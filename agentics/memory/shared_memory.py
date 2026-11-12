"""Shared memory — the common brain of the swarm (SQLite WAL).

Tables:
    messages    — chat/broadcast between agents
    findings    — research results (title, content, citations)
    decisions   — durable decisions (who does what, why)
    claims      — issue claim registry (P1)
    heartbeats  — liveness of each agent (P6)
    traces      — evolutionary traces: what each agent tried, where it failed, outcome (GEA)
    trace_clusters — LifeMem workflow clusters grouping similar traces
    skills      — skills distilled from merged traces of each cluster (LifeMem)
    tool_registry — reusable tools any agent can discover (GEA: anti "innovation dying out")
    merge_queue — persistent, cross-agent merge queue (P4)
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    agent TEXT NOT NULL,
    kind TEXT NOT NULL,           -- broadcast | handoff | alert | review_request
    issue INTEGER,
    content TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS findings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    agent TEXT NOT NULL,
    issue INTEGER,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    citations TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    agent TEXT NOT NULL,
    decision TEXT NOT NULL,
    reason TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS claims (
    issue INTEGER PRIMARY KEY,
    agent TEXT NOT NULL,
    claimed_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    branch TEXT DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active'   -- active | done | released
);
CREATE TABLE IF NOT EXISTS heartbeats (
    agent TEXT PRIMARY KEY,
    role TEXT DEFAULT '',
    last_ts REAL NOT NULL,
    cycle INTEGER DEFAULT 0,
    status TEXT DEFAULT 'idle'
);
CREATE TABLE IF NOT EXISTS traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    agent TEXT NOT NULL,
    role TEXT DEFAULT '',
    cycle INTEGER DEFAULT 0,
    issue INTEGER,
    category TEXT DEFAULT 'other',
    pr_number INTEGER,
    outcome TEXT NOT NULL,        -- pending | approved | merged | rejected | failed | yield | no_change
    attempted TEXT DEFAULT '',    -- what the agent tried (plan / files)
    failure TEXT DEFAULT '',      -- where and why it failed ('' when it did not)
    files TEXT DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_traces_agent ON traces(agent);
CREATE INDEX IF NOT EXISTS idx_traces_pr ON traces(pr_number);
CREATE TABLE IF NOT EXISTS tool_registry (
    name TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    description TEXT NOT NULL,
    agent TEXT NOT NULL,
    issue INTEGER,
    pr_number INTEGER,
    status TEXT NOT NULL DEFAULT 'pending',   -- pending (PR open) | active (merged)
    ts REAL NOT NULL,
    uses INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS merge_queue (
    pr_number INTEGER PRIMARY KEY,
    author TEXT NOT NULL,
    enqueued_at REAL NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'queued'   -- queued | merged | failed
);
CREATE TABLE IF NOT EXISTS trace_clusters (
    cluster_id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL,
    description TEXT DEFAULT '',
    created_ts REAL NOT NULL,
    updated_ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_clusters_cat ON trace_clusters(category);
CREATE TABLE IF NOT EXISTS skills (
    skill_id INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_id INTEGER NOT NULL REFERENCES trace_clusters(cluster_id),
    text TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',   -- active | superseded
    ts REAL NOT NULL,
    uses INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_skills_cluster ON skills(cluster_id, status);
"""


@dataclass
class Claim:
    issue: int
    agent: str
    claimed_at: float
    expires_at: float
    branch: str
    status: str


class SharedMemory:
    """Thread-safe (single connection + lock); opened with WAL for concurrency."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        with self._lock, self._conn:
            self._conn.executescript(_SCHEMA)
            try:      # migration for DBs created before LifeMem clustering
                self._conn.execute("ALTER TABLE traces ADD COLUMN cluster_id INTEGER")
            except sqlite3.OperationalError:
                pass  # column already exists

    # ------------------------------------------------------------ messages
    def post_message(self, agent: str, kind: str, content: str, issue: int | None = None) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO messages (ts, agent, kind, issue, content) VALUES (?,?,?,?,?)",
                (time.time(), agent, kind, issue, content),
            )

    def recent_messages(self, limit: int = 50, since_ts: float = 0.0) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT ts, agent, kind, issue, content FROM messages "
                "WHERE ts > ? ORDER BY id DESC LIMIT ?",
                (since_ts, limit),
            ).fetchall()
        return [
            {"ts": r[0], "agent": r[1], "kind": r[2], "issue": r[3], "content": r[4]}
            for r in reversed(rows)
        ]

    # ------------------------------------------------------------ findings
    def add_finding(
        self, agent: str, title: str, content: str,
        citations: list[str] | None = None, issue: int | None = None,
    ) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO findings (ts, agent, issue, title, content, citations) "
                "VALUES (?,?,?,?,?,?)",
                (time.time(), agent, issue, title, content, json.dumps(citations or [])),
            )
            return int(cur.lastrowid or 0)

    def list_findings(self, limit: int = 100) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, ts, agent, issue, title, content, citations "
                "FROM findings ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [
            {"id": r[0], "ts": r[1], "agent": r[2], "issue": r[3],
             "title": r[4], "content": r[5], "citations": json.loads(r[6])}
            for r in reversed(rows)
        ]

    # ------------------------------------------------------------ decisions
    def add_decision(self, agent: str, decision: str, reason: str = "") -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO decisions (ts, agent, decision, reason) VALUES (?,?,?,?)",
                (time.time(), agent, decision, reason),
            )

    # ------------------------------------------------------------ claims
    def claim_issue(self, issue: int, agent: str, ttl_s: float, branch: str = "") -> bool:
        """Atomic claim: only succeeds if no other active claim exists (P1)."""
        now = time.time()
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT agent, expires_at, status FROM claims WHERE issue=?", (issue,)
            ).fetchone()
            if row:
                prev_agent, expires_at, status = row
                if status == "active" and expires_at > now and prev_agent != agent:
                    return False          # someone else holds a live claim
                self._conn.execute(
                    "UPDATE claims SET agent=?, claimed_at=?, expires_at=?, branch=?, "
                    "status='active' WHERE issue=?",
                    (agent, now, now + ttl_s, branch, issue),
                )
                return True
            self._conn.execute(
                "INSERT INTO claims (issue, agent, claimed_at, expires_at, branch) "
                "VALUES (?,?,?,?,?)",
                (issue, agent, now, now + ttl_s, branch),
            )
            return True

    def release_claim(self, issue: int, agent: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE claims SET status='released' WHERE issue=? AND agent=?",
                (issue, agent),
            )

    def complete_claim(self, issue: int, agent: str, branch: str = "") -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE claims SET status='done', branch=? WHERE issue=? AND agent=?",
                (branch, issue, agent),
            )

    def active_claims(self) -> dict[int, Claim]:
        now = time.time()
        with self._lock:
            rows = self._conn.execute(
                "SELECT issue, agent, claimed_at, expires_at, branch, status "
                "FROM claims WHERE status='active' AND expires_at > ?",
                (now,),
            ).fetchall()
        return {
            r[0]: Claim(r[0], r[1], r[2], r[3], r[4], r[5]) for r in rows
        }

    # ------------------------------------------------------------ heartbeats
    def heartbeat(self, agent: str, role: str, cycle: int, status: str = "idle") -> None:
        now = time.time()
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO heartbeats (agent, role, last_ts, cycle, status) "
                "VALUES (?,?,?,?,?) "
                "ON CONFLICT(agent) DO UPDATE SET role=?, last_ts=?, cycle=?, status=?",
                (agent, role, now, cycle, status, role, now, cycle, status),
            )

    def active_agents(self, stale_after_s: float = 300.0) -> list[dict]:
        cutoff = time.time() - stale_after_s
        with self._lock:
            rows = self._conn.execute(
                "SELECT agent, role, last_ts, cycle, status FROM heartbeats "
                "WHERE last_ts > ? ORDER BY last_ts DESC", (cutoff,)
            ).fetchall()
        return [
            {"agent": r[0], "role": r[1], "last_ts": r[2], "cycle": r[3], "status": r[4]}
            for r in rows
        ]

    # ------------------------------------------------------------ traces (GEA)
    def add_trace(
        self, agent: str, role: str, cycle: int, outcome: str,
        issue: int | None = None, category: str = "other",
        pr_number: int | None = None, attempted: str = "",
        failure: str = "", files: list[str] | None = None,
    ) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO traces (ts, agent, role, cycle, issue, category, pr_number, "
                "outcome, attempted, failure, files) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (time.time(), agent, role, cycle, issue, category, pr_number, outcome,
                 attempted[:3000], failure[:1500], json.dumps(files or [])),
            )
            return int(cur.lastrowid or 0)

    def set_trace_outcome(self, pr_number: int, outcome: str, note: str = "") -> int:
        """Update the outcome of every trace attached to a PR (review / merge result).

        `note` (e.g. the reviewer's comment) is stored in `failure` only for
        negative outcomes, so other agents can learn *why* the PR was refused.
        """
        with self._lock, self._conn:
            if note and outcome in ("rejected", "failed"):
                cur = self._conn.execute(
                    "UPDATE traces SET outcome=?, failure=? WHERE pr_number=?",
                    (outcome, note[:1500], pr_number),
                )
            else:
                cur = self._conn.execute(
                    "UPDATE traces SET outcome=? WHERE pr_number=?", (outcome, pr_number)
                )
            ids = [r[0] for r in self._conn.execute(
                "SELECT id FROM traces WHERE pr_number=?", (pr_number,)
            ).fetchall()]
            return ids

    def recent_traces(
        self, limit: int = 20, issue: int | None = None,
        category: str | None = None, failures_only: bool = False,
        exclude_agent: str | None = None,
    ) -> list[dict]:
        where, args = [], []
        if issue is not None:
            where.append("issue=?")
            args.append(issue)
        if category:
            where.append("category=?")
            args.append(category)
        if failures_only:
            where.append("outcome IN ('failed','rejected')")
        if exclude_agent:
            where.append("agent<>?")
            args.append(exclude_agent)
        sql = (
            "SELECT id, ts, agent, role, cycle, issue, category, pr_number, outcome, "
            "attempted, failure, files FROM traces"
            + (" WHERE " + " AND ".join(where) if where else "")
            + " ORDER BY id DESC LIMIT ?"
        )
        with self._lock:
            rows = self._conn.execute(sql, (*args, limit)).fetchall()
        keys = ("id", "ts", "agent", "role", "cycle", "issue", "category", "pr_number",
                "outcome", "attempted", "failure", "files")
        out = []
        for r in reversed(rows):
            d = dict(zip(keys, r))
            d["files"] = json.loads(d["files"] or "[]")
            out.append(d)
        return out

    def agent_outcomes(self) -> dict[str, list[tuple[str, str]]]:
        """agent -> [(category, outcome)] for *terminal* outcomes only
        (merged = success; rejected/failed = failure; yield/pending/... ignored)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT agent, category, outcome FROM traces "
                "WHERE outcome IN ('merged','rejected','failed') ORDER BY id"
            ).fetchall()
        out: dict[str, list[tuple[str, str]]] = {}
        for agent, category, outcome in rows:
            out.setdefault(agent, []).append((category, outcome))
        return out

    def trace_stats(self, since_ts: float = 0.0) -> dict:
        with self._lock:
            rows = self._conn.execute(
                "SELECT outcome, COUNT(*) FROM traces WHERE ts>=? GROUP BY outcome",
                (since_ts,),
            ).fetchall()
            cat_rows = self._conn.execute(
                "SELECT category, outcome, COUNT(*) FROM traces WHERE ts>=? "
                "GROUP BY category, outcome", (since_ts,),
            ).fetchall()
        by_outcome = {o: n for o, n in rows}
        merged = by_outcome.get("merged", 0)
        judged = merged + by_outcome.get("rejected", 0) + by_outcome.get("failed", 0)
        by_cat: dict[str, dict[str, int]] = {}
        for cat, outcome, n in cat_rows:
            by_cat.setdefault(cat, {})[outcome] = n
        return {
            "by_outcome": by_outcome,
            "merge_rate": (merged / judged) if judged else None,
            "judged": judged,
            "by_category": by_cat,
        }

    # ------------------------------------------------------------ tool registry (GEA)
    def register_tool(
        self, name: str, path: str, description: str, agent: str,
        issue: int | None = None, pr_number: int | None = None,
    ) -> bool:
        """Register a tool as *pending* (visible only once its PR merges, so agents
        never try to reuse code that is not on `main`). False if the name is taken."""
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO tool_registry "
                "(name, path, description, agent, issue, pr_number, status, ts) "
                "VALUES (?,?,?,?,?,?, 'pending', ?)",
                (name, path, description[:500], agent, issue, pr_number, time.time()),
            )
            return cur.rowcount == 1

    def activate_tools(self, pr_number: int) -> int:
        with self._lock, self._conn:
            return self._conn.execute(
                "UPDATE tool_registry SET status='active' WHERE pr_number=? AND status='pending'",
                (pr_number,),
            ).rowcount

    def drop_tools(self, pr_number: int) -> int:
        """PR was rejected / failed verification: its pending tools never existed."""
        with self._lock, self._conn:
            return self._conn.execute(
                "DELETE FROM tool_registry WHERE pr_number=? AND status='pending'", (pr_number,),
            ).rowcount

    def list_tools(self, query: str = "", limit: int = 10) -> list[dict]:
        """Merged tools ranked by keyword hits (name/description/path), then popularity."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT name, path, description, agent, uses FROM tool_registry WHERE status='active'"
            ).fetchall()
        words = {w for w in query.lower().replace("/", " ").replace("_", " ").split() if len(w) > 2}
        scored = []
        for name, path, desc, agent, uses in rows:
            hay = f"{name} {path} {desc}".lower().replace("_", " ")
            hits = sum(1 for w in words if w in hay)
            scored.append((hits, uses, name, path, desc, agent))
        scored.sort(key=lambda t: (-t[0], -t[1], t[2]))
        return [
            {"name": n, "path": p, "description": d, "agent": a, "uses": u}
            for _, u, n, p, d, a in scored[:limit]
        ]

    def bump_tool_use(self, name: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("UPDATE tool_registry SET uses = uses + 1 WHERE name=?", (name,))

    # ------------------------------------------------------------ workflow clusters + skills (LifeMem)
    def create_cluster(self, category: str, description: str = "") -> int:
        now = time.time()
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO trace_clusters (category, description, created_ts, updated_ts) "
                "VALUES (?,?,?,?)", (category, description[:200], now, now),
            )
            return int(cur.lastrowid or 0)

    def get_cluster(self, cluster_id: int) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT cluster_id, category, description, created_ts, updated_ts "
                "FROM trace_clusters WHERE cluster_id=?", (cluster_id,),
            ).fetchone()
        if row is None:
            return None
        return {"cluster_id": row[0], "category": row[1], "description": row[2],
                "created_ts": row[3], "updated_ts": row[4]}

    def list_clusters(self, category: str | None = None) -> list[dict]:
        with self._lock:
            if category:
                rows = self._conn.execute(
                    "SELECT cluster_id, category, description, created_ts, updated_ts "
                    "FROM trace_clusters WHERE category=? ORDER BY cluster_id", (category,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT cluster_id, category, description, created_ts, updated_ts "
                    "FROM trace_clusters ORDER BY cluster_id",
                ).fetchall()
        return [{"cluster_id": r[0], "category": r[1], "description": r[2],
                 "created_ts": r[3], "updated_ts": r[4]} for r in rows]

    def set_trace_cluster(self, trace_id: int, cluster_id: int) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE traces SET cluster_id=? WHERE id=?", (cluster_id, trace_id),
            )
            self._conn.execute(
                "UPDATE trace_clusters SET updated_ts=? WHERE cluster_id=?",
                (time.time(), cluster_id),
            )

    def trace_cluster(self, trace_id: int) -> int | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT cluster_id FROM traces WHERE id=?", (trace_id,),
            ).fetchone()
        return row[0] if row else None

    def cluster_members(self, cluster_id: int) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, outcome, attempted, failure, files FROM traces "
                "WHERE cluster_id=? ORDER BY id", (cluster_id,),
            ).fetchall()
        return [{"id": r[0], "outcome": r[1], "attempted": r[2],
                 "failure": r[3], "files": json.loads(r[4] or "[]")} for r in rows]

    def get_trace(self, trace_id: int) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, category, outcome, attempted, failure, cluster_id "
                "FROM traces WHERE id=?", (trace_id,),
            ).fetchone()
        if row is None:
            return None
        return {"id": row[0], "category": row[1], "outcome": row[2],
                "attempted": row[3], "failure": row[4], "cluster_id": row[5]}

    def save_skill(self, cluster_id: int, text: str) -> None:
        """Insert a new active skill for a cluster, superseding the previous one."""
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE skills SET status='superseded' WHERE cluster_id=? AND status='active'",
                (cluster_id,),
            )
            self._conn.execute(
                "INSERT INTO skills (cluster_id, text, status, ts) VALUES (?,?,'active',?)",
                (cluster_id, text, time.time()),
            )

    def list_skills(self, category: str | None = None, limit: int = 40) -> list[dict]:
        sql = (
            "SELECT s.skill_id, s.cluster_id, s.text, s.uses, c.category, c.description "
            "FROM skills s JOIN trace_clusters c ON s.cluster_id=c.cluster_id "
            "WHERE s.status='active'"
        )
        args: list = []
        if category:
            sql += " AND c.category=?"
            args.append(category)
        sql += " ORDER BY s.skill_id DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return [{"skill_id": r[0], "cluster_id": r[1], "text": r[2], "uses": r[3],
                 "category": r[4], "description": r[5]} for r in rows]

    def bump_skill_use(self, skill_id: int) -> None:
        with self._lock, self._conn:
            self._conn.execute("UPDATE skills SET uses=uses+1 WHERE skill_id=?", (skill_id,))

    def memory_stats(self) -> dict:
        """Counts for `agentics memory-stats` — how big/structured memory is."""
        with self._lock:
            def count(sql: str) -> int:
                return int(self._conn.execute(sql).fetchone()[0])
            top = self._conn.execute(
                "SELECT s.text, s.uses, c.category FROM skills s "
                "JOIN trace_clusters c ON s.cluster_id=c.cluster_id "
                "WHERE s.status='active' ORDER BY s.uses DESC, s.skill_id DESC LIMIT 5"
            ).fetchall()
        return {
            "traces": count("SELECT COUNT(*) FROM traces"),
            "clustered": count("SELECT COUNT(*) FROM traces WHERE cluster_id IS NOT NULL"),
            "clusters": count("SELECT COUNT(*) FROM trace_clusters"),
            "active_skills": count("SELECT COUNT(*) FROM skills WHERE status='active'"),
            "superseded_skills": count("SELECT COUNT(*) FROM skills WHERE status='superseded'"),
            "top_skills": [{"category": r[2], "uses": r[1], "text": r[0][:120]} for r in top],
        }

    # ------------------------------------------------------------ merge queue (P4)
    def mq_enqueue(self, pr_number: int, author: str) -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO merge_queue (pr_number, author, enqueued_at) VALUES (?,?,?)",
                (pr_number, author, time.time()),
            )
            return cur.rowcount == 1

    def mq_pending(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT pr_number, author, enqueued_at, attempts FROM merge_queue "
                "WHERE status='queued' ORDER BY enqueued_at, pr_number"
            ).fetchall()
        return [
            {"pr_number": r[0], "author": r[1], "enqueued_at": r[2], "attempts": r[3]}
            for r in rows
        ]

    def mq_mark(self, pr_number: int, status: str, bump_attempt: bool = False) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE merge_queue SET status=?, attempts = attempts + ? WHERE pr_number=?",
                (status, 1 if bump_attempt else 0, pr_number),
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()
