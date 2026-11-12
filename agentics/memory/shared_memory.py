"""Shared memory — the common brain of the swarm (SQLite WAL).

Tables:
    messages    — chat/broadcast between agents
    findings    — research results (title, content, citations)
    decisions   — durable decisions (who does what, why)
    claims      — task claim registry (P1)
    heartbeats  — liveness of each agent (P6) + live state for the dashboard
                  (current node, task, published intent)
    journal     — append-only activity feed (what/why each agent did) — the
                  owner's window into the swarm (anti blackbox)
    tasks       — LOCAL task queue (replaces GitHub issues in local-only mode)
    reviews     — LOCAL review requests ("MR", replaces GitHub pull requests)
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
    status TEXT DEFAULT 'idle',
    current_node TEXT DEFAULT '',
    task_id INTEGER,
    task_title TEXT DEFAULT '',
    intent TEXT DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS journal (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    agent TEXT NOT NULL,
    event TEXT NOT NULL,
    detail TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    title TEXT NOT NULL,
    body TEXT DEFAULT '',
    category TEXT DEFAULT 'research',
    labels TEXT DEFAULT '',
    status TEXT NOT NULL DEFAULT 'open',
    -- open | claimed | in_review | merged | failed | closed
    claimed_by TEXT DEFAULT '',
    branch TEXT DEFAULT '',
    exempt INTEGER NOT NULL DEFAULT 0,
    -- 1 = owner granted a protected-paths exemption for this task only
    closed_ts REAL
);
CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    task_id INTEGER,
    title TEXT NOT NULL,
    body TEXT DEFAULT '',
    author TEXT NOT NULL,
    branch TEXT NOT NULL,
    files TEXT DEFAULT '[]',
    verification TEXT DEFAULT '',
    status TEXT NOT NULL DEFAULT 'open',
    -- open | approved | changes_requested | merged | failed
    reviewer TEXT DEFAULT '',
    verdict_comment TEXT DEFAULT '',
    approved_by TEXT DEFAULT ''
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
CREATE TABLE IF NOT EXISTS file_locks (
    path TEXT PRIMARY KEY,                  -- normalized relative path
    agent TEXT NOT NULL,
    acquired_at REAL NOT NULL,
    expires_at REAL NOT NULL
);
"""


# Journal growth cap: the append-only activity feed is trimmed to the newest
# rows every N inserts (never on each write — trimming is a rare bulk DELETE).
_JOURNAL_TRIM_EVERY = 500
_JOURNAL_KEEP_ROWS = 5000


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
        self._journal_inserts = 0      # drives the periodic journal trim
        with self._lock, self._conn:
            self._conn.executescript(_SCHEMA)
            for column in (
                "ALTER TABLE traces ADD COLUMN cluster_id INTEGER",
                "ALTER TABLE heartbeats ADD COLUMN current_node TEXT DEFAULT ''",
                "ALTER TABLE heartbeats ADD COLUMN task_id INTEGER",
                "ALTER TABLE heartbeats ADD COLUMN task_title TEXT DEFAULT ''",
                "ALTER TABLE heartbeats ADD COLUMN intent TEXT DEFAULT '{}'",
            ):                 # migrations for DBs created before LifeMem / dashboard
                try:
                    self._conn.execute(column)
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

    def renew_claim(self, issue: int, agent: str, ttl_s: float) -> bool:
        """Extend a live lease only for its current owner; expired leases cannot revive."""
        now = time.time()
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE claims SET expires_at=? "
                "WHERE issue=? AND agent=? AND status='active' AND expires_at>?",
                (now + ttl_s, issue, agent, now),
            )
            return cur.rowcount == 1

    def release_claim(self, issue: int, agent: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE claims SET status='released' "
                "WHERE issue=? AND agent=? AND status='active'",
                (issue, agent),
            )

    def complete_claim(self, issue: int, agent: str, branch: str = "") -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE claims SET status='done', branch=? WHERE issue=? AND agent=?",
                (branch, issue, agent),
            )

    def active_claims(self) -> dict[int, Claim]:
        """Return currently valid leases (for status displays and renewals)."""
        now = time.time()
        with self._lock:
            rows = self._conn.execute(
                "SELECT issue, agent, claimed_at, expires_at, branch, status "
                "FROM claims WHERE status='active' AND expires_at > ?",
                (now,),
            ).fetchall()
        return {r[0]: Claim(r[0], r[1], r[2], r[3], r[4], r[5]) for r in rows}

    def all_active_claims(self) -> dict[int, Claim]:
        """Return every active row, including expired leases awaiting cleanup."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT issue, agent, claimed_at, expires_at, branch, status "
                "FROM claims WHERE status='active'"
            ).fetchall()
        return {r[0]: Claim(r[0], r[1], r[2], r[3], r[4], r[5]) for r in rows}

    def claims_for_agent(self, agent: str) -> dict[int, Claim]:
        """Return active rows owned by one agent, including expired leases."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT issue, agent, claimed_at, expires_at, branch, status "
                "FROM claims WHERE status='active' AND agent=?",
                (agent,),
            ).fetchall()
        return {r[0]: Claim(r[0], r[1], r[2], r[3], r[4], r[5]) for r in rows}

    # ------------------------------------------------------------ heartbeats + live state
    def heartbeat(self, agent: str, role: str, cycle: int, status: str = "idle") -> None:
        now = time.time()
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO heartbeats (agent, role, last_ts, cycle, status) "
                "VALUES (?,?,?,?,?) "
                "ON CONFLICT(agent) DO UPDATE SET role=?, last_ts=?, cycle=?, status=?",
                (agent, role, now, cycle, status, role, now, cycle, status),
            )

    def report(
        self, agent: str, role: str, node: str = "", status: str = "working",
        cycle: int = 0, task_id: int | None = None, task_title: str | None = None,
        intent: dict | str | None = None,
    ) -> None:
        """Live agent state for the dashboard: node, task, and published intent.

        ``None`` fields keep their previous value; pass ``""`` / ``{}`` to clear.
        """
        now = time.time()
        if isinstance(intent, dict):
            intent = json.dumps(intent, ensure_ascii=False)
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO heartbeats (agent, role, last_ts, cycle, status,
                                        current_node, task_id, task_title, intent)
                VALUES (?,?,?,?,?,?,?,?,?)
                ON CONFLICT(agent) DO UPDATE SET
                    role=?, last_ts=?, cycle=?, status=?,
                    current_node=CASE WHEN ?='' THEN current_node ELSE ? END,
                    task_id=CASE WHEN ? IS NULL THEN task_id ELSE ? END,
                    task_title=CASE WHEN ? IS NULL THEN task_title ELSE ? END,
                    intent=CASE WHEN ? IS NULL THEN intent ELSE ? END
                """,
                (agent, role, now, cycle, status, node, task_id, task_title or "",
                 intent or "{}",
                 role, now, cycle, status, node, node, task_id, task_id,
                 task_title, task_title, intent, intent),
            )

    def active_agents(self, stale_after_s: float = 300.0) -> list[dict]:
        cutoff = time.time() - stale_after_s
        with self._lock:
            rows = self._conn.execute(
                "SELECT agent, role, last_ts, cycle, status, current_node, task_id, "
                "task_title, intent FROM heartbeats "
                "WHERE last_ts > ? ORDER BY last_ts DESC", (cutoff,)
            ).fetchall()
        out = []
        for r in rows:
            try:
                intent = json.loads(r[8] or "{}")
            except json.JSONDecodeError:
                intent = {"raw": str(r[8])[:200]}
            out.append({
                "agent": r[0], "role": r[1], "last_ts": r[2], "cycle": r[3],
                "status": r[4], "node": r[5], "task_id": r[6], "task_title": r[7],
                "intent": intent,
            })
        return out

    # ------------------------------------------------------------ journal (anti blackbox)
    def journal(self, agent: str, event: str, detail: str | dict = "") -> None:
        if isinstance(detail, dict):
            detail = json.dumps(detail, ensure_ascii=False)
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO journal (ts, agent, event, detail) VALUES (?,?,?,?)",
                (time.time(), agent, event, str(detail)[:2000]),
            )
            # The lab runs forever; the journal is an append-only feed the
            # dashboard reads through a small window. Keep the table bounded
            # (checked every _JOURNAL_TRIM_EVERY inserts, not on each write).
            self._journal_inserts += 1
            if self._journal_inserts % _JOURNAL_TRIM_EVERY == 0:
                self._conn.execute(
                    "DELETE FROM journal WHERE id NOT IN "
                    "(SELECT id FROM journal ORDER BY id DESC LIMIT ?)",
                    (_JOURNAL_KEEP_ROWS,),
                )

    def recent_journal(self, limit: int = 50, agent: str | None = None) -> list[dict]:
        sql = "SELECT id, ts, agent, event, detail FROM journal"
        args: list = []
        if agent:
            sql += " WHERE agent=?"
            args.append(agent)
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return [
            {"id": r[0], "ts": r[1], "agent": r[2], "event": r[3], "detail": r[4]}
            for r in reversed(rows)
        ]

    # ------------------------------------------------------------ tasks (local issue queue)
    def create_task(
        self, title: str, body: str = "", category: str = "research",
        labels: str = "", agent: str = "owner",
    ) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO tasks (ts, title, body, category, labels) VALUES (?,?,?,?,?)",
                (time.time(), title[:300], body[:8000], category, labels[:200]),
            )
            task_id = int(cur.lastrowid or 0)
        self.journal(agent, "task_created", {"task": task_id, "title": title})
        return task_id

    def create_task_if_absent(
        self, title: str, body: str = "", category: str = "research",
        labels: str = "", agent: str = "owner",
    ) -> int | None:
        """Open a task unless an OPEN one with the same title already exists."""
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM tasks WHERE title=? AND status IN "
                "('open','claimed','in_review')", (title,)
            ).fetchone()
        if row:
            return None
        return self.create_task(title, body, category, labels, agent)

    def get_task(self, task_id: int) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, ts, title, body, category, labels, status, claimed_by, "
                "branch, exempt, closed_ts FROM tasks WHERE id=?", (task_id,)
            ).fetchone()
        if row is None:
            return None
        return {
            "id": row[0], "ts": row[1], "title": row[2], "body": row[3],
            "category": row[4], "labels": row[5], "status": row[6],
            "claimed_by": row[7], "branch": row[8], "exempt": bool(row[9]),
            "closed_ts": row[10],
        }

    def scan_open_tasks(self, limit: int = 30) -> list[dict]:
        """Open, unclaimed tasks, oldest first (the local replacement for
        scanning open unclaimed GitHub issues)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, ts, title, body, category, labels, exempt FROM tasks "
                "WHERE status='open' ORDER BY ts, id LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {"id": r[0], "ts": r[1], "title": r[2], "body": r[3],
             "category": r[4], "labels": r[5], "exempt": bool(r[6])}
            for r in rows
        ]

    def list_tasks(self, limit: int = 30, status: str | None = None) -> list[dict]:
        sql = (
            "SELECT id, ts, title, body, category, labels, status, claimed_by, "
            "branch, exempt, closed_ts FROM tasks"
        )
        args: list = []
        if status:
            sql += " WHERE status=?"
            args.append(status)
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return [
            {"id": r[0], "ts": r[1], "title": r[2], "body": r[3], "category": r[4],
             "labels": r[5], "status": r[6], "claimed_by": r[7], "branch": r[8],
             "exempt": bool(r[9]), "closed_ts": r[10]}
            for r in rows
        ]

    def set_task_status(
        self, task_id: int, status: str, claimed_by: str | None = None,
        branch: str | None = None,
    ) -> None:
        sets, args = ["status=?"], [status]
        if claimed_by is not None:
            sets.append("claimed_by=?")
            args.append(claimed_by)
        if branch is not None:
            sets.append("branch=?")
            args.append(branch)
        if status in ("merged", "failed", "closed"):
            sets.append("closed_ts=?")
            args.append(time.time())
        args.append(task_id)
        with self._lock, self._conn:
            self._conn.execute(f"UPDATE tasks SET {', '.join(sets)} WHERE id=?", args)

    def set_task_exempt(self, task_id: int, exempt: bool, agent: str = "owner") -> None:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE tasks SET exempt=? WHERE id=?", (1 if exempt else 0, task_id)
            )
            changed = cur.rowcount
        if changed:
            self.journal(agent, "task_exempt_set" if exempt else "task_exempt_cleared",
                         {"task": task_id})

    def task_counts(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT status, COUNT(*) FROM tasks GROUP BY status"
            ).fetchall()
        return {status: n for status, n in rows}

    # ------------------------------------------------------------ reviews (local MRs)
    def open_review(
        self, task_id: int | None, title: str, body: str, author: str,
        branch: str, files: list[str], verification: str = "",
    ) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO reviews (ts, task_id, title, body, author, branch, "
                "files, verification) VALUES (?,?,?,?,?,?,?,?)",
                (time.time(), task_id, title[:300], body[:8000], author, branch,
                 json.dumps(files), verification[:500]),
            )
            mr_id = int(cur.lastrowid or 0)
        if task_id is not None:
            self.set_task_status(task_id, "in_review")
        return mr_id

    def _review_row(self, row_id: int) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, ts, task_id, title, body, author, branch, files, "
                "verification, status, reviewer, verdict_comment, approved_by "
                "FROM reviews WHERE id=?", (row_id,)
            ).fetchone()
        if row is None:
            return None
        return {
            "id": row[0], "ts": row[1], "task_id": row[2], "title": row[3],
            "body": row[4], "author": row[5], "branch": row[6],
            "files": json.loads(row[7] or "[]"), "verification": row[8],
            "status": row[9], "reviewer": row[10], "verdict_comment": row[11],
            "approved_by": row[12],
        }

    def get_review(self, mr_id: int) -> dict | None:
        return self._review_row(mr_id)

    def open_reviews(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM reviews WHERE status IN ('open','changes_requested') "
                "ORDER BY id"
            ).fetchall()
        return [self._review_row(r[0]) for r in rows]

    def recent_reviews(self, limit: int = 20) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM reviews ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._review_row(r[0]) for r in rows]

    def set_review_verdict(
        self, mr_id: int, status: str, reviewer: str, comment: str = "",
    ) -> None:
        """status: approved | changes_requested | merged | failed."""
        if status == "approved":
            sql = ("UPDATE reviews SET reviewer=?, verdict_comment=?, "
                   "status='approved', approved_by=? WHERE id=?")
            args = [reviewer, comment[:3000], reviewer, mr_id]
        else:
            sql = "UPDATE reviews SET reviewer=?, verdict_comment=?, status=? WHERE id=?"
            args = [reviewer, comment[:3000], status, mr_id]
        with self._lock, self._conn:
            self._conn.execute(sql, args)

    def reviews_by_author(self, author: str, status: str | None = None) -> list[dict]:
        """MRs written by one agent, oldest first (the rework queue)."""
        sql, args = "SELECT id FROM reviews WHERE author=?", [author]
        if status:
            sql += " AND status=?"
            args.append(status)
        sql += " ORDER BY id"
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return [self._review_row(r[0]) for r in rows]

    def reviews_by_task(self, task_id: int) -> list[dict]:
        """Every MR raised against a task, oldest first (orphan detection)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM reviews WHERE task_id=? ORDER BY id", (task_id,)
            ).fetchall()
        return [self._review_row(r[0]) for r in rows]

    def reopen_review(self, mr_id: int, files: list[str], verification: str = "") -> bool:
        """Rework: an author pushes fixes and puts their MR back into review.

        Only a changes_requested MR authored by ``agent`` is reopened; reviewer
        fields are cleared so the (possibly different) reviewer starts fresh.
        The task goes back to in_review if it was parked elsewhere.
        """
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT author, status, task_id FROM reviews WHERE id=?", (mr_id,)
            ).fetchone()
            if row is None or row[1] != "changes_requested":
                return False
            author, _, task_id = row
            self._conn.execute(
                "UPDATE reviews SET status='open', files=?, verification=?, "
                "reviewer='', verdict_comment='', approved_by='' WHERE id=?",
                (json.dumps(files), verification[:500], mr_id),
            )
        if task_id is not None:
            self.set_task_status(task_id, "in_review")
        self.journal(author, "mr_reopened", {"mr": mr_id, "task": task_id})
        return True

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

    # ------------------------------------------------------------ file locks (P13)
    def acquire_file_lock(self, path: str, agent: str, ttl_s: float) -> bool:
        """Atomic per-file lock: a live lock held by another agent wins."""
        now = time.time()
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT agent, expires_at FROM file_locks WHERE path=?", (path,)
            ).fetchone()
            if row:
                holder, expires_at = row
                if expires_at > now and holder != agent:
                    return False
                self._conn.execute(
                    "UPDATE file_locks SET agent=?, acquired_at=?, expires_at=? "
                    "WHERE path=?", (agent, now, now + ttl_s, path),
                )
                return True
            try:
                self._conn.execute(
                    "INSERT INTO file_locks (path, agent, acquired_at, expires_at) "
                    "VALUES (?,?,?,?)", (path, agent, now, now + ttl_s),
                )
                return True
            except sqlite3.IntegrityError:
                return False

    def release_file_lock(self, path: str, agent: str) -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "DELETE FROM file_locks WHERE path=? AND agent=?", (path, agent),
            )
            return cur.rowcount == 1

    def release_agent_locks(self, agent: str) -> list[str]:
        with self._lock, self._conn:
            rows = self._conn.execute(
                "SELECT path FROM file_locks WHERE agent=?", (agent,)
            ).fetchall()
            self._conn.execute("DELETE FROM file_locks WHERE agent=?", (agent,))
        return [r[0] for r in rows]

    def active_file_locks(self) -> list[dict]:
        now = time.time()
        with self._lock:
            rows = self._conn.execute(
                "SELECT path, agent, acquired_at, expires_at FROM file_locks "
                "WHERE expires_at > ? ORDER BY path", (now,),
            ).fetchall()
        return [{"path": r[0], "agent": r[1], "acquired_at": r[2],
                 "expires_at": r[3]} for r in rows]

    def sweep_file_locks(self) -> list[str]:
        """Drop expired locks (dead agents must not freeze files forever)."""
        now = time.time()
        with self._lock, self._conn:
            rows = self._conn.execute(
                "SELECT path FROM file_locks WHERE expires_at <= ?", (now,)
            ).fetchall()
            if rows:
                self._conn.execute(
                    "DELETE FROM file_locks WHERE expires_at <= ?", (now,))
        return [r[0] for r in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
