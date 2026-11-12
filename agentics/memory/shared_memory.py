"""Shared memory — the common brain of the swarm (SQLite WAL).

Tables:
    messages    — chat/broadcast between agents
    findings    — research results (title, content, citations)
    decisions   — durable decisions (who does what, why)
    claims      — issue claim registry (P1)
    heartbeats  — liveness of each agent (P6)
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

    def close(self) -> None:
        with self._lock:
            self._conn.close()
