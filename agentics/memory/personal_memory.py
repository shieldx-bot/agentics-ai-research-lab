"""Personal memory — each agent's private notebook (SQLite, one file per agent)."""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS lessons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    issue INTEGER,
    lesson TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at REAL NOT NULL
);
"""


class PersonalMemory:
    def __init__(self, agent_id: str, base_dir: Path) -> None:
        base_dir.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(base_dir / f"{agent_id}.db"))
        self._conn.execute("PRAGMA journal_mode=WAL")
        with self._conn:
            self._conn.executescript(_SCHEMA)

    # lessons
    def add_lesson(self, lesson: str, issue: int | None = None) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO lessons (ts, issue, lesson) VALUES (?,?,?)",
                (time.time(), issue, lesson),
            )

    def recent_lessons(self, limit: int = 20) -> list[str]:
        rows = self._conn.execute(
            "SELECT lesson FROM lessons ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [r[0] for r in reversed(rows)]

    # key-value state (survives restarts → continuity across infinite cycles)
    def set_state(self, key: str, value: str) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO state (key, value, updated_at) VALUES (?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=?, updated_at=?",
                (key, value, time.time(), value, time.time()),
            )

    def get_state(self, key: str, default: str = "") -> str:
        row = self._conn.execute(
            "SELECT value FROM state WHERE key=?", (key,)
        ).fetchone()
        return row[0] if row else default

    def close(self) -> None:
        self._conn.close()
