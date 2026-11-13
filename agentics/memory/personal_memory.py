"""Personal memory — each agent's private notebook (SQLite, one file per agent).

v2 hardening (same ideas, closed loops — nothing new is invented):

- **Dedup**: a recurring failure ("no LLM account available" on every cycle)
  used to insert one identical row per cycle. A lesson is now identified by
  its normalized hash — the *same* dedup idea the shared memory already uses
  for ``claim_hash`` — and a recurrence bumps ``hits`` / ``last_seen``
  instead of adding a row.
- **Recall**: ``recent_lessons`` (recency) and ``relevant_lessons(query)``
  (keyword overlap — the ranking the shared memory's tool/skill lookups
  already use) are actually consumed by the graph now. Before v2 lessons
  were written at reflect time and never read back.
- **Bounded**: the lessons table gets the journal/evidence trim treatment
  (checked every N inserts, keep the newest M rows by last_seen).
- **Migration**: v1 (pre-dedup) databases are upgraded in place — new
  columns, hash backfill, duplicate consolidation — on first open.

v3 (same dedup/recall ideas, one notch stronger):

- **Number-blind identity**: recurring failures embed varying numbers
  ("budget cap reached (57/50)", "PR #12", cycle ids). The identity hash now
  collapses digit runs, so the SAME failure with different counts resolves
  to one row and shows up as ``(xN)`` instead of N near-identical rows. The
  stored text stays verbatim — only identity is number-blind.
- **Issue-aware recall**: ``relevant_lessons`` boosts lessons recorded on the
  same issue — the ``issue`` column was written since v1 but never read back.
- **Migration**: v2 databases re-normalize their hashes in place (one sweep,
  idempotent) and the duplicates that only now collapse are consolidated.
"""
from __future__ import annotations

import hashlib
import re
import sqlite3
import threading
import time
from pathlib import Path

# Tables first, index later: a v1 (pre-dedup) database has no lesson_hash
# column yet — the index is only created after the ALTER TABLE migration
# below added the column, so an old notebook opens without error.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS lessons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    issue INTEGER,
    lesson TEXT NOT NULL,
    lesson_hash TEXT NOT NULL DEFAULT '',
    hits INTEGER NOT NULL DEFAULT 1,
    last_seen REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at REAL NOT NULL
);
"""

_SCHEMA_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_lessons_hash ON lessons(lesson_hash);
"""

# Lesson book-keeping: identical failures repeat cycle after cycle, so the
# dedup already caps growth; the periodic trim below is the same belt-and-
# braces the journal/evidence feeds use (never per write — a bulk DELETE
# every N inserts).
_LESSONS_TRIM_EVERY = 50
_LESSONS_KEEP_ROWS = 500

_WORD_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]{2,}")
_DIGITS_RE = re.compile(r"\d+")

# Identity-hash normalization version. v2 hashed the whitespace/case-normalized
# text only; v3 additionally collapses digit runs (recurring failures embed
# varying counts/ids). Bump this whenever normalization changes so old DBs
# re-normalize + re-consolidate exactly once on open.
_HASH_NORM_VERSION = "3"


def lesson_hash(lesson: str) -> str:
    """Stable identity of a lesson: case/whitespace-normalized text with
    digit runs collapsed — "budget cap reached (57/50)" and "(63/50)" are the
    SAME recurring failure, not two lessons. The same normalization idea the
    shared memory's ``claim_hash`` applies to hypotheses; digits are identity-
    blind because they carry counts and ids that vary per cycle, while the
    verbatim text stays stored for display."""
    normalized = _DIGITS_RE.sub("#", " ".join((lesson or "").lower().split()))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]


class PersonalMemory:
    def __init__(self, agent_id: str, base_dir: Path) -> None:
        base_dir.mkdir(parents=True, exist_ok=True)
        # Graph nodes run in LangGraph's executor threads, so the connection must
        # be usable across threads; a lock serialises concurrent access.
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(base_dir / f"{agent_id}.db"), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._lesson_inserts = 0      # drives the periodic lesson trim
        with self._lock, self._conn:
            self._conn.executescript(_SCHEMA)
            for column in (
                "ALTER TABLE lessons ADD COLUMN lesson_hash TEXT NOT NULL DEFAULT ''",
                "ALTER TABLE lessons ADD COLUMN hits INTEGER NOT NULL DEFAULT 1",
                "ALTER TABLE lessons ADD COLUMN last_seen REAL NOT NULL DEFAULT 0",
            ):                # migrations for v1 (pre-dedup) databases
                try:
                    self._conn.execute(column)
                except sqlite3.OperationalError:
                    pass      # column already exists
            self._migrate_v1_rows()
            self._renormalize_hashes_if_needed()
            self._conn.executescript(_SCHEMA_INDEXES)   # index AFTER the columns exist

    # ------------------------------------------------------------ migration
    def _migrate_v1_rows(self) -> None:
        """Upgrade rows written by v1: backfill their identity hash, set
        ``last_seen`` to the write time, and collapse duplicates that were
        already sitting in the table into one row with summed hits."""
        rows = self._conn.execute(
            "SELECT id, lesson, ts FROM lessons WHERE lesson_hash=''").fetchall()
        for rid, text, ts in rows:
            self._conn.execute(
                "UPDATE lessons SET lesson_hash=?, last_seen=? WHERE id=?",
                (lesson_hash(text), ts, rid),
            )
        self._consolidate_duplicate_lessons()

    def _renormalize_hashes_if_needed(self) -> None:
        """v3: digit-blind identity. A DB written by v2 carries digit-sensitive
        hashes; recompute every row's hash under the new normalization and
        consolidate the duplicates that only now collapse (hits roll up).
        Guarded by a state key so the sweep runs exactly once per DB.

        Runs inside __init__'s lock, so it talks to the connection directly —
        get_state/set_state would re-acquire the (non-reentrant) lock."""
        row = self._conn.execute(
            "SELECT value FROM state WHERE key='lesson_hash_norm'").fetchone()
        if row is not None and row[0] == _HASH_NORM_VERSION:
            return
        for rid, text in self._conn.execute(
                "SELECT id, lesson FROM lessons").fetchall():
            self._conn.execute(
                "UPDATE lessons SET lesson_hash=? WHERE id=?",
                (lesson_hash(text), rid),
            )
        self._consolidate_duplicate_lessons()
        self._conn.execute(
            "INSERT INTO state (key, value, updated_at) VALUES ('lesson_hash_norm',?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=?, updated_at=?",
            (_HASH_NORM_VERSION, time.time(), _HASH_NORM_VERSION, time.time()),
        )

    def _consolidate_duplicate_lessons(self) -> None:
        """Collapse rows sharing a lesson_hash into the newest one, with the
        group's hits summed — the same consolidation the v1 migration uses."""
        by_hash: dict[str, list[tuple[int, int]]] = {}
        for rid, lh, hits in self._conn.execute(
                "SELECT id, lesson_hash, hits FROM lessons").fetchall():
            by_hash.setdefault(lh, []).append((rid, hits))
        for lh, members in by_hash.items():
            if len(members) < 2:
                continue
            keep = max(rid for rid, _ in members)          # newest row survives
            total = max(1, sum(h for _, h in members))     # hits roll up into it
            self._conn.execute(
                "UPDATE lessons SET hits=? WHERE id=?", (total, keep))
            self._conn.execute(
                "DELETE FROM lessons WHERE lesson_hash=? AND id<>?", (lh, keep))

    # ------------------------------------------------------------ lessons
    def add_lesson(self, lesson: str, issue: int | None = None) -> int:
        """Record a lesson; a recurrence of a known lesson bumps its
        ``hits`` / ``last_seen`` (dedup) instead of inserting a duplicate
        row. Returns the row id (0 for an empty lesson)."""
        text = " ".join((lesson or "").split())
        if not text:
            return 0
        now = time.time()
        h = lesson_hash(text)
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT id FROM lessons WHERE lesson_hash=? "
                "ORDER BY id DESC LIMIT 1", (h,),
            ).fetchone()
            if row:
                self._conn.execute(
                    "UPDATE lessons SET hits=hits+1, last_seen=?, "
                    "issue=COALESCE(?, issue) WHERE id=?",
                    (now, issue, row[0]),
                )
                return int(row[0])
            cur = self._conn.execute(
                "INSERT INTO lessons (ts, issue, lesson, lesson_hash, hits, last_seen) "
                "VALUES (?,?,?,?,1,?)",
                (now, issue, text[:4000], h, now),
            )
            lid = int(cur.lastrowid or 0)
            # Bounded table (same policy as the shared journal/evidence feeds):
            # checked every N *new* lessons, trimmed to the M most recently seen.
            self._lesson_inserts += 1
            if self._lesson_inserts % _LESSONS_TRIM_EVERY == 0:
                self._conn.execute(
                    "DELETE FROM lessons WHERE id NOT IN "
                    "(SELECT id FROM lessons ORDER BY last_seen DESC, id DESC LIMIT ?)",
                    (_LESSONS_KEEP_ROWS,),
                )
            return lid

    def recent_lessons(self, limit: int = 20) -> list[str]:
        """Most recently seen lessons, newest first (deduped). A lesson that
        recurred is suffixed ``(xN)`` so the agent reads it as a repeated
        pattern, not one bad day."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT lesson, hits FROM lessons "
                "ORDER BY last_seen DESC, id DESC LIMIT ?", (limit,),
            ).fetchall()
        return [f"{r[0]} (x{r[1]})" if r[1] > 1 else r[0] for r in rows]

    def relevant_lessons(
        self, query: str, limit: int = 5, issue: int | None = None,
    ) -> list[str]:
        """Lessons ranked by keyword overlap with the current task, then by
        how often they recurred (``hits``), then recency — the ranking the
        shared memory's tool/skill lookups already use. The agent's own
        failures on similar work surface first.

        v3: lessons recorded on the SAME issue get a fixed boost — the
        ``issue`` column was written since v1 but never read back, so a
        rework/retry of issue N now finds this agent's earlier cycles on N
        even when the wording shares no keywords."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT lesson, hits, last_seen, issue FROM lessons "
                "ORDER BY last_seen DESC, id DESC LIMIT 200",
            ).fetchall()
        words = set(_WORD_RE.findall((query or "").lower()))
        scored = []
        for text, hits, last_seen, row_issue in rows:
            hay = set(_WORD_RE.findall(text.lower()))
            overlap = len(words & hay)
            if issue is not None and row_issue is not None and int(row_issue) == int(issue):
                overlap += 3      # same-issue history outweighs a couple of keywords
            scored.append((overlap, hits, last_seen, text))
        scored.sort(key=lambda t: (-t[0], -t[1], -t[2]))
        return [
            f"{t} (x{h})" if h > 1 else t
            for _o, h, _ts, t in scored[:limit]
        ]

    def lesson_stats(self) -> dict:
        """Counts for ``agentics memory-stats``: distinct lessons kept, total
        recurrences recorded, and how many lessons are recurring patterns."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(hits),0), "
                "COALESCE(SUM(hits>1),0) FROM lessons",
            ).fetchone()
        return {"lessons": int(row[0]), "recurrences": int(row[1]),
                "recurring": int(row[2])}

    # key-value state (survives restarts → continuity across infinite cycles)
    def set_state(self, key: str, value: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO state (key, value, updated_at) VALUES (?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=?, updated_at=?",
                (key, value, time.time(), value, time.time()),
            )

    def get_state(self, key: str, default: str = "") -> str:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM state WHERE key=?", (key,)
            ).fetchone()
        return row[0] if row else default

    def close(self) -> None:
        with self._lock:
            self._conn.close()
