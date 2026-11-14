"""Journal growth cap — the forever-running lab must not grow its feed unbounded."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402

from agentics.memory import shared_memory as sm  # noqa: E402
from agentics.memory.shared_memory import SharedMemory  # noqa: E402


@pytest.fixture()
def mem(tmp_path):
    m = SharedMemory(tmp_path / "m.db")
    yield m
    m.close()


def test_journal_is_trimmed_to_the_newest_rows(mem, monkeypatch):
    monkeypatch.setattr(sm, "_JOURNAL_TRIM_EVERY", 10)
    monkeypatch.setattr(sm, "_JOURNAL_KEEP_ROWS", 12)
    for i in range(10):
        mem.journal("a01", "event", {"i": i})
    assert len(mem.recent_journal(limit=100)) == 10       # below the cap: untouched
    for i in range(10, 30):
        mem.journal("a01", "event", {"i": i})
    rows = mem.recent_journal(limit=100)
    assert len(rows) == 12                                # capped at KEEP_ROWS
    assert rows[0]["detail"] == '{"i": 18}'               # oldest surviving row
    assert rows[-1]["detail"] == '{"i": 29}'              # newest row is intact
