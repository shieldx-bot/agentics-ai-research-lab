"""Evidence store hardening — trim cap, stats and agent filter (canonical
record_evidence API from the P16 wiring commit)."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402

from agentics.memory import shared_memory as sm  # noqa: E402
from agentics.memory.shared_memory import SharedMemory  # noqa: E402
from agentics.verify_ops import Evidence  # noqa: E402


@pytest.fixture()
def mem(tmp_path):
    m = SharedMemory(tmp_path / "m.db")
    yield m
    m.close()


def _row(**over) -> dict:
    row = Evidence(cmd="python experiments/exp.py", exit_code=0, verdict="confirmed",
                   output='{"metric": 0.87}\n', duration_s=1.2).as_row()
    row.update(over)
    return row


def _record(mem, agent="a04", task=3, **over):
    r = _row(**over)
    return mem.record_evidence(
        agent, task, cmd=r["cmd"], exit_code=r["exit_code"],
        verdict=r["verdict"], hypothesis="retry budget cuts failures",
        method=r["method"], output=r["output"], duration_s=r["duration_s"],
    )


def test_evidence_table_is_trimmed(mem, monkeypatch):
    monkeypatch.setattr(sm, "_EVIDENCE_TRIM_EVERY", 10)
    monkeypatch.setattr(sm, "_EVIDENCE_KEEP_ROWS", 6)
    for i in range(30):
        _record(mem, task=i)
    rows = mem.latest_evidence(limit=100)
    assert len(rows) == 6
    assert rows[0]["task_id"] == 29          # newest survives
    assert rows[-1]["task_id"] == 24         # oldest surviving


def test_evidence_stats_counts_verdicts(mem):
    _record(mem, verdict="confirmed", exit_code=0)
    _record(mem, agent="a07", verdict="refuted", exit_code=1)
    _record(mem, agent="a07", verdict="inconclusive", exit_code=124)
    assert mem.evidence_stats() == {"total": 3, "confirmed": 1,
                                    "refuted": 1, "inconclusive": 1}


def test_latest_evidence_agent_filter(mem):
    _record(mem, agent="a04", task=1)
    _record(mem, agent="a07", task=2)
    feed = mem.latest_evidence(limit=10, agent="a04")
    assert [r["agent"] for r in feed] == ["a04"]
    assert len(mem.latest_evidence(limit=10)) == 2
