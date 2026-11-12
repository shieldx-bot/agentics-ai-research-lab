"""Evidence store — P16 records wired into shared memory (hypothesis 3)."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402

from agentics.memory import shared_memory as sm  # noqa: E402
from agentics.memory.shared_memory import SharedMemory  # noqa: E402
from agentics.verify_ops import Evidence, run_experiment  # noqa: E402


@pytest.fixture()
def mem(tmp_path):
    m = SharedMemory(tmp_path / "m.db")
    yield m
    m.close()


def _as_row(**over) -> dict:
    row = Evidence(cmd="python experiments/exp.py", exit_code=0, verdict="confirmed",
                   output='{"metric": 0.87}\n', duration_s=1.2).as_row()
    row.update(over)
    return row


def test_record_evidence_from_as_row_contract(mem):
    """The wiring contract: verify_ops.Evidence.as_row() plugs straight in."""
    eid = mem.record_evidence("a04", "coder", 3, 7, "retry budget cuts failures",
                              _as_row(metric=0.87, threshold=0.9, comparator="lt"))
    rows = mem.recent_evidence()
    assert len(rows) == 1 and rows[0]["id"] == eid
    assert rows[0]["verdict"] == "confirmed"
    assert rows[0]["cmd"].endswith("exp.py") and rows[0]["exit_code"] == 0
    assert rows[0]["metric"] == 0.87 and rows[0]["comparator"] == "lt"
    # H3: the evidence is broadcast + journalled for every agent to see
    assert any("Evidence [confirmed]" in m["content"] for m in mem.recent_messages(limit=5))
    assert any(e["event"] == "evidence_recorded" for e in mem.recent_journal(limit=5))


def test_record_from_a_real_run_experiment(mem, tmp_path):
    """End-to-end against the real runner: a blocked command can never pose
    as confirmed (126 -> inconclusive), matching verify_ops semantics."""
    from types import SimpleNamespace

    from agentics.github_ops.repo_workspace import RepoWorkspace

    ws = RepoWorkspace("a04", SimpleNamespace(
        workspaces_dir=tmp_path, data_dir=tmp_path,
        local_repo_path=tmp_path / "lab.git"))
    ev = run_experiment(ws, ["python", "definitely_missing_script.py"])
    eid = mem.record_evidence("a04", "coder", 1, None, "h", ev.as_row())
    row = [r for r in mem.recent_evidence() if r["id"] == eid][0]
    assert row["verdict"] in ("refuted", "inconclusive")
    assert row["verdict"] != "confirmed"


def test_invalid_verdict_and_comparator_are_sanitised(mem):
    mem.record_evidence("a04", "coder", 1, None, "h",
                        _as_row(verdict="nonsense", comparator="nonsense",
                                metric=None, threshold=None))
    row = mem.recent_evidence()[0]
    assert row["verdict"] == "inconclusive" and row["comparator"] == "ge"
    assert row["metric"] is None


def test_recent_evidence_filter_and_stats(mem):
    mem.record_evidence("a04", "coder", 1, None, "h", _as_row())
    mem.record_evidence("a07", "reviewer", 1, None, "h",
                        _as_row(verdict="refuted", exit_code=1))
    mem.record_evidence("a07", "reviewer", 1, None, "h",
                        _as_row(verdict="inconclusive", exit_code=124))
    assert [r["agent"] for r in mem.recent_evidence(agent="a04")] == ["a04"]
    assert mem.evidence_stats() == {"total": 3, "confirmed": 1, "refuted": 1,
                                    "inconclusive": 1}


def test_evidence_table_is_trimmed(mem, monkeypatch):
    monkeypatch.setattr(sm, "_EVIDENCE_TRIM_EVERY", 10)
    monkeypatch.setattr(sm, "_EVIDENCE_KEEP_ROWS", 6)
    for i in range(30):
        mem.record_evidence("a04", "coder", i, None, f"h{i}", _as_row())
    rows = mem.recent_evidence(limit=100)
    assert len(rows) == 6
    assert rows[0]["hypothesis"] == "h29"          # newest survives
    assert rows[-1]["hypothesis"] == "h24"         # oldest surviving
