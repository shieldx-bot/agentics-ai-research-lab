"""P24bis — the owner surfaces: prompts weigh replicated claims, the
dashboard shows the replication ledger.

The consumption side of H3: a settled claim's independence status must be
VISIBLE where decisions happen — in every agent's prompt (SUPPORTED+ vs
SUPPORTED) and on the owner's dashboard (Vietnamese panel, counts + recent
provenance rows).
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from agentics.config import Settings
from agentics.dashboard.app import _state
from agentics.graph.nodes import verified_knowledge_text
from agentics.memory.shared_memory import SharedMemory

fastapi = pytest.importorskip("fastapi")


@pytest.fixture()
def mem(tmp_path: Path):
    m = SharedMemory(tmp_path / "m.db")
    yield m
    m.close()


def _settled(mem: SharedMemory) -> tuple[int, int]:
    hyp = mem.add_hypothesis(
        "researcher-1", "the method reaches >=0.9 precision",
        {"metric": "precision", "op": ">=", "target": 0.9},
        status="testing", protocol_code="print('x')")
    ev = mem.record_evidence("researcher-1", None, "python p.py", 0,
                             "confirmed", hypothesis="the method reaches >=0.9 precision",
                             method="protocol")
    mem.update_hypothesis(hyp, "supported", evidence_id=ev, note="measured 0.91")
    return hyp, ev


def test_prompt_marks_independently_replicated_claims(mem):
    hyp, ev = _settled(mem)
    ctx = SimpleNamespace(memory=mem)
    text = verified_knowledge_text(ctx)
    assert "[SUPPORTED]" in text and "[SUPPORTED+ (independently replicated)]" not in text

    mem.record_replication(hyp, "confirmed")
    mem.record_replication_detail(ev, hyp, "reviewer-1", "confirmed",
                                  {"precision": 0.93}, [])
    text = verified_knowledge_text(ctx)
    assert "[SUPPORTED+ (independently replicated)]" in text
    assert "strongest kind" in text


def test_dashboard_state_exposes_replications(mem, tmp_path: Path):
    hyp, ev = _settled(mem)
    s = Settings(data_dir=tmp_path / "data")
    s.ensure_dirs()
    state = _state(mem, s)
    assert state["replications"]["counts"] == {
        "replicated": 0, "refuted": 0, "pending": 1}
    assert state["replications"]["recent"] == []

    mem.record_replication(hyp, "confirmed")
    mem.record_replication_detail(ev, hyp, "reviewer-1", "confirmed",
                                  {"precision": 0.93}, [])
    state = _state(mem, s)
    counts = state["replications"]["counts"]
    assert counts["replicated"] == 1 and counts["pending"] == 0
    row = state["replications"]["recent"][0]
    assert row["replicator"] == "reviewer-1"
    assert row["measured"] == {"precision": 0.93}
    # the hypothesis row itself carries the ledger status for the table
    assert state["hypotheses"]["recent"][0]["replication"] == "confirmed"
