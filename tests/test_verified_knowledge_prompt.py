"""H3 consumption — the swarm's verified knowledge reaches every agent's prompt.

The memory layer settles hypotheses (supported/refuted, exit-code verdicts);
`verified_knowledge_text` renders them into prompt text and node_research /
node_plan inject it. These tests prove the full loop with the real memory
API: one agent's confirmed experiment changes what every other agent reads.
"""

import pytest

from agentics.graph.nodes import verified_knowledge_text
from agentics.memory.shared_memory import SharedMemory


@pytest.fixture()
def lab(tmp_path):
    mem = SharedMemory(tmp_path / "mem.db")
    ctx = _Ctx(mem)
    yield mem, ctx
    mem.close()


class _Ctx:
    """Minimal stand-in — the helper only touches ctx.memory."""

    def __init__(self, mem):
        self.memory = mem


def _settle(mem, claim, status, agent="a03", task_id=9, note=""):
    """Register a hypothesis and settle it the way the lifecycle does."""
    hyp_id = mem.add_hypothesis(agent, claim, {"metric": "acc", "op": ">=", "target": 0.9},
                                task_id=task_id)
    assert mem.update_hypothesis(hyp_id, status, note=note or f"exit-code check")
    return hyp_id


def test_cold_start_returns_empty_text(lab):
    mem, ctx = lab
    assert verified_knowledge_text(ctx) == ""


def test_supported_and_refuted_claims_are_rendered(lab):
    mem, ctx = lab
    _settle(mem, "GBDT beats LSTM on the 5-min failure horizon", "supported", note="acc 0.93>0.9")
    _settle(mem, "Sliding windows of 60s are enough", "refuted", note="F1 dropped 12%")
    text = verified_knowledge_text(ctx)
    assert "SUPPORTED" in text and "GBDT beats LSTM" in text
    assert "REFUTED" in text and "Sliding windows" in text
    assert "exit-code verdicts" in text            # provenance reminder


def test_proposed_hypotheses_do_not_leak_into_prompts(lab):
    mem, ctx = lab
    mem.add_hypothesis("a03", "still being tested", {"metric": "x", "op": ">", "target": 1})
    assert verified_knowledge_text(ctx) == ""


def test_bounded_even_with_a_long_history(lab):
    mem, ctx = lab
    for i in range(12):
        _settle(mem, f"claim number {i} with a fairly long sentence " * 3,
                "supported", task_id=i)
    text = verified_knowledge_text(ctx, limit=6)
    assert len(text) <= 1400
    assert text.count("- [SUPPORTED]") == 6        # newest window only


def test_failing_memory_never_breaks_the_cycle(lab):
    class Boom:
        def verified_knowledge(self, limit=20):
            raise RuntimeError("db locked")
    ctx = _Ctx(Boom())
    assert verified_knowledge_text(ctx) == ""
