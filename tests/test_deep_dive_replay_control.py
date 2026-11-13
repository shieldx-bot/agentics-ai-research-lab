"""Deep-dive + replay flood-control tests (2026-11-14 live regression).

Live symptoms this suite locks in:
- two replayable claims were re-run by BOTH supervisors every sweep ->
  hundreds of duplicate evidence rows (cooldown + cap fix);
- 4 `proposed` hypotheses sat untested for 400+ cycles while researchers
  skipped the saturated topic (registry deep-dive fix).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agentics.memory.shared_memory import SharedMemory  # noqa: E402
from agentics.verify.replay import replay_settled  # noqa: E402


@pytest.fixture()
def mem(tmp_path):
    return SharedMemory(tmp_path / "m.db")


def _settled(mem, claim: str, code: str = "print('x')") -> int:
    return mem.add_hypothesis("a01", claim, {"metric": "m", "op": ">=", "target": 1},
                              status="supported", protocol_code=code)


def _replay_evidence(mem, hyp_id: int, agent: str = "a07",
                     ts_offset: float = 0.0) -> int:
    ev_id = mem.record_evidence(
        agent, None, f"replay_h{hyp_id}.py", 0, "confirmed",
        hypothesis="c", method="replay")
    # the hypothesis link is created by the protocol-outcome attach, exactly
    # like the real replay engine does (record_evidence leaves it NULL)
    mem.attach_protocol_outcome(ev_id, {}, [], hyp_id)
    # evidence.ts is set to now at insert; backdate directly for cooldown tests
    with mem._lock, mem._conn:
        mem._conn.execute("UPDATE evidence SET ts=? WHERE id=?",
                          (time.time() - ts_offset, ev_id))
    return ev_id


def test_settled_hypotheses_cooldown_skips_recent_replays(mem):
    h1 = _settled(mem, "claim one")
    _replay_evidence(mem, h1, ts_offset=60)          # replayed 1 min ago
    h2 = _settled(mem, "claim two")
    _replay_evidence(mem, h2, ts_offset=3600)        # replayed 1 h ago
    got = mem.settled_hypotheses(limit=10, require_code=True, cooldown_s=1800)
    assert [h["id"] for h in got] == [h2]            # h1 is on cooldown


def test_settled_hypotheses_max_replays_caps_repeat_replays(mem):
    h1 = _settled(mem, "over-replayed claim")
    for _ in range(3):
        _replay_evidence(mem, h1)
    got = mem.settled_hypotheses(limit=10, require_code=True, max_replays=3)
    assert got == []
    got = mem.settled_hypotheses(limit=10, require_code=True, max_replays=5)
    assert [h["id"] for h in got] == [h1]


def test_settled_hypotheses_no_filters_keeps_everything(mem):
    h1 = _settled(mem, "recent claim")
    _replay_evidence(mem, h1, ts_offset=10)
    got = mem.settled_hypotheses(limit=10, require_code=True)
    assert [h["id"] for h in got] == [h1]


def test_proposed_hypotheses_oldest_first_no_code(mem):
    a = mem.add_hypothesis("a02", "oldest proposed", {"metric": "m", "op": ">=", "target": 1},
                           status="proposed", protocol_code="")
    _settled(mem, "settled claim")                                    # not proposed
    mem.add_hypothesis("a03", "has code", {"metric": "m", "op": ">=", "target": 1},
                       status="proposed", protocol_code="print('x')")  # has protocol
    b = mem.add_hypothesis("a02", "newer proposed", {"metric": "m", "op": ">=", "target": 1},
                           status="proposed", protocol_code="")
    got = mem.proposed_hypotheses(limit=3)
    assert [h["id"] for h in got] == [a, b]


def test_replay_settled_respects_cooldown(mem, tmp_path):
    """Sweep must come back EMPTY for a claim replayed moments ago."""
    h1 = _settled(mem, "recently replayed claim")
    _replay_evidence(mem, h1, ts_offset=30)

    class NoopWs:
        def write_file(self, path, code):
            raise ValueError("must not be called — claim is on cooldown")

    report = replay_settled(mem, NoopWs(), limit=5, agent="a07",
                            cooldown_s=1800, max_replays=10)
    assert all(not r.get("replayed") for r in report)


def test_deep_dive_helper_skips_empty_registry(tmp_path):
    """With no proposed hypotheses the helper must report 'nothing to do'."""
    from agentics.graph.nodes import _test_proposed_hypothesis

    mem = SharedMemory(tmp_path / "m2.db")

    class FakeMem:
        def proposed_hypotheses(self, limit):
            return mem.proposed_hypotheses(limit)

    ctx = SimpleNamespace(llm=SimpleNamespace(), memory=FakeMem(),
                          workspace=None, agent_id="a02")
    state: dict = {}
    import asyncio
    assert asyncio.run(_test_proposed_hypothesis(ctx, state)) is False
