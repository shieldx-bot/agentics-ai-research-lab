"""P19 — node_defend: the author answers the panel with runs, not prose.

The defense round is the swarm's shared answer to "can you actually defend
your point?": every objection must be answered, every declared extra check
really runs in the author's own workspace, and anything less ends the MR.
"""

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402

from agentics.agents.roles import Role  # noqa: E402
from agentics.graph import nodes  # noqa: E402
from agentics.memory.shared_memory import SharedMemory  # noqa: E402
from agentics.peer_review import PeerPanel  # noqa: E402


@pytest.fixture()
def mem(tmp_path):
    m = SharedMemory(tmp_path / "m.db")
    yield m
    m.close()


class FakeWS:
    """run_command canned — records what the defense really executed."""

    def __init__(self, code=0, out="all green"):
        self.code, self.out, self.calls = code, out, []

    def run_command(self, argv, timeout=300):
        self.calls.append(argv)
        return self.code, self.out


def _ctx(mem, ws, answer=None, error=False):
    async def fake_llm(ctx_, prompt, mode="concise", purpose="defend", state=None):
        if error:
            state["error"] = "LLM unreachable"
            return "", []
        return answer, []

    return SimpleNamespace(
        agent_id="a04", role=Role.CODER, memory=mem,
        llm=SimpleNamespace(answer="", account=None), meta={}, cycle=2,
        settings=SimpleNamespace(research_topic="NFV", peer_panel_size=2,
                                 peer_defense_ttl_minutes=30),
        workspace=ws, local=None, merge_queue=SimpleNamespace(enqueue=lambda *a, **k: None),
        panel=PeerPanel(mem, panel_size=2, defense_ttl_minutes=30),
        report=lambda **k: None,
    ), fake_llm


def _defense_state(mem, panel_size=2):
    """Round 1 done: one accept + one reject -> the MR is in defense."""
    for agent in ("a04", "a07", "a09"):
        mem.heartbeat(agent, "coder", 1)
    mr_id = mem.open_review(task_id=mem.create_task("work"), title="[a04] work",
                            body="b", author="a04", branch="agent/a04/task-1-x",
                            files=["experiments/a.py"])
    panel = PeerPanel(mem, panel_size=panel_size, defense_ttl_minutes=30)
    panel.cast_vote(mr_id, "a07", "accept")
    panel.cast_vote(mr_id, "a09", "reject", "metric not measured on the branch")
    assert mem.get_panel(mr_id)["state"] == "defense"
    return mr_id, panel


def test_defense_without_pending_mr_is_a_noop(mem):
    ctx, _ = _ctx(mem, FakeWS())
    state = asyncio.run(nodes.node_defend(ctx, {}))
    assert state["summary"] == "no defense pending" and state["done"]


def test_complete_defense_runs_checks_and_reopens_voting(mem, monkeypatch):
    mr_id, panel = _defense_state(mem)
    ws = FakeWS(code=0, out="fail_rate 0.42 < 0.9")
    ctx, fake_llm = _ctx(mem, ws, answer=json.dumps({
        "concede": False,
        "rebuttals": {"a09": "re-ran on the MR branch: fail_rate 0.42 < 0.9"},
        "extra_checks": ["python experiments/check.py"],
        "note": "measurement attached",
    }))
    monkeypatch.setattr(nodes, "_llm_ask", fake_llm)
    state = asyncio.run(nodes.node_defend(ctx, {}))
    assert state["defense_outcome"] == "re_voting"
    # the declared check really ran in the author's workspace
    assert ws.calls == [[sys.executable, "experiments/check.py"]]
    ev = [e for e in mem.latest_evidence(10) if e["method"] == "defense_check"]
    assert len(ev) == 1 and ev[0]["exit_code"] == 0 and ev[0]["agent"] == "a04"
    p = mem.get_panel(mr_id)
    assert p["state"] == "voting" and p["round"] == 2
    assert p["defense"]["rebuttals"]["a09"].startswith("re-ran")
    assert p["defense"]["evidence_ids"] == [ev[0]["id"]]
    assert p["due_ts"] is None


def test_incomplete_rebuttals_end_the_mr_immediately(mem, monkeypatch):
    mr_id, _ = _defense_state(mem)
    ctx, fake_llm = _ctx(mem, FakeWS(), answer=json.dumps({
        "concede": False, "rebuttals": {"a07": "wrong objector"}, "note": ""}))
    monkeypatch.setattr(nodes, "_llm_ask", fake_llm)
    state = asyncio.run(nodes.node_defend(ctx, {}))
    assert state["defense_outcome"] == "rejected"
    assert "defense incomplete" in state["summary"]
    assert mem.get_panel(mr_id)["state"] == "rejected"


def test_concede_ends_the_mr_honestly(mem, monkeypatch):
    mr_id, _ = _defense_state(mem)
    ctx, fake_llm = _ctx(mem, FakeWS(), answer=json.dumps({
        "concede": True, "rebuttals": {}, "note": "the objection is right"}))
    monkeypatch.setattr(nodes, "_llm_ask", fake_llm)
    state = asyncio.run(nodes.node_defend(ctx, {}))
    assert state["defense_outcome"] == "rejected"
    assert "conceded" in state["summary"]


def test_llm_failure_leaves_the_defense_open_for_retry(mem, monkeypatch):
    mr_id, _ = _defense_state(mem)
    ctx, fake_llm = _ctx(mem, FakeWS(), error=True)
    monkeypatch.setattr(nodes, "_llm_ask", fake_llm)
    state = asyncio.run(nodes.node_defend(ctx, {}))
    assert state.get("done") and "defense_outcome" not in state
    assert mem.get_panel(mr_id)["state"] == "defense"   # deadline still looms
    assert mem.panel_votes(mr_id, round=2) == []        # nothing was submitted


def test_blocked_extra_check_is_recorded_as_inconclusive_evidence(mem, monkeypatch):
    """A defense check the sandbox refuses is evidence too — never silent."""
    mr_id, _ = _defense_state(mem)
    ws = FakeWS(code=126, out="command rejected: not allowlisted")
    ctx, fake_llm = _ctx(mem, ws, answer=json.dumps({
        "concede": False,
        "rebuttals": {"a09": "see the attached run output"},
        "extra_checks": ["./run-everything.sh"],   # not allowlisted -> coerced fail
        "note": ""}))
    monkeypatch.setattr(nodes, "_llm_ask", fake_llm)
    state = asyncio.run(nodes.node_defend(ctx, {}))
    assert state["defense_outcome"] == "re_voting"
    ev = [e for e in mem.latest_evidence(10) if e["method"] == "defense_check"]
    assert ev and ev[0]["verdict"] == "inconclusive"
    assert "rejected" in ev[0]["output"]
