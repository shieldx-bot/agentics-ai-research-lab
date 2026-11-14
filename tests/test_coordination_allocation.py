"""Contract-Net-lite allocation (A2X module 1) — the supervisor allocates.

Before this pack the supervisor's scan was a janitor (sweep, top-up) and
task picking was a pure race over self-history profiles: nobody routed work
to the agents best able to do it. These tests pin the new mechanics:

* awards pair the best live claimant per open task (P-N score over real
  merge history — one award per task, one per agent, deterministic),
* the award table is REWRITTEN every scan (a stale award cannot survive),
* node_scan journals + broadcasts the awards it just made,
* node_claim lifts an awarded task by `coord_award_bonus` (the manager's
  award actually moves work) without changing the no-award ordering.

Fake roster via real heartbeats, real SharedMemory, no git, no LLM.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from agentics.agents.roles import Role
from agentics.coordination.allocation import award_digest, recommend_awards
from agentics.graph import nodes
from agentics.local_ops import LocalClaimPolicy
from agentics.memory.shared_memory import SharedMemory


@pytest.fixture()
def mem(tmp_path: Path):
    m = SharedMemory(tmp_path / "m.db")
    yield m
    m.close()


def _beat(mem: SharedMemory, agent: str, role: str) -> None:
    mem.heartbeat(agent, role, cycle=1, status="idle")


def _hist(mem: SharedMemory, agent: str, category: str, outcome: str) -> None:
    mem.add_trace(agent, "coder", 1, outcome, category=category)


def test_awards_pair_the_best_agent_per_task(mem):
    bug = mem.create_task("Fix crash in claims", category="bugfix")
    docs = mem.create_task("Write docs for the bench", category="docs")
    _beat(mem, "a01", "coder")
    _beat(mem, "a02", "coder")
    # a01 is the bugfix expert, a02 the docs expert (mirrored histories)
    for _ in range(3):
        _hist(mem, "a01", "bugfix", "merged")
    _hist(mem, "a01", "docs", "failed")
    for _ in range(3):
        _hist(mem, "a02", "docs", "merged")
    _hist(mem, "a02", "bugfix", "failed")

    awards = recommend_awards(mem)

    by_agent = {a["agent"]: a["task_id"] for a in awards}
    assert by_agent == {"a01": bug, "a02": docs}
    assert mem.awards_for("a01") == {bug: awards[0]["score"]}
    assert mem.awards_for("a02") == {docs: awards[1]["score"]}


def test_awards_one_per_agent_and_deterministic(mem):
    for title in ("Fix bug A", "Fix bug B", "Fix bug C"):
        mem.create_task(title, category="bugfix")
    _beat(mem, "a01", "coder")
    _hist(mem, "a01", "bugfix", "merged")

    first = recommend_awards(mem)
    second = recommend_awards(mem)
    assert first == second
    assert len(first) == 1                     # one award per agent
    assert first[0]["agent"] == "a01"


def test_awards_skip_writer_on_evolution_tasks(mem):
    mem.create_task("self-evolution: refine prompts", category="evolution")
    _beat(mem, "a01", "writer")
    assert recommend_awards(mem) == []         # only the coder may take these

    _beat(mem, "a02", "coder")
    awards = recommend_awards(mem)
    assert len(awards) == 1 and awards[0]["agent"] == "a02"


def test_awards_clear_when_backlog_or_roster_dries_up(mem):
    task = mem.create_task("Fix bug A", category="bugfix")
    mem.replace_task_awards([{"task_id": task, "agent": "a01", "score": 1.0}])
    recommend_awards(mem)                      # no live claimants -> cleared
    assert mem.awards_for("a01") == {}

    _beat(mem, "a01", "coder")
    recommend_awards(mem)
    assert task in mem.awards_for("a01")

    mem.set_task_status(task, "in_progress", claimed_by="a01")   # no longer open
    recommend_awards(mem)
    assert mem.awards_for("a01") == {}         # claimed tasks are un-awarded


def test_node_scan_journals_and_broadcasts_awards(mem):
    task = mem.create_task("Fix crash in worker loop", category="bugfix")
    _beat(mem, "a02", "coder")
    ctx = SimpleNamespace(
        agent_id="a01", role=Role.SUPERVISOR, memory=mem,
        claims=SimpleNamespace(sweep_stale=lambda: []), cycle=1,
        meta={"personal": None},
        settings=SimpleNamespace(research_topic="NFV", evolution_every_n_cycles=0,
                                 replay_sweep_size=0),
        heartbeat=lambda status="working": None,
        report=lambda **kw: None, local=None, workspace=None,
    )
    state = nodes.node_scan(ctx, {})
    assert "Contract-net awards:" in state["summary"]
    assert str(task) in award_digest(mem.awards_for("a02") and
                                     [{"task_id": task, "agent": "a02",
                                       "score": mem.awards_for("a02")[task]}])
    events = [j["event"] for j in mem.recent_journal(limit=20)]
    assert "contract_net_awards" in events
    msgs = [m["content"] for m in mem.recent_messages(limit=5)]
    assert any("Contract-net awards" in m for m in msgs)


def _claim_ctx(mem: SharedMemory, agent_id="a04"):
    class _Personal:
        def add_lesson(self, lesson, issue=None):
            pass

    return SimpleNamespace(
        agent_id=agent_id, role=Role.CODER, memory=mem,
        claims=LocalClaimPolicy(mem, ttl_minutes=60), cycle=3,
        meta={"personal": _Personal()},
        settings=SimpleNamespace(research_topic="NFV", evolution_every_n_cycles=10),
        heartbeat=lambda status="working": None,
        report=lambda **kw: None, local=None,
    )


def test_node_claim_award_lifts_task_over_profile_choice(mem):
    """docs is this agent's proven category; the supervisor's award on the
    bugfix task must flip the choice (that IS the allocation working)."""
    bug = mem.create_task("Fix crash in claims", category="bugfix")
    docs = mem.create_task("Write docs for policies", category="docs")
    _hist(mem, "a04", "bugfix", "failed")
    _hist(mem, "a04", "bugfix", "failed")
    _hist(mem, "a04", "docs", "merged")

    without_award = nodes.node_claim(_claim_ctx(mem), {})
    assert without_award["issue_number"] == docs       # profile alone picks docs

    mem.set_task_status(docs, "open", claimed_by="")   # reset both to open
    mem.set_task_status(bug, "open", claimed_by="")
    mem.replace_task_awards([{"task_id": bug, "agent": "a04", "score": 0.15}])
    with_award = nodes.node_claim(_claim_ctx(mem), {})
    assert with_award["issue_number"] == bug           # the award flipped it
    reason = mem._conn.execute(
        "SELECT reason FROM decisions ORDER BY id DESC LIMIT 1").fetchone()[0]
    assert "+award" in reason


def test_node_claim_without_award_keeps_profile_order_untouched(mem):
    bug = mem.create_task("Fix crash in claims", category="bugfix")
    docs = mem.create_task("Write docs for policies", category="docs")
    _hist(mem, "a04", "bugfix", "failed")
    _hist(mem, "a04", "bugfix", "failed")
    _hist(mem, "a04", "docs", "merged")
    # an award aimed at SOMEONE ELSE must not influence my ranking
    mem.replace_task_awards([{"task_id": bug, "agent": "a99", "score": 0.15}])
    out = nodes.node_claim(_claim_ctx(mem, agent_id="a04"), {})
    assert out["issue_number"] == docs
    assert bug not in mem.awards_for("a04")
