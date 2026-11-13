"""Rework lifecycle at the node level — a changes_requested MR is never orphaned.

    request_changes → the AUTHOR picks the rework up first (no new claim,
    task parked in in_review) → research skipped → plan carries the feedback
    → implement resumes the pushed branch → the SAME MR is reopened.

Fake LLM, real SharedMemory, no git needed except where noted.
"""

from types import SimpleNamespace

import pytest

from agentics.agents.roles import Role
from agentics.graph import nodes
from agentics.local_ops import LocalClaimPolicy
from agentics.memory.shared_memory import SharedMemory


@pytest.fixture()
def mem(tmp_path):
    m = SharedMemory(tmp_path / "m.db")
    yield m
    m.close()


def _ctx(mem, role=Role.CODER, agent_id="a04", tasks_src=None):
    return SimpleNamespace(
        agent_id=agent_id,
        role=role,
        memory=mem,
        claims=LocalClaimPolicy(mem, ttl_minutes=60),
        cycle=3,
        meta={"personal": SimpleNamespace(add_lesson=lambda *a, **k: None)},
        settings=SimpleNamespace(research_topic="NFV", evolution_every_n_cycles=0),
        heartbeat=lambda status="working": None,
        report=lambda **kw: None,
        local=SimpleNamespace(tasks=tasks_src) if tasks_src else SimpleNamespace(tasks=None),
    )


def _changes_requested_mr(mem, author="a04", task_status=None):
    tid = mem.create_task("bench skeleton", body="make bench", category="code")
    mr_id = mem.open_review(tid, f"[{author}] bench (#{tid})", "b",
                            author=author, branch=f"agent/{author}/task-{tid}-bench",
                            files=["experiments/bench.py"])
    mem.set_review_verdict(mr_id, "changes_requested", "a07", "return 42 please")
    if task_status:                        # applied after open_review (it flips in_review)
        mem.set_task_status(tid, task_status)
    return tid, mr_id


def test_node_claim_reworks_own_mr_before_claiming_new_work(mem):
    tid, mr_id = _changes_requested_mr(mem, task_status="in_review")
    fresh = mem.create_task("another fresh task")          # also claimable
    state = {"done": False}
    out = nodes.node_claim(_ctx(mem), state)

    assert out.get("rework_mr") == mr_id                   # rework wins over fresh claim
    assert out["branch"] == f"agent/a04/task-{tid}-bench"  # same branch, not a new one
    assert "Review feedback" in out["issue_body"] and "return 42" in out["issue_body"]
    assert out["issue_number"] == tid
    # no claim was taken; the task stays parked in review
    assert mem.active_claims() == {}
    assert mem.get_task(tid)["status"] == "in_review"
    assert mem.get_task(fresh)["status"] == "open"
    events = [j["event"] for j in mem.recent_journal()]
    assert "rework_start" in events


def test_node_claim_skips_foreign_mr_and_closed_tasks(mem):
    _changes_requested_mr(mem, author="a09", task_status="in_review")   # not mine
    _changes_requested_mr(mem, author="a04", task_status="merged")      # mine, but closed
    state = {"done": False}
    nodes.node_claim(_ctx(mem), state)
    assert "rework_mr" not in state                      # nothing reworkable -> normal claim


def test_node_research_skips_search_for_rework(mem, monkeypatch):
    import asyncio

    async def fail(*a, **k):                             # any search attempt is a bug
        raise AssertionError("rework must not search papers")

    monkeypatch.setattr(nodes.papers_mod, "dedup_query", fail)
    out = asyncio.run(nodes.node_research(_ctx(mem), {"rework_mr": 7, "done": False}))
    assert out["action"] == "plan" and not out.get("done")


def test_node_open_pr_reopens_the_same_mr_instead_of_a_second_one(mem):
    tid, mr_id = _changes_requested_mr(mem, task_status="in_review")
    ctx = _ctx(mem)
    state = {"rework_mr": mr_id, "issue_number": tid, "branch": "agent/a04/b",
             "changed_files": ["experiments/bench.py"],
             "verification": "tier-1 passed", "done": False}
    out = nodes.node_open_pr(ctx, state)

    assert out["pr_number"] == mr_id
    assert out["summary"] == f"reworked MR #{mr_id}"
    review = mem.get_review(mr_id)
    assert review["status"] == "open" and review["reviewer"] == ""
    assert review["files"] == ["experiments/bench.py"]
    assert [r["id"] for r in mem.reviews_by_task(tid)] == [mr_id]   # no MR duplication
    events = [j["event"] for j in mem.recent_journal()]
    assert "rework_mr" in events and "mr_reopened" in events


def test_node_reflect_failed_rework_leaves_the_parked_task_alone(mem):
    tid, mr_id = _changes_requested_mr(mem, task_status="in_review")
    ctx = _ctx(mem)
    state = {"rework_mr": mr_id, "issue_number": tid,
             "error": "tier-1 tests failed", "done": True}
    nodes.node_reflect(ctx, state)

    # no claim existed and none must be created/released; the task keeps its
    # in_review status and the SAME author reworks again next cycle
    assert mem.active_claims() == {}
    assert mem.get_task(tid)["status"] == "in_review"
    events = [j["event"] for j in mem.recent_journal()]
    assert "release" not in events


def test_node_scan_repairs_orphans_via_local_tasks(mem):
    from agentics.config import Settings
    from agentics.local_ops import TaskSource

    tasks_src = TaskSource(mem, Settings(data_dir=None))
    stalled = mem.create_task("stalled claim")
    mem.set_task_status(stalled, "claimed", claimed_by="a04")   # no lease row
    ctx = _ctx(mem, role=Role.SUPERVISOR, tasks_src=tasks_src)
    out = nodes.node_scan(ctx, {"done": False})

    assert mem.get_task(stalled)["status"] == "open"
    assert f"Repaired stalled task(s): [{stalled}]" in out["summary"]
