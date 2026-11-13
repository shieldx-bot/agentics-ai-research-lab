"""Reviewer handoff dossier (A2X module 2) — the verifier sees the past.

The reviewer's prompt used to carry only the diff, the author's recorded
verification and the P16 evidence rows. The task's own history — earlier
failed attempts, the objections a panel already raised on earlier MRs for
this very task — was stored but never handed over, so an MR could re-run a
known-failed approach and pass review. These tests pin:

* the dossier carries earlier attempts (agent, outcome, failure note),
* it carries ONLY prior MR objections (never the MR under review itself),
* it is empty for a fresh task (cold-start prompts stay unchanged),
* it is bounded, and node_review really injects it into the prompt.
"""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentics.agents.roles import Role
from agentics.coordination.dossier import review_dossier
from agentics.graph import nodes
from agentics.memory.shared_memory import SharedMemory


@pytest.fixture()
def mem(tmp_path: Path):
    m = SharedMemory(tmp_path / "m.db")
    yield m
    m.close()


def test_dossier_carries_attempts_and_prior_objections(mem):
    task = mem.create_task("Add retry backoff", category="code")
    mem.add_trace("a01", "coder", 1, "failed", issue=task, category="code",
                  attempted="fixed sleep backoff", failure="thundering herd: all retries land at once")
    mem.add_trace("a03", "coder", 2, "merged", issue=task, category="code",
                  attempted="jittered exponential backoff")
    prior = mem.open_review(task, "MR one", "b", author="a01",
                            branch="agent/a01/task-1-x", files=["experiments/r.py"])
    mem.set_review_verdict(prior, "changes_requested", "a07", "no test for the herd case")

    text = review_dossier(mem, task)
    assert "Earlier attempts on this task:" in text
    assert "a01: failed — thundering herd" in text
    assert "a03: merged" in text
    assert f"!#{prior} changes_requested: no test for the herd case" in text


def test_dossier_excludes_the_mr_under_review(mem):
    task = mem.create_task("Add retry backoff", category="code")
    first = mem.open_review(task, "MR one", "b", author="a01",
                            branch="agent/a01/b1", files=["experiments/r.py"])
    mem.set_review_verdict(first, "changes_requested", "a07", "tests missing")
    second = mem.open_review(task, "MR two", "b", author="a01",
                             branch="agent/a01/b2", files=["experiments/r.py"])

    text = review_dossier(mem, task, mr_id=second)
    assert f"!#{first}" in text and f"!#{second}" not in text
    # an OPEN MR is never an objection (only changes_requested / rejected /
    # failed verdicts carry one), boundary or not
    assert f"!#{second}" not in review_dossier(mem, task)


def test_dossier_empty_for_fresh_task(mem):
    task = mem.create_task("Brand new idea", category="code")
    assert review_dossier(mem, task) == ""
    assert review_dossier(mem, None) == ""


def test_dossier_is_bounded(mem):
    task = mem.create_task("Add retry backoff", category="code")
    for i in range(6):
        mem.add_trace(f"a0{i % 5 + 1}", "coder", i, "failed", issue=task,
                      category="code", failure="x" * 300)
    text = review_dossier(mem, task, max_attempts=3, max_chars=500)
    assert len(text) <= 500
    assert text.endswith("…")


def test_node_review_prompt_includes_task_history(mem, monkeypatch):
    """The wiring: reviewing MR #2 of a twice-attempted task carries the
    dossier (the communication bridge), reviewing a fresh task does not."""
    seen = {}

    async def fake_llm(ctx, prompt, mode="concise", purpose="general", state=None):
        seen["p"] = prompt
        return json.dumps({"verdict": "approve", "comment": "lgtm"}), []

    monkeypatch.setattr(nodes, "_llm_ask", fake_llm)

    task = mem.create_task("Add retry backoff", category="code")
    mem.add_trace("a01", "coder", 1, "failed", issue=task, category="code",
                  failure="thundering herd: all retries land at once")
    prior = mem.open_review(task, "MR one", "b", author="a01",
                            branch="agent/a01/b1", files=["experiments/r.py"])
    mem.set_review_verdict(prior, "changes_requested", "a07", "tests missing")
    mr_id = mem.open_review(task, "MR two", "b", author="a01",
                            branch="agent/a01/b2", files=["experiments/r.py"])

    ctx = SimpleNamespace(
        agent_id="a07", role=Role.REVIEWER, memory=mem, claims=None, cycle=1,
        meta={"personal": None},
        settings=SimpleNamespace(research_topic="NFV", evolution_every_n_cycles=0),
        heartbeat=lambda status="working": None, report=lambda **kw: None,
        local=None, workspace=None,
        merge_queue=SimpleNamespace(enqueue=lambda mr, author: None),
    )
    state = asyncio.run(nodes.node_review(ctx, {}))
    assert state["review_verdict"] in ("approve", "vote_cast")
    assert "Task history (earlier attempts and prior objections" in seen["p"]
    assert "thundering herd" in seen["p"]
    assert "!#" in seen["p"] and "tests missing" in seen["p"]

    # fresh task: no dossier block at all — the prompt stays byte-compatible
    fresh = mem.create_task("Fresh task", category="code")
    mem.open_review(fresh, "MR fresh", "b", author="a02",
                    branch="agent/a02/b", files=["experiments/f.py"])
    seen.clear()
    asyncio.run(nodes.node_review(ctx, {}))
    assert "Task history" not in seen["p"]
