"""Reviewer independent re-run (P16/H1) + verified-evidence feed (P16/H3).

H1: a real reviewer RUNS the author's declared checks in their own workspace
against the MR branch — evidence recorded with method="reviewer_rerun".
H3: the swarm's verified knowledge flows into every plan prompt.
"""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402

from agentics.agents.roles import Role  # noqa: E402
from agentics.graph import nodes  # noqa: E402
from agentics.memory.shared_memory import SharedMemory  # noqa: E402
from agentics.verify_ops import cmds_from_evidence, verdict_for  # noqa: E402


@pytest.fixture()
def mem(tmp_path):
    m = SharedMemory(tmp_path / "m.db")
    yield m
    m.close()


class FakeLLM:
    def __init__(self, answer: str):
        self.answer, self.account = answer, None

    async def ask(self, *args, **kwargs):
        return SimpleNamespace(answer=self.answer, citations=[])


class FakeReviewerWS:
    """Canned run_checks_on_branch — records what the reviewer tried."""

    def __init__(self, results):
        self.results, self.calls = results, []

    def run_checks_on_branch(self, branch, cmds, timeout=300):
        self.calls.append((branch, [list(c) for c in cmds]))
        return [(cmd, code, out, 0.1) for cmd, code, out, _s in self.results]


def _ctx(mem, ws, llm_answer):
    mq = SimpleNamespace(enqueued=[])
    mq.enqueue = lambda mr_id, author: mq.enqueued.append((mr_id, author))
    return SimpleNamespace(
        agent_id="a07", role=Role.REVIEWER, memory=mem, llm=FakeLLM(llm_answer),
        meta={}, cycle=3, settings=SimpleNamespace(research_topic="NFV"),
        workspace=ws, merge_queue=mq, local=SimpleNamespace(guard=None),
        report=lambda **k: None,
    )


def _mr_with_author_evidence(mem, exit_code=0):
    mr_id = mem.open_review(
        task_id=1, title="[a04] retry budget (#1)", body="b", author="a04",
        branch="agent/a04/task-1-x", files=["experiments/check.py"],
        verification="tier-1 ok",
    )
    mem.record_evidence(
        "a04", 1, "python experiments/check.py", exit_code,
        "confirmed" if exit_code == 0 else "refuted",
        hypothesis="retry budget cuts failures below 0.9",
    )
    return mr_id


def _approve_json():
    return '{"verdict": "approve", "comment": "looks fine"}'


def test_reviewer_reruns_author_checks_and_records_evidence(mem):
    _mr_with_author_evidence(mem)
    ws = FakeReviewerWS([(["python", "experiments/check.py"], 0, "all green", 0.1)])
    state = asyncio.run(nodes.node_review(
        _ctx(mem, ws, _approve_json()), {}))
    assert state["review_verdict"] == "approve"
    assert ws.calls == [("agent/a04/task-1-x", [["python", "experiments/check.py"]])]
    rerun = [e for e in mem.latest_evidence(10) if e["agent"] == "a07"]
    assert len(rerun) == 1
    assert rerun[0]["method"] == "reviewer_rerun"
    assert rerun[0]["verdict"] == "confirmed"
    assert rerun[0]["hypothesis"] == "retry budget cuts failures below 0.9"
    assert "[P16 evidence:" in state["review_comment"]


def test_refuted_rerun_overrides_the_reviewing_llm(mem):
    _mr_with_author_evidence(mem, exit_code=1)
    ws = FakeReviewerWS([(["python", "experiments/check.py"], 1, "assertion failed", 0.1)])
    state = asyncio.run(nodes.node_review(
        _ctx(mem, ws, _approve_json()), {}))
    assert state["review_verdict"] == "request_changes"
    assert "evidence overrides model opinion" in state["review_comment"]
    assert not any(mr["status"] == "approved" for mr in mem.recent_reviews(5))


def test_no_author_evidence_means_no_rerun(mem):
    mem.open_review(task_id=2, title="[a03] docs (#2)", body="b", author="a03",
                    branch="agent/a03/task-2-d", files=[], verification="")
    ws = FakeReviewerWS([])
    state = asyncio.run(nodes.node_review(
        _ctx(mem, ws, _approve_json()), {}))
    assert state["review_verdict"] == "approve" and ws.calls == []


# ------------------------------------------------------------ helpers
def test_cmds_from_evidence_dedupes_and_excludes_self(mem):
    _mr_with_author_evidence(mem)
    mem.record_evidence("a07", 1, "python own.py", 0, "confirmed")
    mem.record_evidence("a04", 1, "python experiments/check.py", 0, "confirmed")
    rows = mem.evidence_for_task(1)
    out = cmds_from_evidence(rows, exclude_agent="a07")
    assert [c for c, _, _ in out] == ["python experiments/check.py"]
    assert out[0][2] == "retry budget cuts failures below 0.9"   # hypothesis rides along
    assert out[0][1] == ["python", "experiments/check.py"]        # shlex argv


def test_verdict_for_matches_verify_ops_semantics():
    assert verdict_for(0) == "confirmed"
    assert verdict_for(124) == "inconclusive"     # timeout can never pose as confirmed
    assert verdict_for(126) == "inconclusive"     # guard-blocked likewise
    assert verdict_for(1) == "refuted"


# ------------------------------------------------------------ H3 feed
def test_memory_context_carries_verified_knowledge(mem):
    mem.add_hypothesis(
        "a04", "retry budget cuts failures below 0.9",
        {"metric": "fail_rate", "op": "<", "target": 0.9},
        task_id=1, status="supported",
    )
    ctx = SimpleNamespace(memory=mem)
    feed = nodes._memory_context(ctx)["verified"]
    assert "SUPPORTED" in feed and "retry budget cuts failures below 0.9" in feed


def test_memory_context_verified_is_empty_before_any_runs(mem):
    ctx = SimpleNamespace(memory=mem)
    assert nodes._memory_context(ctx)["verified"] == "(nothing verified yet)"


def test_plan_prompt_contains_verified_feed(mem, monkeypatch):
    captured = []

    async def fake_ask(ctx, prompt, mode="concise", purpose="general", **kwargs):
        captured.append(prompt)
        return ('{"files": [], "steps": [], "commit_message": "m"}', [])

    monkeypatch.setattr(nodes, "_llm_ask", fake_ask)
    mem.add_hypothesis(
        "a04", "retry budget cuts failures below 0.9",
        {"metric": "fail_rate", "op": "<", "target": 0.9},
        task_id=1, status="supported",
    )
    ctx = SimpleNamespace(
        agent_id="a04", role=Role.CODER, memory=mem, llm=None, meta={},
        cycle=1, settings=SimpleNamespace(research_topic="NFV"),
        workspace=FakeReviewerWS([]), local=None,
        report=lambda **k: None,
    )
    state = {
        "issue_number": 1, "issue_title": "retry budget", "issue_body": "b",
        "papers": [{"answer": "s"}], "category": "code",
        "memory_context": nodes._memory_context(ctx),
    }
    asyncio.run(nodes.node_plan(ctx, state))
    assert "Verified knowledge of this lab" in captured[0]
    assert "retry budget cuts failures below 0.9" in captured[0]
