"""Traces, tool registry and the persistent merge queue — no network needed."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from agentics.evolution.traces import (
    build_evolution_issue,
    evolution_path_allowed,
    tool_context,
    trace_context,
    valid_tool_name,
)
from agentics.memory.shared_memory import SharedMemory
from agentics.policies import merge_queue as mq_mod
from agentics.policies.merge_queue import MergeQueue


@pytest.fixture()
def mem(tmp_path: Path):
    m = SharedMemory(tmp_path / "m.db")
    yield m
    m.close()


# ------------------------------------------------------------ traces
def test_trace_lifecycle_and_terminal_outcomes_only(mem):
    mem.add_trace(
        "a04",
        "coder",
        1,
        "pending",
        issue=5,
        category="code",
        pr_number=11,
        attempted="add parser",
        files=["agentics/x.py"],
    )
    mem.add_trace("a04", "coder", 2, "yield", issue=6, category="docs")  # not judged
    mem.add_trace("a05", "coder", 1, "failed", issue=7, category="code", failure="pytest red")
    assert mem.agent_outcomes() == {"a05": [("code", "failed")]}  # pending/yield ignored

    mem.set_trace_outcome(11, "approved")
    mem.set_trace_outcome(11, "merged")
    assert mem.agent_outcomes()["a04"] == [("code", "merged")]

    mem.add_trace("a06", "coder", 1, "pending", issue=8, category="code", pr_number=12)
    mem.set_trace_outcome(12, "rejected", "missing tests for new module")
    t = [x for x in mem.recent_traces() if x["pr_number"] == 12][0]
    assert t["outcome"] == "rejected" and "missing tests" in t["failure"]  # reason is kept


def test_trace_stats(mem):
    for outcome in ("merged", "merged", "rejected", "failed", "yield"):
        mem.add_trace("a04", "coder", 1, outcome, category="code")
    st = mem.trace_stats()
    assert st["judged"] == 4 and st["merge_rate"] == 0.5
    assert st["by_category"]["code"]["merged"] == 2
    assert mem.trace_stats(since_ts=9e12)["merge_rate"] is None


def test_trace_context_shares_others_experience_but_not_your_own(mem):
    mem.add_trace(
        "a04",
        "coder",
        1,
        "failed",
        issue=5,
        category="code",
        attempted="rewrite parser",
        failure="regex catastrophic backtracking",
    )
    mem.add_trace(
        "a05",
        "coder",
        1,
        "merged",
        issue=6,
        category="code",
        attempted="use json.loads with tolerant fallback",
    )
    mem.add_trace("a06", "coder", 1, "failed", issue=9, category="docs", failure="unrelated")

    ctx = trace_context(mem, agent_id="a06", issue=5, category="code")
    assert "backtracking" in ctx  # failure of a04 on the same issue
    assert "tolerant fallback" in ctx  # approach that got merged
    assert "unrelated" not in ctx  # own trace, other category
    assert trace_context(mem, "a04", 5, "code").count("backtracking") == 0  # never echo own
    assert trace_context(mem, "a01", None, "research") == "(no traces yet)"


# ------------------------------------------------------------ tool registry
def test_registry_only_lists_merged_tools(mem):
    assert mem.register_tool(
        "snippet_extract",
        "agentics/tools/snippet.py",
        "extract code snippets around a line range",
        "a04",
        3,
        pr_number=21,
    )
    assert not mem.register_tool("snippet_extract", "other.py", "dup", "a05")  # name taken
    assert mem.list_tools("snippet") == []  # PR not merged yet
    assert tool_context(mem, "snippet") == "(registry empty)"

    assert mem.activate_tools(21) == 1
    assert [t["name"] for t in mem.list_tools("extract snippet")] == ["snippet_extract"]


def test_rejected_pr_never_pollutes_registry(mem):
    mem.register_tool("ghost_tool", "agentics/ghost.py", "does nothing", "a04", pr_number=30)
    assert mem.drop_tools(30) == 1
    mem.activate_tools(30)
    assert mem.list_tools("ghost") == []


def test_registry_ranks_by_query_then_popularity(mem):
    mem.register_tool(
        "bibtex_fmt", "agentics/research/bib.py", "format citations as bibtex", "a1", pr_number=1
    )
    mem.register_tool(
        "diff_stats", "agentics/diffstat.py", "summarise git diff sizes", "a1", pr_number=2
    )
    mem.activate_tools(1)
    mem.activate_tools(2)
    assert mem.list_tools("citations bibtex")[0]["name"] == "bibtex_fmt"
    assert mem.list_tools("git diff")[0]["name"] == "diff_stats"
    mem.bump_tool_use("diff_stats")
    assert mem.list_tools("")[0]["name"] == "diff_stats"  # no query -> most used first
    mem.bump_tool_use("does_not_exist")  # silent no-op


def test_tool_name_validation():
    assert valid_tool_name("snippet_extract")
    for bad in ("", "Bad-Name", "x", "../etc/passwd", "a" * 80, "1abc"):
        assert not valid_tool_name(bad)


# ------------------------------------------------------------ evolution guard / issue
def test_evolution_allowlist():
    assert evolution_path_allowed("agentics/agents/roles.py")
    assert evolution_path_allowed("tests/test_roles.py")
    assert evolution_path_allowed("docs/EVOLUTION.md")
    for forbidden in (
        "agentics/stop.py",
        "agentics/policies/ownership.py",
        "agentics/github_ops/pr_ops.py",
        "agentics/llm/pool.py",
        "agentics/agents/roles.py.bak",
        ".github/workflows/x.yml",
    ):
        assert not evolution_path_allowed(forbidden), forbidden


def test_evolution_issue_embeds_evidence(mem):
    mem.add_trace(
        "a04", "coder", 1, "failed", category="code", attempted="x", failure="plan parsing failed"
    )
    title, body = build_evolution_issue(mem)
    assert title.startswith("self-evolution")
    assert "plan parsing failed" in body and "agentics/agents/roles.py" in body


# ------------------------------------------------------------ persistent merge queue
class _FakePR:
    def __init__(self):
        self.comments = []

    def create_issue_comment(self, text):
        self.comments.append(text)


def _queues(mem):
    pr = _FakePR()
    gh = SimpleNamespace(repo=SimpleNamespace(get_pull=lambda n: pr))
    # two separate instances == the reviewer's context and the integrator's context
    return MergeQueue(gh, mem), MergeQueue(gh, mem), pr


def test_queue_is_shared_between_agent_contexts(mem, monkeypatch):
    reviewer_q, integrator_q, _ = _queues(mem)
    monkeypatch.setattr(mq_mod.pr_ops, "merge_when_ready", lambda *a, **k: True)
    mem.add_trace("a04", "coder", 1, "approved", pr_number=40, category="code")
    mem.register_tool("t_one", "agentics/t.py", "a tool", "a04", pr_number=40)

    reviewer_q.enqueue(40, "a04")
    reviewer_q.enqueue(40, "a04")  # idempotent
    assert [i.pr_number for i in integrator_q.pending()] == [40]  # the old design saw []
    assert integrator_q.drain("a09") == 1
    assert integrator_q.pending() == []
    assert mem.agent_outcomes()["a04"] == [("code", "merged")]
    assert [t["name"] for t in mem.list_tools("")] == ["t_one"]  # tool became active on merge


def test_tier2_failure_blocks_merge_and_teaches_the_group(mem, monkeypatch):
    q1, q2, pr = _queues(mem)
    called = []
    monkeypatch.setattr(mq_mod.pr_ops, "merge_when_ready", lambda *a, **k: called.append(1) or True)
    mem.add_trace("a04", "coder", 1, "approved", pr_number=41, category="code")
    mem.register_tool("t_two", "agentics/t2.py", "a tool", "a04", pr_number=41)
    q1.enqueue(41, "a04")

    assert q2.drain("a09", verify=lambda n: (False, "tests failed on main+PR (exit 1)")) == 0
    assert not called  # never reached the merge call
    assert mem.mq_pending() == []
    assert mem.agent_outcomes()["a04"] == [("code", "failed")]
    fail = mem.recent_traces(failures_only=True)[0]
    assert "tier-2 verification failed" in fail["failure"]
    assert mem.list_tools("") == []  # pending tool dropped
    assert pr.comments and "tier-2" in pr.comments[0]


def test_unmergeable_pr_is_retried_then_given_up(mem, monkeypatch):
    q1, q2, _ = _queues(mem)
    monkeypatch.setattr(mq_mod.pr_ops, "merge_when_ready", lambda *a, **k: False)
    mem.add_trace("a04", "coder", 1, "approved", pr_number=42, category="code")
    q1.enqueue(42, "a04")
    for expected_attempts in (1, 2):
        q2.drain("a09")
        assert q2.pending()[0].attempts == expected_attempts
    q2.drain("a09")  # 3rd failure -> give up
    assert q2.pending() == []
    assert mem.agent_outcomes()["a04"] == [("code", "failed")]
