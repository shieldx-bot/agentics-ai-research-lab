"""Wiring of the new behaviour inside the graph nodes (fake GitHub / LLM)."""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentics.agents.roles import Role
from agentics.graph import nodes
from agentics.memory.shared_memory import SharedMemory


class FakeClaims:
    def __init__(self, deny=()):
        self.deny, self.claimed = set(deny), []

    def try_claim(self, number, agent, branch):
        if number in self.deny:
            return False
        self.claimed.append(number)
        return True


class FakePersonal:
    def __init__(self):
        self.lessons = []

    def add_lesson(self, lesson, issue=None):
        self.lessons.append(lesson)


def _issue(number, title, day, labels=(), body=""):
    return SimpleNamespace(
        number=number,
        title=title,
        body=body,
        pull_request=None,
        labels=[SimpleNamespace(name=n) for n in labels],
        created_at=datetime(2026, 11, day, tzinfo=timezone.utc),
    )


def _ctx(mem, issues=(), role=Role.CODER, claims=None, evolution_n=10, cycle=1):
    repo = SimpleNamespace(get_issues=lambda **kw: list(issues))
    return SimpleNamespace(
        agent_id="a04",
        role=role,
        memory=mem,
        gh=SimpleNamespace(repo=repo),
        claims=claims or FakeClaims(),
        cycle=cycle,
        meta={"personal": FakePersonal()},
        settings=SimpleNamespace(research_topic="NFV", evolution_every_n_cycles=evolution_n),
        heartbeat=lambda status="working": None,
    )


@pytest.fixture()
def mem(tmp_path: Path):
    m = SharedMemory(tmp_path / "m.db")
    yield m
    m.close()


# ------------------------------------------------------------ claim
def test_claim_steers_a_saturated_agent_to_new_work_and_logs_why(mem):
    for _ in range(5):  # a04 only ever merges bugfixes
        mem.add_trace("a04", "coder", 1, "merged", category="bugfix")
    issues = [_issue(1, "Fix crash in claims", 1), _issue(2, "Write docs for policies", 2)]
    ctx = _ctx(mem, issues)
    state = nodes.node_claim(ctx, {})
    assert state["issue_number"] == 2 and state["category"] == "docs"
    assert (
        "score=" in mem._conn.execute("SELECT reason FROM decisions ORDER BY id DESC").fetchone()[0]
    )


def test_claim_without_history_is_still_oldest_first(mem):
    issues = [_issue(7, "Fix bug A", 1), _issue(8, "Fix bug B", 2)]
    assert nodes.node_claim(_ctx(mem, issues), {})["issue_number"] == 7


def test_claim_falls_through_to_next_best_when_race_is_lost(mem):
    issues = [_issue(1, "Fix bug A", 1), _issue(2, "Fix bug B", 2)]
    state = nodes.node_claim(_ctx(mem, issues, claims=FakeClaims(deny={1})), {})
    assert state["issue_number"] == 2


def test_only_the_coder_may_take_self_evolution_issues(mem):
    issues = [_issue(1, "self-evolution: refine agent prompts", 1, ["kind/evolution"])]
    assert nodes.node_claim(_ctx(mem, issues, role=Role.WRITER), {})["done"] is True
    assert nodes.node_claim(_ctx(mem, issues, role=Role.CODER), {})["issue_number"] == 1


# ------------------------------------------------------------ plan
def test_plan_prompt_carries_group_traces_and_registry_and_counts_reuse(mem, monkeypatch):
    mem.add_trace(
        "a05",
        "coder",
        1,
        "failed",
        issue=3,
        category="code",
        attempted="regex parser",
        failure="catastrophic backtracking",
    )
    mem.register_tool(
        "snippet_extract",
        "agentics/tools/snip.py",
        "extract snippets by line range",
        "a06",
        pr_number=9,
    )
    mem.activate_tools(9)
    seen = {}

    async def fake_llm(ctx, prompt, mode="concise"):
        seen["prompt"] = prompt
        return json.dumps(
            {
                "files": [],
                "steps": ["s1"],
                "commit_message": "m",
                "reuses": ["snippet_extract"],
                "reusable_tools": [],
            }
        ), []

    monkeypatch.setattr(nodes, "_llm_ask", fake_llm)
    state = asyncio.run(nodes.node_plan(
        _ctx(mem),
        {
            "issue_number": 3,
            "issue_title": "Extract snippet code",
            "issue_body": "",
            "papers": [{"answer": ""}],
            "category": "code",
        },
    ))
    assert "catastrophic backtracking" in seen["prompt"]  # a05's failure reaches a04
    assert "snippet_extract" in seen["prompt"]  # registry is in the prompt
    assert state["reuses"] == ["snippet_extract"]
    assert mem.list_tools("snippet")[0]["uses"] == 1


def test_evolution_plan_prompt_states_the_allowed_paths(mem, monkeypatch):
    seen = {}

    async def fake_llm(ctx, prompt, mode="concise"):
        seen["p"] = prompt
        return "{}", []

    monkeypatch.setattr(nodes, "_llm_ask", fake_llm)
    asyncio.run(nodes.node_plan(
        _ctx(mem),
        {
            "issue_number": 1,
            "issue_title": "self-evolution: x",
            "issue_body": "",
            "papers": [{"answer": ""}],
            "category": "evolution",
        },
    ))
    assert "SELF-EVOLUTION" in seen["p"] and "agentics/agents/roles.py" in seen["p"]


# ------------------------------------------------------------ reflect → traces
@pytest.mark.parametrize(
    "extra,outcome",
    [
        ({"pr_number": 5}, "pending"),
        ({"yield_reason": "foreign files"}, "yield"),
        ({"error": "tier-1 tests failed"}, "failed"),
        ({}, "no_change"),
    ],
)
def test_reflect_records_a_structured_trace(mem, extra, outcome):
    state = {
        "issue_number": 4,
        "issue_title": "Write docs",
        "category": "docs",
        "plan": json.dumps({"steps": ["draft", "polish"]}),
        "changed_files": ["docs/a.md"],
        **extra,
    }
    nodes.node_reflect(_ctx(mem), state)
    t = mem.recent_traces()[-1]
    assert t["outcome"] == outcome and t["category"] == "docs"
    assert "draft; polish" in t["attempted"] and t["files"] == ["docs/a.md"]


def test_reflect_writes_no_trace_for_non_work_cycles(mem):
    nodes.node_reflect(_ctx(mem, role=Role.REVIEWER), {"summary": "reviewed PR"})
    nodes.node_reflect(_ctx(mem), {"summary": "no unclaimed issues"})  # no issue_number
    assert mem.recent_traces() == []


# ------------------------------------------------------------ supervisor → evolution issue
def test_supervisor_opens_evolution_issue_only_with_evidence_and_on_schedule(mem, monkeypatch):
    opened = []
    monkeypatch.setattr(nodes.issue_ops, "scan_open_issues", lambda repo: [])
    monkeypatch.setattr(
        nodes.issue_ops,
        "create_issue_if_absent",
        lambda repo, title, body, labels=None: (
            opened.append((title, labels)) or SimpleNamespace(number=99)
        ),
    )

    def run(cycle):
        ctx = _ctx(mem, role=Role.SUPERVISOR, cycle=cycle, evolution_n=10)
        ctx.claims = SimpleNamespace(sweep_stale=lambda: [])
        return nodes.node_scan(ctx, {})

    run(10)
    assert opened == []  # on schedule but no failure evidence yet
    mem.add_trace("a04", "coder", 1, "failed", category="code", failure="boom")
    run(7)
    assert opened == []  # evidence but off schedule
    state = run(10)
    assert opened == [
        ("self-evolution: refine agent prompts from failure traces", ["kind/evolution"])
    ]
    assert "#99" in state["summary"]


# ------------------------------------------------------------ implement (tier-1, guards)
class FakeWorkspace:
    def __init__(self, tmp_path, changed, cmd_results=None):
        self.path, self._changed = tmp_path, changed
        self.results = cmd_results or {}
        self.cmds, self.pushed, self.synced = [], [], 0
        for f in changed:
            (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / f).write_text("x = 1\n")

    def sync_main(self):
        self.synced += 1

    def checkout_new_branch(self, b):
        pass

    def read_file(self, p):
        return ""

    def write_file(self, p, c):
        pass

    def changed_files(self):
        return list(self._changed)

    def run_command(self, cmd, timeout=120):
        self.cmds.append(cmd)
        key = "pytest" if "pytest" in cmd else "compile"
        return self.results.get(key, (0, "ok"))

    def commit_and_push(self, branch, msg):
        self.pushed.append(branch)
        return True


def _impl_ctx(mem, ws, role=Role.CODER):
    ctx = _ctx(mem, role=role)
    ctx.workspace = ws
    ctx.ownership = __import__("agentics.policies.ownership", fromlist=["x"]).OwnershipPolicy()
    ctx.claims = SimpleNamespace(
        release=lambda *a, **k: ctx.__dict__.setdefault("released", []).append(a)
    )
    return ctx


def _impl_state(category="code"):
    return {
        "branch": "agent/a04/issue-1-x",
        "issue_number": 1,
        "issue_title": "t",
        "category": category,
        "plan": json.dumps({"files": [], "commit_message": "m"}),
    }


@pytest.fixture(autouse=False)
def no_branch(monkeypatch):
    monkeypatch.setattr(nodes.branch_ops, "create_branch", lambda *a, **k: None)


def test_implement_happy_path_runs_both_tier1_checks_and_pushes(mem, tmp_path, no_branch):
    ws = FakeWorkspace(tmp_path, ["agentics/a.py", "docs/x.md"])
    state = asyncio.run(nodes.node_implement(_impl_ctx(mem, ws), _impl_state()))
    assert ws.pushed and state["action"] == "open_pr"
    assert "tier-1 passed" in state["verification"] and "1 file" in state["verification"]
    assert state["changed_files"] == ["agentics/a.py", "docs/x.md"]
    assert any("py_compile" in c for c in ws.cmds) and any("pytest" in c for c in ws.cmds)


def test_implement_blocks_push_when_tier1_tests_fail(mem, tmp_path, no_branch):
    ws = FakeWorkspace(tmp_path, ["agentics/a.py"], {"pytest": (1, "1 failed")})
    state = asyncio.run(nodes.node_implement(_impl_ctx(mem, ws), _impl_state()))
    assert not ws.pushed and state["done"] and "tier-1 tests failed" in state["error"]


def test_implement_blocks_push_on_syntax_error_before_running_pytest(mem, tmp_path, no_branch):
    ws = FakeWorkspace(tmp_path, ["agentics/a.py"], {"compile": (1, "SyntaxError")})
    state = asyncio.run(nodes.node_implement(_impl_ctx(mem, ws), _impl_state()))
    assert "tier-1 compile failed" in state["error"]
    assert not any("pytest" in c for c in ws.cmds)  # fail fast


def test_implement_skips_tier1_for_docs_only_changes(mem, tmp_path, no_branch):
    ws = FakeWorkspace(tmp_path, ["docs/x.md"])
    state = asyncio.run(nodes.node_implement(_impl_ctx(mem, ws), _impl_state()))
    assert ws.pushed and "skipped" in state["verification"] and ws.cmds == []


def test_evolution_pr_touching_forbidden_files_is_yielded(mem, tmp_path, no_branch):
    ws = FakeWorkspace(tmp_path, ["agentics/agents/roles.py", "agentics/stop.py"])
    state = asyncio.run(nodes.node_implement(_impl_ctx(mem, ws), _impl_state("evolution")))
    assert not ws.pushed and "outside allowed paths" in state["yield_reason"]
    assert "agentics/stop.py" in state["yield_reason"]


def test_evolution_pr_limited_to_allowed_files_goes_through(mem, tmp_path, no_branch):
    ws = FakeWorkspace(tmp_path, ["agentics/agents/roles.py", "tests/test_roles.py"])
    state = asyncio.run(nodes.node_implement(_impl_ctx(mem, ws), _impl_state("evolution")))
    assert ws.pushed and not state.get("yield_reason")


def test_ownership_yield_now_actually_triggers(mem, tmp_path, no_branch):
    """Regression: P3 compared an always-empty file list, so it never fired."""
    mem.heartbeat("a06", "writer", 1)  # an active writer owns docs/
    ws = FakeWorkspace(tmp_path, ["agentics/a.py", "docs/a.md", "docs/b.md"])
    state = asyncio.run(nodes.node_implement(_impl_ctx(mem, ws), _impl_state()))
    assert not ws.pushed and "foreign files" in state["yield_reason"]
