"""Wiring of the graph nodes in local-only mode (fake LLM, real SharedMemory)."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentics.agents.roles import Role
from agentics.graph import nodes
from agentics.memory.shared_memory import SharedMemory
from agentics.policies.guard import ProtectedPathsGuard


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


def _task(mem, title, body="", category="research", labels=""):
    return mem.create_task(title, body=body, category=category, labels=labels)


def _local_claims(mem, deny=()):
    """Real LocalClaimPolicy (so task-status transitions are exercised too)."""
    from agentics.local_ops import LocalClaimPolicy

    policy = LocalClaimPolicy(mem, ttl_minutes=60)
    if deny:
        original = policy.try_claim

        def try_claim(task_id, agent_id, branch=""):
            if task_id in deny:
                return False
            return original(task_id, agent_id, branch)

        policy.try_claim = try_claim
    return policy


def _ctx(mem, role=Role.CODER, claims=None, evolution_n=10, cycle=1,
         agent_id="a04", guard=None, with_local=True):
    return SimpleNamespace(
        agent_id=agent_id,
        role=role,
        memory=mem,
        claims=claims if claims is not None else _local_claims(mem),
        cycle=cycle,
        meta={"personal": FakePersonal()},
        settings=SimpleNamespace(research_topic="NFV", evolution_every_n_cycles=evolution_n),
        heartbeat=lambda status="working": None,
        report=lambda **kw: None,
        local=SimpleNamespace(guard=guard) if guard is not None
              else (SimpleNamespace(guard=ProtectedPathsGuard()) if with_local else None),
    )


def _llm_json(monkeypatch, payload_by_purpose=None, payload=None):
    """Route _llm_ask to canned JSON answers.

    Signature matches the resilient ``_llm_ask(ctx, ..., state=None)`` ported
    from main: a fake that raises must record the failure in ``state`` so the
    cycle degrades to reflect exactly like the real helper does."""
    async def fake_llm(ctx, prompt, mode="concise", purpose="general", state=None):
        if payload_by_purpose and purpose in payload_by_purpose:
            return json.dumps(payload_by_purpose[purpose]), []
        return json.dumps(payload if payload is not None else {}), []
    monkeypatch.setattr(nodes, "_llm_ask", fake_llm)


@pytest.fixture()
def mem(tmp_path: Path):
    m = SharedMemory(tmp_path / "m.db")
    yield m
    m.close()


# ------------------------------------------------------------ claim
def test_claim_steers_a_saturated_agent_to_new_work_and_logs_why(mem):
    for _ in range(5):  # a04 only ever merges bugfixes
        mem.add_trace("a04", "coder", 1, "merged", category="bugfix")
    _task(mem, "Fix crash in claims")
    _task(mem, "Write docs for policies")
    state = nodes.node_claim(_ctx(mem), {})
    assert state["issue_number"] == 2 and state["category"] == "docs"
    assert (
        "score=" in mem._conn.execute("SELECT reason FROM decisions ORDER BY id DESC").fetchone()[0]
    )


def test_claim_without_history_is_still_oldest_first(mem):
    _task(mem, "Fix bug A")
    _task(mem, "Fix bug B")
    assert nodes.node_claim(_ctx(mem), {})["issue_number"] == 1


def test_claim_falls_through_to_next_best_when_race_is_lost(mem):
    _task(mem, "Fix bug A")
    _task(mem, "Fix bug B")
    state = nodes.node_claim(
        _ctx(mem, claims=_local_claims(mem, deny={1})), {})
    assert state["issue_number"] == 2


def test_claim_publishes_intent_and_task_status(mem):
    _task(mem, "Survey SOTA for NFV")
    state = nodes.node_claim(_ctx(mem), {})
    assert state["issue_number"] == 1
    task = mem.get_task(1)
    assert task["status"] == "claimed" and task["claimed_by"] == "a04"


def test_only_the_coder_may_take_self_evolution_tasks(mem):
    _task(mem, "self-evolution: refine agent prompts", labels="kind/evolution")
    assert nodes.node_claim(_ctx(mem, role=Role.WRITER), {})["done"] is True
    assert nodes.node_claim(_ctx(mem, role=Role.CODER), {})["issue_number"] == 1


def test_claim_records_owner_exemption_flag(mem):
    tid = _task(mem, "self-evolution: prompts", labels="kind/evolution")
    mem.set_task_exempt(tid, True, agent="owner")
    state = nodes.node_claim(_ctx(mem, role=Role.CODER), {})
    assert state["task_exempt"] is True


# ------------------------------------------------------------ plan
def test_plan_prompt_carries_group_traces_and_registry_and_counts_reuse(mem, monkeypatch):
    mem.add_trace(
        "a05", "coder", 1, "failed", issue=3, category="code",
        attempted="regex parser", failure="catastrophic backtracking",
    )
    mem.register_tool(
        "snippet_extract", "agentics/tools/snip.py",
        "extract snippets by line range", "a06", pr_number=9,
    )
    mem.activate_tools(9)
    seen = {}

    async def fake_llm(ctx, prompt, mode="concise", purpose="general", state=None):
        seen["prompt"] = prompt
        return json.dumps(
            {"files": [], "steps": ["s1"], "commit_message": "m",
             "reuses": ["snippet_extract"], "reusable_tools": []}), []

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

    async def fake_llm(ctx, prompt, mode="concise", purpose="general", state=None):
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
    # anti-blackbox: the cycle outcome reaches the journal too
    assert mem.recent_journal()[-1]["event"] == "cycle_done"


def test_reflect_writes_no_trace_for_non_work_cycles(mem):
    nodes.node_reflect(_ctx(mem, role=Role.REVIEWER), {"summary": "reviewed MR"})
    nodes.node_reflect(_ctx(mem), {"summary": "no unclaimed tasks"})  # no issue_number
    assert mem.recent_traces() == []


# ------------------------------------------------------------ supervisor → evolution task
def test_supervisor_opens_evolution_task_only_with_evidence_and_on_schedule(mem, monkeypatch):
    monkeypatch.setattr(
        nodes, "build_evolution_issue",
        lambda memory: ("self-evolution: refine agent prompts from failure traces", "body"),
    )

    def run(cycle):
        ctx = _ctx(mem, role=Role.SUPERVISOR, cycle=cycle, evolution_n=10)
        ctx.claims = SimpleNamespace(sweep_stale=lambda: [])
        return nodes.node_scan(ctx, {})

    run(10)
    assert mem.list_tasks() == []  # on schedule but no failure evidence yet
    mem.add_trace("a04", "coder", 1, "failed", category="code", failure="boom")
    run(7)
    assert mem.list_tasks() == []  # evidence but off schedule
    state = run(10)
    tasks = mem.list_tasks()
    assert [t["title"] for t in tasks] == [
        "self-evolution: refine agent prompts from failure traces"
    ]
    assert tasks[0]["category"] == "evolution"
    assert tasks[0]["exempt"] is False   # owner must grant the code exemption
    assert f"Opened self-evolution task #{tasks[0]['id']}" in state["summary"]


def test_supervisor_tops_up_empty_backlog_from_local_source(mem):
    from agentics.local_ops import TaskSource

    class _Settings:
        research_topic = "NFV"

    ctx = _ctx(mem, role=Role.SUPERVISOR)
    ctx.local = SimpleNamespace(guard=ProtectedPathsGuard(),
                                tasks=TaskSource(mem, _Settings()))
    ctx.claims = SimpleNamespace(sweep_stale=lambda: [])
    state = nodes.node_scan(ctx, {})
    assert "Opened 1 new survey task(s)" in state["summary"]
    assert mem.scan_open_tasks()      # the backlog is no longer empty


# ------------------------------------------------------------ implement (tier-1, guards)
class FakeWorkspace:
    def __init__(self, tmp_path, changed, cmd_results=None):
        self.path, self._changed = tmp_path, changed
        self.results = cmd_results or {}
        self.cmds, self.pushed, self.synced = [], [], 0
        self.guard = None
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
        if self.guard is not None and self.guard.is_protected(p):
            raise ValueError(f"P10 protected path: '{p}' is lab code")
        (self.path / p).parent.mkdir(parents=True, exist_ok=True)
        (self.path / p).write_text(c, encoding="utf-8")

    def changed_files(self):
        return list(self._changed)

    def run_command(self, cmd, timeout=120):
        self.cmds.append(cmd)
        if len(cmd) >= 3 and cmd[1] == "-m":
            key = {"py_compile": "compile"}.get(cmd[2], cmd[2])   # pytest/py_compile
        else:
            key = "script"                     # python <workspace .py> (P16 experiment)
        return self.results.get(key, (0, "ok"))

    def commit_and_push(self, branch, msg, allow_protected=False):
        self.pushed.append(branch)
        self.allow_protected = allow_protected
        return True


def _impl_ctx(mem, ws, role=Role.CODER):
    ctx = _ctx(mem, role=role)
    ws.guard = ctx.local.guard
    ctx.workspace = ws
    ctx.ownership = __import__("agentics.policies.ownership", fromlist=["x"]).OwnershipPolicy()
    ctx.claims = SimpleNamespace(
        release=lambda *a, **k: ctx.__dict__.setdefault("released", []).append(a)
    )
    return ctx


def _impl_state(category="code", issue_number=1):
    return {
        "branch": f"agent/a04/task-{issue_number}-x",
        "issue_number": issue_number,
        "issue_title": "t",
        "category": category,
        "plan": json.dumps({"files": [], "commit_message": "m"}),
    }


def test_implement_happy_path_runs_both_tier1_checks_and_pushes(mem, tmp_path):
    ws = FakeWorkspace(tmp_path, ["experiments/a.py", "docs/x.md"])
    state = asyncio.run(nodes.node_implement(_impl_ctx(mem, ws), _impl_state()))
    assert ws.pushed and state["action"] == "open_pr"
    assert "tier-1 passed" in state["verification"] and "1 file" in state["verification"]
    assert state["changed_files"] == ["experiments/a.py", "docs/x.md"]
    assert any("py_compile" in c for c in ws.cmds) and any("pytest" in c for c in ws.cmds)


def test_implement_blocks_push_when_tier1_tests_fail(mem, tmp_path):
    ws = FakeWorkspace(tmp_path, ["experiments/a.py"], {"pytest": (1, "1 failed")})
    state = asyncio.run(nodes.node_implement(_impl_ctx(mem, ws), _impl_state()))
    assert not ws.pushed and state["done"] and "tier-1 tests failed" in state["error"]


def test_implement_blocks_push_on_syntax_error_before_running_pytest(mem, tmp_path):
    ws = FakeWorkspace(tmp_path, ["experiments/a.py"], {"compile": (1, "SyntaxError")})
    state = asyncio.run(nodes.node_implement(_impl_ctx(mem, ws), _impl_state()))
    assert "tier-1 compile failed" in state["error"]
    assert not any("pytest" in c for c in ws.cmds)  # fail fast


def test_implement_skips_tier1_for_docs_only_changes(mem, tmp_path):
    ws = FakeWorkspace(tmp_path, ["docs/x.md"])
    state = asyncio.run(nodes.node_implement(_impl_ctx(mem, ws), _impl_state()))
    assert ws.pushed and "skipped" in state["verification"] and ws.cmds == []


# ------------------------------------------------------------ P10 guard
def test_implement_yields_when_diff_touches_agent_code(mem, tmp_path):
    """P10 layer 2: agent code is owner-only — the diff is refused."""
    ws = FakeWorkspace(tmp_path, ["agentics/stop.py"])
    state = asyncio.run(nodes.node_implement(_impl_ctx(mem, ws), _impl_state()))
    assert not ws.pushed and "P10 protected paths in diff" in state["yield_reason"]


def test_implement_yields_on_protected_write_even_mid_plan(mem, tmp_path, monkeypatch):
    """P10 layer 1: the plan cannot even write into a protected path."""
    _llm_json(monkeypatch, payload_by_purpose={
        "implement": "x = 1\n",
        "plan": {"files": [{"path": "agentics/policies/claims.py", "change": "edit"}],
                 "steps": ["hack the claims"], "commit_message": "self-modify"},
    })
    ws = FakeWorkspace(tmp_path, [])
    state = asyncio.run(nodes.node_implement(
        _impl_ctx(mem, ws),
        {**_impl_state(), "plan": json.dumps(
            {"files": [{"path": "agentics/policies/claims.py", "change": "edit"}],
             "commit_message": "self-modify"})},
    ))
    assert not ws.pushed and "P10 protected path" in state["yield_reason"]


def test_exempted_evolution_task_may_touch_allowed_agent_files(mem, tmp_path):
    """Owner exception: task flagged exempt + evolution allowlist → passes.

    The exemption must travel to the push as well: commit_and_push receives
    allow_protected=True so the P10 origin hook (layer 4) lets the branch
    through — an owner-exempted task stays viable end-to-end."""
    tid = _task(mem, "self-evolution: refine prompts", category="evolution")
    mem.set_task_exempt(tid, True, agent="owner")
    ws = FakeWorkspace(tmp_path, ["agentics/agents/roles.py", "tests/test_roles.py"])
    state = asyncio.run(nodes.node_implement(
        _impl_ctx(mem, ws), _impl_state("evolution", issue_number=tid)))
    assert ws.pushed and not state.get("yield_reason")
    assert ws.allow_protected is True


def test_non_exempt_evolution_task_cannot_touch_agent_files(mem, tmp_path):
    tid = _task(mem, "self-evolution: refine prompts", category="evolution")
    ws = FakeWorkspace(tmp_path, ["agentics/agents/roles.py"])
    state = asyncio.run(nodes.node_implement(
        _impl_ctx(mem, ws), _impl_state("evolution", issue_number=tid)))
    assert not ws.pushed and "P10 protected paths in diff" in state["yield_reason"]


def test_evolution_pr_touching_forbidden_files_is_yielded(mem, tmp_path):
    """Exempt task still constrained by the evolution allowlist (stop.py)."""
    tid = _task(mem, "self-evolution: prompts", category="evolution")
    mem.set_task_exempt(tid, True, agent="owner")
    ws = FakeWorkspace(tmp_path, ["agentics/agents/roles.py", "agentics/stop.py"])
    state = asyncio.run(nodes.node_implement(
        _impl_ctx(mem, ws), _impl_state("evolution", issue_number=tid)))
    assert not ws.pushed and "outside allowed paths" in state["yield_reason"]
    assert "agentics/stop.py" in state["yield_reason"]


def test_credentials_paths_stay_protected_even_when_exempt(mem, tmp_path):
    tid = _task(mem, "steal cookies", category="other")
    mem.set_task_exempt(tid, True, agent="owner")
    ws = FakeWorkspace(tmp_path, ["accounts/account-01.json"])
    state = asyncio.run(nodes.node_implement(_impl_ctx(mem, ws), _impl_state("other", tid)))
    assert not ws.pushed and "P10 protected paths in diff" in state["yield_reason"]


def test_ownership_yield_now_actually_triggers(mem, tmp_path):
    """Regression: P3 compared an always-empty file list, so it never fired."""
    mem.heartbeat("a06", "writer", 1)  # an active writer owns docs/
    ws = FakeWorkspace(tmp_path, ["experiments/a.py", "docs/a.md", "docs/b.md"])
    state = asyncio.run(nodes.node_implement(_impl_ctx(mem, ws), _impl_state()))
    assert not ws.pushed and "foreign files" in state["yield_reason"]


# ------------------------------------------------------------ open_mr (local)
def test_open_mr_creates_review_record_and_completes_claim(mem):
    completed = []
    task_id = _task(mem, "Add experiment")
    ctx = _ctx(mem)
    ctx.claims = SimpleNamespace(
        complete=lambda *a, **k: completed.append(a),
        release=lambda *a, **k: None,
    )
    state = {
        "issue_number": task_id,
        "issue_title": "Add experiment",
        "branch": f"agent/a04/task-{task_id}-add",
        "changed_files": ["experiments/a.py"],
        "verification": "tier-1 passed",
    }
    out = nodes.node_open_pr(ctx, state)
    mr_id = out["pr_number"]
    assert out["pr_url"] == f"local://mr/{mr_id}"
    assert completed == [(task_id, "a04", f"agent/a04/task-{task_id}-add")]
    mr = mem.get_review(mr_id)
    assert mr["author"] == "a04" and mr["files"] == ["experiments/a.py"]
    assert mem.get_task(task_id)["status"] == "in_review"
    assert any(j["event"] == "open_mr" for j in mem.recent_journal())


# ------------------------------------------------------------ review (local, P5+P10)
def _mr(mem, author="a04", task_id=None, files=("experiments/a.py",), title="Add experiment"):
    if task_id is None:
        task_id = _task(mem, title)
    mr_id = mem.open_review(
        task_id=task_id, title=f"[{author}] {title} (#{task_id})",
        body="body", author=author, branch=f"agent/{author}/task-{task_id}-x",
        files=list(files), verification="tier-1 passed",
    )
    return mr_id


def test_reviewer_approves_foreign_mr_and_enqueues(mem, monkeypatch):
    mr_id = _mr(mem, author="a04")
    enqueued = []
    _llm_json(monkeypatch, payload={"verdict": "approve", "comment": "lgtm"})
    ctx = _ctx(mem, role=Role.REVIEWER, agent_id="a07")
    ctx.merge_queue = SimpleNamespace(
        enqueue=lambda mr, author: enqueued.append((mr, author)))
    state = asyncio.run(nodes.node_review(ctx, {}))
    assert state["review_verdict"] == "approve"
    assert enqueued == [(mr_id, "a04")]
    review = mem.get_review(mr_id)
    assert review["status"] == "approved" and review["approved_by"] == "a07"
    assert any(j["event"] == "review" for j in mem.recent_journal())


def test_reviewer_never_reviews_own_mr(mem):
    _mr(mem, author="a07")  # same agent as the reviewer
    state = asyncio.run(nodes.node_review(_ctx(mem, role=Role.REVIEWER, agent_id="a07"), {}))
    assert state["summary"] == "no MR awaiting review"


def test_reviewer_cannot_approve_mr_touching_protected_paths(mem, monkeypatch):
    """P10 layer 3 (review side): an 'approve' is downgraded on protected diffs."""
    mr_id = _mr(mem, author="a04", files=("agentics/policies/claims.py",))
    _llm_json(monkeypatch, payload={"verdict": "approve", "comment": "ok"})
    state = asyncio.run(nodes.node_review(_ctx(mem, role=Role.REVIEWER, agent_id="a07"), {}))
    assert state["review_verdict"] == "request_changes"
    assert "P10 protected paths" in state["review_comment"]
    assert mem.get_review(mr_id)["status"] == "changes_requested"
    assert mem.mq_pending() == []


def test_requested_changes_recorded_for_author(mem, monkeypatch):
    mr_id = _mr(mem, author="a04")
    _llm_json(monkeypatch, payload={"verdict": "request_changes", "comment": "tests missing"})
    state = asyncio.run(nodes.node_review(_ctx(mem, role=Role.REVIEWER, agent_id="a07"), {}))
    assert state["review_verdict"] == "request_changes"
    review = mem.get_review(mr_id)
    assert review["status"] == "changes_requested"
    assert "tests missing" in review["verdict_comment"]


# ------------------------------------------------------------ integrate (local)
def _integrator(mem):
    from agentics.local_ops import LocalMergeQueue

    ctx = _ctx(mem, role=Role.INTEGRATOR, agent_id="a09")
    ctx.merge_queue = LocalMergeQueue(mem, auto_merge=True,
                                      guard=ProtectedPathsGuard())
    ctx.workspace = SimpleNamespace(
        verify_merge=lambda branch: (True, ""),
        merge_branch_to_main=lambda branch, msg: True,
    )
    return ctx


def test_integrator_drains_queue_with_local_merge(mem):
    mr_id = _mr(mem, author="a04")
    mem.mq_enqueue(mr_id, author="a04")
    merged_branches = []

    ctx = _integrator(mem)
    ctx.workspace = SimpleNamespace(
        verify_merge=lambda branch: (True, ""),
        merge_branch_to_main=lambda branch, msg: merged_branches.append(branch) or True,
    )
    state = nodes.node_integrate(ctx, {})
    assert state["summary"] == "merged 1 MR(s)"
    assert merged_branches == [f"agent/a04/task-1-x"]
    assert mem.get_review(mr_id)["status"] == "merged"
    assert mem.get_task(1)["status"] == "merged"
    assert any(j["event"] == "merge" for j in mem.recent_journal())


def test_integrator_refuses_merge_of_protected_diff(mem):
    """P10 layer 3 (merge side): protected paths never reach main."""
    mr_id = _mr(mem, author="a04", files=("agentics/policies/claims.py",))
    mem.mq_enqueue(mr_id, author="a04")
    ctx = _integrator(mem)
    nodes.node_integrate(ctx, {})
    assert mem.get_review(mr_id)["status"] == "failed"
    assert mem.get_task(1)["status"] == "failed"


def test_implement_survives_a_hallucinated_edit_path(mem, tmp_path, monkeypatch):
    """read_file returns None when the plan says 'edit' but the file does not
    exist in the workspace — the node must treat it as a new file instead of
    crashing with TypeError on current[:6000] (found while integrating)."""
    _llm_json(monkeypatch, payload_by_purpose={"implement": "x = 1\n"})

    class NoFileWorkspace(FakeWorkspace):
        def read_file(self, p):
            return None               # the path was hallucinated by the plan

    ws = NoFileWorkspace(tmp_path, ["experiments/hallucinated.py"])
    state = asyncio.run(nodes.node_implement(
        _impl_ctx(mem, ws),
        {**_impl_state(), "plan": json.dumps(
            {"files": [{"path": "experiments/hallucinated.py", "change": "edit"}],
             "commit_message": "new experiment"})},
    ))
    assert state.get("error") is None, state
    assert ws.pushed and state["action"] == "open_pr"


# ------------------------------------------------------------ P16 empirical evidence
import sys as _sys

_PY = _sys.executable


def _with_cmds(state, cmds):
    state["verify_cmds"] = cmds
    return state


def test_p16_confirmed_evidence_is_recorded_and_pushed(mem, tmp_path):
    """H1+H2: the agent RUNS its declared check; exit 0 -> shared evidence."""
    ws = FakeWorkspace(tmp_path, ["experiments/a.py"])
    state = asyncio.run(nodes.node_implement(
        _impl_ctx(mem, ws), _with_cmds(_impl_state(), [[_PY, "exp.py"]])))
    assert ws.pushed and state["action"] == "open_pr"
    rows = mem.evidence_for_task(1)
    assert len(rows) == 1 and rows[0]["verdict"] == "confirmed"
    assert rows[0]["agent"] == mem.evidence_for_task(1)[0]["agent"]
    assert "P16" in state["verification"]
    assert any(c[:2] == [_PY, "exp.py"] for c in ws.cmds)      # really ran
    assert any(m["kind"] == "evidence" for m in mem.recent_messages())   # H3 broadcast


def test_p16_refuted_blocks_the_push(mem, tmp_path):
    """A nonzero exit is REAL evidence against the hypothesis: no push."""
    ws = FakeWorkspace(tmp_path, ["experiments/a.py"], {"script": (1, "boom")})
    state = asyncio.run(nodes.node_implement(
        _impl_ctx(mem, ws), _with_cmds(_impl_state(), [[_PY, "exp.py"]])))
    assert not ws.pushed and "refuted" in state.get("yield_reason", "")
    assert mem.evidence_for_task(1)[0]["verdict"] == "refuted"


def test_p16_inconclusive_warns_but_does_not_block(mem, tmp_path):
    """Blocked/timeout (exit 124/126) -> inconclusive: reviewer decides,
    pipeline never livelocks on a command the allowlist rejects."""
    ws = FakeWorkspace(tmp_path, ["experiments/a.py"], {"script": (126, "command rejected")})
    state = asyncio.run(nodes.node_implement(
        _impl_ctx(mem, ws), _with_cmds(_impl_state(), [[_PY, "exp.py"]])))
    assert ws.pushed
    assert mem.evidence_for_task(1)[0]["verdict"] == "inconclusive"
    assert any(j["event"] == "p16_inconclusive" for j in mem.recent_journal(limit=20))


def test_p16_no_declared_check_keeps_old_behavior(mem, tmp_path):
    """Backward compatible: plans without verify_cmds behave exactly as before."""
    ws = FakeWorkspace(tmp_path, ["experiments/a.py"])
    state = asyncio.run(nodes.node_implement(_impl_ctx(mem, ws), _impl_state()))
    assert ws.pushed and mem.latest_evidence() == []


def test_reviewer_sees_the_recorded_evidence(mem, monkeypatch):
    """H3: the reviewer weighs REAL runs, and its comment cites them."""
    mr_id = _mr(mem, author="a04")
    mem.record_evidence("a04", mem.get_review(mr_id)["task_id"],
                        "python -m pytest exp.py", 0, "confirmed",
                        hypothesis="t holds")
    mem.record_evidence("a04", mem.get_review(mr_id)["task_id"],
                        "python flaky.py", 124, "inconclusive")
    _llm_json(monkeypatch, payload={"verdict": "approve", "comment": "lgtm"})
    ctx = _ctx(mem, role=Role.REVIEWER, agent_id="a07")
    ctx.merge_queue = SimpleNamespace(enqueue=lambda mr, author: None)
    state = asyncio.run(nodes.node_review(ctx, {}))
    assert state["review_verdict"] == "approve"
    review = mem.get_review(mr_id)
    assert "P16 evidence: 2 run(s)" in review["verdict_comment"]
    assert "confirmed" in review["verdict_comment"]
