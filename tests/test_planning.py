"""Adaptive planning & role prompts — LLM-authored missions, LLM-proposed plans."""

import asyncio
import json
import time

import pytest

from agentics import planning
from agentics.agents.roles import GUARDRAILS, Role, build_system_prompt
from agentics.memory.shared_memory import SharedMemory


class FakeLLM:
    """Records ask() calls; answers from a script."""

    def __init__(self, answers=None, error: Exception | None = None):
        self.answers = list(answers or [])
        self.error = error
        self.calls: list[dict] = []

    async def ask(self, query, *, mode="concise", system=None, purpose="general"):
        self.calls.append({"query": query, "system": system, "purpose": purpose})
        if self.error is not None:
            raise self.error
        answer = self.answers.pop(0) if self.answers else ""
        return type("R", (), {"answer": answer, "citations": []})()


GOOD_MISSION = (
    "You own code tasks in this lab. Read the traces first: another agent "
    "already failed at regex backtracking, so you benchmark before parsing. "
    "Declare one verify_cmd per claim and let the runner judge it."
)


@pytest.fixture()
def mem(tmp_path):
    m = SharedMemory(tmp_path / "m.db")
    yield m
    m.close()


def _ctx(mem, llm=None, role=Role.CODER, **overrides):
    from types import SimpleNamespace

    base = dict(
        agent_id="a04", role=role, memory=mem, llm=llm,
        settings=SimpleNamespace(
            research_topic="NFV", adaptive_prompts=True,
            role_prompt_ttl_hours=24.0, role_prompt_max_chars=1600,
            role_prompt_rework_failures=2,
        ),
        meta={"personal": None},
        cycle=1,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# ------------------------------------------------------------ shared memory API
def test_role_prompt_store_supersedes_and_get(mem):
    first = mem.store_role_prompt("coder", "code", "old mission", reason="cold_start")
    assert mem.get_role_prompt("coder", "code")["mission"] == "old mission"
    mem.store_role_prompt("coder", "code", "new mission", reason="rework")
    active = mem.get_role_prompt("coder", "code")
    assert active["mission"] == "new mission" and active["reason"] == "rework"
    # the old row is kept as superseded (history, not deletion)
    with mem._lock:
        n = mem._conn.execute(
            "SELECT COUNT(*) FROM role_prompts WHERE id=?", (first,)).fetchone()[0]
    assert n == 1
    # different category does not collide
    mem.store_role_prompt("coder", "evolution", "evo mission")
    assert mem.get_role_prompt("coder", "evolution")["mission"] == "evo mission"
    assert mem.get_role_prompt("coder", "code")["mission"] == "new mission"


def test_bump_role_prompt_success_fails_and_rework(mem):
    mem.store_role_prompt("coder", "code", GOOD_MISSION)
    assert mem.bump_role_prompt("coder", "code", success=True) == "active"
    row = mem.get_role_prompt("coder", "code")
    assert (row["uses"], row["successes"], row["failures"]) == (1, 1, 0)
    # a failure increments; threshold=2 supersedes → next resolve re-synthesizes
    assert mem.bump_role_prompt("coder", "code", success=False) == "active"
    assert mem.bump_role_prompt("coder", "code", success=False) == "superseded"
    assert mem.get_role_prompt("coder", "code") is None
    # success decays failures (one bad cycle must not thrash a good prompt)
    mem.store_role_prompt("reviewer", "review", "m")
    mem.bump_role_prompt("reviewer", "review", success=False)
    mem.bump_role_prompt("reviewer", "review", success=True)
    row = mem.get_role_prompt("reviewer", "review")
    assert row["failures"] == 0 and row["successes"] == 1


def test_role_experience_digest(mem):
    mem.add_trace("a04", "coder", 1, "failed", issue=3, category="code",
                  attempted="regex parser", failure="backtracking")
    mem.add_trace("a09", "coder", 2, "merged", issue=4, category="code",
                  attempted="chunked parser")
    digest = mem.role_experience("coder", "code")
    assert "[failed] regex parser" in digest
    assert "backtracking" in digest
    assert "[merged] chunked parser" in digest
    assert mem.role_experience("writer", "docs") == ""


# ------------------------------------------------------------ sanitize & compose
def test_sanitize_mission_strips_control_chars_and_caps():
    dirty = "line1\x00\x08line2\n\n\n\nline3" + " word" * 500
    out = planning.sanitize_mission(dirty, max_chars=100)
    assert "\x00" not in out and "\x08" not in out
    assert out.count("\n\n\n") == 0
    assert len(out) <= 101  # cap + ellipsis


def test_compose_prompt_appends_guardrails_after_synthesized_mission():
    prompt = planning.compose_system_prompt("a04", "coder", "NFV", GOOD_MISSION)
    assert GOOD_MISSION in prompt and "a04" in prompt and "coder" in prompt
    assert GUARDRAILS in prompt
    assert prompt.endswith(GUARDRAILS)  # code owns the tail, not the model


def test_build_system_prompt_static_unchanged_shape():
    sp = build_system_prompt("a04", Role.CODER, "NFV")
    assert "Turn research findings into working code" in sp
    assert GUARDRAILS in sp
    assert planning.static_mission("nonsense-role") == planning.static_mission(Role.RESEARCHER)


# ------------------------------------------------------------ resolve/synthesize
def test_resolve_without_llm_falls_back_to_static(mem):
    sp = asyncio.run(planning.resolve_system_prompt(_ctx(mem)))
    assert "Turn research findings into working code" in sp and GUARDRAILS in sp


def test_resolve_synthesizes_caches_and_shares_across_agents(mem):
    llm = FakeLLM(answers=[GOOD_MISSION])
    ctx = _ctx(mem, llm=llm)
    state = {"category": "code"}
    sp1 = asyncio.run(planning.resolve_system_prompt(ctx, state))
    assert GOOD_MISSION in sp1 and GUARDRAILS in sp1
    assert llm.calls[0]["purpose"] == "role_synthesis"
    meta_prompt = llm.calls[0]["query"]
    assert "Role: coder" in meta_prompt
    assert "Task category: code" in meta_prompt
    assert "cold_start" == mem.get_role_prompt("coder", "code")["reason"]
    # stored active for the swarm
    assert mem.get_role_prompt("coder", "code")["mission"] == GOOD_MISSION
    # second agent, same role/category: cache hit — no second synthesis call
    llm2 = FakeLLM(answers=["should not be used"])
    sp2 = asyncio.run(planning.resolve_system_prompt(_ctx(mem, llm=llm2), state))
    assert GOOD_MISSION in sp2 and llm2.calls == []
    # per-cycle cache: same ctx asks again without any LLM call
    calls_before = len(llm.calls)
    assert asyncio.run(planning.resolve_system_prompt(ctx, state)) == sp1
    assert len(llm.calls) == calls_before


def test_resolve_synthesis_failure_falls_back_to_static(mem):
    llm = FakeLLM(error=RuntimeError("provider down"))
    sp = asyncio.run(planning.resolve_system_prompt(_ctx(mem, llm=llm)))
    assert "Turn research findings into working code" in sp and GUARDRAILS in sp
    assert mem.get_role_prompt("coder", "code") is None


def test_resolve_rejects_refusal_junk(mem):
    llm = FakeLLM(answers=["I cannot comply"])  # < MIN_MISSION_CHARS
    sp = asyncio.run(planning.resolve_system_prompt(_ctx(mem, llm=llm)))
    assert "Turn research findings into working code" in sp
    assert mem.get_role_prompt("coder", "code") is None


def test_rework_after_failures_synthesizes_again_with_record(mem):
    llm1 = FakeLLM(answers=[GOOD_MISSION])
    ctx1 = _ctx(mem, llm=llm1)
    asyncio.run(planning.resolve_system_prompt(ctx1, {"category": "code"}))
    mem.bump_role_prompt("coder", "code", success=False, rework_after=2)
    status = mem.bump_role_prompt("coder", "code", success=False, rework_after=2)
    assert status == "superseded"
    llm2 = FakeLLM(answers=["Better mission that keeps what worked and fixes the rest here."])
    ctx2 = _ctx(mem, llm=llm2)
    sp2 = asyncio.run(planning.resolve_system_prompt(ctx2, {"category": "code"}))
    assert "Better mission" in sp2
    # the rework synthesis saw the measured record of the failed mission
    meta2 = llm2.calls[0]["query"]
    assert "has run 2 cycle(s): 0 ended well, 2 failed" in meta2
    assert mem.get_role_prompt("coder", "code")["reason"] == "rework"


def test_ttl_expiry_triggers_resynthesis(mem):
    mem.store_role_prompt("coder", "code", GOOD_MISSION)
    with mem._lock:  # age the row past the TTL
        mem._conn.execute("UPDATE role_prompts SET ts=? WHERE role='coder'",
                          (time.time() - 25 * 3600,))
    llm = FakeLLM(answers=["Fresh mission written for the current situation of the lab."])
    sp = asyncio.run(planning.resolve_system_prompt(_ctx(mem, llm=llm), {"category": "code"}))
    assert "Fresh mission" in sp and len(llm.calls) == 1
    assert mem.get_role_prompt("coder", "code")["reason"] == "ttl_refresh"


def test_bump_prompt_feedback_skips_yields_and_disabled(mem):
    ctx = _ctx(mem)
    mem.store_role_prompt("coder", "code", GOOD_MISSION)
    planning.bump_prompt_feedback(ctx, {"category": "code", "yield_reason": "lock held"})
    assert mem.get_role_prompt("coder", "code")["uses"] == 0  # neutral
    planning.bump_prompt_feedback(ctx, {"category": "code", "error": "boom"})
    assert mem.get_role_prompt("coder", "code")["failures"] == 1
    ctx.settings.adaptive_prompts = False
    planning.bump_prompt_feedback(ctx, {"category": "code"})
    assert mem.get_role_prompt("coder", "code")["successes"] == 0  # disabled


# ------------------------------------------------------------ normalize_plan
def test_normalize_plan_extracts_contract_and_preserves_extras():
    parsed = {
        "approach": "Chunk the input, then parse incrementally.",
        "files": [{"path": "experiments/p.py", "change": "new"}],
        "steps": ["write", "run", "measure"],
        "commit_message": "feat: chunked parser",
        "verify_cmds": ["python experiments/p.py"],
        "hypotheses": [{"hypothesis_id": "speed", "claim": "faster"}],
        "reuses": ["snippet_extract"],
        "risks": ["memory blowup on 1GB input"],
        "alternatives": ["regex (rejected: backtracking)"],
    }
    core = planning.normalize_plan(parsed)
    assert core.approach.startswith("Chunk the input")
    assert core.expected_files == ["experiments/p.py"]
    assert core.verify_cmds == ["python experiments/p.py"]
    assert len(core.hypotheses) == 1
    assert core.reuses == ["snippet_extract"]
    assert core.commit_message == "feat: chunked parser"
    # model-authored fields beyond the contract survive, grouped as extras
    assert core.extras == {"risks": ["memory blowup on 1GB input"],
                           "alternatives": ["regex (rejected: backtracking)"]}
    assert core.parsed is parsed  # full proposal kept verbatim for state["plan"]


def test_normalize_plan_defensive_caps_and_junk():
    core = planning.normalize_plan({
        "verify_cmds": [f"cmd{i}" for i in range(9)],
        "hypotheses": [{"claim": i} for i in range(5)] + ["not-a-dict"],
        "files": [{"path": f"f{i}.py"} for i in range(20)],
        "reuses": [1, "tool", "tool2"],
        "commit_message": None,
    })
    assert len(core.verify_cmds) == 5
    assert len(core.hypotheses) == 2
    assert len(core.expected_files) == 15
    assert core.reuses == ["tool", "tool2"]  # non-str dropped
    assert core.commit_message == "update"
    # empty / non-dict input degrades to an inert plan
    empty = planning.normalize_plan("garbage")
    assert empty.parsed == {} and empty.expected_files == []


def test_normalize_plan_backward_compatible_without_verify():
    core = planning.normalize_plan({"files": [], "steps": ["s1"],
                                    "commit_message": "m"})
    assert core.verify_cmds == [] and core.hypotheses == []
    assert core.expected_files == [] and core.commit_message == "m"


# ------------------------------------------------------------ graph wiring
def _plan_ctx(mem, llm=None):
    from types import SimpleNamespace

    class _Personal:
        def add_lesson(self, lesson, issue=None):
            pass

    return SimpleNamespace(
        agent_id="a04", role=Role.CODER, memory=mem, llm=llm,
        claims=None, cycle=1,
        meta={"personal": _Personal()},
        settings=SimpleNamespace(
            research_topic="NFV", evolution_every_n_cycles=10,
            adaptive_prompts=True, role_prompt_ttl_hours=24.0,
            role_prompt_max_chars=1600, role_prompt_rework_failures=2,
        ),
        heartbeat=lambda status="working": None,
        report=lambda **kw: _plan_ctx.reports.append(kw),
        local=None,
    )


_plan_ctx.reports = []


def test_node_plan_prompt_proposes_instead_of_template(mem, monkeypatch):
    from agentics.graph import nodes

    seen = {}

    async def fake_llm(ctx, prompt, mode="concise", purpose="general", state=None):
        seen["p"] = prompt
        return json.dumps({
            "approach": "Chunk first, then parse; regex rejected (backtracking).",
            "files": [{"path": "experiments/p.py", "change": "new"}],
            "steps": ["chunk", "parse"],
            "commit_message": "feat: chunked parser",
            "verify_cmds": ["python experiments/p.py"],
            "risks": ["1GB inputs"],
        }), []

    monkeypatch.setattr(nodes, "_llm_ask", fake_llm)
    state = asyncio.run(nodes.node_plan(_plan_ctx(mem), {
        "issue_number": 3, "issue_title": "Parser", "issue_body": "",
        "papers": [{"answer": ""}], "category": "code",
    }))
    # the prompt ASKS for a self-proposed approach, not a fill-in template
    assert "PROPOSE the plan yourself" in seen["p"]
    assert '"approach" (required, first)' in seen["p"]
    # model-authored proposal lands in the state for the owner to read
    assert state["plan_approach"].startswith("Chunk first")
    assert json.loads(state["plan_extras"]) == {"risks": ["1GB inputs"]}
    # machine contract intact for the downstream machinery
    assert state["expected_files"] == ["experiments/p.py"]
    assert state["verify_cmds"] == ["python experiments/p.py"]
    assert state["commit_message"] == "feat: chunked parser"
    intent = _plan_ctx.reports[-1]["intent"]
    assert intent["approach"].startswith("Chunk first")
    assert intent["extras"] == ["risks"]


def test_node_plan_backward_compatible_minimal_json(mem, monkeypatch):
    from agentics.graph import nodes

    async def fake_llm(ctx, prompt, mode="concise", purpose="general", state=None):
        return json.dumps({"files": [], "steps": ["s1"],
                           "commit_message": "m"}), []

    monkeypatch.setattr(nodes, "_llm_ask", fake_llm)
    state = asyncio.run(nodes.node_plan(_plan_ctx(mem), {
        "issue_number": 1, "issue_title": "T", "issue_body": "",
        "papers": [{"answer": ""}], "category": "code",
    }))
    assert state["verify_cmds"] == [] and state["plan_approach"] == ""
    assert "plan_extras" not in state


def test_llm_ask_resolves_adaptive_system_prompt(mem, monkeypatch):
    from agentics.graph import nodes

    llm = FakeLLM(answers=[GOOD_MISSION, "plan json"])
    ctx = _plan_ctx(mem, llm=llm)
    answer, _ = asyncio.run(nodes._llm_ask(
        ctx, "do the plan", purpose="plan", state={"category": "code"}))
    assert answer == "plan json"
    # call 1: mission synthesis (static system), call 2: the plan itself with
    # the SYNTHESIZED mission + code-owned guardrails as the system prompt
    assert llm.calls[0]["purpose"] == "role_synthesis"
    assert GOOD_MISSION in llm.calls[1]["system"]
    assert GUARDRAILS in llm.calls[1]["system"]


def test_reflect_feeds_cycle_outcome_into_the_mission(mem):
    from agentics.graph import nodes

    mem.store_role_prompt("coder", "code", GOOD_MISSION)
    ctx = _plan_ctx(mem)
    ctx.heartbeat = lambda status="working": None
    ctx.claims = None
    nodes.node_reflect(ctx, {"category": "code", "error": "tier-1 tests failed"})
    assert mem.get_role_prompt("coder", "code")["failures"] == 1
    nodes.node_reflect(ctx, {"category": "code", "summary": "ok"})
    row = mem.get_role_prompt("coder", "code")
    assert row["successes"] == 1 and row["failures"] == 0  # decayed
