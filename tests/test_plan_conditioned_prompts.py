"""Prompts conditioned on the model's OWN plan (continuing the de-hardcoding).

The per-file implement ask used to be a fixed minimal template — the
implementer never saw why the plan exists or what will judge it — and the
research query ignored the issue body. Both now carry the planner's own
words (approach, P16 checks, hypotheses) / the issue's problem statement:
bounded, backward compatible, zero extra LLM calls."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from agentics.agents.roles import Role
from agentics.graph import nodes
from agentics.llm.perplexity_client import LLMResponse
from agentics.memory.shared_memory import SharedMemory
from agentics.policies.guard import ProtectedPathsGuard
from agentics.policies.ownership import OwnershipPolicy

import pytest


@pytest.fixture()
def mem(tmp_path: Path):
    m = SharedMemory(tmp_path / "m.db")
    yield m
    m.close()


# ------------------------------------------------------------ _implement_brief
def test_implement_brief_carries_planner_own_words():
    state = {"plan_approach": "Cache the parsed config once per cycle."}
    plan = {
        "verify_cmds": ["python experiments/check_cache.py"],
        "hypotheses": [{"claim": "cache cuts parse time",
                        "prediction": "under 0.5s"}],
    }
    brief = nodes._implement_brief(state, plan)
    assert "Cache the parsed config once per cycle." in brief
    assert "python experiments/check_cache.py" in brief
    assert "cache cuts parse time" in brief
    assert "under 0.5s" in brief


def test_implement_brief_falls_back_to_plan_approach_field():
    """plan_approach missing in state -> the plan's own `approach` is used."""
    brief = nodes._implement_brief({}, {"approach": "Stream rows in batches."})
    assert "Stream rows in batches." in brief


def test_implement_brief_empty_for_minimal_plan():
    assert nodes._implement_brief({}, {}) == ""
    assert nodes._implement_brief({"plan_approach": ""}, {"files": []}) == ""


def test_implement_brief_caps_a_runaway_plan():
    plan = {
        "verify_cmds": [f"python check{i}.py" for i in range(9)],
        "hypotheses": [{"claim": f"c{i}", "prediction": ""} for i in range(5)],
    }
    brief = nodes._implement_brief({"plan_approach": "x" * 5000}, plan)
    assert brief.count("- python check") == 5      # cmds capped at 5
    assert brief.count("- claim: c") == 2          # hypotheses capped at 2
    assert "x" * 701 not in brief                  # approach capped at 700
    assert len(brief) < 2200


# ----------------------------------------------- end-to-end: node_implement
class FakeWorkspace:
    def __init__(self, tmp_path, changed):
        self.path = tmp_path
        self._changed = changed
        self.pushed = []

    def sync_main(self):
        pass

    def checkout_new_branch(self, b):
        pass

    def read_file(self, p):
        return ""

    def write_file(self, p, c, allow=False):
        if not allow and ProtectedPathsGuard().is_protected(p):
            raise ValueError(f"P10 protected path: '{p}' is lab code")
        (self.path / p).parent.mkdir(parents=True, exist_ok=True)
        (self.path / p).write_text(c, encoding="utf-8")

    def changed_files(self):
        return list(self._changed)

    def run_command(self, cmd, timeout=120):
        key = ({"py_compile": "compile"}.get(cmd[2], cmd[2])
               if len(cmd) >= 3 and cmd[1] == "-m" else "script")
        return (0, "ok")

    def commit_and_push(self, branch, msg, allow_protected=False):
        self.pushed.append(branch)
        return True


def _impl_ctx(mem, ws):
    return SimpleNamespace(
        agent_id="a04", role=Role.CODER, memory=mem, cycle=1,
        meta={"personal": SimpleNamespace(add_lesson=lambda *a, **k: None)},
        settings=SimpleNamespace(research_topic="NFV"),
        heartbeat=lambda status="working": None,
        report=lambda **kw: None,
        local=SimpleNamespace(guard=ProtectedPathsGuard()),
        workspace=ws,
        ownership=OwnershipPolicy(),
        claims=SimpleNamespace(release=lambda *a, **k: None),
    )


def _recording_llm(monkeypatch, prompts, content="x = 1\n"):
    async def fake(ctx, prompt, mode="concise", purpose="general", state=None):
        prompts.append((purpose, prompt))
        return content, []
    monkeypatch.setattr(nodes, "_llm_ask", fake)


def _impl_state(plan_obj, **extra):
    state = {
        "branch": "agent/a04/task-1-x",
        "issue_number": 1, "issue_title": "t", "category": "code",
        "plan": json.dumps(plan_obj),
    }
    state.update(extra)
    return state


FULL_PLAN = {
    "files": [{"path": "experiments/cache.py", "change": "new"}],
    "steps": ["write the cache"],
    "commit_message": "m",
    "verify_cmds": ["python experiments/check_cache.py"],
    "hypotheses": [{"claim": "cache cuts parse time"}],
    "approach": "Cache the parsed config once per cycle.",
}


def test_implement_prompt_carries_planner_brief(tmp_path, mem, monkeypatch):
    ws = FakeWorkspace(tmp_path, ["experiments/cache.py"])
    prompts: list[tuple[str, str]] = []
    _recording_llm(monkeypatch, prompts)
    out = asyncio.run(nodes.node_implement(
        _impl_ctx(mem, ws), _impl_state(FULL_PLAN,
                                        plan_approach=FULL_PLAN["approach"])))
    assert ws.pushed, "happy path still pushes"
    body = "\n".join(p for purpose, p in prompts if purpose == "implement")
    assert "Cache the parsed config once per cycle." in body
    assert "python experiments/check_cache.py" in body
    assert "cache cuts parse time" in body
    assert out["action"] == "open_pr"


def test_implement_prompt_unchanged_for_minimal_plan(tmp_path, mem, monkeypatch):
    """A legacy plan (no approach / verify_cmds / hypotheses) keeps the old
    prompt: no brief sections appear."""
    ws = FakeWorkspace(tmp_path, ["experiments/a.py"])
    prompts: list[tuple[str, str]] = []
    _recording_llm(monkeypatch, prompts)
    asyncio.run(nodes.node_implement(
        _impl_ctx(mem, ws),
        _impl_state({"files": [{"path": "experiments/a.py", "change": "new"}],
                     "commit_message": "m"})))
    body = "\n".join(p for purpose, p in prompts if purpose == "implement")
    assert "declared approach" not in body
    assert "WILL run against this code" not in body
    assert "Hypotheses the lab will measure" not in body


# ------------------------------------------------- end-to-end: node_research
class RecordingHost:
    """BrowserSessionHost stand-in that records every research query."""

    def __init__(self, answer="llm survey"):
        self.account = SimpleNamespace(name="account-02")
        self.queries: list[str] = []
        self._resp = LLMResponse(answer=answer, citations=[])

    async def ask(self, query, *, mode="concise", system=None, purpose=None):
        self.queries.append(query)
        return self._resp

    def mark_failed(self, error, cooldown_s=900):
        pass


def _research_ctx(mem, host):
    return SimpleNamespace(
        agent_id="a04", role=Role.RESEARCHER, memory=mem, vector=FakeVector(),
        llm=host,
        meta={"personal": SimpleNamespace(add_lesson=lambda *a, **k: None)},
        settings=SimpleNamespace(research_topic="NFV automation"),
        heartbeat=lambda status="working": None,
        report=lambda **k: None,
    )


class FakeVector:
    available = False

    def __init__(self):
        self.added = []

    def add(self, content, meta):
        self.added.append((content, meta))


def test_research_query_carries_issue_body(tmp_path, mem, monkeypatch):
    host = RecordingHost()
    state = {
        "issue_title": "Speed up config parsing",
        "issue_body": "Parsing takes 8s per cycle because the config is "
                      "re-read on every node call. Target: under 0.5s.",
        "papers": [], "citations": [],
    }
    out = asyncio.run(nodes.node_research(_research_ctx(mem, host), state))
    assert host.queries, "the research ask reached the host"
    query = host.queries[0]
    assert "re-read on every node call" in query      # body excerpt carried
    assert "Speed up config parsing" in query          # title still there
    assert out["action"] in ("verify", "plan")


def test_research_query_without_body_keeps_old_shape(tmp_path, mem):
    host = RecordingHost()
    asyncio.run(nodes.node_research(
        _research_ctx(mem, host),
        {"issue_title": "Survey NFV", "papers": [], "citations": []}))
    query = host.queries[0]
    assert "Problem statement from the issue" not in query
    assert "Survey NFV" in query


def test_research_body_excerpt_is_bounded(tmp_path, mem):
    host = RecordingHost()
    asyncio.run(nodes.node_research(
        _research_ctx(mem, host),
        {"issue_title": "T", "issue_body": "y" * 5000,
         "papers": [], "citations": []}))
    query = host.queries[0]
    assert "y" * 401 not in query
