"""Swarm-wide lessons (A2X module 3) — failure patterns across ALL agents.

The planner already saw its own lessons and a few raw trace rows from other
agents; what no prompt ever showed is the PATTERN: the same failure
signature repeated by several different agents proves the approach never
worked. These tests pin the aggregation (signature -> xN count -> last
agent), the exclusion of the planner's own rows (they arrive as personal
lessons), the empty/bounded edges, and the node_plan wiring.
"""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentics.agents.roles import Role
from agentics.coordination.lessons import swarm_lessons
from agentics.graph import nodes
from agentics.memory.shared_memory import SharedMemory


@pytest.fixture()
def mem(tmp_path: Path):
    m = SharedMemory(tmp_path / "m.db")
    yield m
    m.close()


def _fail(mem, agent, category, failure, outcome="failed"):
    mem.add_trace(agent, "coder", 1, outcome, category=category, failure=failure)


def test_swarm_lessons_aggregates_repeats_with_last_agent(mem):
    _fail(mem, "a01", "bugfix", "timeout under load")
    _fail(mem, "a02", "bugfix", "timeout under load")
    _fail(mem, "a01", "bugfix", "timeout under load")
    _fail(mem, "a03", "bugfix", "regex backtracks on nested input")

    text = swarm_lessons(mem, "bugfix")
    lines = text.splitlines()
    assert lines[0] == "- timeout under load (x3; last: a01)"
    assert lines[1] == "- regex backtracks on nested input (x1; last: a03)"


def test_swarm_lessons_excludes_the_planners_own_rows(mem):
    _fail(mem, "a01", "bugfix", "timeout under load")
    _fail(mem, "a02", "bugfix", "timeout under load")
    _fail(mem, "a01", "bugfix", "regex backtracks")
    text = swarm_lessons(mem, "bugfix", exclude_agent="a01")
    assert "- timeout under load (x1; last: a02)" in text
    assert "regex backtracks" not in text


def test_swarm_lessons_empty_cases(mem):
    assert swarm_lessons(mem, None) == ""
    assert swarm_lessons(mem, "bugfix") == ""          # nothing recorded
    _fail(mem, "a02", "bugfix", "boom")
    assert swarm_lessons(mem, "docs") == ""            # wrong category
    assert swarm_lessons(mem, "bugfix", exclude_agent="a02") == ""


def test_swarm_lessons_rejected_mrs_count_too(mem):
    _fail(mem, "a02", "code", "panel rejected: no P16 checks", outcome="rejected")
    text = swarm_lessons(mem, "code")
    assert "panel rejected: no P16 checks (x1; last: a02)" in text


def test_swarm_lessons_is_bounded(mem):
    for i in range(12):
        _fail(mem, f"a{i % 5 + 1:02d}", "code", "failure " + "z" * 200 + f" #{i}")
    text = swarm_lessons(mem, "code", limit=4, max_chars=300)
    assert len(text) <= 300


# ------------------------------------------------------------ node_plan wiring
def _plan_ctx(mem, reports):
    return SimpleNamespace(
        agent_id="a04", role=Role.CODER, memory=mem, llm=None, claims=None,
        cycle=1, meta={"personal": SimpleNamespace(add_lesson=lambda *a, **k: None)},
        settings=SimpleNamespace(
            research_topic="NFV", evolution_every_n_cycles=10,
            adaptive_prompts=True, role_prompt_ttl_hours=24.0,
            role_prompt_max_chars=1600, role_prompt_rework_failures=2),
        heartbeat=lambda status="working": None,
        report=lambda **kw: reports.append(kw), local=None,
    )


def _capture_llm(monkeypatch, seen):
    async def fake_llm(ctx, prompt, mode="concise", purpose="general", state=None):
        seen["p"] = prompt
        return json.dumps({"files": [], "steps": ["s1"], "commit_message": "m"}), []
    monkeypatch.setattr(nodes, "_llm_ask", fake_llm)


def test_node_plan_prompt_carries_swarm_patterns(mem, monkeypatch):
    seen, reports = {}, []
    _capture_llm(monkeypatch, seen)
    _fail(mem, "a02", "code", "plan parsing failed twice on nested JSON")
    _fail(mem, "a03", "code", "plan parsing failed twice on nested JSON")
    state = asyncio.run(nodes.node_plan(_plan_ctx(mem, reports), {
        "issue_number": 1, "issue_title": "Parser", "issue_body": "",
        "papers": [{"answer": ""}], "category": "code",
    }))
    assert "Swarm-wide failure patterns in this category" in seen["p"]
    assert "plan parsing failed twice on nested JSON (x2; last: a03)" in seen["p"]
    assert state["commit_message"] == "m"


def test_node_plan_prompt_clean_when_no_swarm_failures(mem, monkeypatch):
    seen, reports = {}, []
    _capture_llm(monkeypatch, seen)
    _fail(mem, "a02", "docs", "panel rejected: no P16 checks")   # other category
    state = asyncio.run(nodes.node_plan(_plan_ctx(mem, reports), {
        "issue_number": 1, "issue_title": "Parser", "issue_body": "",
        "papers": [{"answer": ""}], "category": "code",
    }))
    assert "Swarm-wide failure patterns" not in seen["p"]
    assert state["commit_message"] == "m"
