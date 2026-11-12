"""A node that finishes or fails the cycle must send it to `reflect`."""

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("langgraph")

import agentics.graph.research_graph as rg  # noqa: E402
from agentics.agents.roles import Role  # noqa: E402

ALL = [
    "start",
    "scan",
    "claim",
    "research",
    "plan",
    "implement",
    "open_pr",
    "review",
    "integrate",
    "reflect",
]


def _run(monkeypatch, role, updates=None):
    updates = updates or {}
    calls: list[str] = []

    def make(name):
        if name in {"research", "plan", "implement", "review"}:
            async def async_node(ctx, state):
                calls.append(name)
                state.update(updates.get(name, {}))
                return state
            return async_node

        def node(ctx, state):
            calls.append(name)
            state.update(updates.get(name, {}))
            return state

        return node

    for n in ALL:
        monkeypatch.setattr(rg, f"node_{n}", make(n))
    asyncio.run(rg.build_research_graph(SimpleNamespace(role=role)).compile().ainvoke({}))
    return calls


def test_happy_path_runs_every_step(monkeypatch):
    assert _run(monkeypatch, Role.CODER) == [
        "start",
        "claim",
        "research",
        "plan",
        "implement",
        "open_pr",
        "reflect",
    ]


def test_no_issue_goes_straight_to_reflect(monkeypatch):
    calls = _run(monkeypatch, Role.CODER, {"claim": {"done": True}})
    assert calls == ["start", "claim", "reflect"]


def test_plan_error_skips_implement_and_open_pr(monkeypatch):
    calls = _run(monkeypatch, Role.CODER, {"plan": {"error": "plan parsing failed"}})
    assert calls == ["start", "claim", "research", "plan", "reflect"]


def test_yield_in_implement_skips_open_pr(monkeypatch):
    calls = _run(monkeypatch, Role.CODER, {"implement": {"done": True, "yield_reason": "foreign"}})
    assert calls[-2:] == ["implement", "reflect"] and "open_pr" not in calls


def test_researcher_does_not_fall_through_into_plan(monkeypatch):
    calls = _run(monkeypatch, Role.RESEARCHER, {"research": {"done": True}})
    assert calls == ["start", "research", "reflect"]
