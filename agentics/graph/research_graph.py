"""LangGraph research cycle — the heart of every agent thread.

    start → scan | claim → research → plan → implement → open_pr → reflect
    start → review → reflect
    start → integrate → reflect
    (no-LLM / trivial cycles skip straight to reflect)

The infinite loop lives in the orchestrator: it re-invokes this graph cycle
after cycle until the owner writes the STOP file.
"""
from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph

from agentics.agents.base import AgentContext
from agentics.agents.roles import Role
from agentics.graph.nodes import (
    node_claim,
    node_implement,
    node_integrate,
    node_open_pr,
    node_plan,
    node_reflect,
    node_research,
    node_review,
    node_scan,
    node_start,
)
from agentics.graph.state import ResearchState

log = logging.getLogger("graph.cycle")


def continue_or_reflect(next_node: str):
    """Router: go on to `next_node` unless the cycle is already finished or failed."""
    def route(state: ResearchState) -> str:
        return "reflect" if state.get("done") or state.get("error") else next_node
    return route


def build_research_graph(ctx: AgentContext) -> StateGraph:
    g = StateGraph(ResearchState)

    g.add_node("start", lambda s: node_start(ctx, s))
    g.add_node("scan", lambda s: node_scan(ctx, s))
    g.add_node("claim", lambda s: node_claim(ctx, s))
    # Async nodes must be registered as async callables. A sync lambda that merely
    # returns a coroutine is treated as a sync node and leaks the coroutine object.
    async def research_node(state: ResearchState) -> ResearchState:
        return await node_research(ctx, state)

    async def plan_node(state: ResearchState) -> ResearchState:
        return await node_plan(ctx, state)

    async def implement_node(state: ResearchState) -> ResearchState:
        return await node_implement(ctx, state)

    g.add_node("research", research_node)
    g.add_node("plan", plan_node)
    g.add_node("implement", implement_node)
    g.add_node("open_pr", lambda s: node_open_pr(ctx, s))
    g.add_node("review", lambda s: node_review(ctx, s))
    g.add_node("integrate", lambda s: node_integrate(ctx, s))
    g.add_node("reflect", lambda s: node_reflect(ctx, s))

    g.set_entry_point("start")

    def route_from_start(state: ResearchState) -> str:
        return {
            Role.SUPERVISOR: "scan",
            Role.RESEARCHER: "research",
            Role.CODER: "claim",
            Role.WRITER: "claim",
            Role.REVIEWER: "review",
            Role.INTEGRATOR: "integrate",
        }.get(ctx.role, "reflect")

    g.add_conditional_edges("start", route_from_start, {
        "scan": "scan", "research": "research", "claim": "claim",
        "review": "review", "integrate": "integrate", "reflect": "reflect",
    })
    g.add_edge("scan", "reflect")
    # A node that sets `done` (nothing to do, lost a race, yielded) or `error` must
    # short-circuit to reflect; unconditional edges used to push such cycles into
    # plan/implement/open_pr with missing state.
    for src, dst in (("claim", "research"), ("research", "plan"),
                     ("plan", "implement"), ("implement", "open_pr")):
        g.add_conditional_edges(src, continue_or_reflect(dst), {dst: dst, "reflect": "reflect"})
    g.add_edge("open_pr", "reflect")
    g.add_edge("review", "reflect")
    g.add_edge("integrate", "reflect")
    g.add_edge("reflect", END)

    return g


async def run_cycle(ctx: AgentContext) -> ResearchState:
    graph = build_research_graph(ctx).compile()
    # LangGraph runs synchronous I/O nodes in its executor and awaits async
    # LLM/research nodes on the graph event loop.
    return await graph.ainvoke({})
