"""LangGraph research cycle — the heart of every agent thread.

    start → scan | claim → research → plan → implement → open_pr → reflect
    start → research (researcher) → verify → reflect          # P16bis
    start → review → reflect
    start → integrate → reflect
    (no-LLM / trivial cycles skip straight to reflect)

The infinite loop lives in the orchestrator: it re-invokes this graph cycle
after cycle until the owner writes the STOP file.
"""
from __future__ import annotations

import json
import logging

from langgraph.graph import END, StateGraph

from agentics.agents.base import AgentContext
from agentics.agents.roles import Role
from agentics.graph.nodes import (
    node_claim,
    node_defend,
    node_implement,
    node_integrate,
    node_open_pr,
    node_plan,
    node_reflect,
    node_research,
    node_review,
    node_scan,
    node_start,
    node_verify,
    pending_defense,
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

    async def verify_node(state: ResearchState) -> ResearchState:
        return await node_verify(ctx, state)

    g.add_node("research", research_node)
    g.add_node("verify", verify_node)
    g.add_node("plan", plan_node)
    g.add_node("implement", implement_node)
    g.add_node("open_pr", lambda s: node_open_pr(ctx, s))

    async def review_node(state: ResearchState) -> ResearchState:
        return await node_review(ctx, state)

    g.add_node("review", review_node)

    async def defend_node(state: ResearchState) -> ResearchState:
        return await node_defend(ctx, state)

    g.add_node("defend", defend_node)
    g.add_node("integrate", lambda s: node_integrate(ctx, s))
    g.add_node("reflect", lambda s: node_reflect(ctx, s))

    g.set_entry_point("start")

    def route_from_start(state: ResearchState) -> str:
        # P19: defending my challenged MR outranks every default role action —
        # one unanswered objection rejects the MR, so the author goes first.
        try:
            if pending_defense(ctx) is not None:
                return "defend"
        except Exception:  # noqa: BLE001 — a routing read must never kill the cycle
            pass
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
        "review": "review", "integrate": "integrate", "defend": "defend",
        "reflect": "reflect",
    })
    g.add_edge("scan", "reflect")
    # A node that sets `done` (nothing to do, lost a race, yielded) or `error` must
    # short-circuit to reflect; unconditional edges used to push such cycles into
    # plan/implement/open_pr with missing state.
    for src, dst in (("claim", "research"),
                     ("plan", "implement"), ("implement", "open_pr")):
        g.add_conditional_edges(src, continue_or_reflect(dst), {dst: dst, "reflect": "reflect"})

    def route_after_research(state: ResearchState) -> str:
        """P16bis: a researcher cycle holding a fresh finding goes to `verify`
        (claims checked by RUNNING code before they become common knowledge);
        coder/writer cycles continue into plan."""
        if state.get("done") or state.get("error"):
            return "reflect"
        return "verify" if state.get("action") == "verify" else "plan"

    g.add_conditional_edges("research", route_after_research,
                            {"verify": "verify", "plan": "plan",
                             "reflect": "reflect"})
    g.add_edge("verify", "reflect")
    g.add_edge("open_pr", "reflect")
    g.add_edge("review", "reflect")
    g.add_edge("integrate", "reflect")
    g.add_edge("reflect", END)

    return g


def _node_digest(node: str, out: dict) -> str:
    """One line the owner can read: what did this node just do?"""
    task = out.get("issue_number")
    head = f"#{task} " if task else ""
    if node == "claim":
        return f"{head}claimed '{out.get('issue_title', '')}'"[:200]
    if node == "research":
        return f"{head}researched '{out.get('issue_title', '')[:80]}'"[:200]
    if node == "plan":
        return f"{head}planned: {out.get('summary', '')[:140]}"
    if node == "implement":
        if out.get("yield_reason"):
            return f"{head}yielded: {out['yield_reason'][:120]}"
        if out.get("error"):
            return f"{head}implement failed: {out['error'][:120]}"
        return f"{head}implemented {len(out.get('changed_files', []))} file(s)"[:200]
    if node == "open_pr":
        return f"{head}opened MR !{out.get('pr_number')}"
    if node == "review":
        return f"reviewed !{out.get('pr_number', '')}: {out.get('review_verdict', '')}"[:200]
    if node == "defend":
        return f"defended !{out.get('pr_number', '')}: {out.get('defense_outcome', '')}"[:200]
    if node == "integrate":
        return f"merged {out.get('summary', '')}"[:200]
    if out.get("error"):
        return f"{head}error: {out['error'][:140]}"
    if out.get("summary"):
        return f"{head}{out['summary'][:140]}"
    return head + (out.get("action") or "done")


def _node_payload(node: str, out: dict) -> dict:
    """Structured fields for the journal row (dashboards filter on these)."""
    payload: dict = {"node": node, "action": out.get("action", "")}
    for key in ("issue_number", "pr_number", "yield_reason",
                "experiment_verdict", "mechanical_review", "review_verdict",
                "panel_decision", "defense_outcome", "panel_digest"):
        if out.get(key):
            payload[key] = out[key]
    if out.get("error"):
        payload["error"] = out["error"][:400]
    return payload


async def run_cycle(ctx: AgentContext) -> ResearchState:
    """Run one graph cycle, streaming node completions into the shared journal
    (what/why each node just did) — `agentics watch` and the dashboard read
    that feed, so the swarm is never a blackbox. Streaming must never break
    a cycle: journal failures are swallowed (observability, not control flow).
    """
    graph = build_research_graph(ctx).compile()
    final: dict = {}
    # LangGraph runs synchronous I/O nodes in its executor and awaits async
    # LLM/research nodes on the graph event loop.
    async for chunk in graph.astream({}):
        for node_name, update in chunk.items():
            if not isinstance(update, dict):
                continue
            for key, value in update.items():
                if key in ("papers", "citations") and isinstance(value, list):
                    merged, seen = [], set()
                    for item in [*final.get(key, []), *value]:
                        k = json.dumps(item, sort_keys=True, ensure_ascii=False,
                                       default=str) if not isinstance(item, str) else item
                        if k not in seen:
                            seen.add(k)
                            merged.append(item)
                    final[key] = merged
                else:
                    final[key] = value
            try:
                ctx.memory.journal(
                    ctx.agent_id, "node",
                    {"digest": _node_digest(node_name, final),
                     **_node_payload(node_name, final)},
                )
            except Exception as exc:  # noqa: BLE001 — observability, not control flow
                log.debug("could not journal node %s: %s", node_name, exc)
    return final
