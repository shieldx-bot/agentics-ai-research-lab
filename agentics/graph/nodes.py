"""Graph nodes — each node is a step of one agent's research cycle.

Local-only mode: tasks/reviews/claims live in shared memory, git lives in the
shared bare repo (data/lab.git) — this module performs ZERO network calls.
Sync Git/SQLite operations run in LangGraph's executor; LLM and paper-search
nodes are async and await their providers directly.
"""
from __future__ import annotations

import asyncio
import json
import logging
import shlex
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from agentics.agents.base import AgentContext
from agentics.agents.roles import Role, parse_json_block
from agentics.evolution.scoring import IssueInfo, build_profile, categorize_issue, rank_issues
from agentics.evolution.traces import (
    EVOLUTION_ALLOWED_PATHS,
    build_evolution_issue,
    evolution_path_allowed,
    tool_context,
    trace_context,
    valid_tool_name,
)
from agentics.graph.state import ResearchState
from agentics.local_ops import branch_name
from agentics.peer_review import (
    DEFENSE, OUT_APPROVED, OUT_DEFENSE, OUT_REJECTED, PeerPanel,
)
from agentics.planning import (
    PLAN_INSTRUCTION,
    bump_prompt_feedback,
    extras_state_json,
    normalize_plan,
    plan_state_json,
    resolve_system_prompt,
)
from agentics.policies.guard import is_credential_path
from agentics.memory import lifemem
from agentics.policies.intent import scope_violations
from agentics.research import citations as cite_fmt
from agentics.research import free_tools
from agentics.research import papers as papers_mod
from agentics.verify_metrics import PROTOCOL_CONTRACT, Criterion, run_protocol
from agentics.verify_ops import (
    Evidence, cmds_from_evidence, run_experiment, run_experiments, summarize,
    verdict_for,
)

log = logging.getLogger("graph.nodes")

STOP_YIELD = "yield"


def _panel_engine(ctx: AgentContext) -> PeerPanel:
    """P19 engine wired from the ctx. The orchestrator injects ctx.panel;
    tests and older callers get one built on the fly (same rulebook)."""
    injected = getattr(ctx, "panel", None)
    if injected is not None:
        return injected
    s = getattr(ctx, "settings", None)
    return PeerPanel(
        ctx.memory,
        panel_size=int(getattr(s, "peer_panel_size", 2) or 2),
        defense_ttl_minutes=int(getattr(s, "peer_defense_ttl_minutes", 30) or 30),
    )


# ---------------------------------------------------------------- helpers
def _lessons_text(ctx: AgentContext, query: str = "", limit: int = 5,
                  issue: int | None = None) -> str:
    """The agent's OWN lessons from past cycles — the personal memory was
    write-only until now; this closes the loop so an agent stops repeating
    its own failures. With a ``query`` the lessons are ranked by relevance
    to the current task (plus a same-issue boost, v3); without one, by
    recency. Defensive: contexts (or test doubles) without the recall
    methods degrade to "" — never a crash."""
    personal = (getattr(ctx, "meta", None) or {}).get("personal")
    if personal is None:
        return ""
    try:
        if query and hasattr(personal, "relevant_lessons"):
            try:      # v3 signature (query, limit, issue); v2 fakes lack `issue`
                lessons = personal.relevant_lessons(query, limit, issue=issue)
            except TypeError:
                lessons = personal.relevant_lessons(query, limit)
        elif hasattr(personal, "recent_lessons"):
            lessons = personal.recent_lessons(limit)
        else:
            return ""
    except Exception:  # noqa: BLE001 — memory must never break a cycle
        return ""
    return "\n".join(f"- {str(l)[:220]}" for l in lessons if l)


def finding_recall_text(ctx: AgentContext, query: str, limit: int = 3) -> str:
    """Relevance recall of stored findings — the shared semantic memory.

    The vector store already holds every finding the researcher ever wrote
    (``node_research`` adds it); until now that index was only queried for
    DEDUP. This makes the same index answer "what do we already know about
    this?" at plan time. Vector hits whose SQLite row was trimmed out are
    skipped, so recall never cites a ghost finding. Returns "" when chromadb
    is absent or nothing matches — the plan prompt is unchanged at cold start.
    """
    vector = getattr(ctx, "vector", None)
    if not (query or "").strip() or vector is None or not getattr(vector, "available", False):
        return ""
    try:
        hits = vector.search(query, n=max(limit * 2, limit))
        lines: list[str] = []
        for h in hits:
            meta = h.get("meta") or {}
            fid = str(meta.get("finding_id", ""))
            if not fid.isdigit():
                continue
            f = ctx.memory.get_finding(int(fid))
            if f is None:
                continue          # vector entry outlived its trimmed SQLite row
            digest = " ".join((f["content"] or "").split())[:220]
            lines.append(f"- #{f['id']} {f['title']} (by {f['agent']}): {digest}")
            if len(lines) >= limit:
                break
    except Exception:  # noqa: BLE001 — recall must never break planning
        return ""
    return "\n".join(lines)


def _memory_context(ctx: AgentContext) -> dict:
    msgs = ctx.memory.recent_messages(limit=15)
    findings = ctx.memory.list_findings(limit=10)
    evs = ctx.memory.latest_evidence(limit=6)
    return {
        "messages": "\n".join(
            f"- [{m['agent']}|{m['kind']}] {m['content'][:200]}" for m in msgs
        ) or "(empty)",
        "findings": "\n".join(
            f"- #{f['id']} {f['title']} (by {f['agent']})" for f in findings
        ) or "(empty)",
        # v2: the agent's own deduped lessons — reflect writes them, this
        # snapshot (and node_plan) read them back. Recurring failures show
        # as (xN) patterns instead of N identical rows.
        "lessons": _lessons_text(ctx) or "(none yet)",
        # H3: every agent's snapshot carries the swarm's settled knowledge —
        # supported claims are reusable common ground, refuted ones are dead
        # ends (rendered by verified_knowledge_text, below).
        "verified": verified_knowledge_text(ctx) or "(nothing verified yet)",
    }


def verified_knowledge_text(ctx: AgentContext, limit: int = 6) -> str:
    """H3 consumption side — the swarm's settled knowledge as prompt text.

    Composes the memory layer's `verified_knowledge()` (settled hypotheses:
    supported become reusable common ground, refuted stop being repeated)
    with the newest evidence rows (real runs by real agents). Returns ""
    when nothing has been verified yet, so prompts stay unchanged at cold
    start. Bounded: a prompt must not drown in the lab's whole history."""
    try:
        hyps = ctx.memory.verified_knowledge(limit=limit)
    except Exception:  # noqa: BLE001 — knowledge injection must never break a cycle
        return ""
    lines: list[str] = []
    for h in hyps:
        # P24bis: an INDEPENDENTLY replicated claim is stronger common
        # ground than an author-confirmed one — the prompt says so.
        mark = ("SUPPORTED+ (independently replicated)"
                if h["status"] == "supported" and h.get("replicated")
                else "SUPPORTED" if h["status"] == "supported" else "REFUTED")
        note = f" — {h['note'][:110]}" if h.get("note") else ""
        lines.append(f"- [{mark}] {h['claim'][:180]} (h#{h['id']}{note})")
    if not lines:
        return ""
    return ("Empirically verified by this lab (exit-code verdicts, P16 — build on "
            "SUPPORTED facts, never re-assert a REFUTED one; SUPPORTED+ survived "
            "an independent agent's re-run, the strongest kind):\n" + "\n".join(lines))[:1400]


async def _llm_ask(
    ctx: AgentContext, prompt: str, mode: str = "concise", purpose: str = "general",
    state: ResearchState | None = None,
) -> tuple[str, list[str]]:
    """Await the BrowserSessionHost directly; never create or nest an event
    loop in a graph worker. Purpose tags the call for provider routing
    (research wants citations; plan/implement/review want the code model).

    Resilience: a provider failure (cookie cooldown, challenge page, WAF,
    exhausted pool) must NOT kill the LangGraph task — it is recorded in
    ``state["error"]`` so the cycle degrades to `reflect` (claim released,
    trace + personal lesson written) and the swarm retries next cycle.

    P20 — budget gate: every ask passes the shared BudgetPolicy (pool-wide
    hourly cap + per-agent circuit breaker) and is recorded in `llm_usage`
    either way, so a runaway agent is a bounded event, not an infinite burn.
    """
    if ctx.llm is None:
        if state is not None:
            state["error"] = "no LLM account available"
        return "", []
    budget = getattr(ctx, "budget", None)
    started = time.monotonic()
    if budget is not None:
        allowed, why = budget.can_ask(ctx.agent_id)
        if not allowed:
            log.warning("[%s] %s blocked by budget: %s", ctx.agent_id, purpose, why)
            if state is not None and not state.get("error"):
                state["error"] = f"P20 budget: {why}"
            try:
                ctx.memory.journal(ctx.agent_id, "budget_blocked",
                                   {"purpose": purpose, "why": why[:200]})
            except Exception:  # noqa: BLE001 — telemetry must never gate harder
                pass
            return "", []
    try:
        system = await resolve_system_prompt(ctx, state)
        resp = await ctx.llm.ask(
            prompt, mode=mode, purpose=purpose, system=system
        )
        if budget is not None:
            budget.record(ctx.agent_id, purpose, True, time.monotonic() - started)
    except Exception as exc:  # noqa: BLE001 — degrade, don't crash the cycle
        log.warning("[%s] %s llm call failed: %s", ctx.agent_id, purpose, exc)
        if budget is not None:
            budget.record(ctx.agent_id, purpose, False, time.monotonic() - started)
        if state is not None and not state.get("error"):
            state["error"] = f"llm {purpose} call failed: {exc}"
        return "", []
    acc = getattr(ctx.llm, "account", None)
    ctx.meta["account"] = getattr(acc, "name", "none") if acc else "none"
    return resp.answer, resp.citations


# ---------------------------------------------------------------- nodes
def node_start(ctx: AgentContext, state: ResearchState) -> ResearchState:
    ctx.cycle += 1
    ctx.heartbeat("cycle-start")
    msgs = _memory_context(ctx)
    out: ResearchState = {
        "agent_id": ctx.agent_id,
        "role": ctx.role.value,
        "memory_context": msgs,
        "papers": [],
        "citations": [],
        "changed_files": [],
        "done": False,
        "error": "",
        "yield_reason": "",
    }
    # role → default action for this cycle
    out["action"] = {
        Role.SUPERVISOR: "scan",
        Role.RESEARCHER: "research",
        Role.CODER: "claim",
        Role.WRITER: "claim",
        Role.REVIEWER: "review",
        Role.INTEGRATOR: "integrate",
    }.get(ctx.role, "idle")
    # P19: defending my challenged MR outranks the default action — one
    # unanswered objection rejects the MR for good, so the author goes first.
    try:
        if pending_defense(ctx) is not None:
            out["action"] = "defend"
    except Exception:  # noqa: BLE001 — display only, routing decides separately
        pass
    ctx.report(
        node="start", status="working",
        intent={"goal": f"{out['action']} cycle as {ctx.role.value}", "steps": []},
    )
    return out


def node_scan(ctx: AgentContext, state: ResearchState) -> ResearchState:
    """Supervisor: sweep stale claims, keep the local backlog fed."""
    state["action"] = "scan"
    freed = ctx.claims.sweep_stale()
    try:                       # P13 hygiene: dead agents must not hold files
        stale_locks = ctx.memory.sweep_file_locks()
    except Exception:  # noqa: BLE001
        stale_locks = []
    # repair tasks stalled without a lease / without a live MR (after the sweep,
    # so expired leases released above count as repaired orphans)
    tasks_src = getattr(getattr(ctx, "local", None), "tasks", None)
    repaired = tasks_src.repair_orphan_tasks() if tasks_src is not None else []
    # P19 hygiene: defense deadlines and vanished-MR panels are swept the way
    # stale leases are — an undefended MR dies here, the task returns to the
    # open pool, and every agent sees why.
    try:
        swept_panels = _panel_engine(ctx).sweep()
    except Exception:  # noqa: BLE001 — panel sweep must never break the scan
        swept_panels = []
    if swept_panels:
        ctx.memory.journal(ctx.agent_id, "panel_sweep",
                           {"events": [list(e) for e in swept_panels[:10]]})
    # P22 retention: workspaces of dead agents that hold nothing and carry no
    # unpushed work get pruned here (agentics/policies/retention.py).
    try:
        from agentics.policies.retention import sweep_stale_workspaces
        pruned_ws = sweep_stale_workspaces(
            ctx.memory, ctx.settings.workspaces_dir,
            ctx.settings.workspace_retention_minutes,
            ctx.memory.active_agents(stale_after_s=300),
            agent_id=ctx.agent_id)
        if pruned_ws:
            ctx.memory.post_message(
                ctx.agent_id, "alert",
                f"P22 pruned {len(pruned_ws)} idle workspace(s): {pruned_ws}")
    except Exception:  # noqa: BLE001 — retention must never break the scan
        pruned_ws = []
    # P23 replication: on schedule, re-run settled claims from their stored
    # protocol source — the swarm's knowledge must survive re-measurement,
    # not be asserted once. Drift (supported->refuted), overturns and
    # contradictions land in the shared evidence feed (method='replay'), so
    # every agent sees replication results, not just the replayer.
    replay_report: list[dict] = []
    try:
        sweep_size = int(getattr(ctx.settings, "replay_sweep_size", 0) or 0)
        every_n = int(getattr(ctx.settings, "replay_every_n_cycles", 1) or 1)
        workspace = getattr(ctx, "workspace", None)
        if sweep_size > 0 and every_n > 0 and ctx.cycle % every_n == 0 and workspace is not None:
            from agentics.verify.replay import replay_settled
            replay_report = replay_settled(
                ctx.memory, workspace, limit=sweep_size, agent=ctx.agent_id)
    except Exception:  # noqa: BLE001 — replication must never break the scan
        replay_report = []
    replayed_n = sum(1 for r in replay_report if r.get("replayed"))
    replay_events = [r.get("event", "") for r in replay_report if r.get("replayed")]
    tasks = ctx.memory.scan_open_tasks(limit=30)
    if freed:
        ctx.memory.post_message(
            ctx.agent_id, "alert",
            f"Freed {len(freed)} stale claim(s): {freed}"
        )
    topped_up = 0
    if not tasks and tasks_src is not None:
        topped_up = tasks_src.supervisor_top_up()
        tasks = ctx.memory.scan_open_tasks(limit=30)
    summary = (
        f"Backlog scan: {len(tasks)} open unclaimed task(s)."
        + (f" Stale claims freed: {freed}." if freed else "")
        + (f" Stale file locks freed: {stale_locks}." if stale_locks else "")
        + (f" Repaired stalled task(s): {repaired}." if repaired else "")
        + (f" Opened {topped_up} new survey task(s)." if topped_up else "")
        + (f" P19 panel events: {len(swept_panels)}." if swept_panels else "")
        + (f" P23 replayed {replayed_n} claim(s): {replay_events}."
           if replayed_n else "")
    )
    ctx.memory.post_message(ctx.agent_id, "broadcast", summary)
    state["summary"] = summary
    _maybe_open_evolution_task(ctx, state)
    _check_integrity(ctx)
    state["done"] = True
    return state


def _check_integrity(ctx: AgentContext) -> None:
    """DETECT layer (every 5th supervisor cycle): hash-check the tree the lab
    itself runs from. P10 changeset protection stops agent writes into the
    engine; this catches everything else (botched merge, escaped process,
    manual edit). Drift is an owner-visible alert, never a cycle error."""
    try:
        if ctx.cycle % 5 != 0:
            return
        from agentics.config import PROJECT_ROOT
        from agentics.sandbox.integrity import IntegrityGuard

        guard = IntegrityGuard(PROJECT_ROOT, ctx.settings.integrity_path)
        _, drift = guard.ensure_baseline()
        if drift:
            ctx.memory.journal(ctx.agent_id, "integrity_alert", {"drift": drift[:10]})
            ctx.memory.post_message(
                ctx.agent_id, "alert",
                f"Runtime integrity drift: {len(drift)} file(s) differ from the "
                "baseline - run `agentics verify` (owner decision)",
            )
    except Exception as exc:  # noqa: BLE001
        log.debug("integrity check skipped: %s", exc)


def _maybe_open_evolution_task(ctx: AgentContext, state: ResearchState) -> None:
    """Every N supervisor cycles, open ONE evidence-based self-evolution task.

    In local mode a self-evolution task may only touch agent code when the
    OWNER explicitly grants a protected-paths exemption
    (`agentics.cli allow-protected <task_id>`) — the guard (P10) enforces it
    at write time, at diff time and again before the merge."""
    n = int(getattr(ctx.settings, "evolution_every_n_cycles", 0) or 0)
    if n <= 0 or ctx.cycle % n != 0:
        return
    if not ctx.memory.recent_traces(limit=1, failures_only=True):
        return                      # no failure evidence yet → nothing to evolve from
    title, body = build_evolution_issue(ctx.memory)
    task_id = ctx.memory.create_task_if_absent(
        title, body=body, category="evolution", labels="kind/evolution",
        agent=ctx.agent_id,
    )
    if task_id is not None:
        ctx.memory.add_decision(
            ctx.agent_id, f"open self-evolution task #{task_id}",
            "periodic GEA-style prompt refinement (needs owner exemption to touch code)")
        state["summary"] += f" Opened self-evolution task #{task_id}."


def _pick_rework(ctx: AgentContext) -> tuple[dict, dict] | None:
    """My oldest changes_requested MR whose task still exists (author rework).

    P5 hygiene: address the review feedback before claiming new work. The task
    stays parked in in_review during the rework — no new claim, so a crash
    mid-rework leaves the MR exactly where it was and the same author simply
    retries next cycle.
    """
    for mr in ctx.memory.reviews_by_author(ctx.agent_id, status="changes_requested"):
        task = ctx.memory.get_task(mr["task_id"]) if mr.get("task_id") else None
        if task is None or task["status"] not in ("in_review",):
            continue
        return mr, task
    return None


def node_claim(ctx: AgentContext, state: ResearchState) -> ResearchState:
    """Coder/Writer: claim a task (P1), ranked by Performance–Novelty.

    Replaces "oldest first": each task is scored with this agent's own merge
    history (see agentics/evolution/scoring.py); ties fall back to oldest-first.
    Only the Coder may take self-evolution tasks.

    A changes_requested MR authored by this agent comes first: the review
    feedback must be reworked (same branch, same MR) before new claims.
    """
    if ctx.role in (Role.CODER, Role.WRITER):
        rework = _pick_rework(ctx)
        if rework is not None:
            mr, task = rework
            feedback = (mr.get("verdict_comment") or "").strip()
            state["rework_mr"] = mr["id"]
            state["issue_number"] = task["id"]
            state["issue_title"] = task["title"]
            state["issue_body"] = (
                (task["body"] or "")[:4000]
                + f"\n\n## Review feedback on MR #{mr['id']} to address\n{feedback[:2000]}"
            )
            state["branch"] = mr["branch"]
            state["category"] = task.get("category") or "other"
            state["task_exempt"] = bool(task.get("exempt"))
            state["action"] = "research"
            ctx.memory.add_decision(
                ctx.agent_id, f"rework MR #{mr['id']} (task #{task['id']})",
                f"feedback: {feedback[:200]}")
            ctx.memory.journal(ctx.agent_id, "rework_start", {
                "mr": mr["id"], "task": task["id"], "feedback": feedback[:300],
            })
            ctx.report(
                node="claim", task_id=task["id"], task_title=task["title"],
                intent={
                    "goal": f"Rework MR #{mr['id']} on task #{task['id']}",
                    "why": f"review feedback: {feedback[:160]}",
                    "steps": [], "expected_files": [],
                },
            )
            return state
    tasks = ctx.memory.scan_open_tasks(limit=30)
    infos = [
        IssueInfo(
            number=t["id"], title=t["title"], body=(t["body"] or "")[:800],
            labels=tuple(t["labels"].split(",")) if t.get("labels") else (),
            created_ts=t["ts"],
        )
        for t in tasks
    ]
    if ctx.role != Role.CODER:
        infos = [i for i in infos if i.category != "evolution"]
    if not infos:
        state["summary"] = "no unclaimed tasks; resting this cycle"
        state["done"] = True
        return state
    by_id = {i.number: i for i in infos}
    profile = build_profile(ctx.memory.agent_outcomes().get(ctx.agent_id, []))
    for info, score in rank_issues(infos, profile)[:3]:      # lost race → next best
        task = next(t for t in tasks if t["id"] == info.number)
        branch = branch_name(ctx.agent_id, task["id"], task["title"])
        if not ctx.claims.try_claim(task["id"], ctx.agent_id, branch):
            continue
        state["issue_number"] = task["id"]
        state["issue_title"] = task["title"]
        state["issue_body"] = (task["body"] or "")[:4000]
        state["branch"] = branch
        state["category"] = info.category
        state["task_exempt"] = bool(task.get("exempt"))
        state["action"] = "research"
        ctx.memory.add_decision(
            ctx.agent_id, f"claim task #{task['id']}",
            f"category={info.category} score={score:.2f} (alpha*sqrt(nov)*priority)")
        ctx.report(
            node="claim", task_id=task["id"], task_title=task["title"],
            intent={
                "goal": f"Complete task #{task['id']}: {task['title']}",
                "why": f"category={info.category} score={score:.2f}",
                "steps": [], "expected_files": [],
            },
        )
        return state
    state["summary"] = "lost race for top-ranked tasks; retry next cycle"
    state["done"] = True
    return state


# ---------------------------------------------------------------- research helpers
async def _grounded_bibtex(ctx: AgentContext, urls: list[str]) -> str:
    """BibTeX from REAL publisher metadata (arXiv/Crossref) with a safe
    fallback to the legacy <title> scrape for unknown URLs — grounded
    citations beat scraped ones, but a broken source must never break the
    cycle (docs/FREE_TOOLS.md)."""
    try:
        return await free_tools.enriched_bibtex(list(urls), settings=ctx.settings)
    except Exception as exc:  # noqa: BLE001
        log.warning("[%s] bibtex grounding fell back to legacy: %s", ctx.agent_id, exc)
        return cite_fmt.to_bibtex(cite_fmt.parse_citations(list(urls)))


# ---------------------------------------------------------------- research fan-out (P18b)
_FANOUT_ANGLES = (
    ("benchmarks & datasets",
     "Focus on benchmark suites, public datasets and reproducible evaluation "
     "protocols for the topic. Name concrete datasets and metric numbers."),
    ("limitations & open problems",
     "Focus on failure modes, limitations, negative results and open research "
     "problems reported for the topic. Cite the papers that report them."),
)


def _fanout_views(ctx: AgentContext, n_extra: int) -> list:
    """One extra BrowserSessionHost view per research angle, each on a
    DIFFERENT pool identity than the agent's own pinned view — with the P18
    token pool that is a different Hyperbrowser token (a genuinely parallel
    cloud session); on the local websocket backend it is a different cookie
    account. Fewer slots than angles -> fewer angles, never contention."""
    host = ctx.llm
    if host is None or not hasattr(host, "view"):
        return []
    backend = (getattr(ctx.settings, "browser_backend", "") or "local").lower()
    if backend == "hyperbrowser":
        capacity = max(1, len(getattr(ctx.settings, "hyperbrowser_api_keys", None) or []))
    else:
        try:
            capacity = len(host.pools["perplexity"].accounts) or 1
        except Exception:  # noqa: BLE001 — probing the pool must not break research
            capacity = 1
    if capacity < 2 or n_extra < 1:
        return []                       # a single identity cannot fan out
    base = getattr(host, "prefer_index", 0) or 0
    views = []
    for k in range(1, min(n_extra + 1, capacity)):
        views.append(host.view(prefer_index=(base + k) % capacity))
    return views


async def _fanout_angles(ctx: AgentContext, base_query: str, n_extra: int,
                         budget) -> list[tuple[str, object]]:
    """Run the complementary research angles in parallel on the extra views.

    Every angle passes the same P20 budget gate as the primary ask and is
    recorded in the usage ledger; a blocked, failed or empty angle simply
    contributes nothing instead of breaking the finding."""
    views = _fanout_views(ctx, n_extra)
    if not views:
        return []

    async def one(view, angle):
        label, hint = angle
        if budget is not None:
            allowed, why = budget.can_ask(ctx.agent_id)
            if not allowed:
                log.info("[%s] fan-out angle '%s' skipped by budget: %s",
                         ctx.agent_id, label, why)
                return None
        started = time.monotonic()
        try:
            r = await papers_mod.search_papers(
                view,
                f"{base_query}\n\n[Angle: {label}] {hint}",
                purpose="research")
        except Exception as exc:  # noqa: BLE001 — one angle must not kill the finding
            log.warning("[%s] fan-out angle '%s' failed: %s", ctx.agent_id, label, exc)
            return None
        if budget is not None:
            budget.record(ctx.agent_id, "research", bool(r and r.ok),
                          time.monotonic() - started)
        if r is None or not r.ok:
            return None
        return label, r

    results = await asyncio.gather(
        *(one(v, a) for v, a in zip(views, _FANOUT_ANGLES[:len(views)])))
    ran = [r for r in results if r is not None]
    if ran:
        ctx.meta["research_fanout"] = 1 + len(ran)
        try:
            ctx.memory.journal(ctx.agent_id, "research_fanout",
                               {"angles": [label for label, _ in ran]})
        except Exception:  # noqa: BLE001 — telemetry must never gate harder
            pass
    return ran


def _merge_research(primary, extras: list[tuple[str, object]]):
    """Merge parallel angle answers into the primary finding: primary text
    first, then each angle under its own heading; citations deduped in
    order. The merged shape is exactly the LLMResponse the rest of the
    pipeline already consumes."""
    from agentics.llm.perplexity_client import LLMResponse

    parts = [primary.answer.rstrip()]
    citations = list(primary.citations)
    seen = set(citations)
    for label, r in extras:
        parts.append(f"## Supplementary angle — {label}\n{r.answer.strip()}")
        for c in r.citations:
            if c and c not in seen:
                seen.add(c)
                citations.append(c)
    return LLMResponse(
        answer="\n\n".join(parts),
        citations=citations,
        raw=getattr(primary, "raw", None),
        elapsed_s=getattr(primary, "elapsed_s", 0.0),
    )


async def _finish_research(ctx: AgentContext, state: ResearchState, topic: str,
                           answer: str, citations: list[str], source: str) -> ResearchState:
    """Shared tail of node_research: persist the finding and route the cycle.

    One code path for both evidence sources (LLM cookies and the free
    keyless APIs) so shared memory / vector store rows keep the exact same
    shape regardless of where the metadata came from."""
    state["papers"] = [{"title": topic, "answer": answer}]
    state["citations"] = list(citations)
    title = f"Findings: {topic}"
    content = f"{answer[:6000]}\n\n## BibTeX\n\n{state['bibtex'][:3000]}"
    fid = await asyncio.to_thread(
        ctx.memory.add_finding,
        ctx.agent_id, title, content, list(citations), state.get("issue_number"),
    )
    # v3 sync-trim: remember WHICH vector doc indexes this row, so the day the
    # findings trim evicts it the vector tier deletes the same doc (the ghost
    # rows the recall path used to skip at read time never accumulate).
    doc_id = await asyncio.to_thread(
        ctx.vector.add, content,
        {"agent": ctx.agent_id, "finding_id": fid, "issue": state.get("issue_number")},
    )
    if doc_id:
        linker = getattr(ctx.memory, "set_finding_vector_doc", None)
        if linker is not None:
            try:
                await asyncio.to_thread(linker, fid, doc_id)
            except Exception:  # noqa: BLE001 — linking is best-effort
                pass
    await asyncio.to_thread(
        ctx.memory.post_message,
        ctx.agent_id, "broadcast",
        f"New finding #{fid} on '{topic}' with {len(citations)} citations (source: {source}).",
        state.get("issue_number"),
    )
    if ctx.role == Role.RESEARCHER:
        # P16bis (H2): the finding's claims must be VERIFIED by running code,
        # not trusted because an LLM wrote them — the cycle continues into
        # node_verify, which pre-registers hypotheses and adjudicates.
        state["finding_id"] = fid
        state["finding_text"] = content[:4000]
        state["action"] = "verify"
    else:
        state["action"] = "plan"
    return state


async def _free_tools_fallback(ctx: AgentContext, state: ResearchState,
                               topic: str, reason: str) -> ResearchState:
    """LLM unavailable -> real-metadata research via the free keyless APIs.

    arXiv/OpenAlex/Crossref ground the finding in verifiable metadata so a
    dead cookie pool no longer wastes the whole cycle (docs/FREE_TOOLS.md).
    When the free tools are disabled (or every source comes up empty) the
    original degrade-to-error behaviour is preserved exactly."""
    if not getattr(ctx.settings, "free_tools_enabled", True):
        state["error"] = reason or "research unavailable"
        state["done"] = True
        return state
    try:
        fb = await free_tools.fallback_research(topic, settings=ctx.settings)
    except Exception as exc:  # noqa: BLE001 — a source outage must not crash the cycle
        log.warning("[%s] free-tools fallback failed: %s", ctx.agent_id, exc)
        fb = None
    if fb is None or not fb.ok:
        state["error"] = reason or "research unavailable (free-tools fallback empty)"
        state["done"] = True
        return state
    log.info("[%s] research continued via free tools (%s)", ctx.agent_id, reason)
    acc = getattr(ctx.llm, "account", None)
    ctx.meta["account"] = getattr(acc, "name", "none") if acc else "none"
    ctx.meta["research_source"] = "free-tools"
    state["bibtex"] = fb.bibtex
    return await _finish_research(ctx, state, topic, fb.answer, fb.citations, "free-tools")


async def node_research(ctx: AgentContext, state: ResearchState) -> ResearchState:
    """Researcher: search papers, dedup vs vector memory, store findings.

    A rework cycle skips research entirely: the topic is already covered (the
    dedup guard would otherwise abort the cycle) and the review feedback, not
    new literature, drives the fix.
    """
    if state.get("rework_mr"):
        state["action"] = "plan"
        return state
    topic = state.get("issue_title") or ctx.settings.research_topic
    # H3: the research LLM must know what the swarm already PROVED by running
    # code — supported facts are common ground, refuted claims are dead ends.
    knowledge = verified_knowledge_text(ctx)
    # The query carries the issue's own problem statement (bounded excerpt),
    # not just the title + a fixed recipe — the body is where the concrete
    # constraints live. Dedup still keys on the title-based topic, so this
    # cannot weaken the P7 skip guard.
    body = (state.get("issue_body") or "").strip()
    query = (
        f"{topic} — context: {ctx.settings.research_topic}. "
        "Focus on methods, benchmarks, reproducible results."
        + (f"\n\nProblem statement from the issue:\n{body[:400]}" if body else "")
        + (f"\n\n{knowledge}" if knowledge else "")
    )
    # P7: dedup — skip if the swarm already covered this. When the vector
    # store is offline (chromadb absent / init failed) the SAME gate falls
    # back to keyword search over the findings table — the SQLite fallback
    # the policies layer already uses — so dedup survives without embeddings
    # instead of re-researching every topic blind.
    known = await papers_mod.dedup_query(ctx.vector, topic)
    if not getattr(ctx.vector, "available", False):
        fallback = getattr(ctx.memory, "finding_search", None)
        if fallback is not None:
            try:
                known = [m for m in fallback(topic, limit=5) if m.get("hits", 0) >= 2]
            except Exception:  # noqa: BLE001 — dedup fallback must never break research
                known = []
    if len(known) >= 3:
        state["summary"] = f"topic already covered by {len(known)} findings; skipping"
        state["done"] = True
        return state

    if ctx.llm is None:
        return await _free_tools_fallback(ctx, state, topic,
                                          reason="no LLM account available")
    # P20 — research calls bypass _llm_ask (provider routing differs), so the
    # budget gate and the usage ledger are applied here as well. A BLOCKED ask
    # still falls back to the free tools: the owner keeps getting research
    # without burning a single budgeted call.
    budget = getattr(ctx, "budget", None)
    if budget is not None:
        allowed, why = budget.can_ask(ctx.agent_id)
        if not allowed:
            log.warning("[%s] research blocked by budget: %s", ctx.agent_id, why)
            try:
                ctx.memory.journal(ctx.agent_id, "budget_blocked",
                                   {"purpose": "research", "why": why[:200]})
            except Exception:  # noqa: BLE001 — telemetry must never gate harder
                pass
            return await _free_tools_fallback(ctx, state, topic,
                                              reason=f"P19 budget: {why}")
    import time as _time
    _t0 = _time.monotonic()
    resp = None
    llm_error = ""
    try:
        resp = await papers_mod.search_papers(ctx.llm, query, purpose="research")
    except Exception as exc:  # noqa: BLE001 — degrade, don't crash the cycle
        log.warning("[%s] research llm call failed: %s", ctx.agent_id, exc)
        if budget is not None:
            budget.record(ctx.agent_id, "research", False, _time.monotonic() - _t0)
        llm_error = f"research llm call failed: {exc}"
    if budget is not None and resp is not None and resp.ok:
        budget.record(ctx.agent_id, "research", True, _time.monotonic() - _t0)
    if resp is None or not resp.ok:
        return await _free_tools_fallback(
            ctx, state, topic,
            reason=llm_error or "research llm returned an empty answer",
        )
    # P18b fan-out: extra complementary angles asked in PARALLEL on other
    # pool identities (different Hyperbrowser tokens / cookie accounts).
    # Each angle is budget-gated; the merged answer feeds the same finding
    # pipeline (deduped citations -> one batched BibTeX grounding call).
    fanout = max(1, int(getattr(ctx.settings, "research_fanout", 1) or 1))
    if fanout > 1:
        extras = await _fanout_angles(ctx, query, fanout - 1, budget)
        if extras:
            log.info("[%s] research fan-out merged %d extra angle(s)",
                     ctx.agent_id, len(extras))
            resp = _merge_research(resp, extras)
    acc = getattr(ctx.llm, "account", None)
    ctx.meta["account"] = getattr(acc, "name", "none") if acc else "none"
    ctx.meta["research_source"] = "llm"
    state["bibtex"] = await _grounded_bibtex(ctx, resp.citations)
    return await _finish_research(ctx, state, topic, resp.answer, resp.citations, "llm")


# ---------------------------------------------------------------- verify (P16bis)
VERIFY_STATUS = {
    "confirmed": "supported",
    "refuted": "refuted",
    "inconclusive": "inconclusive",
}


# P24 node wiring — the engine is verify/replication.py (P24bis): it writes
# BOTH ledgers (record_replication bookkeeping + record_replication_detail
# provenance), refuses self-replication, and closes a refuted replication
# the P17 way. The wiring only feeds the P24 selector and reports.


async def _replicate_settled(ctx: AgentContext, limit: int = 2) -> list[dict]:
    """P24 — independent replication of the swarm's settled claims (zero LLM).

    One agent's verdict is not knowledge; an INDEPENDENT re-run by another
    agent is. Feeds the P24 selector (oldest settled 'supported' claims this
    agent did NOT author, once-per-swarm bookkeeping, 2-attempt retry
    budget) into the P24bis engine (`verify.replication.replicate_pending`),
    which re-runs the stored protocol in THIS agent's own workspace through
    the full two-run reproducibility gate. Never raises — every failure mode
    is a journaled, queryable outcome."""
    if not getattr(ctx.workspace, "sync_main", None):
        return []
    from agentics.verify.replication import replicate_pending
    reports = await asyncio.to_thread(
        replicate_pending, ctx.memory, ctx.workspace, ctx.agent_id, limit)
    results: list[dict] = []
    for report in reports:
        verdict = report.get("verdict")
        hyp_id = report.get("hypothesis_id")
        if verdict is None and hyp_id:
            # Engine decline (write refused / runner crash / unshareable):
            # the engine journals the blocked event itself but leaves the
            # LEDGER untouched — mark the attempt inconclusive here so the
            # P24 retry budget advances and a dead claim stops looping.
            # Idempotence declines ("already replicated") keep a non-empty
            # ledger column and are skipped.
            row = await asyncio.to_thread(ctx.memory.get_hypothesis, hyp_id)
            if row is not None and not (row.get("replication") or ""):
                await asyncio.to_thread(
                    ctx.memory.record_replication, hyp_id, "inconclusive")
        claim = ""
        if hyp_id:
            hyp = await asyncio.to_thread(ctx.memory.get_hypothesis, hyp_id)
            claim = (hyp or {}).get("claim", "")
        results.append({
            "claim": claim, "status": verdict or "inconclusive",
            "note": (report.get("note") or "")[:200],
            "replication": bool(report.get("replicated"))
                           or verdict == "refuted",
            "hypothesis_id": hyp_id,
            "evidence_id": report.get("peer_evidence")
                           or report.get("evidence_id"),
        })
    return results


def _replication_summary(rep: list[dict]) -> str:
    """One digest line for the P24 replication block of a verify cycle."""
    counts: dict[str, int] = {}
    for r in rep:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    breakdown = ", ".join(f"{s}:{n}" for s, n in sorted(counts.items()))
    return f"P24 replicated {len(rep)} claim(s) [{breakdown}]"


async def node_verify(ctx: AgentContext, state: ResearchState) -> ResearchState:
    """Researcher (P16bis, H2): verify the finding's claims by RUNNING code.

    Claims are pre-registered as hypotheses with a quantitative criterion
    (metric/op/target); one self-contained protocol per claim is written into
    experiments/ and executed twice through the guarded workspace runner;
    the verdict is a machine comparison (measured vs criterion) — never an
    LLM's opinion — and lands in shared evidence + verified knowledge (H3).

    P24: BEFORE any of that, this agent may serve as an independent replica —
    re-running someone else's settled claim from its stored protocol (zero
    LLM, needs no fresh finding). A researcher cycle with nothing new to
    verify still advances the swarm's trust in what it already knows."""
    finding_text = state.get("finding_text") or ""
    # P24 independent replication first — needs neither finding nor LLM.
    try:
        repl = await _replicate_settled(ctx)
    except Exception:  # noqa: BLE001 — replication must never break verification
        repl = []
    if not finding_text or ctx.llm is None:
        state["verifications"] = repl
        state["summary"] = (_replication_summary(repl) if repl else
                            "verification skipped (no finding or no LLM)")
        state["done"] = True
        return state
    ctx.report(
        node="verify", task_id=state.get("issue_number"),
        intent={
            "goal": (f"Verify the claims of finding #{state.get('finding_id')} "
                     "by running protocols"),
            "why": "P16bis: verdicts from measurements, not model opinion",
            "steps": ["extract claims", "register criteria", "run protocols",
                      "record evidence"],
            "expected_files": [],
        },
    )
    prompt = (
        f"Research finding to verify:\n{finding_text[:3000]}\n\n"
        "Extract up to 2 QUANTITATIVE claims from this finding that a small, "
        "self-contained python simulation can check (simulate the method on "
        "synthetic data, measure the metric). For each claim write its "
        "protocol.\n\n"
        f"{PROTOCOL_CONTRACT}\n"
        'Reply ONLY JSON: {"hypotheses": [{"claim": "...", "metric": "<name>", '
        '"op": ">=", "target": 0.9, "protocol": "<full python code>"}]}'
    )
    answer, _ = await _llm_ask(ctx, prompt, mode="concise", purpose="verify")
    parsed = parse_json_block(answer)
    specs = [h for h in (parsed.get("hypotheses") or []) if isinstance(h, dict)][:3]
    if not specs:
        state["summary"] = "no verifiable claim extracted; nothing verified this cycle"
        state["done"] = True
        return state

    await asyncio.to_thread(ctx.workspace.sync_main)    # workspace must exist
    results: list[dict] = []
    for spec in specs:
        claim = str(spec.get("claim", "")).strip()[:300]
        criterion, malformed = None, ""
        try:
            criterion = Criterion.from_dict(spec)
        except ValueError as exc:
            malformed = str(exc)
        protocol_code = str(spec.get("protocol", ""))
        if not claim or criterion is None or not protocol_code.strip():
            results.append({"claim": claim, "status": "inconclusive",
                            "note": malformed or "malformed hypothesis spec"})
            continue
        # H3 dedup — the swarm never re-runs a SETTLED claim: whichever agent
        # meets the same hypothesis later inherits the first verdict instead of
        # burning two protocol runs on it (verified_knowledge keeps the claim
        # reusable; this gate keeps the ledger free of duplicate evidence).
        prior = await asyncio.to_thread(
            ctx.memory.settled_hypothesis_for_claim, claim)
        if prior is not None:
            await asyncio.to_thread(
                ctx.memory.journal, ctx.agent_id, "hypothesis_reused",
                {"claim": claim[:160], "settled_by": prior["id"],
                 "status": prior["status"], "evidence": prior["evidence_id"],
                 "why": "claim already settled — reused, not re-run"})
            results.append({"claim": claim, "status": prior["status"],
                            "note": (f"reused settled verdict of h#{prior['id']} "
                                     f"(by {prior['agent']}): "
                                     f"{prior['verdict_note'][:120]}"),
                            "reused": True, "hypothesis_id": prior["id"],
                            "evidence_id": prior["evidence_id"]})
            continue
        hyp_id = await asyncio.to_thread(
            ctx.memory.add_hypothesis, ctx.agent_id, claim, criterion.as_dict(),
            task_id=state.get("issue_number"), finding_id=state.get("finding_id"),
            status="testing", protocol_code=protocol_code[:20000])
        path = f"experiments/verify_h{hyp_id}.py"
        try:
            await asyncio.to_thread(ctx.workspace.write_file, path, protocol_code)
        except ValueError as exc:            # P10 layer 1: protected / traversal
            await asyncio.to_thread(ctx.memory.update_hypothesis, hyp_id,
                                    "inconclusive", note=str(exc)[:300])
            results.append({"claim": claim, "status": "inconclusive",
                            "note": str(exc)[:200]})
            continue
        try:
            ev = await asyncio.to_thread(run_protocol, ctx.workspace, path, criterion)
        except Exception as exc:  # noqa: BLE001 — one bad protocol must not kill the cycle
            await asyncio.to_thread(ctx.memory.update_hypothesis, hyp_id,
                                    "inconclusive", note=f"runner error: {exc}"[:300])
            results.append({"claim": claim, "status": "inconclusive",
                            "note": f"runner error: {exc}"[:200]})
            continue
        ev_id = await asyncio.to_thread(
            ctx.memory.record_evidence, ctx.agent_id, state.get("issue_number"),
            ev.cmd, ev.exit_code, ev.verdict,
            hypothesis=claim, method=ev.method, output=ev.output,
            duration_s=ev.duration_s, artifact=path)
        await asyncio.to_thread(
            ctx.memory.attach_protocol_outcome, ev_id,
            ev.meta.get("metrics", {}), ev.meta.get("runs", []), hyp_id)
        status = VERIFY_STATUS[ev.verdict]
        note = str(ev.meta.get("note", ""))
        await asyncio.to_thread(
            ctx.memory.update_hypothesis, hyp_id, status,
            evidence_id=ev_id, note=note, protocol_path=path)
        await asyncio.to_thread(
            ctx.memory.journal, ctx.agent_id, "hypothesis_verified",
            {"hyp": hyp_id, "verdict": status, "claim": claim[:160],
             "note": note[:200], "evidence": ev_id,
             "criterion": criterion.as_dict()},
        )
        results.append({"claim": claim, "status": status, "note": note,
                        "hypothesis_id": hyp_id, "evidence_id": ev_id,
                        "replication": False})
    # P24: the independent-replication block rides along on this cycle too
    if repl:
        results = repl + results
        state["summary"] = (
            f"P16bis verified {len(results) - len(repl)} claim(s); "
            + _replication_summary(repl))
    else:
        counts: dict[str, int] = {}
        for r in results:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        breakdown = ", ".join(f"{s}:{n}" for s, n in sorted(counts.items()))
        state["summary"] = f"P16bis verified {len(results)} claim(s) [{breakdown}]"
    state["verifications"] = results
    state["done"] = True
    return state


def pending_replication(ctx: AgentContext, limit: int = 1) -> list[dict]:
    """Settled claims the P24 selector offers ME for independent replication.

    Routing helper — must never crash a cycle (same contract as
    pending_defense): a read failure simply means "nothing to replicate"."""
    try:
        limit = int(limit or 0)
        if limit <= 0:
            return []
        return ctx.memory.hypotheses_needing_replication(ctx.agent_id,
                                                         limit=limit)
    except Exception:  # noqa: BLE001 — routing must never crash on a read
        return []


async def node_replicate(ctx: AgentContext, state: ResearchState) -> ResearchState:
    """Reviewer (P24bis): replicate a settled claim INDEPENDENTLY.

    The swarm's common knowledge used to rest on ONE agent's protocol run —
    the dedup gate made every other agent REUSE it, so a buggy or gaming
    protocol poisoned verified_knowledge forever. This node gives the swarm
    the scientific fix: a peer rebuilds the shareable protocol in its own
    workspace (no LLM involved), runs the two-run reproducibility gate, and
    the machine verdict is written to BOTH ledgers (P24 bookkeeping +
    provenance detail) and broadcast (H3). A refuted replication closes the
    ledger the P17 way: evidence re-verdicted, hypothesis downgraded; an
    inconclusive one never refutes (P17 asymmetry) and the P24 retry budget
    keeps the claim available for another peer."""
    limit = int(getattr(ctx.settings, "replication_limit", 1) or 0)
    if limit <= 0:
        state["summary"] = "replication disabled (REPLICATION_LIMIT=0)"
        state["done"] = True
        return state
    candidates = pending_replication(ctx, limit=limit)
    if not candidates:
        state["summary"] = "no claim awaiting independent replication"
        state["done"] = True
        return state
    ctx.report(
        node="replicate",
        intent={
            "goal": (f"Independently replicate {len(candidates)} settled "
                     "claim(s) from their stored protocols"),
            "why": ("P24bis: a supported claim must survive a DIFFERENT "
                    "agent's re-run, not just its author's"),
            "steps": ["pick oldest unreplicated claim", "rebuild protocol in "
                      "my workspace", "run the 2-run gate", "record both "
                      "ledgers + broadcast"],
            "expected_files": [],
        },
    )
    from agentics.verify.replication import replicate_pending
    # the peer's workspace must be a real clone BEFORE the engine writes the
    # protocol into it — write_file creates directories eagerly, and a
    # later sync_main would then refuse to clone into the non-empty dir
    # (the local_mode_demo caught exactly this ordering bug).
    await asyncio.to_thread(ctx.workspace.sync_main)
    reports = await asyncio.to_thread(
        replicate_pending, ctx.memory, ctx.workspace, ctx.agent_id,
        limit=limit)
    counts: dict[str, int] = {}
    for r in reports:
        v = r.get("verdict") or ("skipped" if not r.get("replicated") else "?")
        counts[v] = counts.get(v, 0) + 1
    breakdown = ", ".join(f"{s}:{n}" for s, n in sorted(counts.items()))
    state["replication_verdict"] = breakdown
    state["summary"] = (f"P24bis replicated {len(reports)} claim(s) "
                        f"[{breakdown}]")
    state["done"] = True
    return state


async def node_plan(ctx: AgentContext, state: ResearchState) -> ResearchState:
    """Coder: turn research into an implementation plan (LLM, JSON)."""
    category = state.get("category") or categorize_issue(
        state.get("issue_title", ""), state.get("issue_body", ""))
    state["category"] = category
    traces_txt = trace_context(ctx.memory, ctx.agent_id, state.get("issue_number"), category)
    skills_txt = lifemem.skill_context(
        ctx.memory, f"{state.get('issue_title', '')} {state.get('issue_body', '')[:200]}",
        category=category)
    tools_txt = tool_context(
        ctx.memory, f"{state.get('issue_title', '')} {state.get('issue_body', '')[:200]}")
    # v2 memory loops: the planner sees its OWN relevant lessons (personal
    # memory) and the swarm's RELATED stored findings (vector recall) — both
    # bounded, both defensive, both empty at cold start. v3: lessons recorded
    # on this very issue rank first (the retry/rework finds its own history).
    _query = f"{state.get('issue_title', '')} {state.get('issue_body', '')[:200]}"
    lessons_txt = _lessons_text(ctx, _query, issue=state.get("issue_number"))
    recall_txt = finding_recall_text(ctx, _query)
    personal_rule = (
        f"\nYour own lessons from past cycles (personal memory — never repeat "
        f"these failures; (xN) marks how often each recurred):\n{lessons_txt}\n"
        if lessons_txt else ""
    )
    recall_rule = (
        f"\nFindings this lab already stored about similar topics (relevance "
        f"recall — reuse this knowledge, cite finding ids):\n{recall_txt}\n"
        if recall_txt else ""
    )
    # H3: plans must stand on what the lab has PROVEN, not on fresh guesses —
    # and the declared verify_cmds should re-check exactly the new claims.
    knowledge_txt = verified_knowledge_text(ctx)
    knowledge_rule = (f"\nVerified knowledge of this lab (P16, exit-code verdicts — "
                      f"your plan must be consistent with these and your verify_cmds "
                      f"must check the NEW claims):\n{knowledge_txt}\n"
                      if knowledge_txt else "")
    evolution_rule = (
        "\nThis is a SELF-EVOLUTION issue: you may ONLY change these paths: "
        + ", ".join(EVOLUTION_ALLOWED_PATHS) + ".\n"
        if category == "evolution" else ""
    )
    rework_rule = (
        f"\nThis is a REWORK of MR #{state.get('rework_mr')}: the branch already "
        "carries the reviewed implementation — plan ONLY the changes that answer "
        "the review feedback below; do not rewrite unrelated files.\n"
        if state.get("rework_mr") else ""
    )
    prompt = (
        f"Issue #{state.get('issue_number')}: {state.get('issue_title')}\n"
        f"{state.get('issue_body')}\n\nResearch summary:\n"
        f"{state.get('papers', [{}])[0].get('answer', '')[:2500]}\n\n"
        f"Traces from other agents (learn from them, do not repeat their failures):\n"
        f"{traces_txt}\n\n"
        f"Distilled skills from past workflows that led to merges (LifeMem tier — "
        f"follow what worked, avoid what is listed):\n{skills_txt}\n\n"
        f"Reusable tools already registered in this repo (reuse before writing new code):\n"
        f"{tools_txt}\n{personal_rule}{recall_rule}{evolution_rule}{rework_rule}{knowledge_rule}\n"
        f"{PLAN_INSTRUCTION}"
    )
    answer, _ = await _llm_ask(ctx, prompt, mode="concise", purpose="plan", state=state)
    parsed = parse_json_block(answer)
    if not parsed and not state.get("error"):
        # One strict retry: the live model occasionally answers in prose or
        # wraps the object in commentary (a real live run lost its plan this
        # way). Provider-level failures (state error set) do NOT retry — a
        # cooling-down cookie needs time, not another immediate ask.
        retry_answer, _ = await _llm_ask(
            ctx,
            prompt + "\n\nIMPORTANT: your previous reply was not one JSON "
            "object. Reply AGAIN with ONLY the JSON object — no prose, no "
            "markdown fence, nothing else.",
            mode="concise", purpose="plan", state=state)
        parsed = parse_json_block(retry_answer)
        if parsed:
            try:
                ctx.memory.journal(ctx.agent_id, "plan_retry_recovered",
                                   {"task": state.get("issue_number")})
            except Exception:  # noqa: BLE001 — telemetry must never gate
                pass
    if not parsed:
        state["error"] = state.get("error") or "plan parsing failed"
        state["done"] = True
        return state
    # The model proposed the plan (approach-first, extras of its own naming);
    # normalize_plan extracts just the machine contract the lab consumes.
    core = normalize_plan(parsed)
    state["plan"] = plan_state_json(core)
    state["commit_message"] = core.commit_message
    state["verify_cmds"] = core.verify_cmds
    state["hypotheses"] = core.hypotheses
    state["reuses"] = core.reuses
    state["expected_files"] = core.expected_files
    state["plan_approach"] = core.approach
    if core.extras:
        state["plan_extras"] = extras_state_json(core.extras)
    for name in core.reuses:
        ctx.memory.bump_tool_use(name)      # no-op if the name is not registered
    ctx.report(
        node="plan",
        intent={
            "goal": f"Implement task #{state.get('issue_number')}: {state.get('issue_title', '')}",
            "approach": core.approach[:200],
            "steps": core.steps[:8],
            "expected_files": core.expected_files,
            "verify_cmds": core.verify_cmds,
            "hypotheses": [str(h.get("claim", ""))[:120] for h in core.hypotheses],
            "extras": sorted(core.extras)[:6],
            "why": f"category={state.get('category', 'other')}",
        },
    )
    # P13 — lock the declared scope across the swarm: another agent already
    # planning the same files is stopped BEFORE burning LLM cycles on implement.
    locks = getattr(getattr(ctx, "local", None), "locks", None)
    if locks is not None and core.expected_files:
        ok, conflicts = await asyncio.to_thread(
            locks.try_acquire_all, ctx.agent_id, core.expected_files)
        if not ok:
            ctx.memory.journal(ctx.agent_id, "lock_conflict", {
                "task": state.get("issue_number"),
                "files": core.expected_files,
                "held_by": conflicts,
            })
            return await _yield(
                ctx, state,
                f"P13 files already locked by other agents: "
                f"{list(conflicts.items())[:3]} — pick another angle",
            )
        ctx.memory.journal(ctx.agent_id, "intent_bound", {
            "task": state.get("issue_number"), "files": core.expected_files,
        })
    return state


async def _yield(ctx: AgentContext, state: ResearchState, reason: str) -> ResearchState:
    # node_reflect owns the single terminal cleanup path for non-PR cycles.
    await asyncio.to_thread(ctx.workspace.sync_main)  # discard local changes
    state["yield_reason"] = reason
    state["done"] = True
    ctx.memory.add_decision(ctx.agent_id, "yield", f"issue #{state['issue_number']}: {reason}")
    return state


def _tier1(ctx: AgentContext, changed: list[str]) -> tuple[bool, str]:
    """Fast sanity gate: py_compile on changed files, then pytest -x. Docs-only
    changes skip it. Returns (ok, one-line summary used in the PR body)."""
    py_files = [f for f in changed if f.endswith(".py") and (ctx.workspace.path / f).exists()]
    if not py_files:
        return True, "tier-1: no Python changes, skipped"
    code, out = ctx.workspace.run_command(
        [sys.executable, "-m", "py_compile", *py_files], timeout=60)
    if code != 0:
        return False, f"tier-1 compile failed: {out[-400:]}"
    code, out = ctx.workspace.run_command([sys.executable, "-m", "pytest", "-x", "-q"], timeout=300)
    if code not in (0, 5):             # 5 = no tests collected
        return False, f"tier-1 tests failed (exit {code}): {out[-400:]}"
    return True, f"tier-1 passed: compiled {len(py_files)} file(s), pytest exit {code}"


# ---------------------------------------------------------------- P16 wiring
def normalize_verify_cmds(cmds: list) -> list[list[str]]:
    """Turn the plan's declared hypothesis checks into safe argv lists.

    The plan LLM writes commands like ``"python -m pytest tests/x.py -q"``;
    run_command only ever executes the ACTIVE interpreter, so ``python`` /
    ``python3`` / ``<sys.executable>`` are all rewritten to ``sys.executable``
    and anything else is dropped (a blocked command would only burn the run).
    At most 5 checks, 12 argv words each."""
    out: list[list[str]] = []
    for raw in (cmds or [])[:8]:
        try:
            argv = shlex.split(raw) if isinstance(raw, str) else [str(a) for a in raw]
        except ValueError:
            continue
        if not argv:
            continue
        exe = argv[0]
        if exe in ("python", "python3") or Path(exe).resolve() == Path(sys.executable):
            argv[0] = sys.executable
        elif exe == sys.executable:
            pass
        else:
            continue                  # only the active interpreter is runnable
        if len(argv) > 12:
            argv = argv[:12]
        if argv not in out:
            out.append(argv)
    return out[:5]


def _implement_brief(state: ResearchState, plan: dict) -> str:
    """The planner's OWN words, carried into every per-file implement ask.

    The per-file prompt used to be a fixed minimal template: the implementer
    wrote code without knowing why the plan exists or what will judge it.
    Now the prompt is conditioned on the model's own proposal — the declared
    approach, the P16 checks that will run against the code and the
    pre-registered hypotheses — everything the planner itself wrote, bounded
    by defensive caps. An empty brief (a minimal/legacy plan) reproduces the
    old prompt byte for byte."""
    parts: list[str] = []
    approach = str(state.get("plan_approach")
                   or plan.get("approach") or "").strip()
    if approach:
        parts.append(
            "Your declared approach (from your own plan — follow it):\n"
            + approach[:700])
    cmds = [str(c) for c in (plan.get("verify_cmds") or [])
            if str(c).strip()][:5]
    if cmds:
        parts.append(
            "These checks WILL run against this code (P16 — the verdict "
            "comes from their exit codes, so make them pass by doing the "
            "real work, never by faking output):\n"
            + "\n".join(f"- {c[:160]}" for c in cmds))
    hyps = [h for h in (plan.get("hypotheses") or [])
            if isinstance(h, dict)][:2]
    if hyps:
        lines = []
        for h in hyps:
            claim = str(h.get("claim", ""))[:200]
            pred = str(h.get("prediction", ""))[:200]
            lines.append(f"- claim: {claim}"
                         + (f" | predicted: {pred}" if pred else ""))
        parts.append(
            "Hypotheses the lab will measure on this code (P16 protocol — "
            "pre-registered criteria, run twice and re-adjudicated):\n"
            + "\n".join(lines))
    return ("\n\n".join(parts) + "\n\n") if parts else ""


async def node_implement(ctx: AgentContext, state: ResearchState) -> ResearchState:
    """Coder: write code from plan into a local branch, run tests, push (P2/P3/P10)."""
    branch = state["branch"]
    await asyncio.to_thread(ctx.workspace.sync_main)
    if state.get("rework_mr"):
        # resume the pushed branch — the reviewed work must survive the rework
        await asyncio.to_thread(ctx.workspace.checkout_branch_from_origin, branch)
    else:
        await asyncio.to_thread(ctx.workspace.checkout_new_branch, branch)

    try:
        plan = json.loads(state.get("plan") or "{}")
    except json.JSONDecodeError:
        state["error"] = "plan state is not valid JSON (corrupted dump)"
        state["done"] = True
        return state
    # The implement prompt is conditioned on the model's OWN proposal, not a
    # fixed template: every per-file ask carries the approach the planner
    # declared and the P16 checks/hypotheses that will judge the code.
    brief = _implement_brief(state, plan)
    for spec in plan.get("files", []):
        if state.get("error"):           # LLM died mid-plan: push nothing
            state["done"] = True
            return state
        path, change = spec.get("path", ""), spec.get("change", "new")
        # `or ""`: read_file returns None for a hallucinated path (the plan
        # says "edit" but the file does not exist in the workspace) — treat
        # it as a new file instead of crashing the whole cycle.
        current = (
            await asyncio.to_thread(ctx.workspace.read_file, path)
            if change == "edit" else ""
        ) or ""
        prompt = (
            f"Task #{state.get('issue_number')}: {state.get('issue_title')}\nPlan step for this file:\n"
            f"{json.dumps(spec, ensure_ascii=False)}\n\n"
            f"{brief}"
            f"Current content:\n{current[:6000]}\n\n"
            "Return the FULL new file content only, no explanations."
        )
        answer, _ = await _llm_ask(ctx, prompt, mode="concise", purpose="implement",
                                   state=state)
        content = answer.strip().removeprefix("```").removesuffix("```")
        if content.strip():
            try:
                # owner-exempt task ('agentics allow-protected'): the write
                # gate opens for the task's agent-code files — credentials
                # (.env, STOP, accounts/, configs/secrets.yaml) stay blocked.
                # The diff gate (layer 2) re-verifies the whole set after.
                allow = (bool(state.get("task_exempt"))
                         and not is_credential_path(path))
                await asyncio.to_thread(ctx.workspace.write_file, path, content, allow)
            except ValueError as exc:      # P10 layer 1: protected path / traversal
                return await _yield(ctx, state, str(exc))

    ctx.report(node="implement", status="verifying",
               intent={"goal": f"Verify and push branch {branch}", "steps": []})

    # NOTE: must be the working-tree status — `diff_files_vs_main()` compares
    # commits and is empty before the commit, which silently disabled P3.
    changed = await asyncio.to_thread(ctx.workspace.changed_files)
    state["changed_files"] = changed

    # P10 layer 2: a diff that touches the lab's own code is refused here —
    # an exemption comes from the task row (owner-granted), never from the agent.
    guard = getattr(getattr(ctx, "local", None), "guard", None)
    allow_protected = False
    if guard is not None:
        violations = guard.check_task(
            ctx.memory, state.get("issue_number"), changed)
        if violations:
            return await _yield(
                ctx, state,
                f"P10 protected paths in diff: {violations[:5]} "
                "(agent code is owner-only; the owner can exempt a task)",
            )
        # owner-granted exemption on this task: the push must carry the same
        # sanction so the P10 origin hook (layer 4) lets the branch through
        task_id = state.get("issue_number")
        task = ctx.memory.get_task(task_id) if task_id is not None else None
        allow_protected = bool(task and task.get("exempt"))

    # P14 — the diff must stay inside the declared intent scope. An undeclared
    # file in the diff means the plan drifted; refuse and journal it.
    expected = state.get("expected_files") or []
    if expected:
        drift = scope_violations(changed, expected)
        if drift:
            ctx.memory.journal(ctx.agent_id, "scope_violation", {
                "task": state.get("issue_number"),
                "undeclared": drift[:10],
                "declared": expected,
            })
            return await _yield(
                ctx, state,
                f"P14 files outside the declared intent: {drift[:5]} "
                "(re-plan or declare them)",
            )

    # Guard: self-evolution PRs may only touch the allow-listed paths.
    if state.get("category") == "evolution":
        out_of_bounds = [f for f in changed if not evolution_path_allowed(f)]
        if out_of_bounds:
            return await _yield(
                ctx, state, f"evolution PR outside allowed paths {out_of_bounds[:5]}"
            )

    # P3: ownership check against other active roles
    active_agents = await asyncio.to_thread(ctx.memory.active_agents, stale_after_s=300)
    active_roles = {
        a["role"] for a in active_agents if a["agent"] != ctx.agent_id
    }
    ok, foreign = await asyncio.to_thread(
        ctx.ownership.evaluate, ctx.role.value, changed, active_roles
    )
    if not ok:
        return await _yield(ctx, state, f"foreign files {foreign[:5]}")

    # Tier 1 (author, before push): compile + fast tests. Tier 2 (full suite on
    # main+PR) runs later, in the Integrator, right before merge.
    t1_ok, verification = await asyncio.to_thread(_tier1, ctx, changed)
    state["verification"] = verification
    if not t1_ok:
        state["error"] = verification
        state["done"] = True
        return state

    # P16 — empirical evidence: run the plan's declared verification commands
    # and record their REAL outcomes as shared evidence (owner hypotheses 1-3).
    # The verdict comes from exit codes, never from an LLM's self-assessment.
    # `refuted` blocks the push (the hypothesis did not hold under test);
    # `inconclusive` (blocked/timeout) journals a warning and proceeds — the
    # reviewer sees the evidence and decides, the pipeline never livelocks.
    # normalize_verify_cmds rewrites "python ..." to the ACTIVE interpreter
    # and drops anything run_command would refuse: a raw "/bin/sh -c" wrap
    # would be rejected by the guard (exit 126) and every check would sit at
    # "inconclusive" forever — the plan's strings must become real argv first.
    verify_cmds = normalize_verify_cmds(state.get("verify_cmds") or [])
    if verify_cmds:
        evs = await asyncio.to_thread(
            run_experiments, ctx.workspace, verify_cmds, timeout=300)
        task_id = state.get("issue_number")
        for ev in evs:
            await asyncio.to_thread(
                ctx.memory.record_evidence, ctx.agent_id, task_id,
                ev.cmd, ev.exit_code, ev.verdict,
                hypothesis=state.get("issue_title", ""),
                method=ev.method, output=ev.output,
                duration_s=ev.duration_s)
        digest = summarize(evs)
        state["verification"] = f"{verification} | {digest}"[:500]
        refuted = [e for e in evs if e.verdict == "refuted"]
        if refuted:
            return await _yield(
                ctx, state,
                f"P16 hypothesis refuted by experiment: {digest} "
                f"(exit codes {[e.exit_code for e in refuted][:5]}) "
                "— fix the code or re-plan",
            )
        inconclusive = [e for e in evs if e.verdict == "inconclusive"]
        if inconclusive:
            ctx.memory.journal(ctx.agent_id, "p16_inconclusive", {
                "task": task_id,
                "cmds": [e.cmd[:120] for e in inconclusive][:5]})

    # P16 (hypothesis contract) — the richer experiment form: the plan may
    # declare MEASURABLE hypotheses (metric + threshold + protocol code).
    # The register -> run -> settle composition (agentics/verify.experiment)
    # pre-registers the claim with a frozen criterion, runs the protocol
    # through verify_metrics.run_protocol (2 runs, reproducibility gate), and
    # the verdict is re-adjudicated from the MEASUREMENT — a lying script
    # cannot claim success. Every verdict lands in the shared evidence table
    # (H3) the reviewer, dashboard and CLI read.
    hyp_raws = state.get("hypotheses") or []
    if hyp_raws:
        from agentics.verify.experiment import (
            register_hypothesis, run_hypothesis, settle_hypothesis,
        )
        from agentics.verify.hypothesis import parse_hypothesis

        hyp_task = state.get("issue_number")
        for raw in hyp_raws:
            hyp, reason = parse_hypothesis(raw)
            if hyp is None:                     # malformed proposal: never guessed through
                ctx.memory.journal(ctx.agent_id, "p16_hypothesis_invalid",
                                   {"task": hyp_task, "reason": reason[:200]})
                continue
            # a plan-declared hypothesis belongs to THIS task — the proposal
            # may omit the binding, the cycle supplies it (evidence rows must
            # be findable per task by the reviewer / dashboard / CLI)
            if hyp.task is None:
                hyp.task = hyp_task
            # H3 dedup: a claim the swarm already settled is inherited, not
            # re-run. A supported claim must not cost two more protocol runs;
            # a refuted claim must not sneak past the push gate by rerunning it.
            prior = await asyncio.to_thread(
                ctx.memory.settled_hypothesis_for_claim, hyp.claim)
            if prior is not None:
                ctx.memory.journal(ctx.agent_id, "hypothesis_reused", {
                    "task": hyp_task, "claim": hyp.claim[:160],
                    "settled_by": prior["id"], "status": prior["status"],
                    "evidence": prior["evidence_id"],
                })
                if prior["status"] == "refuted":
                    return await _yield(
                        ctx, state,
                        f"P16 hypothesis '{hyp.hypothesis_id}' is ALREADY REFUTED "
                        f"by the swarm (h#{prior['id']}, evidence "
                        f"#{prior['evidence_id']}) — do not re-assert a refuted "
                        "claim; fix the code or re-plan",
                    )
                continue            # supported: reuse the verdict, proceed to push
            # pre-registration (broadcast) -> protocol run -> settlement with
            # evidence_id; supported claims become verified_knowledge, refuted
            # ones stop being repeated by the swarm.
            hyp_id = await asyncio.to_thread(
                register_hypothesis, ctx.memory, ctx.agent_id, hyp)
            ev = await asyncio.to_thread(run_hypothesis, ctx.workspace, hyp)
            await asyncio.to_thread(
                settle_hypothesis, ctx.memory, ctx.agent_id, hyp_id, hyp, ev,
                method="hypothesis_contract")
            # the protocol file rides the branch — the MR record must list it,
            # or the reviewer sees a diff file the record does not explain
            # (the protocol IS the evidence's source code)
            state["changed_files"] = sorted(
                set(state.get("changed_files") or []) | {hyp.script_path})
            note = (ev.meta.get("note") or ev.output[:160]) if ev.meta else ev.output[:160]
            # the verdict rides the MR body too: the reviewer and the owner
            # see it in the verification line, not only in the evidence table
            state["verification"] = (
                f"{state.get('verification', '')} | "
                f"P16 {hyp.hypothesis_id}: {ev.verdict} ({note[:120]})"
            )[:500]
            if ev.verdict == "refuted":
                return await _yield(
                    ctx, state,
                    f"P16 hypothesis '{hyp.hypothesis_id}' REFUTED by measurement: "
                    f"{note} — the idea did not hold; fix the code or re-plan",
                )
            if ev.verdict == "inconclusive":
                ctx.memory.journal(ctx.agent_id, "p16_inconclusive", {
                    "task": hyp_task, "hypothesis": hyp.hypothesis_id,
                    "reason": note[:200]})

    try:
        pushed = await asyncio.to_thread(
            ctx.workspace.commit_and_push,
            branch, f"agent({ctx.agent_id}): {state.get('commit_message', 'work')} "
                    f"(#{state['issue_number']})",
            allow_protected=allow_protected,
        )
    except RuntimeError as exc:
        state["error"] = str(exc)
        state["done"] = True
        return state
    if not pushed:
        state["summary"] = "no changes to push"
        state["done"] = True
        return state
    state["action"] = "open_pr"
    return state


def _register_tools(ctx: AgentContext, state: ResearchState, pr_number: int) -> None:
    """Tools declared in the plan become *pending* registry entries tied to the PR;
    they turn active only if the PR merges. Declarations must point at a file this
    PR really changed — a hallucinated path is ignored."""
    try:
        plan = json.loads(state.get("plan") or "{}")
    except json.JSONDecodeError:
        return
    changed = set(state.get("changed_files", []))
    for spec in (plan.get("reusable_tools") or [])[:5]:
        if not isinstance(spec, dict):
            continue
        name, path, desc = spec.get("name", ""), spec.get("path", ""), spec.get("description", "")
        if valid_tool_name(name) and path in changed and desc:
            if ctx.memory.register_tool(name, path, desc, ctx.agent_id,
                                        state.get("issue_number"), pr_number):
                ctx.memory.post_message(
                    ctx.agent_id, "broadcast",
                    f"Tool `{name}` ({path}) will be reusable once PR #{pr_number} merges.",
                    state.get("issue_number"))


def node_open_pr(ctx: AgentContext, state: ResearchState) -> ResearchState:
    """Open a local MR (merge request) — the reviews table replaces GitHub PRs.

    Rework cycles do not open a second MR: the fixes are pushed to the same
    branch and the changes_requested MR goes back to 'open' for re-review.
    """
    rework_id = state.get("rework_mr")
    if rework_id:
        review = ctx.memory.get_review(rework_id)
        if (review and review["author"] == ctx.agent_id
                and review["status"] == "changes_requested"):
            files = sorted(
                set(review.get("files") or []) | set(state.get("changed_files") or []))
            ctx.memory.reopen_review(
                rework_id, files=files, verification=state.get("verification", ""))
            state["pr_number"] = rework_id
            state["pr_url"] = f"local://mr/{rework_id}"
            _register_tools(ctx, state, rework_id)
            ctx.memory.post_message(
                ctx.agent_id, "handoff",
                f"MR #{rework_id} reworked on branch {state['branch']} — "
                "ready for re-review",
                state.get("issue_number"),
            )
            ctx.memory.journal(ctx.agent_id, "rework_mr", {
                "mr": rework_id, "task": state.get("issue_number"),
                "files": state.get("changed_files", [])[:20],
            })
            ctx.report(node="open_mr", status="handoff",
                       task_id=state.get("issue_number"),
                       intent={"goal": f"MR #{rework_id} reworked, awaiting re-review",
                              "steps": []})
            state["done"] = True
            state["summary"] = f"reworked MR #{rework_id}"
            return state
        # MR vanished / already handled — fall through and open a fresh MR.
    mr_id = ctx.memory.open_review(
        task_id=state.get("issue_number"),
        title=f"[{ctx.agent_id}] {state.get('issue_title', 'work')} (#{state['issue_number']})",
        body=(
            f"Closes task #{state['issue_number']}\n\n"
            f"Authored by agent `{ctx.agent_id}` (role: {ctx.role.value}).\n\n"
            f"## Verification\n{state.get('verification', '(not run)')}\n"
            "Tier 2 (full tests on main+MR) runs in the Integrator before merge.\n\n"
            f"## Changed files\n" + "\n".join(f"- {f}" for f in state.get("changed_files", []))
        ),
        author=ctx.agent_id,
        branch=state["branch"],
        files=list(state.get("changed_files", [])),
        verification=state.get("verification", ""),
    )
    state["pr_number"] = mr_id
    state["pr_url"] = f"local://mr/{mr_id}"
    # The task is now handed off to the MR lifecycle; mark the lease complete
    # before any non-critical memory/notification work can fail.
    ctx.claims.complete(state["issue_number"], ctx.agent_id, state["branch"])
    _register_tools(ctx, state, mr_id)
    ctx.memory.post_message(
        ctx.agent_id, "handoff",
        f"MR #{mr_id} ready for cross-review (branch {state['branch']})",
        state["issue_number"],
    )
    ctx.memory.journal(ctx.agent_id, "open_mr", {
        "mr": mr_id, "task": state.get("issue_number"),
        "branch": state["branch"], "files": state.get("changed_files", [])[:20],
    })
    ctx.report(node="open_mr", status="handoff",
               task_id=state.get("issue_number"),
               intent={"goal": f"MR #{mr_id} awaiting cross-review", "steps": []})
    state["done"] = True
    state["summary"] = f"opened MR #{mr_id}"
    return state


async def node_review(ctx: AgentContext, state: ResearchState) -> ResearchState:
    """Reviewer: pick an open MR not authored by me, review it (P5).

    P10 layers 3+4: before any LLM cost is spent, the reviewer re-scans the
    branch that was ACTUALLY pushed (in its own workspace) and hard-rejects
    any MR whose real diff touches protected paths — no matter what the file
    list recorded by the author's workspace claims."""
    # P19 — panel review: an MR is judged by a PANEL of peers, not one
    # reviewer. This agent votes only when it is an eligible, not-yet-voted
    # member of the current round (round 2 belongs to the original objectors).
    engine = _panel_engine(ctx)
    candidate = None
    for mr in ctx.memory.open_reviews():
        if mr["author"] == ctx.agent_id:
            continue  # never review own MR (P5 — the engine refuses it too)
        if mr["status"] != "open":
            continue  # changes_requested awaits author rework, not re-review
        prow = ctx.memory.get_panel(mr["id"]) or ctx.memory.ensure_panel(mr["id"])
        if prow["state"] != "voting":
            continue  # defense awaits the AUTHOR (node_defend); closed panels stay closed
        if any(v["voter"] == ctx.agent_id
               for v in ctx.memory.panel_votes(mr["id"], round=prow["round"])):
            continue  # one vote per reviewer per round
        if prow["round"] >= 2 and ctx.agent_id not in {
                o["voter"] for o in ctx.memory.panel_objections(mr["id"], round=1)}:
            continue  # after a defense only the original objectors hold a veto
        candidate = mr
        break
    if candidate is None:
        state["summary"] = "no MR awaiting review"
        state["done"] = True
        return state
    mr_id = candidate["id"]
    ctx.report(
        node="review",
        intent={"goal": f"Cross-review MR #{mr_id} by {candidate['author']}",
                "why": "P5: no self-approval", "steps": []},
    )

    # P10 layer 4 (defense in depth): each workspace is a separate trust
    # domain, so the reviewer re-scans the pushed branch itself instead of
    # trusting the file list the author's workspace recorded at open_mr time.
    diff_files = candidate.get("files", [])
    scan = getattr(getattr(ctx, "workspace", None), "branch_files_vs_main", None)
    if scan is not None:
        try:
            scanned = await asyncio.to_thread(scan, candidate["branch"])
            if scanned:
                diff_files = scanned
        except RuntimeError as exc:
            log.warning("[%s] layer-4 re-scan of %s failed: %s",
                        ctx.agent_id, candidate["branch"], exc)

    # P16: show the reviewer the REAL runs recorded for this task (shared
    # evidence), so the verdict weighs measurements, not just author claims.
    ev_rows = ctx.memory.evidence_for_task(candidate.get("task_id"))

    # P16 / hypothesis 1: the reviewer RE-RUNS the author's declared checks in
    # their OWN workspace against the MR branch — a real reviewer runs the
    # paper's code instead of trusting the recorded output. Re-runs are
    # recorded as evidence with method="reviewer_rerun" (hypothesis 3).
    rerun_digest = rerun_note = ""
    rerun_refuted = False
    refuted_cmds: list[str] = []                 # original rows to re-verdict
    rerun_rows = cmds_from_evidence(ev_rows, max_cmds=3, exclude_agent=ctx.agent_id)
    rerun_ws = getattr(ctx, "workspace", None)
    if rerun_rows and candidate.get("branch") and hasattr(rerun_ws, "run_checks_on_branch"):
        try:
            reruns = await asyncio.to_thread(
                rerun_ws.run_checks_on_branch, candidate["branch"],
                [argv for _, argv, _ in rerun_rows], 300)
            rerun_counts: dict[str, int] = {}
            for (cmd_str, _argv, hypothesis), (argv, code, out, secs) in zip(rerun_rows, reruns):
                ev_verdict = verdict_for(code)
                rerun_counts[ev_verdict] = rerun_counts.get(ev_verdict, 0) + 1
                rerun_refuted = rerun_refuted or ev_verdict == "refuted"
                if ev_verdict == "refuted":
                    refuted_cmds.append(cmd_str)
                await asyncio.to_thread(
                    ctx.memory.record_evidence, ctx.agent_id, candidate.get("task_id"),
                    cmd_str, code, ev_verdict,
                    hypothesis=hypothesis or candidate.get("title", ""),
                    method="reviewer_rerun", output=out, duration_s=secs)
            rerun_note = (f"[P16 reviewer re-ran {len(reruns)} check(s) on branch "
                          f"{candidate['branch']}: {rerun_counts}]")
        except Exception as exc:  # noqa: BLE001 — a re-run failure must never kill the review
            rerun_note = f"[P16 reviewer re-run unavailable: {exc}]"
            try:
                await asyncio.to_thread(
                    ctx.memory.record_evidence, ctx.agent_id,
                    candidate.get("task_id"), "reviewer re-run", 128,
                    "inconclusive",
                    hypothesis=candidate.get("title", ""),
                    method="reviewer_rerun", output=str(exc))
            except Exception:  # noqa: BLE001
                pass

    # P17 — protocol reproduction: an exit-code re-run cannot judge a
    # quantitative claim (a protocol always exits 0; the criterion is the
    # verdict). For the author's CONFIRMED protocol rows the reviewer re-runs
    # the full two-run adjudication against the ledger's frozen criterion, on
    # the MR branch. A refuted reproduction re-verdicts the original row,
    # downgrades the hypothesis and blocks the approval — self-reported
    # evidence is never the last word.
    proto_rows = [
        e for e in ev_rows
        if e["agent"] == candidate.get("author") and e["verdict"] == "confirmed"
        and e.get("method") == "protocol" and e.get("hypothesis_id")
    ][-2:]
    if proto_rows and candidate.get("branch") and hasattr(rerun_ws, "run_protocols_on_branch"):
        protocols = []
        for row in proto_rows:
            hyp = ctx.memory.get_hypothesis(row["hypothesis_id"])
            if hyp and hyp.get("protocol_path"):
                try:
                    protocols.append(
                        (row, hyp, Criterion.from_dict(hyp.get("criterion") or {})))
                except ValueError:
                    continue
        if protocols:
            try:
                repros = await asyncio.to_thread(
                    rerun_ws.run_protocols_on_branch, candidate["branch"],
                    [(hyp["protocol_path"], crit) for _row, hyp, crit in protocols])
            except Exception as exc:  # noqa: BLE001 — reproduction must never kill the review
                repros = None
                rerun_note += f" [P17 protocol reproduction unavailable: {exc}]"
            if repros is not None:
                for (row, hyp, crit), ev in zip(protocols, repros):
                    note = str((ev.meta or {}).get("note", ""))[:200]
                    rerun_refuted = rerun_refuted or ev.verdict == "refuted"
                    await asyncio.to_thread(
                        ctx.memory.record_evidence, ctx.agent_id,
                        candidate.get("task_id"),
                        f"protocol rerun of evidence #{row['id']} ({crit.describe()})",
                        ev.exit_code, ev.verdict,
                        hypothesis=row.get("hypothesis", "")[:200],
                        method="reviewer_rerun", output=note or ev.output[-200:],
                        duration_s=ev.duration_s, artifact=hyp["protocol_path"])
                    if ev.verdict == "refuted":
                        await asyncio.to_thread(
                            ctx.memory.set_evidence_verdict, row["id"], "refuted",
                            note=f"reviewer reproduction failed: {note}",
                            by=ctx.agent_id)
                        await asyncio.to_thread(
                            ctx.memory.update_hypothesis, row["hypothesis_id"],
                            "refuted",
                            note=f"reviewer reproduction failed: {note}")
                rerun_note += (f" [P17 protocol reproduction: {len(repros)} "
                               f"claim(s) re-adjudicated on branch "
                               f"{candidate['branch']}]")
    rerun_digest = rerun_note

    # P10 layers 3+4: protected paths never reach main via any MR — rejected
    # here, before the LLM is even asked.
    guard = getattr(getattr(ctx, "local", None), "guard", None)
    violations = (
        guard.check_task(ctx.memory, candidate.get("task_id"), diff_files)
        if guard is not None else []
    )
    if violations:
        verdict = "request_changes"
        comment = (f"P10 protected paths in diff (branch re-scan): {violations[:5]} "
                   "— cannot be approved (owner-only code)")
    else:
        # Mechanical tier-2 gate — the reviewer is a real reviewer: it RUNS the
        # full suite (merge main+MR in a throw-away copy + pytest) BEFORE any
        # LLM opinion, complementing the declared-checks re-run above (which
        # only re-executes what the author declared — this gate catches an MR
        # with NO declared checks at all). Mirrors the owner's `agentics mr
        # merge` and the Integrator's drain gate.
        ws = getattr(ctx, "workspace", None)
        verify_tier2 = getattr(ws, "verify_merge", None)
        mech_ok, mech_note = True, "tier-2 not run: workspace cannot verify_merge"
        if verify_tier2 is not None:
            try:
                mech_ok, mech_note = await asyncio.to_thread(
                    verify_tier2, candidate["branch"])
            except Exception as exc:  # noqa: BLE001 — an untestable MR is a failing MR
                mech_ok, mech_note = False, f"mechanical check error: {exc}"
        state["mechanical_review"] = "pass" if mech_ok else f"fail: {mech_note[:300]}"
        if not mech_ok:
            verdict = "request_changes"
            comment = f"Mechanical review failed before LLM review: {mech_note}"
        else:
            files_desc = "\n".join(f"- {f}" for f in diff_files[:30])
            ev_desc = ""
            if ev_rows:
                ev_lines = [
                    f"- [{e['verdict']}] `{e['cmd'][:100]}` (exit {e['exit_code']}, "
                    f"by {e['agent']}{', note: ' + e['note'][:120] if e['note'] else ''})"
                    for e in ev_rows[-6:]
                ]
                ev_desc = ("\n\nEmpirical evidence recorded for this task (P16):\n"
                           + "\n".join(ev_lines))
                if rerun_note:
                    ev_desc += f"\n\nReviewer's own re-runs: {rerun_note}"
            prompt = (
                f"Review MR #{mr_id}: {candidate['title']}\n{candidate.get('body') or ''}\n\n"
                f"Files:\n{files_desc}\n\n"
                f"Verification recorded by the author: {candidate.get('verification') or '(none)'}"
                f"{ev_desc}\n\n"
                + ("Mechanical check: tier-2 tests PASSED on main+MR (you ran them "
                   "yourself — trust measurements over prose).\n\n"
                   if verify_tier2 is not None else
                   "Mechanical check: tier-2 NOT run in this context — weigh the "
                   "author's recorded verification and the P16 evidence.\n\n")
                + 'Reply ONLY JSON: {"verdict": "approve|request_changes", '
                '"comment": "..."}'
            )
            answer, _ = await _llm_ask(ctx, prompt, mode="concise", purpose="review", state=state)
            if state.get("error"):             # reviewer unreachable — retry next cycle
                state["done"] = True
                return state
            parsed = parse_json_block(answer)
            verdict = parsed.get("verdict", "request_changes")
            comment = parsed.get("comment", "auto review")
            if verify_tier2 is not None:
                comment = f"[mechanical: tier-2 passed] {comment}"
            if ev_rows:                        # P16: make the evidence trail explicit
                counts: dict[str, int] = {}
                for e in ev_rows:
                    counts[e["verdict"]] = counts.get(e["verdict"], 0) + 1
                comment = f"{comment} [P16 evidence: {len(ev_rows)} run(s) {counts}]"


    if rerun_refuted:
        # The author's confirmed rows that failed the independent re-run are
        # re-verdicted (with a verdict_by trail), so the shared feed stops
        # presenting them as confirmed and every agent learns the refutation.
        for cmd_str in refuted_cmds:
            for e in ev_rows:
                if e["cmd"] == cmd_str and e["agent"] == candidate.get("author") \
                        and e["verdict"] == "confirmed":
                    await asyncio.to_thread(
                        ctx.memory.set_evidence_verdict, e["id"], "refuted",
                        note="reviewer re-run failed on the MR branch",
                        by=ctx.agent_id)
        ctx.memory.journal(ctx.agent_id, "reviewer_rerun_refuted", {
            "mr": mr_id, "task": candidate.get("task_id"),
            "branch": candidate.get("branch")})
        if verdict == "approve":
            # Evidence overrides model opinion: an independent re-run that refutes
            # the author's check can never be approved, whatever the LLM says.
            verdict = "request_changes"
            comment = (f"{rerun_digest} the author's check FAILED in the reviewer's "
                       f"own workspace — evidence overrides model opinion. {comment}")
    elif rerun_note:
        # the MR record shows what the reviewer independently re-ran (owner
        # visibility: a green MR now says WHICH checks were reproduced)
        comment = f"{comment} {rerun_note}"

    # P19 — the panel: this reviewer's verdict becomes ONE VOTE. One reject
    # opens the defense round immediately; approval needs the panel's
    # unanimous accepts; after a defense, one objector still rejecting is
    # final (the engine handles every consequence).
    vote = "accept" if verdict == "approve" else "reject"
    decision = await asyncio.to_thread(
        engine.cast_vote, mr_id, ctx.agent_id, vote, comment)
    state["panel_decision"] = decision.outcome
    state["panel_digest"] = engine.panel_digest(mr_id)
    if decision.outcome == OUT_APPROVED:
        await asyncio.to_thread(ctx.merge_queue.enqueue, mr_id,
                                author=candidate["author"])
        state["review_verdict"] = "approve"
    elif decision.outcome in (OUT_DEFENSE, OUT_REJECTED):
        state["review_verdict"] = "request_changes"
    else:                              # waiting / ignored — panel still deciding
        state["review_verdict"] = "vote_cast"
    if decision.outcome == OUT_DEFENSE:
        due_s = int(max(0, (decision.panel.get("due_ts") or 0) - time.time()))
        ctx.memory.post_message(
            ctx.agent_id, "alert",
            f"P19: MR #{mr_id} enters DEFENSE — author {candidate['author']} must "
            f"answer EVERY objection within {due_s // 60}m or the MR is rejected "
            f"for good. Objections: {decision.reason[:300]}",
            candidate.get("task_id"),
        )
    ctx.memory.post_message(
        ctx.agent_id, "broadcast",
        f"Review of MR #{mr_id}: {verdict} ({decision.outcome}) — {comment[:200]}",
        candidate.get("task_id"),
    )
    ctx.memory.journal(ctx.agent_id, "review", {
        "mr": mr_id, "verdict": state["review_verdict"], "vote": vote,
        "panel": decision.outcome, "comment": comment[:300],
        "task": candidate.get("task_id"),
    })
    state["review_comment"] = comment
    state["summary"] = f"reviewed MR #{mr_id}: {state['review_verdict']}"
    state["done"] = True
    return state


def pending_defense(ctx: AgentContext) -> tuple[dict, dict] | None:
    """My MR whose panel is in the defense round (P19 — I must defend it)."""
    try:
        for mr in ctx.memory.reviews_by_author(ctx.agent_id):
            panel = ctx.memory.get_panel(mr["id"])
            if panel and panel["state"] == DEFENSE:
                return mr, panel
    except Exception:  # noqa: BLE001 — routing must never crash on a read
        return None
    return None


async def node_defend(ctx: AgentContext, state: ResearchState) -> ResearchState:
    """P19 — defend my MR: the panel raised objections, I answer EVERY one.

    One shot per MR: the defense must cover each objection (keyed by the
    objector's id) and may attach fresh evidence from REAL runs; anything
    less — or silence past the deadline — ends the MR as rejected (final),
    and the task returns to the open pool for a fresh attempt.
    """
    picked = pending_defense(ctx)
    if picked is None:
        state["summary"] = "no defense pending"
        state["done"] = True
        return state
    mr, panel = picked
    mr_id = mr["id"]
    objections = ctx.memory.panel_objections(mr_id, round=1)
    obj_desc = "\n".join(
        f"- [{o['voter']}] {o['objection']}" for o in objections) or "- (none recorded)"
    ev_rows = ctx.memory.evidence_for_task(mr.get("task_id"))
    ev_desc = "\n".join(
        f"- [{e['verdict']}] `{e['cmd'][:90]}` (exit {e['exit_code']}, by {e['agent']})"
        for e in ev_rows[-6:])
    ctx.report(
        node="defend",
        intent={"goal": f"Defend MR #{mr_id} against {len(objections)} objection(s)",
                "why": "P19: one unanswered objection rejects the MR for good",
                "steps": []},
    )
    prompt = (
        f"Your MR #{mr_id} ({mr['title']}) was challenged by the peer panel.\n\n"
        f"Objections you must answer:\n{obj_desc}\n\n"
        f"MR body:\n{(mr.get('body') or '')[:2000]}\n\n"
        f"Recorded evidence:\n{ev_desc or '(none)'}\n\n"
        'Reply ONLY JSON: {"concede": false, "rebuttals": '
        '{"<objector-id>": "your factual answer, with numbers where possible"}, '
        '"extra_checks": ["python -m pytest tests/x.py"], '
        '"note": "one-line summary"}\n'
        "Rules: answer EVERY objection by its objector id (an unanswered one "
        "rejects the MR immediately); extra_checks may only invoke python or "
        "pytest (the workspace runs no shells); if an objection is right, set "
        "concede=true — the MR is rejected and the task reopens for a fresh, "
        "better attempt."
    )
    answer, _ = await _llm_ask(ctx, prompt, mode="concise", purpose="defend", state=state)
    if state.get("error"):             # LLM hiccup — retry next cycle (deadline looms)
        state["done"] = True
        return state
    parsed = parse_json_block(answer) or {}
    engine = _panel_engine(ctx)
    if parsed.get("concede"):
        decision = await asyncio.to_thread(
            engine.concede, mr_id, ctx.agent_id, str(parsed.get("note", "")))
        state["defense_outcome"] = decision.outcome
        state["summary"] = f"conceded MR #{mr_id}: {decision.outcome}"
        state["done"] = True
        return state
    rebuttals = {str(k): str(v)[:2000]
                 for k, v in (parsed.get("rebuttals") or {}).items() if str(v).strip()}
    # the defense is backed by RUNS, not prose: every declared extra check is
    # executed in the author's own workspace and lands in the shared feed
    evidence_ids: list[int] = []
    for cmd in list(parsed.get("extra_checks") or [])[:3]:
        try:
            ev = await asyncio.to_thread(run_experiment, ctx.workspace, cmd, 300)
        except Exception as exc:  # noqa: BLE001 — a failed check is evidence too
            ev = Evidence(cmd=str(cmd)[:400], exit_code=-1, verdict="inconclusive",
                          output=f"defense check error: {exc}")
        row_id = await asyncio.to_thread(
            ctx.memory.record_evidence, ctx.agent_id, mr.get("task_id"),
            ev.cmd, ev.exit_code, ev.verdict,
            hypothesis=f"defense of MR #{mr_id}", method="defense_check",
            output=ev.output, duration_s=ev.duration_s)
        evidence_ids.append(row_id)
    decision = await asyncio.to_thread(
        engine.submit_defense, mr_id, ctx.agent_id, rebuttals,
        note=str(parsed.get("note", ""))[:2000], evidence_ids=evidence_ids)
    state["defense_outcome"] = decision.outcome
    state["summary"] = (
        f"defended MR #{mr_id}: {decision.outcome} {decision.reason[:120]}".strip())
    state["done"] = True
    return state


def node_integrate(ctx: AgentContext, state: ResearchState) -> ResearchState:
    """Integrator: drain the merge queue serially (P4) — local merges only."""
    def _verify(mr_id: int) -> tuple[bool, str]:
        review = ctx.memory.get_review(mr_id)
        if review is None:
            return False, f"MR #{mr_id} not found"
        return ctx.workspace.verify_merge(review["branch"])

    def _merge(mr_id: int) -> bool:
        review = ctx.memory.get_review(mr_id)
        if review is None:
            return False
        return ctx.workspace.merge_branch_to_main(
            review["branch"],
            f"merge MR #{mr_id} ({review['author']}) — task #{review.get('task_id')}",
        )

    def _scan_branch(branch: str) -> list[str]:
        # P10 layer 4: the Integrator re-scans the pushed branch in its own
        # workspace — the queue's guard verdict then rests on the real diff,
        # not on the file list the author's workspace recorded.
        return ctx.workspace.branch_files_vs_main(branch)

    ctx.report(node="integrate",
               intent={"goal": "Drain the merge queue (serial, tier-2 first)", "steps": []})
    scan = getattr(ctx.workspace, "branch_files_vs_main", None)
    merged = ctx.merge_queue.drain(
        ctx.agent_id, verify=_verify, merge=_merge,
        scan_branch=_scan_branch if scan is not None else None)
    state["summary"] = f"merged {merged} MR(s)"
    state["done"] = True
    return state


def _plan_digest(state: ResearchState) -> str:
    """What the agent tried, compressed: the plan's steps (or commit message)."""
    try:
        plan = json.loads(state.get("plan") or "{}")
    except json.JSONDecodeError:
        plan = {}
    steps = "; ".join(str(x) for x in plan.get("steps", []))[:600]
    return steps or state.get("commit_message", "") or state.get("issue_title", "")


def _trace_outcome(state: ResearchState) -> str:
    if state.get("pr_number"):
        return "pending"             # review/merge will update it
    if state.get("yield_reason"):
        return "yield"
    if state.get("error"):
        return "failed"
    return "no_change"


def node_reflect(ctx: AgentContext, state: ResearchState) -> ResearchState:
    """Write lessons into personal memory AND a structured trace into shared
    memory (what was tried, where it failed, how it ended) → every agent can
    learn from it (GEA's group experience sharing)."""
    lesson_bits = []
    if state.get("summary"):
        lesson_bits.append(state["summary"])
    if state.get("error"):
        lesson_bits.append(f"error: {state['error']}")
    if state.get("yield_reason"):
        lesson_bits.append(f"yielded: {state['yield_reason']}")
    lesson = "; ".join(lesson_bits) or "cycle completed"
    # Any claimed task that did not reach MR creation must be unlocked, including
    # early exits (no LLM, duplicate research, invalid plan, failed tests).
    # A rework cycle holds no claim and must leave its parked in_review task alone.
    issue_number = state.get("issue_number")
    if issue_number and not state.get("pr_number") and not state.get("rework_mr"):
        try:
            ctx.claims.release(issue_number, ctx.agent_id, "cycle ended without an MR")
        except Exception as exc:  # noqa: BLE001
            log.warning("could not release claim #%s for %s: %s",
                        issue_number, ctx.agent_id, exc)
    ctx.meta["personal"].add_lesson(lesson, issue_number)
    ctx.memory.add_decision(ctx.agent_id, f"cycle {ctx.cycle} done", lesson[:200])
    # Adaptive prompts: feed the cycle outcome back into the LLM-authored
    # mission (failures accumulate → the mission is reworked next cycle).
    bump_prompt_feedback(ctx, state)
    ctx.memory.journal(ctx.agent_id, "cycle_done", {
        "cycle": ctx.cycle, "summary": lesson[:300],
        "outcome": _trace_outcome(state),
    })
    if state.get("issue_number") and ctx.role in (Role.CODER, Role.WRITER):
        trace_id = ctx.memory.add_trace(
            ctx.agent_id, ctx.role.value, ctx.cycle, _trace_outcome(state),
            issue=state["issue_number"],
            category=state.get("category") or categorize_issue(
                state.get("issue_title", ""), state.get("issue_body", "")),
            pr_number=state.get("pr_number"),
            attempted=_plan_digest(state),
            failure=state.get("error") or state.get("yield_reason") or "",
            files=state.get("changed_files", []),
        )
        try:      # LifeMem: file this trajectory into its workflow cluster now,
            lifemem.assign_trace(   # distillation happens later, on merge
                ctx.memory, trace_id, state.get("category", "other"),
                _plan_digest(state))
        except Exception as exc:  # noqa: BLE001 — memory hiccups must not kill a cycle
            log.warning("could not cluster trace #%s: %s", trace_id, exc)
    ctx.report(node="reflect", status="cycle-end",
               intent={"goal": "cycle complete", "steps": []})
    locks = getattr(getattr(ctx, "local", None), "locks", None)
    if locks is not None:
        try:
            released = locks.release_agent(ctx.agent_id)
            if released:
                ctx.memory.journal(ctx.agent_id, "locks_released",
                                   {"files": released[:15]})
        except Exception as exc:  # noqa: BLE001
            log.warning("could not release file locks for %s: %s",
                        ctx.agent_id, exc)
    ctx.heartbeat("cycle-end")
    state["done"] = True
    return state


NODES = {
    "start": node_start,
    "scan": node_scan,
    "claim": node_claim,
    "research": node_research,
    "verify": node_verify,
    "plan": node_plan,
    "implement": node_implement,
    "open_pr": node_open_pr,
    "review": node_review,
    "integrate": node_integrate,
    "reflect": node_reflect,
}

TIMESTAMP = datetime.now(timezone.utc).isoformat()
