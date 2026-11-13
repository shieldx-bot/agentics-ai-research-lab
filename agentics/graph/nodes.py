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
from datetime import datetime, timezone
from pathlib import Path

from agentics.agents.base import AgentContext
from agentics.agents.roles import Role, build_system_prompt, parse_json_block
from agentics.evolution.scoring import IssueInfo, build_profile, categorize_issue, rank_issues
from agentics.evolution.traces import (
    EVOLUTION_ALLOWED_PATHS, build_evolution_issue, tool_context, trace_context,
    evolution_path_allowed, valid_tool_name,
)
from agentics.local_ops import branch_name
from agentics.policies.intent import scope_violations
from agentics.verify_ops import run_experiments, summarize
from agentics.memory import lifemem
from agentics.memory.vector_store import VectorStore
from agentics.research import citations as cite_fmt
from agentics.research import papers as papers_mod
from agentics.graph.state import ResearchState

log = logging.getLogger("graph.nodes")

STOP_YIELD = "yield"


# ---------------------------------------------------------------- helpers
def _memory_context(ctx: AgentContext) -> dict:
    msgs = ctx.memory.recent_messages(limit=15)
    findings = ctx.memory.list_findings(limit=10)
    return {
        "messages": "\n".join(
            f"- [{m['agent']}|{m['kind']}] {m['content'][:200]}" for m in msgs
        ) or "(empty)",
        "findings": "\n".join(
            f"- #{f['id']} {f['title']} (by {f['agent']})" for f in findings
        ) or "(empty)",
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
        mark = "SUPPORTED" if h["status"] == "supported" else "REFUTED"
        note = f" — {h['note'][:110]}" if h.get("note") else ""
        lines.append(f"- [{mark}] {h['claim'][:180]} (h#{h['id']}{note})")
    if not lines:
        return ""
    return ("Empirically verified by this lab (exit-code verdicts, P16 — build on "
            "SUPPORTED facts, never re-assert a REFUTED one):\n" + "\n".join(lines))[:1400]


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
    """
    if ctx.llm is None:
        if state is not None:
            state["error"] = "no LLM account available"
        return "", []
    try:
        resp = await ctx.llm.ask(
            prompt, mode=mode, purpose=purpose, system=build_system_prompt(
                ctx.agent_id, ctx.role, ctx.settings.research_topic)
        )
    except Exception as exc:  # noqa: BLE001 — degrade, don't crash the cycle
        log.warning("[%s] %s llm call failed: %s", ctx.agent_id, purpose, exc)
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
    query = (
        f"{topic} — context: {ctx.settings.research_topic}. "
        "Focus on methods, benchmarks, reproducible results."
        + (f"\n\n{knowledge}" if knowledge else "")
    )
    # P7: dedup — skip if the swarm already covered this
    known = await papers_mod.dedup_query(ctx.vector, topic)
    if len(known) >= 3:
        state["summary"] = f"topic already covered by {len(known)} findings; skipping"
        state["done"] = True
        return state

    if ctx.llm is None:
        state["error"] = "no LLM account available"
        state["done"] = True
        return state
    try:
        resp = await papers_mod.search_papers(ctx.llm, query, purpose="research")
    except Exception as exc:  # noqa: BLE001 — degrade, don't crash the cycle
        log.warning("[%s] research llm call failed: %s", ctx.agent_id, exc)
        state["error"] = f"research llm call failed: {exc}"
        state["done"] = True
        return state
    acc = getattr(ctx.llm, "account", None)
    ctx.meta["account"] = getattr(acc, "name", "none") if acc else "none"
    state["papers"] = [{"title": topic, "answer": resp.answer}]
    state["citations"] = list(resp.citations)
    state["bibtex"] = cite_fmt.to_bibtex(cite_fmt.parse_citations(resp.citations))

    title = f"Findings: {topic}"
    content = f"{resp.answer[:6000]}\n\n## BibTeX\n\n{state['bibtex'][:3000]}"
    fid = await asyncio.to_thread(
        ctx.memory.add_finding,
        ctx.agent_id, title, content, resp.citations, state.get("issue_number"),
    )
    await asyncio.to_thread(
        ctx.vector.add, content,
        {"agent": ctx.agent_id, "finding_id": fid, "issue": state.get("issue_number")},
    )
    await asyncio.to_thread(
        ctx.memory.post_message,
        ctx.agent_id, "broadcast",
        f"New finding #{fid} on '{topic}' with {len(resp.citations)} citations.",
        state.get("issue_number"),
    )
    if ctx.role == Role.RESEARCHER:
        state["done"] = True
        state["summary"] = f"stored finding #{fid}"
    else:
        state["action"] = "plan"
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
        f"{tools_txt}\n{evolution_rule}{rework_rule}{knowledge_rule}\n"
        "Propose a concrete implementation plan for this repo. Reply ONLY with "
        'JSON: {"files": [{"path": "...", "change": "new|edit"}], '
        '"steps": ["..."], "commit_message": "...", '
        '"verify_cmds": ["<python -m pytest ... or python <workspace .py script> '
        'that CHECKS the idea; exit 0 must mean the hypothesis holds>"], '
        '"hypotheses": [{"hypothesis_id": "slug", "claim": "one testable claim", '
        '"prediction": "what the measurement will show", '
        '"metric": "snake_case_metric", "comparison": ">|>=|<|<=|==", '
        '"threshold": 0.5, "code": "<FULL python source: writes '
        'json.dumps({\\"value\\": <number>}) to '
        'research_notes/experiments/results/<hypothesis_id>.json>"}]} '
        "('reuses' and 'reusable_tools' may be empty lists; only list a tool in "
        "'reusable_tools' if it is a genuinely reusable utility. 'verify_cmds' "
        "is how your idea gets VERIFIED by running real code instead of an LLM's "
        "opinion — declare at least one check for a code task. 'hypotheses' "
        "(optional, max 2) makes the claim MEASURABLE: the runner re-judges the "
        "criterion from the result JSON, a lying script cannot pass.)"
    )
    answer, _ = await _llm_ask(ctx, prompt, mode="concise", purpose="plan", state=state)
    parsed = parse_json_block(answer)
    if not parsed:
        state["error"] = state.get("error") or "plan parsing failed"
        state["done"] = True
        return state
    state["plan"] = json.dumps(parsed, ensure_ascii=False)[:8000]
    state["commit_message"] = parsed.get("commit_message", "update")
    verify_cmds = [str(c) for c in (parsed.get("verify_cmds") or [])][:5]
    state["verify_cmds"] = verify_cmds
    hyp_raws = [h for h in (parsed.get("hypotheses") or []) if isinstance(h, dict)][:2]
    state["hypotheses"] = hyp_raws
    reuses = [n for n in parsed.get("reuses", []) if isinstance(n, str)][:10]
    state["reuses"] = reuses
    for name in reuses:
        ctx.memory.bump_tool_use(name)      # no-op if the name is not registered
    expected_files = [f.get("path", "") for f in (parsed.get("files") or [])][:15]
    expected_files = [f for f in expected_files if f]
    state["expected_files"] = expected_files
    ctx.report(
        node="plan",
        intent={
            "goal": f"Implement task #{state.get('issue_number')}: {state.get('issue_title', '')}",
            "steps": [str(s) for s in (parsed.get("steps") or [])][:8],
            "expected_files": expected_files,
            "verify_cmds": verify_cmds,
            "hypotheses": [str(h.get("claim", ""))[:120] for h in hyp_raws],
            "why": f"category={state.get('category', 'other')}",
        },
    )
    # P13 — lock the declared scope across the swarm: another agent already
    # planning the same files is stopped BEFORE burning LLM cycles on implement.
    locks = getattr(getattr(ctx, "local", None), "locks", None)
    if locks is not None and expected_files:
        ok, conflicts = await asyncio.to_thread(
            locks.try_acquire_all, ctx.agent_id, expected_files)
        if not ok:
            ctx.memory.journal(ctx.agent_id, "lock_conflict", {
                "task": state.get("issue_number"),
                "files": expected_files,
                "held_by": conflicts,
            })
            return await _yield(
                ctx, state,
                f"P13 files already locked by other agents: "
                f"{list(conflicts.items())[:3]} — pick another angle",
            )
        ctx.memory.journal(ctx.agent_id, "intent_bound", {
            "task": state.get("issue_number"), "files": expected_files,
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


async def node_implement(ctx: AgentContext, state: ResearchState) -> ResearchState:
    """Coder: write code from plan into a local branch, run tests, push (P2/P3/P10)."""
    branch = state["branch"]
    await asyncio.to_thread(ctx.workspace.sync_main)
    if state.get("rework_mr"):
        # resume the pushed branch — the reviewed work must survive the rework
        await asyncio.to_thread(ctx.workspace.checkout_branch_from_origin, branch)
    else:
        await asyncio.to_thread(ctx.workspace.checkout_new_branch, branch)

    plan = json.loads(state.get("plan") or "{}")
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
            f"Current content:\n{current[:6000]}\n\n"
            "Return the FULL new file content only, no explanations."
        )
        answer, _ = await _llm_ask(ctx, prompt, mode="concise", purpose="implement",
                                   state=state)
        content = answer.strip().removeprefix("```").removesuffix("```")
        if content.strip():
            try:
                await asyncio.to_thread(ctx.workspace.write_file, path, content)
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
    # declare MEASURABLE hypotheses (metric + threshold + experiment code).
    # ExperimentRunner re-judges the criterion from the experiment's result
    # JSON, so a lying script cannot claim success. Every verdict lands in the
    # same shared evidence table (H3) the reviewer, dashboard and CLI read.
    hyp_raws = state.get("hypotheses") or []
    if hyp_raws:
        from agentics.verify.hypothesis import parse_hypothesis
        from agentics.verify.runner import ExperimentRunner

        runner = ExperimentRunner(ctx.workspace)
        hyp_task = state.get("issue_number")
        for raw in hyp_raws:
            hyp, reason = parse_hypothesis(raw)
            if hyp is None:                     # malformed proposal: never guessed through
                ctx.memory.journal(ctx.agent_id, "p16_hypothesis_invalid",
                                   {"task": hyp_task, "reason": reason[:200]})
                continue
            vr = await asyncio.to_thread(runner.run, hyp)
            rec = vr.to_record()
            await asyncio.to_thread(
                ctx.memory.record_evidence, ctx.agent_id, hyp_task,
                f"{hyp.script_path} -> {hyp.metric} {hyp.comparison} {hyp.threshold}",
                rec["exit_code"] if rec["exit_code"] is not None else -1,
                vr.verdict,
                hypothesis=f"{hyp.claim} | prediction: {hyp.prediction}",
                method="hypothesis_contract",
                output=rec["reason"] or rec["evidence"][:300],
                duration_s=vr.duration_s, artifact=hyp.result_path)
            if vr.verdict == "refuted":
                return await _yield(
                    ctx, state,
                    f"P16 hypothesis '{hyp.hypothesis_id}' REFUTED by measurement: "
                    f"{rec['reason']} — the idea did not hold; fix the code or re-plan",
                )
            if vr.verdict == "inconclusive":
                ctx.memory.journal(ctx.agent_id, "p16_inconclusive", {
                    "task": hyp_task, "hypothesis": hyp.hypothesis_id,
                    "reason": rec["reason"][:200]})

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
    candidate = None
    for mr in ctx.memory.open_reviews():
        if mr["author"] == ctx.agent_id:
            continue  # never review own MR
        if mr["status"] != "open":
            continue  # changes_requested awaits author rework, not re-review
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
        prompt = (
            f"Review MR #{mr_id}: {candidate['title']}\n{candidate.get('body') or ''}\n\n"
            f"Files:\n{files_desc}\n\n"
            f"Verification recorded by the author: {candidate.get('verification') or '(none)'}"
            f"{ev_desc}\n\n"
            'Reply ONLY JSON: {"verdict": "approve|request_changes", '
            '"comment": "..."}'
        )
        answer, _ = await _llm_ask(ctx, prompt, mode="concise", purpose="review", state=state)
        if state.get("error"):             # reviewer unreachable — retry next cycle
            state["done"] = True
            return state
        parsed = parse_json_block(answer)
        verdict = parsed.get("verdict", "request_changes")
        comment = parsed.get("comment", "auto review")
        if ev_rows:                        # P16: make the evidence trail explicit
            counts: dict[str, int] = {}
            for e in ev_rows:
                counts[e["verdict"]] = counts.get(e["verdict"], 0) + 1
            comment = f"{comment} [P16 evidence: {len(ev_rows)} run(s) {counts}]"

    if verdict == "approve":
        await asyncio.to_thread(
            ctx.memory.set_review_verdict, mr_id, "approved", ctx.agent_id, comment
        )
        await asyncio.to_thread(ctx.merge_queue.enqueue, mr_id,
                                author=candidate["author"])
        await asyncio.to_thread(ctx.memory.set_trace_outcome, mr_id, "approved")
        state["review_verdict"] = "approve"
    else:
        await asyncio.to_thread(
            ctx.memory.set_review_verdict, mr_id, "changes_requested",
            ctx.agent_id, comment,
        )
        # the reviewer's reason becomes a failure trace every agent can learn from
        await asyncio.to_thread(ctx.memory.set_trace_outcome, mr_id, "rejected", comment)
        await asyncio.to_thread(ctx.memory.drop_tools, mr_id)
        state["review_verdict"] = "request_changes"
    ctx.memory.post_message(
        ctx.agent_id, "broadcast",
        f"Review of MR #{mr_id}: {verdict} — {comment[:200]}",
        candidate.get("task_id"),
    )
    ctx.memory.journal(ctx.agent_id, "review", {
        "mr": mr_id, "verdict": verdict, "comment": comment[:300],
        "task": candidate.get("task_id"),
    })
    state["review_comment"] = comment
    state["summary"] = f"reviewed MR #{mr_id}: {verdict}"
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
    "plan": node_plan,
    "implement": node_implement,
    "open_pr": node_open_pr,
    "review": node_review,
    "integrate": node_integrate,
    "reflect": node_reflect,
}

TIMESTAMP = datetime.now(timezone.utc).isoformat()
