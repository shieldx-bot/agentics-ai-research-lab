"""Graph nodes — each node is a step of one agent's research cycle.

GitHub/Git/SQLite operations are synchronous and run in LangGraph's executor.
LLM and paper-search nodes are async and await their providers directly.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

from agentics.agents.base import AgentContext
from agentics.agents.roles import Role, build_system_prompt, parse_json_block
from agentics.evolution.scoring import IssueInfo, build_profile, categorize_issue, rank_issues
from agentics.evolution.traces import (
    EVOLUTION_ALLOWED_PATHS, build_evolution_issue, tool_context, trace_context,
    evolution_path_allowed, valid_tool_name,
)
from agentics.memory import lifemem
from agentics.github_ops import branch_ops, issue_ops, pr_ops
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


async def _llm_ask(ctx: AgentContext, prompt: str, mode: str = "concise") -> tuple[str, list[str]]:
    """Await the provider directly; never create or nest an event loop in a graph worker."""
    if ctx.llm is None:
        return "", []
    resp = await ctx.llm.ask(
        prompt, mode=mode, system=build_system_prompt(
            ctx.agent_id, ctx.role, ctx.settings.research_topic)
    )
    ctx.meta["account"] = ctx.llm.account.name
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
    return out


def node_scan(ctx: AgentContext, state: ResearchState) -> ResearchState:
    """Supervisor: sweep stale claims, broadcast backlog status."""
    state["action"] = "scan"
    freed = ctx.claims.sweep_stale()
    issues = issue_ops.scan_open_issues(ctx.gh.repo)
    if freed:
        ctx.memory.post_message(
            ctx.agent_id, "alert",
            f"Freed {len(freed)} stale claim(s): {freed}"
        )
    summary = (
        f"Backlog scan: {len(issues)} open unclaimed issue(s). "
        + (f"Stale claims freed: {freed}." if freed else "")
    )
    ctx.memory.post_message(ctx.agent_id, "broadcast", summary)
    state["summary"] = summary
    _maybe_open_evolution_issue(ctx, state)
    state["done"] = True
    return state


def _maybe_open_evolution_issue(ctx: AgentContext, state: ResearchState) -> None:
    """Every N supervisor cycles, open ONE evidence-based self-evolution issue.
    It is worked like any other issue (claim -> PR -> cross-review -> merge)."""
    n = int(getattr(ctx.settings, "evolution_every_n_cycles", 0) or 0)
    if n <= 0 or ctx.cycle % n != 0:
        return
    if not ctx.memory.recent_traces(limit=1, failures_only=True):
        return                      # no failure evidence yet → nothing to evolve from
    title, body = build_evolution_issue(ctx.memory)
    try:
        issue = issue_ops.create_issue_if_absent(
            ctx.gh.repo, title, body, labels=["kind/evolution"])
    except Exception as exc:  # noqa: BLE001
        log.warning("could not open self-evolution issue: %s", exc)
        return
    if issue is not None:
        ctx.memory.add_decision(ctx.agent_id, f"open self-evolution issue #{issue.number}",
                                "periodic GEA-style prompt refinement")
        state["summary"] += f" Opened self-evolution issue #{issue.number}."


def node_claim(ctx: AgentContext, state: ResearchState) -> ResearchState:
    """Coder/Writer: claim an issue (P1), ranked by Performance–Novelty.

    Replaces "oldest first": each issue is scored with this agent's own merge
    history (see agentics/evolution/scoring.py); ties fall back to oldest-first.
    Only the Coder may take self-evolution issues.
    """
    issues = issue_ops.scan_open_issues(ctx.gh.repo)
    infos = [
        IssueInfo(
            number=i.number, title=i.title, body=(i.body or "")[:800],
            labels=tuple(lb.name for lb in i.labels),
            created_ts=i.created_at.timestamp() if i.created_at else 0.0,
        )
        for i in issues
    ]
    if ctx.role != Role.CODER:
        infos = [i for i in infos if i.category != "evolution"]
    if not infos:
        state["summary"] = "no unclaimed issues; resting this cycle"
        state["done"] = True
        return state
    by_number = {i.number: i for i in issues}
    profile = build_profile(ctx.memory.agent_outcomes().get(ctx.agent_id, []))
    for info, score in rank_issues(infos, profile)[:3]:      # lost race → next best
        issue = by_number[info.number]
        branch = branch_ops.branch_name(ctx.agent_id, issue.number, issue.title)
        if not ctx.claims.try_claim(issue.number, ctx.agent_id, branch):
            continue
        state["issue_number"] = issue.number
        state["issue_title"] = issue.title
        state["issue_body"] = (issue.body or "")[:4000]
        state["branch"] = branch
        state["category"] = info.category
        state["action"] = "research"
        ctx.memory.add_decision(
            ctx.agent_id, f"claim #{issue.number}",
            f"category={info.category} score={score:.2f} (alpha*sqrt(nov)*priority)")
        return state
    state["summary"] = "lost race for top-ranked issues; retry next cycle"
    state["done"] = True
    return state


async def node_research(ctx: AgentContext, state: ResearchState) -> ResearchState:
    """Researcher: search papers, dedup vs vector memory, store findings."""
    topic = state.get("issue_title") or ctx.settings.research_topic
    query = (
        f"{topic} — context: {ctx.settings.research_topic}. "
        "Focus on methods, benchmarks, reproducible results."
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
    resp = await papers_mod.search_papers(ctx.llm, query)
    ctx.meta["account"] = ctx.llm.account.name
    state["papers"] = [{"title": topic, "answer": resp.answer}]
    state["citations"] = list(resp.citations)
    state["bibtex"] = cite_fmt.to_bibtex(cite_fmt.parse_citations(resp.citations))

    title = f"Findings: {topic}"
    content = f"{resp.answer[:6000]}\n\n## BibTeX\n\n{state['bibtex'][:3000]}"
    fid = ctx.memory.add_finding(
        ctx.agent_id, title, content, resp.citations, state.get("issue_number")
    )
    ctx.vector.add(content, {"agent": ctx.agent_id, "finding_id": fid,
                             "issue": state.get("issue_number")})
    ctx.memory.post_message(
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
    evolution_rule = (
        "\nThis is a SELF-EVOLUTION issue: you may ONLY change these paths: "
        + ", ".join(EVOLUTION_ALLOWED_PATHS) + ".\n"
        if category == "evolution" else ""
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
        f"{tools_txt}\n{evolution_rule}\n"
        "Propose a concrete implementation plan for this repo. Reply ONLY with "
        'JSON: {"files": [{"path": "...", "change": "new|edit"}], '
        '"steps": ["..."], "commit_message": "...", '
        '"reuses": ["<registered tool name you rely on>"], '
        '"reusable_tools": [{"name": "snake_case_name", "path": "<file you create>", '
        '"description": "what it does, in one line"}]} '
        "('reuses' and 'reusable_tools' may be empty lists; only list a tool in "
        "'reusable_tools' if it is a genuinely reusable utility.)"
    )
    answer, _ = await _llm_ask(ctx, prompt, mode="concise")
    parsed = parse_json_block(answer)
    if not parsed:
        state["error"] = "plan parsing failed"
        state["done"] = True
        return state
    state["plan"] = json.dumps(parsed, ensure_ascii=False)[:8000]
    state["commit_message"] = parsed.get("commit_message", "update")
    reuses = [n for n in parsed.get("reuses", []) if isinstance(n, str)][:10]
    state["reuses"] = reuses
    for name in reuses:
        ctx.memory.bump_tool_use(name)      # no-op if the name is not registered
    return state


def _yield(ctx: AgentContext, state: ResearchState, reason: str) -> ResearchState:
    ctx.claims.release(state["issue_number"], ctx.agent_id, f"yield: {reason}")
    ctx.workspace.sync_main()          # discard local changes
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


async def node_implement(ctx: AgentContext, state: ResearchState) -> ResearchState:
    """Coder: create branch, write code from plan, run tests, push (P2/P3)."""
    branch = state["branch"]
    ctx.workspace.sync_main()
    branch_ops.create_branch(ctx.gh.repo, branch)
    ctx.workspace.checkout_new_branch(branch)

    plan = json.loads(state.get("plan") or "{}")
    for spec in plan.get("files", []):
        path, change = spec.get("path", ""), spec.get("change", "new")
        current = ctx.workspace.read_file(path) if change == "edit" else ""
        prompt = (
            f"Issue: {state.get('issue_title')}\nPlan step for this file:\n"
            f"{json.dumps(spec, ensure_ascii=False)}\n\n"
            f"Current content:\n{current[:6000]}\n\n"
            "Return the FULL new file content only, no explanations."
        )
        answer, _ = await _llm_ask(ctx, prompt, mode="concise")
        content = answer.strip().removeprefix("```").removesuffix("```")
        if content.strip():
            ctx.workspace.write_file(path, content)

    # NOTE: must be the working-tree status — `diff_files_vs_main()` compares
    # commits and is empty before the commit, which silently disabled P3.
    changed = ctx.workspace.changed_files()
    state["changed_files"] = changed

    # Guard: self-evolution PRs may only touch the allow-listed paths.
    if state.get("category") == "evolution":
        out_of_bounds = [f for f in changed if not evolution_path_allowed(f)]
        if out_of_bounds:
            return _yield(ctx, state, f"evolution PR outside allowed paths {out_of_bounds[:5]}")

    # P3: ownership check against other active roles
    active_roles = {
        a["role"] for a in ctx.memory.active_agents(stale_after_s=300)
        if a["agent"] != ctx.agent_id
    }
    ok, foreign = ctx.ownership.evaluate(ctx.role.value, changed, active_roles)
    if not ok:
        return _yield(ctx, state, f"foreign files {foreign[:5]}")

    # Tier 1 (author, before push): compile + fast tests. Tier 2 (full suite on
    # main+PR) runs later, in the Integrator, right before merge.
    t1_ok, verification = _tier1(ctx, changed)
    state["verification"] = verification
    if not t1_ok:
        state["error"] = verification
        state["done"] = True
        return state

    try:
        pushed = ctx.workspace.commit_and_push(
            branch, f"agent({ctx.agent_id}): {state.get('commit_message', 'work')} "
                    f"(#{state['issue_number']})"
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
    repo = ctx.gh.repo
    pr = pr_ops.open_pr(
        repo,
        title=f"[{ctx.agent_id}] {state.get('issue_title', 'work')} (#{state['issue_number']})",
        head=state["branch"],
        base=repo.default_branch,
        body=(
            f"Closes #{state['issue_number']}\n\n"
            f"Authored by agent `{ctx.agent_id}` (role: {ctx.role.value}).\n\n"
            f"## Verification\n{state.get('verification', '(not run)')}\n"
            "Tier 2 (full tests on main+PR) runs in the Integrator before merge.\n\n"
            f"## Changed files\n" + "\n".join(f"- {f}" for f in state.get("changed_files", []))
        ),
        author_agent=ctx.agent_id,
    )
    state["pr_number"] = pr.number
    state["pr_url"] = pr.html_url
    _register_tools(ctx, state, pr.number)
    ctx.memory.post_message(
        ctx.agent_id, "handoff",
        f"PR #{pr.number} ready for cross-review: {pr.html_url}",
        state["issue_number"],
    )
    ctx.claims.complete(state["issue_number"], ctx.agent_id, state["branch"])
    state["done"] = True
    state["summary"] = f"opened PR #{pr.number}"
    return state


async def node_review(ctx: AgentContext, state: ResearchState) -> ResearchState:
    """Reviewer: pick an open PR not authored by me, review it (P5)."""
    repo = ctx.gh.repo
    candidate = None
    for pr in repo.get_pulls(state="open"):
        if pr.user and pr.user.login == "shieldx-bot" and f"[{ctx.agent_id}]" in pr.title:
            continue  # never review own PR
        labels = {lb.name for lb in pr.get_labels()}
        if "reviewed" in labels:
            continue
        candidate = pr
        break
    if candidate is None:
        state["summary"] = "no PR awaiting review"
        state["done"] = True
        return state

    diff = candidate.get_files()
    files_desc = "\n".join(f"- {f.filename} (+{f.additions}/-{f.deletions})" for f in diff[:30])
    prompt = (
        f"Review PR #{candidate.number}: {candidate.title}\n{candidate.body or ''}\n\n"
        f"Files:\n{files_desc}\n\n"
        'Reply ONLY JSON: {"verdict": "approve|request_changes", '
        '"comment": "..."}'
    )
    answer, _ = await _llm_ask(ctx, prompt, mode="concise")
    parsed = parse_json_block(answer)
    verdict = parsed.get("verdict", "request_changes")
    comment = parsed.get("comment", "auto review")

    pr_to_review = ctx.gh.repo.get_pull(candidate.number)
    if verdict == "approve":
        pr_ops.approve_review(pr_to_review, comment)
        author = candidate.title.split("]")[0].lstrip("[") or "agent"
        ctx.merge_queue.enqueue(candidate.number, author=author)
        ctx.memory.set_trace_outcome(candidate.number, "approved")
        state["review_verdict"] = "approve"
    else:
        pr_to_review.create_issue_comment(f"Review ({ctx.agent_id}): {comment}")
        # the reviewer's reason becomes a failure trace every agent can learn from
        ctx.memory.set_trace_outcome(candidate.number, "rejected", comment)
        ctx.memory.drop_tools(candidate.number)
        state["review_verdict"] = "request_changes"
    try:                                   # without this the same PR is re-reviewed every cycle
        candidate.add_to_labels("reviewed")
    except Exception as exc:  # noqa: BLE001
        log.warning("could not label PR #%s reviewed: %s", candidate.number, exc)
    state["review_comment"] = comment
    state["summary"] = f"reviewed PR #{candidate.number}: {verdict}"
    state["done"] = True
    return state


def node_integrate(ctx: AgentContext, state: ResearchState) -> ResearchState:
    """Integrator: drain the merge queue serially (P4)."""
    def _verify(pr_number: int) -> tuple[bool, str]:
        try:
            head = ctx.gh.repo.get_pull(pr_number).head.ref
        except Exception as exc:  # noqa: BLE001
            return False, f"cannot read PR head: {exc}"
        return ctx.workspace.verify_merge(head)

    merged = ctx.merge_queue.drain(ctx.agent_id, verify=_verify)
    state["summary"] = f"merged {merged} PR(s)"
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
    lesson = "; ".join(lesson_bits) or "cycle completed"
    ctx.meta["personal"].add_lesson(lesson, state.get("issue_number"))
    ctx.memory.add_decision(ctx.agent_id, f"cycle {ctx.cycle} done", lesson[:200])
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
