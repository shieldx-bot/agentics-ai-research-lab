"""Graph nodes — each node is a step of one agent's research cycle.

Nodes are synchronous (GitHub/Git/SQLite are sync libs) and wrapped by the
graph in a thread executor; only LLM calls are async (kept as coroutines
invoked from within async nodes).
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone

from agentics.agents.base import AgentContext
from agentics.agents.roles import Role, build_system_prompt, parse_json_block
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


def _llm_ask(ctx: AgentContext, prompt: str, mode: str = "concise") -> tuple[str, list[str]]:
    if ctx.llm is None:
        return "", []
    resp = asyncio.get_event_loop().run_until_complete(
        ctx.llm.ask(prompt, mode=mode, system=build_system_prompt(
            ctx.agent_id, ctx.role, ctx.settings.research_topic))
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
    state["done"] = True
    return state


def node_claim(ctx: AgentContext, state: ResearchState) -> ResearchState:
    """Coder/Writer: claim the oldest unclaimed issue (P1)."""
    issues = issue_ops.scan_open_issues(ctx.gh.repo)
    if not issues:
        state["summary"] = "no unclaimed issues; resting this cycle"
        state["done"] = True
        return state
    issue = issues[0]
    branch = branch_ops.branch_name(ctx.agent_id, issue.number, issue.title)
    if not ctx.claims.try_claim(issue.number, ctx.agent_id, branch):
        state["summary"] = f"lost race for issue #{issue.number}; retry next cycle"
        state["done"] = True
        return state
    state["issue_number"] = issue.number
    state["issue_title"] = issue.title
    state["issue_body"] = (issue.body or "")[:4000]
    state["branch"] = branch
    state["action"] = "research"
    return state


def node_research(ctx: AgentContext, state: ResearchState) -> ResearchState:
    """Researcher: search papers, dedup vs vector memory, store findings."""
    topic = state.get("issue_title") or ctx.settings.research_topic
    query = (
        f"{topic} — context: {ctx.settings.research_topic}. "
        "Focus on methods, benchmarks, reproducible results."
    )
    # P7: dedup — skip if the swarm already covered this
    known = asyncio.get_event_loop().run_until_complete(
        papers_mod.dedup_query(ctx.vector, topic)
    )
    if len(known) >= 3:
        state["summary"] = f"topic already covered by {len(known)} findings; skipping"
        state["done"] = True
        return state

    if ctx.llm is None:
        state["error"] = "no LLM account available"
        state["done"] = True
        return state
    resp = asyncio.get_event_loop().run_until_complete(
        papers_mod.search_papers(ctx.llm, query)
    )
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


def node_plan(ctx: AgentContext, state: ResearchState) -> ResearchState:
    """Coder: turn research into an implementation plan (LLM, JSON)."""
    prompt = (
        f"Issue #{state.get('issue_number')}: {state.get('issue_title')}\n"
        f"{state.get('issue_body')}\n\nResearch summary:\n"
        f"{state.get('papers', [{}])[0].get('answer', '')[:2500]}\n\n"
        "Propose a concrete implementation plan for this repo. Reply ONLY with "
        'JSON: {"files": [{"path": "...", "change": "new|edit"}], '
        '"steps": ["..."], "commit_message": "..."}'
    )
    answer, _ = _llm_ask(ctx, prompt, mode="concise")
    parsed = parse_json_block(answer)
    if not parsed:
        state["error"] = "plan parsing failed"
        state["done"] = True
        return state
    state["plan"] = json.dumps(parsed, ensure_ascii=False)[:4000]
    state["commit_message"] = parsed.get("commit_message", "update")
    return state


def node_implement(ctx: AgentContext, state: ResearchState) -> ResearchState:
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
        answer, _ = _llm_ask(ctx, prompt, mode="concise")
        content = answer.strip().removeprefix("```").removesuffix("```")
        if content.strip():
            ctx.workspace.write_file(path, content)

    # safe verification: run pytest inside workspace when tests changed
    if any(f.startswith("tests/") for f in plan.get("files", [])):
        code, out = ctx.workspace.run_command(["python", "-m", "pytest", "-x", "-q"], timeout=300)
        if code not in (0, 5):   # 5 = no tests collected
            state["error"] = f"workspace tests failed (exit {code}): {out[-500:]}"
            state["done"] = True
            return state

    changed = ctx.workspace.diff_files_vs_main()
    state["changed_files"] = changed
    # P3: ownership check against other active roles
    active_roles = {
        a["role"] for a in ctx.memory.active_agents(stale_after_s=300)
        if a["agent"] != ctx.agent_id
    }
    ok, foreign = ctx.ownership.evaluate(ctx.role.value, changed, active_roles)
    if not ok:
        ctx.claims.release(state["issue_number"], ctx.agent_id, "yield: foreign files")
        ctx.workspace.sync_main()          # discard local changes
        state["yield_reason"] = f"foreign files {foreign[:5]}"
        state["done"] = True
        ctx.memory.add_decision(ctx.agent_id, "yield", f"issue #{state['issue_number']}: {foreign[:5]}")
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
            f"## Changed files\n" + "\n".join(f"- {f}" for f in state.get("changed_files", []))
        ),
        author_agent=ctx.agent_id,
    )
    state["pr_number"] = pr.number
    state["pr_url"] = pr.html_url
    ctx.memory.post_message(
        ctx.agent_id, "handoff",
        f"PR #{pr.number} ready for cross-review: {pr.html_url}",
        state["issue_number"],
    )
    ctx.claims.complete(state["issue_number"], ctx.agent_id, state["branch"])
    state["done"] = True
    state["summary"] = f"opened PR #{pr.number}"
    return state


def node_review(ctx: AgentContext, state: ResearchState) -> ResearchState:
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
    answer, _ = _llm_ask(ctx, prompt, mode="concise")
    parsed = parse_json_block(answer)
    verdict = parsed.get("verdict", "request_changes")
    comment = parsed.get("comment", "auto review")

    pr_to_review = ctx.gh.repo.get_pull(candidate.number)
    if verdict == "approve":
        pr_ops.approve_review(pr_to_review, comment)
        ctx.merge_queue.enqueue(candidate.number, author=candidate.title.split("]")[0].lstrip("[") or "agent")
        state["review_verdict"] = "approve"
    else:
        pr_to_review.create_issue_comment(f"Review ({ctx.agent_id}): {comment}")
        state["review_verdict"] = "request_changes"
    state["review_comment"] = comment
    state["summary"] = f"reviewed PR #{candidate.number}: {verdict}"
    state["done"] = True
    return state


def node_integrate(ctx: AgentContext, state: ResearchState) -> ResearchState:
    """Integrator: drain the merge queue serially (P4)."""
    merged = ctx.merge_queue.drain(ctx.agent_id)
    state["summary"] = f"merged {merged} PR(s)"
    state["done"] = True
    return state


def node_reflect(ctx: AgentContext, state: ResearchState) -> ResearchState:
    """Write lessons into personal + shared memory → compounding intelligence."""
    lesson_bits = []
    if state.get("summary"):
        lesson_bits.append(state["summary"])
    if state.get("error"):
        lesson_bits.append(f"error: {state['error']}")
    lesson = "; ".join(lesson_bits) or "cycle completed"
    ctx.meta["personal"].add_lesson(lesson, state.get("issue_number"))
    ctx.memory.add_decision(ctx.agent_id, f"cycle {ctx.cycle} done", lesson[:200])
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
