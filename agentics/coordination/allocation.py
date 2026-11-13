"""Contract-Net-lite task allocation — the supervisor becomes a manager.

Gap closed (owner, 2026-11-13): "cách các agent phối hợp còn chưa hiệu quả".
The supervisor's scan was a janitor only — it swept stale claims and topped
up the backlog — while task picking was a pure race between claimants, each
looking only at its OWN history (``node_claim`` ranks with its own profile).
Nobody routed the work to the agents best able to do it.

The classic fix is the Contract Net Protocol (Smith 1980): a manager
announces tasks, contractors bid, the manager awards to the best bid. This
module is that protocol, mechanically and without LLM calls:

* ``announce`` — the open, unclaimed backlog (``scan_open_tasks``).
* ``bid``     — every LIVE claimant agent (fresh heartbeat, coder/writer
  role) scores every task with the lab's own Performance–Novelty model
  (``agentics.evolution.scoring.score_issue``) over its merge history.
* ``award``   — the best (task, agent) pairs win, greedy over the global
  score matrix: one award per task, one per agent, deterministic tie-break.
  Awards are persisted (``task_awards``) and REWRITTEN wholesale on every
  scan, so a stale award can never outlive one supervisor cycle.

The award is a market signal, not a lock (keeping writes single-threaded and
decisions local — Cognition, "Multi-Agents: What's Actually Working"): the
claimant still races and still owns the decision; the award only lifts the
task's rank in ``node_claim`` by ``settings.coord_award_bonus``.
"""
from __future__ import annotations

from agentics.agents.roles import Role
from agentics.evolution.scoring import IssueInfo, build_profile, score_issue
from agentics.memory.shared_memory import SharedMemory

CLAIMANT_ROLES = frozenset({Role.CODER.value, Role.WRITER.value})
# Only the coder may take self-evolution tasks (same rule as node_claim).
CODER_ONLY_CATEGORIES = frozenset({"evolution"})


def recommend_awards(
    memory: SharedMemory,
    max_awards: int = 4,
    stale_after_s: float = 300.0,
) -> list[dict]:
    """Rewrite the swarm's award table from the current backlog + roster.

    Returns the awarded rows (``{task_id, agent, score}``, best first) —
    the caller journals/broadcasts them. Deterministic: equal scores are
    broken by task id, then agent id, so two scans of the same state never
    disagree. With no tasks or no live claimants the table is emptied (the
    cleanup IS the semantics: awards are worth exactly one scan cycle).
    """
    tasks = memory.scan_open_tasks(limit=30)
    agents = [
        a for a in memory.active_agents(stale_after_s=stale_after_s)
        if a.get("role") in CLAIMANT_ROLES
    ]
    if not tasks or not agents:
        memory.replace_task_awards([])
        return []
    outcomes = memory.agent_outcomes()
    profiles = {
        a["agent"]: build_profile(outcomes.get(a["agent"], [])) for a in agents
    }
    triples: list[tuple[float, int, str]] = []
    for t in tasks:
        info = IssueInfo(
            number=t["id"], title=t["title"], body=(t["body"] or "")[:800],
            labels=tuple(lb for lb in (t.get("labels") or "").split(",") if lb),
            created_ts=t["ts"],
        )
        for a in agents:
            if info.category in CODER_ONLY_CATEGORIES and a["role"] != Role.CODER.value:
                continue
            triples.append((score_issue(profiles[a["agent"]], info), t["id"], a["agent"]))
    triples.sort(key=lambda x: (-round(x[0], 9), x[1], x[2]))
    chosen: list[dict] = []
    used_tasks: set[int] = set()
    used_agents: set[str] = set()
    for score, task_id, agent in triples:
        if task_id in used_tasks or agent in used_agents:
            continue
        used_tasks.add(task_id)
        used_agents.add(agent)
        chosen.append({"task_id": task_id, "agent": agent,
                       "score": round(float(score), 4)})
        if len(chosen) >= max_awards:
            break
    memory.replace_task_awards(chosen)
    return chosen


def award_digest(awards: list[dict]) -> str:
    """One owner-readable line: ``#12→a03 (1.23), #7→a05 (0.87)``."""
    return ", ".join(
        f"#{a['task_id']}→{a['agent']} ({a['score']:.2f})" for a in awards)
