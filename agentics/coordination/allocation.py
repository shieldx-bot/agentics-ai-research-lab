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
  scan, so a stale award can never outlive one supervisor cycle. A pair
  re-awarded more than ``ttl`` consecutive scans without being claimed is
  DROPPED (award TTL — the manager stops nagging a task nobody claims).

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
    ttl: int = 3,
) -> list[dict]:
    """Rewrite the swarm's award table from the current backlog + roster.

    Returns the awarded rows (``{task_id, agent, score}``, best first) —
    the caller journals/broadcasts them. Deterministic: equal scores are
    broken by task id, then agent id, so two scans of the same state never
    disagree. With no tasks or no live claimants the table is emptied (the
    cleanup IS the semantics: awards are worth exactly one scan cycle).

    Award TTL (owner dashboard 2026-11-14: task #4 was re-awarded to the
    same coder every scan forever because the awarded agent never claimed
    it). An (task, agent) pair that already sits in the table with
    ``hits`` >= ``ttl`` is dropped from the allocation — the manager stops
    re-awarding a task nobody claims and the task falls back to the
    ordinary profile race. ``ttl <= 0`` disables the TTL. Expired pairs are
    journalled (``award_ttl_expired``) so the owner sees WHY the award
    disappeared.
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
    # TTL bookkeeping: how many consecutive scans each (task, agent) pair
    # has been re-awarded without being claimed (a claim changes the task's
    # status, so it leaves the open backlog and the pair vanishes naturally).
    ttl = int(ttl)
    prior: dict[tuple[int, str], int] = {}
    if ttl > 0:
        prior = {(r["task_id"], r["agent"]): int(r.get("hits", 1))
                 for r in memory.task_awards_rows()}
    chosen: list[dict] = []
    expired: list[dict] = []
    used_tasks: set[int] = set()
    used_agents: set[str] = set()
    for score, task_id, agent in triples:
        if task_id in used_tasks or agent in used_agents:
            continue
        used_tasks.add(task_id)
        used_agents.add(agent)
        row = {"task_id": task_id, "agent": agent,
               "score": round(float(score), 4)}
        hits = prior.get((task_id, agent), 0) + 1
        if ttl > 0 and hits > ttl:
            expired.append(row)
            continue
        row["hits"] = hits
        chosen.append(row)
        if len(chosen) >= max_awards:
            break
    for row in expired:
        try:
            memory.journal(
                "supervisor", "award_ttl_expired",
                {"task": row["task_id"], "agent": row["agent"],
                 "why": f"re-awarded {ttl}x without a claim — allocation "
                        "stopped nagging (task stays open for the profile race)"})
        except Exception:  # noqa: BLE001 — telemetry must never break the scan
            pass
    memory.replace_task_awards(chosen)
    # The public row contract stays {task_id, agent, score} (two scans of
    # the same state return EQUAL lists — determinism the caller can pin);
    # `hits` is internal TTL bookkeeping, readable via task_awards_rows().
    return [{k: v for k, v in row.items() if k != "hits"} for row in chosen]


def award_digest(awards: list[dict]) -> str:
    """One owner-readable line: ``#12→a03 (1.23), #7→a05 (0.87)``."""
    return ", ".join(
        f"#{a['task_id']}→{a['agent']} ({a['score']:.2f})" for a in awards)
