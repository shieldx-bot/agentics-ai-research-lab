"""Swarm evaluation — how good is this swarm AS a swarm?

Output quality alone cannot judge a multi-agent research lab: a single
correct answer can come from luck, premature convergence or one agent
carrying the rest. This module grades the COLLABORATION TRAJECTORY from the
lab's own immutable traces (journal, tasks, MRs, P19 panels, LLM ledger) —
no LLM, no new machinery, purely read-only aggregation:

1. coordination & task   — merge rate, cross-agent panel verification,
                           citation-domain coverage, author spread;
2. convergence & dynamics — panel rounds/cycles to a verdict, approval rate,
                           defense save rate;
3. trajectory & handoff  — productive-vs-looping cycles, trajectory length,
                           claim churn/friction, cycle-error share;
4. resilience & scale    — LLM ok-rate, budget-breaker trips, error
                           recovery, replication counts, scalability
                           elasticity S = (%ΔP / %ΔN) across snapshots;
5. governance & cost     — quality gap (objectors on rejected vs approved),
                           peer-substantiated incoherence rate, cost per
                           merged MR in the honest units the ledger has
                           (calls + wall-clock seconds; cookie providers
                           expose no token counts).

`evaluate_swarm` returns a plain JSON-able dict grouped exactly like the
taxonomy above; `save_snapshot`/`scalability_S` persist each evaluation so
S becomes computable once the lab has run at two different swarm sizes.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from agentics.memory.shared_memory import SharedMemory

# journal writers that are machinery, not swarm members
_SYSTEM_AGENTS = {"orchestrator", "owner", "guard", "peer-panel"}

# cycle_done outcomes that mean "this cycle moved the lab forward"
_PRODUCTIVE_OUTCOMES = {"merged", "rejected", "failed", "yield", "rework"}


def _rate(num: float | int | None, den: float | int | None) -> float | None:
    if den is None or den <= 0 or num is None:
        return None
    return float(num) / float(den)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _distinct_domains(citations: list[str]) -> set[str]:
    """arxiv.org/html/x and arxiv.org/abs/x are ONE source domain."""
    domains: set[str] = set()
    for url in citations:
        if "://" not in str(url):
            continue                       # not a URL — not a source domain
        try:
            netloc = str(url).split("://", 1)[-1].split("/", 1)[0].lower()
        except Exception:  # noqa: BLE001 — a malformed URL is still a URL
            continue
        netloc = netloc.removeprefix("www.").removeprefix("dx.")
        if netloc:
            domains.add(netloc)
    return domains


def evaluate_swarm(mem: SharedMemory, *, since_ts: float = 0.0,
                   now: float | None = None) -> dict:
    """Compute the five metric groups from the lab's retained traces.

    Read-only: nothing is written. Every value is either a real measurement
    or ``None`` (honest "not enough data" — never a fabricated number)."""
    now = now if now is not None else time.time()
    events = mem.journal_events(since_ts)
    tstats = mem.trace_stats(since_ts)
    pstats = mem.panel_stats(since_ts)
    llm = mem.llm_usage_stats(since_ts)
    findings = [f for f in mem.list_findings(limit=2000) if f.get("ts", 0) >= since_ts]
    traces = [t for t in mem.recent_traces(limit=5000) if t.get("ts", 0) >= since_ts]
    # cycle failures are posted to the message bus (kind=alert), not the journal
    messages = [m for m in mem.recent_messages(limit=5000, since_ts=since_ts)]

    # ---- journal decompositions --------------------------------------
    cycle_done = [e for e in events if e["event"] == "cycle_done"]
    productive = [e for e in cycle_done
                  if (e["detail"].get("outcome") or "") in _PRODUCTIVE_OUTCOMES]
    node_events = [e for e in events if e["event"] == "node"]
    claim_events = [e for e in events if e["event"] in ("claim", "handoff")]
    release_events = [e for e in events if e["event"] == "release"]
    friction_events = [e for e in events if e["event"] in (
        "lock_conflict", "claim_unstable", "claim_replay_blocked")]
    breaker_events = [e for e in events if e["event"] == "budget_breaker_open"]
    retry_recovered = [e for e in events if e["event"] == "plan_retry_recovered"]
    rework_starts = [e for e in events if e["event"] == "rework_start"]
    cycle_errors = [m for m in messages
                    if m["kind"] == "alert"
                    and m["content"].startswith("cycle error")]

    agents_active = sorted({e["agent"] for e in events
                            if e["agent"] not in _SYSTEM_AGENTS})
    agent_count = len(agents_active)

    # ---- panel-level facts (voter breadth, objections, rounds) --------
    decided_panels = (mem.panels_overview(limit=100, state="approved")
                      + mem.panels_overview(limit=100, state="rejected"))
    voters_per_panel: list[tuple[str, int]] = []      # (state, distinct voters)
    objectors_per_panel: list[tuple[str, int]] = []
    rounds: list[float] = []
    for p in decided_panels:
        voters = {v.get("voter") for v in mem.panel_votes(p["mr_id"])}
        voters.discard(None)
        voters_per_panel.append((p["state"], len(voters)))
        objectors = {o.get("voter")
                     for o in mem.panel_objections(p["mr_id"])}
        objectors.discard(None)
        objectors_per_panel.append((p["state"], len(objectors)))
        if p.get("round"):
            rounds.append(float(p["round"]))
    cross_agent_panels = sum(1 for s, n in voters_per_panel if n >= 2)

    # ---- findings coverage -------------------------------------------
    all_citations = [c for f in findings for c in (f.get("citations") or [])]
    domains = _distinct_domains(all_citations)

    # ---- error recovery (resilience) ----------------------------------
    recovered, errored_agents = 0, 0
    for m in cycle_errors:
        agent = m["agent"]
        errored_agents += 1
        nxt = next((c for c in cycle_done
                    if c["agent"] == agent and c["ts"] > m["ts"]), None)
        if nxt is not None:
            recovered += 1

    merged = tstats["by_outcome"].get("merged", 0)

    report = {
        "ts": now,
        "generated": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
        "since_ts": since_ts,
        "agents_active": agent_count,
        "agents_active_names": agents_active,
        "groups": {
            # ---------------------------------------------------- G1
            "coordination_task": {
                "task_completion_rate": tstats["merge_rate"],
                "judged_outcomes": tstats["judged"],
                "coordination_effectiveness": _rate(
                    cross_agent_panels, len(voters_per_panel)),
                "panels_with_cross_review": cross_agent_panels,
                "panels_decided": len(voters_per_panel),
                "exploration_coverage": _rate(len(domains),
                                              max(1, len(all_citations))),
                "findings": len(findings),
                "citation_domains": len(domains),
                "solution_diversity": _rate(
                    len({t["agent"] for t in traces if t["outcome"] in
                         ("merged", "rejected", "failed")}),
                    max(1, agent_count)),
                "category_spread": len(tstats["by_category"]),
            },
            # ---------------------------------------------------- G2
            "convergence_dynamics": {
                "convergence_correctness": pstats["approval_rate"],
                "mean_panel_rounds": _mean(rounds),
                "mean_trace_cycle": _mean(
                    [float(t["cycle"]) for t in traces
                     if t["outcome"] in ("merged", "rejected", "failed")]),
                "defense_save_rate": pstats["defense_save_rate"],
                "panels_defended": pstats["defended"],
            },
            # ---------------------------------------------------- G3
            "trajectory_handoff": {
                "productive_vs_looping": _rate(len(productive), len(cycle_done)),
                "productive_cycles": len(productive),
                "total_cycles": len(cycle_done),
                "trajectory_length_nodes": _rate(len(node_events), len(cycle_done)),
                "handoff_churn": _rate(len(release_events), len(claim_events)),
                "claim_friction_events": len(friction_events),
                "cycle_error_share": _rate(len(cycle_errors),
                                           len(cycle_done) + len(cycle_errors)),
            },
            # ---------------------------------------------------- G4
            "resilience_scalability": {
                "llm_ok_rate": llm["ok_rate"],
                "budget_breaker_opens": len(breaker_events),
                "error_recovery_rate": _rate(recovered, errored_agents),
                "plan_retry_recovered": len(retry_recovered),
                "rework_starts": len(rework_starts),
                "replication_counts": mem.replication_counts(),
            },
            # ---------------------------------------------------- G5
            "governance_cost": {
                "quality_gap_objectors": _quality_gap(objectors_per_panel),
                "rejected_why": pstats["rejected_why"],
                "incoherence_rate": _incoherence_rate(pstats["rejected_why"],
                                                      pstats["decided"]),
                "llm_calls": llm["calls"],
                "llm_mean_duration_s": llm["mean_duration_s"],
                "llm_seconds_per_merged": (llm["total_duration_s"] / merged)
                                          if merged else None,
                "llm_calls_per_merged": _rate(llm["calls"], merged),
                "cost_unit_note": "calls + wall-clock seconds (cookie "
                                  "providers expose no token counts)",
            },
        },
    }
    return report


def _quality_gap(objectors_per_panel: list[tuple[str, int]]) -> float | None:
    """Mean distinct objectors on REJECTED panels minus APPROVED ones.

    A healthy panel raises objections exactly where work is defective: a
    positive gap means rejections carry the evidence, not the mood."""
    rej = [n for s, n in objectors_per_panel if s == "rejected"]
    app = [n for s, n in objectors_per_panel if s == "approved"]
    if not rej or not app:
        return None
    return _mean(rej) - _mean(app)


def _incoherence_rate(rejected_why: dict, decided: int | None) -> float | None:
    """Share of decided panels rejected for PEER-SUBSTANTIATED defects
    (objection stood after defense, or the author conceded) — the swarm
    catching its own bad knowledge. Deadline/incomplete misses are process
    failures and are excluded on purpose."""
    if not decided:
        return None
    substantive = rejected_why.get("objector_unmoved", 0) \
        + rejected_why.get("conceded", 0)
    return substantive / decided


# ------------------------------------------------------------------ history
def scalability_S(history: list[dict], current: dict,
                  performance_key: str = "task_completion_rate") -> dict:
    """S = (%Δ performance) / (%Δ swarm size) across two evaluations.

    Compares the CURRENT report against the closest earlier snapshot with a
    DIFFERENT agent count (same-size snapshots say nothing about scaling).
    S > 1 → superlinear scaling; 0 < S < 1 → sublinear; S < 0 → adding
    agents hurt. Performance is merge rate by default, panel approval rate
    as fallback when no judged work exists yet."""
    perf = (current.get("groups", {}).get("coordination_task", {})
            .get(performance_key))
    if perf is None:
        perf = (current.get("groups", {}).get("convergence_dynamics", {})
                .get("convergence_correctness"))
    if perf is None:
        return {"S": None, "note": "no performance measurement in current report"}
    n2 = current.get("agents_active", 0)
    best: dict | None = None
    for snap in history:                      # newest first
        n1 = snap.get("agent_count", 0)
        if n1 and n2 and n1 != n2:
            best = snap
            break
    if best is None:
        return {"S": None,
                "note": "need one earlier snapshot at a different agent_count"}
    p1 = (best.get("report", {}).get("groups", {})
          .get("coordination_task", {}).get(performance_key))
    if p1 is None:
        p1 = (best.get("report", {}).get("groups", {})
              .get("convergence_dynamics", {}).get("convergence_correctness"))
    if not p1 or not perf:
        return {"S": None, "note": "earlier snapshot lacks a performance value"}
    pct_p = (perf - p1) / p1
    pct_n = (n2 - n1) / n1
    if pct_n == 0:
        return {"S": None, "note": "agent count unchanged"}
    return {"S": pct_p / pct_n, "performance_key": performance_key,
            "from": {"agent_count": n1, "performance": p1, "ts": best.get("ts")},
            "to": {"agent_count": n2, "performance": perf}}


def attach_scalability(mem: SharedMemory, report: dict) -> dict:
    """Compute S from persisted snapshots and embed it into the report's
    resilience group (called by the CLI before saving the new snapshot)."""
    history = mem.list_eval_snapshots(limit=50)
    s = scalability_S(history, report)
    report["groups"]["resilience_scalability"]["scalability_S"] = s.get("S")
    report["groups"]["resilience_scalability"]["scalability_note"] = s.get("note", "")
    return report


def save_snapshot(mem: SharedMemory, report: dict) -> int:
    """Persist one evaluation row (history for S and for trend reading)."""
    return mem.save_eval_snapshot(report.get("agents_active", 0), report)
