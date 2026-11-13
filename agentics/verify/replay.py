"""P23 — replay: a settled claim must survive being re-run.

Real researchers replicate — including their OWN older experiments. The
swarm's settled verdicts (supported / refuted) are stored with the protocol
source (protocol_code, P23) so anyone can re-run them in a fresh workspace
long after P21 pruned the original:

* same verdict        -> `claim_reconfirmed`  (the claim holds over time)
* supported -> refuted-> `claim_falsified`    (drift! the swarm's knowledge
                                               is corrected automatically)
* refuted -> supported-> `claim_overturned`   (the original refutation was
                                               wrong or the code improved)
* now inconclusive    -> `claim_unstable`     (protocol no longer runs
                                               deterministically — reviewer)

Every replay lands in the SAME shared evidence table (method='replay') and
bumps the hypothesis's attempts — the ledger shows how many times a claim
survived scrutiny, which is exactly what a reviewer wants to see.
"""
from __future__ import annotations

from agentics.verify_metrics import Criterion, run_protocol

REPLAY_PATH = "experiments/replay_h{hyp_id}.py"

_STATUS_BY_EVIDENCE = {
    "confirmed": "supported",
    "refuted": "refuted",
    "inconclusive": "inconclusive",
}


def _drift_event(old: str, new: str) -> str:
    if old == new:
        return "claim_reconfirmed"
    if old == "supported" and new == "refuted":
        return "claim_falsified"
    if old == "refuted" and new == "supported":
        return "claim_overturned"
    return "claim_unstable"          # settled -> inconclusive


def replay_hypothesis(
    memory, workspace, hyp_id: int, agent: str = "owner",
    timeout: int = 120, repeat: int = 2,
) -> dict:
    """Re-run ONE settled hypothesis from its stored protocol source.

    Never raises: every failure mode becomes a journaled, queryable outcome.
    Returns a small report dict {hyp_id, status, replayed, event, evidence_id,
    note}."""
    hyp = memory.get_hypothesis(hyp_id)
    if hyp is None:
        return {"hyp_id": hyp_id, "replayed": False,
                "note": "no such hypothesis"}
    if hyp["status"] not in ("supported", "refuted"):
        return {"hyp_id": hyp_id, "replayed": False,
                "note": f"not settled ({hyp['status']}) — nothing to replicate"}
    code = hyp.get("protocol_code") or ""
    if not code.strip():
        return {"hyp_id": hyp_id, "replayed": False,
                "note": "no stored protocol (registered before P23)"}
    try:
        criterion = Criterion.from_dict(hyp["criterion"])
    except ValueError as exc:
        return {"hyp_id": hyp_id, "replayed": False,
                "note": f"stored criterion unusable: {exc}"}

    old_status = hyp["status"]
    path = REPLAY_PATH.format(hyp_id=hyp_id)
    try:
        workspace.write_file(path, code)
    except ValueError as exc:              # guard refused the write
        memory.journal(agent, "claim_replay_blocked",
                       {"hyp": hyp_id, "reason": str(exc)[:200]})
        return {"hyp_id": hyp_id, "replayed": False,
                "note": f"write refused: {exc}"[:200]}
    try:
        ev = run_protocol(workspace, path, criterion, repeat=repeat,
                          timeout=timeout)
    except Exception as exc:  # noqa: BLE001 — a broken protocol must not break the sweep
        memory.journal(agent, "claim_unstable",
                       {"hyp": hyp_id, "reason": f"replay crashed: {exc}"[:200]})
        return {"hyp_id": hyp_id, "replayed": False,
                "note": f"runner error: {exc}"[:200]}

    ev_id = memory.record_evidence(
        agent, hyp.get("task_id"), ev.cmd, ev.exit_code, ev.verdict,
        hypothesis=hyp["claim"], method="replay", output=ev.output,
        duration_s=ev.duration_s, artifact=path)
    memory.attach_protocol_outcome(
        ev_id, ev.meta.get("metrics", {}), ev.meta.get("runs", []), hyp_id)
    new_status = _STATUS_BY_EVIDENCE[ev.verdict]
    note = f"replay: {ev.meta.get('note', '')}"[:600]
    memory.update_hypothesis(hyp_id, new_status, evidence_id=ev_id, note=note)
    event = _drift_event(old_status, new_status)
    memory.journal(agent, event, {
        "hyp": hyp_id, "claim": hyp["claim"][:160],
        "was": old_status, "now": new_status,
        "criterion": criterion.describe(), "evidence": ev_id,
        "measured": ev.meta.get("metrics", {}),
    })
    return {"hyp_id": hyp_id, "replayed": True, "status": new_status,
            "was": old_status, "event": event, "evidence_id": ev_id,
            "note": note}


def replay_settled(memory, workspace, limit: int = 20, agent: str = "owner",
                   timeout: int = 120) -> list[dict]:
    """Replicate the newest settled claims (that carry stored protocols).

    The owner's / supervisor's maintenance sweep: one broken claim never
    aborts the batch — the report list is what the reviewer reads."""
    out: list[dict] = []
    for hyp in memory.settled_hypotheses(limit=limit, require_code=True):
        try:
            out.append(replay_hypothesis(memory, workspace, hyp["id"],
                                         agent=agent, timeout=timeout))
        except Exception as exc:  # noqa: BLE001 — never break the sweep
            out.append({"hyp_id": hyp["id"], "replayed": False,
                        "note": f"sweep error: {type(exc).__name__}: {exc}"[:200]})
    return out
