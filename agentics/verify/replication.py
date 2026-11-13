"""P24bis — independent replication: a DIFFERENT agent re-runs the claim.

Composes with the predecessor's stack instead of duplicating it:

* P23 replay (agentics/verify/replay.py) re-runs settled claims over TIME
  from the same ledger (drift detection, supervisor sweep);
* P24 ledger (SharedMemory.hypotheses_needing_replication +
  record_replication) is the canonical once-per-swarm bookkeeping with
  per-agent INDEPENDENCE — the author cannot replicate their own claim;
* THIS module is the engine the P24 node wiring feeds: it rebuilds the
  shareable protocol in the PEER's own workspace, runs it through the full
  two-run reproducibility gate (verify_metrics.run_protocol), and writes
  BOTH ledgers — record_replication (bookkeeping) AND
  record_replication_detail (provenance: who, measured metrics, runs,
  duration, peer evidence row).

Verdict handling mirrors the P17 asymmetry policy:
* replication CONFIRMS  -> verified_knowledge flags the claim `replicated`
  (author-confirmed becomes independently replicated);
* replication REFUTES   -> the engine closes the ledger as the P24 contract
  requires: the author's evidence row is re-verdicted and the hypothesis
  downgraded to refuted — verified_knowledge stops presenting a dead claim
  as common ground; journaled `replication_refuted`;
* replication INCONCLUSIVE (sandbox hiccup, unparsable output) only
  annotates — it NEVER refutes, and the P24 retry budget (2 attempts) keeps
  the claim available for another peer.
"""
from __future__ import annotations

from agentics.verify_metrics import Criterion, run_protocol

REPLICATION_PATH = "experiments/replicate_h{hyp_id}.py"


def replicate_candidate(memory, workspace, hyp: dict, agent: str,
                        timeout: int = 120, repeat: int = 2) -> dict:
    """Independently replicate ONE settled claim in MY workspace.

    `hyp` is a row of SharedMemory.hypotheses_needing_replication (the P24
    selector: settled 'supported', shareable protocol, not my own claim).
    Never raises — every failure mode becomes a journaled, queryable
    outcome. Returns a report dict {hypothesis_id, evidence_id, verdict,
    event, replicated, note}."""
    hyp_id = hyp.get("id") or 0
    ev_id = hyp.get("evidence_id")
    author = hyp.get("agent") or ""
    if not agent or agent == author:
        return {"hypothesis_id": hyp_id, "evidence_id": ev_id,
                "replicated": False,
                "note": "refusing self-replication (independence required)"}
    if (hyp.get("replication") or "") == "confirmed":
        return {"hypothesis_id": hyp_id, "evidence_id": ev_id,
                "replicated": False,
                "note": "already replicated — no second confirmation needed"}
    try:
        criterion = Criterion.from_dict(hyp.get("criterion") or {})
    except ValueError as exc:
        return {"hypothesis_id": hyp_id, "evidence_id": ev_id,
                "replicated": False,
                "note": f"stored criterion unusable: {exc}"}
    code = hyp.get("protocol_code") or ""
    if not code.strip():
        return {"hypothesis_id": hyp_id, "evidence_id": ev_id,
                "replicated": False,
                "note": "protocol not shareable — cannot replicate"}

    path = REPLICATION_PATH.format(hyp_id=hyp_id)
    try:
        workspace.write_file(path, code)
    except ValueError as exc:              # P10: guard refused the write
        memory.journal(agent, "replication_blocked",
                       {"hyp": hyp_id, "reason": str(exc)[:200]})
        return {"hypothesis_id": hyp_id, "evidence_id": ev_id,
                "replicated": False, "note": f"write refused: {exc}"[:200]}
    try:
        ev = run_protocol(workspace, path, criterion, repeat=repeat,
                          timeout=timeout)
    except Exception as exc:  # noqa: BLE001 — a broken protocol must not break the cycle
        memory.journal(agent, "replication_inconclusive",
                       {"hyp": hyp_id, "reason": f"runner error: {exc}"[:200]})
        return {"hypothesis_id": hyp_id, "evidence_id": ev_id,
                "replicated": False, "note": f"runner error: {exc}"[:200]}

    note = str(ev.meta.get("note", ""))[:600]
    # provenance row: one per (evidence, replicator) — the UNIQUE constraint
    # is the only gate (the P24 retry budget legitimately allows a SECOND
    # peer to attempt after an inconclusive first attempt, so a per-evidence
    # 'taken' pre-check would break the budget). False = I already recorded
    # this evidence myself; another peer's row never blocks mine.
    if not memory.record_replication_detail(
            ev_id, hyp_id, agent, ev.verdict,
            ev.meta.get("metrics", {}), ev.meta.get("runs", []),
            note=note, duration_s=ev.duration_s):
        return {"hypothesis_id": hyp_id, "evidence_id": ev_id,
                "verdict": ev.verdict, "replicated": False,
                "note": "already recorded my replication of this evidence"}

    # the peer's own evidence row: method='replication', linked to the claim
    rep_ev = memory.record_evidence(
        agent, hyp.get("task_id"), ev.cmd, ev.exit_code, ev.verdict,
        hypothesis=hyp.get("claim") or "", method="replication",
        output=ev.output, duration_s=ev.duration_s, artifact=path,
        broadcast=False)
    memory.attach_protocol_outcome(
        rep_ev, ev.meta.get("metrics", {}), ev.meta.get("runs", []), hyp_id)
    # canonical P24 bookkeeping — never re-picked (or retry-budgeted)
    memory.record_replication(hyp_id, ev.verdict)

    if ev.verdict == "confirmed":
        event = "replication_confirmed"
        memory.journal(agent, event, {
            "hyp": hyp_id, "claim": (hyp.get("claim") or "")[:160],
            "measured": ev.meta.get("metrics", {}),
            "criterion": criterion.describe(), "peer_evidence": rep_ev,
        })
    elif ev.verdict == "refuted":
        # P24 contract: the caller closes a refuted replication — re-verdict
        # the author's evidence and downgrade the claim (verified_knowledge
        # self-corrects; the P23 contradiction alert may fire too).
        event = "replication_refuted"
        memory.set_evidence_verdict(
            ev_id, "refuted",
            note=f"independent replication by {agent}: {note}"[:600], by=agent)
        memory.update_hypothesis(hyp_id, "refuted",
                                 note=f"refuted by independent replication "
                                      f"({agent}): {note}"[:600])
        memory.journal(agent, event, {
            "hyp": hyp_id, "claim": (hyp.get("claim") or "")[:160],
            "was_evidence": ev_id, "measured": ev.meta.get("metrics", {}),
            "criterion": criterion.describe(), "peer_evidence": rep_ev,
        })
    else:
        event = "replication_inconclusive"
        memory.journal(agent, event, {
            "hyp": hyp_id, "claim": (hyp.get("claim") or "")[:160],
            "note": note[:300],
            "why": "sandbox/parse hiccup — never a false refutation; the "
                   "P24 retry budget keeps the claim available",
        })
    return {"hypothesis_id": hyp_id, "evidence_id": ev_id,
            "verdict": ev.verdict, "event": event,
            "replicated": ev.verdict == "confirmed",
            "peer_evidence": rep_ev, "note": note}


def replicate_pending(memory, workspace, agent: str, limit: int = 1,
                      timeout: int = 120) -> list[dict]:
    """Replicate up to `limit` claims the P24 selector offers ME (oldest
    first — the longest-unreplicated common ground goes first). One broken
    candidate never aborts the batch."""
    out: list[dict] = []
    for hyp in memory.hypotheses_needing_replication(agent, limit=limit):
        try:
            out.append(replicate_candidate(
                memory, workspace, hyp, agent, timeout=timeout))
        except Exception as exc:  # noqa: BLE001 — never break the batch
            out.append({"hypothesis_id": hyp.get("id"),
                        "evidence_id": hyp.get("evidence_id"),
                        "replicated": False,
                        "note": f"replication error: {exc}"[:200]})
    return out
