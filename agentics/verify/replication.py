"""P23bis — independent replication: a DIFFERENT agent re-runs the claim.

Composes with the predecessor's P23 replay (agentics/verify/replay.py):
replay re-runs settled claims over TIME from the same ledger (drift
detection, supervisor sweep); replication adds the social dimension — the
re-runner must be a peer, never the author, so a bug in the author's
environment (stale cache, wrong seed, lucky draw) cannot confirm its own
claim. The machine contract is identical to every other layer here:

* the protocol SOURCE travels with the hypothesis (protocol_code, stored
  at registration) — evidence must be shareable, not just the verdict;
* the peer re-runs it through verify_metrics.run_protocol — the FULL two-run
  reproducibility gate applies to the replicator too;
* the verdict is `measured <op> target` against the frozen criterion —
  never an LLM's opinion.

Verdict handling mirrors the P17 asymmetry policy:
* replication CONFIRMS  -> the claim is flagged `replicated` in
  verified_knowledge (author-confirmed becomes independently replicated);
* replication REFUTES   -> the author's evidence row is re-verdicted and the
  hypothesis downgraded to refuted — verified_knowledge stops presenting a
  dead claim as common ground; journaled `replication_refuted`;
* replication INCONCLUSIVE (sandbox hiccup, unparsable output) only
  annotates — it NEVER refutes, and the slot stays open for another peer.
"""
from __future__ import annotations

from agentics.verify_metrics import Criterion, run_protocol

REPLICATION_PATH = "experiments/replicate_h{hyp_id}.py"


def replicate_candidate(memory, workspace, candidate: dict, agent: str,
                        timeout: int = 120, repeat: int = 2) -> dict:
    """Independently replicate ONE pending candidate in MY workspace.

    `candidate` is a row of SharedMemory.pending_replications (claim,
    criterion, protocol_code, author, evidence_id, hypothesis_id).
    Never raises — every failure mode becomes a journaled, queryable outcome.
    Returns a report dict {hypothesis_id, evidence_id, verdict, event,
    replicated, note}."""
    hyp_id = candidate.get("hypothesis_id") or 0
    ev_id = candidate.get("evidence_id")
    author = candidate.get("author") or ""
    if not agent or agent == author:
        return {"hypothesis_id": hyp_id, "evidence_id": ev_id,
                "replicated": False,
                "note": "refusing self-replication (independence required)"}
    try:
        criterion = Criterion.from_dict(candidate.get("criterion") or {})
    except ValueError as exc:
        return {"hypothesis_id": hyp_id, "evidence_id": ev_id,
                "replicated": False,
                "note": f"stored criterion unusable: {exc}"}
    code = candidate.get("protocol_code") or ""
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
    recorded = memory.record_replication(
        ev_id, hyp_id, agent, ev.verdict,
        ev.meta.get("metrics", {}), ev.meta.get("runs", []),
        note=note, duration_s=ev.duration_s)
    if not recorded:
        return {"hypothesis_id": hyp_id, "evidence_id": ev_id,
                "verdict": ev.verdict, "replicated": False,
                "note": "slot taken — another peer replicated first"}

    # the peer's own evidence row: method='replication', linked to the claim
    rep_ev = memory.record_evidence(
        agent, candidate.get("task_id"), ev.cmd, ev.exit_code, ev.verdict,
        hypothesis=candidate.get("claim") or "", method="replication",
        output=ev.output, duration_s=ev.duration_s, artifact=path,
        broadcast=False)
    memory.attach_protocol_outcome(
        rep_ev, ev.meta.get("metrics", {}), ev.meta.get("runs", []), hyp_id)

    if ev.verdict == "confirmed":
        event = "replication_confirmed"
        memory.journal(agent, event, {
            "hyp": hyp_id, "claim": (candidate.get("claim") or "")[:160],
            "measured": ev.meta.get("metrics", {}),
            "criterion": criterion.describe(), "peer_evidence": rep_ev,
        })
    elif ev.verdict == "refuted":
        # evidence overrides the author's verdict (P17 asymmetry: only a real
        # refutation re-verdicts — an inconclusive replication never does)
        event = "replication_refuted"
        memory.set_evidence_verdict(
            ev_id, "refuted",
            note=f"independent replication by {agent}: {note}"[:600], by=agent)
        memory.update_hypothesis(hyp_id, "refuted",
                                 note=f"refuted by independent replication "
                                      f"({agent}): {note}"[:600])
        memory.journal(agent, event, {
            "hyp": hyp_id, "claim": (candidate.get("claim") or "")[:160],
            "was_evidence": ev_id, "measured": ev.meta.get("metrics", {}),
            "criterion": criterion.describe(), "peer_evidence": rep_ev,
        })
    else:
        event = "replication_inconclusive"
        memory.journal(agent, event, {
            "hyp": hyp_id, "claim": (candidate.get("claim") or "")[:160],
            "note": note[:300],
            "why": "sandbox/parse hiccup — slot left open for another peer",
        })
    return {"hypothesis_id": hyp_id, "evidence_id": ev_id,
            "verdict": ev.verdict, "event": event,
            "replicated": ev.verdict == "confirmed",
            "peer_evidence": rep_ev, "note": note}


def replicate_pending(memory, workspace, agent: str, limit: int = 1,
                      timeout: int = 120) -> list[dict]:
    """Replicate up to `limit` pending claims (the reviewer's bounded share
    per cycle — one broken candidate never aborts the batch)."""
    out: list[dict] = []
    for candidate in memory.pending_replications(agent, limit=limit):
        try:
            out.append(replicate_candidate(
                memory, workspace, candidate, agent, timeout=timeout))
        except Exception as exc:  # noqa: BLE001 — never break the batch
            out.append({"hypothesis_id": candidate.get("hypothesis_id"),
                        "evidence_id": candidate.get("evidence_id"),
                        "replicated": False,
                        "note": f"replication error: {exc}"[:200]})
    return out
