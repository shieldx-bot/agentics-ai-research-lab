"""The register -> run -> settle glue for research-side hypotheses (P16).

Composes ONLY predecessor primitives — no second runner, no second verdict
logic:

* pre-registration lands the claim + structured criterion in the shared
  `hypotheses` table BEFORE the protocol runs (the swarm can see what is
  being tested while it is being tested — and the goalposts are frozen);
* `run_hypothesis` writes the protocol script into the researcher's own
  territory (research_notes/, P3) and judges it through
  `verify_metrics.run_protocol` — two runs, reproducibility gate, the SHARED
  machine verdict (adjudicate), never model opinion. Two protocol output
  contracts are honoured: the stdout ``AGENTICS_RESULT:`` marker (the coder
  path's contract) and, as a fallback, the result JSON the hypothesis
  contract writes to ``h.result_path`` (the bridge contract the reviewer and
  dashboard read) — a protocol that prints nothing but measures correctly
  still settles;
* `settle_hypothesis` turns the run into shared knowledge: one `evidence`
  row (broadcast to every agent, method ``hypothesis_contract``), protocol
  metrics/runs attached, and the hypothesis lifecycle moved to
  supported | refuted | inconclusive.

The SETTLE ANCHOR LAW (verdict provenance round): `supported` requires at
least one verdict input the agent did not author (self_authored_ratio < 1)
— an anchorless supported is downgraded to `unanchored` and waits for
independent replication / owner approval. See agentics/verify/anchors.py.
"""
from __future__ import annotations

import json

from agentics.verify.anchors import (
    VerdictProvenance, apply_settle_law, detect_pinned_data,
    provenance_for_settle,
)
from agentics.verify.hypothesis import Hypothesis, RESULT_DIR
from agentics.verify_metrics import (
    RESULT_MARKER, adjudicate, run_protocol,
)
from agentics.verify_ops import Evidence

# evidence (process) verdict -> hypothesis (claim) lifecycle status
_STATUS_BY_VERDICT = {
    "confirmed": "supported",
    "refuted": "refuted",
    "inconclusive": "inconclusive",
}


def register_hypothesis(memory, agent: str, h: Hypothesis) -> int:
    """Pre-register the hypothesis (status 'testing') and announce it.

    Called BEFORE the protocol runs — the whole swarm sees WHAT will be
    tested and against WHICH criterion while it is running."""
    hyp_id = memory.add_hypothesis(
        agent, h.claim, h.criterion.as_dict(),
        task_id=h.task, finding_id=h.source_finding,
        protocol_path=h.script_path, status="testing",
        protocol_code=h.code[:20000],   # P23: the claim stays replayable
    )
    memory.post_message(
        agent, "evidence",
        f"[HYPOTHESIS] #{hyp_id} pre-registered: {h.claim[:160]} — "
        f"criterion {h.criterion.describe()} (protocol {h.script_path})",
        issue=h.task,
    )
    memory.journal(agent, "hypothesis_registered", {
        "id": hyp_id, "task": h.task, "finding": h.source_finding,
        "criterion": h.criterion.describe(),
    })
    return hyp_id


def run_hypothesis(workspace, h: Hypothesis, repeat: int = 2,
                   timeout: int = 120) -> Evidence:
    """Write the protocol and judge it against the pre-registered criterion.

    Primary contract: the protocol prints ``AGENTICS_RESULT: {...}`` and
    ``verify_metrics.run_protocol`` adjudicates it (2 runs, reproducibility).
    Fallback contract: the protocol instead WRITES its measurement JSON to
    ``h.result_path`` — the value is read back and adjudicated by the same
    ``adjudicate`` function, so the shared verdict logic stays single-source.
    Never raises: a guard-refused write (P10/P14) becomes an inconclusive
    Evidence row with the reason, exactly like a blocked command — the swarm
    must see WHY nothing was measured."""
    try:
        workspace.write_file(h.script_path, h.code)
    except ValueError as exc:
        return Evidence(
            cmd=h.script_path, exit_code=126, verdict="inconclusive",
            method="protocol",
            output=f"protocol write refused by guard: {exc}",
            meta={"note": f"protocol write refused: {exc}",
                  "criterion": h.criterion.as_dict(), "runs": [], "metrics": {}},
        )
    evidence = run_protocol(workspace, h.script_path, h.criterion,
                            repeat=repeat, timeout=timeout)
    meta = evidence.meta if isinstance(evidence.meta, dict) else {}
    # pinned-data anchor: the protocol may declare it measured against a
    # hash-locked dataset — verify the declaration against the ACTUAL file
    # the protocol reads, in the SAME code that is executed.
    if detect_pinned_data(workspace, h.code):
        meta = {**meta, "anchor_pinned_data": True}
        evidence.meta = meta
    if evidence.verdict == "inconclusive" and not meta.get("metrics"):
        # the stdout contract produced no measurement — before declaring
        # "measured nothing", honour the file contract: a protocol that
        # writes its result JSON but prints no marker still measured.
        value = _file_measurement(workspace, h)
        if value is not None:
            metrics = {h.criterion.metric: value}
            verdict, note = adjudicate(metrics, h.criterion)
            evidence_verdict = "confirmed" if verdict == "supported" else verdict
            runs = evidence.meta.get("runs", []) if evidence.meta else []
            return Evidence(
                cmd=f"{h.script_path} ({h.criterion.describe()}; "
                    f"{len(runs)} run(s), result file)",
                exit_code=evidence.exit_code, verdict=evidence_verdict,
                output=f"{note} [result file: {h.result_path}]",
                duration_s=evidence.duration_s,
                ok=evidence_verdict == "confirmed",
                method="protocol",
                meta={"note": note, "runs": runs, "metrics": metrics,
                      "criterion": h.criterion.as_dict()},
            )
    return evidence


def _file_measurement(workspace, h: Hypothesis):
    """Read the contract result JSON and return the numeric measured value
    (the criterion's metric key, else the generic ``value`` key), or None."""
    try:
        text = workspace.read_file(h.result_path)
    except (ValueError, OSError):
        return None
    if not text:
        return None
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict):
        return None
    for key in (h.criterion.metric, "value"):
        v = raw.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return v
    return None


def settle_hypothesis(memory, agent: str, hyp_id: int, h: Hypothesis,
                      evidence: Evidence, method: str | None = None,
                      citations: int = 0, peer_judge: bool = False,
                      require_anchor: bool = True) -> int:
    """Record the run as shared knowledge and close the hypothesis lifecycle.

    Returns the evidence row id. Every agent sees the verdict via the
    record_evidence broadcast (H3) and the settled claim via
    verified_knowledge() — refuted claims stop being repeated, supported
    claims become reusable common ground.

    `method` overrides the evidence row's method label (the coder-path bridge
    records "hypothesis_contract"; the researcher protocol path keeps the
    runner's own "protocol").

    SETTLE ANCHOR LAW: `supported` requires self_authored_ratio < 1 — the
    verdict must carry at least one input the agent did not author. Anchor
    sources here: external citations attached to the claim's origin answer
    (`citations`), a peer judge opinion (`peer_judge`), an already-confirmed
    P24 replication or owner approval (read from the hypothesis row), and a
    pinned-hash dataset the protocol declares (detected in run_hypothesis).
    An anchorless supported is downgraded to `unanchored` + journaled
    `settle_unanchored`; the ledger row records the ratio + anchors either
    way."""
    meta = evidence.meta if isinstance(evidence.meta, dict) else {}
    anchor_state = memory.settle_anchor_state(hyp_id) \
        if hasattr(memory, "settle_anchor_state") else \
        {"replicated": False, "owner_approved": False}
    prov: VerdictProvenance = provenance_for_settle(
        citations=citations, peer_judge=peer_judge,
        replicated=bool(anchor_state.get("replicated")),
        owner_approved=bool(anchor_state.get("owner_approved")),
        pinned_data=bool(meta.get("anchor_pinned_data")),
    )
    status = _STATUS_BY_VERDICT.get(evidence.verdict, "inconclusive")
    status, downgraded = apply_settle_law(status, prov, require_anchor)
    if downgraded:
        try:
            memory.journal(agent, "settle_unanchored", {
                "hyp": hyp_id, "claim": h.claim[:160],
                "self_authored_ratio": prov.ratio,
                "why": "verdict passed the criterion but every deciding "
                       "input was agent-authored — supported withheld "
                       "until independent replication / owner approval",
            })
        except Exception:  # noqa: BLE001 — the law must never break a settle
            pass
    ev_id = memory.record_evidence(
        agent, h.task, evidence.cmd, evidence.exit_code, evidence.verdict,
        hypothesis=f"{h.claim} | prediction: {h.prediction}",
        method=method or evidence.method, output=evidence.output,
        duration_s=evidence.duration_s, artifact=h.script_path,
    )
    memory.attach_protocol_outcome(
        ev_id,
        evidence.meta.get("metrics", {}) if evidence.meta else {},
        evidence.meta.get("runs", []) if evidence.meta else [],
        hypothesis_id=hyp_id,
    )
    status_by_law = status
    note = (evidence.meta or {}).get("note", "") or evidence.output[:200]
    if downgraded:
        note = (f"{note} [unanchored: self_authored_ratio={prov.ratio}, "
                f"no external anchor]")[:600]
    # Wave-2 #4: the verdict carries its EMPIRICAL confidence floor — the
    # lab's own re-measurement agreement (replays + replications, Wilson-95
    # lower bound). Empty until the lab has enough decisive re-runs.
    try:
        from agentics.verify.calibration import calibration_suffix
        note = f"{note}{calibration_suffix(memory)}"[:600]
    except Exception:  # noqa: BLE001 — calibration must never break a settle
        pass
    memory.update_hypothesis(hyp_id, status_by_law, evidence_id=ev_id, note=note)
    # autoresearch ledger (keep/discard history of every settle) — the
    # coder-path twin of the deep-dive's record_experiment call. The ledger
    # row carries the verdict's PROVENANCE next to the verdict.
    try:
        metrics = (evidence.meta or {}).get("metrics", {}) or {}
        crit = h.criterion.as_dict()
        memory.record_experiment(
            agent, hyp_id, status_by_law, metric=str(crit.get("metric", "")),
            measured=metrics.get(str(crit.get("metric", ""))),
            op=str(crit.get("op", "")), target=crit.get("target"),
            verdict_note=str(note)[:200], protocol_path=h.script_path,
            source=method or evidence.method, description=h.claim[:200],
            self_authored_ratio=prov.ratio, anchors=prov.anchors_json())
    except Exception:  # noqa: BLE001 — ledger must never break a settle
        pass
    return ev_id


__all__ = ["register_hypothesis", "run_hypothesis", "settle_hypothesis"]
