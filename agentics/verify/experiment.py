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
"""
from __future__ import annotations

import json

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
    if evidence.verdict == "inconclusive" and not evidence.meta.get("metrics"):
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
                      evidence: Evidence, method: str | None = None) -> int:
    """Record the run as shared knowledge and close the hypothesis lifecycle.

    Returns the evidence row id. Every agent sees the verdict via the
    record_evidence broadcast (H3) and the settled claim via
    verified_knowledge() — refuted claims stop being repeated, supported
    claims become reusable common ground.

    `method` overrides the evidence row's method label (the coder-path bridge
    records "hypothesis_contract"; the researcher protocol path keeps the
    runner's own "protocol")."""
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
    status = _STATUS_BY_VERDICT.get(evidence.verdict, "inconclusive")
    note = (evidence.meta or {}).get("note", "") or evidence.output[:200]
    memory.update_hypothesis(hyp_id, status, evidence_id=ev_id, note=note)
    return ev_id


__all__ = ["register_hypothesis", "run_hypothesis", "settle_hypothesis"]
