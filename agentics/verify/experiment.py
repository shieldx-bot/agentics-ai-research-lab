"""The register -> run -> settle glue for research-side hypotheses (P16).

Composes ONLY predecessor primitives — no second runner, no second verdict
logic:

* pre-registration lands the claim + structured criterion in the shared
  `hypotheses` table BEFORE the protocol runs (the swarm can see what is
  being tested while it is being tested — and the goalposts are frozen);
* `run_hypothesis` writes the protocol script into the researcher's own
  territory (research_notes/, P3) and executes it through
  `verify_metrics.run_protocol` (two runs, reproducibility gate, machine
  adjudication against the criterion);
* `settle_hypothesis` turns the run into shared knowledge: one `evidence`
  row (broadcast to every agent), protocol metrics/runs attached, and the
  hypothesis lifecycle moved to supported | refuted | inconclusive.
"""
from __future__ import annotations

from agentics.verify.hypothesis import Hypothesis
from agentics.verify_metrics import Evidence, run_protocol

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
    """Write the protocol and run it through the predecessor's runner.

    Never raises: a guard-refused write (P10/P14) becomes an inconclusive
    Evidence row with the reason, exactly like a blocked command — the swarm
    must see WHY nothing was measured."""
    try:
        workspace.write_file(h.script_path, h.code)
    except ValueError as exc:
        return Evidence(
            cmd=h.script_path, exit_code=126, verdict="inconclusive",
            output=f"protocol write refused by guard: {exc}",
            method="protocol",
            meta={"note": f"protocol write refused: {exc}",
                  "criterion": h.criterion.as_dict(), "runs": [],
                  "metrics": {}},
        )
    return run_protocol(workspace, h.script_path, h.criterion,
                        repeat=repeat, timeout=timeout)


def settle_hypothesis(memory, agent: str, hyp_id: int, h: Hypothesis,
                      evidence: Evidence) -> int:
    """Record the run as shared knowledge and close the hypothesis lifecycle.

    Returns the evidence row id. Every agent sees the verdict via the
    record_evidence broadcast (H3) and the settled claim via
    verified_knowledge() — refuted claims stop being repeated, supported
    claims become reusable common ground."""
    ev_id = memory.record_evidence(
        agent, h.task, evidence.cmd, evidence.exit_code, evidence.verdict,
        hypothesis=h.claim, method=evidence.method, output=evidence.output,
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
