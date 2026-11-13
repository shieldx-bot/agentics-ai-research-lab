"""P16 ExperimentRunner — standalone executor for a pre-registered hypothesis.

NOTE (superseded in the live cycle): node_implement now composes the
register → run → settle API (agentics/verify/experiment.py). This module
remains the reference executor with the RESULT-FILE fallback convention
and is exercised by tests/test_experiment_runner.py.

Composes the predecessor's stack (no second safety surface, no second
adjudicator):

    verify_metrics.run_protocol — runs the script twice, parses the
        AGENTICS_RESULT marker, adjudicates the pre-registered Criterion
        (machine verdict, 2-run reproducibility gate);
    RepoWorkspace.run_command   — the only execution path (allow-listed
        interpreter, scrubbed env, bounded lifetime, path-jailed).

What this module adds on top:

* it materializes the hypothesis (writes the protocol code to the
  hypothesis' script_path in the workspace — the open research_notes/ zone);
* the RESULT-FILE convention: a protocol may either print the
  ``AGENTICS_RESULT: {...}`` marker (preferred) or write its measured value
  as ``{"value": <number>}`` to ``research_notes/experiments/results/<id>.json``.
  The marker path is judged by run_protocol; when the marker is missing the
  runner falls back to the result file and adjudicates that number against
  the SAME pre-registered criterion — a script that forgets both is
  `inconclusive`, never refuted or confirmed by guesswork.

The verdict vocabulary matches verify_ops at the EVIDENCE level
(confirmed / refuted / inconclusive); the hypothesis-level `supported`
wording belongs to the shared-memory lifecycle (hypotheses.settle).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from agentics.verify.hypothesis import Hypothesis
from agentics.verify_metrics import RESULT_MARKER, adjudicate, run_protocol

log = logging.getLogger("verify.runner")

# key conventions accepted inside the result JSON file
_VALUE_KEYS = ("value", "measured", "result", "metric")


@dataclass
class RunnerResult:
    """One executed hypothesis: the machine verdict plus the raw material
    any other agent (or the owner) needs to audit the run."""

    verdict: str                     # confirmed | refuted | inconclusive
    exit_code: int | None = None
    duration_s: float = 0.0
    reason: str = ""
    evidence: str = ""               # raw output tail of the last run
    metrics: dict[str, Any] = field(default_factory=dict)
    runs: int = 0
    source: str = "marker"           # marker | result_file | none

    def to_record(self) -> dict:
        """Flat dict for the node bridge (nodes.node_implement)."""
        return {
            "exit_code": self.exit_code,
            "reason": self.reason,
            "evidence": self.evidence,
            "metrics": self.metrics,
            "runs": self.runs,
            "source": self.source,
        }


class ExperimentRunner:
    """Run one Hypothesis inside a workspace and judge it mechanically."""

    def __init__(self, workspace, repeat: int = 2, timeout: int = 120):
        self.workspace = workspace
        self.repeat = max(1, int(repeat))
        self.timeout = max(1, int(timeout))

    # ------------------------------------------------------------ execution
    def run(self, hyp: Hypothesis) -> RunnerResult:
        """Write the protocol, execute it, judge the measurement.

        Never raises for run failures — a broken run is an inconclusive
        verdict, which IS a result (the cycle must not livelock on one
        un-runnable protocol)."""
        try:
            self.workspace.write_file(hyp.script_path, hyp.code)
        except Exception as exc:  # noqa: BLE001 — protected-path/traversal etc.
            log.warning("protocol write refused for %s: %s", hyp.script_path, exc)
            return RunnerResult(
                verdict="inconclusive", exit_code=None, reason=f"protocol write refused: {exc}",
            )
        try:
            ev = run_protocol(self.workspace, hyp.script_path, hyp.criterion,
                              repeat=self.repeat, timeout=self.timeout)
        except Exception as exc:  # noqa: BLE001 — a broken runner is inconclusive
            log.warning("protocol runner error for %s: %s", hyp.hypothesis_id, exc)
            return RunnerResult(
                verdict="inconclusive", exit_code=None, reason=f"runner error: {exc}",
            )
        evidence_tail = ev.output[-1500:]
        if ev.verdict != "inconclusive":
            note = str(ev.meta.get("note", ""))
            return RunnerResult(
                verdict=ev.verdict, exit_code=ev.exit_code,
                duration_s=ev.duration_s, reason=note, evidence=evidence_tail,
                metrics=ev.meta.get("metrics", {}), runs=len(ev.meta.get("runs", [])),
                source="marker",
            )
        # Inconclusive via the marker path — try the RESULT-FILE convention
        # before giving up (a script may write its measurement to disk).
        return self._from_result_file(hyp, ev)

    # ------------------------------------------------------------ fallbacks
    def _from_result_file(self, hyp: Hypothesis, ev) -> RunnerResult:
        """Adjudicate from the ``results/<id>.json`` file the protocol wrote.

        Accepted shapes: {"value": <num>} (or measured/result/metric), or
        {<criterion.metric>: <num>}. A missing/unparsable file stays
        inconclusive with the run_protocol note attached."""
        base = RunnerResult(
            verdict="inconclusive", exit_code=ev.exit_code,
            duration_s=ev.duration_s,
            reason=str(ev.meta.get("note", "")) or f"no parsable {RESULT_MARKER} line",
            evidence=ev.output[-1500:], runs=len(ev.meta.get("runs", [])),
            source="none",
        )
        raw = None
        try:
            raw = self.workspace.read_file(hyp.result_path)
        except Exception as exc:  # noqa: BLE001 — unreadable workspace stays inconclusive
            log.warning("result file unreadable for %s: %s", hyp.hypothesis_id, exc)
        if not raw:
            return base
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return RunnerResult(**{**base.__dict__, "reason":
                                   f"result file at {hyp.result_path} is not valid JSON"})
        if not isinstance(data, dict):
            return RunnerResult(**{**base.__dict__,
                                   "reason": "result file must contain a JSON object"})
        value: Any = None
        for key in _VALUE_KEYS:
            if key in data:
                value = data[key]
                break
        else:
            value = data.get(hyp.criterion.metric)
        if value is None:
            return RunnerResult(**{**base.__dict__,
                                   "reason": f"result file carries no measurement for "
                                             f"'{hyp.criterion.metric}'"})
        verdict, note = adjudicate({hyp.criterion.metric: value}, hyp.criterion)
        # adjudicate speaks the HYPOTHESIS level (supported/refuted/inconclusive);
        # evidence rows use the verify_ops vocabulary (confirmed/refuted/...)
        verdict = "confirmed" if verdict == "supported" else verdict
        return RunnerResult(
            verdict=verdict, exit_code=ev.exit_code, duration_s=ev.duration_s,
            reason=note, evidence=f"{raw[-500:]}\n[{note}]"[-1500:],
            metrics={hyp.criterion.metric: value}, runs=len(ev.meta.get("runs", [])),
            source="result_file",
        )


__all__ = ["ExperimentRunner", "RunnerResult"]
