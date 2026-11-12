"""ExperimentRunner — execute the experiment, judge it from the evidence.

This is the part that makes the swarm behave like real researchers (P16):

* the experiment runs as a workspace script through ``RepoWorkspace.run_command``,
  so it inherits every P2 guardrail (only workspace .py scripts, scrubbed
  environment — no GitHub token, no cookies — bounded lifetime, process-group
  kill on timeout). The runner adds NO new privileges of its own.
* the experiment must leave its measurement in a JSON result file
  (``research_notes/experiments/results/<id>.json``) with a numeric ``value``;
* the runner then RE-EVALUATES the success criterion from that raw value. A
  script that claims ``"passed": true`` without meeting the threshold is still
  REFUTED — the verdict is computed from evidence, never self-reported, and
  never asked of an LLM.

Verdicts:
    confirmed    exit 0 + valid result + criterion holds
    refuted      exit 0 + valid result + criterion does NOT hold
    inconclusive crash / timeout / refused write / missing or bad evidence
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from agentics.verify.hypothesis import Hypothesis

# Result directory is relative to the agent workspace root (the experiment's
# CWD) — the researcher's own P3 territory, so no ownership conflicts.
RESULT_DIR = "research_notes/experiments/results"

VERDICT_CONFIRMED = "confirmed"
VERDICT_REFUTED = "refuted"
VERDICT_INCONCLUSIVE = "inconclusive"
VERDICTS = (VERDICT_CONFIRMED, VERDICT_REFUTED, VERDICT_INCONCLUSIVE)

# Do not leak unbounded experiment output into the ledger: keep tails only.
_OUTPUT_TAIL_CHARS = 800


def evaluate_criteria(value: Any, threshold: float, comparison: str) -> bool | None:
    """Mechanically re-evaluate the criterion. Returns None when the value is
    not a real number (bools are rejected) — an incomparable value can never
    confirm anything."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if comparison == ">=":
        return value >= threshold
    if comparison == "<=":
        return value <= threshold
    if comparison == "==":
        return value == threshold
    if comparison == ">":
        return value > threshold
    if comparison == "<":
        return value < threshold
    return None


@dataclass
class VerificationResult:
    """The outcome of one experiment run — what goes into the shared ledger."""

    hypothesis_id: str
    verdict: str                  # confirmed | refuted | inconclusive
    metric: str
    value: Any = None
    threshold: float = 0.0
    comparison: str = ""
    exit_code: int | None = None
    duration_s: float = 0.0
    reason: str = ""              # why inconclusive / short human verdict note
    evidence: dict = field(default_factory=dict)

    def to_record(self) -> dict:
        """Plain-dict shape the shared-memory ledger stores (H3)."""
        return {
            "hypothesis_id": self.hypothesis_id,
            "verdict": self.verdict,
            "metric": self.metric,
            "value": self.value,
            "threshold": self.threshold,
            "comparison": self.comparison,
            "exit_code": self.exit_code,
            "duration_s": round(self.duration_s, 3),
            "reason": self.reason[:500],
            "evidence": json.dumps(self.evidence, ensure_ascii=False)[:4000],
        }


class ExperimentRunner:
    """Runs ONE hypothesis's experiment inside ONE agent's workspace."""

    def __init__(self, workspace, python: str = sys.executable,
                 timeout_s: int = 300) -> None:
        self.workspace = workspace
        self.python = python
        self.timeout_s = max(1, min(int(timeout_s), 3600))

    # ------------------------------------------------------------ run
    def run(self, h: Hypothesis) -> VerificationResult:
        started = time.monotonic()
        base = {
            "hypothesis_id": h.hypothesis_id,
            "metric": h.metric,
            "threshold": h.threshold,
            "comparison": h.comparison,
        }

        try:
            self.workspace.write_file(h.script_path, h.code)
        except ValueError as exc:          # P10/P14 guard refused the write
            return VerificationResult(
                hypothesis_id=h.hypothesis_id, verdict=VERDICT_INCONCLUSIVE,
                metric=h.metric, exit_code=None,
                threshold=h.threshold, comparison=h.comparison,
                duration_s=time.monotonic() - started,
                reason=f"experiment write refused by guard: {exc}",
                evidence={**base, "script_path": h.script_path},
            )

        # The contract only asks the script to WRITE its result file — the
        # results directory is prepared here so experiments cannot fail on a
        # missing parent dir (fewer malformed proposals to debug).
        results_root = self.workspace.path / RESULT_DIR
        try:
            results_root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return VerificationResult(
                hypothesis_id=h.hypothesis_id, verdict=VERDICT_INCONCLUSIVE,
                metric=h.metric, exit_code=None,
                threshold=h.threshold, comparison=h.comparison,
                duration_s=time.monotonic() - started,
                reason=f"cannot prepare the results directory: {exc}",
                evidence={**base, "script_path": h.script_path},
            )

        exit_code, output = self.workspace.run_command(
            [self.python, h.script_path], timeout=self.timeout_s)

        raw = self._read_result(h)
        value = raw.get("value") if isinstance(raw, dict) else None
        criterion = evaluate_criteria(value, h.threshold, h.comparison)
        duration = time.monotonic() - started

        evidence = {
            **base,
            "script_path": h.script_path,
            "result_path": h.result_path,
            "exit_output_tail": (output or "")[-_OUTPUT_TAIL_CHARS:],
            "result": raw if isinstance(raw, dict) else None,
        }

        if exit_code != 0:
            reason = ("experiment timed out" if exit_code == 124
                      else f"experiment exited with code {exit_code}")
            return VerificationResult(
                hypothesis_id=h.hypothesis_id, verdict=VERDICT_INCONCLUSIVE,
                metric=h.metric, value=value, threshold=h.threshold,
                comparison=h.comparison,
                exit_code=exit_code, duration_s=duration, reason=reason,
                evidence=evidence,
            )
        if raw is None:
            return VerificationResult(
                hypothesis_id=h.hypothesis_id, verdict=VERDICT_INCONCLUSIVE,
                metric=h.metric, value=value, threshold=h.threshold,
                comparison=h.comparison,
                exit_code=exit_code, duration_s=duration,
                reason="no readable result JSON at the contract path",
                evidence=evidence,
            )
        if criterion is None:
            return VerificationResult(
                hypothesis_id=h.hypothesis_id, verdict=VERDICT_INCONCLUSIVE,
                metric=h.metric, value=value, threshold=h.threshold,
                comparison=h.comparison,
                exit_code=exit_code, duration_s=duration,
                reason="measured value is not a number",
                evidence=evidence,
            )
        if criterion:
            return VerificationResult(
                hypothesis_id=h.hypothesis_id, verdict=VERDICT_CONFIRMED,
                metric=h.metric, value=value, threshold=h.threshold,
                comparison=h.comparison,
                exit_code=exit_code, duration_s=duration,
                reason=f"{h.metric}={value} {h.comparison} {h.threshold} holds",
                evidence=evidence,
            )
        return VerificationResult(
            hypothesis_id=h.hypothesis_id, verdict=VERDICT_REFUTED,
            metric=h.metric, value=value, threshold=h.threshold,
            comparison=h.comparison,
            exit_code=exit_code, duration_s=duration,
            reason=f"{h.metric}={value} violates {h.comparison} {h.threshold}",
            evidence=evidence,
        )

    # ------------------------------------------------------------ evidence
    def _read_result(self, h: Hypothesis) -> dict | None:
        """Read + parse the result JSON the contract demands. Any deviation
        (missing file, invalid JSON, non-object) counts as missing evidence."""
        try:
            text = self.workspace.read_file(h.result_path)
        except ValueError:
            return None
        if not text:
            return None
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            return None
        return raw if isinstance(raw, dict) else None
