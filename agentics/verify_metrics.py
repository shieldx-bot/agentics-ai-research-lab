"""P16 — metric-based verification, composed on top of verify_ops.

verify_ops.run_experiment already turns a real exit code into a verdict
(confirmed / refuted / inconclusive). That answers "does the declared check
pass?" but a research hypothesis is usually quantitative: "Random Forest
reaches >= 0.9 precision on the demo set". An assert-based script CAN encode
that comparison, but then the criterion lives inside the generated code —
the reviewer cannot see what was actually claimed.

This module keeps the criterion OUTSIDE the protocol, as a pre-registered
`Criterion` (metric, op, target):

* the protocol script prints a final machine-readable line
  ``AGENTICS_RESULT: {"precision": 0.93, ...}``;
* `parse_metrics` extracts it from the process output (no model involved);
* `adjudicate` compares the measured number against the criterion — the
  verdict is computed from data, never from an LLM's opinion;
* `run_protocol` runs the script TWICE (a real researcher reruns the
  experiment): non-deterministic or unparsable output is `inconclusive`,
  never silently accepted.

Everything rides on verify_ops primitives (RepoWorkspace.run_command with
its allowlist, scrubbed environment and bounded lifetime) — no second
runner, no new safety surface.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from typing import Any

from agentics.verify_ops import Evidence, run_experiment

# The protocol contract: the LAST line matching this marker carries the
# measured metrics as a JSON object. Everything before it is human output.
RESULT_MARKER = "AGENTICS_RESULT:"

# ops the adjudicator understands; "between" uses target as the lower bound
# and tolerance as the upper bound (inclusive both ends).
_ALLOWED_OPS = ("<=", ">=", "<", ">", "==", "!=", "between")

_NUMBER_RE = re.compile(r"^-?\d+(?:\.\d+)?$")


@dataclass
class Criterion:
    """A pre-registered acceptance criterion for one hypothesis.

    Registered BEFORE the protocol runs (like trial pre-registration):
    the verdict is `metric <op> target` over the MEASURED value, so the
    agent cannot move the goalposts after seeing the numbers."""
    metric: str
    op: str
    target: float
    tolerance: float | None = None     # only for "between" (upper bound)

    def __post_init__(self) -> None:
        if self.op not in _ALLOWED_OPS:
            raise ValueError(
                f"unsupported op {self.op!r} — allowed: {', '.join(_ALLOWED_OPS)}")
        try:
            self.target = float(self.target)
        except (TypeError, ValueError) as exc:
            raise ValueError("criterion target must be a number") from exc
        if self.tolerance is not None:
            self.tolerance = float(self.tolerance)

    def as_dict(self) -> dict:
        d = {"metric": self.metric, "op": self.op, "target": self.target}
        if self.tolerance is not None:
            d["tolerance"] = self.tolerance
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Criterion:
        return cls(
            metric=str(d.get("metric", "")),
            op=str(d.get("op", ">=")),
            target=d.get("target"),
            tolerance=d.get("tolerance"),
        )

    def describe(self) -> str:
        if self.op == "between" and self.tolerance is not None:
            return f"{self.metric} between {self.target} and {self.tolerance}"
        return f"{self.metric} {self.op} {self.target}"


def parse_metrics(output: str) -> dict[str, Any]:
    """Extract the metrics dict from the last RESULT_MARKER line.

    Returns {} when no parsable marker line exists (protocol did not
    follow the contract) — the caller must treat that as inconclusive."""
    found: dict[str, Any] = {}
    for line in output.splitlines():
        idx = line.find(RESULT_MARKER)
        if idx == -1:
            continue
        payload = line[idx + len(RESULT_MARKER):].strip()
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            found = data
    return found


def _num(value: Any) -> float | None:
    """Coerce a measured value to float; bools and text are not numbers."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and _NUMBER_RE.match(value.strip()):
        return float(value.strip())
    return None


def adjudicate(metrics: dict[str, Any], criterion: Criterion) -> tuple[str, str]:
    """Compare the measured metric against the criterion — machine verdict.

    Returns (verdict, note):
      * supported   — the measured value satisfies the criterion;
      * refuted     — the measured value violates it;
      * inconclusive — the metric is missing / not a number (the protocol
        did not measure what the criterion asked for; a missing measurement
        must never read as success)."""
    if criterion.metric not in metrics:
        return "inconclusive", f"metric '{criterion.metric}' not measured"
    value = _num(metrics[criterion.metric])
    if value is None:
        return "inconclusive", f"metric '{criterion.metric}' is not a number"
    op, target = criterion.op, criterion.target
    if op == ">=":
        ok = value >= target
    elif op == "<=":
        ok = value <= target
    elif op == ">":
        ok = value > target
    elif op == "<":
        ok = value < target
    elif op == "==":
        ok = abs(value - target) <= 1e-9
    elif op == "!=":
        ok = abs(value - target) > 1e-9
    else:  # between
        upper = criterion.tolerance if criterion.tolerance is not None else target
        ok = target <= value <= upper
    note = f"measured {criterion.metric}={value} vs {criterion.describe()}"
    return ("supported" if ok else "refuted"), note


def run_protocol(
    workspace, script: str, criterion: Criterion,
    repeat: int = 2, timeout: int = 120,
) -> Evidence:
    """Run one protocol script `repeat` times and adjudicate the metrics.

    Verdict rules (in order — safety first):
      * any run blocked (126) / timed out (124)  -> inconclusive;
      * any run that failed to parse a RESULT    -> inconclusive;
      * runs disagree on the metrics             -> inconclusive
        (non-reproducible evidence must not become `supported`);
      * otherwise the machine verdict of adjudicate() stands.

    `script` is a workspace-relative path to an existing .py file (the
    allowlist in RepoWorkspace.run_command enforces it)."""
    runs: list[dict] = []
    output_parts: list[str] = []
    last_exit = -1
    total_duration = 0.0
    metrics: dict[str, Any] = {}
    for i in range(max(1, int(repeat))):
        ev = run_experiment(workspace, [sys.executable, script], timeout=timeout)
        last_exit = ev.exit_code
        total_duration += ev.duration_s
        output_parts.append(ev.output[-400:])
        runs.append({"exit_code": ev.exit_code, "verdict": ev.verdict})
        if ev.verdict == "inconclusive":      # timeout / blocked: stop early
            return _protocol_evidence(
                script, last_exit, "inconclusive",
                f"run {i + 1} inconclusive (exit {ev.exit_code})",
                runs, {}, output_parts, total_duration, criterion)
        parsed = parse_metrics(ev.output)
        if not parsed:
            return _protocol_evidence(
                script, last_exit, "inconclusive",
                f"run {i + 1} printed no parsable {RESULT_MARKER} line",
                runs, {}, output_parts, total_duration, criterion)
        runs[-1]["metrics"] = parsed
        metrics = parsed
    if len(runs) > 1 and any(r["metrics"] != runs[0]["metrics"] for r in runs[1:]):
        metrics = runs[0]["metrics"]
        return _protocol_evidence(
            script, last_exit, "inconclusive",
            "runs disagree — protocol is not reproducible",
            runs, {}, output_parts, total_duration, criterion)
    verdict, note = adjudicate(metrics, criterion)
    return _protocol_evidence(
        script, last_exit, verdict, note,
        runs, metrics, output_parts, total_duration, criterion)


def _protocol_evidence(script: str, exit_code: int, verdict: str, note: str,
                       runs: list[dict], metrics: dict,
                       output_parts: list[str], duration_s: float,
                       criterion: Criterion) -> Evidence:
    """The combined Evidence row for one protocol verification (never raises)."""
    combined = "\n---\n".join(output_parts)
    return Evidence(
        cmd=f"{script} ({criterion.describe()}; {len(runs)} run(s))",
        exit_code=int(exit_code),
        verdict=verdict,
        output=f"{combined}\n[{note}]"[-1500:],
        duration_s=duration_s,
        ok=verdict == "supported",
        method="protocol",
        meta={"runs": runs, "metrics": metrics, "note": note,
              "criterion": criterion.as_dict()},
    )


PROTOCOL_CONTRACT = """\
Protocol contract (the runner enforces it, violations become `inconclusive`):
- one self-contained python file, standard library only, no network, no
  reads/writes outside the current directory, deterministic output;
- compute the metric you claim, then print EXACTLY one final line:
  AGENTICS_RESULT: {"<metric>": <number>}
- the protocol must measure the metric named in the criterion — a missing
  measurement is recorded as inconclusive, never as success.
"""
