"""P25 — the falsification layer: a claim is verified only if it SURVIVES.

A real researcher does not run once and conclude — they state the claim as
a universal property ("holds for EVERY input of class X") and then try to
BREAK it: varied inputs, and transformations that must not matter. This
module mechanizes exactly that on top of the existing stack (no second
runner, no second adjudicator — one vocabulary):

    verify_metrics.run_experiment — the ONLY execution path (allow-listed
        interpreter, scrubbed env, bounded lifetime);
    verify_metrics.adjudicate     — the ONLY verdict logic (machine verdict
        from the pre-registered Criterion);
    verify_metrics.parse_metrics  — the AGENTICS_RESULT contract (plus the
        result-file convention the P16 bridge honors).

How a falsifiable hypothesis is declared (opt-in field of the hypothesis
JSON contract, validated mechanically in hypothesis.parse_hypothesis):

    "falsification": {
      "cases": [ <JSON case>, ... ],            # the input class X
      "relations": [                            # metamorphic relations
        {"transform": "shuffle",      "expect": "same_within", "tolerance": 1e-9},
        {"transform": "double",       "expect": "criterion_holds"},
        {"transform": "noise",        "expect": "same_within", "tolerance": 0.05}
      ],
      "mutation": true,                          # quality gate (mutation.py)
      "mutation_threshold": 0.5
    }

Case delivery contract: the runner writes each case as JSON to
``research_notes/experiments/cases/<hypothesis_id>.json`` and runs the
protocol once per case — the protocol READS that file and prints its
AGENTS_RESULT measurement FOR THAT CASE (or writes the result JSON, the
fallback contract). run_hypothesis pre-writes the FIRST case before the
base protocol runs, so the 2-run reproducibility gate stays meaningful.

Verdict laws (Popper, mechanically):
  * one counterexample (a base case whose measurement violates the
    criterion) refutes the UNIVERSAL claim — a single case is enough;
  * one violated metamorphic relation refutes the claim too — the
    measurement changed under a transformation that must not change it
    (fragile evidence);
  * a broken/unparsable case run is inconclusive, NEVER refuted (the
    b359557 law: marker-less is inconclusive, never refuted by guesswork);
  * everything survived -> the confirmed verdict stands, enriched with the
    falsification record (cases x relations, violations = 0);
  * base runs that never vary across DISTINCT cases get the meta flag
    ``case_binding: false`` — the protocol plausibly ignored its input
    class, so the suite measured nothing about X (flagged honestly, kept
    confirmed: some claims are genuinely input-independent).

Determinism: every relation run is seeded from a stable hash of
(hypothesis_id, case, relation) so any agent can replay the exact suite —
evidence must be reproducible, not lucky.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
import sys
from dataclasses import dataclass, field
from typing import Any

from agentics.verify_metrics import (
    RESULT_MARKER, adjudicate, metrics_agree, parse_metrics,
)
from agentics.verify_ops import Evidence, run_experiment

CASES_DIR = "research_notes/experiments/cases"
MAX_CASES = 16                 # budget guard: cases x relations runs, opt-in
MAX_RELATIONS = 8

TRANSFORMS = (
    "shuffle", "double", "negate", "subset_half", "duplicate", "noise",
)
EXPECTS = ("criterion_holds", "same", "same_within")

_RESULT_KEYS = ("value", "measured", "result", "metric")


def case_path(hypothesis_id: str) -> str:
    """Where the runner parks the CURRENT case for the protocol to read."""
    return f"{CASES_DIR}/{hypothesis_id}.json"


@dataclass
class Relation:
    """One metamorphic relation: transform the case, then demand an outcome.

    expect:
      criterion_holds — the transformed run must still satisfy the
                        pre-registered Criterion (weakest, default);
      same            — the metrics dict must agree with the base run of
                        the same case (order-invariance à la metrics_agree);
      same_within     — the criterion metric must stay within `tolerance`
                        of the base run's value (robustness to noise)."""
    transform: str
    expect: str = "criterion_holds"
    tolerance: float | None = None     # required for same_within
    eps: float = 0.01                  # relative noise size (transform=noise)
    seed: int | None = None            # None -> stable hash seed

    def __post_init__(self) -> None:
        if self.transform not in TRANSFORMS:
            raise ValueError(
                f"unknown transform {self.transform!r} — allowed: "
                f"{', '.join(TRANSFORMS)}")
        if self.expect not in EXPECTS:
            raise ValueError(
                f"unknown expect {self.expect!r} — allowed: "
                f"{', '.join(EXPECTS)}")
        if self.expect == "same_within" and self.tolerance is None:
            raise ValueError("expect 'same_within' requires a tolerance")
        try:
            self.eps = float(self.eps)
        except (TypeError, ValueError) as exc:
            raise ValueError("relation eps must be a number") from exc
        if not (0.0 <= self.eps <= 1.0):
            raise ValueError("relation eps must be within [0, 1]")

    @classmethod
    def from_dict(cls, raw: object) -> "Relation":
        if not isinstance(raw, dict):
            raise ValueError("relation must be an object")
        return cls(
            transform=str(raw.get("transform", "")),
            expect=str(raw.get("expect", "criterion_holds")),
            tolerance=raw.get("tolerance"),
            eps=raw.get("eps", 0.01),
            seed=int(raw["seed"]) if isinstance(raw.get("seed"), int) else None,
        )

    def as_dict(self) -> dict:
        d: dict[str, Any] = {"transform": self.transform, "expect": self.expect}
        if self.tolerance is not None:
            d["tolerance"] = self.tolerance
        return d


@dataclass
class FalsificationSpec:
    """The declared input class X (cases) + metamorphic relations + gates."""
    cases: list[Any]
    relations: list[Relation] = field(default_factory=list)
    mutation: bool = True              # run the mutation quality gate too
    mutation_threshold: float = 0.5

    def __post_init__(self) -> None:
        if not isinstance(self.cases, list) or not self.cases:
            raise ValueError("falsification needs a non-empty cases list")
        if len(self.cases) > MAX_CASES:
            raise ValueError(f"cases exceeds the {MAX_CASES}-case budget")
        if len(self.relations) > MAX_RELATIONS:
            raise ValueError(f"relations exceeds the {MAX_RELATIONS} budget")
        try:
            self.mutation_threshold = float(self.mutation_threshold)
        except (TypeError, ValueError) as exc:
            raise ValueError("mutation_threshold must be a number") from exc
        if not (0.0 < self.mutation_threshold <= 1.0):
            raise ValueError("mutation_threshold must be within (0, 1]")

    @classmethod
    def from_dict(cls, raw: object) -> "FalsificationSpec":
        if not isinstance(raw, dict):
            raise ValueError("falsification must be an object")
        relations_raw = raw.get("relations", [])
        if not isinstance(relations_raw, list):
            raise ValueError("relations must be a list")
        return cls(
            cases=raw.get("cases"),
            relations=[Relation.from_dict(r) for r in relations_raw],
            mutation=bool(raw.get("mutation", True)),
            mutation_threshold=raw.get("mutation_threshold", 0.5),
        )

    def as_dict(self) -> dict:
        return {
            "cases": self.cases,
            "relations": [r.as_dict() for r in self.relations],
            "mutation": self.mutation,
            "mutation_threshold": self.mutation_threshold,
        }


# ------------------------------------------------------------- transforms
def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _map_numbers(x: Any, fn) -> Any:
    """Deep-copy x applying fn to every number (bools are not numbers)."""
    if _is_num(x):
        return fn(x)
    if isinstance(x, list):
        return [_map_numbers(v, fn) for v in x]
    if isinstance(x, dict):
        return {k: _map_numbers(v, fn) for k, v in x.items()}
    return copy.deepcopy(x)


def _map_lists(x: Any, fn) -> Any:
    """Deep-copy x applying fn to every list (returns the new structure)."""
    if isinstance(x, list):
        return fn(x)
    if isinstance(x, dict):
        return {k: _map_lists(v, fn) for k, v in x.items()}
    return copy.deepcopy(x)


def transform_case(case: Any, transform: str, rng: random.Random,
                   eps: float = 0.01) -> Any | None:
    """Apply one built-in metamorphic transform; None when the transform
    cannot change this case (the relation is skipped, never invented)."""
    if transform == "shuffle":
        eligible = _count_shufflable(case)
        if not eligible:
            return None
        def _shuffle(lst: list) -> list:
            out = [copy.deepcopy(v) for v in lst]
            rng.shuffle(out)
            return out
        return _map_lists(case, _shuffle)
    if transform == "double":
        if not _count_numbers(case):
            return None
        return _map_numbers(case, lambda v: v * 2)
    if transform == "negate":
        if not _count_numbers(case):
            return None
        return _map_numbers(case, lambda v: -v)
    if transform == "duplicate":
        def _dup(lst: list) -> list:
            return lst + [copy.deepcopy(v) for v in lst]
        if not _count_lists(case):
            return None
        return _map_lists(case, _dup)
    if transform == "subset_half":
        def _half(lst: list) -> list:
            keep = max(1, (len(lst) + 1) // 2)
            return [copy.deepcopy(v) for v in lst[:keep]]
        if not _count_shufflable(case):
            return None
        return _map_lists(case, _half)
    if transform == "noise":
        def _noisy(v: float) -> float:
            delta = rng.uniform(-eps, eps) * max(1.0, abs(v))
            return v + delta
        if not _count_numbers(case):
            return None
        return _map_numbers(case, _noisy)
    return None                      # unreachable (Relation validates)


def _count_numbers(x: Any) -> int:
    if _is_num(x):
        return 1
    if isinstance(x, list):
        return sum(_count_numbers(v) for v in x)
    if isinstance(x, dict):
        return sum(_count_numbers(v) for v in x.values())
    return 0


def _count_lists(x: Any) -> int:
    if isinstance(x, list):
        return 1
    if isinstance(x, dict):
        return sum(_count_lists(v) for v in x.values())
    return 0


def _count_shufflable(x: Any) -> int:
    """Lists with >= 2 elements (shuffleable / halvable)."""
    if isinstance(x, list):
        return (1 if len(x) >= 2 else 0) + sum(_count_shufflable(v) for v in x)
    if isinstance(x, dict):
        return sum(_count_shufflable(v) for v in x.values())
    return 0


def stable_seed(hypothesis_id: str, case_idx: int, rel_idx: int) -> int:
    """Replayable per-run seed: same suite on every agent, every replay."""
    digest = hashlib.sha1(
        f"{hypothesis_id}|{case_idx}|{rel_idx}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


# ------------------------------------------------------------------ runner
def _measure(workspace, script_path: str, result_path: str,
             timeout: int, metric_name: str) -> tuple[dict[str, Any] | None, int, str]:
    """Run the protocol ONCE and read its measurement (marker first, then
    the result-file convention — rekeyed to the criterion metric exactly
    like experiment.py's fallback does). Returns (metrics | None, exit, tail)."""
    ev = run_experiment(workspace, [sys.executable, script_path], timeout=timeout)
    metrics = parse_metrics(ev.output or "")
    if not metrics:
        try:
            raw = workspace.read_file(result_path)
        except (ValueError, OSError):
            raw = None
        if raw:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = None
            if isinstance(data, dict):
                for key in (metric_name, *_RESULT_KEYS):
                    v = data.get(key)
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        metrics = {metric_name: v}
                        break
    return (metrics if metrics else None), ev.exit_code, (ev.output or "")[-400:]


@dataclass
class FalsificationReport:
    """The audit record of one falsification suite run."""
    verdict: str                     # confirmed | refuted | inconclusive
    note: str
    runs: int = 0
    cases_run: int = 0
    relations_run: int = 0
    violations: list[str] = field(default_factory=list)
    case_binding: bool | None = None # None: <2 cases; False: all metrics equal

    def to_meta(self) -> dict:
        return {
            "falsification": {
                "verdict": self.verdict,
                "note": self.note,
                "runs": self.runs,
                "cases_run": self.cases_run,
                "relations_run": self.relations_run,
                "violations": self.violations,
                "case_binding": self.case_binding,
            }
        }


class FalsificationRunner:
    """Run the declared cases + metamorphic relations against ONE protocol.

    Every run rides the existing execution path (run_experiment) and the
    existing adjudicator — this module only ORCHESTRATES the loop and the
    comparisons, it never invents a verdict of its own."""

    def __init__(self, workspace, timeout: int = 120):
        self.workspace = workspace
        self.timeout = max(1, int(timeout))

    def run(self, h, spec: FalsificationSpec) -> FalsificationReport:
        script = h.script_path
        result_path = h.result_path
        cpath = case_path(h.hypothesis_id)
        crit = h.criterion
        report = FalsificationReport(verdict="confirmed", note="")
        try:
            # materialize the protocol (idempotent: run_hypothesis already
            # wrote the same source — a standalone runner call still works)
            self.workspace.write_file(script, h.code)
        except ValueError as exc:  # noqa: PERF203 — guard refusal is honest
            report.verdict = "inconclusive"
            report.note = f"protocol write refused by guard: {exc}"
            return report

        # ---- base phase: the universal claim over the whole case class
        base_metrics: list[dict[str, Any]] = []
        for idx, case in enumerate(spec.cases):
            self.workspace.write_file(cpath, json.dumps(case, ensure_ascii=False))
            metrics, exit_code, tail = _measure(
                self.workspace, script, result_path, self.timeout, crit.metric)
            report.runs += 1
            if metrics is None:
                report.verdict = "inconclusive"
                report.note = (
                    f"falsification case {idx} printed no parsable "
                    f"{RESULT_MARKER} line (exit {exit_code})")
                return report
            verdict, note = adjudicate(metrics, crit)
            if verdict == "refuted":
                report.verdict = "refuted"
                report.violations.append(f"case {idx}: {note}")
                report.note = (
                    f"counterexample: case {idx} measured "
                    f"{metrics.get(crit.metric)!r} — {crit.describe()} fails")
                report.cases_run = idx + 1
                return report
            if verdict == "inconclusive":
                report.verdict = "inconclusive"
                report.note = f"falsification case {idx}: {note}"
                report.cases_run = idx + 1
                return report
            base_metrics.append(metrics)
            report.cases_run = idx + 1

        # ---- case-binding audit: did the protocol actually consume X?
        if len(base_metrics) >= 2:
            varied = any(not metrics_agree(base_metrics[0], m)
                         for m in base_metrics[1:])
            report.case_binding = varied
            if not varied:
                report.note = ("protocol produced identical metrics across "
                               "distinct cases — the case input class was "
                               "likely not consumed (flagged, not downgraded)")

        # ---- relation phase: transformations that must not matter
        for ci, case in enumerate(spec.cases):
            for ri, rel in enumerate(spec.relations):
                rng = random.Random(rel.seed if rel.seed is not None
                                    else stable_seed(h.hypothesis_id, ci, ri))
                tcase = transform_case(case, rel.transform, rng, eps=rel.eps)
                if tcase is None or tcase == case:
                    continue                  # not applicable to this case
                self.workspace.write_file(
                    cpath, json.dumps(tcase, ensure_ascii=False))
                metrics, exit_code, _tail = _measure(
                    self.workspace, script, result_path, self.timeout, crit.metric)
                report.runs += 1
                report.relations_run += 1
                if metrics is None:
                    report.verdict = "inconclusive"
                    report.note = (f"relation {rel.transform} on case {ci} "
                                   f"printed no parsable {RESULT_MARKER} line "
                                   f"(exit {exit_code})")
                    return report
                broken = self._check_expectation(metrics, base_metrics[ci], crit, rel)
                if broken:
                    report.verdict = "refuted"
                    report.violations.append(f"case {ci}/{rel.transform}: {broken}")
                    report.note = (
                        f"metamorphic violation: {rel.transform} on case {ci} "
                        f"(expect {rel.expect}) — {broken}")
                    return report

        if not report.note:
            report.note = (f"survived {report.cases_run} case(s) x "
                           f"{report.relations_run} relation run(s), "
                           f"violations = 0")
        return report

    def _check_expectation(self, metrics: dict, base: dict,
                           crit, rel: Relation) -> str | None:
        """Return the violation text, or None when the relation holds."""
        if rel.expect == "criterion_holds":
            verdict, note = adjudicate(metrics, crit)
            if verdict == "refuted":
                return note
            if verdict == "inconclusive":
                return f"transformed run inconclusive: {note}"
            return None
        if rel.expect == "same":
            if not metrics_agree(metrics, base):
                return "metrics differ from the base run"
            return None
        # same_within: the criterion metric stays within tolerance
        bval, nval = base.get(crit.metric), metrics.get(crit.metric)
        if not (_is_num(bval) and _is_num(nval)):
            return f"metric {crit.metric!r} missing or non-numeric"
        tol = float(rel.tolerance or 0.0)
        if abs(float(nval) - float(bval)) > tol:
            return (f"{crit.metric} moved {bval} -> {nval} "
                    f"(tolerance {tol})")
        return None


# ------------------------------------------------------------------ hook
def apply_falsification(workspace, h, evidence: Evidence,
                        timeout: int = 120) -> Evidence:
    """Run the declared falsification suite against a CONFIRMED verdict.

    Only a confirmed verdict is falsified — refuted/inconclusive evidence
    passes through untouched (a failed run is not evidence OF the claim).
    The suite may downgrade:
      confirmed -> refuted      (counterexample / violated relation)
      confirmed -> inconclusive (broken case run)
    and enriches evidence.meta with the falsification record either way.
    When `spec.mutation` is on and the claim still stands, the mutation
    quality gate runs as the second line of defense (mutation.py)."""
    spec = h.falsification
    if evidence is None or spec is None or evidence.verdict != "confirmed":
        return evidence
    try:
        report = FalsificationRunner(workspace, timeout=timeout).run(h, spec)
    except Exception as exc:  # noqa: BLE001 — falsification must never break a settle
        meta = dict(evidence.meta or {})
        meta["falsification"] = {"error": str(exc)}
        return Evidence(
            cmd=evidence.cmd, exit_code=evidence.exit_code,
            verdict=evidence.verdict,
            output=f"{evidence.output} | falsification error: {exc}"[:1500],
            duration_s=evidence.duration_s, ok=evidence.ok,
            method=evidence.method, meta=meta)
    meta = dict(evidence.meta or {})
    meta.update(report.to_meta())
    if report.verdict == "confirmed":
        out = f"{evidence.output} | falsification: {report.note}"[:1500]
        stood = Evidence(
            cmd=evidence.cmd, exit_code=evidence.exit_code,
            verdict="confirmed", output=out,
            duration_s=evidence.duration_s, ok=True,
            method=evidence.method, meta=meta)
        if spec.mutation:
            # second line of defense (lazy import: no cycle at module load;
            # a missing module degrades to falsification-only, never breaks
            # a settle) — sabotage the protocol, does the verdict move?
            try:
                from agentics.verify.mutation import apply_mutation_gate
                return apply_mutation_gate(
                    workspace, h, stood, timeout=timeout)
            except Exception as exc:  # noqa: BLE001 — gate must never break a settle
                meta["mutation"] = {"error": str(exc)}
        return stood
    # downgraded — the machine verdict changed for a STATED reason
    out = (f"{evidence.output} | falsification downgraded to "
           f"{report.verdict}: {report.note}")[:1500]
    return Evidence(
        cmd=evidence.cmd, exit_code=evidence.exit_code,
        verdict=report.verdict, output=out,
        duration_s=evidence.duration_s, ok=False,
        method=evidence.method, meta=meta)


__all__ = [
    "CASES_DIR",
    "EXPECTS",
    "TRANSFORMS",
    "FalsificationReport",
    "FalsificationRunner",
    "FalsificationSpec",
    "Relation",
    "apply_falsification",
    "case_path",
    "stable_seed",
    "transform_case",
]
