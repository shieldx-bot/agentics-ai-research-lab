"""P25b — the mutation quality gate: test yếu thì bằng chứng vô giá trị.

Classic mutation testing asks one question about a test suite: "if the code
under test were subtly broken, would the tests NOTICE?" This module asks
the same question of a verified hypothesis, with the protocol in the role
of the test and the measurement in the role of the code under test:

    the lab SABOTAGES the protocol (built-in mutation operators — the
    mutmut idea, inlined and deterministic), re-runs it once per mutant,
    and counts how many mutants are KILLED. A mutant is killed when its
    run stops confirming the claim (refuted or inconclusive — a crashed
    measurement IS a detection). Survived mutants mean the verdict barely
    depends on the protocol's actual logic — the "evidence" would have
    said confirmed for a sabotaged measurement, so it proves nothing.

    mutation score = killed / total
    score < threshold  ->  confirmed is downgraded to INCONCLUSIVE with
    the score in the note ("weak protocol"). A downgraded claim is not a
    refutation of the IDEA — the idea simply was not measured well enough
    to count as confirmed.

Design laws:
  * the gate only ever fires on a CONFIRMED verdict (a refuted claim is
    evidence of failure and needs no quality gate; an inconclusive run is
    already honest);
  * mutation runs are OPT-IN — they cost runs, so they execute only for
    hypotheses that declared a falsification spec (spec.mutation, on by
    default);
  * everything rides the existing execution path (verify_ops.run_experiment)
    and the existing adjudicator (verify_metrics.adjudicate) — no second
    runner, no second verdict logic;
  * mutants are written into the researcher's own territory
    (research_notes/experiments/mutants/, P3) and the operator set is
    deterministic, so any agent can replay the exact mutant list;
  * no mutable sites found -> `applicable=False` and the gate passes with
    an honest note — an empty measurement script is NOT silently scored
    1.0 (that would laud the absence of anything to sabotage).

Built-in operators (first occurrence, comment lines skipped, deterministic
order, capped at MAX_MUTANTS per protocol):
    flip_comparison  <  <->  > ,  <=  <->  >=
    swap_boolean     True  <->  False
    perturb_constant the first numeric literal x -> x * 1.1
    swap_add_sub     the first binary + -> -
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from agentics.verify_metrics import adjudicate
from agentics.verify_ops import Evidence, run_experiment

MUTANTS_DIR = "research_notes/experiments/mutants"
DEFAULT_THRESHOLD = 0.5
MAX_MUTANTS = 4

_CMP_RE = re.compile(r"<=|>=|<|>")
_NUM_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_BOOL_RE = re.compile(r"\b(True|False)\b")


def _code_lines(code: str) -> list[tuple[int, str]]:
    """(index, line) pairs, skipping pure comment lines — mutations must
    not live in prose."""
    return [(i, ln) for i, ln in enumerate(code.splitlines())
            if ln.lstrip() and not ln.lstrip().startswith("#")]


def flip_comparison(code: str) -> str | None:
    """Flip the FIRST comparison operator: < <-> > and <= <-> >=."""
    for i, ln in _code_lines(code):
        m = _CMP_RE.search(ln)
        if not m:
            continue
        swap = {"<=": ">", ">=": "<", "<": ">=", ">": "<="}[m.group(0)]
        lines = code.splitlines()
        lines[i] = ln[:m.start()] + swap + ln[m.end():]
        return "\n".join(lines) + ("\n" if code.endswith("\n") else "")
    return None


def swap_boolean(code: str) -> str | None:
    """Flip the first boolean literal: True <-> False."""
    for i, ln in _code_lines(code):
        m = _BOOL_RE.search(ln)
        if not m:
            continue
        swap = "False" if m.group(1) == "True" else "True"
        lines = code.splitlines()
        lines[i] = ln[:m.start()] + swap + ln[m.end():]
        return "\n".join(lines) + ("\n" if code.endswith("\n") else "")
    return None


def _inside_string(line: str, idx: int) -> bool:
    """Best-effort string-literal guard: an odd number of quote chars
    before position idx means idx sits inside a quoted segment. Mutating
    numbers inside strings (e.g. a data path) produces mutants that only
    break file lookups — detections without semantic content."""
    prefix = line[:idx]
    return prefix.count("'") % 2 == 1 or prefix.count('"') % 2 == 1


def perturb_constant(code: str) -> str | None:
    """Scale the first numeric literal x -> x * 1.1 (0.8 -> 0.88, 42 -> 46.2).
    Numeric fragments inside quoted strings are skipped (see _inside_string)."""
    for i, ln in _code_lines(code):
        for m in _NUM_RE.finditer(ln):
            if _inside_string(ln, m.start()):
                continue
            value = float(m.group(0)) * 1.1
            text = f"{value:g}"
            lines = code.splitlines()
            lines[i] = ln[:m.start()] + text + ln[m.end():]
            return "\n".join(lines) + ("\n" if code.endswith("\n") else "")
    return None


def swap_add_sub(code: str) -> str | None:
    """Swap the first binary + into - (aggregation becomes cancellation)."""
    for i, ln in _code_lines(code):
        idx = ln.find("+")
        while idx != -1:                      # skip ++, +=, +-, "1e+5"-style
            prev = ln[idx - 1] if idx else ""
            nxt = ln[idx + 1] if idx + 1 < len(ln) else ""
            if prev not in "+-=" and nxt not in "+=":
                lines = code.splitlines()
                lines[i] = ln[:idx] + "-" + ln[idx + 1:]
                return "\n".join(lines) + ("\n" if code.endswith("\n") else "")
            idx = ln.find("+", idx + 1)
    return None


# Deterministic order: semantic sabotage first (comparisons, booleans),
# then numeric (constants, aggregation).
OPERATORS = (
    ("flip_comparison", flip_comparison),
    ("swap_boolean", swap_boolean),
    ("perturb_constant", perturb_constant),
    ("swap_add_sub", swap_add_sub),
)


def mutants_for(code: str, cap: int = MAX_MUTANTS) -> list[tuple[str, str]]:
    """The deterministic mutant list: one mutation per operator, applied to
    the ORIGINAL code (never chained), deduped by resulting text."""
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for name, op in OPERATORS:
        if len(out) >= cap:
            break
        mutated = op(code)
        if mutated is None or mutated == code or mutated in seen:
            continue
        seen.add(mutated)
        out.append((name, mutated))
    return out


# ------------------------------------------------------------------ lab
def _verdict_for(workspace, script_path: str, result_path: str, crit,
                 timeout: int) -> str:
    """Run one mutant once and return its verdict through the SHARED
    adjudicator (missing metrics -> inconclusive, never refuted)."""
    from agentics.verify.falsify import _measure   # single measurement path
    metrics, _exit, _tail = _measure(workspace, script_path, result_path,
                                     timeout, crit.metric)
    if metrics is None:
        return "inconclusive"
    verdict, _note = adjudicate(metrics, crit)
    # adjudicate speaks the HYPOTHESIS level (supported/refuted/inconclusive);
    # the gate compares against the EVIDENCE level (confirmed/...) — the
    # same single translation run_hypothesis applies to its verdicts.
    return "confirmed" if verdict == "supported" else verdict


@dataclass
class MutationReport:
    """The audit record of one mutation gate run."""
    applicable: bool
    score: float | None = None
    killed: int = 0
    total: int = 0
    threshold: float = DEFAULT_THRESHOLD
    details: list[dict] = field(default_factory=list)

    def to_meta(self) -> dict:
        return {
            "mutation": {
                "applicable": self.applicable,
                "score": self.score,
                "killed": self.killed,
                "total": self.total,
                "threshold": self.threshold,
                "details": self.details,
            }
        }

    def gate(self) -> tuple[bool, str]:
        """(gate_ok, note) — the ONE decision of this module."""
        if not self.applicable:
            return True, "mutation gate not applicable (no mutable sites)"
        if self.score >= self.threshold:
            return True, (f"mutation score {self.score:.2f} >= "
                          f"{self.threshold:.2f} ({self.killed}/{self.total} "
                          f"mutants killed)")
        return False, (f"weak protocol: mutation score {self.score:.2f} < "
                       f"{self.threshold:.2f} ({self.killed}/{self.total} "
                       f"mutants killed) — a sabotaged measurement would "
                       f"have confirmed the claim too")


class MutationLab:
    """Sabotage the protocol, re-run it, score the detections."""

    def __init__(self, workspace, timeout: int = 120,
                 threshold: float = DEFAULT_THRESHOLD):
        self.workspace = workspace
        self.timeout = max(1, int(timeout))
        self.threshold = float(threshold)

    def run(self, h) -> MutationReport:
        mutants = mutants_for(h.code)
        if not mutants:
            return MutationReport(applicable=False, threshold=self.threshold)
        killed = 0
        details: list[dict] = []
        for name, mcode in mutants:
            mpath = f"{MUTANTS_DIR}/{h.hypothesis_id}.{name}.py"
            self.workspace.write_file(mpath, mcode)
            verdict = _verdict_for(self.workspace, mpath, h.result_path,
                                   h.criterion, self.timeout)
            is_killed = verdict != "confirmed"
            killed += int(is_killed)
            details.append({"op": name, "verdict": verdict,
                            "killed": is_killed})
        total = len(mutants)
        return MutationReport(
            applicable=True, score=killed / total, killed=killed,
            total=total, threshold=self.threshold, details=details)


# ------------------------------------------------------------------ hook
def apply_mutation_gate(workspace, h, evidence: Evidence,
                        timeout: int = 120) -> Evidence:
    """Run the mutation lab against a CONFIRMED verdict; downgrade to
    inconclusive when the protocol scores below the spec's threshold.

    Non-confirmed evidence and falsification-less hypotheses pass through
    untouched. The machinery must never break a settle: an unexpected
    error keeps the original verdict with the error recorded in meta."""
    spec = getattr(h, "falsification", None)
    if evidence is None or spec is None or evidence.verdict != "confirmed":
        return evidence
    try:
        report = MutationLab(workspace, timeout=timeout,
                             threshold=spec.mutation_threshold).run(h)
    except Exception as exc:  # noqa: BLE001 — the gate must never break a settle
        meta = dict(evidence.meta or {})
        meta["mutation"] = {"error": str(exc)}
        return Evidence(
            cmd=evidence.cmd, exit_code=evidence.exit_code,
            verdict=evidence.verdict,
            output=f"{evidence.output} | mutation gate error: {exc}"[:1500],
            duration_s=evidence.duration_s, ok=evidence.ok,
            method=evidence.method, meta=meta)
    meta = dict(evidence.meta or {})
    meta.update(report.to_meta())
    ok, note = report.gate()
    out = f"{evidence.output} | mutation gate: {note}"[:1500]
    if ok:
        return Evidence(
            cmd=evidence.cmd, exit_code=evidence.exit_code,
            verdict="confirmed", output=out,
            duration_s=evidence.duration_s, ok=True,
            method=evidence.method, meta=meta)
    return Evidence(
        cmd=evidence.cmd, exit_code=evidence.exit_code,
        verdict="inconclusive", output=out,
        duration_s=evidence.duration_s, ok=False,
        method=evidence.method, meta=meta)


__all__ = [
    "DEFAULT_THRESHOLD",
    "MAX_MUTANTS",
    "MUTANTS_DIR",
    "MutationLab",
    "MutationReport",
    "OPERATORS",
    "apply_mutation_gate",
    "flip_comparison",
    "mutants_for",
    "perturb_constant",
    "swap_add_sub",
    "swap_boolean",
]
