"""P25 tests — the falsification layer (agentics/verify/falsify.py).

A claim is verified only if it SURVIVES its declared input class (cases)
and its metamorphic relations; one counterexample refutes the universal
claim; a broken run is inconclusive, never refuted (the b359557 law).

The suite exercises real subprocess runs through the same composition the
runner tests pin (RunnerWorkspace: write_file / read_file / run_command) —
a broken orchestration fails here, not in production.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from agentics.verify.falsify import (
    FalsificationRunner, FalsificationSpec, Relation,
    apply_falsification, case_path, stable_seed, transform_case,
)
from agentics.verify.hypothesis import parse_hypothesis

PY = "python3"


# ------------------------------------------------------------ workspace
class RunnerWorkspace:
    """Real subprocess execution in tmp_path (same shape the runner tests
    pin — write_file / read_file / run_command are all the falsify layer
    is allowed to touch)."""

    def __init__(self, root: Path):
        self.path = root

    def write_file(self, rel: str, content: str) -> None:
        f = self.path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(content, encoding="utf-8")

    def read_file(self, rel: str) -> str | None:
        f = self.path / rel
        return f.read_text(encoding="utf-8") if f.exists() else None

    def run_command(self, cmd, timeout=120):
        import subprocess
        proc = subprocess.run(cmd, cwd=self.path, capture_output=True,
                              text=True, timeout=timeout, check=False)
        return proc.returncode, (proc.stdout + proc.stderr)[-1000:]


# protocol template: reads the CURRENT case file and prints the mean
MEAN_PROTOCOL = (
    "import json\n"
    f"case = json.load(open({case_path('mean-claim-01')!r}))\n"
    "vals = case['vals'] if isinstance(case, dict) else case\n"
    "mean = sum(vals) / len(vals)\n"
    "print('AGENTICS_RESULT: ' + json.dumps({'mean': mean}))\n"
)


def _hypothesis(raw_overrides: dict | None = None, code: str | None = None):
    raw = {
        "claim": "the mean of every declared case stays >= 0.5",
        "prediction": "mean >= 0.5 for every case in the class",
        "criterion": {"metric": "mean", "op": ">=", "target": 0.5},
        "hypothesis_id": "mean-claim-01",
        "code": code or MEAN_PROTOCOL,
        "falsification": {
            "cases": [
                {"vals": [1.0, 2.0, 3.0]},
                {"vals": [2.0, 2.0, 2.0]},
                {"vals": [0.6, 0.6]},
            ],
            "relations": [
                {"transform": "shuffle", "expect": "same_within",
                 "tolerance": 1e-9},
            ],
        },
    }
    if raw_overrides:
        raw.update(raw_overrides)
    return parse_hypothesis(raw)


# ------------------------------------------------------------ spec parsing
def test_spec_parses_and_round_trips():
    hyp, reason = _hypothesis()
    assert hyp is not None, reason
    assert hyp.falsifies is True
    spec = hyp.falsification
    assert len(spec.cases) == 3
    assert spec.relations[0].transform == "shuffle"
    assert spec.mutation is True                    # default gate on
    restored = FalsificationSpec.from_dict(spec.as_dict())
    assert restored.as_dict() == spec.as_dict()


def test_spec_absent_is_none_and_contract_unchanged():
    hyp, _ = _hypothesis({"falsification": None})
    assert hyp.falsification is None
    assert hyp.falsifies is False
    assert "falsification" in hyp.to_json()          # explicit null in JSON


@pytest.mark.parametrize("bad", [
    {"cases": []},
    {"cases": "not-a-list"},
    {"cases": [{"vals": [1]}] * 17},                 # over the budget
    {"cases": [[1]], "relations": [{"transform": "teleport"}]},
    {"cases": [[1]], "relations": [{"transform": "shuffle",
                                    "expect": "identical"}]},
    {"cases": [[1]], "relations": [{"transform": "shuffle",
                                    "expect": "same_within"}]},  # no tolerance
    {"cases": [[1]], "relations": "not-a-list"},
])
def test_malformed_spec_refused_with_reason(bad):
    hyp, reason = _hypothesis({"falsification": bad})
    assert hyp is None
    assert "invalid falsification spec" in reason


# ------------------------------------------------------------- transforms
def test_transforms_are_deterministic_and_sound():
    case = {"vals": [1.0, 2.0, 3.0], "tag": "x"}
    shuffled = transform_case(case, "shuffle", random.Random(7))
    assert sorted(shuffled["vals"]) == [1.0, 2.0, 3.0]
    assert transform_case(case, "shuffle", random.Random(7)) == shuffled
    assert transform_case(case, "double", random.Random(1))["vals"] == \
        [2.0, 4.0, 6.0]
    assert transform_case(case, "negate", random.Random(1))["vals"] == \
        [-1.0, -2.0, -3.0]
    assert transform_case(case, "duplicate", random.Random(1))["vals"] == \
        [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
    assert transform_case(case, "subset_half", random.Random(1))["vals"] == \
        [1.0, 2.0]
    noisy = transform_case(case, "noise", random.Random(1), eps=0.01)
    assert all(abs(a - b) <= 0.01 * max(1.0, abs(b))
               for a, b in zip(noisy["vals"], case["vals"]))
    assert noisy["tag"] == "x"                       # non-numeric untouched


def test_transform_not_applicable_returns_none():
    scalar = 3.14
    assert transform_case(scalar, "shuffle", random.Random(1)) is None
    assert transform_case(scalar, "duplicate", random.Random(1)) is None
    assert transform_case(scalar, "subset_half", random.Random(1)) is None
    assert transform_case("no numbers here", "double", random.Random(1)) is None
    assert transform_case({"vals": [1.0]}, "subset_half",
                          random.Random(1)) is None          # len < 2


def test_stable_seed_is_replayable():
    assert stable_seed("hyp", 0, 1) == stable_seed("hyp", 0, 1)
    assert stable_seed("hyp", 0, 1) != stable_seed("hyp", 1, 0)


# ------------------------------------------------------------- runner E2E
def test_survivor_keeps_confirmed_with_record(tmp_path):
    hyp, _ = _hypothesis()
    report = FalsificationRunner(RunnerWorkspace(tmp_path)).run(
        hyp, hyp.falsification)
    assert report.verdict == "confirmed"
    assert report.cases_run == 3
    # only case 0's shuffle produces a DIFFERENT case ([2,2,2] and
    # [0.6,0.6] shuffle to themselves) — identical transforms are skipped,
    # never invented
    assert report.relations_run == 1
    assert report.runs == 4                          # 3 base + 1 relation
    assert report.violations == []
    assert report.case_binding is True               # means differ per case
    assert "survived" in report.note


def test_single_counterexample_refutes(tmp_path):
    code = MEAN_PROTOCOL
    hyp, _ = _hypothesis(
        {"falsification": {"cases": [{"vals": [0.1, 0.1, 0.1]},
                                       {"vals": [2.0, 2.0]}]}},
        code=code)
    report = FalsificationRunner(RunnerWorkspace(tmp_path)).run(
        hyp, hyp.falsification)
    assert report.verdict == "refuted"
    assert "counterexample" in report.note
    assert report.cases_run == 1                     # stopped at case 0
    assert len(report.violations) == 1


def test_violated_metamorphic_relation_refutes(tmp_path):
    # doubling the inputs must change the mean by x2 — demanding
    # same_within 1e-9 makes the relation FAIL for a mean statistic.
    hyp, _ = _hypothesis(
        {"falsification": {
            "cases": [{"vals": [1.0, 1.0, 1.0]}],
            "relations": [{"transform": "double", "expect": "same_within",
                           "tolerance": 1e-9}],
        }})
    report = FalsificationRunner(RunnerWorkspace(tmp_path)).run(
        hyp, hyp.falsification)
    assert report.verdict == "refuted"
    assert "metamorphic violation" in report.note
    assert "double" in report.violations[0]


def test_broken_case_run_is_inconclusive_never_refuted(tmp_path):
    code = "print('no marker today')\n"
    hyp, _ = _hypothesis(code=code)
    report = FalsificationRunner(RunnerWorkspace(tmp_path)).run(
        hyp, hyp.falsification)
    assert report.verdict == "inconclusive"
    assert "no parsable" in report.note


def test_case_binding_flagged_when_protocol_ignores_input(tmp_path):
    code = ("import json\n"
            "print('AGENTICS_RESULT: ' + json.dumps({'mean': 0.9}))\n")
    hyp, _ = _hypothesis(code=code)
    report = FalsificationRunner(RunnerWorkspace(tmp_path)).run(
        hyp, hyp.falsification)
    assert report.verdict == "confirmed"             # flagged, not downgraded
    assert report.case_binding is False
    assert "not consumed" in report.note


def test_result_file_contract_honored_per_case(tmp_path):
    # a protocol that prints no marker but WRITES its measurement — the
    # fallback contract the P16 bridge honors, per falsification case.
    code = (
        "import json, pathlib\n"
        f"case = json.load(open({case_path('mean-claim-01')!r}))\n"
        "vals = case['vals']\n"
        "pathlib.Path('research_notes/experiments/results').mkdir("
        "parents=True, exist_ok=True)\n"
        "pathlib.Path('research_notes/experiments/results/mean-claim-01.json')"
        ".write_text(json.dumps({'value': sum(vals) / len(vals)}))\n"
    )
    hyp, _ = _hypothesis({"falsification": {
        "cases": [{"vals": [1.0, 1.0]}, {"vals": [0.2, 0.2]}]}}, code=code)
    report = FalsificationRunner(RunnerWorkspace(tmp_path)).run(
        hyp, hyp.falsification)
    assert report.verdict == "refuted"               # case 1 mean = 0.2
    assert "counterexample" in report.note


# ------------------------------------------------------------------ hook
def test_apply_falsification_passes_non_confirmed_through(tmp_path):
    hyp, _ = _hypothesis()
    from agentics.verify_ops import Evidence
    refuted = Evidence(cmd="x", exit_code=1, verdict="refuted")
    assert apply_falsification(RunnerWorkspace(tmp_path), hyp,
                               refuted) is refuted
    assert apply_falsification(RunnerWorkspace(tmp_path), hyp,
                               refuted, timeout=5) is refuted


def test_apply_falsification_downgrades_on_violation(tmp_path):
    hyp, _ = _hypothesis({
        "falsification": {
            "cases": [{"vals": [0.1, 0.1, 0.1]}],        # mean 0.1 < 0.5
            "relations": [],
            "mutation": False,                            # isolate the layer
        }})
    from agentics.verify_ops import Evidence
    confirmed = Evidence(cmd="x", exit_code=0, verdict="confirmed",
                         ok=True, method="protocol",
                         meta={"metrics": {"mean": 0.9}})
    out = apply_falsification(RunnerWorkspace(tmp_path), hyp, confirmed)
    assert out.verdict == "refuted"
    assert out.ok is False
    assert out.meta["falsification"]["verdict"] == "refuted"
    assert "counterexample" in out.output


def test_apply_falsification_enriches_survivor_meta(tmp_path):
    hyp, _ = _hypothesis({"falsification": {
        "cases": [{"vals": [1.0, 1.0, 1.0]}],
        "relations": [],
        "mutation": False,
    }})
    from agentics.verify_ops import Evidence
    confirmed = Evidence(cmd="x", exit_code=0, verdict="confirmed",
                         ok=True, method="protocol")
    out = apply_falsification(RunnerWorkspace(tmp_path), hyp, confirmed)
    assert out.verdict == "confirmed"
    assert out.meta["falsification"]["violations"] == []
    assert "falsification:" in out.output


# ------------------------------------------------- run_hypothesis wiring
def test_run_hypothesis_end_to_end_survivor(tmp_path):
    """The live path: case 0 parked BEFORE the base run (the 2-run gate
    measures the same case), the suite re-measures the whole class, the
    mutation gate runs last — one confirmed verdict, fully audited."""
    from agentics.verify.experiment import run_hypothesis
    hyp, _ = _hypothesis()
    ev = run_hypothesis(RunnerWorkspace(tmp_path), hyp, repeat=2, timeout=60)
    assert ev.verdict == "confirmed"
    assert ev.meta["falsification"]["violations"] == []
    assert ev.meta["falsification"]["case_binding"] is True
    assert ev.meta["mutation"]["applicable"] is True      # gate ran too
    assert "falsification:" in ev.output


def test_run_hypothesis_end_to_end_counterexample_refutes(tmp_path):
    """Base adjudication catches the counterexample in case 0 itself —
    the falsification layer never touches a non-confirmed verdict."""
    from agentics.verify.experiment import run_hypothesis
    hyp, _ = _hypothesis({"falsification": {
        "cases": [{"vals": [0.2, 0.2]}, {"vals": [1.0, 1.0]}],
        "relations": [], "mutation": False}})
    ev = run_hypothesis(RunnerWorkspace(tmp_path), hyp, repeat=2, timeout=60)
    assert ev.verdict == "refuted"
    assert "0.2" in ev.output
