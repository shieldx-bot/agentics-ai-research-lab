"""P25b tests — the mutation quality gate (agentics/verify/mutation.py).

"Test yếu thì bằng chứng vô giá trị": the lab sabotages a CONFIRMED
protocol with deterministic mutation operators; survived mutants mean the
verdict barely depends on the protocol's logic, so the gate downgrades
confirmed -> inconclusive (never refuted — the idea was not measured well
enough to count as confirmed, but it was not disproven either).
"""
from __future__ import annotations

import json
from pathlib import Path

from agentics.verify.falsify import case_path
from agentics.verify.mutation import (
    MutationLab, apply_mutation_gate, flip_comparison, mutants_for,
    perturb_constant, swap_add_sub, swap_boolean,
)
from agentics.verify_ops import Evidence

HYP_ID = "mut-gate-alpha"           # digit-free: mutants must bite the LOGIC,
# not the data-path string (perturb_constant skips string literals anyway)


class RunnerWorkspace:
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


# ------------------------------------------------------------ operators
def test_flip_comparison_first_occurrence_only():
    # the operator applies the exact NEGATION: not(a < b) == a >= b
    code = "if a < b:\n    hit = a <= b\n"
    assert flip_comparison(code) == "if a >= b:\n    hit = a <= b\n"
    assert flip_comparison("no comparisons") is None


def test_swap_boolean_and_perturb_constant():
    assert swap_boolean("flag = True\n") == "flag = False\n"
    assert swap_boolean("flag = False\n") == "flag = True\n"
    assert swap_boolean("nothing") is None
    assert perturb_constant("x = 0.8\n") == "x = 0.88\n"
    assert perturb_constant("x = 42\n") == "x = 46.2\n"
    assert perturb_constant("x = 'none here'\n") is None


def test_perturb_skips_numbers_inside_strings():
    # a data path is not logic: sabotaging '01' would only break a file
    # lookup — a detection without semantic content, so it is skipped
    code = ("path = 'research_notes/experiments/cases/mut-gate-01.json'\n"
            "score = 0.9\n")
    assert perturb_constant(code) == (
        "path = 'research_notes/experiments/cases/mut-gate-01.json'\n"
        "score = 0.99\n")


def test_swap_add_sub_skips_non_binary_plus():
    assert swap_add_sub("total = a + b\n") == "total = a - b\n"
    assert swap_add_sub("i += 1\n") is None           # += is not binary
    assert swap_add_sub("n = i++\n") is None          # ++ untouched
    assert swap_add_sub("a = b\n") is None


def test_mutants_deterministic_capped_and_deduped():
    code = ("if hits < limit:\n"
            "    flag = True\n"
            "    total = hits + 1\n"
            "score = 0.9\n"
            "print('AGENTICS_RESULT:', json.dumps({'score': score}))\n")
    names = [name for name, _ in mutants_for(code)]
    assert names == ["flip_comparison", "swap_boolean",
                     "perturb_constant", "swap_add_sub"]
    assert [n for n, _ in mutants_for(code)] == names     # deterministic
    assert mutants_for("print('x')\n") == []              # no mutable sites


# ------------------------------------------------------------- lab E2E
def _hypothesis(code: str, mutation: bool = True):
    from agentics.verify.hypothesis import parse_hypothesis
    raw = {
        "claim": "the protocol measures what it says it measures",
        "prediction": "verdict follows the protocol logic",
        "criterion": {"metric": "mean", "op": ">=", "target": 0.5},
        "hypothesis_id": HYP_ID,
        "code": code,
        "falsification": {
            "cases": [{"vals": [1.0, 1.0, 1.0]}],
            "relations": [],
            "mutation": mutation,
            "mutation_threshold": 0.5,
        },
    }
    hyp, reason = parse_hypothesis(raw)
    assert hyp is not None, reason
    return hyp


KILLABLE_PROTOCOL = (
    "import json\n"
    f"case = json.load(open({case_path(HYP_ID)!r}))\n"
    "vals = case['vals']\n"
    "flag = True\n"
    "mean = sum(vals) / len(vals) if flag else 0\n"
    "print('AGENTICS_RESULT:', json.dumps({'mean': mean}))\n"
)

WEAK_PROTOCOL = (
    "import json\n"
    f"case = json.load(open({case_path(HYP_ID)!r}))\n"
    "flag = True\n"                    # unused: sabotage-proof => evidence-free
    "print('AGENTICS_RESULT:', json.dumps({'mean': 0.9}))\n"
)

PURE_PROTOCOL = (
    "import json\n"
    f"case = json.load(open({case_path(HYP_ID)!r}))\n"
    "vals = case['vals']\n"
    "mean = sum(vals) / len(vals)\n"
    "print('AGENTICS_RESULT:', json.dumps({'mean': mean}))\n"
)


def _seed_case(ws) -> None:
    ws.write_file(case_path(HYP_ID),
                  json.dumps({"vals": [1.0, 1.0, 1.0]}))


def test_meaningful_protocol_scores_perfect(tmp_path):
    hyp = _hypothesis(KILLABLE_PROTOCOL)
    ws = RunnerWorkspace(tmp_path)
    _seed_case(ws)
    report = MutationLab(ws).run(hyp)
    assert report.applicable is True
    assert report.killed == report.total >= 1        # the flag matters
    assert report.score == 1.0
    ok, note = report.gate()
    assert ok is True
    assert "mutation score 1.00" in note


def test_weak_protocol_is_downgraded_to_inconclusive(tmp_path):
    hyp = _hypothesis(WEAK_PROTOCOL)
    ws = RunnerWorkspace(tmp_path)
    _seed_case(ws)
    report = MutationLab(ws).run(hyp)
    assert report.applicable is True
    assert report.score == 0.0                       # sabotage changes nothing
    ok, note = report.gate()
    assert ok is False
    assert "weak protocol" in note
    confirmed = Evidence(cmd="x", exit_code=0, verdict="confirmed",
                         ok=True, method="protocol")
    out = apply_mutation_gate(ws, hyp, confirmed)
    assert out.verdict == "inconclusive"             # downgraded, not refuted
    assert out.ok is False
    assert out.meta["mutation"]["score"] == 0.0
    assert "weak protocol" in out.output


def test_no_mutable_sites_passes_with_honest_note(tmp_path):
    # no operator can bite this protocol — the gate passes WITHOUT
    # inventing a perfect score (absence of sabotage sites is not strength)
    hyp = _hypothesis(PURE_PROTOCOL)
    ws = RunnerWorkspace(tmp_path)
    _seed_case(ws)
    report = MutationLab(ws).run(hyp)
    assert report.applicable is False
    ok, note = report.gate()
    assert ok is True
    assert "not applicable" in note
    confirmed = Evidence(cmd="x", exit_code=0, verdict="confirmed",
                         ok=True, method="protocol")
    out = apply_mutation_gate(RunnerWorkspace(tmp_path), hyp, confirmed)
    assert out.verdict == "confirmed"
    assert out.meta["mutation"]["applicable"] is False


def test_gate_passes_through_non_confirmed_and_specless():
    from agentics.verify.hypothesis import parse_hypothesis
    plain, _ = parse_hypothesis({
        "claim": "no falsification declared",
        "prediction": "p",
        "criterion": {"metric": "mean", "op": ">=", "target": 0.5},
        "code": "print('AGENTICS_RESULT:', json.dumps({'mean': 0.9}))\n",
        "hypothesis_id": "plain-hyp-01",
    })
    refuted = Evidence(cmd="x", exit_code=1, verdict="refuted")
    assert apply_mutation_gate(None, plain, refuted) is refuted
    assert apply_mutation_gate(None, plain, refuted, timeout=5) is refuted


def test_mutation_gate_error_keeps_verdict(tmp_path):
    # an exploding lab must never break a settle: verdict survives, error
    # recorded in meta
    hyp = _hypothesis(KILLABLE_PROTOCOL)
    ws = RunnerWorkspace(tmp_path)

    class Broken:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, h):  # noqa: ARG002 — signature parity with MutationLab
            raise RuntimeError("lab exploded")

    import agentics.verify.mutation as mut
    original = mut.MutationLab
    mut.MutationLab = Broken
    try:
        confirmed = Evidence(cmd="x", exit_code=0, verdict="confirmed",
                             ok=True, method="protocol")
        out = apply_mutation_gate(ws, hyp, confirmed)
    finally:
        mut.MutationLab = original
    assert out.verdict == "confirmed"
    assert "lab exploded" in out.meta["mutation"]["error"]
