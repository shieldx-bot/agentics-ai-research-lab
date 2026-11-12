"""P16 — hypothesis contract + experiment runner (no LLM involved here:
the runner must judge evidence mechanically, whatever wrote the code)."""

import sys
import textwrap
from pathlib import Path

import pytest

from agentics.config import Settings
from agentics.github_ops.repo_workspace import RepoWorkspace
from agentics.verify import (
    ExperimentRunner,
    Hypothesis,
    RESULT_DIR,
    evaluate_criteria,
    hypothesis_from_json,
    parse_hypothesis,
)


# ------------------------------------------------------------- fixtures
@pytest.fixture()
def ws(tmp_path: Path):
    """A minimal local-only workspace (bare 'origin' + seed, like the real lab)."""
    import subprocess

    def git(cwd, *args):
        subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)

    bare = tmp_path / "remote.git"
    git(tmp_path, "init", "--bare", "-b", "main", str(bare))
    seed = tmp_path / "seed"
    git(tmp_path, "init", "-b", "main", str(seed))
    git(seed, "config", "user.email", "t@t")
    git(seed, "config", "user.name", "t")
    (seed / "README.md").write_text("seed\n")
    git(seed, "add", "-A")
    git(seed, "commit", "-m", "init")
    git(seed, "remote", "add", "origin", str(bare))
    git(seed, "push", "-q", "origin", "main")

    settings = Settings(github_token="x", github_repo="o/r")
    settings.data_dir = tmp_path / "data"
    w = RepoWorkspace("a02", settings)
    w._remote = str(bare)
    w.sync_main()
    return w


def _hyp(code: str, comparison: str = ">=", threshold: float = 3.0, **over) -> Hypothesis:
    base = dict(
        hypothesis_id="baseline-throughput",
        claim="The naive baseline beats the threshold",
        prediction="mean throughput stays above the threshold",
        metric="throughput",
        comparison=comparison,
        threshold=threshold,
        code=textwrap.dedent(code),
        source_finding=7,
        task=3,
    )
    base.update(over)
    return Hypothesis(**base)


# ------------------------------------------------------------- hypothesis
def test_parse_hypothesis_accepts_a_valid_proposal():
    raw = {
        "claim": "WAL mode improves write throughput",
        "prediction": "measured tps >= 100 on the synthetic workload",
        "metric": "tps",
        "comparison": ">=",
        "threshold": 100,
        "code": "print('experiment')",
        "source_finding": 4,
        "task": 9,
    }
    h, why = parse_hypothesis(raw)
    assert h is not None and why == ""
    assert h.hypothesis_id == "wal-mode-improves-write-throughput"
    assert h.script_path == f"research_notes/experiments/{h.hypothesis_id}.py"
    assert h.result_path == f"{RESULT_DIR}/{h.hypothesis_id}.json"
    assert h.threshold == 100.0 and h.source_finding == 4 and h.task == 9


def test_parse_hypothesis_rejects_malformed_proposals_with_a_reason():
    good_code = "print('x')"
    bads = [
        ({}, "claim"),
        ({"claim": "c", "prediction": "p", "metric": "m", "comparison": "~",
          "threshold": 1, "code": good_code}, "comparison"),
        ({"claim": "c", "prediction": "p", "metric": "m", "comparison": ">=",
          "threshold": "abc", "code": good_code}, "threshold"),
        ({"claim": "c", "prediction": "p", "metric": "m", "comparison": ">=",
          "threshold": 1, "code": "   "}, "code"),
        ({"claim": "c", "prediction": "p", "metric": "m", "comparison": ">=",
          "threshold": True, "code": good_code}, "threshold"),
        ({"claim": "c", "prediction": "", "metric": "m", "comparison": ">=",
          "threshold": 1, "code": good_code}, "prediction"),
    ]
    for raw, needle in bads:
        h, why = parse_hypothesis(raw)
        assert h is None, raw
        assert needle in why


def test_hypothesis_from_json_roundtrip():
    h = _hyp("print(1)")
    h2, why = hypothesis_from_json(h.to_json())
    assert h2 is not None and why == ""
    assert h2.claim == h.claim and h2.threshold == h.threshold
    broken, why = hypothesis_from_json("{not json")
    assert broken is None and "invalid" in why


# ------------------------------------------------------------- criteria
def test_evaluate_criteria_mechanical_all_operators():
    assert evaluate_criteria(4, 3, ">=") is True
    assert evaluate_criteria(2, 3, ">=") is False
    assert evaluate_criteria(2, 3, "<=") is True
    assert evaluate_criteria(3, 3, "==") is True
    assert evaluate_criteria(4, 3, ">") is True
    assert evaluate_criteria(2, 3, "<") is True
    # not-a-number can never confirm anything
    assert evaluate_criteria("4", 3, ">=") is None
    assert evaluate_criteria(True, 3, ">=") is None
    assert evaluate_criteria(None, 3, ">=") is None


# ------------------------------------------------------------- runner
def test_runner_confirmed_from_real_execution(ws):
    code = """
        import json, pathlib
        value = 21 * 2
        pathlib.Path("{result}").write_text(json.dumps({{"value": value}}))
    """.format(result=f"{RESULT_DIR}/baseline-throughput.json")
    result = ExperimentRunner(ws).run(_hyp(code))
    assert result.verdict == "confirmed", result.reason
    assert result.value == 42 and result.exit_code == 0
    assert result.duration_s >= 0
    assert (ws.path / f"{RESULT_DIR}/baseline-throughput.json").exists()


def test_runner_refuted_when_evidence_misses_the_threshold(ws):
    code = f"""
        import json, pathlib
        pathlib.Path("{RESULT_DIR}/baseline-throughput.json").write_text(
            json.dumps({{"value": 1.0}}))
    """
    result = ExperimentRunner(ws).run(_hyp(code))
    assert result.verdict == "refuted"
    assert "violates" in result.reason


def test_runner_overrides_the_script_self_report(ws):
    # The script claims success — the runner still re-evaluates the criterion.
    code = f"""
        import json, pathlib
        pathlib.Path("{RESULT_DIR}/baseline-throughput.json").write_text(
            json.dumps({{"value": 1.0, "passed": True, "note": "trust me"}}))
    """
    result = ExperimentRunner(ws).run(_hyp(code))
    assert result.verdict == "refuted"


def test_runner_inconclusive_on_crash(ws):
    result = ExperimentRunner(ws).run(_hyp("raise RuntimeError('boom')"))
    assert result.verdict == "inconclusive"
    assert "exited with code 1" in result.reason
    assert "boom" in result.evidence["exit_output_tail"]


def test_runner_inconclusive_on_missing_evidence(ws):
    result = ExperimentRunner(ws).run(_hyp("print('wrote nothing')"))
    assert result.verdict == "inconclusive"
    assert "result JSON" in result.reason


def test_runner_inconclusive_on_non_numeric_value(ws):
    code = f"""
        import json, pathlib
        pathlib.Path("{RESULT_DIR}/baseline-throughput.json").write_text(
            json.dumps({{"value": "fast"}}))
    """
    result = ExperimentRunner(ws).run(_hyp(code))
    assert result.verdict == "inconclusive"
    assert "not a number" in result.reason


def test_runner_inconclusive_on_timeout(ws):
    result = ExperimentRunner(ws, timeout_s=1).run(_hyp("import time; time.sleep(10)"))
    assert result.verdict == "inconclusive"
    assert result.exit_code == 124 and "timed out" in result.reason


def test_runner_inconclusive_when_guard_refuses_the_write(ws, monkeypatch):
    def refuse(path, content):
        raise ValueError("P10 protected path: 'agentics/' is lab code")

    monkeypatch.setattr(ws, "write_file", refuse)
    result = ExperimentRunner(ws).run(_hyp("print(1)"))
    assert result.verdict == "inconclusive"
    assert "guard" in result.reason and result.exit_code is None


def test_runner_equivalence_experiment_example(ws):
    """A real-shaped experiment: compare two algorithms on the same workload."""
    code = f"""
        import json, pathlib, random
        random.seed(0)
        data = [random.random() for _ in range(2000)]
        # "optimized": single-pass min/max vs the naive double-scan baseline
        lo, hi = data[0], data[0]
        for x in data:
            if x < lo: lo = x
            if x > hi: hi = x
        assert (min(data), max(data)) == (lo, hi)
        pathlib.Path("{RESULT_DIR}/baseline-throughput.json").write_text(
            json.dumps({{"value": 1.0, "detail": "single-pass matches naive"}}))
    """
    result = ExperimentRunner(ws).run(_hyp(code, comparison="==", threshold=1.0))
    assert result.verdict == "confirmed", result.reason
