"""P16 — ExperimentRunner turns real exit codes into structured verdicts.

The runner is a thin, honest adapter over RepoWorkspace.run_command: no model
opinion involved — `confirmed` can ONLY mean "the declared check exited 0".
These tests also pin the safety contract: a blocked command (allowlist
rejection) is `inconclusive`, never a fake `confirmed`.
"""
from __future__ import annotations

import sys
from pathlib import Path

from agentics.config import Settings
from agentics.github_ops.repo_workspace import RepoWorkspace
from agentics.verify_ops import (
    Evidence, run_experiment, run_experiments, summarize,
)

PY = sys.executable


def _workspace(tmp_path: Path) -> RepoWorkspace:
    settings = Settings(data_dir=tmp_path / "data")
    settings.ensure_dirs()
    ws = RepoWorkspace("a01", settings)
    ws.path.mkdir(parents=True, exist_ok=True)
    return ws


def _script(ws: RepoWorkspace, code: str, name: str = "exp.py") -> None:
    (ws.path / name).write_text(code)


def test_confirmed_comes_from_a_zero_exit(tmp_path):
    ws = _workspace(tmp_path)
    _script(ws, "import sys\nsys.exit(0)\n")
    ev = run_experiment(ws, [PY, "exp.py"])
    assert ev.verdict == "confirmed" and ev.ok and ev.exit_code == 0
    assert ev.duration_s > 0
    assert ev.cmd.startswith(PY)


def test_refuted_comes_from_a_nonzero_exit(tmp_path):
    ws = _workspace(tmp_path)
    _script(ws, "import sys\nsys.exit(3)\n")
    ev = run_experiment(ws, [PY, "exp.py"])
    assert ev.verdict == "refuted" and not ev.ok and ev.exit_code == 3


def test_timeout_is_inconclusive_not_refuted(tmp_path):
    ws = _workspace(tmp_path)
    _script(ws, "import time\ntime.sleep(5)\n")
    ev = run_experiment(ws, [PY, "exp.py"], timeout=1)
    assert ev.exit_code == 124 and ev.verdict == "inconclusive"


def test_blocked_command_is_inconclusive_never_confirmed(tmp_path):
    ws = _workspace(tmp_path)
    ev = run_experiment(ws, [PY, "-m", "pytest", "--token=secret"])
    assert ev.exit_code == 126 and ev.verdict == "inconclusive"
    assert "rejected" in ev.output


def test_output_is_bounded_in_the_row(tmp_path):
    ws = _workspace(tmp_path)
    _script(ws, "print('x' * 20000)\n")
    ev = run_experiment(ws, [PY, "exp.py"])
    row = ev.as_row()
    assert len(row["output"]) <= 1500
    assert row["cmd"] == ev.cmd and row["verdict"] == "confirmed"


def test_run_experiments_never_raises(tmp_path):
    ws = _workspace(tmp_path)
    _script(ws, "print('ok')\n")
    results = run_experiments(ws, [
        [PY, "exp.py"],                                  # confirmed
        [PY, "-m", "pytest", "--cookie=junk"],           # blocked
        "not-a-list-command",                            # non-python head (no shell)
    ])
    assert [e.verdict for e in results] == \
        ["confirmed", "inconclusive", "inconclusive"]
    # a non-python command is rejected BEFORE any shell exists — no crash,
    # just honest evidence about the sandbox contract
    assert results[2].exit_code == 126
    assert "rejected" in results[2].output


# ------------------------------------------------------------ string cmds (plan JSON)
# Regression: plans are LLM JSON, so verify_cmds arrive as STRINGS. The old
# runner wrapped them in /bin/sh -c, which the workspace allowlist (active
# python only) always rejected — every declared check silently became
# `inconclusive` in production. P16 was a no-op on real workspaces.
def test_string_verify_cmd_from_a_plan_actually_runs(tmp_path):
    ws = _workspace(tmp_path)
    _script(ws, "import sys\nsys.exit(0)\n")
    ev = run_experiment(ws, "python exp.py")
    assert ev.verdict == "confirmed" and ev.exit_code == 0
    assert ev.cmd.startswith(PY) and "exp.py" in ev.cmd


def test_string_pytest_cmd_is_coerced_to_the_module_form(tmp_path):
    from agentics.verify_ops import coerce_cmd
    argv, reason = coerce_cmd("python -m pytest tests/test_x.py -q")
    assert argv is not None and argv[:3] == [PY, "-m", "pytest"]
    argv, _ = coerce_cmd("pytest tests/test_x.py")
    assert argv[:3] == [PY, "-m", "pytest"]


def test_string_command_never_uses_a_shell(tmp_path):
    ws = _workspace(tmp_path)
    ev = run_experiment(ws, "rm -rf /")            # catastrophic under sh -c
    assert ev.verdict == "inconclusive" and ev.exit_code == 126
    assert "rejected" in ev.output and "python" in ev.output


def test_non_string_non_list_command_is_rejected(tmp_path):
    ws = _workspace(tmp_path)
    ev = run_experiment(ws, 42)
    assert ev.verdict == "inconclusive" and ev.exit_code == 126


def test_summarize_digest(tmp_path):
    assert summarize([]) == ""
    evs = [Evidence(cmd="a", exit_code=0, verdict="confirmed", ok=True),
           Evidence(cmd="b", exit_code=1, verdict="refuted"),
           Evidence(cmd="b", exit_code=1, verdict="refuted")]
    digest = summarize(evs)
    assert "3 check(s)" in digest and "refuted:2" in digest and "-> refuted" in digest
    all_ok = [Evidence(cmd="a", exit_code=0, verdict="confirmed", ok=True)]
    assert "-> confirmed" in summarize(all_ok)
