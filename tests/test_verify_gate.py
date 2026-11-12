"""P16 graph wiring — the plan's declared hypothesis checks run for real.

Covers the seam between the LLM's plan text and the exit-code verdicts:
`normalize_verify_cmds` (plan text -> safe argv, active interpreter only)
and the production flow node_implement uses (normalize -> run_experiments
-> record_evidence), proven against a real RepoWorkspace. Without the
normalizer, run_experiments wraps the plan's string as "/bin/sh -c", the
guard rejects it (exit 126) and EVERY check would sit at "inconclusive".
"""

import sys
from pathlib import Path

import pytest

from agentics.config import Settings
from agentics.github_ops.repo_workspace import RepoWorkspace
from agentics.graph.nodes import normalize_verify_cmds
from agentics.local_ops import LocalRepo
from agentics.memory.shared_memory import SharedMemory
from agentics.verify_ops import run_experiments, summarize


@pytest.fixture()
def lab(tmp_path: Path):
    """A real workspace clone of a tiny bare origin + shared memory."""
    settings = Settings(data_dir=tmp_path / "data")
    settings.ensure_dirs()
    src = tmp_path / "src"
    src.mkdir()
    import subprocess
    subprocess.run(["git", "init", "-b", "main", str(src)],
                   check=True, capture_output=True)
    (src / "docs").mkdir()
    (src / "docs" / "a.md").write_text("seed\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(src), "add", "."], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(src), "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-m", "seed"], check=True, capture_output=True)
    repo = LocalRepo(settings)
    repo.ensure(src=src)
    ws = RepoWorkspace("a03", settings)
    ws.ensure_cloned()
    mem = SharedMemory(tmp_path / "mem.db")
    yield ws, mem
    mem.close()


# --------------------------------------------------------------- normalize
def test_normalize_rewrites_the_interpreter():
    argv = normalize_verify_cmds(["python -m pytest tests/x.py -q"])
    assert argv == [[sys.executable, "-m", "pytest", "tests/x.py", "-q"]]


def test_normalize_drops_non_python_commands():
    cmds = ["rm -rf /", "bash -c 'evil'", "python -m pytest ok.py -q"]
    assert normalize_verify_cmds(cmds) == [[sys.executable, "-m", "pytest", "ok.py", "-q"]]


def test_normalize_accepts_argv_lists_and_dedups_caps():
    cmds = [["python", "-m", "pytest", "a.py", "-q"],
            ["python", "-m", "pytest", "a.py", "-q"],
            " ".join(["python", "s.py", *[f"a{i}" for i in range(20)]])]
    out = normalize_verify_cmds(cmds)
    assert len(out) == 2 and out[0][0] == sys.executable
    assert all(len(a) <= 12 for a in out)


# -------------------------------------------- production flow (their block)
def test_plan_string_reaches_confirmed_through_the_production_flow(lab):
    """The exact sequence node_implement runs: normalize -> run_experiments ->
    record_evidence. A plan's plain 'python check_ok.py' must CONFIRM (not
    the eternal 'inconclusive' a raw /bin/sh wrap would produce)."""
    ws, mem = lab
    (ws.path / "check_ok.py").write_text("print(40 + 2)\n", encoding="utf-8")
    evs = run_experiments(ws, normalize_verify_cmds(["python check_ok.py"]),
                          timeout=240)
    assert evs[0].verdict == "confirmed" and evs[0].exit_code == 0
    mem.record_evidence("a03", 5, evs[0].cmd, evs[0].exit_code, evs[0].verdict,
                        hypothesis="two plus two", method=evs[0].method,
                        output=evs[0].output, duration_s=evs[0].duration_s)
    (row,) = mem.evidence_for_task(5)
    assert row["verdict"] == "confirmed"


def test_raw_plan_string_without_normalizer_would_be_blocked(lab):
    """Documents the bug the normalizer fixes: the un-normalized string goes
    through /bin/sh and is refused by the guard (126 -> inconclusive)."""
    ws, mem = lab
    (ws.path / "check_ok.py").write_text("print(40 + 2)\n", encoding="utf-8")
    evs = run_experiments(ws, ["python check_ok.py"], timeout=240)
    assert evs[0].exit_code == 126 and evs[0].verdict == "inconclusive"


def test_refuted_check_is_recorded_for_everyone_to_see(lab):
    ws, mem = lab
    (ws.path / "check_bad.py").write_text("raise SystemExit(3)\n", encoding="utf-8")
    evs = run_experiments(ws, normalize_verify_cmds(["python check_bad.py"]),
                          timeout=240)
    assert evs[0].verdict == "refuted" and evs[0].exit_code == 3
    mem.record_evidence("a03", 6, evs[0].cmd, evs[0].exit_code, evs[0].verdict,
                        hypothesis="a false idea", output=evs[0].output)
    (row,) = mem.evidence_for_task(6)
    assert row["verdict"] == "refuted"
    assert "REFUTED" in f"{row['verdict'].upper()}"        # what the feed shows
    assert summarize(evs).startswith("P16 1 check(s)")
