"""RepoWorkspace against a real (local, bare) git remote — no GitHub involved."""

import subprocess
import sys
from pathlib import Path

import pytest

from agentics.config import Settings
from agentics.github_ops.repo_workspace import RepoWorkspace


def git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


@pytest.fixture()
def ws(tmp_path: Path):
    bare = tmp_path / "remote.git"
    git(tmp_path, "init", "--bare", "-b", "main", str(bare))
    seed = tmp_path / "seed"
    git(tmp_path, "init", "-b", "main", str(seed))
    git(seed, "config", "user.email", "t@t")
    git(seed, "config", "user.name", "t")
    (seed / "a.txt").write_text("base\n")
    git(seed, "add", "-A")
    git(seed, "commit", "-m", "init")
    git(seed, "remote", "add", "origin", str(bare))
    git(seed, "push", "origin", "main")

    settings = Settings(github_token="x", github_repo="o/r")
    settings.data_dir = tmp_path / "data"
    w = RepoWorkspace("a01", settings)
    w._remote = str(bare)  # point the workspace at the local remote
    w.sync_main()
    w.seed, w.bare = seed, bare  # test-only handles
    return w


def _push_branch(w, name, files: dict[str, str]):
    git(w.seed, "checkout", "-q", "main")
    git(w.seed, "pull", "-q", "origin", "main")
    git(w.seed, "checkout", "-q", "-B", name)
    for path, text in files.items():
        (w.seed / path).write_text(text)
    git(w.seed, "add", "-A")
    git(w.seed, "commit", "-m", name)
    git(w.seed, "push", "-q", "-f", "origin", name)


def test_changed_files_lists_each_new_file_not_the_directory(ws):
    ws.checkout_new_branch("agent/a01/x")
    ws.write_file("pkg/one.py", "x=1\n")
    ws.write_file("pkg/two.py", "y=2\n")
    ws.write_file("a.txt", "edited\n")
    assert set(ws.changed_files()) == {"a.txt", "pkg/one.py", "pkg/two.py"}


def test_verify_merge_runs_checks_on_main_plus_pr_and_cleans_up(ws):
    _push_branch(ws, "feat-ok", {"feature.txt": "hello\n"})
    probe = [
        sys.executable,
        "-c",
        "import pathlib,sys; sys.exit(0 if pathlib.Path('feature.txt').exists() else 1)",
    ]
    ok, note = ws.verify_merge("feat-ok", cmd=probe)
    assert ok, note  # the PR's file was present in the merged tree
    cur = subprocess.run(
        ["git", "branch", "--show-current"], cwd=ws.path, capture_output=True, text=True
    ).stdout.strip()
    branches = subprocess.run(["git", "branch"], cwd=ws.path, capture_output=True, text=True).stdout
    assert cur == "main" and "verify/tmp" not in branches


def test_verify_merge_reports_failing_checks(ws):
    _push_branch(ws, "feat-bad", {"feature.txt": "hello\n"})
    ok, note = ws.verify_merge("feat-bad", cmd=[sys.executable, "-c", "raise SystemExit(3)"])
    assert not ok and "exit 3" in note


def test_verify_merge_detects_conflict_with_current_main(ws):
    _push_branch(ws, "feat-conflict", {"a.txt": "from branch\n"})
    git(ws.seed, "checkout", "-q", "main")
    (ws.seed / "a.txt").write_text("from main\n")
    git(ws.seed, "add", "-A")
    git(ws.seed, "commit", "-m", "main moved on")
    git(ws.seed, "push", "-q", "origin", "main")
    ok, note = ws.verify_merge("feat-conflict", cmd=[sys.executable, "-c", "pass"])
    assert not ok and "conflict" in note
    assert (ws.path / "a.txt").read_text() == "from main\n"  # workspace left clean
