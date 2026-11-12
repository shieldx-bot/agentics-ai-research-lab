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

    settings = Settings(github_token="x", github_repo="o/r", sandbox_enabled=False)
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
    _push_branch(ws, "feat-ok", {
        "feature.txt": "hello\n",
        "probe.py": "import pathlib, sys; sys.exit(0 if pathlib.Path('feature.txt').exists() else 1)\n",
    })
    ok, note = ws.verify_merge("feat-ok", cmd=[sys.executable, "probe.py"])
    assert ok, note  # the PR's file was present in the merged tree
    cur = subprocess.run(
        ["git", "branch", "--show-current"], cwd=ws.path, capture_output=True, text=True
    ).stdout.strip()
    branches = subprocess.run(["git", "branch"], cwd=ws.path, capture_output=True, text=True).stdout
    assert cur == "main" and "verify/tmp" not in branches


def test_verify_merge_reports_failing_checks(ws):
    _push_branch(ws, "feat-bad", {
        "feature.txt": "hello\n",
        "fail.py": "raise SystemExit(3)\n",
    })
    ok, note = ws.verify_merge("feat-bad", cmd=[sys.executable, "fail.py"])
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



def test_workspace_rejects_path_traversal(ws):
    with pytest.raises(ValueError):
        ws.write_file("../outside.txt", "nope")
    with pytest.raises(ValueError):
        ws.read_file("/etc/passwd")


def test_workspace_rejects_symlink_escape(ws, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (ws.path / "escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        ws.write_file("escape/pwned.txt", "nope")


def test_run_command_strips_credentials_from_environment(ws, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "must-not-leak")
    monkeypatch.setenv("PPLX_COOKIE", "must-not-leak")
    ws.write_file("env_probe.py", (
        "import os, sys\n"
        "sys.exit(1 if os.getenv('GITHUB_TOKEN') or os.getenv('PPLX_COOKIE') else 0)\n"
    ))
    code, output = ws.run_command([sys.executable, "env_probe.py"])
    assert code == 0, output


def test_run_command_rejects_inline_python(ws):
    code, output = ws.run_command([sys.executable, "-c", "raise SystemExit(0)"])
    assert code == 126 and "rejected" in output


def test_run_command_kills_timed_out_process(ws):
    ws.write_file("sleep.py", "import time; time.sleep(10)\n")
    code, output = ws.run_command([sys.executable, "sleep.py"], timeout=1)
    assert code == 124 and "timeout" in output


def test_agent_id_cannot_escape_workspace(ws):
    with pytest.raises(ValueError):
        RepoWorkspace("../outside", ws.settings)


def test_git_network_environment_contains_only_git_auth_not_provider_cookies(ws, monkeypatch):
    monkeypatch.setenv("PPLX_COOKIE", "must-not-leak")
    env = ws._git_env()
    assert "PPLX_COOKIE" not in env
    assert env["GIT_CONFIG_KEY_0"] == "http.extraheader"
    assert "must-not-leak" not in repr(env)
