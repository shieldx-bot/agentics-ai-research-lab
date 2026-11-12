"""Workspace credential handling and command allowlist regression tests."""
import base64
import sys

from agentics.config import Settings
from agentics.github_ops.repo_workspace import RepoWorkspace


def test_git_credentials_are_not_embedded_in_remote_url_or_config(tmp_path, monkeypatch):
    token = "test-token-never-print"
    workspace = RepoWorkspace("agent-test", Settings(github_token=token, data_dir=tmp_path))

    assert token not in workspace._remote
    assert "x-access-token:" not in workspace._remote

    monkeypatch.setenv("GITHUB_TOKEN", "ambient-secret")
    env = workspace._git_env()
    assert "GITHUB_TOKEN" not in env
    index = int(env["GIT_CONFIG_COUNT"]) - 1
    header = env[f"GIT_CONFIG_VALUE_{index}"]
    encoded = header.split("basic ", 1)[1]
    assert base64.b64decode(encoded).decode() == f"x-access-token:{token}"


def test_existing_clone_remote_is_normalized_before_reuse(tmp_path, monkeypatch):
    workspace = RepoWorkspace("agent-test", Settings(
        github_token="old-secret", github_repo="owner/repo", data_dir=tmp_path
    ))
    (workspace.path / ".git").mkdir(parents=True)
    calls = []
    monkeypatch.setattr(workspace, "_git", lambda *args, **kwargs: calls.append(args))
    workspace.ensure_cloned()
    assert calls == [("remote", "set-url", "origin", "https://github.com/owner/repo.git")]


def test_command_allowlist_accepts_only_project_validation_commands(tmp_path):
    workspace = RepoWorkspace("agent-test", Settings(data_dir=tmp_path))
    ok = workspace._validate_command  # accepted commands raise nothing
    ok([sys.executable, "-m", "pytest", "-q", "-x"])
    ok([sys.executable, "-m", "py_compile", "agentics/config.py"])
    ok([sys.executable, "probe.py"])  # workspace script bound to the clone

    def rejected(*cmd) -> None:
        try:
            workspace._validate_command(list(cmd))
        except ValueError:
            return
        raise AssertionError(f"command was accepted: {cmd}")

    rejected(["bash", "-c", "echo unsafe"])
    rejected([sys.executable, "-c", "print('unsafe')"])
    rejected([sys.executable, "-m", "py_compile", "../outside.py"])
    rejected([sys.executable, "-m", "pytest", "-c", "import os"])
    rejected([sys.executable, "-m", "pytest", "--cache-dir=token-leak", "-q"])


def test_rejected_command_never_starts_subprocess(tmp_path, monkeypatch):
    workspace = RepoWorkspace("agent-test", Settings(data_dir=tmp_path))

    def forbidden(*args, **kwargs):
        raise AssertionError("subprocess must not run for a rejected command")

    monkeypatch.setattr("agentics.github_ops.repo_workspace.subprocess.Popen", forbidden)
    code, output = workspace.run_command(["bash", "-c", "echo unsafe"])
    assert code == 126
    assert "command rejected" in output
