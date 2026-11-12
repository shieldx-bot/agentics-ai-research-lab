"""Workspace credential handling and command allowlist regression tests."""
import base64
import sys

from agentics.config import Settings
from agentics.github_ops.repo_workspace import RepoWorkspace


def test_git_credentials_are_not_embedded_in_remote_url_or_config(tmp_path):
    token = "test-token-never-print"
    workspace = RepoWorkspace("agent-test", Settings(github_token=token, data_dir=tmp_path))

    assert token not in workspace._remote
    assert "x-access-token:" not in workspace._remote

    env = workspace._git_env()
    index = int(env["GIT_CONFIG_COUNT"]) - 1
    header = env[f"GIT_CONFIG_VALUE_{index}"]
    encoded = header.split("basic ", 1)[1]
    assert base64.b64decode(encoded).decode() == f"x-access-token:{token}"


def test_command_allowlist_accepts_only_project_validation_commands(tmp_path):
    workspace = RepoWorkspace("agent-test", Settings(data_dir=tmp_path))
    assert workspace._command_allowed([sys.executable, "-m", "pytest", "-q", "-x"])
    assert workspace._command_allowed([sys.executable, "-m", "py_compile", "agentics/config.py"])
    assert not workspace._command_allowed(["bash", "-c", "echo unsafe"])
    assert not workspace._command_allowed([sys.executable, "-c", "print('unsafe')"])
    assert not workspace._command_allowed([
        sys.executable, "-m", "py_compile", "../outside.py"
    ])
    assert not workspace._command_allowed([
        sys.executable, "-m", "pytest", "--pyargs", "os"
    ])


def test_rejected_command_never_starts_subprocess(tmp_path, monkeypatch):
    workspace = RepoWorkspace("agent-test", Settings(data_dir=tmp_path))

    def forbidden(*args, **kwargs):
        raise AssertionError("subprocess must not run for a rejected command")

    monkeypatch.setattr("agentics.github_ops.repo_workspace.subprocess.run", forbidden)
    code, output = workspace.run_command(["bash", "-c", "echo unsafe"])
    assert code == 126
    assert "allowlist" in output
