"""Unit tests for fail-closed rootless Docker execution (no daemon required)."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentics.config import Settings
from agentics.github_ops.sandbox import DockerSandboxRunner, SandboxError
from agentics.github_ops.repo_workspace import RepoWorkspace


def _settings(tmp_path: Path) -> Settings:
    settings = Settings(data_dir=tmp_path / "data")
    return settings


def test_docker_command_enforces_isolation_and_resource_limits(tmp_path):
    settings = _settings(tmp_path)
    runner = DockerSandboxRunner(settings)
    command = runner.build_command(
        "/usr/bin/docker", tmp_path / "workspace",
        ["/usr/bin/python3", "-m", "pytest", "-q"], tmp_path / "run.cid",
    )
    joined = " ".join(command)
    assert "--network=none" in command
    assert "--read-only" in command
    assert "--cap-drop=ALL" in command
    assert "--security-opt=no-new-privileges:true" in command
    assert "--pids-limit=128" in command
    assert "--memory=1024m" in command
    assert "--cpus=1.000" in command
    assert "--mount" in command and "target=/workspace" in joined
    assert command[-4:] == ["python", "-m", "pytest", "-q"]
    assert "GITHUB_TOKEN" not in joined and "PPLX_COOKIE" not in joined


def test_docker_command_rejects_invalid_resource_limits(tmp_path):
    settings = _settings(tmp_path)
    settings.sandbox_memory_mb = 64
    runner = DockerSandboxRunner(settings)
    with pytest.raises(SandboxError, match="SANDBOX_MEMORY_MB"):
        runner.build_command(
            "docker", tmp_path / "workspace", ["/python", "probe.py"], tmp_path / "x.cid"
        )


def test_preflight_rejects_non_rootless_daemon(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    runner = DockerSandboxRunner(settings)
    monkeypatch.setattr("agentics.github_ops.sandbox.shutil.which", lambda _: "/usr/bin/docker")
    monkeypatch.setattr(
        "agentics.github_ops.sandbox.subprocess.run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout='["name=seccomp"]', stderr=""),
    )
    with pytest.raises(SandboxError, match="not rootless"):
        runner._preflight()


def test_preflight_rejects_missing_local_image(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    runner = DockerSandboxRunner(settings)
    monkeypatch.setattr("agentics.github_ops.sandbox.shutil.which", lambda _: "/usr/bin/docker")
    calls = iter([
        SimpleNamespace(returncode=0, stdout='["name=rootless"]', stderr=""),
        SimpleNamespace(returncode=1, stdout="", stderr="image not found"),
    ])
    monkeypatch.setattr(
        "agentics.github_ops.sandbox.subprocess.run", lambda *a, **k: next(calls)
    )
    with pytest.raises(SandboxError, match="not available locally"):
        runner._preflight()


def test_workspace_fails_closed_when_sandbox_is_unavailable(tmp_path, monkeypatch):
    settings = Settings(
        github_token="", github_repo="owner/repo", data_dir=tmp_path / "data",
        sandbox_enabled=True,
    )
    workspace = RepoWorkspace("agent-1", settings)
    workspace.path.mkdir(parents=True)
    workspace.write_file("probe.py", "print('should not run')\\n")

    def refuse(*args, **kwargs):
        raise SandboxError("Docker daemon unavailable")

    monkeypatch.setattr(DockerSandboxRunner, "run", refuse)
    code, output = workspace.run_command(["/usr/local/bin/python", "probe.py"])
    assert code == 126
    assert "execution refused" in output
