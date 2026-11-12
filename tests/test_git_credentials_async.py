"""Regression tests for safe Git authentication and async LLM execution."""

import asyncio
import base64
import subprocess
from pathlib import Path
from types import SimpleNamespace

from agentics.config import Settings
from agentics.github_ops.repo_workspace import RepoWorkspace
from agentics.graph.nodes import _llm_ask
from agentics.agents.roles import Role


def _workspace(tmp_path: Path, token: str = "test-secret") -> RepoWorkspace:
    settings = Settings(github_token=token, github_repo="owner/repo")
    settings.data_dir = tmp_path / "data"
    return RepoWorkspace("agent-1", settings)


def test_remote_url_never_contains_github_token(tmp_path):
    ws = _workspace(tmp_path)
    # local-only mode: origin is the on-disk bare repo, never a remote URL
    assert ws._remote == str(ws.settings.local_repo_path)
    assert ws._remote.endswith("lab.git")
    assert "test-secret" not in ws._remote


def test_git_auth_is_process_scoped_and_not_plaintext(tmp_path):
    ws = _workspace(tmp_path, token="distinct-secret")
    env = ws._git_env()
    assert env["GIT_CONFIG_KEY_0"] == "http.extraheader"
    expected = base64.b64encode(b"x-access-token:distinct-secret").decode("ascii")
    assert env["GIT_CONFIG_VALUE_0"] == f"AUTHORIZATION: basic {expected}"
    assert "distinct-secret" not in env["GIT_CONFIG_VALUE_0"]


def test_existing_clone_rewrites_legacy_credential_bearing_remote(tmp_path):
    ws = _workspace(tmp_path, token="new-secret")
    ws.path.mkdir(parents=True)
    subprocess.run(["git", "init", str(ws.path)], check=True, capture_output=True)
    subprocess.run(
        ["git", "remote", "add", "origin",
         "https://x-access-token:old-secret@github.com/owner/repo.git"],
        cwd=ws.path, check=True, capture_output=True,
    )
    ws.ensure_cloned()
    remote = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=ws.path, check=True, capture_output=True, text=True,
    ).stdout.strip()
    assert remote == str(ws.settings.local_repo_path)
    assert "old-secret" not in remote


def test_local_git_commands_do_not_receive_github_credentials(tmp_path, monkeypatch):
    ws = _workspace(tmp_path, token="local-command-secret")
    ws.path.mkdir(parents=True)
    seen = {}

    def fake_run(args, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    ws._git("status", check=False)
    assert "GIT_CONFIG_VALUE_0" not in seen.get("env", {})


def test_network_git_commands_receive_credentials_only_in_environment(tmp_path, monkeypatch):
    ws = _workspace(tmp_path, token="network-secret")
    ws.path.mkdir(parents=True)
    seen = {}

    def fake_run(args, **kwargs):
        seen["cmd"] = args
        seen.update(kwargs)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    ws._git("fetch", "origin", "main")
    assert "network-secret" not in " ".join(seen["cmd"])
    assert "GIT_CONFIG_VALUE_0" in seen["env"]


def test_clone_uses_clean_url_and_environment_auth(tmp_path, monkeypatch):
    ws = _workspace(tmp_path, token="clone-secret")
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        ws.path.mkdir(parents=True, exist_ok=True)
        (ws.path / ".git").mkdir(exist_ok=True)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(ws, "_git", lambda *args, **kwargs: SimpleNamespace(returncode=0))
    ws.ensure_cloned()
    args, kwargs = calls[0]
    assert args[-2] == str(ws.settings.local_repo_path)
    assert "clone-secret" not in " ".join(args)
    assert "GIT_CONFIG_VALUE_0" in kwargs["env"]


def test_async_llm_helper_awaits_provider_without_event_loop_hacks():
    async def ask(prompt, **kwargs):
        return SimpleNamespace(answer="answer", citations=["https://example.org"])

    ctx = SimpleNamespace(
        llm=SimpleNamespace(ask=ask, account=SimpleNamespace(name="account-1")),
        agent_id="agent-1",
        role=Role.CODER,
        settings=SimpleNamespace(research_topic="NFV"),
        meta={},
    )
    answer, citations = asyncio.run(_llm_ask(ctx, "prompt"))
    assert answer == "answer"
    assert citations == ["https://example.org"]
    assert ctx.meta["account"] == "account-1"
