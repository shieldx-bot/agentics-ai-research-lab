"""Tests for .env auto-loading in agentics.config (GITHUB_TOKEN etc.)."""
import os
import sys
import textwrap
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agentics import config as cfg  # noqa: E402


@pytest.fixture()
def temp_env(tmp_path, monkeypatch):
    """Point config at a throwaway project root with a .env file."""
    monkeypatch.setattr(cfg, "PROJECT_ROOT", tmp_path)
    return tmp_path / ".env"


def _clean_env(monkeypatch, keys):
    for k in keys:
        monkeypatch.delenv(k, raising=False)


def test_real_env_file_exists_and_gitignored():
    """The deployed repo must have .env gitignored and .env.example committed."""
    gitignore = (PROJECT_ROOT / ".gitignore").read_text()
    assert ".env" in gitignore
    assert (PROJECT_ROOT / ".env.example").is_file()


def test_dotenv_fills_missing_keys(temp_env, monkeypatch):
    _clean_env(monkeypatch, ["GITHUB_TOKEN", "GITHUB_REPO"])
    temp_env.write_text(
        textwrap.dedent(
            """\
            # comment line
            GITHUB_TOKEN=ghp_abc123
            GITHUB_REPO="owner/repo"
            """
        ),
        encoding="utf-8",
    )
    s = cfg.Settings()
    assert s.github_token == "ghp_abc123"
    assert s.github_repo == "owner/repo"


def test_real_environment_wins_over_dotenv(temp_env, monkeypatch):
    _clean_env(monkeypatch, ["GITHUB_TOKEN"])
    temp_env.write_text("GITHUB_TOKEN=from_file\n", encoding="utf-8")
    monkeypatch.setenv("GITHUB_TOKEN", "from_shell")
    s = cfg.Settings()
    assert s.github_token == "from_shell"


def test_quotes_and_comments_are_stripped(temp_env, monkeypatch):
    _clean_env(monkeypatch, ["HB_API_KEY"])
    temp_env.write_text(
        "HB_API_KEY = 'hb_single_quoted'  # trailing comment\n",
        encoding="utf-8",
    )
    s = cfg.Settings()
    assert s.hyperbrowser_api_key == "hb_single_quoted"


def test_missing_env_file_is_fine(temp_env, monkeypatch):
    _clean_env(monkeypatch, ["GITHUB_TOKEN"])
    # No .env written — must not raise, token just stays empty.
    s = cfg.Settings()
    assert s.github_token == ""
