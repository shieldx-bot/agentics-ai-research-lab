"""Owner visibility: the `agentics journal` terminal tail (complements the
web dashboard — same shared-memory journal, keyboard-only)."""

from typer.testing import CliRunner

from agentics import cli
from agentics.config import Settings
from agentics.memory.shared_memory import SharedMemory


def _mem(tmp_path) -> SharedMemory:
    s = Settings(data_dir=tmp_path / "data")
    s.ensure_dirs()
    m = SharedMemory(s.db_path)
    m.journal("a01", "claim", {"task": 3, "branch": "agent/a01/task-3-x"})
    m.journal("a02", "merge", {"mr": 9, "author": "a01"})
    m.journal("a02", "backlog_top_up", {"task": 4, "source": "survey_rotation"})
    return m


def test_journal_cli_lists_recent_entries(monkeypatch, tmp_path):
    mem = _mem(tmp_path)
    monkeypatch.setattr(cli, "_settings", lambda: _settings_for(tmp_path))
    result = CliRunner().invoke(cli.app, ["journal", "-n", "10"])
    assert result.exit_code == 0, result.output
    for needle in ("claim", "merge", "backlog_top_up", "a01", "a02"):
        assert needle in result.output
    mem.close()


def test_journal_cli_agent_filter(monkeypatch, tmp_path):
    _mem(tmp_path)
    monkeypatch.setattr(cli, "_settings", lambda: _settings_for(tmp_path))
    result = CliRunner().invoke(cli.app, ["journal", "-a", "a01"])
    assert result.exit_code == 0, result.output
    assert "claim" in result.output
    assert "backlog_top_up" not in result.output      # belongs to a02


def _settings_for(tmp_path) -> Settings:
    s = Settings(data_dir=tmp_path / "data")
    s.ensure_dirs()
    return s
