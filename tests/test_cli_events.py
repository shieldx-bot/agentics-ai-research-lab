"""Owner visibility: the `agentics events` terminal journal reader.

Reconciliation note: a parallel `journal -f` command was folded INTO the
existing `events` command (--follow) instead of shipping two journal readers.
"""

import time as _time

from typer.testing import CliRunner

from agentics import cli
from agentics.config import Settings
from agentics.memory.shared_memory import SharedMemory


def _setup(tmp_path):
    s = Settings(data_dir=tmp_path / "data")
    s.ensure_dirs()
    m = SharedMemory(s.db_path)
    m.journal("a01", "claim", {"task": 3, "branch": "agent/a01/task-3-x"})
    m.journal("a02", "merge", {"mr": 9, "author": "a01"})
    m.journal("a02", "backlog_top_up", {"task": 4, "angle": "Benchmark design and metrics"})
    return s, m


def test_events_cli_lists_recent_entries(monkeypatch, tmp_path):
    s, mem = _setup(tmp_path)
    monkeypatch.setattr(cli, "_settings", lambda: s)
    result = CliRunner().invoke(cli.app, ["events", "-n", "10"])
    assert result.exit_code == 0, result.output
    for needle in ("claim", "merge", "backlog_top_up", "a01", "a02"):
        assert needle in result.output
    mem.close()


def test_events_cli_agent_filter(monkeypatch, tmp_path):
    s, mem = _setup(tmp_path)
    monkeypatch.setattr(cli, "_settings", lambda: s)
    result = CliRunner().invoke(cli.app, ["events", "-a", "a01"])
    assert result.exit_code == 0, result.output
    assert "claim" in result.output
    assert "backlog_top_up" not in result.output      # belongs to a02
    mem.close()


def test_events_follow_streams_then_exits_on_ctrl_c(monkeypatch, tmp_path):
    s, mem = _setup(tmp_path)
    monkeypatch.setattr(cli, "_settings", lambda: s)
    # the follow loop's first sleep raises KeyboardInterrupt -> clean exit
    monkeypatch.setattr(_time, "sleep", lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    result = CliRunner().invoke(cli.app, ["events", "-f"])
    assert result.exit_code == 0, result.output
    assert "streaming new journal entries" in result.output
    assert "journal closed" in result.output
    mem.close()


def test_events_follow_prints_entries_landed_after_start(monkeypatch, tmp_path):
    s, mem = _setup(tmp_path)
    monkeypatch.setattr(cli, "_settings", lambda: s)
    landed = {"n": 0}

    def fake_sleep(seconds):
        # first poll: one new entry lands while following; second: stop
        landed["n"] += 1
        if landed["n"] == 1:
            mem.journal("a03", "review", {"mr": 9, "verdict": "approved"})
        else:
            raise KeyboardInterrupt()

    monkeypatch.setattr(_time, "sleep", fake_sleep)
    result = CliRunner().invoke(cli.app, ["events", "-f"])
    assert result.exit_code == 0, result.output
    # the entry that landed during the stream is printed in the live section
    assert "review" in result.output.split("streaming")[1]
    mem.close()
