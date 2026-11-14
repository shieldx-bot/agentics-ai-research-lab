"""Owner operability: `agentics repair` runs the EXISTING maintenance services
(orphan-task repair + stale-lease sweep) on demand — previously the owner had
to wait for a Supervisor scan cycle (or restart the lab) to unstick tasks."""
from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from agentics.cli import app
from agentics.config import Settings
from agentics.memory.shared_memory import SharedMemory


def test_repair_command_unsticks_tasks_and_journals(tmp_path: Path, monkeypatch):
    settings = Settings(data_dir=tmp_path / "data")
    settings.ensure_dirs()
    monkeypatch.setattr("agentics.cli._settings", lambda: settings)
    mem = SharedMemory(settings.db_path)
    tid = mem.create_task("stuck without a lease", body="b", category="code",
                          agent="owner")
    mem.set_task_status(tid, "claimed", claimed_by="a09")   # no lease row
    mem.close()

    result = CliRunner().invoke(app, ["repair"])
    assert result.exit_code == 0, result.output
    assert f"#{tid}" in result.output

    mem = SharedMemory(settings.db_path)
    try:
        assert mem.get_task(tid)["status"] == "open"
        assert any(j["event"] == "manual_maintenance"
                   for j in mem.recent_journal(limit=10))
    finally:
        mem.close()
