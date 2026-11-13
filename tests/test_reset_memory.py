"""Topic-reset: the owner's memory wipe for switching research topics.

Hermetic — every path derives from Settings(data_dir=tmp), the only
settable root. Covers: dry-run deletes nothing; the running-agent guard
(fresh heartbeat blocks, 'stopped' heartbeat passes); the full wipe
recreates empty dirs, keeps the integrity baseline and honours --keep-repo;
a cold lab (no DB) resets cleanly.
"""
import shutil

import pytest
from typer.testing import CliRunner

from agentics import cli
from agentics.config import Settings
from agentics.memory.reset import ResetBlocked, reset_memory, running_agents
from agentics.memory.shared_memory import SharedMemory


def _seed(tmp_path, *, with_db: bool = True) -> Settings:
    s = Settings(data_dir=tmp_path / "data")
    s.ensure_dirs()
    (s.chroma_path / "index").mkdir(parents=True)
    (s.chroma_path / "index" / "vectors.bin").write_bytes(b"vectors")
    personal = s.data_dir / "personal" / "a01"
    personal.mkdir(parents=True)
    (personal / "lessons.json").write_text("[]", encoding="utf-8")
    ws = s.workspaces_dir / "a01"
    ws.mkdir(parents=True)
    (ws / "clone.txt").write_text("old topic", encoding="utf-8")
    labgit = s.local_repo_path
    labgit.mkdir(parents=True)
    (labgit / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    integrity = s.data_dir / "integrity"
    integrity.mkdir(parents=True)
    (integrity / "baseline.json").write_text("{}", encoding="utf-8")
    if with_db:
        mem = SharedMemory(s.db_path)
        mem.close()
    return s


def test_dry_run_lists_targets_and_deletes_nothing(tmp_path):
    s = _seed(tmp_path)
    report = reset_memory(s, dry_run=True)
    labels = " ".join(label for label, _ in report.removed)
    assert "shared memory" in labels and "vector memory" in labels
    assert "personal memory" in labels and "workspaces" in labels
    assert "local origin repo" in labels
    assert report.freed_bytes > 0
    # nothing actually deleted
    assert s.db_path.exists() and s.chroma_path.exists()
    assert (s.data_dir / "personal").exists() and s.workspaces_dir.exists()
    assert s.local_repo_path.exists()


def test_fresh_heartbeat_blocks_reset(tmp_path):
    s = _seed(tmp_path)
    mem = SharedMemory(s.db_path)
    mem.heartbeat("a01", "supervisor", 3, status="working")
    live = running_agents(s)
    assert [a["agent"] for a in live] == ["a01"]
    with pytest.raises(ResetBlocked):
        reset_memory(s)
    mem.close()


def test_stopped_heartbeat_passes_and_wipe_is_complete(tmp_path):
    s = _seed(tmp_path)
    mem = SharedMemory(s.db_path)
    mem.heartbeat("a01", "supervisor", 3, status="stopped")
    mem.close()
    assert running_agents(s) == []
    report = reset_memory(s)
    # everything topic-bound is gone (incl. sqlite sidecar files);
    # chroma/workspaces/personal are recreated EMPTY for the next boot
    assert not s.db_path.exists()
    assert not s.db_path.with_name(s.db_path.name + "-wal").exists()
    assert not any(s.chroma_path.iterdir())
    assert not (s.data_dir / "personal" / "a01").exists()
    assert not any(s.workspaces_dir.iterdir())
    assert not s.local_repo_path.exists()
    # lab code state survives
    assert (s.data_dir / "integrity" / "baseline.json").exists()
    # empty dirs recreated for the next boot
    assert s.chroma_path.is_dir() and s.workspaces_dir.is_dir()
    assert (s.data_dir / "personal").is_dir()
    labels = " ".join(label for label, _ in report.removed)
    assert "shared memory" in labels and "local origin repo" in labels
    assert any("integrity" in k for k in report.kept)


def test_keep_repo_preserves_lab_git(tmp_path):
    s = _seed(tmp_path)
    reset_memory(s, keep_repo=True)
    assert s.local_repo_path.exists()
    assert (s.local_repo_path / "HEAD").exists()
    assert not s.db_path.exists()


def test_cold_lab_resets_cleanly(tmp_path):
    s = _seed(tmp_path, with_db=False)
    assert running_agents(s) == []
    report = reset_memory(s)
    assert s.chroma_path.is_dir() and not any(s.chroma_path.iterdir())
    assert report.freed_bytes > 0


def test_cli_reset_memory_dry_run(monkeypatch, tmp_path):
    s = _seed(tmp_path)
    monkeypatch.setattr(cli, "_settings", lambda: s)
    result = CliRunner().invoke(cli.app, ["reset-memory", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "DRY RUN" in result.output
    assert "shared memory" in result.output
    assert s.db_path.exists()          # untouched


def test_cli_reset_memory_blocked_by_live_agent(monkeypatch, tmp_path):
    s = _seed(tmp_path)
    mem = SharedMemory(s.db_path)
    mem.heartbeat("a01", "supervisor", 3, status="working")
    mem.close()
    monkeypatch.setattr(cli, "_settings", lambda: s)
    result = CliRunner().invoke(cli.app, ["reset-memory", "--yes"])
    assert result.exit_code == 1
    assert "a01" in result.output and "stop" in result.output
    assert s.db_path.exists()          # nothing deleted


def test_cli_reset_memory_yes_wipes(monkeypatch, tmp_path):
    s = _seed(tmp_path)
    mem = SharedMemory(s.db_path)
    mem.heartbeat("a01", "supervisor", 3, status="stopped")
    mem.close()
    monkeypatch.setattr(cli, "_settings", lambda: s)
    result = CliRunner().invoke(cli.app, ["reset-memory", "--yes"])
    assert result.exit_code == 0, result.output
    assert "Memory đã reset" in result.output
    assert not s.db_path.exists()
    assert s.chroma_path.is_dir()
    assert (s.data_dir / "integrity" / "baseline.json").exists()
