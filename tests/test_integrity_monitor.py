"""P10 layer 4 — IntegrityWatcher: drift detection + self-healing workspaces."""

import asyncio
import subprocess
from pathlib import Path

import pytest

from agentics.config import Settings
from agentics.github_ops.repo_workspace import RepoWorkspace
from agentics.local_ops import LocalRepo
from agentics.memory.shared_memory import SharedMemory
from agentics.policies.guard import ProtectedPathsGuard
from agentics.policies.integrity import IntegrityWatcher


@pytest.fixture()
def env(tmp_path: Path):
    project = tmp_path / "src"
    project.mkdir()
    subprocess.run(["git", "init", "-b", "main", str(project)],
                   check=True, capture_output=True)
    (project / "agentics").mkdir()
    (project / "agentics" / "core.py").write_text("X = 1\n", encoding="utf-8")
    (project / "experiments").mkdir()
    (project / "experiments" / "keep.py").write_text("Y = 2\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(project), "add", "."],
                   check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(project), "-c", "user.email=t@t", "-c", "user.name=t",
         "commit", "-m", "seed"], check=True, capture_output=True)

    settings = Settings(data_dir=tmp_path / "data")
    settings.ensure_dirs()
    memory = SharedMemory(settings.db_path)
    LocalRepo(settings).ensure(project)
    ws = RepoWorkspace("a01", settings)
    ws.ensure_cloned()
    yield settings, memory, ws
    memory.close()


def _watcher(memory, settings) -> IntegrityWatcher:
    return IntegrityWatcher(memory, ProtectedPathsGuard(),
                            settings.workspaces_dir)


def test_snapshot_hashes_only_protected_trees(env):
    settings, memory, _ws = env
    watcher = _watcher(memory, settings)
    snap = watcher.snapshot()
    assert "a01:agentics/core.py" in snap
    assert not any(":experiments/" in key for key in snap)   # research area unwatched


def test_sweep_restores_modified_and_removes_rogue_files(env):
    settings, memory, ws = env
    watcher = _watcher(memory, settings)
    watcher.snapshot()

    (ws.path / "agentics" / "core.py").write_text("X = rogue\n", encoding="utf-8")
    (ws.path / "agentics" / "evil.py").write_text("boom\n", encoding="utf-8")
    # a legit research edit must NOT be flagged
    (ws.path / "experiments" / "keep.py").write_text("Y = 3\n", encoding="utf-8")

    drifted = watcher.sweep()
    assert "a01:agentics/core.py" in drifted
    assert "a01:agentics/evil.py" in drifted
    assert not any("experiments" in key for key in drifted)
    assert (ws.path / "agentics" / "core.py").read_text(encoding="utf-8") == "X = 1\n"
    assert not (ws.path / "agentics" / "evil.py").exists()
    assert watcher.sweep() == []


def test_sweep_restores_deleted_protected_file(env):
    settings, memory, ws = env
    watcher = _watcher(memory, settings)
    watcher.snapshot()
    (ws.path / "agentics" / "core.py").unlink()
    assert "a01:agentics/core.py" in watcher.sweep()
    assert (ws.path / "agentics" / "core.py").exists()


def test_run_journals_armed_and_drift(env):
    settings, memory, ws = env
    watcher = _watcher(memory, settings)
    (ws.path / "agentics" / "core.py").write_text("X = rogue\n", encoding="utf-8")

    async def arm_only():
        task = asyncio.create_task(watcher.run(999))   # huge interval: no auto sweep
        await asyncio.sleep(0)                          # let the arming journal land
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(arm_only())
    assert "integrity_armed" in [e["event"] for e in memory.recent_journal(limit=10)]

    # a NEW rogue file appearing after the arming is caught and journalled
    (ws.path / "agentics" / "evil.py").write_text("boom\n", encoding="utf-8")
    drifted = watcher.sweep_and_journal()
    assert "a01:agentics/evil.py" in drifted
    assert not (ws.path / "agentics" / "evil.py").exists()
    assert "integrity_drift" in [e["event"] for e in memory.recent_journal(limit=10)]
    assert any("integrity drift" in m["content"]
               for m in memory.recent_messages(limit=10))


# ------------------------------------------------- restore-war regression
def test_restore_converges_when_baseline_is_stale(env):
    """The live 2026-11-13 bug: the baseline was captured mid-clone, so it
    disagreed with git HEAD forever — the same keys were flagged and
    'restored' on every sweep for the whole run (a02 churned every 21s for
    half an hour in the owner's log). After a restore, the baseline must
    adopt the restored (HEAD) content so the loop converges."""
    settings, memory, ws = env
    watcher = _watcher(memory, settings)
    watcher.snapshot()
    # the workspace advances to a NEW commit: disk == HEAD, baseline stale
    (ws.path / "agentics" / "core.py").write_text("X = 2\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(ws.path), "add", "."],
                   check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(ws.path), "-c", "user.email=t@t", "-c", "user.name=t",
         "commit", "-m", "advance"], check=True, capture_output=True)
    drifted = watcher.sweep()
    assert "a01:agentics/core.py" in drifted        # flagged once...
    assert watcher.sweep() == []                    # ...never again
    # the restore must NOT fight the workspace's own HEAD
    assert (ws.path / "agentics" / "core.py").read_text(encoding="utf-8") == "X = 2\n"


def test_untracked_protected_file_that_vanishes_is_dropped(env):
    """A protected path that is gone AND untracked cannot be restored by git
    (checkout resurrects tracked paths only, clean only deletes) — dropping
    it from the watch set must end the sweep-after-sweep loop, journalled."""
    settings, memory, ws = env
    watcher = _watcher(memory, settings)
    (ws.path / "agentics" / "scratch.tmp").write_text("x\n", encoding="utf-8")
    watcher.snapshot()                              # scratch is in the baseline
    (ws.path / "agentics" / "scratch.tmp").unlink()
    assert "a01:agentics/scratch.tmp" in watcher.sweep()   # flagged once
    assert watcher.sweep() == []                            # ...never again
    assert "integrity_watch_dropped" in [
        e["event"] for e in memory.recent_journal(limit=10)]
