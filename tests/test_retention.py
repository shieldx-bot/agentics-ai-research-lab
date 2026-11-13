"""P21 — retention: dead agents leave clean workspaces, not disk debt.

The sweep must prune ONLY when every safety condition holds: agent inactive
beyond the idle window, no active claims/locks, and a CLEAN git tree (every
commit already pushed). Anything at risk is skipped and journaled.
"""

import os
import subprocess
import time
from pathlib import Path

import pytest

from agentics.memory.shared_memory import SharedMemory
from agentics.policies.retention import sweep_stale_workspaces


def _git(path: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(path), *args],
                          check=True, capture_output=True, text=True)


def _clone(base: Path, name: str) -> Path:
    """A minimal clean git clone (no remote needed for `status --porcelain`)."""
    ws = base / name
    ws.mkdir(parents=True)
    _git(ws, "init", "-b", "main")
    (ws / "a.txt").write_text("committed\n", encoding="utf-8")
    _git(ws, "add", ".")
    _git(ws, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-m", "seed")
    import os
    old = time.time() - 24 * 3600            # make it idle: mtime yesterday
    os.utime(ws, (old, old))
    for p in ws.rglob("*"):
        os.utime(p, (old, old))
    return ws


@pytest.fixture()
def mem(tmp_path):
    m = SharedMemory(tmp_path / "r.db")
    yield m
    m.close()


def test_prunes_clean_idle_workspace_of_inactive_agent(tmp_path, mem):
    ws = _clone(tmp_path / "workspaces", "a07")
    pruned = sweep_stale_workspaces(mem, tmp_path / "workspaces", 60, [])
    assert pruned == ["a07"]
    assert not ws.exists()
    events = [e["event"] for e in mem.recent_journal(limit=5)]
    assert "workspace_pruned" in events


def test_never_touches_active_agents(tmp_path, mem):
    ws = _clone(tmp_path / "workspaces", "a07")
    pruned = sweep_stale_workspaces(
        mem, tmp_path / "workspaces", 60, [{"agent": "a07"}])
    assert pruned == [] and ws.exists()


def test_never_touches_dirty_tree(tmp_path, mem):
    """Uncommitted results must never be deleted — a stale clone is a
    nuisance, a lost result is a loss."""
    ws = _clone(tmp_path / "workspaces", "a07")
    (ws / "results.txt").write_text("uncommitted measurement\n", encoding="utf-8")
    old = time.time() - 24 * 3600            # writing the file renewed the dir mtime
    os.utime(ws, (old, old))
    pruned = sweep_stale_workspaces(mem, tmp_path / "workspaces", 60, [])
    assert pruned == [] and ws.exists()
    events = [e["event"] for e in mem.recent_journal(limit=5)]
    assert "workspace_retention_skip" in events


def test_never_touches_claim_or_lock_holders(tmp_path, mem):
    ws = _clone(tmp_path / "workspaces", "a07")
    task_id = mem.create_task("held task")
    mem.claim_issue(task_id, "a07", ttl_s=1800)
    pruned = sweep_stale_workspaces(mem, tmp_path / "workspaces", 60, [])
    assert pruned == [] and ws.exists()


def test_fresh_workspace_inside_window_is_kept(tmp_path, mem):
    ws = tmp_path / "workspaces" / "a07"
    ws.mkdir(parents=True)
    (ws / "x.txt").write_text("new\n", encoding="utf-8")
    pruned = sweep_stale_workspaces(mem, tmp_path / "workspaces", 60, [])
    assert pruned == [] and ws.exists()


def test_zero_minutes_disables_the_sweep(tmp_path, mem):
    _clone(tmp_path / "workspaces", "a07")
    pruned = sweep_stale_workspaces(mem, tmp_path / "workspaces", 0, [])
    assert pruned == []


def test_every_prune_is_audited(tmp_path, mem):
    _clone(tmp_path / "workspaces", "a07")
    sweep_stale_workspaces(mem, tmp_path / "workspaces", 60, [])
    events = [r["event"] for r in mem.audit_recent(limit=5)]
    assert "workspace_pruned" in events
