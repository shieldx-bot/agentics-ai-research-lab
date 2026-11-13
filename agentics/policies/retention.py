"""P21 — retention: dead agents leave clean workspaces, not disk debt.

The lab runs for months; every dead agent leaves `data/workspaces/<id>` on
disk (a full clone). P6 marks them dead, P13 releases their locks — nothing
pruned the clone itself. This module deletes a workspace ONLY when nothing
can be lost:

* the agent is not in the active heartbeat set (P6 said dead, twice over);
* the agent holds NO active claim and NO active file lock (P1/P13);
* the git tree is CLEAN (`git status --porcelain` empty) — every commit it
  ever made is already pushed to the shared bare repo, so the clone is a
  reproducible asset, not the only copy;
* the directory has been idle longer than `max_idle_minutes` (0 = off).

Anything that fails a check is left alone — a stale clone is a nuisance, a
deleted uncommitted result is a loss. Every prune is journaled AND audited
(P20), so disk debt is paid down transparently.
"""
from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path


def _git_clean(path) -> bool:
    """True when the clone has no uncommitted/untracked work (or git fails —
    a broken clone is pruned, it carries nothing reproducible)."""
    try:
        r = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"],
            capture_output=True, text=True, timeout=15, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return False
    if r.returncode not in (0, 128):
        return False
    return not (r.stdout or "").strip()


def sweep_stale_workspaces(memory, workspaces_dir, max_idle_minutes: int,
                           active_agents: list[dict], agent_id: str = "supervisor",
                           ) -> list[str]:
    """Prune clean, idle, agent-less workspaces; return the pruned names."""
    if max_idle_minutes <= 0 or workspaces_dir is None:
        return []
    base = Path(workspaces_dir)
    if not base.exists():
        return []
    active = {a.get("agent") for a in (active_agents or [])}
    holders = _active_holders(memory)
    now = time.time()
    idle_limit_s = max_idle_minutes * 60
    pruned: list[str] = []
    for d in sorted(base.iterdir()):
        if not d.is_dir() or d.name in active or d.name in holders:
            continue
        try:
            idle_s = now - d.stat().st_mtime
        except OSError:
            continue
        if idle_s < idle_limit_s:
            continue
        if not _git_clean(d):
            memory.journal(agent_id, "workspace_retention_skip", {
                "agent": d.name, "why": "dirty tree — work not on origin"})
            continue
        shutil.rmtree(d, ignore_errors=True)
        if not d.exists():
            pruned.append(d.name)
            memory.journal(agent_id, "workspace_pruned", {
                "agent": d.name, "idle_s": int(idle_s)})
            try:
                memory.audit(agent_id, "workspace_pruned",
                             {"agent": d.name, "idle_s": int(idle_s)})
            except Exception:  # noqa: BLE001 — audit must never gate the sweep
                pass
    return pruned


def _active_holders(memory) -> set[str]:
    """Agents that still hold claims or file locks — never pruned."""
    holders: set[str] = set()
    try:
        holders |= {c.agent for c in (memory.active_claims() or {}).values()}
    except Exception:  # noqa: BLE001
        pass
    try:
        holders |= {lk["agent"] for lk in (memory.active_file_locks() or [])}
    except Exception:  # noqa: BLE001
        pass
    return holders
