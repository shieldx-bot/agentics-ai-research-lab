"""P13 + P14 — file locks and intent binding across the swarm.

P13 (file locks): two agents must never work the same file concurrently —
in per-agent workspaces that only surfaces as a painful merge conflict, or
worse, as duplicated work discovered after both cycles finished. Locks are
declared at PLAN time (the intent's expected files), live in SQLite with a
TTL, and expire on their own when an agent dies.

P14 (intent binding): the plan's declared file scope is enforced at
implement time — a diff containing files the agent never declared is
yielded and journalled, so the dashboard's intent panel and the actual
diff can never silently diverge.
"""
from __future__ import annotations

import fnmatch

from agentics.memory.shared_memory import SharedMemory


def normalize_rel(path: str) -> str:
    """Relative POSIX path without ./ noise. Raises ValueError on traversal."""
    cleaned = (path or "").strip().replace("\\", "/")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    if cleaned.startswith("/") or (len(cleaned) > 1 and cleaned[1] == ":"):
        raise ValueError(f"absolute paths are not allowed: {path}")
    if ".." in cleaned.split("/"):
        raise ValueError(f"path traversal is not allowed: {path}")
    if not cleaned:
        raise ValueError("empty path")
    return cleaned


def path_covered(path: str, declared: list[str]) -> bool:
    """True when `path` is inside the declared scope: exact match, directory
    prefix, or a shell-like pattern."""
    try:
        p = normalize_rel(path)
    except ValueError:
        return False
    for entry in declared or []:
        try:
            d = normalize_rel(entry)
        except ValueError:
            continue
        if p == d:
            return True
        prefix = d if d.endswith("/") else d + "/"
        if p.startswith(prefix):
            return True
        if any(ch in d for ch in "*?[") and fnmatch.fnmatch(p, d):
            return True
    return False


def scope_violations(changed_files: list[str], declared: list[str]) -> list[str]:
    """Changed files that were never declared in the intent (P14)."""
    return [f for f in changed_files or [] if not path_covered(f, declared)]


class FileLockManager:
    """All-or-nothing multi-file acquisition over the SQLite lock registry."""

    def __init__(self, memory: SharedMemory, ttl_s: float = 1800.0) -> None:
        self.memory = memory
        self.ttl_s = ttl_s

    def try_acquire_all(self, agent: str, paths: list[str]) -> tuple[bool, dict[str, str]]:
        """Acquire every path or nothing. On failure, returns (False, conflicts)
        where conflicts maps path -> current holder; any partial acquisitions
        are rolled back so an agent never half-holds a plan."""
        cleaned: list[str] = []
        for path in paths or []:
            try:
                cleaned.append(normalize_rel(path))
            except ValueError:
                continue                    # the guard will deal with bad paths
        cleaned = sorted(set(cleaned))
        if not cleaned:
            return True, {}
        acquired: list[str] = []
        conflicts: dict[str, str] = {}
        try:
            for path in cleaned:
                if not self.memory.acquire_file_lock(path, agent, self.ttl_s):
                    owner = next(
                        (l["agent"] for l in self.memory.active_file_locks()
                         if l["path"] == path),
                        "unknown",
                    )
                    conflicts[path] = owner
                    break
                acquired.append(path)
        finally:
            if conflicts:
                for path in acquired:       # never half-hold a plan
                    self.memory.release_file_lock(path, agent)
        return (not conflicts), conflicts

    def release_agent(self, agent: str) -> list[str]:
        return self.memory.release_agent_locks(agent)

    def release(self, agent: str, path: str) -> bool:
        return self.memory.release_file_lock(path, agent)

    def sweep_stale(self) -> list[str]:
        return self.memory.sweep_file_locks()

    def active(self) -> list[dict]:
        return self.memory.active_file_locks()
