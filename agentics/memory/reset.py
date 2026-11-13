"""Topic-reset — wipe ALL lab memory when the owner switches research topic.

The lab accumulates topic-bound state everywhere: the shared-memory DB
(tasks/findings/panels/journal/role prompts), the vector store (finding
recall), per-agent personal memory (lessons), agent workspaces (old-topic
checkouts) and the local bare origin (old-topic branches and MRs). None of
that is reusable for a new topic — old findings would be recalled into the
new research, old lessons would steer new plans, old workspaces would shadow
fresh clones. The reset removes exactly this set, keeps lab CODE state
(integrity baseline, accounts/, configs/), and relies on the existing
self-healing boot path: ``LocalRepo.ensure`` re-seeds ``data/lab.git`` from
the checkout, ``SharedMemory`` recreates the schema, the vector store
re-opens empty.

Safety: refuses to run while any agent heartbeat is fresh (status !=
"stopped") unless ``--force`` — a reset under a running swarm would corrupt
live sessions.
"""
from __future__ import annotations

import shutil
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

from agentics.config import Settings

# A supervisor/lease heartbeat lands at most every 30s (min(ttl/3, 30)); a
# lab mid-cycle therefore always refreshes inside this window. Rows with
# status "stopped" are the graceful-shutdown mark and never count as running.
RUNNING_WINDOW_S = 120.0


class ResetBlocked(RuntimeError):
    """Fresh agent heartbeats detected — the lab looks like it is running."""


@dataclass
class ResetReport:
    removed: list[tuple[str, int]] = field(default_factory=list)  # (label, bytes)
    kept: list[str] = field(default_factory=list)

    @property
    def freed_bytes(self) -> int:
        return sum(n for _, n in self.removed)

    @property
    def freed_human(self) -> str:
        n = float(self.freed_bytes)
        for unit in ("B", "KB", "MB", "GB"):
            if n < 1024 or unit == "GB":
                return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
            n /= 1024
        return f"{n:.1f} GB"


def _tree_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return 0


def _remove(path: Path) -> int:
    """Delete a file or directory tree; return bytes freed (0 if absent)."""
    if not path.exists() and not path.is_symlink():
        return 0
    size = _tree_size(path)
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)
    return size


def reset_targets(settings: Settings, *, keep_repo: bool = False) -> list[tuple[str, list[Path]]]:
    """The topic-bound state groups, in wipe order. db_path, chroma_path,
    workspaces_dir and local_repo_path are all derived from data_dir."""
    groups: list[tuple[str, list[Path]]] = [
        ("shared memory (tasks/findings/panels/journal/role prompts)",
         [settings.db_path,
          settings.db_path.with_name(settings.db_path.name + "-wal"),
          settings.db_path.with_name(settings.db_path.name + "-shm")]),
        ("vector memory (finding recall)", [settings.chroma_path]),
        ("personal memory (per-agent lessons)",
         [settings.data_dir / "personal"]),
        ("agent workspaces (old-topic checkouts)", [settings.workspaces_dir]),
    ]
    if not keep_repo:
        groups.append(("local origin repo (old-topic branches/MRs)",
                       [settings.local_repo_path]))
    return groups


def running_agents(settings: Settings,
                   window_s: float = RUNNING_WINDOW_S) -> list[dict]:
    """Agents whose heartbeat is fresher than ``window_s`` and not 'stopped'.

    A missing DB is a cold lab (no agents). A DB that cannot be opened is
    treated as RUNNING (locked/corrupt = a live process may hold it)."""
    if not settings.db_path.exists():
        return []
    from agentics.memory.shared_memory import SharedMemory

    try:
        mem = SharedMemory(settings.db_path)
    except sqlite3.OperationalError as exc:
        raise ResetBlocked(
            f"shared memory DB cannot be opened ({exc}) — a live process may "
            "be holding it. Stop the lab first or use --force."
        ) from exc
    try:
        fresh = mem.active_agents(stale_after_s=window_s)
    finally:
        mem.close()
    return [a for a in fresh if (a.get("status") or "") != "stopped"]


def reset_memory(settings: Settings, *, keep_repo: bool = False,
                 dry_run: bool = False) -> ResetReport:
    """Wipe every topic-bound memory group; recreate empty directories.

    Kept on purpose (lab CODE state, not topic memory): the integrity
    baseline (``data/integrity``), accounts/, configs/, .env, the logs dir.
    The DB is left absent — the next boot recreates a clean schema."""
    report = ResetReport()
    if dry_run:
        for label, paths in reset_targets(settings, keep_repo=keep_repo):
            for p in paths:
                if p.exists():
                    report.removed.append((f"{label}: {p.name}", _tree_size(p)))
        report.kept.append("integrity baseline (lab code hashes)")
        return report

    # refuse BEFORE deleting anything
    live = running_agents(settings)
    if live:
        names = ", ".join(f"{a['agent']}({a['role']})" for a in live)
        raise ResetBlocked(
            f"{len(live)} agent(s) look alive ({names}) — stop the lab first "
            "(python -m agentics.cli stop) or pass --force."
        )

    for label, paths in reset_targets(settings, keep_repo=keep_repo):
        for p in paths:
            freed = _remove(p)
            if freed:
                report.removed.append((f"{label}: {p.name}", freed))

    settings.ensure_dirs()
    (settings.data_dir / "personal").mkdir(parents=True, exist_ok=True)
    report.kept.append("integrity baseline (lab code hashes)")
    return report
