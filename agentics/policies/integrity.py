"""P10 layer 4 — integrity self-healing for agent workspaces.

The other P10 layers (write_file, diff gate, review/merge gate) all act at
*action time*. Nothing repaired a workspace if a rogue script executed through
`run_command` (python-allowlisted, so it can compute paths at runtime) wrote
directly into the lab's own code of ITS clone. This watcher closes that gap:

  - at boot, hash every protected path inside every agent workspace;
  - sweep periodically (default 20 s, 0 = off);
  - any drift (modified / deleted / new-in-protected) is restored from git
    (`checkout --` for tracked, `clean -f --` for untracked) and journalled
    loudly so the owner sees it on the dashboard.

Honest scope: this repairs the CLONES. `main` of the bare repo can only
advance through the guarded merge queue, so the machinery on main is already
protected by the other three layers.

Companion to `agentics/sandbox/integrity.py` (layer 4a): that one DETECTS
drift on the RUNTIME tree the lab itself runs from and alerts the owner;
this watcher (layer 4b) covers the per-agent WORKSPACES and self-heals them
(a clone is reproducible from main, so auto-restore is safe there).
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import subprocess
from pathlib import Path

from agentics.memory.shared_memory import SharedMemory
from agentics.policies.guard import ALWAYS_PROTECTED, ProtectedPathsGuard

log = logging.getLogger("policies.integrity")


class IntegrityWatcher:
    """Hash protected trees in all workspaces, sweep for drift, self-heal."""

    def __init__(
        self, memory: SharedMemory, guard: ProtectedPathsGuard,
        workspaces_dir: Path,
    ) -> None:
        self.memory = memory
        self.guard = guard
        self.workspaces_dir = Path(workspaces_dir)
        self._snapshot: dict[str, str] = {}          # "workspace:relpath" -> sha256

    # ------------------------------------------------------------ watch set
    def _watch_roots(self) -> list[Path]:
        if not self.workspaces_dir.exists():
            return []
        return sorted(
            p for p in self.workspaces_dir.iterdir()
            if (p / ".git").exists() and p.is_dir()
        )

    def _watch_files(self, root: Path) -> list[Path]:
        prefixes = (*self.guard.paths, *ALWAYS_PROTECTED)
        files: list[Path] = []
        for prefix in prefixes:
            base = root / prefix
            if base.is_file():
                files.append(base)
            elif base.is_dir():
                for p in base.rglob("*"):
                    if p.is_file() and ".git" not in p.parts:
                        files.append(p)
        return sorted(set(files))

    @staticmethod
    def _hash(path: Path) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest()

    # ------------------------------------------------------------ public API
    def snapshot(self) -> dict[str, str]:
        """Hash every watched file in every workspace; call once at boot."""
        self._snapshot = {}
        for root in self._watch_roots():
            for p in self._watch_files(root):
                key = f"{root.name}:{p.relative_to(root).as_posix()}"
                try:
                    self._snapshot[key] = self._hash(p)
                except OSError:
                    continue
        return dict(self._snapshot)

    def sweep(self) -> list[str]:
        """Detect drifted protected paths and restore them. Returns drift keys."""
        if not self._snapshot:
            self.snapshot()
            return []
        drifted: list[str] = []
        seen: set[str] = set()
        for root in self._watch_roots():
            for p in self._watch_files(root):
                key = f"{root.name}:{p.relative_to(root).as_posix()}"
                seen.add(key)
                try:
                    current = self._hash(p)
                except OSError:
                    continue
                if self._snapshot.get(key) != current:
                    drifted.append(key)
        for key in sorted(set(self._snapshot) - seen):
            drifted.append(key)               # protected file was deleted
        if drifted:
            self._restore(drifted)
        return drifted

    def _restore(self, keys: list[str]) -> None:
        by_root: dict[Path, list[str]] = {}
        for key in keys:
            name, _, rel = key.partition(":")
            root = self.workspaces_dir / name
            by_root.setdefault(root, []).append(rel)
        for root, paths in by_root.items():
            for path in paths:                 # per path: a mixed batch would
                for args in (["checkout", "--", path],   # fail all-or-nothing
                             ["clean", "-f", "--", path]):
                    proc = subprocess.run(
                        ["git", "-C", str(root), *args],
                        check=False, capture_output=True, text=True, timeout=60,
                    )
                    # A silent no-op restoration hid a baseline-vs-HEAD
                    # restore war for a whole live run (2026-11-13): make
                    # git's verdict visible instead of swallowing stderr.
                    if proc.returncode != 0 and proc.stderr.strip():
                        log.info("integrity restore %s %s: %s",
                                 args[0], path, proc.stderr.strip()[:200])
                self._resnapshot_after_restore(root, path)

    def _resnapshot_after_restore(self, root: Path, rel: str) -> None:
        """Adopt the post-restore truth into the baseline.

        The baseline is a disk snapshot first taken when a workspace is seen
        — often MID-CLONE, so it can disagree with git HEAD forever, and the
        same keys then re-drifted (and were "restored") on every sweep for
        the whole run (live 2026-11-13: one workspace churned every 21s for
        half an hour). After restoring, the snapshot learns the restored
        content when git confirms the working tree matches HEAD: a file the
        restore actually fixed stops drifting, while a file that keeps
        drifting (a real rogue) keeps alerting. A protected path that is
        gone AND untracked cannot be restored by git at all (checkout only
        resurrects tracked paths, clean only deletes) — drop it from the
        watch set and journal once instead of looping forever."""
        key = f"{root.name}:{rel}"
        target = root / rel
        if target.is_file():
            same = subprocess.run(
                ["git", "-C", str(root), "diff", "--quiet", "HEAD", "--", rel],
                check=False, capture_output=True, text=True, timeout=60,
            )
            if same.returncode == 0:           # working tree == HEAD: adopt
                try:
                    self._snapshot[key] = self._hash(target)
                except OSError:
                    pass
            return
        if key not in self._snapshot:
            return                             # new rogue already removed: nothing to adopt
        tracked = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--error-unmatch", rel],
            check=False, capture_output=True, text=True, timeout=60,
        )
        if tracked.returncode != 0:
            self._snapshot.pop(key, None)
            try:
                self.memory.journal("guard", "integrity_watch_dropped", {
                    "workspace": root.name, "path": rel,
                    "reason": "untracked and deleted; nothing to restore",
                })
            except Exception:  # noqa: BLE001 — telemetry must never gate harder
                pass

    def sweep_and_journal(self) -> list[str]:
        """Sweep, and when drift is found: journal + broadcast the alert."""
        drifted = self.sweep()
        if drifted:
            log.warning("[guard] integrity drift: %s — restored", drifted[:5])
            self.memory.journal("guard", "integrity_drift", {
                "drifted": drifted[:20],
                "action": "restored from git",
            })
            self.memory.post_message(
                "guard", "alert",
                f"P10 integrity drift detected and restored: {drifted[:5]}"
            )
        return drifted

    async def run(self, interval_s: int) -> None:
        """Orchestrator loop: snapshot once, then sweep every interval."""
        self.snapshot()
        self.memory.journal("guard", "integrity_armed", {
            "workspaces": [r.name for r in self._watch_roots()],
            "watched_files": len(self._snapshot),
        })
        while True:
            await asyncio.sleep(max(5.0, float(interval_s)))
            try:
                self.sweep_and_journal()
            except Exception as exc:  # noqa: BLE001
                log.warning("integrity sweep failed: %s", exc)
