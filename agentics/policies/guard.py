"""P10 — protected paths: agents must never modify the lab's own code.

Motivation (owner question 1): "Liệu các tác nhân có thể làm việc trực tiếp
trong local mà không dẫn tới tình trạng AI sửa nhầm code agent?" — this guard
is the answer's enforcement half. Three layers use it:

    1. RepoWorkspace.write_file  — a plan cannot even create/edit a protected file
    2. node_implement            — a diff that touches protected paths is yielded
    3. LocalMergeQueue.drain     — a merge of a protected-path branch is refused

An exemption is granted ONLY by the owner, either per task
(`python -m agentics.cli allow-protected <task_id>` — stored on the task row)
or globally via the ALLOW_PROTECTED_PATHS=1 environment variable for the run.
Everything else is rejected with a hard error.
"""
from __future__ import annotations

import fnmatch
import os

from agentics.memory.shared_memory import SharedMemory

# The lab's own machinery: agent code, runtime config, credentials, owner
# scripts and CI. Everything under these paths belongs to humans.
DEFAULT_PROTECTED_PATHS: tuple[str, ...] = (
    "agentics/",
    "configs/",
    "accounts/",
    "scripts/",
    ".github/",
)

# Files agents may never touch even when a task carries an exemption.
ALWAYS_PROTECTED: tuple[str, ...] = (
    "accounts/",          # cookies / credentials
    "configs/secrets.yaml",
)


class ProtectedPathsGuard:
    """Hard gate for the lab's own code, with an owner-only exception."""

    def __init__(
        self,
        paths: tuple[str, ...] | list[str] = DEFAULT_PROTECTED_PATHS,
        allow_env_override: bool = True,
        writable_roots: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        self.paths = tuple(paths) or DEFAULT_PROTECTED_PATHS
        self.allow_env_override = allow_env_override
        # Optional second ring ("cấm tuyệt đối"): when set, ANY file outside
        # these roots is refused even though it is not machinery — the union
        # of the ownership surfaces by default (see configs/lab.yaml).
        self.writable_roots = tuple(writable_roots) if writable_roots else None

    # ------------------------------------------------------------ checks
    def is_protected(self, path: str) -> bool:
        """Exact file match, directory prefix (entries end with '/'), or glob."""
        norm = (path or "").strip()
        while norm.startswith("./"):       # strip only "./", never a real "."
            norm = norm[2:]
        norm = norm.lstrip("/")
        return any(
            norm == p or norm.startswith(p)
            or (p.endswith("/") and norm.startswith(p))
            or (("*" in p or "?" in p) and fnmatch.fnmatch(norm, p))
            for p in (*self.paths, *ALWAYS_PROTECTED)
        )

    def _env_allows_all(self) -> bool:
        return self.allow_env_override and os.environ.get(
            "ALLOW_PROTECTED_PATHS", ""
        ).strip().lower() in ("1", "true", "yes", "on")

    def violations(self, changed_files: list[str], task_exempt: bool = False) -> list[str]:
        """Protected paths inside a diff. Empty list = allowed to proceed.

        Exemption semantics:
        - ``task_exempt=True`` (owner flagged the task) lifts the default list
          but NEVER the ALWAYS_PROTECTED credentials/secret paths;
        - ``ALLOW_PROTECTED_PATHS=1`` (owner env switch) lifts everything.
        """
        if self._env_allows_all() or task_exempt:
            out = [f for f in changed_files if self._always_protected(f)]
        else:
            out = [f for f in changed_files
                   if self.is_protected(f) or self._outside_writable(f)]
        return list(dict.fromkeys(out))          # dedup, keep order

    def _outside_writable(self, path: str) -> bool:
        """True when the whitelist ring is active and `path` is outside it."""
        if not self.writable_roots:
            return False
        norm = (path or "").strip()
        while norm.startswith("./"):
            norm = norm[2:]
        norm = norm.lstrip("/")
        return not any(
            norm == root.rstrip("/")
            or norm.startswith(root)
            or (("*" in root or "?" in root) and fnmatch.fnmatch(norm, root))
            for root in self.writable_roots
        )

    def _always_protected(self, path: str) -> bool:
        norm = (path or "").strip().lstrip("./")
        return any(norm == p or norm.startswith(p) for p in ALWAYS_PROTECTED)

    def check_task(
        self, memory: SharedMemory, task_id: int | None, changed_files: list[str],
    ) -> list[str]:
        """Convenience for graph nodes: read the task's owner exemption flag."""
        exempt = False
        if task_id is not None:
            task = memory.get_task(task_id)
            exempt = bool(task and task.get("exempt"))
        return self.violations(changed_files, task_exempt=exempt)
