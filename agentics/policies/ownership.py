"""P3 — ownership matrix: agents yield when touching someone else's territory."""
from __future__ import annotations

import fnmatch
import logging
from typing import Any

log = logging.getLogger("policy.ownership")

DEFAULT_OWNERSHIP: dict[str, list[str]] = {
    "coder": ["agentics/", "tests/", "scripts/"],
    "researcher": ["research_notes/"],
    "writer": ["docs/", "README.md", "README*"],
    "integrator": [".github/"],
    "supervisor": ["configs/"],
}

YIELD_THRESHOLD = 0.3   # >30% of changed files in foreign territory -> yield


def _matches(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, p) or path.startswith(p.rstrip("*")) for p in patterns)


class OwnershipPolicy:
    def __init__(self, ownership: dict[str, list[str]] | None = None) -> None:
        self.ownership = ownership or DEFAULT_OWNERSHIP

    def owner_of(self, path: str) -> str | None:
        for role, patterns in self.ownership.items():
            if _matches(path, patterns):
                return role
        return None

    def evaluate(
        self,
        agent_role: str,
        changed_files: list[str],
        active_roles: set[str],
    ) -> tuple[bool, list[str]]:
        """Return (ok_to_push, foreign_files).

        ok_to_push=False when too many changed files belong to foreign roles
        that are currently active (P3) — agent should yield and split work.
        """
        foreign: list[str] = []
        for path in changed_files:
            owner = self.owner_of(path)
            if owner and owner != agent_role and owner in active_roles:
                foreign.append(path)
        ratio = len(foreign) / max(1, len(changed_files))
        ok = ratio <= YIELD_THRESHOLD or not (foreign and active_roles)
        if not ok:
            log.info("yield: %d/%d files in foreign territory %s",
                     len(foreign), len(changed_files), foreign[:10])
        return ok, foreign

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> "OwnershipPolicy":
        if cfg and isinstance(cfg, dict):
            return cls({k: list(v) for k, v in cfg.items()})
        return cls()
