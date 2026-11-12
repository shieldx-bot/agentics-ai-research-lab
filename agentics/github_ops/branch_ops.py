"""Branch operations — per-agent namespaced branches (P2)."""
from __future__ import annotations

import logging
import re

from github.Repository import Repository

log = logging.getLogger("gh.branch")

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def branch_name(agent_id: str, issue_number: int, title: str) -> str:
    slug = _SLUG_RE.sub("-", title.lower()).strip("-")[:40]
    return f"agent/{agent_id}/issue-{issue_number}-{slug}"


def create_branch(repo: Repository, branch: str, from_branch: str | None = None) -> None:
    base = from_branch or repo.default_branch
    ref = f"refs/heads/{branch}"
    try:
        repo.create_git_ref(ref, repo.get_branch(base).commit.sha)
        log.info("created branch %s from %s", branch, base)
    except Exception as exc:  # noqa: BLE001
        if "Reference already exists" in str(exc):
            log.info("branch %s already exists", branch)
            return
        raise


def branch_exists(repo: Repository, branch: str) -> bool:
    try:
        repo.get_branch(branch)
        return True
    except Exception:  # noqa: BLE001
        return False
