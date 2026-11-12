"""PyGithub wrapper — one GitHub identity per agent thread is *not* required for
API calls, but branch pushes are attributed per-agent via workspace commits."""
from __future__ import annotations

import logging
from functools import cached_property

from github import Github, Repository

from agentics.config import Settings

log = logging.getLogger("gh.client")


class GitHubClient:
    def __init__(self, settings: Settings) -> None:
        if not settings.github_token:
            raise RuntimeError("GITHUB_TOKEN is not set — cannot talk to GitHub")
        self.settings = settings
        self.gh = Github(settings.github_token, per_page=50)

    @cached_property
    def repo(self) -> Repository.Repository:
        return self.gh.get_repo(self.settings.github_repo)

    @property
    def default_branch(self) -> str:
        return self.repo.default_branch or "main"

    def label_names(self) -> set[str]:
        try:
            return {lb.name for lb in self.repo.get_labels()}
        except Exception:  # noqa: BLE001
            return set()

    def ensure_labels(self, names: list[str]) -> None:
        existing = self.label_names()
        for name in names:
            if name not in existing:
                try:
                    self.repo.create_label(
                        name=name, color="1d76db", description="agentics"
                    )
                    log.info("created label %s", name)
                except Exception as exc:  # noqa: BLE001
                    log.debug("label %s: %s", name, exc)
