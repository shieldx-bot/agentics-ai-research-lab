"""PyGithub wrapper — one GitHub identity per agent thread is *not* required for
API calls, but branch pushes are attributed per-agent via workspace commits."""
from __future__ import annotations

import logging
from functools import cached_property

from github import Auth, Github, GithubRetry, Repository

from agentics.config import Settings

log = logging.getLogger("gh.client")

# Unattended lab etiquette: poll gently, absorb secondary rate limits (403 with
# Retry-After) and transient connection drops instead of failing the cycle.
# - secondary_rate_wait 60s: patient between secondary-limit retries
# - max_rate_limit_wait 900s: give up (raise) rather than freeze a cycle for
#   most of an hour when the primary quota is exhausted
# - timeout 30s: fail fast on dead connections instead of hanging a node
_GITHUB_RETRY = GithubRetry(
    secondary_rate_wait=60,
    max_rate_limit_wait=900,
)
_GITHUB_TIMEOUT_S = 30


class GitHubClient:
    def __init__(self, settings: Settings) -> None:
        if not settings.github_token:
            raise RuntimeError("GITHUB_TOKEN is not set — cannot talk to GitHub")
        self.settings = settings
        self.gh = Github(
            auth=Auth.Token(settings.github_token),
            per_page=50,
            timeout=_GITHUB_TIMEOUT_S,
            retry=_GITHUB_RETRY,
        )

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
