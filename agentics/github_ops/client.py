"""PyGithub wrapper — one GitHub identity per agent thread is *not* required for
API calls, but branch pushes are attributed per-agent via workspace commits.

Production hardening (seen live 2026-11-13, issue #70 run):

* Unattended lab etiquette: poll gently, absorb secondary rate limits (403 with
  Retry-After) and transient connection drops instead of failing the cycle.
  - secondary_rate_wait 60s: patient between secondary-limit retries
  - max_rate_limit_wait 900s: never stall an agent for a full hour on a
    primary rate-limit reset
  - total=6 with backoff_factor=2 so 5xx storms survive instead of exhausting
    the caller ("too many 500 error responses")
* Process-wide pacing (``GH_MIN_API_INTERVAL``, default 1.0 s): all agent
  threads share one PAT, and bursts of concurrent GETs trip GitHub's abuse
  detection. The pacing wrapper serializes every API call through this client
  into a steady ~1 req/s stream — well under the 5 000 req/h core limit.
"""
from __future__ import annotations

import logging
import random
import threading
import time
from functools import cached_property

from github import Auth, Github, GithubRetry, Repository

from agentics.config import Settings

log = logging.getLogger("gh.client")

# Unattended lab etiquette: poll gently, absorb secondary rate limits (403 with
# Retry-After) and transient connection drops instead of failing the cycle.
_GITHUB_RETRY = GithubRetry(
    total=6,
    backoff_factor=2,
    secondary_rate_wait=60,
    max_rate_limit_wait=900,
)
_GITHUB_TIMEOUT_S = 30

# Process-wide pacing state (all GitHubClient instances share the same PAT).
_PACE_LOCK = threading.Lock()
_PACE_LAST: float = 0.0


def make_paced(orig, min_interval: float):
    """Wrap ``orig`` so consecutive calls are at least ``min_interval`` seconds
    apart (plus jitter). The lock is held only while sleeping — the actual
    network call happens outside it, so slow responses cannot block others."""
    def paced(*args, **kwargs):
        global _PACE_LAST
        with _PACE_LOCK:
            wait = min_interval - (time.monotonic() - _PACE_LAST)
            if wait > 0:
                time.sleep(wait + random.uniform(0.0, 0.35))
            _PACE_LAST = time.monotonic()
        return orig(*args, **kwargs)
    return paced


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
        interval = float(getattr(settings, "gh_min_api_interval", 1.0) or 0.0)
        if interval > 0:
            requester = self.gh.requester
            requester.requestJsonAndCheck = make_paced(
                requester.requestJsonAndCheck, interval)
            requester.requestJson = make_paced(requester.requestJson, interval)
            log.info("GitHub API pacing enabled: min interval %.2fs", interval)

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
