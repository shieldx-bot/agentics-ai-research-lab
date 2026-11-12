"""Issue operations: scan, claim (P1), release, close with summary."""
from __future__ import annotations

import logging
import re
import threading
import time

from github.Issue import Issue
from github.Repository import Repository

log = logging.getLogger("gh.issue")

CLAIM_PREFIX = "/claim"
CLAIMED_LABEL_RE = re.compile(r"^claimed/(\S+)$")

# Extra claim-label attempts after PyGithub's own retry pool gave up (5xx
# incidents). A missed claim label would let a second agent double-claim the
# issue, so the add is retried before the claim is given up entirely.
CLAIM_LABEL_RETRY_DELAYS: tuple[float, ...] = (3.0, 10.0)


# All agents share one process. Cache the expensive GitHub issue scan so
# synchronized agent loops do not issue identical paginated requests.
_SCAN_CACHE_TTL_S = 45.0
_SCAN_CACHE_LOCK = threading.Lock()
_SCAN_CACHE: dict[tuple[int, int], tuple[float, list[Issue]]] = {}


def set_scan_cache_ttl(ttl_seconds: float) -> None:
    """Override the scan TTL at runtime (orchestrator applies the Settings
    value, which honors configs/lab.yaml)."""
    global _SCAN_CACHE_TTL_S
    if ttl_seconds and ttl_seconds > 0:
        _SCAN_CACHE_TTL_S = float(ttl_seconds)


def clear_scan_cache() -> None:
    """Drop all cached scans — claim/release invalidate the cache so the next
    scan sees the fresh claim labels immediately."""
    with _SCAN_CACHE_LOCK:
        _SCAN_CACHE.clear()


def scan_open_issues(repo: Repository, limit: int = 30) -> list[Issue]:
    """Open, unclaimed issues; a short shared cache limits API polling.

    The cache is only a discovery optimization. Claim acquisition must still
    re-read the target issue and atomically verify its current claim labels.
    """
    # Duck-typed repo id: real Repository objects always have `.id`; fakes and
    # thin wrappers may not — fall back to object identity for the cache key.
    key = (getattr(repo, "id", id(repo)), int(limit))
    now = time.monotonic()
    with _SCAN_CACHE_LOCK:
        cached = _SCAN_CACHE.get(key)
        if cached and now - cached[0] < _SCAN_CACHE_TTL_S:
            return list(cached[1])

        issues: list[Issue] = []
        for issue in repo.get_issues(state="open", sort="created", direction="asc"):
            if issue.pull_request is not None:
                continue
            labels = {lb.name for lb in issue.labels}
            if any(CLAIMED_LABEL_RE.match(lb) for lb in labels):
                continue
            if "blocked" in labels:
                continue
            issues.append(issue)
            if len(issues) >= limit:
                break
        _SCAN_CACHE[key] = (time.monotonic(), list(issues))
        # Keep the cache bounded if a process watches several repositories.
        if len(_SCAN_CACHE) > 32:
            oldest = min(_SCAN_CACHE, key=lambda k: _SCAN_CACHE[k][0])
            _SCAN_CACHE.pop(oldest, None)
        return issues


def issue_holders(repo: Repository, issue_number: int) -> list[str]:
    """Agents currently holding a claim on the issue (from labels + comments)."""
    holders: list[str] = []
    issue = repo.get_issue(issue_number)
    for lb in issue.labels:
        m = CLAIMED_LABEL_RE.match(lb.name)
        if m:
            holders.append(m.group(1))
    return holders


def claim_issue(repo: Repository, issue_number: int, agent_id: str, ttl_minutes: int) -> bool:
    """Two-layer claim: shared memory is checked by the caller; here we do the
    GitHub-side check+label as atomically as the REST API allows (P1)."""
    issue = repo.get_issue(issue_number)
    holders = issue_holders(repo, issue_number)
    if holders and agent_id not in holders:
        return False
    label = f"claimed/{agent_id}"
    last_exc: Exception | None = None
    for attempt in range(1 + len(CLAIM_LABEL_RETRY_DELAYS)):
        try:
            issue.add_to_labels(label)
            last_exc = None
            break
        except Exception as exc:  # noqa: BLE001 — 5xx storms / rate limits
            last_exc = exc
            log.warning("label add on #%s attempt %d/%d failed: %s",
                        issue_number, attempt + 1, 1 + len(CLAIM_LABEL_RETRY_DELAYS), exc)
            if attempt < len(CLAIM_LABEL_RETRY_DELAYS):
                time.sleep(CLAIM_LABEL_RETRY_DELAYS[attempt])
    if last_exc is not None:
        log.error("cannot add label on #%s after retries: %s", issue_number, last_exc)
        return False
    clear_scan_cache()                      # next scan must see the new claim
    # The label is the authoritative cross-process lock. A claim comment is
    # informational: if it fails (e.g. transient network drop) do NOT roll the
    # claim back — a failed rollback here could orphan the label and lock the
    # issue forever. The owner's release/sweep path removes the label later.
    try:
        issue.create_comment(
            f"{CLAIM_PREFIX} @{agent_id}\n\n"
            f"Claimed at `{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}`, "
            f"TTL {ttl_minutes} min. Branch prefix `agent/{agent_id}/`."
        )
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "claim comment failed on #%s (label stands): %s", issue_number, exc
        )
    return True


def release_issue(repo: Repository, issue_number: int, agent_id: str, reason: str = "") -> None:
    issue = repo.get_issue(issue_number)
    try:
        issue.remove_from_labels(f"claimed/{agent_id}")
    except Exception:  # noqa: BLE001
        pass
    clear_scan_cache()
    try:
        issue.create_comment(f"/release @{agent_id}\n\nReason: {reason or 'released'}")
    except Exception as exc:  # noqa: BLE001
        log.warning("release comment on #%s failed: %s", issue_number, exc)


def close_issue_with_summary(repo: Repository, issue_number: int, summary: str) -> None:
    issue = repo.get_issue(issue_number)
    issue.create_comment(f"## Result\n\n{summary}")
    issue.edit(state="closed")


def create_issue_if_absent(
    repo: Repository, title: str, body: str, labels: list[str] | None = None,
) -> Issue | None:
    """Open an issue unless an open one with the same title already exists
    (keeps the Supervisor from flooding the backlog). Returns the new issue or None."""
    for issue in repo.get_issues(state="open"):
        if issue.pull_request is None and issue.title == title:
            return None
    issue = repo.create_issue(title=title, body=body, labels=labels or [])
    clear_scan_cache()
    return issue
