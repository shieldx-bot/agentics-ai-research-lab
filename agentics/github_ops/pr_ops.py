"""PR operations: open, cross-review request, auto-merge (P4/P5)."""
from __future__ import annotations

import logging
import time

from github.GithubException import GithubException
from github.PullRequest import PullRequest
from github.Repository import Repository

log = logging.getLogger("gh.pr")

READY_LABEL = "ready-to-merge"


def open_pr(
    repo: Repository,
    title: str,
    head: str,
    base: str,
    body: str,
    author_agent: str,
) -> PullRequest:
    pr = repo.create_pull(title=title, body=body, head=head, base=base)
    pr.add_to_labels("agent", f"by/{author_agent}")
    return pr


def request_cross_review(pr: PullRequest, reviewers: list[str]) -> None:
    for r in reviewers:
        if r:
            try:
                pr.create_review_request(reviewers=[r])
            except Exception as exc:  # noqa: BLE001
                log.debug("review request %s failed: %s", r, exc)


def approve_review(pr: PullRequest, body: str, reviewer_agent: str | None = None) -> str:
    """Record a review on ``pr``; returns the GitHub event actually used.

    All agents share ONE GitHub PAT, so every PR belongs to the same GitHub
    user and GitHub rejects APPROVE with 422 "Can not approve your own pull
    request". In that case the review verdict is recorded as a COMMENT plus
    an ``approved-by/<agent>`` label — ``merge_when_ready`` accepts that
    label in place of a real APPROVED review event.
    """
    try:
        pr.create_review(event="APPROVE", body=body[:6000])
        return "APPROVE"
    except GithubException as exc:
        message = str(exc)
        recoverable = "422" in message or "own pull request" in message
        if not recoverable:
            raise
        log.info(
            "PR #%s: APPROVE rejected (shared token self-review) -> COMMENT fallback",
            pr.number,
        )
        pr.create_review(
            event="COMMENT",
            body=("[self-approval not possible with shared token] " + body)[:6000],
        )
        if reviewer_agent:
            try:
                pr.add_to_labels(f"approved-by/{reviewer_agent}")
            except Exception as exc2:  # noqa: BLE001
                log.warning("could not label approved-by on PR #%s: %s",
                            pr.number, exc2)
        return "COMMENT"


def merge_when_ready(
    repo: Repository,
    pr_number: int,
    author_agent: str,
    reviewer_agent: str,
    auto_merge: bool = True,
) -> bool:
    """P5: needs a cross-review approval (someone other than the author).

    Approval can be recorded in two ways:
    - a real APPROVED review event (possible with per-agent PATs), or
    - an ``approved-by/<agent>`` label (fallback for the shared-PAT mode,
      see :func:`approve_review`).
    P4: Integrator merges serially; rebase-merge keeps history flat."""
    pr = repo.get_pull(pr_number)
    reviews = list(pr.get_reviews())
    approvals = [
        r for r in reviews
        if r.state == "APPROVED" and r.user and r.user.login != author_agent
    ]
    approved = bool(approvals)
    if not approved:
        for lb in list(pr.labels):
            if lb.name.startswith("approved-by/"):
                agent = lb.name.split("/", 1)[1]
                if agent and agent != author_agent:
                    approved = True
                    break
    if not approved:
        log.info("PR #%s has no cross-review approval yet", pr_number)
        return False
    if not auto_merge:
        return True
    # update branch then rebase-merge
    try:
        pr.update_branch()
    except Exception as exc:  # noqa: BLE001
        log.warning("update_branch on #%s: %s", pr_number, exc)
    try:
        pr.merge(merge_method="rebase", commit_title=f"merge PR #{pr_number} ({author_agent})")
        log.info("merged PR #%s", pr_number)
        try:
            pr.edit(state="closed")
        except Exception:  # noqa: BLE001
            pass
        return True
    except Exception as exc:  # noqa: BLE001
        log.error("merge #%s failed: %s", pr_number, exc)
        return False


def pr_age_seconds(pr: PullRequest) -> float:
    created = pr.created_at
    return max(0.0, time.time() - created.timestamp())
