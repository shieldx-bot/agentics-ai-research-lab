"""PR operations: open, cross-review request, auto-merge (P4/P5)."""
from __future__ import annotations

import logging
import time

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


def approve_review(pr: PullRequest, body: str) -> None:
    pr.create_review(event="APPROVE", body=body[:6000])


def merge_when_ready(
    repo: Repository,
    pr_number: int,
    author_agent: str,
    reviewer_agent: str,
    auto_merge: bool = True,
) -> bool:
    """P5: needs an approval from someone other than the author.
    P4: Integrator merges serially; rebase-merge keeps history flat."""
    pr = repo.get_pull(pr_number)
    reviews = list(pr.get_reviews())
    approvals = [
        r for r in reviews
        if r.state == "APPROVED" and r.user and r.user.login != author_agent
    ]
    if not approvals:
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
