"""Tests for shared-PAT self-approval fallback (422 "Can not approve your own
pull request") in agentics.github_ops.pr_ops."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from github.GithubException import GithubException

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agentics.github_ops import pr_ops  # noqa: E402


def _422_self_review() -> GithubException:
    return GithubException(
        422,
        {"message": "Unprocessable Entity",
         "errors": ["Review Can not approve your own pull request"]},
        None,
    )


class FakePr:
    def __init__(self, number: int = 12, labels: list[str] | None = None,
                 reviews: list | None = None) -> None:
        self.number = number
        self.labels = [SimpleNamespace(name=n) for n in (labels or [])]
        self._reviews = reviews or []
        self.reviews_called: list[str] = []
        self.labels_added: list[str] = []
        self.merged = False

    def create_review(self, event: str, body: str) -> None:
        self.reviews_called.append(event)
        if event == "APPROVE":
            raise _422_self_review()

    def add_to_labels(self, name: str) -> None:
        self.labels_added.append(name)
        self.labels.append(SimpleNamespace(name=name))

    def get_reviews(self) -> list:
        return self._reviews

    def update_branch(self) -> None:
        pass

    def merge(self, merge_method: str, commit_title: str) -> None:
        self.merged = True


def test_approve_review_falls_back_to_comment_and_label():
    pr = FakePr()
    event = pr_ops.approve_review(pr, "LGTM", reviewer_agent="a08")
    assert event == "COMMENT"
    assert pr.reviews_called == ["APPROVE", "COMMENT"]
    assert f"approved-by/a08" in [lb.name for lb in pr.labels]


def test_approve_review_succeeds_with_per_agent_pat():
    pr = FakePr()
    pr.create_review = lambda event, body: pr.reviews_called.append(event)
    event = pr_ops.approve_review(pr, "LGTM", reviewer_agent="a08")
    assert event == "APPROVE"
    assert pr.reviews_called == ["APPROVE"]
    assert pr.labels_added == []


def test_approve_review_reraises_unrelated_errors():
    pr = FakePr()

    def boom(event: str, body: str) -> None:
        raise GithubException(403, {"message": "forbidden"}, None)

    pr.create_review = boom
    with pytest.raises(GithubException):
        pr_ops.approve_review(pr, "LGTM")


def _repo_with(pr: FakePr):
    return SimpleNamespace(get_pull=lambda n: pr)


def test_merge_when_ready_accepts_approved_by_label():
    pr = FakePr(number=12, labels=["approved-by/a07"], reviews=[])
    ok = pr_ops.merge_when_ready(_repo_with(pr), 12, author_agent="a03",
                                 reviewer_agent="a09", auto_merge=True)
    assert ok is True
    assert pr.merged is True


def test_merge_when_ready_rejects_self_approval_label():
    # author "a03" approving its own PR via label must NOT count
    pr = FakePr(number=13, labels=["approved-by/a03"], reviews=[])
    ok = pr_ops.merge_when_ready(_repo_with(pr), 13, author_agent="a03",
                                 reviewer_agent="a09", auto_merge=True)
    assert ok is False


def test_merge_when_ready_still_accepts_real_review_event():
    review = SimpleNamespace(state="APPROVED", user=SimpleNamespace(login="someone"))
    pr = FakePr(number=14, labels=[], reviews=[review])
    ok = pr_ops.merge_when_ready(_repo_with(pr), 14, author_agent="a03",
                                 reviewer_agent="a09", auto_merge=False)
    assert ok is True          # auto_merge=False: gate passes, no merge done
    assert pr.merged is False
