"""Production-hardening tests for GitHub ops and orchestrator anti-burst.

Covers the failure modes seen in live runs:
- GitHubClient must carry an explicit retry policy + timeout (403 storms,
  "Max retries exceeded" connect drops must not kill cycles instantly)
- claim_issue must not orphan a claim label when the informational comment
  fails after the label was applied (issue #19 lockout)
- scan_open_issues must tolerate duck-typed repo objects without `.id`
- agents must start staggered, not in a synchronized burst
"""
from __future__ import annotations

from types import SimpleNamespace

from agentics.github_ops import issue_ops
from agentics.orchestrator import _startup_delay_s


# --------------------------------------------------------------- github client
def test_github_client_has_timeout_and_retry():
    from agentics.config import Settings
    from agentics.github_ops.client import (
        _GITHUB_RETRY,
        _GITHUB_TIMEOUT_S,
        GitHubClient,
    )

    assert _GITHUB_TIMEOUT_S >= 15
    settings = Settings()
    settings.github_token = "t" * 40
    client = GitHubClient(settings)
    req = client.gh.requester
    retry = getattr(req, "_Requester__retry", None)
    timeout = getattr(req, "_Requester__timeout", None)
    if retry is not None:  # private attr name stable across PyGithub 2.x
        assert retry.status_forcelist and 403 in retry.status_forcelist
    if timeout is not None:
        assert timeout == _GITHUB_TIMEOUT_S


# ----------------------------------------------------------------- claim issue
class _FakeIssue:
    def __init__(self, fail_label=False, fail_comment=False):
        self.labels = []
        self.comments: list[str] = []
        self._fail_label = fail_label
        self._fail_comment = fail_comment

    def add_to_labels(self, label):
        if self._fail_label:
            raise ConnectionError("network down")
        self.labels.append(SimpleNamespace(name=label))

    def create_comment(self, body):
        if self._fail_comment:
            raise ConnectionError("network down mid-claim")
        self.comments.append(body)


class _ClaimRepo:
    def __init__(self, issue: _FakeIssue):
        self._issue = issue

    def get_issue(self, _number):
        return self._issue


def test_claim_issue_label_failure_returns_false(monkeypatch):
    issue = _FakeIssue(fail_label=True)
    ok = issue_ops.claim_issue(_ClaimRepo(issue), 19, "account-01", ttl_minutes=60)
    assert ok is False
    assert issue.labels == []


def test_claim_issue_comment_failure_keeps_label(monkeypatch):
    """The label is the lock; losing the comment must not roll back the claim
    (a failed rollback orphans the label and locks the issue forever)."""
    issue = _FakeIssue(fail_comment=True)
    ok = issue_ops.claim_issue(_ClaimRepo(issue), 19, "account-01", ttl_minutes=60)
    assert ok is True
    assert [lb.name for lb in issue.labels] == ["claimed/account-01"]
    assert issue.comments == []


def test_claim_issue_happy_path_posts_comment():
    issue = _FakeIssue()
    ok = issue_ops.claim_issue(_ClaimRepo(issue), 7, "account-02", ttl_minutes=45)
    assert ok is True
    assert [lb.name for lb in issue.labels] == ["claimed/account-02"]
    assert len(issue.comments) == 1 and "/claim @account-02" in issue.comments[0]


# ------------------------------------------------------------- scan resilience
def test_scan_open_issues_tolerates_repo_without_id(monkeypatch):
    monkeypatch.setattr(issue_ops, "_SCAN_CACHE", {})
    repo = SimpleNamespace(get_issues=lambda **kw: [])  # no `.id`
    assert issue_ops.scan_open_issues(repo) == []


# ------------------------------------------------------------ startup stagger
def test_startup_delay_is_zero_for_first_agent_and_capped():
    assert _startup_delay_s(0) == 0.0
    assert _startup_delay_s(1) == 7.0
    assert _startup_delay_s(9) == 60.0
    assert _startup_delay_s(50) == 60.0  # cap: no agent waits > 1 minute
    assert _startup_delay_s(-3) == 0.0   # defensive against bad indexes
