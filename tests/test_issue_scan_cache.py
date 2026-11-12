from __future__ import annotations

from types import SimpleNamespace

from agentics.github_ops import issue_ops


class FakeRepo:
    id = 123

    def __init__(self):
        self.calls = 0

    def get_issues(self, **_kwargs):
        self.calls += 1
        return [
            SimpleNamespace(
                pull_request=None,
                labels=[],
                number=1,
            )
        ]


def test_scan_open_issues_is_shared_cached_across_agent_calls(monkeypatch):
    monkeypatch.setattr(issue_ops, "_SCAN_CACHE", {})
    repo = FakeRepo()

    first = issue_ops.scan_open_issues(repo)
    second = issue_ops.scan_open_issues(repo)

    assert repo.calls == 1
    assert first == second
    assert first is not second


def test_scan_open_issues_refreshes_after_ttl(monkeypatch):
    monkeypatch.setattr(issue_ops, "_SCAN_CACHE", {})
    monkeypatch.setattr(issue_ops, "_SCAN_CACHE_TTL_S", 0)
    repo = FakeRepo()

    issue_ops.scan_open_issues(repo)
    issue_ops.scan_open_issues(repo)

    assert repo.calls == 2


def test_scan_cache_key_includes_repo_and_limit(monkeypatch):
    monkeypatch.setattr(issue_ops, "_SCAN_CACHE", {})
    repo = FakeRepo()

    issue_ops.scan_open_issues(repo, limit=10)
    issue_ops.scan_open_issues(repo, limit=20)

    assert repo.calls == 2
