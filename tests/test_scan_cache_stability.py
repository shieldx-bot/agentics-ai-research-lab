"""Regression: the shared issue-scan cache must key on STABLE identity only.

Keying on ``id(obj)`` poisons the cache when CPython reuses the address of a
garbage-collected duck-typed repo — a later scan silently resurrects an
earlier (stale) issue list. Contract: objects without a real ``repo.id`` are
NEVER cached; objects with one are cached for the TTL.
"""
from __future__ import annotations

import sys
import time
from types import SimpleNamespace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agentics.github_ops import issue_ops  # noqa: E402


def _issue(number: int, title: str) -> SimpleNamespace:
    return SimpleNamespace(number=number, title=title, body="", pull_request=None,
                           labels=[SimpleNamespace(name=n) for n in ()],
                           created_at=None)


def _repo(issues, *, with_id=None):
    repo = SimpleNamespace(get_issues=lambda **kw: list(issues))
    if with_id is not None:
        repo.id = with_id
    return repo


def test_unstable_identity_repos_are_never_cached():
    issue_ops._SCAN_CACHE.clear()
    issues = [_issue(1, "first")]
    repo = _repo(issues)                       # no .id -> must bypass the cache
    assert [i.number for i in issue_ops.scan_open_issues(repo)] == [1]
    issues.append(_issue(2, "second"))         # same object, mutated in place
    assert [i.number for i in issue_ops.scan_open_issues(repo)] == [1, 2]
    assert issue_ops._SCAN_CACHE == {}         # nothing was stored


def test_stable_identity_repos_are_cached_within_ttl():
    issue_ops._SCAN_CACHE.clear()
    issues = [_issue(7, "cached")]
    repo = _repo(issues, with_id="repo-1")
    assert [i.number for i in issue_ops.scan_open_issues(repo)] == [7]
    issues.append(_issue(8, "later"))          # mutation hidden by the cache
    assert [i.number for i in issue_ops.scan_open_issues(repo)] == [7]
    # expiry makes the mutation visible again
    key = ("repo-1", 30)
    ts, _ = issue_ops._SCAN_CACHE[key]
    issue_ops._SCAN_CACHE[key] = (ts - issue_ops._SCAN_CACHE_TTL_S - 1, [])
    assert [i.number for i in issue_ops.scan_open_issues(repo)] == [7, 8]


def test_different_stable_ids_do_not_share_entries():
    issue_ops._SCAN_CACHE.clear()
    first = issue_ops.scan_open_issues(_repo([_issue(1, "a")], with_id="repo-1"))
    second = issue_ops.scan_open_issues(_repo([_issue(2, "b")], with_id="repo-2"))
    assert [i.number for i in first] == [1]
    assert [i.number for i in second] == [2]


def test_cache_expiry_window_is_short():
    assert 0 < issue_ops._SCAN_CACHE_TTL_S <= 120
