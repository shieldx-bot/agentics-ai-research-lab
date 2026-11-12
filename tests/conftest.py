"""Autouse hygiene: the shared GitHub scan cache is module-level state, so it
must be cleared between tests (fake repo objects can reuse memory addresses,
which would otherwise collide with cached entries)."""
import pytest

from agentics.github_ops import issue_ops


@pytest.fixture(autouse=True)
def _clean_scan_cache():
    issue_ops.clear_scan_cache()
    yield
    issue_ops.clear_scan_cache()
