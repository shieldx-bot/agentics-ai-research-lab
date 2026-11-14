"""P10 whitelist ring — writable_roots ("cấm tuyệt đối") trong ProtectedPathsGuard."""

import pytest

from agentics.policies.guard import ProtectedPathsGuard


WRITABLE = ("research_notes/", "experiments/", "benchmarks/", "tests/",
            "docs/", "reports/", "README.md")


@pytest.fixture()
def whitelist_guard():
    return ProtectedPathsGuard(writable_roots=WRITABLE)


def test_outside_the_whitelist_is_a_violation(whitelist_guard):
    # not machinery, but outside the declared research surface → refused
    violations = whitelist_guard.violations(
        ["random_root_file.txt", "data/thing.bin", "notes/misc.md"])
    assert violations == ["random_root_file.txt", "data/thing.bin", "notes/misc.md"]


def test_inside_the_whitelist_and_machinery_rules_hold(whitelist_guard):
    assert whitelist_guard.violations([
        "experiments/bench.py", "docs/guide.md", "tests/test_x.py",
        "benchmarks/run.py", "reports/summary.md", "README.md",
        "research_notes/rev.md",
    ]) == []
    # machinery stays protected exactly as before
    assert whitelist_guard.violations(["agentics/core.py"]) == ["agentics/core.py"]
    # and credentials remain protected even under the whitelist
    assert whitelist_guard.violations(["accounts/account-01.json"]) == \
        ["accounts/account-01.json"]
    assert whitelist_guard.violations(["configs/secrets.yaml"]) == \
        ["configs/secrets.yaml"]


def test_owner_task_exemption_lifts_the_whitelist_but_never_credentials(
        whitelist_guard):
    # owner exemption lifts the whitelist AND the default protected list,
    # but NEVER the credentials (ALWAYS_PROTECTED)
    violations = whitelist_guard.violations(
        ["random_root_file.txt", "agentics/core.py", "accounts/a.json"],
        task_exempt=True)
    assert violations == ["accounts/a.json"]


def test_env_override_behaviour_is_preserved(whitelist_guard, monkeypatch):
    monkeypatch.setenv("ALLOW_PROTECTED_PATHS", "1")
    assert whitelist_guard.violations(["random_root_file.txt"]) == []


def test_default_guard_is_unchanged_without_whitelist():
    guard = ProtectedPathsGuard()
    assert guard.writable_roots is None
    assert guard.violations(["random_root_file.txt"]) == []   # blacklist-only
    assert guard.violations(["agentics/core.py"]) == ["agentics/core.py"]


def test_dedup_between_the_two_rings(whitelist_guard):
    # a path can be BOTH machinery-protected and outside the whitelist:
    # it must appear exactly once in the violations list
    out = whitelist_guard.violations(["scripts/tool.py"])
    assert out == ["scripts/tool.py"]
