"""P10 ProtectedPathsGuard — the anti 'AI sửa nhầm code agent' gate."""

import pytest

from agentics.policies.guard import ALWAYS_PROTECTED, ProtectedPathsGuard


@pytest.fixture()
def guard():
    return ProtectedPathsGuard()


def test_default_lab_code_paths_are_protected(guard):
    for path in (
        "agentics/config.py",
        "agentics/graph/nodes.py",
        "agentics/policies/claims.py",
        "configs/lab.yaml",
        "configs/secrets.yaml",
        "accounts/account-01.json",
        "scripts/live_research_run.py",
        ".github/workflows/ci.yml",
    ):
        assert guard.is_protected(path), path


def test_research_surface_is_writable(guard):
    for path in (
        "experiments/bench.py",
        "experiments/nested/deep.py",
        "research_notes/survey.md",
        "docs/notes.md",
        "README.md",
        "tests/test_something.py",
        "benchmarks/run.py",
    ):
        assert not guard.is_protected(path), path


def test_traversal_and_dot_prefixes_are_normalised(guard):
    assert guard.is_protected("./agentics/config.py")
    # a sneaky traversal that stays under agentics/ still matches the prefix
    assert guard.is_protected("agentics/sub/../stop.py")
    # (real cross-root traversal is impossible anyway: RepoWorkspace resolves
    #  paths and rejects anything escaping the workspace clone)


def test_task_exempt_lifts_default_list_but_never_credentials(guard):
    files = ["agentics/agents/roles.py", "accounts/account-01.json",
             "experiments/x.py", "configs/secrets.yaml"]
    violations = guard.violations(files, task_exempt=True)
    # accounts/ + secrets stay protected even with an owner exemption
    assert set(violations) == {"accounts/account-01.json", "configs/secrets.yaml"}


def test_no_exempt_blocks_all_lab_code(guard):
    files = ["agentics/agents/roles.py", "experiments/x.py"]
    assert guard.violations(files, task_exempt=False) == ["agentics/agents/roles.py"]


def test_env_override_allows_everything_except_credentials(guard, monkeypatch):
    monkeypatch.setenv("ALLOW_PROTECTED_PATHS", "1")
    files = ["agentics/config.py", "configs/lab.yaml", "accounts/a.json"]
    assert guard.violations(files) == ["accounts/a.json"]
    monkeypatch.setenv("ALLOW_PROTECTED_PATHS", "0")
    assert len(guard.violations(files)) == 3


def test_check_task_reads_the_owner_flag_from_memory(guard, tmp_path):
    from agentics.memory.shared_memory import SharedMemory

    mem = SharedMemory(tmp_path / "g.db")
    try:
        tid = mem.create_task("self-evolution: prompts", category="evolution")
        assert guard.check_task(mem, tid, ["agentics/roles.py"])
        mem.set_task_exempt(tid, True, agent="owner")
        assert guard.check_task(mem, tid, ["agentics/roles.py"]) == []
        assert guard.check_task(mem, tid, ["accounts/a.json"])  # never lifted
        assert guard.check_task(mem, None, ["agentics/roles.py"])  # no task → blocked
    finally:
        mem.close()


def test_always_protected_covers_credentials():
    assert any(p.startswith("accounts/") for p in ALWAYS_PROTECTED)
