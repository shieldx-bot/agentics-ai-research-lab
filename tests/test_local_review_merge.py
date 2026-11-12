"""End-to-end local-mode pipeline — real git, zero GitHub.

    task → claim → branch in a real workspace clone → commit → push to the
    shared bare repo → open MR → cross-review approve → tier-2 verify →
    local merge into main → task merged.
"""

import subprocess
from pathlib import Path

import pytest

from agentics.config import Settings
from agentics.github_ops.repo_workspace import RepoWorkspace
from agentics.local_ops import (
    LocalClaimPolicy, LocalMergeQueue, LocalRepo, branch_name,
)
from agentics.memory.shared_memory import SharedMemory
from agentics.policies.guard import ProtectedPathsGuard


@pytest.fixture()
def lab(tmp_path: Path):
    """A seeded local lab: bare repo + memory + guard + settings."""
    src = tmp_path / "src"
    src.mkdir()
    subprocess.run(["git", "init", "-b", "main", str(src)], check=True, capture_output=True)
    (src / "README.md").write_text("seed project\n", encoding="utf-8")
    (src / "experiments").mkdir()
    (src / "experiments" / ".gitkeep").write_text("", encoding="utf-8")
    subprocess.run(["git", "-C", str(src), "add", "."], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(src), "-c", "user.email=t@t", "-c", "user.name=t",
         "commit", "-m", "seed"], check=True, capture_output=True)

    settings = Settings(data_dir=tmp_path / "data")
    settings.ensure_dirs()
    LocalRepo(settings).ensure(src=src)

    mem = SharedMemory(tmp_path / "m.db")
    try:
        yield Simple(settings=settings, mem=mem,
                     guard=ProtectedPathsGuard(), repo=LocalRepo(settings))
    finally:
        mem.close()


class Simple:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _workspace(lab, agent_id: str) -> RepoWorkspace:
    return RepoWorkspace(agent_id, lab.settings, guard=lab.guard)


def test_full_local_lifecycle_claim_to_merge_on_main(lab):
    mem, guard = lab.mem, lab.guard
    task_id = mem.create_task(
        "Benchmark skeleton for NFV failure prediction",
        body="Create experiments/bench.py with a tiny benchmark skeleton.",
        category="code",
    )

    # ---- author agent (a04/coder): claim → branch → commit → push
    claims = LocalClaimPolicy(mem, ttl_minutes=60)
    branch = branch_name("a04", task_id, "Benchmark skeleton for NFV failure prediction")
    assert claims.try_claim(task_id, "a04", branch)

    author = _workspace(lab, "a04")
    author.sync_main()
    author.checkout_new_branch(branch)
    author.write_file("experiments/bench.py", "def run():\n    return 42\n")
    (author.path / "tests").mkdir(exist_ok=True)
    author.write_file("tests/test_bench.py", "def test_run():\n    assert True\n")
    assert author.commit_and_push(branch, f"agent(a04): bench skeleton (#{task_id})")
    claims.complete(task_id, "a04", branch)

    # ---- author opens the MR
    changed = author.diff_files_vs_main()
    mr_id = mem.open_review(
        task_id=task_id,
        title=f"[a04] Benchmark skeleton (#{task_id})",
        body="Closes the task.", author="a04", branch=branch,
        files=changed, verification="tier-1 passed (py_compile + pytest)",
    )
    assert mem.get_task(task_id)["status"] == "in_review"

    # ---- reviewer agent (a07): cross-review approval (P5: not the author)
    guard.check_task(mem, task_id, changed)          # no violations
    mem.set_review_verdict(mr_id, "approved", "a07", "lgtm")
    mem.set_trace_outcome(mr_id, "approved")
    queue = LocalMergeQueue(mem, auto_merge=True, guard=guard)
    queue.enqueue(mr_id, author="a04")

    # ---- integrator (a09): tier-2 on main+MR, then the local merge
    integrator = _workspace(lab, "a09")
    ok, note = integrator.verify_merge(branch)
    assert ok, note
    assert queue.drain(
        "a09",
        verify=lambda mr: integrator.verify_merge(lab.mem.get_review(mr)["branch"]),
        merge=lambda mr: integrator.merge_branch_to_main(
            lab.mem.get_review(mr)["branch"],
            f"merge MR #{mr} (a04) — task #{task_id}"),
    ) == 1

    # ---- assertions: the work is on main of the shared bare repo
    assert mem.get_review(mr_id)["status"] == "merged"
    assert mem.get_task(task_id)["status"] == "merged"
    log = subprocess.run(
        ["git", "-C", str(lab.repo.path), "log", "--oneline", "main"],
        capture_output=True, text=True).stdout
    assert f"merge MR #{mr_id}" in log
    content = subprocess.run(
        ["git", "-C", str(lab.repo.path), "show", f"main:experiments/bench.py"],
        capture_output=True, text=True).stdout
    assert "def run" in content


def test_author_cannot_push_a_branch_touching_agent_code(lab):
    """Even if a rogue plan writes into agentics/, the guard refuses the write."""
    author = _workspace(lab, "a04")
    author.sync_main()
    with pytest.raises(ValueError, match="P10 protected path"):
        author.write_file("agentics/config.py", "OWNED = True\n")
    # the workspace file was never created
    assert not (author.path / "agentics" / "config.py").exists()


def test_merge_conflict_fails_gracefully_and_marks_mr_failed(lab):
    mem = lab.mem
    task_id = mem.create_task("conflicting docs", category="docs")
    branch = branch_name("a05", task_id, "conflicting docs")

    author = _workspace(lab, "a05")
    author.sync_main()
    author.checkout_new_branch(branch)
    author.write_file("README.md", "agent edit\n")
    assert author.commit_and_push(branch, "agent(a05): edit readme")

    # owner silently advances main → the merge is no longer a fast-forward
    owner = _workspace(lab, "owner")
    owner.sync_main()
    # the P10 origin hook binds to the committer email: a simulated owner must
    # commit under an owner identity (real owner commits are not "agentics+")
    owner._git("config", "user.email", "owner@local")
    owner.write_file("README.md", "owner edit\n")
    owner._git("add", "-A")
    owner._git("commit", "-m", "owner edit")
    owner._git("push", "origin", "main")

    mr_id = mem.open_review(task_id, "[a05] conflicting docs", "b",
                            author="a05", branch=branch, files=["README.md"])
    mem.set_review_verdict(mr_id, "approved", "a07")
    queue = LocalMergeQueue(mem, auto_merge=True, guard=lab.guard)
    queue.enqueue(mr_id, author="a05")

    integrator = _workspace(lab, "a09")
    for _ in range(LocalMergeQueue.MAX_ATTEMPTS):   # conflict → retries → failed
        merged = queue.drain(
            "a09",
            verify=lambda mr: (True, ""),
            merge=lambda mr: integrator.merge_branch_to_main(
                mem.get_review(mr)["branch"], f"merge MR #{mr}"),
        )
    assert merged == 0
    assert mem.get_review(mr_id)["status"] == "failed"
