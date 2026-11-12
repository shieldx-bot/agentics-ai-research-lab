"""Unit tests for the local-only execution layer (agentics/local_ops.py)."""

import subprocess
from pathlib import Path

import pytest

from agentics.config import Settings
from agentics.local_ops import (
    LocalClaimPolicy, LocalMergeQueue, LocalRepo, TaskSource, branch_name,
)
from agentics.memory.shared_memory import SharedMemory
from agentics.policies.guard import ProtectedPathsGuard


@pytest.fixture()
def mem(tmp_path: Path):
    m = SharedMemory(tmp_path / "m.db")
    yield m
    m.close()


@pytest.fixture()
def settings(tmp_path: Path):
    s = Settings(data_dir=tmp_path / "data")
    s.ensure_dirs()
    return s


@pytest.fixture()
def project(tmp_path: Path):
    """A tiny 'lab checkout' to seed the bare repo from."""
    p = tmp_path / "src"
    p.mkdir()
    subprocess.run(["git", "init", "-b", "main", str(p)], check=True, capture_output=True)
    (p / "README.md").write_text("seed\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(p), "add", "."], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(p), "-c", "user.email=t@t", "-c", "user.name=t",
         "commit", "-m", "seed"], check=True, capture_output=True)
    return p


def test_branch_name_is_namespaced_and_slugged():
    assert branch_name("a04", 12, "Fix: the Crash (again)!") == \
        "agent/a04/task-12-fix-the-crash-again"


def test_local_repo_seeds_once_and_is_idempotent(settings, project):
    repo = LocalRepo(settings)
    repo.ensure(src=project)
    assert (repo.path / "HEAD").exists()
    assert repo.branch_exists("main")
    head1 = repo.main_head()
    repo.ensure(src=project)          # second call is a no-op
    assert repo.main_head() == head1


def test_local_repo_refuses_non_git_source(settings, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(RuntimeError, match="not a git checkout"):
        LocalRepo(settings).ensure(src=empty)


def _git(*args: str, env_extra: dict | None = None, cwd: Path | None = None) -> subprocess.CompletedProcess:
    import os
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t",
               GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
    if env_extra:
        env.update(env_extra)
    return subprocess.run(["git", *args], cwd=cwd, env=env,
                          capture_output=True, text=True)


def _clone_and_commit(bare: Path, work: Path) -> None:
    _git("clone", "-q", str(bare), str(work))
    (work / "reports").mkdir(exist_ok=True)
    (work / "reports" / "ok.md").write_text("fine\n", encoding="utf-8")
    _git("add", ".", cwd=work)
    _git("commit", "-qm", "research work", cwd=work)


def test_pre_receive_hook_blocks_direct_push_of_lab_code(settings, project, tmp_path):
    """P10 layer 2.5 (defence in depth): even a raw `git push` straight to
    data/lab.git — no workspace, no review, no merge queue — cannot introduce
    agentics/configs/accounts/scripts/.github paths into the origin."""
    repo = LocalRepo(settings)
    repo.ensure(src=project)
    work = tmp_path / "work"
    _clone_and_commit(repo.path, work)

    (work / "agentics").mkdir()
    (work / "agentics" / "rogue.py").write_text("x = 1\n", encoding="utf-8")
    _git("add", ".", cwd=work)
    _git("commit", "-qm", "sneak lab code", cwd=work)
    refused = _git("push", "-q", "origin", "HEAD:refs/heads/main", cwd=work)
    assert refused.returncode != 0
    assert "protected lab paths" in refused.stderr
    # the refused commit never landed
    assert repo.main_head() != _git("rev-parse", "HEAD", cwd=work).stdout.strip()

    # drop the refused commit, then a clean research-only push goes through
    _git("reset", "-q", "--hard", "origin/main", cwd=work)
    (work / "reports").mkdir(exist_ok=True)
    (work / "reports" / "more.md").write_text("more\n", encoding="utf-8")
    _git("add", ".", cwd=work)
    _git("commit", "-qm", "more research", cwd=work)
    ok = _git("push", "-q", "origin", "HEAD:refs/heads/main", cwd=work)
    assert ok.returncode == 0
    assert repo.main_head() == _git("rev-parse", "HEAD", cwd=work).stdout.strip()


def test_pre_receive_hook_honours_owner_env_override(settings, project, tmp_path):
    """ALLOW_PROTECTED_PATHS=1 in the pushing shell lifts the git-layer block,
    mirroring the app-layer override in policies/guard.py."""
    repo = LocalRepo(settings)
    repo.ensure(src=project)
    work = tmp_path / "work"
    _clone_and_commit(repo.path, work)
    (work / "agentics").mkdir()
    (work / "agentics" / "owner_backport.py").write_text("x = 1\n", encoding="utf-8")
    _git("add", ".", cwd=work)
    _git("commit", "-qm", "owner backport", cwd=work)
    ok = _git("push", "-q", "origin", "HEAD:refs/heads/main", cwd=work,
              env_extra={"ALLOW_PROTECTED_PATHS": "1"})
    assert ok.returncode == 0


def test_pre_receive_hook_never_clobbers_owner_hook(settings, project):
    """A hook that does not carry the agentics sentinel belongs to the owner
    and must survive ensure() untouched."""
    repo = LocalRepo(settings)
    repo.ensure(src=project)
    hook = repo.path / "hooks" / "pre-receive"
    hook.write_text("#!/bin/sh\nexit 0  # mine\n", encoding="utf-8")
    repo.ensure(src=project)                    # repair path must not rewrite
    assert "mine" in hook.read_text(encoding="utf-8")
    assert repo.install_pre_receive().read_text(encoding="utf-8") == \
        "#!/bin/sh\nexit 0  # mine\n"


def test_pre_receive_hook_updated_when_settings_change(settings, project):
    repo = LocalRepo(settings)
    repo.ensure(src=project)
    hook = repo.path / "hooks" / "pre-receive"
    settings.protected_paths = list(settings.protected_paths) + ["docs/"]
    repo.ensure(src=project)                    # repair path picks up new list
    assert "docs/" in hook.read_text(encoding="utf-8")


def test_task_crud_and_scan_ordering(mem):
    t1 = mem.create_task("first survey", body="b1", category="research")
    t2 = mem.create_task("second docs", body="b2", category="docs")
    open_tasks = mem.scan_open_tasks()
    assert [t["id"] for t in open_tasks] == [t1, t2]
    # claimed tasks leave the scan pool
    mem.set_task_status(t1, "claimed", claimed_by="a04")
    assert [t["id"] for t in mem.scan_open_tasks()] == [t2]
    assert mem.get_task(t1)["claimed_by"] == "a04"
    assert mem.task_counts() == {"claimed": 1, "open": 1}


def test_create_task_if_absent_dedupes_against_open_work(mem):
    first = mem.create_task("SOTA survey")
    assert mem.create_task_if_absent("SOTA survey") is None
    mem.set_task_status(first, "merged")          # closed work does not block
    again = mem.create_task_if_absent("SOTA survey")
    assert again is not None and again > first


def test_set_task_exempt_journals_the_owner_action(mem):
    tid = mem.create_task("evolution: prompts")
    mem.set_task_exempt(tid, True, agent="owner")
    assert mem.get_task(tid)["exempt"] is True
    events = [j["event"] for j in mem.recent_journal()]
    assert "task_exempt_set" in events
    mem.set_task_exempt(tid, False, agent="owner")
    assert mem.get_task(tid)["exempt"] is False


def test_claim_policy_transitions_task_status(mem):
    tid = mem.create_task("code task")
    policy = LocalClaimPolicy(mem, ttl_minutes=60)
    branch = branch_name("a04", tid, "code task")
    assert policy.try_claim(tid, "a04", branch)
    assert mem.get_task(tid)["status"] == "claimed"
    assert mem.active_claims()[tid].agent == "a04"
    # someone else cannot steal a live claim
    assert not policy.try_claim(tid, "a05")
    # completing the handoff releases the lease (task moves to in_review when
    # the MR opens — node_open_pr calls open_review)
    policy.complete(tid, "a04", branch)
    assert tid not in mem.all_active_claims()
    # release puts the task back into the open pool
    policy.release(tid, "a04", "yielded")
    assert mem.get_task(tid)["status"] == "open"


def test_claim_sweep_releases_stale_leases(mem):
    tid = mem.create_task("stale work")
    policy = LocalClaimPolicy(mem, ttl_minutes=60)
    assert policy.try_claim(tid, "a04", "agent/a04/x")
    # forge an expired lease
    with mem._lock, mem._conn:
        mem._conn.execute("UPDATE claims SET expires_at=1 WHERE issue=?", (tid,))
    freed = policy.sweep_stale()
    assert freed == [tid]
    assert mem.get_task(tid)["status"] == "open"


def test_task_source_seeds_backlog_once(settings, mem):
    src = TaskSource(mem, settings)
    src.ensure_backlog()
    seeded = mem.scan_open_tasks()
    assert len(seeded) == 3
    src.ensure_backlog()                       # idempotent
    assert len(mem.scan_open_tasks()) == 3
    assert all(t["category"] in ("research", "code", "docs") for t in seeded)


def test_task_source_supervisor_tops_up_when_dry(settings, mem):
    src = TaskSource(mem, settings)
    assert src.supervisor_top_up() == 1        # empty backlog -> survey task
    assert src.supervisor_top_up() == 0        # still open -> nothing new


def test_supervisor_top_up_rotates_survey_angles(settings, mem):
    """Regression (cd8b536 review): top-up used to re-open SURVEY_ANGLES[0]
    forever — the backlog dried, the same survey was re-created, and the other
    five angles were never reached. Least-used-first rotation fixes that."""
    from agentics.local_ops import SURVEY_ANGLES

    src = TaskSource(mem, settings)
    first = src.supervisor_top_up()
    assert first == 1
    t1 = mem.scan_open_tasks()[0]
    assert t1["title"].startswith(SURVEY_ANGLES[0] + ":")

    mem.set_task_status(t1["id"], "merged")    # backlog dry again
    assert src.supervisor_top_up() == 1
    titles = [t["title"] for t in mem.list_tasks(limit=10)]
    assert sum(t.startswith(a + ":") for a in SURVEY_ANGLES for t in titles) == 2
    # the second angle differs from the first and no angle repeats
    angles_used = [a for a in SURVEY_ANGLES
                   if sum(t.startswith(a + ":") for t in titles) >= 1]
    assert len(angles_used) == 2               # two distinct angles so far

    mem.set_task_status(mem.scan_open_tasks()[0]["id"], "merged")
    assert src.supervisor_top_up() == 1
    titles = [t["title"] for t in mem.list_tasks(limit=10)]
    angles_used = [a for a in SURVEY_ANGLES
                   if sum(t.startswith(a + ":") for t in titles) >= 1]
    assert len(angles_used) == 3               # rotation keeps advancing


def test_supervisor_top_up_investigates_newest_failure_first(settings, mem):
    """GEA: the newest failed trace with no task yet becomes a bugfix task
    before any survey angle is opened; an already-investigated failure
    (merged task) is never re-opened."""
    src = TaskSource(mem, settings)
    trace_id = mem.add_trace(
        agent="a02", role="coder", cycle=3, outcome="failed",
        category="tool", attempted="run benchmark", failure="ModuleNotFoundError: no module named 'sklearn'",
    )
    assert src.supervisor_top_up() == 1
    task = mem.scan_open_tasks()[0]
    assert task["category"] == "bugfix"
    assert task["title"].startswith("Investigate failure (tool):")
    assert "ModuleNotFoundError" in task["title"]
    assert task["body"].startswith("Self-generated from the group failure traces")

    mem.set_task_status(task["id"], "merged")  # investigation done & merged
    assert src.supervisor_top_up() == 1        # falls through to the survey angle
    titles = [t["title"] for t in mem.list_tasks(limit=10)]
    assert sum(t.startswith("Investigate failure (tool):") for t in titles) == 1

    _ = trace_id


def test_merge_queue_drains_approved_mrs_serially(mem):
    tid = mem.create_task("some work")
    mr = mem.open_review(tid, "MR title", "body", author="a04",
                         branch="agent/a04/task-1-x", files=["experiments/a.py"])
    mem.mq_enqueue(mr, author="a04")
    merged_into = []
    queue = LocalMergeQueue(mem, auto_merge=True, guard=ProtectedPathsGuard())

    def fake_merge(mr_id):
        review = mem.get_review(mr_id)
        merged_into.append((mr_id, review["branch"]))
        return True

    drained = queue.drain("a09", verify=lambda mr: (True, ""), merge=fake_merge)
    assert drained == 1
    assert merged_into == [(mr, "agent/a04/task-1-x")]
    assert mem.get_review(mr)["status"] == "merged"
    assert mem.get_task(tid)["status"] == "merged"
    assert mem.mq_pending() == []


def test_merge_queue_fails_after_repeated_merge_conflicts(mem):
    tid = mem.create_task("conflicting work")
    mr = mem.open_review(tid, "MR", "body", author="a04",
                         branch="agent/a04/task-1-x", files=[])
    mem.mq_enqueue(mr, author="a04")
    queue = LocalMergeQueue(mem, auto_merge=True, guard=ProtectedPathsGuard())
    assert queue.drain("a09", verify=lambda mr: (True, ""),
                       merge=lambda mr: False) == 0
    assert queue.drain("a09", verify=lambda mr: (True, ""),
                       merge=lambda mr: False) == 0
    queue.drain("a09", verify=lambda mr: (True, ""), merge=lambda mr: False)
    assert mem.get_review(mr)["status"] == "failed"
    assert mem.get_task(tid)["status"] == "failed"


def test_merge_queue_with_auto_merge_off_only_marks_ready(mem):
    tid = mem.create_task("manual work")
    mr = mem.open_review(tid, "MR", "body", author="a04",
                         branch="agent/a04/task-1-x", files=[])
    mem.mq_enqueue(mr, author="a04")
    queue = LocalMergeQueue(mem, auto_merge=False, guard=ProtectedPathsGuard())
    assert queue.drain("a09", verify=lambda mr: (True, "")) == 0
    assert mem.get_review(mr)["status"] == "approved"   # waiting for the owner
    assert mem.get_task(tid)["status"] == "in_review"
