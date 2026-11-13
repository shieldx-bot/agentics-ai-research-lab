"""Unit tests for the local-only execution layer (agentics/local_ops.py)."""

import subprocess
from pathlib import Path

import pytest

from agentics.config import Settings
from agentics.local_ops import (
    SURVEY_ANGLES, LocalClaimPolicy, LocalMergeQueue, LocalRepo, TaskSource,
    branch_name,
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
    """Backlog dry again and again must walk every angle, not repeat angle 0."""
    src = TaskSource(mem, settings)
    titles = []
    for _ in range(len(SURVEY_ANGLES)):        # exactly one full rotation
        created = src.supervisor_top_up()
        if created:
            newest = mem.list_tasks(limit=1)[0]
            titles.append(newest["title"])
            mem.set_task_status(newest["id"], "merged")   # dry the backlog again
        else:
            break
    assert len(titles) >= 3                    # distinct angles over time
    assert len(set(titles)) == len(titles)     # never the same angle twice in a row
    angles_logged = [j["detail"] for j in mem.recent_journal(limit=50)
                     if j["event"] == "backlog_top_up"]
    assert len(angles_logged) == len(titles)


def test_verification_angle_task_points_at_stored_findings(settings, mem):
    """P16bis: the appended 'Empirical verification' angle produces a task
    whose body steers the Researcher at the swarm's OWN unverified findings
    (settling claims by running code), not at another re-survey."""
    src = TaskSource(mem, settings)
    created = None
    for _ in range(len(SURVEY_ANGLES)):        # rotation reaches it eventually
        if not src.supervisor_top_up():        # returns a COUNT, not an id
            continue
        task = mem.list_tasks(limit=1)[0]      # newest = the one just opened
        if task["title"].startswith("Empirical verification"):
            created = task
            break
        mem.set_task_status(task["id"], "merged")          # dry the backlog again
    assert created is not None
    assert "stored in shared memory" in created["body"]
    assert "verified_knowledge" in created["body"]


def test_repair_orphan_tasks_reopens_claimed_without_lease(settings, mem):
    src = TaskSource(mem, settings)
    tid = mem.create_task("crashed claim")
    mem.set_task_status(tid, "claimed", claimed_by="a04")   # no claim row at all
    assert src.repair_orphan_tasks() == [tid]
    assert mem.get_task(tid)["status"] == "open"
    events = [j["event"] for j in mem.recent_journal()]
    assert "task_repair" in events


def test_repair_orphan_tasks_keeps_live_claims_and_reviews(settings, mem):
    src = TaskSource(mem, settings)
    held = mem.create_task("held work")
    policy = LocalClaimPolicy(mem, ttl_minutes=60)
    assert policy.try_claim(held, "a04", "agent/a04/x")     # active lease -> untouched
    in_review = mem.create_task("in review with live MR")
    mr = mem.open_review(in_review, "MR", "b", author="a05",
                         branch="agent/a05/x", files=[])
    mem.set_review_verdict(mr, "changes_requested", "a07", "please fix")
    assert src.repair_orphan_tasks() == []
    assert mem.get_task(held)["status"] == "claimed"
    assert mem.get_task(in_review)["status"] == "in_review"


def test_repair_orphan_tasks_reopens_in_review_without_live_mr(settings, mem):
    src = TaskSource(mem, settings)
    stalled = mem.create_task("MR vanished")
    mem.set_task_status(stalled, "in_review")
    failed_mr = mem.open_review(stalled, "MR", "b", author="a05",
                                branch="agent/a05/x", files=[])
    mem.set_review_verdict(failed_mr, "failed", "a09", "tier-2 failed")
    assert src.repair_orphan_tasks() == [stalled]
    assert mem.get_task(stalled)["status"] == "open"


def test_release_never_reopens_a_task_with_a_live_mr(mem):
    """An in_review task has a live MR — release must not reopen it (P5)."""
    tid = mem.create_task("reviewed work")
    mr = mem.open_review(tid, "MR", "b", author="a04",
                         branch="agent/a04/x", files=[])
    mem.set_review_verdict(mr, "changes_requested", "a07", "fix")
    policy = LocalClaimPolicy(mem, ttl_minutes=60)
    policy.release(tid, "a04", "cycle ended without an MR")
    assert mem.get_task(tid)["status"] == "in_review"


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
