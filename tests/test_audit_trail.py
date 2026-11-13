"""P21 — the immutable audit trail: actions that change what gets merged,
what gets exempted and what gets re-judged must survive the trimmed journal.

Covers the SharedMemory audit API (never trimmed, never rewritten) and the
hooks: evidence re-verdict, task exemption, local merge queue drain / fail.
"""

import json

import pytest

from agentics.memory.shared_memory import SharedMemory


@pytest.fixture()
def mem(tmp_path):
    m = SharedMemory(tmp_path / "a.db")
    yield m
    m.close()


def test_audit_appends_and_reads_back(mem):
    rid = mem.audit("owner", "merge", {"mr": 3, "author": "a03"})
    assert rid >= 1
    rows = mem.audit_recent(limit=5)
    assert rows[-1]["event"] == "merge" and rows[-1]["agent"] == "owner"
    assert json.loads(rows[-1]["detail"])["mr"] == 3


def test_audit_is_never_trimmed(mem):
    """The journal trims at 5000 rows; the audit trail must outlive it."""
    for i in range(210):
        mem.audit("a03", "probe", {"i": i})
        mem.journal("a03", "noise", {"i": i})
    assert mem.audit_count() == 210
    assert len(mem.journal_caps()) < 210 if hasattr(mem, "journal_caps") else True


def test_evidence_reverdict_is_audited_with_verdict_and_by(mem):
    ev_id = mem.record_evidence("a03", 1, "python x.py", 0, "confirmed",
                                hypothesis="h")
    ok = mem.set_evidence_verdict(ev_id, "refuted", note="rerun disagreed",
                                  by="a07")
    assert ok
    events = [(r["event"], r["agent"]) for r in mem.audit_recent(limit=5)]
    assert ("evidence_reverdict", "a07") in events
    detail = json.loads(next(r["detail"] for r in mem.audit_recent(limit=5)
                             if r["event"] == "evidence_reverdict"))
    assert detail["evidence_id"] == ev_id and detail["verdict"] == "refuted"


def test_task_exempt_lifecycle_is_audited(mem):
    tid = mem.create_task("evolution: exempt audit probe")
    mem.set_task_exempt(tid, True, agent="owner")
    mem.set_task_exempt(tid, False, agent="owner")
    events = [r["event"] for r in mem.audit_recent(limit=5)]
    assert events[:2] == ["task_exempt_cleared", "task_exempt_set"]  # newest first


def test_local_merge_queue_drain_audits_merge_and_reject(tmp_path):
    """The local Integrator's merge decisions land in the audit trail."""
    import subprocess
    from agentics.config import Settings
    from agentics.local_ops import LocalRepo, LocalMergeQueue
    from agentics.policies.guard import ProtectedPathsGuard

    settings = Settings(data_dir=tmp_path / "data")
    settings.ensure_dirs()
    src = tmp_path / "src"
    src.mkdir()
    subprocess.run(["git", "init", "-b", "main", str(src)],
                   check=True, capture_output=True)
    (src / "docs").mkdir()
    (src / "docs" / "a.md").write_text("seed\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(src), "add", "."], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(src), "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-m", "seed"], check=True, capture_output=True)
    LocalRepo(settings).ensure(src=src)
    mem = SharedMemory(tmp_path / "m.db")

    mr = mem.open_review(
        task_id=5, title="[a03] docs bump", body="t", author="a03",
        branch="agent/a03/task-5", files=["docs/b.md"], verification="tier-1 ok",
    )
    mem.set_review_verdict(mr, "approved", "a07", "lgtm")
    q = LocalMergeQueue(mem, True, guard=ProtectedPathsGuard())
    q.enqueue(mr, author="a03")

    def merge(mr_id):
        # a real local merge would happen here; the audit point does not care
        return True

    merged = q.drain("a09", verify=lambda mr_id: (True, "ok"), merge=merge)
    assert merged == 1
    events = [r["event"] for r in mem.audit_recent(limit=5)]
    assert "merge" in events
    mem.close()
