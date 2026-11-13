"""P10 layer 4 — the reviewer and the integrator never trust the file list
the author's workspace recorded at open_mr time. Each workspace is a separate
trust domain, so both re-scan the branch that was ACTUALLY pushed
(`RepoWorkspace.branch_files_vs_main`) before approval / merge:

    1. branch_files_vs_main reports the real pushed diff (real git).
    2. node_review fast-rejects a branch whose REAL diff touches protected
       paths — even when the recorded file list looks clean — without
       spending an LLM call.
    3. the reviewer's prompt is built from the actual branch files.
    4. LocalMergeQueue.drain re-scans via `scan_branch` and refuses to merge
       a poisoned branch; owner exemptions (task-level) are still honored,
       but accounts/ + secrets stay blocked for everyone.
"""

import asyncio
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentics.agents.roles import Role
from agentics.config import Settings
from agentics.github_ops.repo_workspace import RepoWorkspace
from agentics.graph import nodes
from agentics.local_ops import LocalMergeQueue, LocalRepo, branch_name
from agentics.memory.shared_memory import SharedMemory
from agentics.policies.guard import ProtectedPathsGuard


@pytest.fixture()
def lab(tmp_path: Path):
    """A seeded local lab: bare repo + memory + guard (real git)."""
    src = tmp_path / "src"
    src.mkdir()
    subprocess.run(["git", "init", "-b", "main", str(src)], check=True, capture_output=True)
    (src / "README.md").write_text("seed project\n", encoding="utf-8")
    (src / "docs").mkdir()
    (src / "docs" / ".gitkeep").write_text("", encoding="utf-8")
    subprocess.run(["git", "-C", str(src), "add", "."], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(src), "-c", "user.email=t@t", "-c", "user.name=t",
         "commit", "-m", "seed"], check=True, capture_output=True)

    settings = Settings(data_dir=tmp_path / "data")
    settings.ensure_dirs()
    LocalRepo(settings).ensure(src=src)
    mem = SharedMemory(tmp_path / "m.db")
    try:
        yield Simple(settings=settings, mem=mem, guard=ProtectedPathsGuard(),
                     repo=LocalRepo(settings))
    finally:
        mem.close()


class Simple:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _workspace(lab, agent_id: str) -> RepoWorkspace:
    return RepoWorkspace(agent_id, lab.settings, guard=lab.guard)


def _review_ctx(lab, reviewer: RepoWorkspace):
    return SimpleNamespace(
        agent_id="a07", role=Role.REVIEWER, memory=lab.mem,
        workspace=reviewer,
        local=SimpleNamespace(guard=lab.guard),
        merge_queue=LocalMergeQueue(lab.mem, auto_merge=True, guard=lab.guard),
        report=lambda **kw: None,
        heartbeat=lambda status="working": None,
        settings=SimpleNamespace(research_topic="NFV"),
        meta={"personal": SimpleNamespace(add_lesson=lambda *a, **k: None)},
        llm=None,
    )


class CannedAnswer:
    def __init__(self, payload: dict, seen_prompts: list):
        self.answer = json.dumps(payload)
        self.citations = []
        self._seen = seen_prompts

    @property
    def prompts(self):
        return self._seen


def _canned_llm(payload: dict, seen_prompts: list):
    async def ask(query, *, mode="concise", system=None, purpose=None):
        seen_prompts.append(query)
        return CannedAnswer(payload, seen_prompts)
    return SimpleNamespace(account=SimpleNamespace(name="acc-1"), ask=ask)


def _push_clean_branch(lab, agent_id: str, task_id: int) -> str:
    branch = branch_name(agent_id, task_id, "docs work")
    author = _workspace(lab, agent_id)
    author.sync_main()
    author.checkout_new_branch(branch)
    author.write_file("docs/notes.md", "findings go here\n")
    assert author.commit_and_push(branch, f"agent({agent_id}): docs (#{task_id})")
    return branch


def test_branch_files_vs_main_reports_pushed_changes(lab):
    """The reviewer's own clone sees exactly what the author pushed."""
    task_id = lab.mem.create_task("docs work", category="docs")
    branch = _push_clean_branch(lab, "a02", task_id)

    reviewer = _workspace(lab, "a07")
    files = reviewer.branch_files_vs_main(branch)
    assert files == ["docs/notes.md"]

    # a branch that does not exist scans as empty (never raises)
    assert reviewer.branch_files_vs_main("agent/ghost/task-999-nope") == []


def test_reviewer_fast_rejects_branch_with_unreported_protected_files(lab, monkeypatch):
    """Layer 4 core: the recorded file list says 'docs only', but the pushed
    branch also touches agentics/ — request_changes WITHOUT an LLM call."""
    task_id = lab.mem.create_task("docs work", category="docs")
    branch = _push_clean_branch(lab, "a02", task_id)

    # poison the pushed branch directly (simulating a write that sneaked past
    # layer 1 in the author's workspace)
    author = _workspace(lab, "a02")
    (author.path / "agentics").mkdir(exist_ok=True)
    (author.path / "agentics" / "sneak.py").write_text("OWNED = True\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(author.path), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(author.path), "-c", "user.email=t@t", "-c", "user.name=t",
         "commit", "-m", "sneak"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(author.path), "push", "origin", branch],
                   check=True, capture_output=True)

    # the MR lies: recorded files mention docs/notes.md only
    mr_id = lab.mem.open_review(
        task_id=task_id, title="[a02] docs work", body="", author="a02",
        branch=branch, files=["docs/notes.md"], verification="tier-1 passed")

    def _no_llm(*a, **k):
        pytest.fail("layer 4 must reject protected diffs before the LLM is called")

    monkeypatch.setattr(nodes, "_llm_ask", _no_llm)
    state = __import__("asyncio").run(nodes.node_review(_review_ctx(lab, _workspace(lab, "a07")), {}))

    assert state["review_verdict"] == "request_changes"
    assert "P10 protected paths" in state["review_comment"]
    assert "agentics/sneak.py" in state["review_comment"]
    assert lab.mem.get_panel(mr_id)["state"] == "defense"   # P18: author must defend
    assert lab.mem.get_review(mr_id)["status"] == "open"
    assert lab.mem.mq_pending() == []


def test_reviewer_prompt_uses_actual_branch_files(lab):
    """With a clean branch the reviewer approves — and the prompt shows the
    files the branch REALLY changes, not the author's recorded list."""
    task_id = lab.mem.create_task("docs work", category="docs")
    branch = _push_clean_branch(lab, "a02", task_id)
    mr_id = lab.mem.open_review(
        task_id=task_id, title="[a02] docs work", body="", author="a02",
        branch=branch, files=["docs/stale-listing.md"],   # stale/wrong list
        verification="tier-1 passed")

    seen_prompts: list = []
    ctx = _review_ctx(lab, _workspace(lab, "a07"))
    ctx.llm = _canned_llm({"verdict": "approve", "comment": "lgtm"}, seen_prompts)

    import asyncio
    state = asyncio.run(nodes.node_review(ctx, {}))

    assert state["review_verdict"] == "approve"
    assert any("docs/notes.md" in p for p in seen_prompts)      # real diff
    assert not any("stale-listing" in p for p in seen_prompts)  # recorded lie
    assert lab.mem.get_review(mr_id)["status"] == "approved"
    assert lab.mem.mq_pending() and lab.mem.mq_pending()[0]["pr_number"] == mr_id


def test_integrator_drain_re_scans_branch_instead_of_recorded_files(lab):
    """Layer 4 at merge time: drain trusts the scan_branch result over the
    recorded list — a poisoned branch is refused even with a clean record."""
    task_id = lab.mem.create_task("docs work", category="docs")
    branch = _push_clean_branch(lab, "a02", task_id)
    author = _workspace(lab, "a02")
    (author.path / "agentics").mkdir(exist_ok=True)
    (author.path / "agentics" / "sneak.py").write_text("OWNED = True\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(author.path), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(author.path), "-c", "user.email=t@t", "-c", "user.name=t",
         "commit", "-m", "sneak"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(author.path), "push", "origin", branch],
                   check=True, capture_output=True)

    mr_id = lab.mem.open_review(
        task_id=task_id, title="[a02] docs work", body="", author="a02",
        branch=branch, files=["docs/notes.md"], verification="tier-1 passed")
    queue = LocalMergeQueue(lab.mem, auto_merge=True, guard=lab.guard)
    queue.enqueue(mr_id, author="a02")

    integrator = _workspace(lab, "a09")
    merged_calls: list = []
    drained = queue.drain(
        "a09",
        verify=lambda mr: (True, ""),
        merge=lambda mr: merged_calls.append(mr) or True,
        scan_branch=integrator.branch_files_vs_main,
    )
    assert drained == 0 and merged_calls == []
    assert lab.mem.get_review(mr_id)["status"] == "failed"
    assert "P10 protected paths" in (lab.mem.get_review(mr_id)["verdict_comment"] or "")

    # without the scanner (old behavior) the queue only sees the recorded
    # list — the same clean-looking record lets the poisoned branch merge.
    # That gap is exactly what layer 4 closes.
    mr2 = lab.mem.open_review(
        task_id=task_id, title="[a02] docs work 2", body="", author="a02",
        branch=branch, files=["docs/notes.md"], verification="tier-1 passed")
    queue2 = LocalMergeQueue(lab.mem, auto_merge=True, guard=lab.guard)
    queue2.enqueue(mr2, author="a02")
    assert queue2.drain("a09", verify=lambda mr: (True, ""),
                        merge=lambda mr: True) == 1
    assert lab.mem.get_review(mr2)["status"] == "merged"


def test_layer4_honors_owner_exemption_but_never_accounts(lab):
    """An owner-exempted task may touch default protected paths; accounts/
    and secrets stay hard-blocked even for exempt tasks."""
    task_id = lab.mem.create_task("refactor agent tools", category="code")
    lab.mem.set_task_exempt(task_id, True, agent="owner")
    branch = _push_clean_branch(lab, "a02", task_id)
    lab.mem.open_review(
        task_id=task_id, title="[a02] docs work", body="", author="a02",
        branch=branch, files=["docs/notes.md"], verification="tier-1 passed")

    seen_prompts: list = []
    ctx = _review_ctx(lab, _workspace(lab, "a07"))
    ctx.llm = _canned_llm({"verdict": "approve", "comment": "ok"}, seen_prompts)

    # fake scan: the branch actually touches agentics/ — exempt task passes
    ctx.workspace = SimpleNamespace(
        branch_files_vs_main=lambda branch_: ["agentics/tools/impl.py"])
    state = asyncio.run(nodes.node_review(ctx, {}))
    assert state["review_verdict"] == "approve"

    # ALWAYS_PROTECTED still applies: accounts/ is rejected even when exempt
    mr2 = lab.mem.open_review(
        task_id=task_id, title="[a02] docs work 2", body="", author="a02",
        branch=branch, files=["docs/notes.md"], verification="tier-1 passed")
    ctx2 = _review_ctx(lab, _workspace(lab, "a07"))
    ctx2.llm = _canned_llm({"verdict": "approve", "comment": "ok"}, seen_prompts)
    ctx2.workspace = SimpleNamespace(
        branch_files_vs_main=lambda branch_: ["accounts/cookies.json"])
    state2 = asyncio.run(nodes.node_review(ctx2, {}))
    assert state2["review_verdict"] == "request_changes"
    assert "accounts/cookies.json" in state2["review_comment"]
    assert lab.mem.get_panel(mr2)["state"] == "defense"     # P18: author must defend