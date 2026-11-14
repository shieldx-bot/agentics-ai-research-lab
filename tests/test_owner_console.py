"""Owner console additions — `agentics watch` TUI + `agentics mr` merge gate.

These are the deltas contributed on top of the local-mode implementation
(after cross-checking it line by line): a terminal live view and the owner
side of the AUTO_MERGE=false flow — the queue records readiness but nothing
ever merged; `mr approve` / `mr merge` complete that interactive loop.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agentics.config import Settings
from agentics.github_ops.repo_workspace import RepoWorkspace
from agentics.local_ops import LocalRepo
from agentics.memory.shared_memory import SharedMemory
from agentics.policies.guard import ProtectedPathsGuard


# ------------------------------------------------------------ fixtures
@pytest.fixture()
def settings(tmp_path: Path):
    s = Settings(data_dir=tmp_path / "data")
    s.ensure_dirs()
    return s


@pytest.fixture()
def lab(tmp_path: Path):
    """Same seeded lab as test_local_review_merge: real git, bare repo."""
    src = tmp_path / "src"
    src.mkdir()
    subprocess.run(["git", "init", "-b", "main", str(src)], check=True,
                   capture_output=True)
    (src / "README.md").write_text("seed project\n", encoding="utf-8")
    (src / "experiments").mkdir()
    (src / "experiments" / ".gitkeep").write_text("", encoding="utf-8")
    subprocess.run(["git", "-C", str(src), "add", "."], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(src), "-c", "user.email=t@t", "-c", "user.name=t",
         "commit", "-m", "seed"], check=True, capture_output=True)

    s = Settings(data_dir=tmp_path / "data")
    s.ensure_dirs()
    LocalRepo(s).ensure(src=src)
    mem = SharedMemory(s.db_path)          # same DB the CLI commands open
    try:
        yield {"settings": s, "mem": mem, "guard": ProtectedPathsGuard()}
    finally:
        mem.close()


def _run(app, *args):
    return CliRunner().invoke(app, [*args])


# ------------------------------------------------------------ watch
def test_watch_once_renders_agents_intents_and_journal(settings, monkeypatch):
    from agentics.cli import app

    mem = SharedMemory(settings.db_path)
    mem.report("a04", "coder", node="implement", status="working", cycle=17,
               task_id=1, task_title="Write NFV survey",
               intent={"goal": "Write survey notes from cited papers",
                       "why": "category=research score=0.87", "steps": []})
    mem.journal("a04", "intent", {"goal": "survey notes", "task": 1})
    mem.journal("a04", "protection", "blocked: agentics/orchestrator.py")
    mem.close()

    monkeypatch.setattr("agentics.cli._settings", lambda: settings)
    result = _run(app, "watch", "--once")
    assert result.exit_code == 0, result.output
    assert "a04" in result.output
    assert "survey notes" in result.output           # the live goal
    assert "0.87" in result.output                   # the WHY
    assert "protection" in result.output             # journal feed


def test_watch_once_works_on_an_empty_lab(settings, monkeypatch):
    from agentics.cli import app

    monkeypatch.setattr("agentics.cli._settings", lambda: settings)
    result = _run(app, "watch", "--once")
    assert result.exit_code == 0, result.output


def test_watch_flags_manual_merge_when_auto_merge_off(settings, monkeypatch):
    from agentics.cli import app

    mem = SharedMemory(settings.db_path)
    mem.report("a09", "integrator", node="integrate", cycle=3,
               intent={"goal": "drain queue"})
    mem.mq_enqueue(4, "a04")
    mem.close()
    settings.auto_merge = False

    monkeypatch.setattr("agentics.cli._settings", lambda: settings)
    result = _run(app, "watch", "--once")
    assert result.exit_code == 0, result.output
    assert "mr merge" in result.output


# ------------------------------------------------------------ mr list / approve
def test_mr_list_and_approve_by_owner(lab, monkeypatch):
    from agentics.cli import app

    mem = lab["mem"]
    mr = mem.open_review(1, "Add benchmark", "body", "a04",
                         "agent/a04/task-1-add-benchmark", ["bench/run.py"])
    monkeypatch.setattr("agentics.cli._settings", lambda: lab["settings"])

    listing = _run(app, "mr", "list")
    assert listing.exit_code == 0 and "Add benchmark" in listing.output

    result = _run(app, "mr", "approve", str(mr), "--note", "looks solid")
    assert result.exit_code == 0, result.output
    review = mem.get_review(mr)
    assert review["status"] == "approved"
    assert review["approved_by"] == "owner"
    assert mem.mq_pending(), "owner approval enqueues the MR"
    journal = mem.recent_journal(limit=5, agent="owner")
    assert journal and journal[-1]["event"] == "review"


def test_mr_approve_rejects_unknown_mr(lab, monkeypatch):
    from agentics.cli import app

    monkeypatch.setattr("agentics.cli._settings", lambda: lab["settings"])
    result = _run(app, "mr", "approve", "999")
    assert result.exit_code == 1


# ------------------------------------------------------------ mr merge (real git)
def _make_mr(lab, agent_id: str, path: str, content: str) -> int:
    """Author a real branch + MR against the lab's bare repo."""
    mem = lab["mem"]
    task_id = mem.create_task(f"work on {path}", "", "research")
    ws = RepoWorkspace(agent_id, lab["settings"], guard=lab["guard"])
    ws.sync_main()
    branch = f"agent/{agent_id}/task-{task_id}-work"
    ws.checkout_new_branch(branch)
    ws.write_file(path, content)
    assert ws.commit_and_push(branch, f"agent({agent_id}): {path} (#{task_id})")
    return mem.open_review(
        task_id, f"[{agent_id}] work (#{task_id})", "verification: tier-1 ok",
        agent_id, branch, [path])


def test_mr_merge_completes_the_owner_loop(lab, monkeypatch):
    from agentics.cli import app

    mem = lab["mem"]
    mr = _make_mr(lab, "a04", "experiments/nfv-bench.md", "# NFV bench\nresult: ok\n")
    mem.set_review_verdict(mr, "approved", "a07", "cross-review ok")
    mem.mq_enqueue(mr, "a04")
    task_id = mem.get_review(mr)["task_id"]

    monkeypatch.setattr("agentics.cli._settings", lambda: lab["settings"])
    result = _run(app, "mr", "merge", str(mr))
    assert result.exit_code == 0, result.output

    review = mem.get_review(mr)
    assert review["status"] == "merged"
    assert mem.get_task(task_id)["status"] == "merged"
    assert not mem.mq_pending(), "queue drained for this MR"
    # the file really landed on the bare repo's main
    bare = lab["settings"].data_dir / "lab.git"
    blob = subprocess.run(
        ["git", "--git-dir", str(bare), "show", "main:experiments/nfv-bench.md"],
        capture_output=True, text=True, check=False).stdout
    assert "result: ok" in blob
    journal = mem.recent_journal(limit=5, agent="owner")
    assert any(e["event"] == "merge" for e in journal)


def test_mr_merge_refuses_unapproved_mr(lab, monkeypatch):
    from agentics.cli import app

    mem = lab["mem"]
    mr = _make_mr(lab, "a04", "experiments/x.md", "x\n")
    monkeypatch.setattr("agentics.cli._settings", lambda: lab["settings"])
    result = _run(app, "mr", "merge", str(mr))
    assert result.exit_code == 1
    assert "approve it first" in result.output


def test_mr_merge_enforces_p10_even_for_the_owner(lab, monkeypatch):
    from agentics.cli import app

    mem = lab["mem"]
    # an MR whose diff touches the lab's own code, WITHOUT the owner exemption
    ws = RepoWorkspace("a04", lab["settings"], guard=None)   # guard bypassed at
    ws.sync_main()                                           # write time on purpose:
    branch = "agent/a04/task-1-agent-code"                   # simulates a crafted push
    ws.checkout_new_branch(branch)
    (ws.path / "experiments").mkdir(exist_ok=True)
    ws.write_file("experiments/ok.md", "ok\n")
    # push a change to a protected path directly with git (bypassing write_file)
    target = ws.path / "agentics" / "planted.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("# planted\n")
    subprocess.run(["git", "-C", str(ws.path), "add", "-A"], check=True,
                   capture_output=True)
    subprocess.run(["git", "-C", str(ws.path), "-c", "user.email=a@b",
                    "-c", "user.name=a", "commit", "-qm", "craft"], check=True,
                   capture_output=True)
    subprocess.run(["git", "-C", str(ws.path), "push", "-q", "origin",
                    f"HEAD:refs/heads/{branch}"], check=True, capture_output=True)

    mr = mem.open_review(mem.create_task("crafted", "", "research"),
                         "crafted MR", "", "a04", branch,
                         ["experiments/ok.md", "agentics/planted.py"])
    mem.set_review_verdict(mr, "approved", "a07", "ok")
    mem.mq_enqueue(mr, "a04")

    monkeypatch.setattr("agentics.cli._settings", lambda: lab["settings"])
    result = _run(app, "mr", "merge", str(mr))
    assert result.exit_code == 1
    assert "P10" in result.output
    assert mem.get_review(mr)["status"] == "approved", "MR left untouched"
