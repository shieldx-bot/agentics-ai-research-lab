"""P10 layer 4 — the pre-receive hook on the shared bare origin.

Layers 1-3 of the guard run inside the agent workspace; this hook runs inside
the bare repo at receive-pack time, so a raw `git push` that bypassed the
workspace API is still judged by the same rulebook:

    * owner-authored pushes (committer email not "agentics+*") always pass;
    * agents may never move or delete the default branch — main moves only
      through the Integrator's sanctioned merge (INTEGRATOR_MERGE_ENV);
    * agent commits may not touch a protected path — checked PER COMMIT, so a
      branch rebased on top of fresh owner commits cannot smuggle changes;
    * agent branch deletions are fine.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from agentics.config import Settings
from agentics.local_ops import (
    INTEGRATOR_MERGE_ENV, LocalRepo, origin_hook_script,
)


def _make_source(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    subprocess.run(["git", "init", "-b", "main", str(src)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(src), "config", "user.email", "owner@local"], check=True)
    subprocess.run(["git", "-C", str(src), "config", "user.name", "owner"], check=True)
    (src / "README.md").write_text("seed project\n")
    (src / "agentics").mkdir()
    (src / "agentics" / "engine.py").write_text("ENGINE = True\n")
    subprocess.run(["git", "-C", str(src), "add", "-A"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(src), "commit", "-qm", "seed"], check=True,
                   capture_output=True)
    return src


@pytest.fixture()
def origin(tmp_path: Path) -> Path:
    settings = Settings(data_dir=tmp_path / "data")
    settings.ensure_dirs()
    repo = LocalRepo(settings)
    repo.ensure(src=_make_source(tmp_path))
    return repo.path


def _clone(origin: Path, work: Path, email: str = "agentics+a04@users.noreply.github.com") -> None:
    subprocess.run(["git", "clone", "-q", str(origin), str(work)], check=True,
                   capture_output=True)
    subprocess.run(["git", "-C", str(work), "config", "user.email", email], check=True)
    subprocess.run(["git", "-C", str(work), "config", "user.name", email.split("@")[0]],
                   check=True)


def _commit(work: Path, files: dict[str, str], message: str = "agent work") -> None:
    for rel, content in files.items():
        target = work / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    subprocess.run(["git", "-C", str(work), "add", "-A"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(work), "commit", "-qm", message], check=True,
                   capture_output=True)


def _push(work: Path, refspec: str, bypass: bool = False):
    env = dict(os.environ)
    env.pop(INTEGRATOR_MERGE_ENV, None)
    if bypass:
        env[INTEGRATOR_MERGE_ENV] = "1"
    return subprocess.run(["git", "-C", str(work), "push", "-q", "origin", refspec],
                          capture_output=True, text=True, env=env)


def test_hook_installed_and_executable(origin):
    hook = origin / "hooks" / "pre-receive"
    assert hook.exists() and os.access(hook, os.X_OK)
    assert "agentics P10 origin guard" in hook.read_text(encoding="utf-8")


def test_hook_reinstall_is_idempotent(tmp_path, origin):
    hook = origin / "hooks" / "pre-receive"
    before = hook.read_text(encoding="utf-8")
    mtime = hook.stat().st_mtime_ns
    settings = Settings(data_dir=tmp_path / "data")
    settings.ensure_dirs()
    LocalRepo(settings).ensure_guard()            # same path, same script
    assert hook.read_text(encoding="utf-8") == before == origin_hook_script()
    assert hook.stat().st_mtime_ns == mtime       # identical -> untouched


def test_hook_script_contains_protected_and_owner_rules():
    script = origin_hook_script()
    assert "refs/heads/main" in script
    assert "agentics" in script and "configs" in script   # protected patterns
    assert INTEGRATOR_MERGE_ENV in script                 # sanctioned bypass


def test_agent_push_to_main_is_rejected(origin, tmp_path):
    work = tmp_path / "w-main"
    _clone(origin, work)
    _commit(work, {"research_notes/x.md": "x\n"}, "agent(a04): direct to main")
    push = _push(work, "HEAD:main")
    assert push.returncode != 0
    assert "cannot move 'main'" in push.stderr


def test_integrator_merge_push_to_main_is_sanctioned(origin, tmp_path):
    work = tmp_path / "w-int"
    _clone(origin, work)
    _commit(work, {"experiments/b.py": "b\n"}, "agent(a09): merge MR #1")
    push = _push(work, "HEAD:main", bypass=True)
    assert push.returncode == 0, push.stderr


def test_owner_push_to_main_is_allowed(origin, tmp_path):
    work = tmp_path / "w-owner"
    _clone(origin, work, email="owner@local")
    _commit(work, {"README.md": "owner edit\n"}, "owner edit")
    push = _push(work, "HEAD:main")
    assert push.returncode == 0, push.stderr


def test_agent_protected_paths_rejected_on_first_push(origin, tmp_path):
    work = tmp_path / "w-protected"
    _clone(origin, work)
    _commit(work, {"agentics/engine.py": "ENGINE = 'sabotaged'\n"},
            "agent(a04): sabotage")
    push = _push(work, "HEAD:agent/a04/task-1-sabotage")
    assert push.returncode != 0
    assert "agentics/engine.py" in push.stderr


def test_agent_protected_paths_rejected_when_rebased_on_owner_commits(
        origin, tmp_path):
    """An agent branch rebased on top of fresh OWNER commits must not slip its
    own protected-path changes through the owner-commit escape hatch — the
    check is per agent commit (diff-tree), not per push."""
    work = tmp_path / "w-mixed"
    _clone(origin, work, email="owner@local")
    _commit(work, {"docs/owner-note.md": "owner note\n"}, "owner note")
    assert _push(work, "HEAD:main").returncode == 0          # owner advances main
    # same clone switches to an agent identity and adds the payload
    subprocess.run(["git", "-C", str(work), "config", "user.email",
                    "agentics+a04@users.noreply.github.com"], check=True)
    _commit(work, {"agentics/engine.py": "ENGINE = 'smuggled'\n"}, "agent(a04): payload")
    push = _push(work, "HEAD:agent/a04/task-2-smuggle")
    assert push.returncode != 0
    assert "agentics/engine.py" in push.stderr


def test_clean_agent_branch_push_is_allowed(origin, tmp_path):
    work = tmp_path / "w-clean"
    _clone(origin, work)
    _commit(work, {"research_notes/survey.md": "survey\n"})
    push = _push(work, "HEAD:agent/a06/task-3-survey")
    assert push.returncode == 0, push.stderr


def test_agent_branch_deletion_is_allowed_main_deletion_is_not(origin, tmp_path):
    work = tmp_path / "w-del"
    _clone(origin, work)
    _commit(work, {"research_notes/d.md": "d\n"})
    assert _push(work, "HEAD:agent/a06/task-4-tmp").returncode == 0
    assert _push(work, ":agent/a06/task-4-tmp").returncode == 0
    push = _push(work, ":main")
    assert push.returncode != 0
    assert "cannot be deleted" in push.stderr


def test_exempt_push_carries_the_sanction(origin, tmp_path):
    """An owner-exempted task may push protected paths — via the env var only."""
    work = tmp_path / "w-exempt"
    _clone(origin, work)
    _commit(work, {"agentics/roles.py": "# owner-exempted tweak\n"},
            "agent(a04): exempted evolution")
    rejected = _push(work, "HEAD:agent/a04/task-5-evolve")
    assert rejected.returncode != 0                          # without the flag
    accepted = _push(work, "HEAD:agent/a04/task-5-evolve", bypass=True)
    assert accepted.returncode == 0, accepted.stderr
