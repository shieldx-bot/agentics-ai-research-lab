"""The P10 origin hook (`LocalRepo.ensure_guard` / `origin_hook_script`),
exercised with REAL git pushes against a real bare repo.

Their rulebook (agentics/local_ops.py):
  * owner-authored tips pass untouched;
  * agents (`agentics+*`) cannot move or delete `main`;
  * agent commits may not touch protected paths (per-commit check);
  * a push with INTEGRATOR_MERGE_ENV=1 in the environment is the
    sanctioned Integrator merge and bypasses the agent rules
    (the bare repo is local, so receive-pack inherits the env).
"""

import os
import subprocess
from pathlib import Path

import pytest

from agentics.config import Settings
from agentics.local_ops import LocalRepo
from agentics.policies.guard import INTEGRATOR_MERGE_ENV


@pytest.fixture()
def lab(tmp_path: Path):
    """Bare origin seeded from a tiny checkout + one agent-style clone."""
    settings = Settings(data_dir=tmp_path / "data")
    settings.ensure_dirs()
    src = tmp_path / "src"
    src.mkdir()
    subprocess.run(["git", "init", "-b", "main", str(src)],
                   check=True, capture_output=True)
    (src / "README.md").write_text("seed\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(src), "add", "."], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(src), "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-m", "seed"], check=True, capture_output=True)
    repo = LocalRepo(settings)
    repo.ensure(src=src)                      # installs the P10 origin guard

    clone = tmp_path / "a04"
    subprocess.run(["git", "clone", "-q", str(repo.path), str(clone)],
                   check=True, capture_output=True)
    subprocess.run(["git", "-C", str(clone), "config", "user.email",
                    "agentics+a04@users.noreply.github.com"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(clone), "config", "user.name", "agentics[a04]"],
                   check=True, capture_output=True)
    return repo, clone


def _commit(clone: Path, files: dict[str, str], message: str) -> None:
    for path, content in files.items():
        target = clone / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    subprocess.run(["git", "-C", str(clone), "add", "-A"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(clone), "commit", "-m", message],
                   check=True, capture_output=True)


def _push(clone: Path, *args: str, env_extra: dict[str, str] | None = None):
    env = {k: v for k, v in os.environ.items()
           if k in ("PATH", "HOME", "LANG", "LC_ALL")}
    env["GIT_TERMINAL_PROMPT"] = "0"
    env.update(env_extra or {})
    return subprocess.run(["git", "-C", str(clone), "push", *args],
                          capture_output=True, text=True, env=env)


def test_hook_is_installed_by_ensure_and_self_heals(lab):
    repo, _ = lab
    hook = repo.path / "hooks" / "pre-receive"
    assert hook.exists() and hook.stat().st_mode & 0o111
    assert "agentics" in hook.read_text(encoding="utf-8")
    repo.ensure(src=repo.path.parent)         # second boot: same hook, no drift
    assert hook.exists()


def test_agent_can_push_a_clean_feature_branch(lab):
    _, clone = lab
    subprocess.run(["git", "-C", str(clone), "checkout", "-q", "-b",
                    "agent/a04/task-1-ok"], check=True, capture_output=True)
    _commit(clone, {"docs/note.md": "ok\n", "tests/t.py": "def t():\n    pass\n"},
            "work")
    assert _push(clone, "-u", "origin", "agent/a04/task-1-ok").returncode == 0


@pytest.mark.parametrize("evil", [
    {"agentics/graph/nodes.py": "# tampered\n"},
    {"configs/lab.yaml": "max_agents: 1\n"},
    {"configs/secrets.yaml": "hb: x\n"},
    {"accounts/cookie.json": "{}"},
    {"scripts/evil.py": "print('no')\n"},
    {".github/workflows/x.yml": "on: push\n"},
])
def test_agent_cannot_push_protected_paths(lab, evil):
    _, clone = lab
    subprocess.run(["git", "-C", str(clone), "checkout", "-q", "-b",
                    "agent/a04/task-2-evil"], check=True, capture_output=True)
    _commit(clone, evil, "try the protected paths")
    proc = _push(clone, "-u", "origin", "agent/a04/task-2-evil")
    assert proc.returncode != 0
    assert "guard P10" in proc.stderr


def test_agent_cannot_move_or_delete_main(lab):
    _, clone = lab
    _commit(clone, {"docs/x.md": "x\n"}, "docs only")
    proc = _push(clone, "origin", "HEAD:main")
    assert proc.returncode != 0 and "cannot move 'main'" in proc.stderr

    proc = _push(clone, "origin", ":main")
    assert proc.returncode != 0 and "cannot be deleted" in proc.stderr


def test_sanctioned_integrator_merge_env_bypasses_agent_rules(lab):
    """The sanctioned merge path: merge_branch_to_main pushes main with
    INTEGRATOR_MERGE_ENV=1 (the constant names the env var) — the hook must
    let it through."""
    _, clone = lab
    subprocess.run(["git", "-C", str(clone), "checkout", "-q", "-b",
                    "merge/tmp"], check=True, capture_output=True)
    _commit(clone, {"docs/merged.md": "merged by the integrator\n"}, "merge result")
    proc = _push(clone, "origin", "merge/tmp:main",
                 env_extra={INTEGRATOR_MERGE_ENV: "1"})
    assert proc.returncode == 0, proc.stderr


def test_owner_identity_is_unrestricted(lab):
    _, clone = lab
    subprocess.run(["git", "-C", str(clone), "config", "user.email", "owner@home"],
                   check=True, capture_output=True)
    subprocess.run(["git", "-C", str(clone), "checkout", "-q", "-b", "owner-refactor"],
                   check=True, capture_output=True)
    _commit(clone, {"agentics/new_module.py": "# owner refactoring\n"},
            "owner may edit agent code")
    assert _push(clone, "-u", "origin", "owner-refactor").returncode == 0
    # and main directly, too
    proc = _push(clone, "origin", "owner-refactor:main")
    assert proc.returncode == 0, proc.stderr
