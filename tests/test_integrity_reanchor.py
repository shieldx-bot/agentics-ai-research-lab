"""Integrity baseline re-anchoring (live 2026-11-14 07:08):

`runtime integrity drift: 5 file(s)` fired right after `git pull --autostash`
— the 5 drifted files were EXACTLY the files the pull changed. The tree
changed because HEAD moved; that is expected operation, not tampering, and
must re-anchor the manifest instead of crying wolf. Drift WITHOUT a HEAD
move (an escaped process, a manual edit no commit explains) stays an alert.
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402

from agentics.sandbox.integrity import (  # noqa: E402
    IntegrityGuard,
    build_manifest,
)


def _git(root: Path, *args: str) -> str:
    proc = subprocess.run(["git", "-C", str(root), *args],
                          check=True, capture_output=True, text=True)
    return proc.stdout.strip()


@pytest.fixture()
def repo(tmp_path):
    """A minimal git repo shaped like the project: agentics/ dir + one file."""
    root = tmp_path / "lab"
    (root / "agentics").mkdir(parents=True)
    (root / "agentics" / "nodes.py").write_text("print('v1')\n")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    _git(root, "config", "user.email", "t@example.org")
    _git(root, "config", "user.name", "t")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "v1")
    return root


def _guard(repo: Path) -> IntegrityGuard:
    return IntegrityGuard(repo, repo / "data" / "integrity" / "manifest.json")


def test_manifest_records_git_head(repo):
    m = build_manifest(repo)
    assert m["git_head"] == _git(repo, "rev-parse", "HEAD")


def test_first_run_creates_baseline_no_drift(repo):
    existed, drift = _guard(repo).ensure_baseline()
    assert existed is False and drift == []


def test_pull_reanchor_silently_adopts_the_new_tree(repo):
    g = _guard(repo)
    assert g.ensure_baseline() == (False, [])
    # owner pulls a new commit: 1 file changes WITH a HEAD move
    (repo / "agentics" / "nodes.py").write_text("print('v2')\n")
    (repo / "agentics" / "config.py").write_text("NEW = 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "v2")
    existed, drift = g.ensure_baseline()
    assert existed is True and drift == []          # no false alarm
    assert g.last_reanchor is not None
    assert g.last_reanchor["changed"] == 2
    assert g.last_reanchor["to"] == _git(repo, "rev-parse", "HEAD")
    # a second run stays quiet: the baseline now matches the tree
    g2 = _guard(repo)
    assert g2.ensure_baseline() == (True, [])
    assert g2.last_reanchor is None


def test_drift_without_head_move_still_alerts(repo, monkeypatch):
    g = _guard(repo)
    assert g.ensure_baseline() == (False, [])
    # a file changes but HEAD does NOT move: no pull/merge explains it
    (repo / "agentics" / "nodes.py").write_text("print('rogue edit')\n")
    head = _git(repo, "rev-parse", "HEAD")
    existed, drift = g.ensure_baseline()
    assert existed is True and len(drift) == 1
    assert drift[0]["kind"] == "changed"
    assert g.last_reanchor is None                  # NOT re-anchored
    assert _git(repo, "rev-parse", "HEAD") == head
    # restore the file and the guard is quiet again
    (repo / "agentics" / "nodes.py").write_text("print('v1')\n")
    assert g.ensure_baseline() == (True, [])


def test_no_git_repo_keeps_the_old_alert_behaviour(tmp_path):
    """Plain dirs (tests, exotic deployments) have no HEAD: drift must
    surface instead of being swallowed by a vacuous re-anchor."""
    root = tmp_path / "plain"
    (root / "agentics").mkdir(parents=True)
    (root / "agentics" / "nodes.py").write_text("print('v1')\n")
    g = IntegrityGuard(root, tmp_path / "manifest.json")
    assert g.ensure_baseline() == (False, [])
    (root / "agentics" / "nodes.py").write_text("print('v2')\n")
    existed, drift = g.ensure_baseline()
    assert existed is True and len(drift) == 1
    assert g.last_reanchor is None
