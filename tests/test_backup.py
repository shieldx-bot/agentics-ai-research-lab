"""Backup pipeline (docs/PRODUCTION.md §5): the lab's state has no second
copy — shared_memory.db (incl. the P21 audit trail) and the bare repo that
IS "origin" in local-only mode.

The script is exercised as a subprocess exactly like an owner would run it
(cron / systemd timer), with --data-dir pointed at a tmp lab.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "backup_lab.py"


@pytest.fixture()
def lab(tmp_path):
    """A small but REAL lab state: shared memory with rows + a bare repo
    holding at least one pushed commit (an empty bare repo has no refs and
    cannot produce a bundle — the script reports that honestly)."""
    from agentics.memory.shared_memory import SharedMemory

    data = tmp_path / "data"
    data.mkdir()
    mem = SharedMemory(data / "shared_memory.db")
    mem.create_task("backup probe")
    mem.audit("owner", "merge", {"mr": 1, "author": "a03"})
    mem.record_evidence("a03", 1, "python x.py", 0, "confirmed", hypothesis="h")
    mem.close()
    bare = data / "lab.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(bare)],
                   check=True, capture_output=True)
    src = tmp_path / "src"
    src.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(src)],
                   check=True, capture_output=True)
    (src / "seed.md").write_text("seed\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(src), "add", "."], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(src), "-c", "user.email=t@t",
                    "-c", "user.name=t", "commit", "-q", "-m", "seed"],
                   check=True, capture_output=True)
    subprocess.run(["git", "-C", str(src), "push", "-q", str(bare), "main"],
                   check=True, capture_output=True)
    return data


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        check=True, capture_output=True, text=True, cwd=str(SCRIPT.parent))


def test_backup_snapshots_db_and_repo_with_manifest(lab, tmp_path):
    dest = tmp_path / "backups"
    proc = _run(["--data-dir", str(lab), "--dest", str(dest), "--keep", "7"])
    out = next(dest.glob("backup-*"))
    assert (out / "shared_memory.db").stat().st_size > 0
    assert (out / "lab.git.bundle").stat().st_size > 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["files"]["shared_memory.db"]["ok"] is True
    assert manifest["files"]["lab.git.bundle"]["ok"] is True
    # the snapshot itself is verifiable: audit trail made it into the copy
    assert manifest["audit_log_rows"] == 1
    assert manifest["evidence_rows"] == 1
    assert manifest["tasks_rows"] == 1
    assert "backup" in proc.stdout


def test_restored_db_opens_and_reads(lab, tmp_path):
    """The backup is only real if the COPY opens and serves the data."""
    import sqlite3

    dest = tmp_path / "backups"
    _run(["--data-dir", str(lab), "--dest", str(dest), "--keep", "7"])
    out = next(dest.glob("backup-*"))
    conn = sqlite3.connect(out / "shared_memory.db")
    try:
        n = conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
        assert n == 1
        row = conn.execute(
            "SELECT detail FROM audit_log LIMIT 1").fetchone()
        assert json.loads(row[0])["author"] == "a03"
    finally:
        conn.close()


def test_bundle_is_clonable(lab, tmp_path):
    """A git bundle that can't be cloned is decoration — verify it restores."""
    dest = tmp_path / "backups"
    _run(["--data-dir", str(lab), "--dest", str(dest), "--keep", "7"])
    bundle = next(dest.glob("backup-*")) / "lab.git.bundle"
    restored = tmp_path / "restored"
    subprocess.run(["git", "clone", "-q", str(bundle), str(restored)],
                   check=True, capture_output=True)
    r = subprocess.run(["git", "-C", str(restored), "rev-parse", "--is-bare-repository"],
                       capture_output=True, text=True)
    assert r.returncode == 0


def test_retention_keeps_only_newest(tmp_path):
    """--keep 2 must leave exactly 2 newest backups, older ones pruned."""
    from agentics.config import Settings
    from agentics.memory.shared_memory import SharedMemory

    data = tmp_path / "data"
    data.mkdir()
    mem = SharedMemory(data / "shared_memory.db")
    mem.create_task("t")
    mem.close()
    dest = tmp_path / "backups"
    for _ in range(3):
        _run(["--data-dir", str(data), "--dest", str(dest), "--keep", "2"])
    dirs = sorted(dest.glob("backup-*"))
    assert len(dirs) == 2


def test_backup_without_bare_repo_still_snapshots_db(tmp_path):
    """Fresh lab (no bare repo yet): db snapshot succeeds, bundle is honestly
    marked failed in the manifest — never a silent gap."""
    from agentics.memory.shared_memory import SharedMemory

    data = tmp_path / "data"
    data.mkdir()
    mem = SharedMemory(data / "shared_memory.db")
    mem.create_task("fresh")
    mem.close()
    dest = tmp_path / "backups"
    _run(["--data-dir", str(data), "--dest", str(dest), "--keep", "7"])
    manifest = json.loads(
        (next(dest.glob("backup-*")) / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["files"]["shared_memory.db"]["ok"] is True
    assert manifest["files"]["lab.git.bundle"]["ok"] is False
