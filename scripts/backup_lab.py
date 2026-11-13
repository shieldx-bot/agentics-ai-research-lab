#!/usr/bin/env python3
"""Backup the lab's state for production (docs/PRODUCTION.md §5).

Everything that CANNOT be rebuilt lives in `data/`:
  * shared_memory.db  — tasks, claims, journal, evidence, hypotheses,
                        audit_log (P21: the trail has no other copy), MR
                        reviews, panels… (SQLite WAL, written live by all
                        agent threads)
  * lab.git           — the bare repo that IS "origin" in local-only mode

This script snapshots both SAFELY while the lab runs:
  * SQLite via the online backup API (consistent copy, no write lock);
  * the bare repo via `git bundle create` (single file, restorable with
    `git clone repo.bundle`).

Each run writes  <dest>/backup-<stamp>/{shared_memory.db, lab.git.bundle,
manifest.json}  and prunes older backups to --keep. The manifest records
file sizes + sha256 + the audit/evidence/agent counts, so an owner can
verify a backup is complete WITHOUT opening it.

Usage:
  python scripts/backup_lab.py [--data-dir data] [--dest backups] [--keep 7]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agentics.config import Settings  # noqa: E402


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _backup_sqlite(src: Path, dst: Path) -> None:
    """Online backup: consistent even while agent threads are writing."""
    if not src.exists():
        raise FileNotFoundError(f"no database at {src}")
    dst_conn = sqlite3.connect(dst)
    try:
        src_conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
        try:
            src_conn.backup(dst_conn)
        finally:
            src_conn.close()
    finally:
        dst_conn.close()


def _bundle_repo(repo: Path, bundle: Path) -> None:
    r = subprocess.run(
        ["git", "bundle", "create", str(bundle), "--all"],
        cwd=str(repo), capture_output=True, text=True)
    if r.returncode != 0:
        err = (r.stderr or r.stdout or "unknown git error").strip().splitlines()
        raise RuntimeError(err[-1] if err else "git bundle failed")


def run_backup(data_dir: Path, dest: Path, keep: int = 7) -> dict:
    """Snapshot db + bare repo into <dest>/backup-<stamp>/; prune to keep.
    Returns the manifest dict (also written as manifest.json)."""
    settings = Settings(data_dir=data_dir)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = dest / f"backup-{stamp}"
    suffix = 1
    while out.exists():          # two runs within the same second — never crash
        suffix += 1
        out = dest / f"backup-{stamp}-{suffix}"
    out.mkdir(parents=True)

    manifest: dict = {"ts": time.time(), "stamp": stamp, "backup_dir": out.name,
                      "files": {}, "data_dir": str(data_dir)}
    db_dst = out / "shared_memory.db"
    try:
        _backup_sqlite(settings.db_path, db_dst)
        manifest["files"]["shared_memory.db"] = {
            "bytes": db_dst.stat().st_size, "sha256": _sha256(db_dst),
            "ok": True}
    except Exception as exc:  # noqa: BLE001 — one failed part must not kill the run
        manifest["files"]["shared_memory.db"] = {"ok": False, "error": str(exc)}

    bundle_dst = out / "lab.git.bundle"
    if settings.local_repo_path.exists():
        try:
            _bundle_repo(settings.local_repo_path, bundle_dst)
            manifest["files"]["lab.git.bundle"] = {
                "bytes": bundle_dst.stat().st_size,
                "sha256": _sha256(bundle_dst), "ok": True}
        except Exception as exc:  # noqa: BLE001
            manifest["files"]["lab.git.bundle"] = {
                "ok": False, "error": str(exc)}
    else:
        manifest["files"]["lab.git.bundle"] = {
            "ok": False, "error": "no bare repo yet (lab never ran)"}

    # counts straight from the snapshot — the owner verifies completeness
    # without opening the backup anywhere else
    if db_dst.exists():
        try:
            conn = sqlite3.connect(db_dst)
            try:
                for table in ("audit_log", "evidence", "tasks", "hypotheses"):
                    n = conn.execute(
                        f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    manifest[f"{table}_rows"] = int(n)
                row = conn.execute(
                    "SELECT MAX(ts) FROM agents").fetchone()
                manifest["last_heartbeat"] = row[0] if row else None
            finally:
                conn.close()
        except Exception as exc:  # noqa: BLE001
            manifest["counts_error"] = str(exc)

    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    _prune_old_backups(dest, keep)
    return manifest


def _prune_old_backups(dest: Path, keep: int) -> list[str]:
    """Newest --keep backups survive; older ones are removed."""
    if keep <= 0:
        return []
    backups = sorted(
        (d for d in dest.glob("backup-*") if d.is_dir()),
        key=lambda d: d.name, reverse=True)
    removed = []
    for old in backups[keep:]:
        shutil.rmtree(old, ignore_errors=True)
        removed.append(old.name)
    return removed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    ap.add_argument("--dest", type=Path, default=PROJECT_ROOT / "backups")
    ap.add_argument("--keep", type=int, default=7,
                    help="how many newest backups to keep (default 7)")
    args = ap.parse_args()

    manifest = run_backup(args.data_dir, args.dest, args.keep)
    parts = ", ".join(
        f"{k}: {'ok' if v.get('ok') else v.get('error', 'FAILED')}"
        for k, v in manifest["files"].items())
    print(f"backup {manifest['stamp']} — {parts}")
    if "audit_log_rows" in manifest:
        print(f"  audit_log={manifest['audit_log_rows']} "
              f"evidence={manifest['evidence_rows']} "
              f"tasks={manifest['tasks_rows']} "
              f"hypotheses={manifest['hypotheses_rows']}")
    failed = [k for k, v in manifest["files"].items() if not v.get("ok")]
    return 1 if failed and failed == ["shared_memory.db"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
