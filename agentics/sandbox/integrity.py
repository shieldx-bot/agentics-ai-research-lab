"""Integrity manifest — DETECT any tampering with the running lab (layer 4).

The orchestrator hashes every runtime-critical file at startup and stores the
manifest under data/integrity/. The Supervisor re-verifies periodically and
the owner can run `agentics verify` at any time. Any drift between the
manifest and the live tree — whether caused by a botched merge, an agent that
escaped the path guard, or a fat-fingered owner edit — is surfaced as an
alert instead of being silently absorbed.

This is detection, not prevention: prevention is the P10 protected-paths
guard (write / diff / merge). Together the layers answer hypothesis 1 of
docs/LOCAL_MODE.md.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

log = logging.getLogger("sandbox.integrity")

# Directories that define how the running lab behaves (relative to PROJECT_ROOT).
RUNTIME_DIRS: tuple[str, ...] = ("agentics", "configs", "scripts")
RUNTIME_FILES: tuple[str, ...] = ("pyproject.toml", "requirements.txt")

_SKIP_PARTS = {"__pycache__", ".pytest_cache"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(root: Path) -> dict:
    """Hash every runtime-critical file under ``root`` (a project directory)."""
    files: dict[str, str] = {}
    for name in RUNTIME_DIRS:
        base = root / name
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or _SKIP_PARTS.intersection(path.parts):
                continue
            files[path.relative_to(root).as_posix()] = _sha256(path)
    for name in RUNTIME_FILES:
        candidate = root / name
        if candidate.is_file():
            files[name] = _sha256(candidate)
    return {
        "version": 1,
        "built_ts": time.time(),
        "file_count": len(files),
        "files": files,
    }


def load_manifest(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def save_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")


def verify(root: Path, manifest: dict) -> list[dict]:
    """Compare the live tree against ``manifest``; return a drift list.

    Each drift item: {"path", "kind": changed|missing|added}.
    """
    live = build_manifest(root)["files"]
    expected = manifest.get("files", {})
    drift: list[dict] = []
    for path, digest in expected.items():
        if path not in live:
            drift.append({"path": path, "kind": "missing"})
        elif live[path] != digest:
            drift.append({"path": path, "kind": "changed"})
    for path in live:
        if path not in expected:
            drift.append({"path": path, "kind": "added"})
    return drift


class IntegrityGuard:
    """Startup baseline + periodic verification for the runtime tree."""

    def __init__(self, root: Path, manifest_path: Path) -> None:
        self.root = root
        self.manifest_path = manifest_path

    def ensure_baseline(self) -> tuple[bool, list[dict]]:
        """Return (baseline_exists, drift). Creates the baseline on first run."""
        manifest = load_manifest(self.manifest_path)
        if manifest is None:
            save_manifest(self.manifest_path, build_manifest(self.root))
            return False, []
        return True, verify(self.root, manifest)

    def load(self) -> dict | None:
        """Read the stored baseline (None when there is none yet)."""
        return load_manifest(self.manifest_path)

    def refresh_baseline(self) -> int:
        manifest = build_manifest(self.root)
        save_manifest(self.manifest_path, manifest)
        return manifest["file_count"]
