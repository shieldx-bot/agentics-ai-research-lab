"""Integrity manifest — DETECT layer (layer 4) for tampering with the running lab."""

import json

from agentics.sandbox.integrity import (
    IntegrityGuard, build_manifest, load_manifest, save_manifest, verify,
)


def _make_runtime(tmp_path):
    (tmp_path / "agentics" / "sub").mkdir(parents=True)
    (tmp_path / "agentics" / "__pycache__").mkdir()
    (tmp_path / "agentics" / "core.py").write_text("VERSION = 1\n")
    (tmp_path / "agentics" / "sub" / "util.py").write_text("def f():\n    return 1\n")
    (tmp_path / "agentics" / "__pycache__" / "core.pyc").write_bytes(b"junk")  # skipped
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "lab.yaml").write_text("max_agents: 3\n")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("docs are NOT runtime — excluded")


def test_manifest_hashes_runtime_only(tmp_path):
    _make_runtime(tmp_path)
    manifest = build_manifest(tmp_path)
    files = manifest["files"]
    assert "agentics/core.py" in files and "agentics/sub/util.py" in files
    assert "configs/lab.yaml" in files and "pyproject.toml" in files
    assert not any(p.startswith("docs/") for p in files)          # product area excluded
    assert not any("__pycache__" in p for p in files)             # caches excluded
    assert manifest["file_count"] == len(files)


def test_verify_detects_changed_missing_added(tmp_path):
    _make_runtime(tmp_path)
    manifest = build_manifest(tmp_path)

    (tmp_path / "agentics" / "core.py").write_text("VERSION = 999\n")   # changed
    (tmp_path / "agentics" / "sub" / "util.py").unlink()                 # missing
    (tmp_path / "agentics" / "new.py").write_text("x = 1\n")             # added

    drift = {d["path"]: d["kind"] for d in verify(tmp_path, manifest)}
    assert drift["agentics/core.py"] == "changed"
    assert drift["agentics/sub/util.py"] == "missing"
    assert drift["agentics/new.py"] == "added"


def test_verify_is_clean_when_nothing_changed(tmp_path):
    _make_runtime(tmp_path)
    manifest = build_manifest(tmp_path)
    assert verify(tmp_path, manifest) == []


def test_guard_roundtrip_and_persistence(tmp_path):
    _make_runtime(tmp_path)
    guard = IntegrityGuard(tmp_path, tmp_path / "integrity" / "baseline.json")

    existed, drift = guard.ensure_baseline()
    assert existed is False and drift == []                 # first run creates it
    saved = load_manifest(tmp_path / "integrity" / "baseline.json")
    assert saved and saved["file_count"] > 0

    existed, drift = guard.ensure_baseline()
    assert existed is True and drift == []                  # second run verifies clean

    (tmp_path / "configs" / "lab.yaml").write_text("max_agents: 99\n")
    _, drift = guard.ensure_baseline()
    assert drift and drift[0]["path"] == "configs/lab.yaml"

    count = guard.refresh_baseline()
    _, drift = guard.ensure_baseline()
    assert drift == [] and count == saved["file_count"]


def test_manifest_survives_a_reload(tmp_path):
    _make_runtime(tmp_path)
    path = tmp_path / "integrity" / "baseline.json"
    save_manifest(path, build_manifest(tmp_path))
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == 1 and load_manifest(path)["files"] == data["files"]


def test_orchestrator_startup_writes_the_integrity_baseline(tmp_path):
    """Integration: `Orchestrator(...)` on a fresh data dir must come up and
    hash the runtime tree (DETECT layer) without any error."""
    from agentics.config import Settings
    from agentics.orchestrator import Orchestrator

    settings = Settings(data_dir=tmp_path / "data",
                        accounts_dir=tmp_path / "accounts")
    orch = Orchestrator(settings)
    assert (tmp_path / "data" / "integrity" / "baseline.json").exists()
    assert orch.memory is not None
    # the orchestrator can journal an integrity alert (consolidated API)
    orch.memory.journal("orchestrator", "alert", {"what": "test"})
    assert any(e["event"] == "alert"
               for e in orch.memory.recent_journal(limit=5))
