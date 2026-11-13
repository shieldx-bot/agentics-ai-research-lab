"""Smoke-check the P16bis owner surfaces: dashboard state + CLI + demo data.

Creates a temp lab, records a supported and a refuted hypothesis through the
REAL flow primitives, then asserts:
  * /api/state carries the hypotheses panel (counts + rows with criterion);
  * the dashboard HTML contains the panel markup;
  * 'agentics hypotheses' (CLI) prints the table without error.
Run:  python scripts/smoke_verification_surfaces.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from agentics.dashboard.app import create_app
from agentics.memory.shared_memory import SharedMemory


def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="agentics-smoke-"))
    mem = SharedMemory(root / "mem.db")

    # two settled hypotheses + their evidence, via the real primitives
    sup = mem.add_hypothesis(
        "a02", "ensemble voting reaches >=0.9 precision",
        {"metric": "precision", "op": ">=", "target": 0.9}, task_id=None)
    ref = mem.add_hypothesis(
        "a02", "single-tree baseline reaches >=0.9 precision",
        {"metric": "precision", "op": ">=", "target": 0.9})
    ev1 = mem.record_evidence("a02", None, "experiments/verify_h1.py", 0,
                              "confirmed", hypothesis="ensemble voting",
                              method="protocol", output="hit_rate=0.9 holds")
    mem.attach_protocol_outcome(ev1, {"precision": 0.9}, [{"exit_code": 0}], sup)
    mem.update_hypothesis(sup, "supported", evidence_id=ev1, note="measured 0.9")
    mem.update_hypothesis(ref, "refuted", note="measured precision=0.4")

    # The REAL Settings class — the dashboard state now carries the P18 LLM
    # budget dial (llm_max_calls_per_hour) and the P20 audit counters, so a
    # hand-rolled fake settings namespace 500s the endpoint and the smoke
    # reads {"error": ...} instead of the state payload.
    from agentics.config import Settings

    app = create_app(mem, Settings(data_dir=root / "data", research_topic="NFV"))
    client = _client(app)
    state = client.get("/api/state").json()
    hyp = state["hypotheses"]
    assert hyp["counts"] == {"refuted": 1, "supported": 1}, hyp["counts"]
    rows = {h["id"]: h for h in hyp["recent"]}
    assert rows[sup]["criterion"]["metric"] == "precision"
    assert rows[sup]["status"] == "supported" and rows[ref]["status"] == "refuted"
    html = client.get("/").text
    assert "Giả thuyết đã đăng ký" in html and 'id="hypotheses"' in html
    assert 'id="evcounts"' in html                      # P16 evidence card intact

    # CLI smoke: 'agentics hypotheses' on this lab's DB path
    env = {**dict(__import__("os").environ), "PYTHONPATH": str(REPO)}
    proc = subprocess.run(
        [sys.executable, "-m", "agentics.cli", "hypotheses"],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=60, check=False)
    # the CLI reads the repo's default data dir; a fresh clone has an empty DB,
    # so the command must succeed regardless and print a header or table
    assert proc.returncode == 0, proc.stderr[-400:]
    assert "Giả thuyết" in proc.stdout or "Pre-registered" in proc.stdout

    mem.close()
    import shutil
    shutil.rmtree(root, ignore_errors=True)
    print("SMOKE OK — dashboard panel, criterion rows and CLI all render")


def _client(app):
    try:
        from fastapi.testclient import TestClient
        return TestClient(app)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"testclient unavailable: {exc}")


if __name__ == "__main__":
    main()
