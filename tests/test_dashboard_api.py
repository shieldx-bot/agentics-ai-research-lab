"""Dashboard API — the owner's window (anti-blackbox) must expose everything."""

from pathlib import Path

import pytest

from agentics.config import Settings
from agentics.dashboard.app import _state, create_app
from agentics.memory.shared_memory import SharedMemory

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")


@pytest.fixture()
def mem(tmp_path: Path):
    m = SharedMemory(tmp_path / "m.db")
    yield m
    m.close()


@pytest.fixture()
def settings(tmp_path):
    s = Settings(data_dir=tmp_path / "data")
    s.ensure_dirs()
    return s


def test_state_contains_the_full_owner_picture(mem, settings):
    tid = mem.create_task("Survey SOTA", category="research")
    mem.claim_issue(tid, "a02", 3600, "agent/a02/task-1-x")
    mem.set_task_status(tid, "claimed", claimed_by="a02", branch="agent/a02/task-1-x")
    mem.report("a02", "researcher", node="research", cycle=3,
               task_id=tid, task_title="Survey SOTA",
               intent={"goal": "Find NFV failure-prediction papers",
                       "why": "task #1", "steps": ["search", "dedup"]})
    mem.journal("a02", "claim", {"task": tid})
    mr_id = mem.open_review(tid, "[a04] work", "b", author="a04",
                            branch="agent/a04/task-1-x", files=["experiments/a.py"])
    mem.mq_enqueue(mr_id, author="a04")

    s = _state(mem, settings)
    assert s["mode"].startswith("local-only")
    agent = next(a for a in s["agents"] if a["agent"] == "a02")
    assert agent["node"] == "research"
    assert agent["intent"]["goal"] == "Find NFV failure-prediction papers"
    assert agent["task_id"] == tid
    assert s["tasks"]["counts"]["in_review"] == 1
    assert any(c["task"] == tid for c in s["claims"])
    assert any(m["pr_number"] == mr_id for m in s["merge_queue"])
    assert any(e["event"] == "claim" for e in s["activity"])
    assert any(r["id"] == mr_id for r in s["reviews"])


def test_dashboard_http_endpoints(mem, settings):
    from fastapi.testclient import TestClient

    app = create_app(mem, settings)
    client = TestClient(app)

    resp = client.get("/")
    assert resp.status_code == 200
    assert "Agentics Lab" in resp.text
    assert "/api/state" in resp.text or "fetch" in resp.text  # polling wired

    mem.create_task("hello task")
    mem.report("a01", "supervisor", node="scan",
               intent={"goal": "scan backlog", "steps": []})

    resp = client.get("/api/state")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["tasks"]["recent"][0]["title"] == "hello task"
    assert payload["agents"][0]["intent"]["goal"] == "scan backlog"


def test_report_keeps_previous_fields_when_not_given(mem):
    mem.report("a03", "coder", node="claim", task_id=5, task_title="T",
               intent={"goal": "g"})
    mem.report("a03", "coder", node="plan", cycle=2)   # no task/intent → kept
    a = mem.active_agents()[0]
    assert a["node"] == "plan" and a["cycle"] == 2
    assert a["task_id"] == 5 and a["task_title"] == "T"
    assert a["intent"] == {"goal": "g"}


def test_stop_endpoint_writes_stop_file_gracefully(mem, settings, monkeypatch):
    """POST /stop = same STOP-file mechanism as `agentics stop` (graceful)."""
    from fastapi.testclient import TestClient

    from agentics import stop as stop_mod

    stop_path = settings.data_dir / "STOP"
    monkeypatch.setattr(stop_mod, "stop_file_path", lambda: stop_path)
    stop_mod._STOP_EVENT.clear()

    client = TestClient(create_app(mem, settings))
    resp = client.post("/stop")
    assert resp.status_code == 200
    body = resp.json()
    assert body["stopped"] is True
    assert stop_path.exists()
    assert stop_mod.should_stop()            # agents exit gracefully between cycles
    stop_path.unlink()
    stop_mod._STOP_EVENT.clear()


def test_stop_endpoint_can_be_disabled(mem, settings, monkeypatch):
    """DASHBOARD_ALLOW_STOP=0 -> 403, nothing written (non-localhost deployments)."""
    from fastapi.testclient import TestClient

    from agentics import stop as stop_mod

    stop_path = settings.data_dir / "STOP"
    monkeypatch.setattr(stop_mod, "stop_file_path", lambda: stop_path)
    monkeypatch.setenv("DASHBOARD_ALLOW_STOP", "0")
    stop_mod._STOP_EVENT.clear()

    client = TestClient(create_app(mem, settings))
    resp = client.post("/stop")
    assert resp.status_code == 403
    assert not stop_path.exists()
    assert not stop_mod.should_stop()


def test_stop_button_present_in_dashboard_html(mem, settings):
    from fastapi.testclient import TestClient

    html = TestClient(create_app(mem, settings)).get("/").text
    assert "stopbtn" in html and "/stop" in html      # button + handler wired
