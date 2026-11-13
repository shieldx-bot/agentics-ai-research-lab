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


def test_state_includes_shared_evidence(mem, settings):
    """P16 on the owner's dashboard: real runs, newest first, with histogram."""
    mem.record_evidence("a02", 3, "python exp.py", 1, "refuted",
                        hypothesis="linear beats quadratic")
    mem.record_evidence("a04", 3, "python check.py", 0, "confirmed")
    s = _state(mem, settings)
    ev = s["evidence"]
    assert ev["counts"] == {"refuted": 1, "confirmed": 1}
    assert [r["id"] for r in ev["recent"]] == [2, 1]          # newest first
    assert ev["recent"][0]["verdict"] == "confirmed"


def test_evidence_card_present_in_dashboard_html(mem, settings):
    from fastapi.testclient import TestClient

    html = TestClient(create_app(mem, settings)).get("/").text
    assert "Kiểm chứng thực nghiệm" in html
    assert 'id="evidence"' in html and 'id="evcounts"' in html


def test_alerts_card_covers_repairs_and_p16_suspicions(mem, settings):
    """The alerts card previously only showed guard events — a Supervisor
    task_repair, a failed merge or a suspicious P16 hypothesis were invisible
    to the owner. All system alerts must surface in one place."""
    for event in ("task_repair", "merge_failed", "p16_inconclusive",
                  "p16_hypothesis_invalid", "integrity_drift"):
        mem.journal("sys", event, {"note": f"{event} happened"})
    s = _state(mem, settings)
    events = [a["event"] for a in s["alerts"]]
    for event in ("task_repair", "merge_failed", "p16_inconclusive",
                  "p16_hypothesis_invalid", "integrity_drift"):
        assert event in events
    assert "lock_conflict" not in events            # none recorded here


def test_state_shows_peer_panel_progress(mem, settings):
    """P19: the owner sees the panel state, the votes and the defense clock."""
    from agentics.peer_review import PeerPanel

    mr_id = mem.open_review(task_id=None, title="[a04] work", body="b",
                            author="a04", branch="agent/a04/task-1-x", files=[])
    for agent in ("a02", "a03"):
        mem.heartbeat(agent, "reviewer", 1)
    panel = PeerPanel(mem, panel_size=2, defense_ttl_minutes=30)
    panel.cast_vote(mr_id, "a02", "accept")            # 1/2 -> waiting
    panel.cast_vote(mr_id, "a03", "reject", "no evidence on the branch")
    s = _state(mem, settings)
    row = next(r for r in s["reviews"] if r["id"] == mr_id)
    assert row["panel"]["state"] == "defense"
    assert row["panel"]["round"] == 1
    assert row["panel"]["votes"] == {"accept": 1, "reject": 1}
    assert row["panel"]["defense_left_s"] >= 0


def test_alerts_card_covers_panel_rejections(mem, settings):
    mem.journal("peer-panel", "panel_rejected", {"mr": 3, "why": "defense incomplete"})
    assert "panel_rejected" in [a["event"] for a in _state(mem, settings)["alerts"]]
