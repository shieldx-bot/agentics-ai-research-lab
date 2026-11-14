"""Owner chat (dashboard) — hỏi đáp qua model CỤC BỘ, đọc shared memory.

Contract being tested:
* build_context: fresh digest, agent-focus narrowing, never invents data.
* ask_lab: local-model happy path (history carried forward), graceful
  digest fallback when Ollama is absent/raising, and history ring bounds.
* endpoint: POST /api/chat answers, empty question is 400, and the whole
  feature stays read-only over shared memory.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentics.dashboard.chat import ChatSession, build_context, detect_agent
from agentics.llm import local_router
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
    s = SimpleNamespace(local_llm_model="", local_llm_url="",
                        local_llm_timeout=120.0, llm_max_calls_per_hour=90)
    return s


def _local_settings():
    return SimpleNamespace(
        local_llm_model="qwen3:1.7b", local_llm_url="http://127.0.0.1:11434",
        local_llm_timeout=120.0, llm_max_calls_per_hour=90)


def _populate(mem: SharedMemory):
    tid = mem.create_task("Survey SOTA", category="research")
    mem.report("a03", "coder", node="plan", cycle=13,
               task_id=tid, task_title="Survey SOTA",
               intent={"goal": "plan the survey", "steps": ["search"]})
    mem.journal("a03", "local_fallback", {"purpose": "plan"})
    mem.journal("a04", "budget_blocked",
                {"purpose": "research", "why": "P20 pool budget exhausted"})
    mem.journal("a01", "integrity_alert", {"drift": ["agentics/cli.py"]})
    mem.record_llm_call("a03", "research", True, 1.2)


# ------------------------------------------------------------ agent focus
def test_detect_agent_variants():
    assert detect_agent("a03 đang làm gì?") == "a03"
    assert detect_agent("trạng thái A3 thế nào") == "a03"
    assert detect_agent("a12 làm gì") == "a12"
    assert detect_agent("ngân sách còn bao nhiêu?") == ""
    assert detect_agent("") == ""
    # không nhả mồi trong từ khác ('task3', 'x1a3')
    assert detect_agent("task3 world") == ""


def test_build_context_sections_and_agent_focus(mem, settings):
    _populate(mem)
    digest, focus = build_context(mem, settings, "tổng quan")
    assert focus == ""
    for section in ("AGENTS", "BUDGET", "TASKS", "EVIDENCE", "HYPOTHESES",
                    "MEMORY"):
        assert section in digest
    assert "cap 90" in digest
    assert "local_fallback" in digest            # general journal slice
    assert "integrity_alert" in digest           # alert surfaced even unfocused

    digest_f, focus_f = build_context(mem, settings, "a03 đang làm gì?")
    assert focus_f == "a03"
    assert "JOURNAL của a03" in digest_f
    assert "local_fallback" in digest_f          # a03's own event
    assert "budget_blocked" not in digest_f      # a04's event filtered out


def test_build_context_empty_memory_does_not_crash(mem, settings):
    digest, focus = build_context(mem, settings, "hello?")
    assert focus == ""
    assert "BUDGET" in digest and "MEMORY" in digest


# ------------------------------------------------------------ ask_lab
def test_ask_lab_local_model_answers_and_carries_history(mem, settings,
                                                         monkeypatch):
    _populate(mem)
    captured = {}

    async def fake_ask(prompt, settings=None, purpose="local", system=""):
        captured["prompt"] = prompt
        captured["purpose"] = purpose
        return SimpleNamespace(answer="a03 đang plan task #1", ok=True)

    monkeypatch.setattr(local_router, "ask", fake_ask)
    session = ChatSession()
    out = asyncio.run(session.ask(mem, _local_settings(), "a03 đang làm gì?"))
    assert out["source"] == "local" and out["focused"] == "a03"
    assert out["answer"] == "a03 đang plan task #1"
    # prompt is grounded: digest + question, and NEVER touches cookies
    assert "DỮ LIỆU LAB" in captured["prompt"]
    assert "a03 đang làm gì?" in captured["prompt"]
    assert "JOURNAL của a03" in captured["prompt"]
    assert captured["purpose"] == "owner_chat"

    # follow-up carries the previous turn as context
    out2 = asyncio.run(session.ask(mem, _local_settings(), "còn a04 thì sao?"))
    assert "HỎI-ĐÁP GẦN ĐÂY" in captured["prompt"]
    assert "a03 đang plan task #1" in captured["prompt"]  # prior answer
    assert "còn a04 thì sao?" in captured["prompt"]
    assert out2["focused"] == "a04"
    assert len(session.turns) == 2


def test_ask_lab_history_ring_is_bounded(mem, settings, monkeypatch):
    async def fake_ask(prompt, settings=None, purpose="local", system=""):
        return SimpleNamespace(answer="ok", ok=True)

    monkeypatch.setattr(local_router, "ask", fake_ask)
    session = ChatSession(maxlen=4)
    for i in range(6):
        asyncio.run(session.ask(mem, _local_settings(), f"câu hỏi {i}"))
    assert len(session.turns) == 4


def test_ask_lab_falls_back_to_digest_when_ollama_absent(mem, settings):
    _populate(mem)
    session = ChatSession()
    out = asyncio.run(session.ask(mem, settings, "tổng quan?"))   # no local_llm_model
    assert out["source"] == "digest"
    assert "không khả dụng" in out["answer"]
    assert "BUDGET" in out["answer"]                 # truthful raw data
    assert out["elapsed_s"] >= 0.0


def test_ask_lab_falls_back_when_local_raises(mem, settings, monkeypatch):
    _populate(mem)

    async def boom(prompt, settings=None, purpose="local", system=""):
        raise ConnectionError("ollama died")

    monkeypatch.setattr(local_router, "ask", boom)
    session = ChatSession()
    out = asyncio.run(session.ask(mem, _local_settings(), "a03 làm gì rồi?"))
    assert out["source"] == "digest"
    assert "không khả dụng" in out["answer"]
    assert len(session.turns) == 1                   # fallback still recorded


def test_ask_lab_ignores_not_ok_local_response(mem, settings, monkeypatch):
    async def dead(prompt, settings=None, purpose="local", system=""):
        return None

    monkeypatch.setattr(local_router, "ask", dead)
    session = ChatSession()
    out = asyncio.run(session.ask(mem, _local_settings(), "hello?"))
    assert out["source"] == "digest"


# ------------------------------------------------------------ HTTP endpoint
def test_chat_endpoint(mem, settings, monkeypatch):
    from fastapi.testclient import TestClient

    from agentics.dashboard.app import create_app

    _populate(mem)

    async def fake_ask(prompt, settings=None, purpose="local", system=""):
        return SimpleNamespace(answer="42", ok=True)

    monkeypatch.setattr(local_router, "ask", fake_ask)
    client = TestClient(create_app(mem, _local_settings()))

    resp = client.post("/api/chat", json={"question": "a03 đang làm gì?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "42" and body["source"] == "local"
    assert body["focused"] == "a03"

    # empty question is a 400, not a crash
    assert client.post("/api/chat", json={"question": "  "}).status_code == 400
    assert client.post("/api/chat", json={}).status_code == 400


def test_chat_is_read_only_over_shared_memory(mem, settings, monkeypatch):
    """The owner asking questions must not perturb the swarm: no journal
    rows, no tasks, no LLM-usage records."""
    from fastapi.testclient import TestClient

    from agentics.dashboard.app import create_app

    _populate(mem)
    before_tasks = mem.task_counts()
    before_journal = len(mem.recent_journal(limit=500))
    before_usage = mem.llm_calls_in_window(3600.0)

    async def fake_ask(prompt, settings=None, purpose="local", system=""):
        return SimpleNamespace(answer="x", ok=True)

    monkeypatch.setattr(local_router, "ask", fake_ask)
    client = TestClient(create_app(mem, _local_settings()))
    for q in ("tổng quan?", "a03 đang làm gì?", "ngân sách thế nào?"):
        assert client.post("/api/chat", json={"question": q}).status_code == 200

    assert mem.task_counts() == before_tasks
    assert len(mem.recent_journal(limit=500)) == before_journal
    assert mem.llm_calls_in_window(3600.0) == before_usage


def test_dashboard_html_wires_the_chat_ui(mem, settings):
    from fastapi.testclient import TestClient

    from agentics.dashboard.app import create_app

    html = TestClient(create_app(mem, settings)).get("/").text
    for needle in ("/api/chat", "chatpanel", "chatin", "sendChat",
                   "Chat AI"):
        assert needle in html
