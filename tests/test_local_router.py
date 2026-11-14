"""Local small-model router (#3): a FREE offline LLM edge for cheap purposes.

Contract under test:
* `ask` never raises; an absent runtime returns None;
* `allowed_for` refuses SCIENCE purposes even when someone lists them in
  LOCAL_LLM_PURPOSES — a weaker verdict is worse than a refused ask;
* purposes outside the configured list never route to the local model;
* the P20 ledger does not count local asks (they burn no cookie quota).
"""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402

from agentics.llm import local_router  # noqa: E402


def _settings(**kw):
    base = dict(local_llm_enabled=True, local_llm_url="http://127.0.0.1:11434",
                local_llm_model="qwen3:1.7b", local_llm_timeout=1.0,
                local_llm_purposes="kg_extract,classify")
    base.update(kw)
    return SimpleNamespace(**base)


# ------------------------------------------------------------ purpose gate
def test_science_purposes_never_route_locally():
    s = _settings(local_llm_purposes="verify,research,kg_extract")
    assert local_router.allowed_for(s, "verify") is False
    assert local_router.allowed_for(s, "research") is False
    assert local_router.allowed_for(s, "ideation") is False
    assert local_router.allowed_for(s, "kg_extract") is True


def test_purposes_outside_the_list_refuse():
    s = _settings()
    assert local_router.allowed_for(s, "kg_extract") is True
    assert local_router.allowed_for(s, "tournament") is False
    assert local_router.allowed_for(s, "") is False


def test_disabled_router_refuses_everything():
    s = _settings(local_llm_enabled=False)
    assert local_router.allowed_for(s, "kg_extract") is False


# ------------------------------------------------------------ ask contract
def test_ask_without_url_or_model_returns_none():
    assert asyncio.run(local_router.ask("hi", _settings(local_llm_model=""))) \
        is None
    assert asyncio.run(local_router.ask("", _settings())) is None
    assert asyncio.run(local_router.ask("hi", _settings(local_llm_url=""))) \
        is None


def test_ask_against_a_dead_runtime_returns_none():
    # nothing listens on this port; must return None, never raise
    s = _settings(local_llm_url="http://127.0.0.1:1", local_llm_timeout=1.0)
    assert asyncio.run(local_router.ask("hello", s)) is None
    assert asyncio.run(local_router.available(s)) is False


def test_ask_parses_ollama_chat_payload(monkeypatch):
    captured = {}

    async def fake_post(url, payload, settings=None):
        captured["url"] = url
        captured["payload"] = payload
        return {"message": {"content": "triple extraction done"}}

    monkeypatch.setattr(local_router, "_post", fake_post)
    resp = asyncio.run(local_router.ask("extract", _settings(),
                                        purpose="kg_extract"))
    assert resp is not None and resp.ok and resp.answer == "triple extraction done"
    assert resp.citations == []
    assert captured["url"].endswith("/api/chat")
    assert captured["payload"]["model"] == "qwen3:1.7b"
    assert captured["payload"]["stream"] is False


def test_ask_accepts_llamacpp_style_payload(monkeypatch):
    async def fake_post(url, payload, settings=None):
        return {"response": "plain completion"}

    monkeypatch.setattr(local_router, "_post", fake_post)
    resp = asyncio.run(local_router.ask("hi", _settings()))
    assert resp is not None and resp.answer == "plain completion"


def test_ask_empty_content_is_none(monkeypatch):
    async def fake_post(url, payload, settings=None):
        return {"message": {"content": "   "}}

    monkeypatch.setattr(local_router, "_post", fake_post)
    assert asyncio.run(local_router.ask("hi", _settings())) is None


def test_post_never_raises_on_garbage(monkeypatch):
    async def boom(*a, **k):
        raise RuntimeError("exploded")

    monkeypatch.setattr(local_router.aiohttp, "ClientSession", boom)
    assert asyncio.run(local_router._post("http://x", {})) is None
