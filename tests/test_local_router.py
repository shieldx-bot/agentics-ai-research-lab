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


# --------------------------------- CPU-realistic completion (2026-11-14 08:51)
def test_ask_payload_is_cpu_bounded(monkeypatch):
    """Live 08:51: every /api/chat timed out (TimeoutError str() is empty —
    hence the silent `failed: ` warnings). The payload must bound
    generation and keep the model warm so CPU inference fits the window."""
    captured = {}

    async def fake_post(url, payload, settings=None):
        captured.update(payload)
        return {"message": {"content": "ok"}}

    monkeypatch.setattr(local_router, "_post", fake_post)
    s = _settings(local_llm_num_predict=256, local_llm_timeout=120.0)
    assert asyncio.run(local_router.ask("hi", s)) is not None
    assert captured["keep_alive"] == "5m"
    assert captured["options"]["num_predict"] == 256
    assert captured["options"]["num_ctx"] == 4096      # pinned, not model default
    assert captured["think"] is False          # thinking off by default


def test_ask_think_param_respects_env(monkeypatch):
    async def fake_post(url, payload, settings=None):
        captured.update(payload)
        return {"message": {"content": "ok"}}

    captured = {}
    monkeypatch.setattr(local_router, "_post", fake_post)
    s = _settings(local_llm_think=True)
    assert asyncio.run(local_router.ask("hi", s)) is not None
    assert "think" not in captured             # runtime decides, model thinks


def test_ask_keep_alive_is_configurable(monkeypatch):
    """Live 2026-11-14 10:35: the kernel OOM-killed llama-server (global
    OOM, anon-rss 6 GB) — while the model is loaded, llama.cpp's prompt
    cache grows toward ~8 GiB and the limit is NOT configurable upstream
    (ollama#17351). A RAM-tight host must be able to shorten keep_alive
    so model + cache unload on idle; the request-level value overrides
    any OLLAMA_KEEP_ALIVE server default, so it has to come from here."""
    captured = {}

    async def fake_post(url, payload, settings=None):
        captured.update(payload)
        return {"message": {"content": "ok"}}

    monkeypatch.setattr(local_router, "_post", fake_post)
    s = _settings(local_llm_keep_alive="30m")
    assert asyncio.run(local_router.ask("hi", s)) is not None
    assert captured["keep_alive"] == "30m"

    # empty/None env must not send a None keep_alive to Ollama
    s = _settings(local_llm_keep_alive="")
    assert asyncio.run(local_router.ask("hi", s)) is not None
    assert captured["keep_alive"] == "5m"


def test_ask_num_ctx_is_capped_and_configurable(monkeypatch):
    """Live 2026-11-14 10:35 + ~13:43 (killed TWICE): llama.cpp cache slots
    scale with the context window — the 10:35 journal showed ~238 MiB
    slots x ~30 LRU ≈ 7 GB RSS → global OOM. num_ctx must be pinned in
    EVERY payload (never left to the model default) and configurable."""
    captured = {}

    async def fake_post(url, payload, settings=None):
        captured.update(payload)
        return {"message": {"content": "ok"}}

    monkeypatch.setattr(local_router, "_post", fake_post)
    s = _settings(local_llm_num_ctx=8192)
    assert asyncio.run(local_router.ask("hi", s)) is not None
    assert captured["options"]["num_ctx"] == 8192

    # empty/0 env must degrade to the 4096 design default, not None/0
    s = _settings(local_llm_num_ctx=0)
    assert asyncio.run(local_router.ask("hi", s)) is not None
    assert captured["options"]["num_ctx"] == 4096


def test_ask_strips_think_blocks(monkeypatch):
    async def fake_post(url, payload, settings=None):
        return {"message": {
            "content": "<think>\nlet me reason about triples...\n</think>\n"
                       "A causes B"}}            # thinking left in by the
                                                     # runtime despite think=0
    monkeypatch.setattr(local_router, "_post", fake_post)
    resp = asyncio.run(local_router.ask("hi", _settings()))
    assert resp is not None
    assert "<think>" not in resp.answer
    assert resp.answer == "A causes B"


def test_timeout_default_is_cpu_realistic():
    import agentics.config as config
    cfg = config.Settings()
    assert cfg.local_llm_timeout >= 60.0        # 20s starved the first call
    assert cfg.local_llm_num_predict > 0


def test_defaults_fit_a_ram_tight_host():
    """2026-11-14 OOM postmortem: the shipped defaults must already be the
    safe ones — keep_alive short enough that idle unloads the prompt
    cache, num_ctx pinned to the lab's 4096 design budget."""
    import agentics.config as config
    cfg = config.Settings()
    assert cfg.local_llm_keep_alive == "5m"
    assert cfg.local_llm_num_ctx == 4096


def test_post_never_raises_on_garbage(monkeypatch):
    async def boom(*a, **k):
        raise RuntimeError("exploded")

    monkeypatch.setattr(local_router.aiohttp, "ClientSession", boom)
    assert asyncio.run(local_router._post("http://x", {})) is None


# --------------------------------------------- health check (2026-11-14 405)
def _fake_session_factory(methods_seen: list, status: int = 200):
    """Build an aiohttp.ClientSession stand-in that records request methods
    and answers `status` for every request."""

    class _Resp:
        status = 200

        async def json(self, *a, **k):
            return {"models": [{"name": "qwen3:1.7b"}]}

    class _Ctx:
        async def __aenter__(self):
            return _Resp()

        async def __aexit__(self, *a):
            return False

    class _Session:
        def __init__(self, *a, **k):
            pass

        def get(self, url):
            methods_seen.append("GET")
            return _Ctx()

        def post(self, url, json=None):
            methods_seen.append("POST")
            return _Ctx()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    return _Session


def _patch_http(monkeypatch, methods_seen: list, status: int = 200):
    monkeypatch.setattr(local_router.aiohttp, "ClientTimeout",
                        lambda *a, **k: 1.0)
    monkeypatch.setattr(local_router.aiohttp, "ClientSession",
                        _fake_session_factory(methods_seen, status))


def test_available_health_check_uses_get(monkeypatch):
    """Ollama's /api/tags is read-only: a POST gets HTTP 405 and made the
    lab treat a RUNNING Ollama as absent (live 08:35). The check must GET."""
    methods: list = []
    _patch_http(monkeypatch, methods)
    assert asyncio.run(local_router.available(_settings())) is True
    assert methods == ["GET"]


def test_available_false_on_405_style_failure(monkeypatch, caplog):
    """A 405 (or any non-200) on /api/tags must read as 'unavailable',
    never raise."""
    methods: list = []
    _patch_http(monkeypatch, methods, status=405)

    class _Resp405:
        status = 405

        async def json(self, *a, **k):
            return {}

    class _Ctx:
        async def __aenter__(self):
            return _Resp405()

        async def __aexit__(self, *a):
            return False

    class _Session405:
        def __init__(self, *a, **k):
            pass

        def get(self, url):
            methods.append("GET")
            return _Ctx()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(local_router.aiohttp, "ClientSession", _Session405)
    assert asyncio.run(local_router.available(_settings())) is False


# ---------------------------------------- scaffolding purposes may fall back
def test_plan_and_role_synthesis_may_fallback_locally():
    """plan/role_synthesis are SCAFFOLDING, not verdicts (live 08:35: every
    cycle starved at `plan blocked by budget`). They may fall back to the
    local model when the pool refuses — the science that follows still runs
    pool-side."""
    s = _settings()  # local_llm_purposes = "kg_extract,classify"
    assert local_router.allowed_for(s, "plan") is False  # not listed yet
    s2 = _settings(local_llm_purposes="kg_extract,plan,role_synthesis")
    assert local_router.allowed_for(s2, "plan") is True
    assert local_router.allowed_for(s2, "role_synthesis") is True
    # real science purposes still refuse, even when someone lists them
    s3 = _settings(local_llm_purposes="plan,role_synthesis,verify,research")
    assert local_router.allowed_for(s3, "verify") is False
    assert local_router.allowed_for(s3, "research") is False
    assert local_router.allowed_for(s3, "ideation") is False


def test_default_purpose_list_includes_scaffolding():
    import agentics.config as config
    cfg = config.Settings()
    assert "plan" in cfg.local_llm_purposes
    assert "role_synthesis" in cfg.local_llm_purposes
    assert "kg_extract" in cfg.local_llm_purposes
