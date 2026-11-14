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

        def get(self, url, headers=None):
            methods_seen.append("GET")
            return _Ctx()

        def post(self, url, json=None, headers=None):
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

        def get(self, url, headers=None):
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


# ------------------------------- OmniRoute / OpenAI-compatible backend (2026)
def _clear_mode_cache():
    local_router._mode_cache.clear()


def test_mode_heuristic_by_url_shape():
    """auto: a base ending /v1 speaks OpenAI chat completions, a bare
    host:port speaks Ollama native; an explicit LOCAL_LLM_BACKEND pin
    overrides the URL shape in both directions."""
    _clear_mode_cache()
    s = _settings()  # no backend attr -> auto
    assert local_router._mode("http://127.0.0.1:20128/v1", s) == "openai"
    assert local_router._mode("http://127.0.0.1:11434", s) == "ollama"
    assert local_router._mode("http://127.0.0.1:20128/v1",
                              _settings(local_llm_backend="ollama")) == "ollama"
    assert local_router._mode("http://127.0.0.1:11434",
                              _settings(local_llm_backend="omniroute")) == "openai"


def test_ask_openai_payload_posts_to_chat_completions(monkeypatch):
    """OmniRoute tier: one /v1/chat/completions call — OpenAI body shape,
    bounded generation (max_tokens), Nemotron thinking OFF by default via
    the NIM chat_template_kwargs knob, no Ollama-only fields."""
    _clear_mode_cache()
    captured = {}

    async def fake_post(url, payload, settings=None):
        captured["url"] = url
        captured["payload"] = payload
        return {"choices": [{"message": {"content": "{\"files\": []}"}}]}

    monkeypatch.setattr(local_router, "_post", fake_post)
    s = _settings(local_llm_url="http://127.0.0.1:20128/v1",
                  local_llm_model="nvidia/nemotron-3-ultra-550b-a55b")
    resp = asyncio.run(local_router.ask("plan this", s, purpose="plan"))
    assert resp is not None and resp.ok
    assert resp.answer == '{"files": []}'
    assert resp.citations == []
    assert captured["url"] == "http://127.0.0.1:20128/v1/chat/completions"
    p = captured["payload"]
    assert p["model"] == "nvidia/nemotron-3-ultra-550b-a55b"
    assert p["stream"] is False
    assert p["max_tokens"] == 512
    assert p["chat_template_kwargs"] == {"enable_thinking": False}
    assert "keep_alive" not in p and "options" not in p and "think" not in p


def test_ask_openai_bare_host_gets_v1_appended(monkeypatch):
    """LOCAL_LLM_BACKEND=omniroute on a bare host still hits /v1/..."""
    _clear_mode_cache()
    captured = {}

    async def fake_post(url, payload, settings=None):
        captured["url"] = url
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(local_router, "_post", fake_post)
    s = _settings(local_llm_url="http://127.0.0.1:20128",
                  local_llm_backend="omniroute")
    assert asyncio.run(local_router.ask("hi", s)) is not None
    assert captured["url"] == "http://127.0.0.1:20128/v1/chat/completions"


def test_ask_openai_think_env_omits_knob(monkeypatch):
    _clear_mode_cache()
    captured = {}

    async def fake_post(url, payload, settings=None):
        captured.update(payload)
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(local_router, "_post", fake_post)
    s = _settings(local_llm_url="http://127.0.0.1:20128/v1",
                  local_llm_think=True)
    assert asyncio.run(local_router.ask("hi", s)) is not None
    assert "chat_template_kwargs" not in captured  # model may think


def test_ask_openai_reasoning_content_is_not_answer(monkeypatch):
    """NIM Nemotron emits reasoning separately; the lab's parsers want the
    answer only — reasoning_content must never leak into `answer`."""
    _clear_mode_cache()

    async def fake_post(url, payload, settings=None):
        return {"choices": [{"message": {
            "content": "", "reasoning_content": "long chain of thought"}}]}

    monkeypatch.setattr(local_router, "_post", fake_post)
    s = _settings(local_llm_url="http://127.0.0.1:20128/v1")
    assert asyncio.run(local_router.ask("hi", s)) is None


def test_ask_openai_strips_think_blocks(monkeypatch):
    _clear_mode_cache()

    async def fake_post(url, payload, settings=None):
        return {"choices": [{"message": {"content": "<think>hmm</think>done"}}]}

    monkeypatch.setattr(local_router, "_post", fake_post)
    s = _settings(local_llm_url="http://127.0.0.1:20128/v1")
    resp = asyncio.run(local_router.ask("hi", s))
    assert resp is not None and resp.answer == "done"


def test_available_auto_probes_openai_then_ollama(monkeypatch):
    """auto health: GET /v1/models first (OmniRoute/vLLM/LM Studio), then
    GET /api/tags (Ollama native); the confirmed mode is cached so ask()
    never probes."""
    seen: list[str] = []

    async def fake_get(url, settings=None):
        seen.append(url)
        return {"data": []} if url.endswith("/models") else None

    monkeypatch.setattr(local_router, "_get", fake_get)
    _clear_mode_cache()
    s = _settings(local_llm_url="http://127.0.0.1:20128/v1")
    assert asyncio.run(local_router.available(s)) is True
    assert seen == ["http://127.0.0.1:20128/v1/models"]
    assert local_router._mode_cache["http://127.0.0.1:20128/v1"] == "openai"

    seen.clear()

    async def ollama_get(url, settings=None):
        seen.append(url)
        return {"models": []} if url.endswith("/api/tags") else None

    monkeypatch.setattr(local_router, "_get", ollama_get)
    _clear_mode_cache()
    s2 = _settings(local_llm_url="http://127.0.0.1:11434")
    assert asyncio.run(local_router.available(s2)) is True
    assert seen == ["http://127.0.0.1:11434/v1/models",
                    "http://127.0.0.1:11434/api/tags"]
    assert local_router._mode_cache["http://127.0.0.1:11434"] == "ollama"


def test_default_config_points_at_omniroute_and_nemotron():
    """The owner's stack: OmniRoute gateway (:20128/v1) + NVIDIA Nemotron
    3 Ultra 550B as the local-tier model id (env still overrides)."""
    import agentics.config as config
    cfg = config.Settings()
    assert cfg.local_llm_backend == "auto"
    assert cfg.local_llm_url.rstrip("/").endswith("/v1")
    assert "nemotron" in cfg.local_llm_model.lower()


# --------------------------------------- gateway auth (OmniRoute API Keys)
def test_headers_bearer_only_when_key_set():
    """LOCAL_LLM_API_KEY empty (the localhost default) must send NO
    Authorization header at all; a set key becomes a Bearer header
    (whitespace-stripped)."""
    assert local_router._headers(_settings()) == {}
    assert local_router._headers(_settings(local_llm_api_key="")) == {}
    assert local_router._headers(
        _settings(local_llm_api_key=" omni-key-123 ")) == {
            "Authorization": "Bearer omni-key-123"}


def test_post_carries_bearer_when_key_set(monkeypatch):
    """The OmniRoute gateway can require its API Key for non-localhost
    access (LAN 192.168.x.x:20128 / tunnels) — every POST must carry it."""
    seen = {}

    class _Resp:
        status = 200

        async def json(self, *a, **k):
            return {"choices": [{"message": {"content": "ok"}}]}

    class _Ctx:
        async def __aenter__(self):
            return _Resp()

        async def __aexit__(self, *a):
            return False

    class _Session:
        def __init__(self, *a, **k):
            pass

        def post(self, url, json=None, headers=None):
            seen["headers"] = headers or {}
            return _Ctx()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(local_router.aiohttp, "ClientTimeout",
                        lambda *a, **k: 1.0)
    monkeypatch.setattr(local_router.aiohttp, "ClientSession", _Session)
    _clear_mode_cache()
    s = _settings(local_llm_url="http://127.0.0.1:20128/v1",
                  local_llm_api_key="omni-key-123")
    assert asyncio.run(local_router.ask("hi", s)) is not None
    assert seen["headers"].get("Authorization") == "Bearer omni-key-123"

    # no key configured -> no Authorization header (plain localhost gateway)
    seen.clear()
    s2 = _settings(local_llm_url="http://127.0.0.1:20128/v1")
    assert asyncio.run(local_router.ask("hi", s2)) is not None
    assert "Authorization" not in seen["headers"]


# ---------------------- live 2026-11-14 21:25 incident: HTTP 502 bursts
def _burst_settings(**kw):
    return _settings(local_llm_url="http://127.0.0.1:20128/v1",
                     local_llm_model="nvidia/nemotron-3-ultra-550b-a55b",
                     **kw)


def test_ask_serializes_concurrent_calls(monkeypatch):
    """Live 21:25: two SAME-SECOND lab asks both came back HTTP 502 while a
    single dashboard-chat completion kept working — the gateway's upstream
    serves one completion at a time. The default gate
    (local_llm_max_concurrent=1) must keep at most ONE local ask in flight."""
    _clear_mode_cache()
    state = {"in_flight": 0, "max_in_flight": 0}

    async def fake_post(url, payload, settings=None):
        state["in_flight"] += 1
        state["max_in_flight"] = max(state["max_in_flight"],
                                     state["in_flight"])
        await asyncio.sleep(0.02)
        state["in_flight"] -= 1
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(local_router, "_post", fake_post)

    async def burst():
        return await asyncio.gather(
            *[local_router.ask("hi", _burst_settings()) for _ in range(4)])

    results = asyncio.run(burst())
    assert all(r is not None and r.ok for r in results)
    assert state["max_in_flight"] == 1


def test_gate_respects_local_llm_max_concurrent(monkeypatch):
    """A host whose upstream PROVABLY handles parallel completions (probe
    --ask --concurrency 3) may raise the gate; the semaphore then admits
    exactly that many in flight."""
    _clear_mode_cache()
    state = {"in_flight": 0, "max_in_flight": 0}

    async def fake_post(url, payload, settings=None):
        state["in_flight"] += 1
        state["max_in_flight"] = max(state["max_in_flight"],
                                     state["in_flight"])
        await asyncio.sleep(0.02)
        state["in_flight"] -= 1
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(local_router, "_post", fake_post)
    s = _burst_settings(local_llm_max_concurrent=2)

    async def burst():
        return await asyncio.gather(*[local_router.ask("hi", s)
                                      for _ in range(4)])

    results = asyncio.run(burst())
    assert all(r is not None for r in results)
    assert state["max_in_flight"] == 2


def test_default_max_concurrent_is_serialized():
    """Shipped default = 1: the honest edge queues instead of bursting."""
    import agentics.config as config
    cfg = config.Settings()
    assert cfg.local_llm_max_concurrent == 1


def _fake_gateway_session(posts: list, statuses: list):
    """aiohttp.ClientSession stand-in answering `statuses` in sequence;
    non-200 bodies carry the upstream's real error text."""
    class _Resp:
        def __init__(self, status):
            self.status = status

        async def text(self):
            return ('{"error": {"message": "upstream provider nvidia: '
                    'model busy"}}')

        async def json(self, *a, **k):
            return {"choices": [{"message": {"content": "recovered"}}]}

    class _Ctx:
        def __init__(self, resp):
            self._resp = resp

        async def __aenter__(self):
            return self._resp

        async def __aexit__(self, *a):
            return False

    class _Session:
        def __init__(self, *a, **k):
            pass

        def post(self, url, json=None, headers=None):
            posts.append(statuses[min(len(posts), len(statuses) - 1)])
            return _Ctx(_Resp(posts[-1]))

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    return _Session


def _patch_gateway(monkeypatch, posts: list, statuses: list):
    monkeypatch.setattr(local_router.aiohttp, "ClientTimeout",
                        lambda *a, **k: 1.0)
    monkeypatch.setattr(local_router.aiohttp, "ClientSession",
                        _fake_gateway_session(posts, statuses))


def test_post_logs_model_and_body_on_502(monkeypatch, caplog):
    """The bare `-> HTTP 502` warning hid the real upstream error for a
    whole incident (2026-11-14 21:25). Non-200 must log the configured
    model AND the gateway's error body (truncated)."""
    import logging as _logging
    posts: list = []
    _patch_gateway(monkeypatch, posts, [502])
    monkeypatch.setattr(local_router, "_RETRY_DELAY_S", 0)
    with caplog.at_level(_logging.WARNING, logger="llm.local_router"):
        assert asyncio.run(local_router._post(
            "http://127.0.0.1:20128/v1/chat/completions",
            {"model": "nvidia/nemotron-3-ultra-550b-a55b"})) is None
    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "model=nvidia/nemotron-3-ultra-550b-a55b" in joined
    assert "upstream provider nvidia" in joined
    assert len(posts) == 2                    # exactly ONE quiet retry


def test_post_retries_once_on_502_then_succeeds(monkeypatch):
    """429/502/503 get exactly ONE quiet retry — the gateway's upstream
    hiccup recovers in seconds; without it the caller loses a whole agent
    cycle. Anything that then succeeds is returned normally."""
    posts: list = []
    _patch_gateway(monkeypatch, posts, [502, 200])
    monkeypatch.setattr(local_router, "_RETRY_DELAY_S", 0)
    data = asyncio.run(local_router._post(
        "http://127.0.0.1:20128/v1/chat/completions", {"model": "m"}))
    assert data is not None
    assert data["choices"][0]["message"]["content"] == "recovered"
    assert posts == [502, 200]


def test_post_does_not_retry_non_retryable(monkeypatch):
    """400/401/404 are CONFIG errors — retrying them is pure noise; one
    attempt, then the warning with the body."""
    posts: list = []
    _patch_gateway(monkeypatch, posts, [400])
    assert asyncio.run(local_router._post(
        "http://127.0.0.1:20128/v1/chat/completions", {"model": "m"})) is None
    assert posts == [400]
