"""Local small-model router (#3 of the advanced-tech integration): a FREE,
offline LLM edge (OmniRoute / vLLM / LM Studio via the OpenAI-compatible
chat-completions API, or native Ollama) for the lab's cheap purposes —
extraction/planning/system-prompt synthesis — so the cookie pool is spent
ONLY on science.

Backends (LOCAL_LLM_BACKEND = auto | omniroute | ollama):
* ``omniroute`` (default tier): OmniRoute is a self-hosted OpenAI-compatible
  gateway (``npm i -g omniroute`` → http://localhost:20128/v1) that routes
  one /v1/chat/completions call across NVIDIA NIM, vLLM, LM Studio,
  llama.cpp and Ollama — so the lab's local tier can be the owner's big
  models (Nemotron 3 Ultra 550B A55B, Nemotron 3.5 Super VL) without the
  lab caring which runtime serves them;
* ``ollama``: the original native /api/chat path, kept verbatim for
  CPU-only setups (qwen3:1.7b);
* ``auto`` (default): a base URL ending in /v1 speaks OpenAI chat
  completions, a bare host:port speaks Ollama native — no probing in the
  ask path; `available()` probes and caches the confirmed mode per base.

Why this exists (live 2026-11-14 07:08): 121/120 pool calls in one hour,
mostly plumbing, and the science paths got budget-refused. A local model
converts the whole class of "pool exhausted" incidents into a no-op: cheap
purposes route here first (or fall back here when the pool refuses), the
cookies stay reserved for research/verify/ideation.

Design constraints (honest edge, same as every pool client):
* never raises — a missing/unresponsive local runtime returns ok=False;
* only the purposes in `local_llm_purposes` may use it (science purposes
  NEVER silently degrade to the small model — a weaker verdict is worse
  than a refused ask; plan/role_synthesis are scaffolding, not verdicts,
  and MAY fall back so a dry pool degrades to a local plan, not an idle
  agent);
* the P20 ledger does NOT count local asks: they consume no shared cookie
  quota, and inflating the ledger would starve the pool cap artificially.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from types import SimpleNamespace

import aiohttp

log = logging.getLogger("llm.local_router")

_local: SimpleNamespace | None = None      # process-wide singleton state
_mode_cache: dict[str, str] = {}           # base url -> confirmed protocol


def _timeout(settings=None) -> float:
    return float(getattr(settings, "local_llm_timeout", 60.0) or 60.0)


def _backend(settings=None) -> str:
    """Explicit protocol pin: 'openai' (OmniRoute/vLLM/LM Studio/llama.cpp
    all speak the same chat-completions shape), 'ollama', or 'auto'."""
    raw = str(getattr(settings, "local_llm_backend", "auto") or "auto")
    raw = raw.strip().lower()
    if raw in ("omniroute", "openai", "openai_compat", "openai-compatible"):
        return "openai"
    if raw == "ollama":
        return "ollama"
    return "auto"


def _openai_base(base: str) -> str:
    """OpenAI-compat bases carry /v1 (OmniRoute ships :20128/v1); tolerate a
    bare host by appending it so both spellings hit /v1/chat/completions."""
    base = base.rstrip("/")
    return base if base.endswith("/v1") else f"{base}/v1"


def _mode(base: str, settings=None) -> str:
    """Which protocol answers at ``base`` — cached when `available()` probed,
    else the URL-shape heuristic (no network in the ask path):
    ``.../v1`` → OpenAI chat completions, bare host:port → Ollama native."""
    cached = _mode_cache.get(base)
    if cached:
        return cached
    backend = _backend(settings)
    if backend != "auto":
        return backend
    return "openai" if base.rstrip("/").endswith("/v1") else "ollama"


async def _post(url: str, payload: dict,
                settings=None) -> dict | None:
    """One POST to the local runtime; None on any failure (never raises)."""
    try:
        timeout = aiohttp.ClientTimeout(total=_timeout(settings))
        async with aiohttp.ClientSession(timeout=timeout) as sess:
            async with sess.post(url, json=payload) as resp:
                if resp.status != 200:
                    log.warning("local_router %s -> HTTP %s",
                                url, resp.status)
                    return None
                return await resp.json(content_type=None)
    except Exception as exc:  # noqa: BLE001 — local outage must never break a cycle
        # type(exc).__name__ first: TimeoutError / ServerDisconnectedError
        # stringify to EMPTY and would log as `failed: ` (live 08:51).
        log.warning("local_router %s failed: %s: %s",
                    url, type(exc).__name__, exc)
        return None


async def _get(url: str, settings=None) -> dict | None:
    """One GET to the local runtime; None on any failure (never raises)."""
    try:
        timeout = aiohttp.ClientTimeout(total=_timeout(settings))
        async with aiohttp.ClientSession(timeout=timeout) as sess:
            async with sess.get(url) as resp:
                if resp.status != 200:
                    log.warning("local_router %s -> HTTP %s", url, resp.status)
                    return None
                return await resp.json(content_type=None)
    except Exception as exc:  # noqa: BLE001 — local outage must never break a cycle
        log.warning("local_router %s failed: %s: %s",
                    url, type(exc).__name__, exc)
        return None


async def available(settings=None) -> bool:
    """True when the local runtime answers a READ-ONLY health endpoint.

    Auto mode probes the OpenAI-compat tier first (OmniRoute/vLLM answer
    GET /v1/models), then falls back to Ollama's GET /api/tags; the
    confirmed protocol is cached so `ask()` never probes.

    GET, not POST: Ollama's /api/tags is a read-only endpoint and answers
    405 Method Not Allowed to a POST (live 2026-11-14 08:35 — the lab
    logged `HTTP 405` every cycle while Ollama was actually RUNNING, so
    kg ingest silently mined nothing)."""
    base = str(getattr(settings, "local_llm_url", "") or "").rstrip("/")
    if not base:
        return False
    backend = _backend(settings)
    if backend == "auto":
        if await _get(f"{_openai_base(base)}/models", settings=settings):
            _mode_cache[base] = "openai"
            return True
        if await _get(f"{base}/api/tags", settings=settings):
            _mode_cache[base] = "ollama"
            return True
        return False
    if backend == "openai":
        ok = bool(await _get(f"{_openai_base(base)}/models", settings=settings))
    else:
        ok = bool(await _get(f"{base}/api/tags", settings=settings))
    if ok:
        _mode_cache[base] = backend
    return ok


def _strip_think(text: str) -> str:
    """qwen3 (and other reasoning models) prepend a <think>...</think>
    block before the answer — the lab's parsers only want the answer."""
    return re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)


async def ask(prompt: str, settings=None, purpose: str = "local",
              system: str = "") -> SimpleNamespace | None:
    """One local completion via the configured protocol. Returns an
    LLMResponse-shaped namespace (answer/citations/ok) or None when the
    runtime is absent/failing. Deliberately NOT budget-recorded: local
    calls burn no cookie quota (see module docstring)."""
    base = str(getattr(settings, "local_llm_url", "") or "").rstrip("/")
    model = str(getattr(settings, "local_llm_model", "") or "").strip()
    if not base or not model or not prompt.strip():
        return None
    if _mode(base, settings) == "openai":
        return await _ask_openai(base, model, prompt, settings, system)
    return await _ask_ollama(base, model, prompt, settings, system)


def _messages(prompt: str, system: str) -> list[dict]:
    messages = []
    if system.strip():
        messages.append({"role": "system", "content": system[:2000]})
    messages.append({"role": "user", "content": prompt[:16000]})
    return messages


async def _ask_openai(base: str, model: str, prompt: str,
                      settings=None, system: str = "") -> SimpleNamespace | None:
    """One OpenAI-compatible /v1/chat/completions completion (OmniRoute,
    vLLM, LM Studio, llama.cpp server — and Ollama's own /v1 if pinned).

    Nemotron-3 reasoning models: thinking is OFF by default via
    ``chat_template_kwargs.enable_thinking`` (the exact NIM knob) so
    plumbing asks don't burn reasoning tokens; LOCAL_LLM_THINK=1 omits
    the field and lets the model think. Reasoning emitted separately as
    ``message.reasoning_content`` is ignored — the lab's parsers only
    want the answer."""
    started = time.monotonic()
    payload: dict = {
        "model": model,
        "messages": _messages(prompt, system),
        "stream": False,
        "temperature": 0.2,
        "max_tokens": int(getattr(settings, "local_llm_num_predict",
                                  512) or 512),
    }
    if not bool(getattr(settings, "local_llm_think", False)):
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    data = await _post(f"{_openai_base(base)}/chat/completions", payload,
                       settings=settings)
    if not isinstance(data, dict):
        return None
    choices = data.get("choices")
    content = ""
    if isinstance(choices, list) and choices:
        message = choices[0].get("message")
        if isinstance(message, dict):
            content = str(message.get("content") or "")
    content = _strip_think(content)
    if not content.strip():
        return None
    return SimpleNamespace(
        answer=content.strip(),
        citations=[],
        ok=True,
        elapsed_s=time.monotonic() - started,
        raw=data,
    )


async def _ask_ollama(base: str, model: str, prompt: str,
                      settings=None, system: str = "") -> SimpleNamespace | None:
    """One native Ollama /api/chat completion (legacy CPU path, verbatim).

    CPU-realistic completion contract (live 2026-11-14 08:51 — three
    silent `api/chat failed:` warnings, all TimeoutError, whose str() is
    empty): the FIRST call also loads the model into RAM (1.7B ≈ 30-60s
    on CPU), qwen3 is a reasoning model that would burn hundreds of
    tokens in <think>, and without num_predict Ollama generates up to
    2048 tokens. So: bounded generation (num_predict), a pinned context
    window (num_ctx, LOCAL_LLM_NUM_CTX — cache slots scale with it),
    thinking off by default (LOCAL_LLM_THINK=1 to re-enable), a short
    keep_alive so idle unloads free model + prompt cache
    (LOCAL_LLM_KEEP_ALIVE), and the default timeout raised to
    120s (LOCAL_LLM_TIMEOUT)."""
    started = time.monotonic()
    payload = {
        "model": model,
        "messages": _messages(prompt, system),
        "stream": False,
        # Configurable (LOCAL_LLM_KEEP_ALIVE, default 5m): while the model
        # is loaded, llama.cpp also keeps its prompt cache in RAM — which
        # can grow to ~8 GiB and is NOT configurable (upstream
        # ollama#17351). On a RAM-tight host the kernel OOM-killer then
        # takes down llama-server mid-request (live 2026-11-14 10:35 AND
        # ~13:43: global OOM, anon-rss 6 GB, ServerDisconnectedError in
        # the lab). A short keep_alive frees model + cache on idle; a GPU
        # reload costs only seconds.
        "keep_alive": str(getattr(settings, "local_llm_keep_alive",
                                  "5m") or "5m"),
        "options": {
            "temperature": 0.2,
            "num_predict": int(getattr(settings, "local_llm_num_predict",
                                       512) or 512),
            # Prompt-cache slots scale with the context window (~238 MiB
            # per slot at the model default ctx → ~7 GB RSS across the
            # ~30-slot LRU → the OOM above). Pin it explicitly; the lab's
            # prompts are designed for a 4096 ctx anyway.
            "num_ctx": int(getattr(settings, "local_llm_num_ctx",
                                   4096) or 4096),
        },
    }
    if not bool(getattr(settings, "local_llm_think", False)):
        payload["think"] = False          # ignored by pre-0.9 Ollama
    data = await _post(f"{base}/api/chat", payload, settings=settings)
    if not isinstance(data, dict):
        return None
    content = ""
    message = data.get("message")
    if isinstance(message, dict):
        content = str(message.get("content") or "")
    if not content and isinstance(data.get("response"), str):
        content = data["response"]          # llama.cpp-style payloads
    content = _strip_think(content)
    if not content.strip():
        return None
    return SimpleNamespace(
        answer=content.strip(),
        citations=[],
        ok=True,
        elapsed_s=time.monotonic() - started,
        raw=data,
    )


# Purposes that may NEVER route to the small model — even if someone
# lists them in LOCAL_LLM_PURPOSES. A weaker verdict/research answer is
# worse than a refused ask (the reserve lane should wait for the pool).
#
# `plan` and `role_synthesis` are deliberately NOT blacklisted (live
# 2026-11-14 08:35 — every cycle starved at `plan blocked by budget`): a
# mission plan / system prompt is SCAFFOLDING, not a verdict — the science
# that follows (research/verify/ideation) still runs pool-side, so a local
# plan beats an idle agent when the pool is dry. They route only as a P20c
# BUDGET FALLBACK (pool refused first), never proactively.
_SCIENCE_PURPOSES = frozenset({
    "research", "verify", "ideation", "implement", "review",
    "repair", "defend", "tournament", "tournament_judge2",
})


def purposes_for(settings) -> tuple[str, ...]:
    """The purposes allowed to route to the local model (config list,
    minus anything science-critical)."""
    raw = str(getattr(settings, "local_llm_purposes",
                      "kg_extract,plan,role_synthesis,"
                      "prompt_evolve,reflect,skill_distill") or "")
    return tuple(p.strip().lower() for p in raw.split(",") if p.strip()
                 and p.strip().lower() not in _SCIENCE_PURPOSES)


def allowed_for(settings, purpose: str) -> bool:
    """Science purposes never degrade to the small model, even by accident."""
    p = (purpose or "").lower()
    if p in _SCIENCE_PURPOSES:
        return False
    return (bool(getattr(settings, "local_llm_enabled", False))
            and p in purposes_for(settings))
