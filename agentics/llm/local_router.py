"""Local small-model router (#3 of the advanced-tech integration): a FREE,
offline LLM edge (Ollama, llama.cpp OpenAI-compatible, LM Studio) for the
lab's cheap purposes — extraction/planning/system-prompt synthesis — so the
cookie pool is spent ONLY on science.

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


def _timeout(settings=None) -> float:
    return float(getattr(settings, "local_llm_timeout", 60.0) or 60.0)


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
    """True when the local runtime answers GET /api/tags (Ollama).

    GET, not POST: Ollama's /api/tags is a read-only endpoint and answers
    405 Method Not Allowed to a POST (live 2026-11-14 08:35 — the lab
    logged `HTTP 405` every cycle while Ollama was actually RUNNING, so
    kg ingest silently mined nothing)."""
    base = str(getattr(settings, "local_llm_url", "") or "").rstrip("/")
    if not base:
        return False
    data = await _get(f"{base}/api/tags", settings=settings)
    return bool(data)


def _strip_think(text: str) -> str:
    """qwen3 (and other reasoning models) prepend a <think>...</think>
    block before the answer — the lab's parsers only want the answer."""
    return re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)


async def ask(prompt: str, settings=None, purpose: str = "local",
              system: str = "") -> SimpleNamespace | None:
    """One Ollama /api/chat completion. Returns an LLMResponse-shaped
    namespace (answer/citations/ok) or None when the runtime is absent/
    failing. Deliberately NOT budget-recorded: local calls burn no cookie
    quota (see module docstring).

    CPU-realistic completion contract (live 2026-11-14 08:51 — three
    silent `api/chat failed:` warnings, all TimeoutError, whose str() is
    empty): the FIRST call also loads the model into RAM (1.7B ≈ 30-60s
    on CPU), qwen3 is a reasoning model that would burn hundreds of
    tokens in <think>, and without num_predict Ollama generates up to
    2048 tokens. So: bounded generation (num_predict), thinking off by
    default (LOCAL_LLM_THINK=1 to re-enable), a long keep_alive so the
    model stays warm between cycles, and the default timeout raised to
    120s (LOCAL_LLM_TIMEOUT)."""
    base = str(getattr(settings, "local_llm_url", "") or "").rstrip("/")
    model = str(getattr(settings, "local_llm_model", "") or "").strip()
    if not base or not model or not prompt.strip():
        return None
    started = time.monotonic()
    messages = []
    if system.strip():
        messages.append({"role": "system", "content": system[:2000]})
    messages.append({"role": "user", "content": prompt[:16000]})
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "keep_alive": "30m",
        "options": {
            "temperature": 0.2,
            "num_predict": int(getattr(settings, "local_llm_num_predict",
                                       512) or 512),
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
                      "prompt_evolve,reflect") or "")
    return tuple(p.strip().lower() for p in raw.split(",") if p.strip()
                 and p.strip().lower() not in _SCIENCE_PURPOSES)


def allowed_for(settings, purpose: str) -> bool:
    """Science purposes never degrade to the small model, even by accident."""
    p = (purpose or "").lower()
    if p in _SCIENCE_PURPOSES:
        return False
    return (bool(getattr(settings, "local_llm_enabled", False))
            and p in purposes_for(settings))
