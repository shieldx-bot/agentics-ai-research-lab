"""Qwen client — cookie-based access to the chat.qwen.ai native web API.

.. warning::
    Uses browser session cookies / tokens to talk to Qwen's internal web API.
    This may conflict with Qwen's Terms of Service and the API can change or
    challenge with a WAF at any time. Use your own accounts for personal /
    academic research, at your own risk.

Live-verified facts (2026-11-12, see docs/LLM_PROVIDERS.md):

* Access tokens (localStorage ``token``) live **~15 minutes**.
  Refresh tokens (cookie ``refresh_token``) live **~30 days**.
* Refresh: ``POST https://chat.qwen.ai/api/v2/auths/refresh``
  ``{"refresh_token": ...}`` -> ``{"data": {"token": ...}}``.
* Chat: ``POST https://chat.qwen.ai/api/v2/chat/completions`` (streaming,
  ``incremental_output`` SSE frames of ``{"data": {"phase": ..., "content": ...}}``).
* Models (``GET /api/models``): ``qwen3.7-plus``, ``qwen3.8-max``,
  ``qwen4.0-max-preview`` (Qwen4-Max-Preview — flagship preview).
* Alibaba's baxia WAF (``RGV587_ERROR``) rejects raw HTTP calls that do not
  carry browser-grade anti-bot tokens (``tfstk`` cookie, ``bx-ua``). Cookies
  exported from a *live* browser session usually pass; stale exports do not.
  When blocked, re-export cookies from the logged-in browser, or drive the
  session through a real browser (Playwright/agent-browser).

Account file format (``accounts/qwen-*.json``)::

    {
      "cookies": [ {"name": "...", "value": "...", "domain": ".qwen.ai"}, ... ],
      "refresh_token": "eyJ...",       // optional, from refresh_token cookie
      "access_token": "eyJ...",        // optional, from localStorage token
      "model": "qwen4.0-max-preview"   // optional override
    }
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
import uuid as _uuid
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import aiohttp

from agentics.llm.perplexity_client import LLMResponse, UA

log = logging.getLogger("llm.qwen")

QWEN_BASE = "https://chat.qwen.ai"
CHAT_URL = f"{QWEN_BASE}/api/v2/chat/completions"
REFRESH_URL = f"{QWEN_BASE}/api/v2/auths/refresh"
MODELS_URL = f"{QWEN_BASE}/api/models"
QWEN_DOMAIN = "qwen.ai"
DEFAULT_MODEL = "qwen4.0-max-preview"

# modes that justify web search on Qwen's side (auto_search feature flag)
SEARCH_MODES = ("academic", "research")

USER_AGENT = UA.replace("Chrome/126", "Chrome/148")


@dataclass
class QwenAccount:
    """One Qwen session: browser cookies + tokens from a manual export."""
    name: str
    cookies: dict[str, str] = field(default_factory=dict)
    refresh_token: str = ""
    access_token: str = ""
    model: str = DEFAULT_MODEL
    path: str = ""

    def cookie_header(self) -> str:
        return "; ".join(f"{k}={v}" for k, v in self.cookies.items())


def load_qwen_account(path) -> QwenAccount:
    """Load ``accounts/qwen-*.json`` (cookies list + optional tokens)."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    items = raw.get("cookies", raw if isinstance(raw, list) else [])
    cookies = {
        str(it.get("name")): str(it.get("value"))
        for it in items
        if isinstance(it, dict) and it.get("name") and it.get("value")
        and QWEN_DOMAIN in str(it.get("domain", ""))
    }
    # a bare refresh_token cookie in the export also works
    rt = str(raw.get("refresh_token") or cookies.get("refresh_token") or "")
    at = str(raw.get("access_token") or "")
    model = str(raw.get("model") or DEFAULT_MODEL)
    return QwenAccount(
        name=path.stem, cookies=cookies, refresh_token=rt,
        access_token=at, model=model, path=str(path),
    )


# ------------------------------------------------------------------ helpers
def jwt_exp(token: str) -> float | None:
    """Expiry (epoch seconds) of a JWT access token; None if unparsable."""
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return float(json.loads(base64.urlsafe_b64decode(payload))["exp"])
    except Exception:  # noqa: BLE001
        return None


def token_expired(token: str, margin_s: float = 120.0) -> bool:
    exp = jwt_exp(token)
    return exp is None or exp - margin_s < time.time()


def build_chat_payload(query: str, model: str, system: str | None,
                       auto_search: bool, chat_id: str | None = None) -> dict:
    """Native v2 payload (verified against chat.qwen.ai web app, 2026-11)."""
    ts = int(time.time())
    feature_config = {
        "thinking_enabled": True, "output_schema": "phase",
        "research_mode": "normal", "auto_thinking": False,
        "thinking_mode": "Thinking", "thinking_format": "summary",
        "auto_search": auto_search,
    }
    if system:
        query = f"{system}\n\n{query}"
    message = {
        "fid": str(_uuid.uuid4()), "parentId": None, "childrenIds": [],
        "role": "user", "content": query, "user_action": "chat",
        "files": [], "timestamp": ts, "models": [model],
        "chat_type": "t2t", "feature_config": feature_config,
        "extra": {"meta": {"subChatType": "t2t"}},
        "sub_chat_type": "t2t", "parent_id": None,
    }
    return {
        "stream": True, "version": "2.1", "incremental_output": True,
        "chat_id": chat_id, "chat_mode": "normal", "model": model,
        "parent_id": None, "messages": [message], "timestamp": ts + 1,
    }


def parse_sse_frames(text: str) -> tuple[str, list[str]]:
    """Extract (answer, citations) from the streamed body.

    Frames are JSON objects; Qwen emits ``{"data": {"phase": "answer",
    "content": <incremental text>}}`` (and sometimes OpenAI-ish
    ``choices[0].delta.content``). Non-JSON lines are ignored.
    """
    parts: list[str] = []
    citations: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            line = line[5:].strip()
        if not line:
            continue
        try:
            frame = json.loads(line)
        except json.JSONDecodeError:
            continue
        for delta in _iter_deltas(frame):
            phase = delta.get("phase")
            # Qwen phase protocol: only "answer" frames carry output text.
            # OpenAI-style deltas carry bare content with no phase at all.
            if (phase == "answer" or phase is None) and isinstance(delta.get("content"), str):
                parts.append(delta["content"])
            url = delta.get("url")
            if isinstance(url, str) and urlparse(url).scheme in ("http", "https"):
                if url not in citations:
                    citations.append(url)
    return "".join(parts).strip(), citations


def _iter_deltas(frame: Any):
    data = frame.get("data") if isinstance(frame, dict) else None
    if isinstance(data, dict):
        yield data
    if isinstance(frame, dict) and frame.get("choices"):
        try:
            yield frame["choices"][0].get("delta", {})
        except (IndexError, AttributeError):
            return


def waf_blocked(error_text: str) -> bool:
    return "RGV587" in error_text or "FAIL_SYS_USER_VALIDATE" in error_text


# ------------------------------------------------------------------ client
class QwenClient:
    """One client = one Qwen session. Not thread-safe; use per asyncio task."""

    def __init__(
        self,
        account: QwenAccount,
        base_url: str = QWEN_BASE,
        timeout: int = 180,
    ) -> None:
        self.account = account
        self.base_url = base_url.rstrip("/")
        self.chat_url = f"{self.base_url}/api/v2/chat/completions"
        self.refresh_url = f"{self.base_url}/api/v2/auths/refresh"
        self.timeout = timeout

    # ------------------------------------------------------------------ auth
    def _headers(self, token: str | None) -> dict[str, str]:
        h = {
            "accept": "application/json, text/plain, */*",
            "content-type": "application/json",
            "user-agent": USER_AGENT,
            "origin": self.base_url,
            "referer": f"{self.base_url}/",
            "source": "web",
            "x-request-id": str(_uuid.uuid4()),
            "bx-v": "2.5.36",
            "cookie": self.account.cookie_header(),
        }
        if token:
            h["authorization"] = f"Bearer {token}"
        return h

    async def refresh_access_token(self, sess: aiohttp.ClientSession) -> bool:
        """Exchange the refresh token for a fresh access token (~15 min TTL)."""
        if not self.account.refresh_token:
            return False
        async with sess.post(self.refresh_url,
                             json={"refresh_token": self.account.refresh_token}) as resp:
            body = await resp.text()
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                log.warning("qwen refresh: unparsable response")
                return False
            token = (data.get("data") or {}).get("token") if isinstance(data.get("data"), dict) else None
            if resp.status == 200 and data.get("success") is not False and token:
                self.account.access_token = token
                new_rt = data["data"].get("refresh_token")
                if new_rt:
                    self.account.refresh_token = new_rt
                log.info("qwen access token refreshed for %s", self.account.name)
                return True
            log.warning("qwen refresh rejected for %s: %s",
                        self.account.name, (data.get("data") or {}).get("details", body[:120]))
            return False

    async def _ensure_token(self, sess: aiohttp.ClientSession) -> str | None:
        """A valid access token when possible; cookie-only fallback otherwise."""
        if self.account.access_token and not token_expired(self.account.access_token):
            return self.account.access_token
        if await self.refresh_access_token(sess):
            return self.account.access_token
        return None                     # cookie-only attempt (WAF permitting)

    # ------------------------------------------------------------------ ask
    async def ask(
        self,
        query: str,
        *,
        mode: str = "concise",
        system: str | None = None,
        purpose: str | None = None,
    ) -> LLMResponse:
        """Send one query; returns the final answer text (+ any link citations).

        purpose: routing hint consumed by BrowserSessionHost; ignored here.
        """
        started = time.monotonic()
        timeout = aiohttp.ClientTimeout(total=self.timeout + 30)
        async with aiohttp.ClientSession(timeout=timeout) as sess:
            token = await self._ensure_token(sess)
            payload = build_chat_payload(
                query, self.account.model, system,
                auto_search=mode in SEARCH_MODES,
            )
            async with sess.post(self.chat_url, json=payload,
                                 headers=self._headers(token)) as resp:
                text = await resp.text()
                if resp.status != 200:
                    raise RuntimeError(
                        f"qwen chat HTTP {resp.status}: {text[:200]}")
            if waf_blocked(text):
                raise RuntimeError(
                    "qwen chat blocked by baxia WAF (RGV587): re-export cookies "
                    "from a LIVE logged-in browser session (must include the "
                    "tfstk cookie) or drive chat.qwen.ai through browser automation")
            answer, citations = parse_sse_frames(text)
            if not answer:
                raise RuntimeError(f"qwen chat empty answer: {text[:200]}")
            return LLMResponse(
                answer=answer, citations=citations,
                elapsed_s=round(time.monotonic() - started, 2),
            )
