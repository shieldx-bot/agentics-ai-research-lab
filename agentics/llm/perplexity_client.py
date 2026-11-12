"""Perplexity client — cookie-based WebSocket (socket.io v4) access.

.. warning::
    Uses browser session cookies to talk to Perplexity's internal socket.io API.
    Perplexity may change its internal endpoints at any time and this access may
    conflict with Perplexity's Terms of Service. Use your own accounts, for
    personal/academic research, at your own risk.

Protocol sketch (EIO=4):
    server -> "0{sid}"                    (open)
    client -> "40"                        (connect namespace)
    server -> "40{sid}"                   (connected)
    client -> '420["perplexity_ask", {…}]'
    server -> '42["message", {...}]'      (streaming chunks & final answer)

The final message carries "text" plus "citations"/"web_results" — parsed into
:class:`LLMResponse`.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any

import aiohttp

log = logging.getLogger("llm.pplx")

ASK_EVENT = "perplexity_ask"
MSG_EVENT = "message"

# frontend progress codes -> human labels (informational only)
_STATUS_LABELS = {
    0: "backend processing",
    1: "querying public index",
    2: "chopping results",
    3: "fetching question",
    4: "fetching related",
    5: "generating related",
    6: "llm talk",
}


@dataclass
class LLMResponse:
    answer: str
    citations: list[str] = field(default_factory=list)
    raw: dict[str, Any] | None = None
    elapsed_s: float = 0.0

    @property
    def ok(self) -> bool:
        return bool(self.answer.strip())


@dataclass
class CookieAccount:
    name: str
    cookies: dict[str, str]
    path: str = ""

    def cookie_header(self) -> str:
        return "; ".join(f"{k}={v}" for k, v in self.cookies.items())


UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)


class PerplexityClient:
    """One client = one cookie account. Not thread-safe; use per asyncio task."""

    def __init__(
        self,
        account: CookieAccount,
        ws_url: str = "wss://www.perplexity.ai/socket.io/?EIO=4&transport=websocket",
        origin: str = "https://www.perplexity.ai",
        timeout: int = 180,
    ) -> None:
        self.account = account
        self.ws_url = ws_url
        self.origin = origin
        self.timeout = timeout
        self._sid: str | None = None

    # ------------------------------------------------------------------ ws
    def _headers(self) -> dict[str, str]:
        return {
            "User-Agent": UA,
            "Origin": self.origin,
            "Cookie": self.account.cookie_header(),
        }

    async def _handshake(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        raw = await asyncio.wait_for(ws.receive(), timeout=30)
        if raw.type != aiohttp.WSMsgType.TEXT or not raw.data.startswith("0"):
            raise ConnectionError(f"unexpected open frame: {raw.data!r:.120}")
        open_msg = json.loads(raw.data[1:])
        self._sid = open_msg.get("sid", "")
        await ws.send_str("40")
        raw = await asyncio.wait_for(ws.receive(), timeout=30)
        if raw.type != aiohttp.WSMsgType.TEXT or not raw.data.startswith("40"):
            raise ConnectionError(f"namespace connect failed: {raw.data!r:.120}")

    async def _emit_ask(self, ws: aiohttp.ClientWebSocketResponse, payload: dict) -> None:
        frame = "42" + json.dumps([ASK_EVENT, payload], ensure_ascii=False)
        await ws.send_str(frame)

    # ------------------------------------------------------------------ ask
    async def ask(
        self,
        query: str,
        *,
        mode: str = "academic",
        system: str | None = None,
        frontend_uuid: str | None = None,
        purpose: str | None = None,
    ) -> LLMResponse:
        """Send one query; returns final answer + citations.

        mode: "academic" biases Perplexity towards scholarly sources (good for
        paper search); "concise" is faster for coding/writing tasks.
        purpose: routing hint consumed by BrowserSessionHost; ignored here.
        """
        started = time.monotonic()
        uuid = frontend_uuid or _fake_uuid()
        payload: dict[str, Any] = {
            "query": query,
            "mode": mode,
            "frontend_uuid": uuid,
            "is_related_query": False,
            "allow_citations": True,
        }
        if system:
            payload["prompt_prefix"] = system

        timeout = aiohttp.ClientTimeout(total=self.timeout + 30)
        async with aiohttp.ClientSession(timeout=timeout, headers=self._headers()) as sess:
            async with sess.ws_connect(self.ws_url, headers=self._headers()) as ws:
                await self._handshake(ws)
                await self._emit_ask(ws, payload)

                answer_parts: list[str] = []
                citations: list[str] = []
                final_raw: dict[str, Any] = {}
                deadline = time.monotonic() + self.timeout

                while time.monotonic() < deadline:
                    raw = await asyncio.wait_for(ws.receive(), timeout=45)
                    if raw.type == aiohttp.WSMsgType.CLOSED or raw.type == aiohttp.WSMsgType.ERROR:
                        break
                    if raw.type != aiohttp.WSMsgType.TEXT:
                        continue
                    data = raw.data
                    if data == "2":            # engine.io ping
                        await ws.send_str("3") # pong
                        continue
                    if data == "3":
                        continue
                    if not data.startswith("42"):
                        continue
                    try:
                        event, msg = json.loads(data[2:])
                    except json.JSONDecodeError:
                        continue
                    if event != MSG_EVENT or not isinstance(msg, dict):
                        continue

                    status = msg.get("status_code")
                    if status in _STATUS_LABELS:      # progress tick
                        continue

                    text = msg.get("text") or ""
                    if text:
                        answer_parts = [text]          # cumulative text frames
                    if msg.get("citations"):
                        citations = list(msg["citations"])
                    if msg.get("web_results"):
                        for wr in msg["web_results"]:
                            url = wr.get("url") if isinstance(wr, dict) else None
                            if url and url not in citations:
                                citations.append(url)

                    if msg.get("mode") in ("copilot", "concise", "academic") and (
                        msg.get("step_idx") is not None or text
                    ):
                        final_raw = msg
                        if text and msg.get("is_completed", True):
                            break

                answer = "\n".join(answer_parts).strip()
                return LLMResponse(
                    answer=answer,
                    citations=citations,
                    raw=final_raw or None,
                    elapsed_s=round(time.monotonic() - started, 2),
                )


def _fake_uuid() -> str:
    return "-".join(
        "".join(random.choices("0123456789abcdef", k=n)) for n in (8, 4, 4, 4, 12)
    )
