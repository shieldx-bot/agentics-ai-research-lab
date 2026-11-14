"""ChatGPT cloud-browser client — the SECOND, independent model family.

Why it exists
-------------
The lab's biggest correlated-error risk: hypothesis generation, tournament
debates, protocol writing, verdicts and P16-R repair all pass through ONE
LLM family (Perplexity). A model favours ideas it generated itself, and a
"simulated panel debate" run by one model is one entity arguing with
itself. This client gives the lab an independent judge: ChatGPT through
the same Hyperbrowser stealth edge already proven for Perplexity
(live-verified 2026-11-13: session created, cookie login accepted, real
answer captured in ~316s wall).

Contract (mirrors HyperbrowserClient / PerplexityClient):

* ``await ask(prompt, mode=..., system=..., purpose=...)`` -> LLMResponse
* one client = one cookie file (``account.name = "chatgpt-01"``)
* one Hyperbrowser session per client, created lazily, reused across asks
  (each ask navigates to the ChatGPT home page = a fresh conversation)

Live-verified mechanics (probe, 2026-11-13 — do not "simplify" these):

* the long-lived ``__Secure-next-auth.session-token`` / ``oai-sc`` cookies
  carry the login; short-lived edge cookies (``__oailb``, ``__cflb``, ~10
  minute life) are DROPPED — an export works for months, not minutes;
* the composer is a React-controlled contenteditable: synthetic
  ``insert_text`` does NOT update its state (the send button stays
  disabled) — real keystrokes with a per-char delay are required;
* frontend selectors change between deploys; the answer is extracted from
  the page TEXT after the last ``You said:`` marker (DOM role attributes
  are a fallback, never the primary).

Every ask consumes one Hyperbrowser session slot while the client is
open — on the one-token lab the judge and the primary edge share the
slot, so sessions are created/used one after another (the pool's key
lock still serializes them safely).
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentics.config import Settings
from agentics.llm.perplexity_client import LLMResponse

log = logging.getLogger("llm.chatgpt")

_HOME = "https://chatgpt.com/"
_COMPOSER_SELECTORS = ("#prompt-textarea", "[contenteditable='true']", "textarea")
_SEND_BUTTON = '[data-testid="send-button"]'
_USER_BLOCK = '[data-message-author-role="user"]'
_ASSISTANT_BLOCK = '[data-message-author-role="assistant"]'

_LOGIN_MARKERS = re.compile(r"\blog in\b|\bsign up\b|create account|get started", re.I)
# UI chrome that follows (or precedes) the answer text on the page.
_NOISE_MARKERS = (
    "Is this conversation helpful", "ChatGPT can make mistakes",
    "ChatGPT is AI and can make mistakes", "Regenerate", "Copy", "Share",
    "Latest response", "Response complete", "Voice", "Listen",
)


class ChatGPTError(RuntimeError):
    """Base class for ChatGPT cloud-session failures."""


class ChatGPTAuthError(ChatGPTError):
    """The injected session is no longer authenticated."""


class ChatGPTUIError(ChatGPTError):
    """Page loaded but the composer/answer could not be driven."""


def load_cookies(path: str | Path) -> tuple[list[dict], dict]:
    """Cookie-Editor export -> Playwright cookies for chatgpt.com only.

    Drops cookies for other domains and cookies already expired (the
    short-lived edge tokens); maps Cookie-Editor's lowercase sameSite
    values to Playwright's. Returns (cookies, stats) — stats is bounded,
    non-secret bookkeeping for logs.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "cookies" in raw:
        raw = raw["cookies"]
    now = time.time()
    stats = {"total": len(raw), "kept": 0, "dropped_other_domain": 0,
             "dropped_expired": 0}
    same_map = {"strict": "Strict", "lax": "Lax", "no_restriction": "None",
                "unspecified": "Lax"}
    out: list[dict] = []
    for c in raw:
        domain = (c.get("domain") or "").lower()
        if "chatgpt.com" not in domain:
            stats["dropped_other_domain"] += 1
            continue
        exp = c.get("expirationDate")
        if exp is not None:
            try:
                exp = float(exp)
            except (TypeError, ValueError):
                exp = None
            if exp is not None and exp <= now:
                stats["dropped_expired"] += 1
                continue
        entry = {
            "name": c["name"],
            "value": c["value"],
            "domain": domain,
            "path": c.get("path", "/"),
            "secure": bool(c.get("secure", True)),
            "httpOnly": bool(c.get("httpOnly", False)),
            "sameSite": same_map.get((c.get("sameSite") or "lax").lower(), "Lax"),
        }
        if exp is not None:
            entry["expires"] = exp
        out.append(entry)
    stats["kept"] = len(out)
    return out, stats


def compose_prompt(prompt: str, system: str | None) -> str:
    """The ChatGPT web UI has no system role — prepend it to the prompt."""
    text = (system or "").strip()
    return f"{text}\n\n{prompt}" if text else prompt


def extract_answer(full_body: str, prompt: str) -> str:
    """The assistant's answer from the page TEXT (deploy-proof).

    Takes everything after the LAST ``You said:`` marker, then after the
    ``ChatGPT said:`` label that follows our prompt echo (live-verified
    page shape); falls back to stripping the echoed prompt itself. Trailing
    UI chrome is cut. Returns '' when no real answer is found.
    """
    if not full_body or "You said:" not in full_body:
        return ""
    tail = full_body.rsplit("You said:", 1)[1].strip()
    if "ChatGPT said:" in tail:
        tail = tail.rsplit("ChatGPT said:", 1)[1].strip()
    else:
        stripped = (prompt or "").strip()
        if stripped and tail.startswith(stripped):
            tail = tail[len(stripped):].strip()
        elif stripped and tail.startswith(stripped[:40]):
            tail = tail[len(stripped[:40]):].strip()
    tail = re.sub(r"^ChatGPT said:\s*", "", tail).strip()
    cuts = [tail.find(m) for m in _NOISE_MARKERS]
    cuts = [i for i in cuts if i > 0]
    if cuts:
        tail = tail[: min(cuts)].strip()
    return tail if len(tail) > 20 else ""


class ChatGPTClient:
    """ChatGPT access via a Hyperbrowser cloud browser session."""

    def __init__(self, cookie_path: str | Path, settings: Settings,
                 answer_wait_s: int = 150, api_key: str | None = None) -> None:
        self.cookie_path = Path(cookie_path)
        self.settings = settings
        self.answer_wait_s = answer_wait_s
        self.api_key = (
            api_key
            or getattr(settings, "hyperbrowser_api_key", "")
            or (getattr(settings, "hyperbrowser_api_keys", None) or [""])[0]
        )
        # pool bookkeeping shape: session_host marks clients by .account.name;
        # the judge client is managed standalone, so account stays None.
        self.account = None
        self._session_id: str | None = None
        self._pw: Any = None
        self._browser: Any = None
        self._ctx: Any = None
        self._page: Any = None
        self._hb: Any = None
        self._ask_lock = asyncio.Lock()   # one UI conversation at a time

    # ------------------------------------------------------------- session
    @property
    def account_name(self) -> str:
        return "chatgpt-01"

    def _session_params(self) -> dict:
        return {
            "stealth": True,
            "proxy": {
                "active": True,
                "country": getattr(self.settings, "hyperbrowser_proxy_country", "US"),
            },
            "timeout": getattr(self.settings, "hyperbrowser_timeout", 900),
        }

    async def _sweep_orphan_sessions(self) -> int:
        """Stop active sessions still counted against this token (a killed
        process otherwise blocks the single slot until server timeout)."""
        try:
            listing = await asyncio.wait_for(
                asyncio.to_thread(self._hb.sessions.list, {"status": "active"}),
                timeout=60)
        except Exception:  # noqa: BLE001
            return 0
        stopped = 0
        for s in (getattr(listing, "sessions", None) or []):
            sid = getattr(s, "id", None)
            if not sid or sid == self._session_id:
                continue
            try:
                await asyncio.to_thread(self._hb.sessions.stop, sid)
                stopped += 1
            except Exception:  # noqa: BLE001
                pass
        return stopped

    async def _ensure_session(self) -> None:
        if self._page is not None:
            return
        from hyperbrowser import Hyperbrowser

        self._hb = Hyperbrowser(api_key=self.api_key)
        try:
            # the SDK create call has NO internal timeout — a contested
            # slot can hang it far past any sane wait; cap it hard
            sess = await asyncio.wait_for(
                asyncio.to_thread(self._hb.sessions.create,
                                  self._session_params()),
                timeout=120)
        except Exception as exc:
            if "maximum number of active sessions" not in str(exc).lower():
                raise
            await self._sweep_orphan_sessions()
            await asyncio.sleep(4)  # server releases the slot asynchronously
            sess = await asyncio.wait_for(
                asyncio.to_thread(self._hb.sessions.create,
                                  self._session_params()),
                timeout=120)
        self._session_id = sess.id
        log.info("chatgpt session %s created", sess.id)
        try:
            from playwright.async_api import async_playwright

            cookies, stats = load_cookies(self.cookie_path)
            if stats["kept"] < 1:
                raise ChatGPTAuthError(
                    f"no usable chatgpt.com cookies in {self.cookie_path}")
            log.info("chatgpt cookies: %s", json.dumps(stats))
            self._pw = await async_playwright().start()
            self._browser = await self._pw.chromium.connect_over_cdp(
                sess.ws_endpoint, timeout=60000)
            self._ctx = self._browser.contexts[0]
            await self._ctx.add_cookies(cookies)
            self._page = await self._ctx.new_page()
        except Exception:
            await self._teardown()
            raise

    async def _teardown(self) -> None:
        self._page = None
        if self._browser is not None:
            try:
                await self._browser.close()
            except Exception:  # noqa: BLE001
                pass
            self._browser = self._ctx = None
        if self._pw is not None:
            try:
                await self._pw.stop()
            except Exception:  # noqa: BLE001
                pass
            self._pw = None
        if self._hb is not None and self._session_id is not None:
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(self._hb.sessions.stop, self._session_id),
                    timeout=60)
            except Exception:  # noqa: BLE001
                pass
            self._hb = None
        self._session_id = None

    async def close(self) -> None:
        """Stop the cloud session (safe to call twice)."""
        await self._teardown()

    # ----------------------------------------------------------------- ui
    async def _body_text(self) -> str:
        try:
            # page.evaluate has NO internal timeout — a blocked page would
            # hang the ask forever; cap it hard.
            return await asyncio.wait_for(
                self._page.evaluate("() => document.body?.innerText || ''"),
                timeout=20)
        except Exception:  # noqa: BLE001
            return ""

    async def _find_composer(self, wait_s: int = 75):
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            for sel in _COMPOSER_SELECTORS:
                loc = self._page.locator(sel).first
                try:
                    if await loc.is_visible():
                        return loc
                except Exception:  # noqa: BLE001
                    continue
            await self._page.wait_for_timeout(3000)
        return None

    async def ask(self, prompt: str, mode: str = "concise",
                  system: str | None = None, purpose: str = "general",
                  ) -> LLMResponse:
        """One fresh ChatGPT conversation per ask; raises ChatGPTError on
        auth/UI failure (callers treat it like any provider failure).

        Hard ceiling: the whole ask is wrapped in ``answer_wait_s + 180``
        — the browser edge has several calls with no internal timeout, so
        an ask must ALWAYS return or raise, never hang a research cycle."""
        started = time.monotonic()
        async with self._ask_lock:
            try:
                return await asyncio.wait_for(
                    self._ask_unlocked(prompt, system, purpose),
                    timeout=self.answer_wait_s + 180)
            except asyncio.TimeoutError:
                raise ChatGPTUIError(
                    f"ask exceeded its hard ceiling "
                    f"({self.answer_wait_s + 180}s) — page edge stuck")
            finally:
                log.info("chatgpt ask done in %.0fs (purpose=%s)",
                         time.monotonic() - started, purpose)

    async def _ask_unlocked(self, prompt: str, system: str | None,
                            purpose: str) -> LLMResponse:
        # caller (ask) already holds _ask_lock — never lock again here
        t0 = time.monotonic()
        await self._ensure_session()
        full_prompt = compose_prompt(prompt, system)
        await self._page.goto(_HOME, wait_until="domcontentloaded",
                              timeout=90000)
        await self._page.wait_for_timeout(5000)
        body = await self._body_text()
        if not re.search(r"Recents|Projects|New chat", body):
            if _LOGIN_MARKERS.search(body):
                raise ChatGPTAuthError(
                    "chatgpt session is no longer authenticated")
            raise ChatGPTUIError("chatgpt home did not render the app shell")
        log.info("chatgpt: app shell live after %.0fs", time.monotonic() - t0)
        await self._page.keyboard.press("Escape")
        await self._page.wait_for_timeout(500)
        composer = await self._find_composer()
        if composer is None:
            raise ChatGPTUIError("composer not found within 75s")
        await composer.click()
        try:
            await composer.focus()
        except Exception:  # noqa: BLE001
            pass
        # React-controlled contenteditable. keyboard.type dispatches REAL
        # keystrokes but the ChatGPT page processes every keypress slowly
        # (live: a 1300-char prompt took 300s+ at 12ms/char delay) — try
        # the fast input-event path first (fill), and keep real keystrokes
        # as the reliable fallback only when the conversation fails to
        # start. Enter submits even while the send button renders disabled
        # (probe-verified live 2026-11-13).
        filled = False
        try:
            await composer.fill(full_prompt)
            filled = True
        except Exception:  # noqa: BLE001
            pass
        if not filled:
            await self._page.keyboard.type(full_prompt, delay=12)
        await self._page.wait_for_timeout(800)
        started_conv = await self._send_and_confirm()
        if not started_conv and filled:
            # fast path failed to register: fall back to real keystrokes
            try:
                await composer.fill("")
            except Exception:  # noqa: BLE001
                pass
            await self._page.keyboard.type(full_prompt, delay=6)
            await self._page.wait_for_timeout(800)
            started_conv = await self._send_and_confirm()
        if not started_conv:
            raise ChatGPTUIError("conversation never started after send")
        log.info("chatgpt: conversation started after %.0fs — waiting answer",
                 time.monotonic() - t0)
        answer = await self._wait_answer(full_prompt)
        return LLMResponse(
            answer=answer,
            citations=[],
            raw={"provider": "chatgpt", "purpose": purpose},
            elapsed_s=round(time.monotonic() - t0, 2),
        )

    async def _send_and_confirm(self, wait_s: int = 25) -> bool:
        """Click send (or Enter) and wait for the conversation to start.

        Conversation creation is observed as the URL moving to /c/<uuid>
        or the user message block rendering — DOM-independent signals.
        Returns True when the ask was accepted."""
        sent = False
        try:
            btn = self._page.locator(_SEND_BUTTON).first
            if await btn.is_enabled():
                await btn.click()
                sent = True
        except Exception:  # noqa: BLE001
            pass
        if not sent:
            await self._page.keyboard.press("Enter")
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            if "/c/" in (self._page.url or ""):
                return True
            try:
                if await self._page.locator(_USER_BLOCK).count() > 0:
                    return True
            except Exception:  # noqa: BLE001
                pass
            await self._page.wait_for_timeout(1500)
        return False

    async def _wait_answer(self, prompt: str) -> str:
        deadline = time.monotonic() + self.answer_wait_s
        stable, text_len = 0, 0
        answer = ""
        while time.monotonic() < deadline:
            await self._page.wait_for_timeout(4000)
            body = await self._body_text()
            answer = extract_answer(body, prompt)
            if not answer:
                blocks = self._page.locator(_ASSISTANT_BLOCK)
                try:
                    n = await blocks.count()
                    if n > 0:
                        answer = (await blocks.nth(n - 1).innerText()
                                  or "").strip()
                except Exception:  # noqa: BLE001
                    pass
            if answer and len(answer) == text_len:
                stable += 1
            else:
                stable, text_len = 0, len(answer)
            if stable >= 2 and text_len > 0:
                try:
                    if await self._page.locator(_SEND_BUTTON).first.is_visible():
                        break
                except Exception:  # noqa: BLE001
                    pass
        if not answer:
            raise ChatGPTUIError(
                f"no assistant answer within {self.answer_wait_s}s")
        return answer


def get_judge_client(settings: Settings) -> ChatGPTClient | None:
    """Build (and cache) the second-judge client when configured.

    Returns None when ``judge_second_provider`` is not "chatgpt"; raises
    FileNotFoundError when the cookie file is missing so callers can
    journal an actionable ``judge_second_unavailable`` event.
    """
    provider = str(getattr(settings, "judge_second_provider", "") or "").strip().lower()
    if provider != "chatgpt":
        return None
    path = str(getattr(settings, "chatgpt_cookie_path", "") or "").strip()
    if not path:
        raise FileNotFoundError(
            "judge_second_provider=chatgpt but chatgpt_cookie_path is empty")
    expanded = Path(path).expanduser()
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    if not expanded.exists():
        raise FileNotFoundError(f"chatgpt cookie file not found: {expanded}")
    cache = getattr(get_judge_client, "_cache", None)
    if cache is None:
        cache = get_judge_client._cache = {}  # type: ignore[attr-defined]
    key = str(expanded.resolve())
    client = cache.get(key)
    if client is None:
        # >= 2 Hyperbrowser tokens -> the judge gets ITS OWN token (pool[1])
        # so the two juries run truly in parallel; on the one-token lab the
        # judge shares the primary token and the tournament serializes the
        # two asks (one session slot).
        pool = list(getattr(settings, "hyperbrowser_api_keys", None) or [])
        api_key = pool[1] if len(pool) >= 2 else None
        client = ChatGPTClient(expanded, settings, api_key=api_key)
        cache[key] = client
    return client
