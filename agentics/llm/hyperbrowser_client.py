"""Hyperbrowser cloud-browser client — Perplexity through a real remote UI.

Why it exists
-------------
Perplexity blocks raw WebSocket access from datacenter IPs (verified live
2026-11-12: ``wss://www.perplexity.ai/socket.io`` → HTTP 403). The cookie
WebSocket client (:mod:`agentics.llm.perplexity_client`) only works from
residential networks. This client keeps the same ``ask()`` contract but runs
the browser on Hyperbrowser's cloud (stealth session + residential proxy),
so the lab can operate from a bare VPS.

Contract (mirrors PerplexityClient):

* one client = one cookie account (``.account`` attribute for pool marking)
* ``await ask(query, mode=..., system=..., purpose=...)`` -> :class:`LLMResponse`

Lifecycle: one Hyperbrowser session per client, created lazily on the first
``ask`` and reused for subsequent asks (each ask starts a fresh thread by
navigating to the Perplexity home page). ``close()`` stops the session —
call it on shutdown; sessions also die by their own timeout.

Token pool (P18): sessions are registered PER TOKEN (``_ACTIVE_BY_KEY``).
Each Hyperbrowser token owns exactly one session slot; clients on the same
token queue and evict each other, clients on different tokens run in
parallel. Give the swarm K tokens via ``HB_API_KEYS=tk1,tk2,...`` and run K
agents (the orchestrator sizes the pool for you). A single ``HB_API_KEY``
keeps the old free-plan behavior (one session, agents queue).
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
from agentics.llm.perplexity_client import CookieAccount, LLMResponse

log = logging.getLogger("llm.hyperbrowser")

# Per-token session registries (P18 token pool): each Hyperbrowser token
# owns ONE session slot. Clients sharing a token queue behind that token's
# lock and evict each other (free-plan semantics on a shared token);
# clients on DIFFERENT tokens run fully parallel — this is what scales the
# swarm from one agent to the whole token pool.
_KEY_LOCKS: dict[str, asyncio.Lock] = {}
_ACTIVE_BY_KEY: dict[str, "HyperbrowserClient"] = {}


def _key_lock(api_key: str) -> asyncio.Lock:
    """One lock per Hyperbrowser token (created lazily, process-wide)."""
    lock = _KEY_LOCKS.get(api_key)
    if lock is None:
        lock = asyncio.Lock()
        _KEY_LOCKS[api_key] = lock
    return lock

_HOME = "https://www.perplexity.ai"
_ACADEMIC_PREFIX = "[Prefer peer-reviewed academic sources: arXiv, IEEE, ACM, Springer.] "
_CITE_RE = re.compile(r"^https?://")

# A challenge or frontend outage is shared by every cookie account. Keep a
# process-wide circuit breaker so agents do not rotate accounts and create
# more cloud sessions against the same unavailable page.
_BACKEND_COOLDOWN_UNTIL = 0.0
_BACKEND_COOLDOWN_REASON = ""


class HyperbrowserError(RuntimeError):
    """Base class for browser-session failures."""


class HyperbrowserAuthError(HyperbrowserError):
    """The injected Perplexity session is no longer authenticated."""


class HyperbrowserBackendUnavailable(HyperbrowserError):
    """The frontend is challenged/unavailable; retry after the circuit opens."""


class HyperbrowserUIError(HyperbrowserError):
    """The page loaded but its expected composer could not be found."""


def _open_backend_circuit(reason: str, cooldown_s: int = 300) -> None:
    global _BACKEND_COOLDOWN_UNTIL, _BACKEND_COOLDOWN_REASON
    _BACKEND_COOLDOWN_UNTIL = max(
        _BACKEND_COOLDOWN_UNTIL, time.monotonic() + cooldown_s
    )
    _BACKEND_COOLDOWN_REASON = reason[:180]


def _session_died(exc: Exception) -> bool:
    """Playwright's target-closed family: the CDP link to the cloud browser
    dropped mid-ask (session duration cap, network blip). The stale handle
    still looks alive on our side, so the ask must rebuild the session."""
    text = str(exc)
    if "Target page, context or browser has been closed" in text:
        return True
    return type(exc).__name__ in ("TargetClosedError", "TargetClosed")


def _raw_cookie_export(account: CookieAccount) -> list | dict:
    """Prefer the raw cookie-editor export (CDP accepts it verbatim); fall
    back to the normalized name->value mapping when no file is available."""
    if account.path:
        try:
            data = json.loads(Path(account.path).read_text(encoding="utf-8"))
            if isinstance(data, dict) and "cookies" in data:
                data = data["cookies"]
            if isinstance(data, list):
                return data
        except (OSError, ValueError):
            pass
    return account.cookies


def _account_cookies(account: CookieAccount) -> list[dict]:
    return to_playwright_cookies(_raw_cookie_export(account))


def _expires_to_epoch(value) -> float | None:
    """expirationDate may be an epoch number or an ISO-8601 string."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt.timestamp() if dt.tzinfo else dt.replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


def to_playwright_cookies(cookies: list | dict) -> list[dict]:
    """Convert a cookie export (list of dicts) or an already-normalized
    ``CookieAccount.cookies`` mapping (name -> value) to Playwright format."""
    if isinstance(cookies, dict):
        raw = [{"name": k, "value": v, "domain": ".perplexity.ai",
                "path": "/", "secure": True}
               for k, v in cookies.items()]
    else:
        raw = cookies
    out = []
    for c in raw:
        if c.get("domain") and "perplexity.ai" not in c["domain"]:
            continue
        same = c.get("sameSite")
        if same not in ("Strict", "Lax", "None"):
            same = "Lax"
        entry = {
            "name": c["name"],
            "value": c["value"],
            "domain": c.get("domain") or ".perplexity.ai",
            "path": c.get("path", "/"),
            "secure": bool(c.get("secure", False)),
            "httpOnly": bool(c.get("httpOnly", False)),
            "sameSite": same,
        }
        expires = _expires_to_epoch(c.get("expirationDate"))
        if expires is not None:
            entry["expires"] = expires
        out.append(entry)
    return out


class HyperbrowserClient:
    """Perplexity access via a Hyperbrowser cloud browser session."""

    def __init__(
        self,
        account: CookieAccount,
        settings: Settings,
        answer_wait_s: int = 120,
        api_key: str | None = None,
    ) -> None:
        self.account = account
        self.settings = settings
        self.answer_wait_s = answer_wait_s
        # This client's Hyperbrowser token. The session host pins agent i to
        # token i of the pool (settings.hyperbrowser_api_keys); without a
        # pool it falls back to the legacy single HB_API_KEY.
        self.api_key = (
            api_key
            or getattr(settings, "hyperbrowser_api_key", "")
            or (getattr(settings, "hyperbrowser_api_keys", None) or [""])[0]
        )
        self._session_id: str | None = None
        self._pw: Any = None
        self._browser: Any = None
        self._ctx: Any = None
        self._page: Any = None
        self._hb: Any = None
        self._ask_lock = asyncio.Lock()      # one UI conversation at a time

    # ------------------------------------------------------------- session
    def _session_params(self) -> dict:
        return {
            "stealth": True,
            "proxy": {
                "active": True,
                "country": getattr(self.settings, "hyperbrowser_proxy_country", "US"),
            },
            "timeout": getattr(self.settings, "hyperbrowser_timeout", 900),
        }

    async def _ensure_session(self) -> None:
        """Caller must hold this token's lock (see _key_lock, non-reentrant)."""
        if self._page is not None:
            return
        from hyperbrowser import Hyperbrowser

        self._hb = Hyperbrowser(api_key=self.api_key)
        holder = _ACTIVE_BY_KEY.get(self.api_key)
        if holder is not None and holder is not self:
            # one session per token: evict the current holder of THIS token
            # (holders on other tokens keep their sessions — that is the pool)
            await holder._teardown()
        sess = None
        for attempt in (1, 2):
            try:
                sess = await asyncio.to_thread(
                    self._hb.sessions.create, self._session_params())
                break
            except Exception as exc:
                if "Maximum number of active sessions" not in str(exc) or attempt == 2:
                    raise
                # A cap-sized token (1 session) hits 429 when a previous ask's
                # session is still propagating its stop() on HB's side, or a
                # crashed run leaked one. Sessions on THIS key can only be
                # ours — one session per token by design (_ACTIVE_BY_KEY) —
                # so stopping them is self-healing, never disruptive.
                log.warning(
                    "hyperbrowser session cap hit for this token; "
                    "stopping leaked session(s) and retrying once")
                await self._stop_leaked_sessions()
        if sess is None:
            raise RuntimeError("hyperbrowser session creation failed")
        self._session_id = sess.id
        log.info("hyperbrowser session %s created for %s", sess.id, self.account.name)
        try:
            from playwright.async_api import async_playwright

            self._pw = await async_playwright().start()
            self._browser = await self._pw.chromium.connect_over_cdp(
                sess.ws_endpoint, timeout=60000)
            self._ctx = self._browser.contexts[0]
            await self._ctx.add_cookies(_account_cookies(self.account))
            self._page = await self._ctx.new_page()
        except Exception:
            await self._teardown()
            raise
        _ACTIVE_BY_KEY[self.api_key] = self

    async def _stop_leaked_sessions(self) -> None:
        """Stop every session still active on THIS token, then give HB's
        active-session counter a few seconds to reflect the stops."""
        try:
            resp = await asyncio.to_thread(self._hb.sessions.list, None)
            dump = resp.model_dump() if hasattr(resp, "model_dump") else {}
            for s in dump.get("sessions", []) or []:
                status = str(s.get("status") or "").lower()
                if status in ("active", "running", "state_active"):
                    try:
                        await asyncio.to_thread(self._hb.sessions.stop, s["id"])
                    except Exception:  # noqa: BLE001
                        pass
        except Exception:  # noqa: BLE001
            pass
        await asyncio.sleep(5)

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
                await asyncio.to_thread(self._hb.sessions.stop, self._session_id)
            except Exception:  # noqa: BLE001
                pass
            self._hb = None
        if _ACTIVE_BY_KEY.get(self.api_key) is self:
            _ACTIVE_BY_KEY.pop(self.api_key, None)
        self._session_id = None

    async def close(self) -> None:
        """Stop the cloud session (safe to call twice)."""
        await self._teardown()

    # ----------------------------------------------------------------- ui
    async def _page_diagnostics(self) -> dict[str, str]:
        """Return bounded, non-secret page metadata for actionable failures."""
        try:
            data = await self._page.evaluate("""() => ({
                url: location.href,
                title: document.title || "",
                body: (document.body?.innerText || "").slice(0, 1200)
            })""")
        except Exception as exc:  # noqa: BLE001
            return {"url": "", "title": "", "body": f"page inspection failed: {type(exc).__name__}"}
        return {k: str(data.get(k, ""))[:1200] for k in ("url", "title", "body")}

    @staticmethod
    def _challenge_text(value: str) -> bool:
        return bool(re.search(
            r"access verification|verify you are human|captcha|"
            r"checking your browser|security challenge|unusual traffic",
            value,
            re.IGNORECASE,
        ))

    async def _goto_home(self) -> None:
        await self._page.goto(_HOME, timeout=60000, wait_until="domcontentloaded")
        await asyncio.sleep(6)
        diag = await self._page_diagnostics()
        combined = " ".join(diag.values())
        if self._challenge_text(combined):
            reason = f"Perplexity verification page detected ({diag['title']!r}, {diag['url']})"
            _open_backend_circuit(reason, cooldown_s=600)
            raise HyperbrowserBackendUnavailable(
                reason + "; complete any required verification in a normal user browser "
                "and retry later. Automated challenge solving is not supported."
            )
        names = [c["name"] for c in await self._ctx.cookies(_HOME)]
        if not any("session-token" in n for n in names):
            raise HyperbrowserAuthError(
                f"{self.account.name}: session-token cookie missing after load "
                f"(page={diag['title']!r}, url={diag['url']!r}; account session may be expired)"
            )

    async def _wait_for_composer(self):
        # Keep the selector broad, but wait longer for the SPA to hydrate.
        selectors = (
            "textarea:visible, [contenteditable='true']:visible, "
            "[role='textbox']:visible"
        )
        box = self._page.locator(selectors).first
        timeout_s = max(5, int(getattr(self.settings, "hb_input_timeout", 45) or 45))
        try:
            await box.wait_for(state="visible", timeout=timeout_s * 1000)
            return box
        except Exception as first_exc:  # noqa: BLE001
            # One in-place recovery BEFORE escalating: a slow SPA hydration or
            # a late frontend deploy often renders the composer a beat after
            # the first wait gives up. Escalating straight to the shared
            # circuit breaker used to stall EVERY agent for minutes on what a
            # simple reload heals — the 25s-timeout failure class from the
            # 2026-11-12 incident, now self-healing instead of circuit-opening.
            try:
                await self._goto_home()
                await box.wait_for(state="visible", timeout=timeout_s * 1000)
                log.info("composer recovered after one reload (%s)", self.account.name)
                return box
            except Exception:  # noqa: BLE001 — fall through to the diagnostics
                exc = first_exc
            diag = await self._page_diagnostics()
            combined = " ".join(diag.values())
            if self._challenge_text(combined):
                reason = f"Perplexity verification page detected ({diag['title']!r}, {diag['url']})"
                _open_backend_circuit(reason, cooldown_s=600)
                raise HyperbrowserBackendUnavailable(
                    reason + "; automated challenge solving is not supported"
                ) from exc
            # A visible sign-in state means this particular cookie is stale;
            # a generic missing composer is a frontend/UI outage, not bad auth.
            if re.search(r"sign\s?in|log\s?in", diag["body"][:500], re.IGNORECASE):
                raise HyperbrowserAuthError(
                    f"{self.account.name}: Perplexity rendered a sign-in page "
                    f"(title={diag['title']!r}, url={diag['url']!r})"
                ) from exc
            reason = (
                f"Perplexity composer unavailable after {timeout_s}s "
                f"(title={diag['title']!r}, url={diag['url']!r})"
            )
            _open_backend_circuit(reason, cooldown_s=180)
            raise HyperbrowserUIError(
                reason + "; inspect the page state before retrying"
            ) from exc

    async def _submit(self, text: str) -> None:
        box = await self._wait_for_composer()
        try:
            await box.click(timeout=4000)
        except Exception:  # noqa: BLE001
            # A click interception can be caused by a normal modal. Do not
            # attempt to dismiss or interact with security challenges.
            await self._page.keyboard.press("Escape")
            await asyncio.sleep(1)
            await box.focus()
        await self._page.keyboard.insert_text(text)   # Lexical editor
        await asyncio.sleep(1)
        try:
            await self._page.locator(
                "button[aria-label*='submit' i], button:has-text('Submit')"
            ).first.click(timeout=5000)
        except Exception:  # noqa: BLE001
            await box.press("Enter")

    _ANSWER_JS = """() => {
        const els = document.querySelectorAll('[data-testid="answer-text"], .prose');
        return els.length ? els[els.length - 1].innerText : '';
    }"""
    _CITES_JS = """() => {
        const out = [];
        for (const a of document.querySelectorAll('a[href]')) {
            const h = a.href;
            if (h.startsWith('http') && !h.includes('perplexity.ai')) out.push(h);
        }
        return out;
    }"""

    async def _collect_answer(self) -> tuple[str, list[str]]:
        """Poll until the answer text is stable, then grab citations."""
        deadline = time.monotonic() + self.answer_wait_s
        text, stable = "", 0
        while time.monotonic() < deadline:
            await asyncio.sleep(4)
            new = (await self._page.evaluate(self._ANSWER_JS) or "").strip()
            if new and new == text:
                stable += 1
                if stable >= 2:                      # unchanged for ~8 s: done
                    break
            else:
                stable = 0
            text = new
        cites = await self._page.evaluate(self._CITES_JS)
        uniq: list[str] = []
        for u in cites or []:
            if u not in uniq and _CITE_RE.match(u):
                uniq.append(u)
        return text.strip(), uniq[:12]

    # ---------------------------------------------------------------- ask
    async def ask(
        self,
        query: str,
        *,
        mode: str = "concise",
        system: str | None = None,
        purpose: str | None = None,
        frontend_uuid: str | None = None,
    ) -> LLMResponse:
        """Submit one query through the remote Perplexity UI.

        ``mode="academic"`` prepends an academic-sources hint (the web UI has
        no per-request focus parameter). ``purpose``/``frontend_uuid`` are
        accepted for interface parity and ignored.
        """
        started = time.monotonic()
        # One UI conversation at a time PER TOKEN: the token's session slot is
        # exclusive, so same-token asks queue behind each other (and may evict
        # the previous holder). Different tokens run in parallel — the pool.
        async with _key_lock(self.api_key):
            if time.monotonic() < _BACKEND_COOLDOWN_UNTIL:
                remaining = int(_BACKEND_COOLDOWN_UNTIL - time.monotonic())
                raise HyperbrowserBackendUnavailable(
                    f"Hyperbrowser/Perplexity frontend circuit open for {remaining}s: "
                    f"{_BACKEND_COOLDOWN_REASON}"
                )
            for attempt in (0, 1):
                try:
                    await self._ensure_session()
                    await self._goto_home()
                    prefix = _ACADEMIC_PREFIX if mode == "academic" else ""
                    if system:
                        prefix += "[Instructions] " + system.strip()[:800] + " "
                    await self._submit(prefix + query)
                    answer, citations = await self._collect_answer()
                    if not answer:
                        # A silent submit failure (Enter swallowed, answer markup
                        # moved) must not masquerade as an LLM answer: downstream
                        # nodes would parse "" into a JSON crash (node_plan) or a
                        # wasted cycle (node_verify). Raised as a UI outage so the
                        # session host fails over without cooling the account.
                        raise HyperbrowserUIError(
                            "Perplexity produced no answer text after "
                            f"{self.answer_wait_s}s (submit may have failed silently)"
                        )
                    break
                except Exception as exc:
                    await self._teardown()               # next ask rebuilds cleanly
                    # A dead CDP link (cloud session hit its duration cap, or a
                    # network blip) still LOOKS alive on our side — without the
                    # rebuild, every later node of the cycle would fail with
                    # "Target page ... closed" against the stale handle.
                    if attempt == 0 and _session_died(exc):
                        log.warning("hyperbrowser session died mid-ask (%s); "
                                    "rebuilding the cloud session once", str(exc)[:120])
                        continue
                    raise
        return LLMResponse(
            answer=answer,
            citations=citations,
            raw={"session_id": self._session_id, "backend": "hyperbrowser"},
            elapsed_s=round(time.monotonic() - started, 2),
        )
