"""Probe: can the lab talk to ChatGPT through the same Hyperbrowser edge?

Research task (owner, 2026-11-13): evaluate ChatGPT as the SECOND, independent
model family — the dual-judge recommendation (#12) against single-LLM
correlated errors. This script is a feasibility probe, NOT an integration:

  1. create ONE Hyperbrowser stealth session (same edge as Perplexity runs)
  2. inject the owner's chatgpt.com cookie export (Cookie-Editor JSON)
  3. navigate to https://chatgpt.com/ and detect the auth state
  4. if the composer is live: send one small research prompt, capture the answer
  5. report a verdict + diagnostics (never prints cookie values)

Usage:
    .venv/bin/python scripts/probe_chatgpt.py <cookies.json> [--prompt "..."]

Secrets stay in the file passed on the CLI; nothing is written back.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentics.config import Settings  # noqa: E402

HOME = "https://chatgpt.com/"
COMPOSER = "#prompt-textarea"
SEND_BUTTON = '[data-testid="send-button"]'
ASSISTANT_MSG = '[data-message-author-role="assistant"]'


def _same_site(value: str | None) -> str:
    return {
        "strict": "Strict",
        "lax": "Lax",
        "no_restriction": "None",
        "unspecified": "Lax",
    }.get((value or "lax").lower(), "Lax")


def load_chatgpt_cookies(path: Path) -> tuple[list[dict], dict]:
    """Cookie-Editor export -> Playwright cookies for chatgpt.com only.
    Returns (cookies, stats). Expired cookies are dropped (the __oailb /
    __cflb edge tokens live ~10 minutes; Playwright rejects stale expires)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "cookies" in raw:
        raw = raw["cookies"]
    now = time.time()
    stats = {"total": len(raw), "kept": 0, "dropped_other_domain": 0, "dropped_expired": 0}
    out: list[dict] = []
    for c in raw:
        domain = (c.get("domain") or "").lower()
        if "chatgpt.com" not in domain:
            stats["dropped_other_domain"] += 1
            continue
        entry = {
            "name": c["name"],
            "value": c["value"],
            "domain": domain,
            "path": c.get("path", "/"),
            "secure": bool(c.get("secure", True)),
            "httpOnly": bool(c.get("httpOnly", False)),
            "sameSite": _same_site(c.get("sameSite")),
        }
        exp = c.get("expirationDate")
        if exp is not None:
            try:
                exp = float(exp)
            except (TypeError, ValueError):
                exp = None
            if exp is not None:
                if exp <= now:
                    stats["dropped_expired"] += 1
                    continue
                entry["expires"] = exp
        out.append(entry)
    stats["kept"] = len(out)
    return out, stats


async def run(cookies_path: Path, prompt: str, answer_wait_s: int) -> int:
    from hyperbrowser import Hyperbrowser
    from playwright.async_api import async_playwright

    settings = Settings()
    api_key = (
        getattr(settings, "hyperbrowser_api_key", "")
        or (getattr(settings, "hyperbrowser_api_keys", None) or [""])[0]
    )
    if not api_key:
        print("VERDICT: NO_TOKEN — no Hyperbrowser token in settings/.env")
        return 2

    cookies, stats = load_chatgpt_cookies(cookies_path)
    print(f"cookies: {json.dumps(stats)}")
    if not any(c["name"].startswith("__Secure-next-auth.session-token") for c in cookies):
        print("VERDICT: NO_SESSION_TOKEN — export lacks __Secure-next-auth.session-token")
        return 2

    hb = Hyperbrowser(api_key=api_key)
    params = {
        "stealth": True,
        "proxy": {"active": True,
                  "country": getattr(settings, "hyperbrowser_proxy_country", "US")},
        "timeout": getattr(settings, "hyperbrowser_timeout", 900),
    }
    print("creating Hyperbrowser session …")
    sess = await asyncio.to_thread(hb.sessions.create, params)
    print(f"session: {sess.id}")
    pw = browser = None
    try:
        pw = await async_playwright().start()
        browser = await pw.chromium.connect_over_cdp(sess.ws_endpoint, timeout=60000)
        ctx = browser.contexts[0]
        await ctx.add_cookies(cookies)
        page = await ctx.new_page()

        t0 = time.monotonic()
        await page.goto(HOME, wait_until="domcontentloaded", timeout=90000)
        await page.wait_for_timeout(5000)

        diag = await page.evaluate(
            "() => ({url: location.href, title: document.title || '',"
            " body: (document.body?.innerText || '').slice(0, 600)})")
        print(f"landed: url={diag['url'][:110]}")
        print(f"        title={diag['title'][:80]}")

        # Auth proof: the sidebar only renders the account's real chat
        # history when the session-token cookie was accepted. Composer
        # selectors lag the login state — ChatGPT hydrates the UI lazily.
        body = diag.get("body", "")
        authed = bool(re.search(r"Recents|Projects|New chat", body)) \
            and not re.search(r"log in|sign up|Get started|Create account", body, re.I)
        if not authed:
            print(f"VERDICT: NOT_AUTHENTICATED body[:400]={body[:400]!r}")
            return 3
        print("auth state: LOGGED IN (sidebar history present)")

        composer = None
        deadline = time.monotonic() + 75
        while time.monotonic() < deadline and composer is None:
            for sel in (COMPOSER, "[contenteditable='true']", "textarea"):
                loc = page.locator(sel).first
                try:
                    if await loc.is_visible():
                        composer = loc
                        break
                except Exception:
                    continue
            if composer is None:
                await page.wait_for_timeout(3000)
        if composer is None:
            print("VERDICT: AUTHED_BUT_COMPOSER_MISSING — logged in, composer not found in 75s")
            print(f"body[:600]={body[:600]!r}")
            return 3
        print(f"composer live after {time.monotonic() - t0:.0f}s")

        # React-controlled contenteditable: synthetic insert_text does NOT
        # update the editor state (send button stays disabled) — type real
        # keystrokes instead. Escape first: dismiss any upsell popover that
        # could swallow the click/Enter.
        await page.keyboard.press("Escape")
        await page.wait_for_timeout(500)
        await composer.click()
        try:
            await composer.focus()
        except Exception:
            pass
        await page.keyboard.type(prompt, delay=12)
        await page.wait_for_timeout(800)
        typed = ""
        try:
            typed = (await composer.inner_text() or "").strip()
        except Exception:
            pass
        print(f"composer holds {len(typed)} chars of prompt {len(prompt)}")
        sent = False
        try:
            btn = page.locator(SEND_BUTTON).first
            if await btn.is_enabled():
                await btn.click()
                sent = True
        except Exception:
            pass
        if not sent:
            await page.keyboard.press("Enter")
        print("prompt sent, waiting for assistant reply …")

        # Conversation creation is most reliably observed as the URL moving
        # to /c/<uuid>; fall back to the user message block.
        user_deadline = time.monotonic() + 25
        user_seen = False
        while time.monotonic() < user_deadline:
            url_now = page.url or ""
            if "/c/" in url_now:
                user_seen = True
                print(f"conversation started: {url_now[:90]}")
                break
            try:
                if await page.locator('[data-message-author-role="user"]').count() > 0:
                    user_seen = True
                    break
            except Exception:
                pass
            await page.wait_for_timeout(1500)
        if not user_seen:
            diag2 = await page.evaluate(
                "() => ({body: (document.body?.innerText || '').slice(0, 500)})")
            print("VERDICT: SEND_FAILED — conversation never started")
            print(f"body[:500]={diag2['body'][:500]!r}")
            return 4

        deadline = time.monotonic() + answer_wait_s
        stable, text_len = 0, 0
        answer = ""
        while time.monotonic() < deadline:
            await page.wait_for_timeout(4000)
            # Primary: text-slice after the LAST "You said:" marker — immune
            # to frontend DOM changes. Fallback: role blocks / .markdown.
            full = ""
            try:
                full = await page.evaluate(
                    "() => document.body?.innerText || ''")
            except Exception:
                pass
            if full:
                parts = full.rsplit("You said:", 1)
                tail = parts[1].strip() if len(parts) == 2 else ""
                # drop our own prompt echo at the start of the tail
                if tail.startswith(prompt[:60]):
                    tail = tail[len(prompt):].strip()
                # strip trailing UI noise after the answer
                for marker in ("Regenerate", "Copy", "Share", "ChatGPT can make mistakes"):
                    idx = tail.find(marker)
                    if idx > 0:
                        tail = tail[:idx].strip()
                if len(tail) > 40:  # a real answer, not just UI chrome
                    answer = tail
            if not answer:
                blocks = page.locator(ASSISTANT_MSG)
                n = await blocks.count()
                if n > 0:
                    answer = (await blocks.nth(n - 1).innerText() or "").strip()
            if answer and len(answer) == text_len:
                stable += 1
            else:
                stable, text_len = 0, len(answer)
            if stable >= 2 and text_len > 0:
                try:
                    if await page.locator(SEND_BUTTON).first.is_visible():
                        break
                except Exception:
                    pass
        if not answer:
            diag3 = await page.evaluate(
                "() => ({url: location.href,"
                " body: (document.body?.innerText || '').slice(0, 2600)})")
            print("VERDICT: NO_ANSWER — nothing after 'You said:' in time")
            print(f"url={diag3['url'][:90]}")
            print(f"FULL_BODY_BEGIN\n{diag3['body']}\nFULL_BODY_END")
            return 4
        print(f"VERDICT: OK — answer_len={len(answer)}")
        print("---ANSWER-BEGIN---")
        print(answer[:2500])
        print("---ANSWER-END---")
        return 0
    finally:
        try:
            if browser is not None:
                await browser.close()
        except Exception:
            pass
        try:
            if pw is not None:
                await pw.stop()
        except Exception:
            pass
        try:
            await asyncio.to_thread(hb.sessions.stop, sess.id)
            print("session stopped (token slot freed)")
        except Exception:
            print("WARN: session stop failed — sweep with scripts/hb_session_cleanup.py")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cookies", type=Path)
    ap.add_argument("--prompt", default=(
        "In one short paragraph: name the sorting algorithm used by Python's "
        "built-in sorted(), and state one measurable weakness of that algorithm "
        "on nearly-sorted integer arrays. Be precise and factual."))
    ap.add_argument("--answer-wait", type=int, default=180)
    args = ap.parse_args()
    started = datetime.now(timezone.utc)
    rc = asyncio.run(run(args.cookies, args.prompt, args.answer_wait))
    print(f"rc={rc} wall={(datetime.now(timezone.utc) - started).total_seconds():.0f}s")
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
