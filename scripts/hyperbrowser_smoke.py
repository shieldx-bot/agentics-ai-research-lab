"""Hyperbrowser cloud-browser effectiveness test for Agentics Perplexity accounts.

Flow per account:
  1. create a stealth Hyperbrowser session (residential proxy, US)
  2. connect Playwright over CDP
  3. inject the account's Perplexity cookies
  4. open https://www.perplexity.ai and detect login state
  5. if logged in, submit a real query via the UI and capture the answer
  6. stop the session

Usage:
  HB_API_KEY=... python hb_perplexity_test.py --accounts 1-2 [--query "..."]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path("/home/z/my-project/agentics-ai-research-lab")
sys.path.insert(0, str(REPO))

from hyperbrowser import Hyperbrowser  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

ACCOUNTS_DIR = REPO / "accounts"
OUT = Path("/home/z/my-project/download/hb_test")
OUT.mkdir(parents=True, exist_ok=True)


def load_accounts(indices: list[int]) -> list[tuple[str, list[dict]]]:
    out = []
    for i in indices:
        p = ACCOUNTS_DIR / f"account-{i:02d}.json"
        if not p.exists():
            continue
        raw = json.loads(p.read_text())
        out.append((p.stem, raw))
    return out


def to_playwright_cookies(raw: list[dict]) -> list[dict]:
    """Convert storage-format cookies to Playwright add_cookies format."""
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
            "domain": c["domain"],
            "path": c.get("path", "/"),
            "secure": bool(c.get("secure", False)),
            "httpOnly": bool(c.get("httpOnly", False)),
            "sameSite": same,
        }
        try:
            entry["expires"] = float(c["expirationDate"])
        except (KeyError, TypeError, ValueError):
            pass  # session cookie: no expires
        out.append(entry)
    return out


def detect_login(page) -> str:
    state = page.evaluate(
        """() => {
        const body = document.body ? document.body.innerText : '';
        const hasSignin = /sign\\s?in|log\\s?in/i.test(body.slice(0, 3000));
        const hasSessionCookie = document.cookie.includes('session-token');
        const textarea = !!document.querySelector('textarea, [contenteditable="true"]');
        const avatar = !!document.querySelector('img[alt*="avatar" i], [data-testid*="avatar" i]');
        return {hasSignin, hasSessionCookie, textarea, avatar,
                title: document.title, url: location.href};
    }"""
    )
    if state["hasSessionCookie"] and not state["hasSignin"]:
        return "logged_in", state
    if state["hasSignin"] and not state["hasSessionCookie"]:
        return "not_logged_in", state
    return "unknown", state


def submit_query(page, query: str, wait_s: int = 90) -> str:
    box = page.locator("textarea:visible, [contenteditable='true']:visible").first
    box.wait_for(state="visible", timeout=25000)
    box.click()
    page.keyboard.insert_text(query)  # Lexical editor: type() can stall
    time.sleep(1)
    try:
        page.locator("button[aria-label*='submit' i], button:has-text('Submit')").first.click(timeout=5000)
    except Exception:  # noqa: BLE001
        box.press("Enter")
    time.sleep(wait_s)
    return page.evaluate(
        """() => {
        const sels = ['[data-testid="answer-text"]', '.prose', '[class*="answer"]'];
        for (const s of sels) {
          const els = document.querySelectorAll(s);
          if (els.length) return els[els.length - 1].innerText.slice(0, 2000);
        }
        return document.body.innerText.slice(0, 2000);
    }"""
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--accounts", default="1", help="e.g. 1-3 or 1,4")
    ap.add_argument("--query", default="What is network function virtualization? Answer in 2 sentences with one citation.")
    ap.add_argument("--wait", type=int, default=75, help="seconds to wait for answer")
    args = ap.parse_args()

    idx: list[int] = []
    for part in args.accounts.split(","):
        if "-" in part:
            a, b = part.split("-")
            idx.extend(range(int(a), int(b) + 1))
        else:
            idx.append(int(part))

    hb = Hyperbrowser(api_key=os.environ["HB_API_KEY"])
    accounts = load_accounts(idx)
    print(f"testing {len(accounts)} account(s) via Hyperbrowser cloud browser")

    with sync_playwright() as pw:
        for name, raw in accounts:
            sess = None
            print(f"\n=== {name} ===")
            try:
                sess = hb.sessions.create({
                    "stealth": True,
                    "proxy": {"active": True, "country": "US"},
                })
                print(f"session {sess.id} ({sess.status})")
                browser = pw.chromium.connect_over_cdp(sess.ws_endpoint, timeout=60000)
                ctx = browser.contexts[0]
                ctx.add_cookies(to_playwright_cookies(raw))

                page = ctx.new_page()
                page.goto("https://www.perplexity.ai", timeout=60000, wait_until="domcontentloaded")
                time.sleep(8)
                names = [c["name"] for c in ctx.cookies("https://www.perplexity.ai")]
                has_session = any("session-token" in n for n in names)
                signin_visible = page.locator("a[href*='signup'], a[href*='signin'], button:has-text('Sign in')").count() > 0
                state_name = ("logged_in" if has_session else
                              "not_logged_in" if signin_visible else "unknown")
                print(f"login state: {state_name} | session-token={has_session} | signin_visible={signin_visible} | title={page.title()!r}")

                page.screenshot(path=str(OUT / f"{name}-home.png"))

                if state_name == "logged_in":
                    t0 = time.monotonic()
                    answer = submit_query(page, args.query, wait_s=args.wait)
                    print(f"round-trip latency: {time.monotonic() - t0:.0f}s")
                    page.screenshot(path=str(OUT / f"{name}-answer.png"))
                    print(f"answer captured ({len(answer)} chars):")
                    print(answer[:600])
                else:
                    print("-> skip query (not logged in)")
            except Exception as exc:  # noqa: BLE001
                print(f"ERROR: {type(exc).__name__}: {str(exc)[:300]}")
            finally:
                if sess is not None:
                    try:
                        hb.sessions.stop(sess.id)
                    except Exception:  # noqa: BLE001
                        pass
                    print("session stopped")


if __name__ == "__main__":
    main()
