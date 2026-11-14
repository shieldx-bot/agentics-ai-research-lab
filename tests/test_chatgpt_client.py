"""ChatGPT second-judge client — offline tests for the pure mechanics.

Covers the deploy-proof pieces the live probe validated (2026-11-13):
cookie loading (domain filter, expiry drop, sameSite mapping), the
system-prompt composition (the ChatGPT UI has no system role), and the
page-TEXT answer extraction. No HTTP, no browser sessions.
"""

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402

from agentics.config import Settings  # noqa: E402
from agentics.llm.chatgpt_client import (  # noqa: E402
    ChatGPTClient,
    compose_prompt,
    extract_answer,
    get_judge_client,
    load_cookies,
)


def _write_cookies(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "cookies.json"
    p.write_text(json.dumps(rows), encoding="utf-8")
    return p


def test_load_cookies_filters_domain_drops_expired_maps_samesite(tmp_path):
    now = time.time()
    rows = [
        {"name": "__Secure-next-auth.session-token.0", "value": "tok-a",
         "domain": ".chatgpt.com", "path": "/", "secure": True,
         "httpOnly": True, "sameSite": "lax", "expirationDate": now + 86400},
        {"name": "__cflb", "value": "edge",
         "domain": "chatgpt.com", "path": "/", "secure": True,
         "httpOnly": True, "sameSite": "no_restriction",
         "expirationDate": now - 10},          # expired -> dropped
        {"name": "oai-sc", "value": "sc",
         "domain": ".chatgpt.com", "path": "/", "secure": True,
         "httpOnly": False, "sameSite": "no_restriction",
         "expirationDate": now + 86400},
        {"name": "unrelated", "value": "x",
         "domain": ".perplexity.ai", "path": "/", "secure": True,
         "httpOnly": False, "sameSite": "lax", "expirationDate": None},
    ]
    cookies, stats = load_cookies(_write_cookies(tmp_path, rows))
    names = [c["name"] for c in cookies]
    assert names == ["__Secure-next-auth.session-token.0", "oai-sc"]
    assert stats == {"total": 4, "kept": 2, "dropped_other_domain": 1,
                     "dropped_expired": 1}
    # no_restriction -> None (Playwright), lax -> Lax; expires preserved
    assert cookies[0]["sameSite"] == "Lax"
    assert cookies[0]["expires"] > now
    assert cookies[1]["sameSite"] == "None"


def test_load_cookies_accepts_cookie_editor_wrapper(tmp_path):
    rows = [{"name": "oai-did", "value": "d", "domain": ".chatgpt.com",
             "path": "/", "secure": True, "httpOnly": False,
             "sameSite": "lax", "expirationDate": None}]
    p = tmp_path / "wrapped.json"
    p.write_text(json.dumps({"cookies": rows}), encoding="utf-8")
    cookies, stats = load_cookies(p)
    assert stats["kept"] == 1 and cookies[0]["name"] == "oai-did"


def test_compose_prompt_prepends_system_and_handles_empty():
    assert compose_prompt("Q?", "Be honest.") == "Be honest.\n\nQ?"
    assert compose_prompt("Q?", None) == "Q?"
    assert compose_prompt("Q?", "   ") == "Q?"


PROMPT = "In one short paragraph: name the sorting algorithm used by Python's sorted()."
ANSWER = ("Python's built-in sorted() uses Timsort, a stable adaptive hybrid "
          "of merge sort and insertion sort. One measurable weakness: it "
          "needs O(n) auxiliary memory, unlike in-place heapsort.")


def test_extract_answer_strips_echo_label_and_noise():
    body = ("ChatGPT\nNew chat\nRecents\n"
            "Python sorted algorithm weakness\n"
            "You said:\n" + PROMPT + "\n"
            "ChatGPT said:\n" + ANSWER + "\n"
            "Is this conversation helpful so far?\n"
            "ChatGPT can make mistakes. Check important info.")
    assert extract_answer(body, PROMPT) == ANSWER


def test_extract_answer_handles_missing_marker_and_chrome_only():
    assert extract_answer("no marker here", PROMPT) == ""
    # tail shorter than the 20-char threshold -> not an answer
    assert extract_answer("You said:\n" + PROMPT + "\nok", PROMPT) == ""


def test_get_judge_client_none_off_and_raises_on_missing_file(tmp_path):
    assert get_judge_client(Settings()) is None  # provider off by default
    s = Settings()
    s.judge_second_provider = "chatgpt"
    s.chatgpt_cookie_path = str(tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError):
        get_judge_client(s)


def test_get_judge_client_builds_and_caches(tmp_path, monkeypatch):
    monkeypatch.chdir(PROJECT_ROOT)
    cookie = _write_cookies(tmp_path, [
        {"name": "__Secure-next-auth.session-token.0", "value": "tok",
         "domain": ".chatgpt.com", "path": "/", "secure": True,
         "httpOnly": True, "sameSite": "lax", "expirationDate": None}])
    s = Settings()
    s.judge_second_provider = "chatgpt"
    s.chatgpt_cookie_path = str(cookie)
    get_judge_client._cache = None  # reset module cache between tests
    try:
        c1 = get_judge_client(s)
        c2 = get_judge_client(s)
        assert isinstance(c1, ChatGPTClient)
        assert c1 is c2                      # cached per resolved path
        assert c1.answer_wait_s == 150       # default
    finally:
        get_judge_client._cache = None
