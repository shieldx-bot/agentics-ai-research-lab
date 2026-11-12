"""Qwen client — offline tests (pure helpers + account loading + payload)."""

import json
import time
from pathlib import Path

import pytest

from agentics.llm.qwen_client import (
    QwenAccount, QwenClient, build_chat_payload, jwt_exp, load_qwen_account,
    parse_sse_frames, token_expired, waf_blocked,
)


# ---------------------------------------------------------------- token math
def _jwt(payload: dict) -> str:
    import base64

    def enc(o):
        raw = json.dumps(o).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    return f"{enc({'alg': 'HS256'})}.{enc(payload)}.sig"


def test_jwt_exp_and_expiry_margin():
    future = _jwt({"exp": time.time() + 3600})
    past = _jwt({"exp": time.time() - 10})
    assert jwt_exp(future) == pytest.approx(time.time() + 3600, rel=0.01)
    assert token_expired(past) is True
    assert token_expired(future) is False
    # margin: token expiring within 120s counts as expired
    soon = _jwt({"exp": time.time() + 60})
    assert token_expired(soon) is True
    # unparsable token is treated as expired
    assert token_expired("garbage") is True


# ---------------------------------------------------------------- payload
def test_build_chat_payload_shape():
    p = build_chat_payload("hello", "qwen4.0-max-preview", "SYS", auto_search=True)
    assert p["stream"] is True and p["version"] == "2.1"
    assert p["incremental_output"] is True and p["chat_id"] is None
    assert p["model"] == "qwen4.0-max-preview"
    msg = p["messages"][0]
    assert msg["role"] == "user" and msg["chat_type"] == "t2t"
    assert msg["models"] == ["qwen4.0-max-preview"]
    assert msg["feature_config"]["auto_search"] is True
    assert msg["content"].startswith("SYS")          # system prompt prepended
    assert msg["sub_chat_type"] == "t2t"


def test_build_payload_search_off_by_default():
    p = build_chat_payload("hi", "m", None, auto_search=False)
    assert p["messages"][0]["feature_config"]["auto_search"] is False


# ---------------------------------------------------------------- SSE parsing
def test_parse_sse_frames_phase_protocol():
    body = "\n".join([
        json.dumps({"data": {"phase": "thinking", "content": "hmm"}}),
        json.dumps({"data": {"phase": "answer", "content": "PONG"}}),
        json.dumps({"data": {"phase": "answer", "content": "-42"}}),
        json.dumps({"data": {"phase": "finish", "content": ""}}),
    ])
    answer, citations = parse_sse_frames(body)
    assert answer == "PONG-42"                       # thinking phase excluded
    assert citations == []


def test_parse_sse_frames_citations_and_noise():
    body = "\n".join([
        "data: " + json.dumps({"data": {"phase": "answer", "content": "see ",
                                        "url": "https://example.com/a"}}),
        "data: [junk, not-an-object]",             # junk frame ignored
        json.dumps({"choices": [{"delta": {"content": "docs"}}]}),
        ": keep-alive comment",
    ])
    answer, citations = parse_sse_frames(body)
    assert answer == "see docs"
    assert citations == ["https://example.com/a"]


# ---------------------------------------------------------------- WAF
def test_waf_blocked_detection():
    assert waf_blocked('{"ret":["FAIL_SYS_USER_VALIDATE","RGV587_ERROR::SM::x"]}')
    assert not waf_blocked('{"data":{"phase":"answer","content":"ok"}}')


# ---------------------------------------------------------------- accounts
def test_load_qwen_account(tmp_path: Path):
    acc_file = tmp_path / "qwen-account-01.json"
    acc_file.write_text(json.dumps({
        "cookies": [
            {"name": "cna", "value": "v1", "domain": ".qwen.ai"},
            {"name": "acw_tc", "value": "v2", "domain": "chat.qwen.ai"},
            {"name": "session", "value": "nope", "domain": "perplexity.ai"},  # filtered out
        ],
        "refresh_token": "rt-123",
        "model": "qwen3.8-max",
    }))
    acc = load_qwen_account(acc_file)
    assert acc.name == "qwen-account-01"
    assert acc.cookies == {"cna": "v1", "acw_tc": "v2"}
    assert acc.refresh_token == "rt-123"
    assert acc.model == "qwen3.8-max"
    # refresh_token cookie fallback works too
    acc2_file = tmp_path / "qwen-account-02.json"
    acc2_file.write_text(json.dumps({
        "cookies": [{"name": "refresh_token", "value": "rt-cookie", "domain": ".qwen.ai"}],
    }))
    assert load_qwen_account(acc2_file).refresh_token == "rt-cookie"


def test_client_headers_carry_cookie_and_token():
    acc = QwenAccount(name="t", cookies={"cna": "v1"}, access_token="tok")
    client = QwenClient(acc)
    h = client._headers("tok")
    assert h["cookie"] == "cna=v1"
    assert h["authorization"] == "Bearer tok"
    assert h["origin"] == "https://chat.qwen.ai"
    assert "x-request-id" in h
    assert "authorization" not in client._headers(None)


# ---------------------------------------------------------------- pool wiring
def test_pool_loads_qwen_accounts(tmp_path: Path):
    from agentics.config import Settings
    from agentics.llm.pool import AccountPool

    acc_file = tmp_path / "qwen-account-01.json"
    acc_file.write_text(json.dumps({
        "cookies": [{"name": "cna", "value": "v1", "domain": ".qwen.ai"}],
        "refresh_token": "rt",
    }))
    (tmp_path / "qwen-account-02.json").write_text(json.dumps({
        "cookies": [{"name": "cna", "value": "v2", "domain": ".qwen.ai"}],
    }))
    s = Settings(accounts_dir=tmp_path, llm_provider="qwen")
    pool = AccountPool(s)
    assert len(pool) == 2
    client = pool.client_for(pool.lease(prefer_index=0))
    assert type(client).__name__ == "QwenClient"
    assert client.account.refresh_token == "rt"
