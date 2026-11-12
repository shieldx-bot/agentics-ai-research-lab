"""BrowserSessionHost — dual-provider routing: research->Perplexity, code->Qwen."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from agentics.config import Settings
from agentics.llm.perplexity_client import CookieAccount, LLMResponse, PerplexityClient
from agentics.llm.qwen_client import QwenClient
from agentics.llm.session_host import PURPOSE_PROVIDER, BrowserSessionHost


def _settings(tmp_path: Path) -> Settings:
    acc = tmp_path / "accounts"
    acc.mkdir(exist_ok=True)
    # two perplexity accounts
    for i in (1, 2):
        (acc / f"account-0{i}.json").write_text(json.dumps([
            {"name": "session", "value": f"pplx-{i}", "domain": ".perplexity.ai"},
        ]))
    # one qwen account
    (acc / "qwen-01.json").write_text(json.dumps({
        "cookies": [{"name": "refresh_token", "value": "rt-1", "domain": ".qwen.ai"}],
        "refresh_token": "rt-1",
        "model": "qwen4.0-max-preview",
    }))
    s = Settings()
    s.accounts_dir = acc
    return s


def _resp(text: str = "ok") -> LLMResponse:
    return LLMResponse(answer=text, citations=["https://arxiv.org/a"], elapsed_s=0.01)


# ------------------------------------------------------------- routing table
def test_provider_for_purposes():
    assert PURPOSE_PROVIDER["research"] == "perplexity"
    for p in ("plan", "implement", "review", "distill", "general"):
        assert PURPOSE_PROVIDER[p] == "qwen"
    assert BrowserSessionHost.provider_for(object(), "unknown-thing") == "qwen"


def test_len_is_largest_pool(tmp_path):
    host = BrowserSessionHost.from_settings(_settings(tmp_path))
    assert len(host) == 2                      # 2 pplx vs 1 qwen
    assert len(host.pools["perplexity"]) == 2
    assert len(host.pools["qwen"]) == 1


# ------------------------------------------------------------- leasing
def test_lease_routes_by_purpose(tmp_path):
    host = BrowserSessionHost.from_settings(_settings(tmp_path))
    prov, client = host._lease("research")
    assert prov == "perplexity" and isinstance(client, PerplexityClient)
    prov, client = host._lease("implement")
    assert prov == "qwen" and isinstance(client, QwenClient)
    prov, client = host._lease("review")
    assert prov == "qwen" and isinstance(client, QwenClient)


def test_view_pins_identity(tmp_path):
    host = BrowserSessionHost.from_settings(_settings(tmp_path))
    _prov, c0 = host.view(prefer_index=0)._lease("research")
    _prov, c1 = host.view(prefer_index=1)._lease("research")
    assert c0.account.name == "account-01"
    assert c1.account.name == "account-02"


def test_fallback_qwen_to_perplexity(tmp_path):
    """No qwen accounts at all -> code purposes degrade to Perplexity."""
    s = _settings(tmp_path)
    for f in s.accounts_dir.glob("qwen-*.json"):
        f.unlink()
    host = BrowserSessionHost.from_settings(s)
    prov, client = host._lease("plan")
    assert prov == "perplexity" and isinstance(client, PerplexityClient)


def test_fallback_perplexity_to_qwen(tmp_path):
    """No pplx accounts -> research degrades to Qwen (citations may be lost)."""
    s = _settings(tmp_path)
    for f in s.accounts_dir.glob("account-*.json"):
        f.unlink()
    host = BrowserSessionHost.from_settings(s)
    prov, client = host._lease("research")
    assert prov == "qwen" and isinstance(client, QwenClient)


def test_lease_none_when_both_pools_empty(tmp_path):
    s = _settings(tmp_path)
    for f in s.accounts_dir.glob("*.json"):
        f.unlink()
    host = BrowserSessionHost.from_settings(s)
    assert host._lease("research") == (None, None)
    with pytest.raises(RuntimeError, match="no usable browser-session account"):
        asyncio.run(host.ask("x", purpose="research"))


# ------------------------------------------------------------- ask dispatch
def test_ask_dispatch_and_purpose_passthrough(tmp_path, monkeypatch):
    s = _settings(tmp_path)
    host = BrowserSessionHost.from_settings(s)
    seen: list[tuple[str, str]] = []              # (provider, purpose)

    async def fake_pplx(self, query, *, mode="academic", system=None,
                        frontend_uuid=None, purpose=None):
        seen.append(("perplexity", purpose or ""))
        return _resp("pplx-answer")

    async def fake_qwen(self, query, *, mode="concise", system=None, purpose=None):
        seen.append(("qwen", purpose or ""))
        return _resp("qwen-answer")

    monkeypatch.setattr(PerplexityClient, "ask", fake_pplx)
    monkeypatch.setattr(QwenClient, "ask", fake_qwen)

    r1 = asyncio.run(host.ask("survey papers", mode="academic", purpose="research"))
    r2 = asyncio.run(host.ask("write code", mode="concise", purpose="implement"))
    assert r1.answer == "pplx-answer" and r2.answer == "qwen-answer"
    assert seen == [("perplexity", "research"), ("qwen", "implement")]
    assert host.last_provider == "qwen"
    assert host.account.name == "qwen-01"


def test_ask_marks_failure_on_cooldown(tmp_path, monkeypatch):
    host = BrowserSessionHost.from_settings(_settings(tmp_path))

    async def boom(self, query, **kwargs):
        raise RuntimeError("qwen chat HTTP 401: bad cookie")

    monkeypatch.setattr(QwenClient, "ask", boom)
    with pytest.raises(RuntimeError):
        asyncio.run(host.view(prefer_index=0).ask("code", purpose="implement"))
    h = host.pools["qwen"].health["qwen-01"]
    assert h.failures == 1 and h.in_cooldown()


def test_ask_marks_ok_after_success(tmp_path, monkeypatch):
    host = BrowserSessionHost.from_settings(_settings(tmp_path))

    async def ok(self, query, **kwargs):
        return _resp()

    monkeypatch.setattr(PerplexityClient, "ask", ok)
    asyncio.run(host.view(prefer_index=0).ask("survey", purpose="research"))
    h = host.pools["perplexity"].health["account-01"]
    assert h.calls == 1 and h.ok


# ------------------------------------------------------------- failover API
def test_mark_failed_after_last_call(tmp_path, monkeypatch):
    host = BrowserSessionHost.from_settings(_settings(tmp_path))

    async def ok(self, query, **kwargs):
        return _resp()

    monkeypatch.setattr(PerplexityClient, "ask", ok)
    view = host.view(prefer_index=0)               # one agent's identity
    asyncio.run(view.ask("survey", purpose="research"))
    view.mark_failed("auth expired")               # orchestrator failover path
    h = host.pools["perplexity"].health["account-01"]
    assert h.in_cooldown() and "auth expired" in h.last_error
    # next lease skips the cooled-down account -> agent 0 falls to account-02
    _prov, client = host.view(prefer_index=0)._lease("research")
    assert client.account.name == "account-02"


def test_mark_failed_noop_without_call(tmp_path):
    host = BrowserSessionHost.from_settings(_settings(tmp_path))
    host.mark_failed("nothing served yet")          # must not raise


def test_status_lines_cover_both_pools(tmp_path):
    host = BrowserSessionHost.from_settings(_settings(tmp_path))
    lines = "\n".join(host.status_lines())
    assert "perplexity (research, citations)" in lines
    assert "qwen (plan/implement/review" in lines
    assert "account-01" in lines and "qwen-01" in lines


def test_client_cache_reuses_instances(tmp_path):
    host = BrowserSessionHost.from_settings(_settings(tmp_path))
    view = host.view(prefer_index=0)
    _p, c1 = view._lease("research")
    _p, c2 = view._lease("research")
    assert c1 is c2                                # keeps socket/token state
