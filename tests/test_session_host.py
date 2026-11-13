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
    # Unit tests must not depend on the machine's runtime backend choice:
    # .env / lab.yaml may say "hyperbrowser" (datacenter deployments), which
    # makes _lease() wrap accounts in HyperbrowserClient instead of
    # PerplexityClient. Pin the raw-socket backend these tests assert on.
    s.browser_backend = "local"
    return s                                  # default: perplexity-only mode


def _hybrid_settings(tmp_path: Path) -> Settings:
    s = _settings(tmp_path)
    s.llm_provider = "qwen"                   # hybrid: research->pplx, code->qwen
    return s


def _resp(text: str = "ok") -> LLMResponse:
    return LLMResponse(answer=text, citations=["https://arxiv.org/a"], elapsed_s=0.01)


# ------------------------------------------------------------- routing table
def test_purpose_table_hybrid():
    assert PURPOSE_PROVIDER["research"] == "perplexity"
    for p in ("plan", "implement", "review", "distill", "general"):
        assert PURPOSE_PROVIDER[p] == "qwen"


def test_perplexity_only_mode_routes_everything_to_pplx(tmp_path):
    host = BrowserSessionHost.from_settings(_settings(tmp_path))
    assert host.qwen_enabled is False
    for purpose in ("research", "plan", "implement", "review", "distill", "general"):
        assert host.provider_for(purpose) == "perplexity"
    prov, client = host._lease("implement")
    assert prov == "perplexity" and isinstance(client, PerplexityClient)
    prov, client = host._lease("plan")
    assert prov == "perplexity" and isinstance(client, PerplexityClient)


def test_perplexity_only_never_falls_back_to_qwen(tmp_path):
    """Only qwen accounts exist + perplexity-only mode -> no Qwen, hard error."""
    s = _settings(tmp_path)
    for f in s.accounts_dir.glob("account-*.json"):
        f.unlink()
    host = BrowserSessionHost.from_settings(s)
    assert host._lease("research") == (None, None)
    with pytest.raises(RuntimeError, match="perplexity-only"):
        asyncio.run(host.ask("x", purpose="research"))


def test_hybrid_mode_routes_by_purpose(tmp_path):
    host = BrowserSessionHost.from_settings(_hybrid_settings(tmp_path))
    assert host.qwen_enabled is True
    assert host.provider_for("research") == "perplexity"
    for p in ("plan", "implement", "review"):
        assert host.provider_for(p) == "qwen"
    prov, client = host._lease("research")
    assert prov == "perplexity" and isinstance(client, PerplexityClient)
    prov, client = host._lease("implement")
    assert prov == "qwen" and isinstance(client, QwenClient)


def test_len_respects_mode(tmp_path):
    assert len(BrowserSessionHost.from_settings(_settings(tmp_path))) == 2      # pplx-only: 2
    assert len(BrowserSessionHost.from_settings(_hybrid_settings(tmp_path))) == 2  # max(2,1)
    s = _hybrid_settings(tmp_path)
    for f in s.accounts_dir.glob("account-*.json"):
        f.unlink()
    assert len(BrowserSessionHost.from_settings(s)) == 1                        # hybrid: qwen only


# ------------------------------------------------------------- leasing
def test_lease_routes_by_purpose(tmp_path):
    host = BrowserSessionHost.from_settings(_hybrid_settings(tmp_path))
    prov, client = host._lease("research")
    assert prov == "perplexity" and isinstance(client, PerplexityClient)
    prov, client = host._lease("implement")
    assert prov == "qwen" and isinstance(client, QwenClient)
    prov, client = host._lease("review")
    assert prov == "qwen" and isinstance(client, QwenClient)


def test_view_pins_identity(tmp_path):
    host = BrowserSessionHost.from_settings(_hybrid_settings(tmp_path))
    _prov, c0 = host.view(prefer_index=0)._lease("research")
    _prov, c1 = host.view(prefer_index=1)._lease("research")
    assert c0.account.name == "account-01"
    assert c1.account.name == "account-02"


def test_fallback_qwen_to_perplexity(tmp_path):
    """No qwen accounts at all -> code purposes degrade to Perplexity (hybrid)."""
    s = _hybrid_settings(tmp_path)
    for f in s.accounts_dir.glob("qwen-*.json"):
        f.unlink()
    host = BrowserSessionHost.from_settings(s)
    prov, client = host._lease("plan")
    assert prov == "perplexity" and isinstance(client, PerplexityClient)


def test_fallback_perplexity_to_qwen(tmp_path):
    """No pplx accounts -> research degrades to Qwen (hybrid only)."""
    s = _hybrid_settings(tmp_path)
    for f in s.accounts_dir.glob("account-*.json"):
        f.unlink()
    host = BrowserSessionHost.from_settings(s)
    prov, client = host._lease("research")
    assert prov == "qwen" and isinstance(client, QwenClient)


def test_lease_none_when_both_pools_empty(tmp_path):
    s = _hybrid_settings(tmp_path)
    for f in s.accounts_dir.glob("*.json"):
        f.unlink()
    host = BrowserSessionHost.from_settings(s)
    assert host._lease("research") == (None, None)
    with pytest.raises(RuntimeError, match="no usable browser-session account"):
        asyncio.run(host.ask("x", purpose="research"))


# ------------------------------------------------------------- ask dispatch
def test_ask_dispatch_and_purpose_passthrough(tmp_path, monkeypatch):
    s = _hybrid_settings(tmp_path)
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
    host = BrowserSessionHost.from_settings(_hybrid_settings(tmp_path))

    async def boom(self, query, **kwargs):
        raise RuntimeError("qwen chat HTTP 401: bad cookie")

    monkeypatch.setattr(QwenClient, "ask", boom)
    with pytest.raises(RuntimeError):
        asyncio.run(host.view(prefer_index=0).ask("code", purpose="implement"))
    h = host.pools["qwen"].health["qwen-01"]
    assert h.failures == 1 and h.in_cooldown()


def test_transient_non_auth_failure_does_not_cool_account(tmp_path, monkeypatch):
    # A websocket read timeout / dropped socket from the local backend is NOT
    # a dead cookie: the account must stay healthy so the next cycle (or the
    # client's own retry) can recover it — mirrors the Hyperbrowser
    # provider-outage policy for backends that raise plain RuntimeErrors.
    host = BrowserSessionHost.from_settings(_hybrid_settings(tmp_path))

    async def flaky(self, query, **kwargs):
        raise RuntimeError("websocket read timed out (op=60s)")

    monkeypatch.setattr(QwenClient, "ask", flaky)
    with pytest.raises(RuntimeError):
        asyncio.run(host.view(prefer_index=0).ask("code", purpose="implement"))
    h = host.pools["qwen"].health["qwen-01"]
    assert not h.in_cooldown()
    # the very next lease still gets the same (healthy) account
    _prov, client = host.view(prefer_index=0)._lease("implement")
    assert client.account.name == "qwen-01"


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


def test_status_lines_perplexity_only_marks_qwen_disabled(tmp_path):
    host = BrowserSessionHost.from_settings(_settings(tmp_path))
    lines = "\n".join(host.status_lines())
    assert "perplexity (research, citations)" in lines
    assert "qwen: DISABLED" in lines
    assert "account-01" in lines


def test_status_lines_hybrid_covers_both_pools(tmp_path):
    host = BrowserSessionHost.from_settings(_hybrid_settings(tmp_path))
    lines = "\n".join(host.status_lines())
    assert "perplexity (research, citations)" in lines
    assert "qwen (plan/implement/review" in lines
    assert "qwen-01" in lines


def test_client_cache_reuses_instances(tmp_path):
    host = BrowserSessionHost.from_settings(_hybrid_settings(tmp_path))
    view = host.view(prefer_index=0)
    _p, c1 = view._lease("research")
    _p, c2 = view._lease("research")
    assert c1 is c2                                # keeps socket/token state
