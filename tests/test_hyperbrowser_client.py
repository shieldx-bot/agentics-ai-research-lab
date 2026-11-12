"""Hyperbrowser cloud-browser backend: cookie conversion + host wiring.

All tests are offline — no session is created (sessions only spawn inside
``ask()``); the routing and conversion layers are exercised directly.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentics.config import Settings
from agentics.llm.hyperbrowser_client import HyperbrowserClient, to_playwright_cookies
from agentics.llm.perplexity_client import CookieAccount
from agentics.llm.session_host import BrowserSessionHost


def _settings(tmp_path: Path, **overrides) -> Settings:
    acc = tmp_path / "accounts"
    acc.mkdir(exist_ok=True)
    (acc / "account-01.json").write_text(json.dumps([
        {"name": "session", "value": "pplx-1", "domain": ".perplexity.ai"},
    ]))
    s = Settings()
    s.accounts_dir = acc
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


# ------------------------------------------------------------ cookies
def test_cookie_conversion_full():
    raw = [{
        "name": "session-token", "value": "abc", "domain": "www.perplexity.ai",
        "path": "/", "secure": True, "httpOnly": True,
        "sameSite": "unspecified", "expirationDate": "1794573205.123",
    }]
    [c] = to_playwright_cookies(raw)
    assert c["expires"] == 1794573205.123 and isinstance(c["expires"], float)
    assert c["sameSite"] == "Lax" and c["secure"] is True and c["httpOnly"] is True


def test_cookie_conversion_session_cookie_and_foreign_domain_filtered():
    raw = [
        {"name": "tmp", "value": "1", "domain": ".perplexity.ai"},   # no expires
        {"name": "other", "value": "2", "domain": "example.com"},    # foreign
    ]
    out = to_playwright_cookies(raw)
    assert len(out) == 1 and "expires" not in out[0]


# ------------------------------------------------------------ host wiring
def test_host_uses_hyperbrowser_client_when_backend_set(tmp_path):
    s = _settings(tmp_path, browser_backend="hyperbrowser", hyperbrowser_api_key="hb_x")
    host = BrowserSessionHost.from_settings(s)
    acc = host.pools["perplexity"].accounts[0]
    client = host._client_for("perplexity", acc)
    assert isinstance(client, HyperbrowserClient)
    assert client.account.name == acc.name


def test_host_client_is_cached_per_account(tmp_path):
    s = _settings(tmp_path, browser_backend="hyperbrowser", hyperbrowser_api_key="hb_x")
    host = BrowserSessionHost.from_settings(s)
    acc = host.pools["perplexity"].accounts[0]
    assert host._client_for("perplexity", acc) is host._client_for("perplexity", acc)


def test_host_local_backend_keeps_websocket_client(tmp_path):
    s = _settings(tmp_path)                      # BROWSER_BACKEND default = local
    host = BrowserSessionHost.from_settings(s)
    acc = host.pools["perplexity"].accounts[0]
    from agentics.llm.perplexity_client import PerplexityClient
    assert isinstance(host._client_for("perplexity", acc), PerplexityClient)


def test_hyperbrowser_client_ask_raises_without_session(tmp_path, monkeypatch):
    """No API key -> SDK error surfaces; client tears down its half-built state."""
    s = _settings(tmp_path, browser_backend="hyperbrowser", hyperbrowser_api_key="")
    client = HyperbrowserClient(
        CookieAccount(name="account-01", cookies=[
            {"name": "session", "value": "pplx-1", "domain": ".perplexity.ai"}]),
        s,
    )
    with pytest.raises(Exception):
        asyncio_run(client.ask("hi"))
    assert client._page is None and client._session_id is None  # clean teardown


def asyncio_run(coro):
    import asyncio
    return asyncio.run(coro)


# ------------------------------------------------------------ ask interface
def test_ask_signature_matches_host_contract(tmp_path):
    """The host passes mode/system/purpose; the client must accept all of them."""
    import inspect
    sig = inspect.signature(HyperbrowserClient.ask)
    for name in ("mode", "system", "purpose", "frontend_uuid"):
        assert name in sig.parameters


# ------------------------------------------------------------ shutdown
def test_close_all_tears_down_cached_clients(tmp_path):
    """Orchestrator shutdown must stop cloud sessions (no leaked sessions)."""
    s = _settings(tmp_path, browser_backend="hyperbrowser", hyperbrowser_api_key="hb_x")
    host = BrowserSessionHost.from_settings(s)
    closed = []

    class FakeClient:
        async def close(self):
            closed.append(True)

    host._clients[("perplexity", "account-01")] = FakeClient()
    asyncio_run(host.close_all())
    assert closed == [True] and not host._clients


def test_close_all_tolerates_clients_without_close(tmp_path):
    s = _settings(tmp_path)
    host = BrowserSessionHost.from_settings(s)
    host._clients[("perplexity", "account-01")] = object()  # no close()
    asyncio_run(host.close_all())                            # must not raise
    assert not host._clients


# ------------------------------------------------------------ config flow
def test_lab_yaml_and_secrets_yaml_merge(tmp_path, monkeypatch):
    """The committed lab.yaml sets browser_backend; the gitignored
    secrets.yaml supplies the API key — no env vars needed."""
    import agentics.config as cfg

    cfgs = tmp_path / "configs"
    cfgs.mkdir()
    (cfgs / "lab.yaml").write_text(
        "browser_backend: hyperbrowser\nhyperbrowser_proxy_country: SG\n")
    (cfgs / "secrets.yaml").write_text('hyperbrowser_api_key: "hb_test"\n')
    monkeypatch.setattr(cfg, "PROJECT_ROOT", tmp_path)
    s = cfg.get_settings()
    assert s.browser_backend == "hyperbrowser"
    assert s.hyperbrowser_api_key == "hb_test"
    assert s.hyperbrowser_proxy_country == "SG"


def test_lab_yaml_without_secrets_still_loads(tmp_path, monkeypatch):
    import agentics.config as cfg

    cfgs = tmp_path / "configs"
    cfgs.mkdir()
    (cfgs / "lab.yaml").write_text("browser_backend: local\n")
    monkeypatch.setattr(cfg, "PROJECT_ROOT", tmp_path)
    s = cfg.get_settings()
    assert s.browser_backend == "local"
