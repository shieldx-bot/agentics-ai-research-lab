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



# ------------------------------------------------------ failure classification
def test_challenge_detection_is_explicit():
    from agentics.llm.hyperbrowser_client import HyperbrowserClient
    assert HyperbrowserClient._challenge_text("Access Verification")
    assert HyperbrowserClient._challenge_text("Verify you are human")
    assert not HyperbrowserClient._challenge_text("Welcome to Perplexity")


def test_composer_timeout_on_challenge_opens_backend_circuit(tmp_path, monkeypatch):
    import agentics.llm.hyperbrowser_client as hb

    monkeypatch.setattr(hb, "_BACKEND_COOLDOWN_UNTIL", 0.0)
    monkeypatch.setattr(hb, "_BACKEND_COOLDOWN_REASON", "")
    client = HyperbrowserClient(
        CookieAccount(name="account-01", cookies=[]),
        _settings(tmp_path),
    )

    class MissingComposer:
        def locator(self, _selector):
            class Locator:
                # Playwright's Locator.first returns another Locator; the fake
                # mirrors that so the client's chained lookup works.
                @property
                def first(self):
                    return self

                async def wait_for(self, **_kwargs):
                    raise TimeoutError("no composer")
            return Locator()

    client._page = MissingComposer()

    async def diagnostics():
        return {
            "url": "https://www.perplexity.ai/",
            "title": "Access Verification",
            "body": "Verify you are human",
        }

    monkeypatch.setattr(client, "_page_diagnostics", diagnostics)
    with pytest.raises(hb.HyperbrowserBackendUnavailable):
        asyncio_run(client._wait_for_composer())
    assert hb._BACKEND_COOLDOWN_UNTIL > 0
    assert "verification" in hb._BACKEND_COOLDOWN_REASON.lower()


def test_backend_circuit_rejects_new_account_sessions(tmp_path, monkeypatch):
    import agentics.llm.hyperbrowser_client as hb

    monkeypatch.setattr(hb, "_BACKEND_COOLDOWN_UNTIL", __import__("time").monotonic() + 60)
    monkeypatch.setattr(hb, "_BACKEND_COOLDOWN_REASON", "verification page")
    client = HyperbrowserClient(
        CookieAccount(name="account-02", cookies=[]),
        _settings(tmp_path),
    )
    with pytest.raises(hb.HyperbrowserBackendUnavailable, match="circuit open"):
        asyncio_run(client.ask("test"))
    assert client._session_id is None



def test_provider_ui_outage_does_not_cool_down_cookie_account(tmp_path, monkeypatch):
    from agentics.llm.hyperbrowser_client import HyperbrowserBackendUnavailable

    host = BrowserSessionHost.from_settings(_settings(tmp_path))
    account = host.pools["perplexity"].accounts[0]
    calls = []

    class BrokenClient:
        def __init__(self, account):
            self.account = account

        async def ask(self, *_args, **_kwargs):
            raise HyperbrowserBackendUnavailable("verification page")

    pool = host.pools["perplexity"]
    monkeypatch.setattr(pool, "mark_failed", lambda *args, **kwargs: calls.append(args))
    monkeypatch.setattr(host, "_lease", lambda _purpose: ("perplexity", BrokenClient(account)))

    with pytest.raises(HyperbrowserBackendUnavailable):
        asyncio_run(host.ask("test", purpose="research"))
    assert calls == []
    assert not pool.health[account.name].in_cooldown()


# ------------------------------------------------------ composer self-recovery
def test_composer_recovers_after_one_reload(tmp_path, monkeypatch):
    """A slow SPA hydration must not open the shared circuit breaker: one
    in-place reload + re-wait heals the transient miss (2026-11-12 incident
    class: composer TimeoutError killed the whole swarm's availability)."""
    import agentics.llm.hyperbrowser_client as hb

    monkeypatch.setattr(hb, "_BACKEND_COOLDOWN_UNTIL", 0.0)
    monkeypatch.setattr(hb, "_BACKEND_COOLDOWN_REASON", "")
    client = HyperbrowserClient(
        CookieAccount(name="account-01", cookies=[]),
        _settings(tmp_path, hb_input_timeout=5),
    )

    class FlakyComposer:
        def __init__(self):
            self.calls = 0

        def locator(self, _selector):
            locator = self

            class Locator:
                @property
                def first(self):
                    return self

                async def wait_for(self, **_kwargs):
                    locator.calls += 1
                    if locator.calls == 1:
                        raise TimeoutError("not hydrated yet")

            return Locator()

    client._page = FlakyComposer()
    reloads = []

    async def fake_goto_home():
        reloads.append(1)

    monkeypatch.setattr(client, "_goto_home", fake_goto_home)
    box = asyncio_run(client._wait_for_composer())
    assert box is not None
    assert len(reloads) == 1
    assert hb._BACKEND_COOLDOWN_UNTIL == 0.0      # no circuit opened


def test_composer_failure_after_reload_still_classifies(tmp_path, monkeypatch):
    """Recovery is one-shot: a page that fails twice still reaches the
    predecessor's diagnostics (challenge -> backend circuit, sign-in -> auth
    error, generic miss -> UI error)."""
    import agentics.llm.hyperbrowser_client as hb

    monkeypatch.setattr(hb, "_BACKEND_COOLDOWN_UNTIL", 0.0)
    monkeypatch.setattr(hb, "_BACKEND_COOLDOWN_REASON", "")
    client = HyperbrowserClient(
        CookieAccount(name="account-01", cookies=[]),
        _settings(tmp_path, hb_input_timeout=5),
    )

    class MissingComposer:
        def locator(self, _selector):
            class Locator:
                @property
                def first(self):
                    return self

                async def wait_for(self, **_kwargs):
                    raise TimeoutError("no composer")
            return Locator()

    client._page = MissingComposer()

    async def diagnostics():
        return {"url": "u", "title": "t", "body": "some text"}

    async def failing_goto_home():
        raise TimeoutError("reload also failed")

    monkeypatch.setattr(client, "_page_diagnostics", diagnostics)
    monkeypatch.setattr(client, "_goto_home", failing_goto_home)
    with pytest.raises(hb.HyperbrowserUIError):
        asyncio_run(client._wait_for_composer())
    assert hb._BACKEND_COOLDOWN_UNTIL > 0


# ------------------------------------------------------------ empty-answer guard
def test_ask_raises_ui_error_on_silent_empty_answer(tmp_path, monkeypatch):
    """A submit that silently produces NO answer must raise a UI outage, not
    return LLMResponse(answer='') that crashes node_plan's json.loads."""
    import agentics.llm.hyperbrowser_client as hb

    monkeypatch.setattr(hb, "_BACKEND_COOLDOWN_UNTIL", 0.0)
    monkeypatch.setattr(hb, "_BACKEND_COOLDOWN_REASON", "")
    client = HyperbrowserClient(
        CookieAccount(name="account-01", cookies=[]),
        _settings(tmp_path),
    )

    async def ok(*_args, **_kwargs):
        return None

    async def empty_answer():
        return ("", [])

    monkeypatch.setattr(client, "_ensure_session", ok)
    monkeypatch.setattr(client, "_goto_home", ok)
    monkeypatch.setattr(client, "_submit", ok)
    monkeypatch.setattr(client, "_collect_answer", empty_answer)
    torn_down = []

    async def fake_teardown():
        torn_down.append(1)

    monkeypatch.setattr(client, "_teardown", fake_teardown)
    with pytest.raises(hb.HyperbrowserUIError, match="no answer text"):
        asyncio_run(client.ask("hello"))
    assert torn_down == [1]           # session rebuilt on the next ask


# ------------------------------------------------- orphan session sweep
class _FakeSessions:
    """sessions.create/list/stop fake — records every call."""

    def __init__(self, create_error=None, active_ids=(), stop_ok=True):
        self.create_error = create_error
        self.active_ids = list(active_ids)
        self.stop_ok = stop_ok
        self.created = 0
        self.stopped: list[str] = []

    def create(self, _params):
        self.created += 1
        if self.create_error is not None and self.created <= 1:
            raise self.create_error

        class _S:
            id = f"s{self.created}"
            ws_endpoint = "wss://cdp"
        return _S()

    def list(self, _params):
        class _R:
            sessions = [type("D", (), {"id": sid, "status": "active"})()
                        for sid in self.active_ids]
        return _R()

    def stop(self, sid):
        if not self.stop_ok:
            raise RuntimeError("already stopped")
        self.stopped.append(sid)


def _client_with(tmp_path, sessions, monkeypatch):
    """A client whose Hyperbrowser SDK constructor returns the fake."""
    import agentics.llm.hyperbrowser_client as hb
    import hyperbrowser as hb_sdk

    fake_hb = type("HB", (), {"sessions": sessions})()
    # _ensure_session does `from hyperbrowser import Hyperbrowser` at call
    # time, so the SDK module itself must be patched.
    monkeypatch.setattr(hb_sdk, "Hyperbrowser", lambda api_key=None: fake_hb)
    client = HyperbrowserClient(
        CookieAccount(name="account-01", cookies=[]),
        _settings(tmp_path, browser_backend="hyperbrowser",
                  hyperbrowser_api_key="hb_x"),
    )
    hb._ACTIVE_BY_KEY.pop(client.api_key, None)
    return client


def _stop_at_playwright(monkeypatch):
    """_ensure_session builds the CDP connection after create — stop there."""
    import playwright.async_api as papi

    class _NoPW:
        async def start(self):
            raise RuntimeError("stop at playwright")

    monkeypatch.setattr(papi, "async_playwright", lambda: _NoPW())


def test_ensure_session_sweeps_orphans_and_retries(tmp_path, monkeypatch):
    """The live incident: a killed process leaves its session running; the
    token's single slot is busy; create fails with 'Maximum number of
    active sessions (1) reached'. The client sweeps and retries ONCE."""
    _stop_at_playwright(monkeypatch)
    fake = _FakeSessions(
        create_error=RuntimeError(
            "Maximum number of active sessions (1) reached. Please close "
            "some sessions before creating new ones."),
        active_ids=("orphan-1", "orphan-2"),
    )
    client = _client_with(tmp_path, fake, monkeypatch)
    with pytest.raises(RuntimeError, match="stop at playwright"):
        asyncio_run(client._ensure_session())
    assert fake.created == 2                       # create -> sweep -> retry
    # the sweep stops exactly the two orphans; the later teardown also stops
    # the freshly created session, so only the first two entries matter
    assert fake.stopped[:2] == ["orphan-1", "orphan-2"]


def test_ensure_session_only_sweeps_on_the_session_cap_error(tmp_path, monkeypatch):
    fake = _FakeSessions(create_error=RuntimeError("quota exhausted"))
    client = _client_with(tmp_path, fake, monkeypatch)
    with pytest.raises(RuntimeError, match="quota"):
        asyncio_run(client._ensure_session())
    assert fake.created == 1 and fake.stopped == []


def test_ensure_session_raises_when_the_sweep_frees_nothing(tmp_path, monkeypatch):
    fake = _FakeSessions(
        create_error=RuntimeError("Maximum number of active sessions (1) reached."),
        active_ids=(),                                # server lists nothing
    )
    client = _client_with(tmp_path, fake, monkeypatch)
    with pytest.raises(RuntimeError, match="Maximum number of active sessions"):
        asyncio_run(client._ensure_session())
    assert fake.created == 1


def test_sweep_tolerates_a_failing_stop(tmp_path, monkeypatch):
    fake = _FakeSessions(stop_ok=False, active_ids=("ghost",))
    client = _client_with(tmp_path, fake, monkeypatch)
    assert asyncio_run(client._sweep_orphan_sessions()) == 0
