"""Hyperbrowser token pool — one cloud session per token, N parallel agents.

P18: the free-plan single-session bottleneck is replaced by a per-token
pool. K tokens (HB_API_KEYS) scale the swarm to K truly parallel agents;
clients sharing one token keep the old single-session semantics (queue +
evict), clients on different tokens run fully parallel. All tests offline.
"""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402

from agentics.config import Settings  # noqa: E402
from agentics.llm import hyperbrowser_client as hb_mod  # noqa: E402
from agentics.llm.hyperbrowser_client import HyperbrowserClient  # noqa: E402
from agentics.llm.perplexity_client import CookieAccount  # noqa: E402
from agentics.orchestrator import _effective_agent_count  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_registries(monkeypatch):
    """Isolate the process-wide per-key registries per test."""
    monkeypatch.setattr(hb_mod, "_KEY_LOCKS", {})
    monkeypatch.setattr(hb_mod, "_ACTIVE_BY_KEY", {})
    monkeypatch.delenv("HB_API_KEYS", raising=False)
    monkeypatch.delenv("HB_API_KEY", raising=False)
    yield


_FAKE_SETTINGS = SimpleNamespace(
    hyperbrowser_api_key="", hyperbrowser_api_keys=[],
    hyperbrowser_proxy_country="US", hyperbrowser_timeout=900,
    hb_input_timeout=45,
)


def _fake_sdk(monkeypatch):
    """Fake hyperbrowser SDK + playwright import so _ensure_session runs
    offline and records created session ids."""
    created: list[str] = []

    class FakeSessions:
        def create(self, params):
            created.append(params.get("proxy", {}).get("country", ""))
            return SimpleNamespace(id=f"s-{len(created)}", ws_endpoint="ws://fake")

        def stop(self, session_id):
            pass

    class FakeHyperbrowser:
        def __init__(self, api_key=None):
            self.api_key = api_key
            self.sessions = FakeSessions()

    class FakePage:
        pass

    class FakeCtx:
        async def add_cookies(self, cookies):
            pass

        async def new_page(self):
            return FakePage()

    class FakeBrowser:
        def __init__(self):
            self.contexts = [FakeCtx()]

    class FakeChromium:
        async def connect_over_cdp(self, endpoint, timeout=None):
            return FakeBrowser()

    class FakePW:
        chromium = FakeChromium()

        async def stop(self):
            pass

    class FakeFactory:
        async def start(self):
            return FakePW()

    monkeypatch.setitem(sys.modules, "hyperbrowser",
                        SimpleNamespace(Hyperbrowser=FakeHyperbrowser))
    fake_api = SimpleNamespace(async_playwright=lambda: FakeFactory())
    monkeypatch.setitem(sys.modules, "playwright", SimpleNamespace(async_api=fake_api))
    monkeypatch.setitem(sys.modules, "playwright.async_api", fake_api)
    return created


def _client(api_key):
    return HyperbrowserClient(
        CookieAccount(name=f"acc-{api_key}", cookies={}), _FAKE_SETTINGS,
        api_key=api_key)


class _Standin:
    """A fake active-session holder whose teardown is observable."""

    def __init__(self):
        self.torn = False

    async def _teardown(self):
        self.torn = True


# ------------------------------------------------------------ config
def test_hb_api_keys_env_comma_separated(monkeypatch):
    monkeypatch.setenv("HB_API_KEYS", "tk_a, tk_b ,,tk_c")
    assert Settings().hyperbrowser_api_keys == ["tk_a", "tk_b", "tk_c"]


def test_hb_api_keys_falls_back_to_single_key(monkeypatch):
    monkeypatch.setenv("HB_API_KEY", "tk_single")
    assert Settings().hyperbrowser_api_keys == ["tk_single"]


def test_hb_api_keys_empty_when_no_env(monkeypatch):
    assert Settings().hyperbrowser_api_keys == []


def test_secrets_yaml_list_override(tmp_path, monkeypatch):
    """configs/secrets.yaml wins: a YAML list replaces the env-derived pool."""
    monkeypatch.setenv("HB_API_KEYS", "tk_env")
    monkeypatch.setattr("agentics.config.PROJECT_ROOT", tmp_path)
    cfg = tmp_path / "configs"
    cfg.mkdir()
    (cfg / "secrets.yaml").write_text(
        "hyperbrowser_api_keys:\n  - tk_y1\n  - tk_y2\n", encoding="utf-8")
    s = __import__("agentics.config", fromlist=["load_yaml_overrides"]) \
        .load_yaml_overrides(Settings())
    assert s.hyperbrowser_api_keys == ["tk_y1", "tk_y2"]


# ------------------------------------------------------------ agent count
def _host_settings(tmp_path, keys):
    import json as _json

    acc = tmp_path / "accounts"
    acc.mkdir(exist_ok=True)
    (acc / "account-01.json").write_text(_json.dumps([
        {"name": "session", "value": "pplx-1", "domain": ".perplexity.ai"}]))
    s = Settings()
    s.accounts_dir = acc
    s.browser_backend = "hyperbrowser"
    s.hyperbrowser_api_key = ""
    s.hyperbrowser_api_keys = keys
    return s


def test_host_pins_agent_to_its_token(tmp_path):
    """Agent i runs on token i — that is the whole point of the pool."""
    from agentics.llm.session_host import BrowserSessionHost

    host = BrowserSessionHost.from_settings(_host_settings(tmp_path, ["tk0", "tk1", "tk2"]))
    acc = host.pools["perplexity"].accounts[0]
    for i, expected in enumerate(("tk0", "tk1", "tk2")):
        view = host.view(prefer_index=i)
        client = view._client_for("perplexity", acc)
        assert isinstance(client, HyperbrowserClient)
        assert client.api_key == expected
    # unpinned (standalone CLI) host falls back to the first token
    assert host._client_for("perplexity", acc).api_key == "tk0"


def test_host_pin_wraps_around_the_pool(tmp_path):
    """More agents than tokens: tokens are reused round-robin (same-token
    clients then queue, per free-plan semantics)."""
    from agentics.llm.session_host import BrowserSessionHost

    host = BrowserSessionHost.from_settings(_host_settings(tmp_path, ["tk0", "tk1"]))
    acc = host.pools["perplexity"].accounts[0]
    assert host.view(prefer_index=2)._client_for("perplexity", acc).api_key == "tk0"
    assert host.view(prefer_index=3)._client_for("perplexity", acc).api_key == "tk1"


def test_host_without_pool_falls_back_to_legacy_key(tmp_path):
    from agentics.llm.session_host import BrowserSessionHost

    host = BrowserSessionHost.from_settings(_host_settings(tmp_path, []))
    host.settings.hyperbrowser_api_key = "tk_legacy"
    acc = host.pools["perplexity"].accounts[0]
    client = host.view(prefer_index=1)._client_for("perplexity", acc)
    assert client.api_key == "tk_legacy"


# ------------------------------------------------------------ per-key locking
def test_count_scales_with_the_token_pool():
    settings = SimpleNamespace(browser_backend="hyperbrowser", max_agents=10,
                               hyperbrowser_api_keys=["a", "b", "c"])
    assert _effective_agent_count(settings, available_accounts=10) == 3


def test_count_capped_by_accounts_and_max_agents():
    settings = SimpleNamespace(browser_backend="hyperbrowser", max_agents=2,
                               hyperbrowser_api_keys=["a", "b", "c"])
    assert _effective_agent_count(settings, available_accounts=10) == 2
    settings2 = SimpleNamespace(browser_backend="hyperbrowser", max_agents=10,
                                hyperbrowser_api_keys=["a", "b", "c"])
    assert _effective_agent_count(settings2, available_accounts=2) == 2


def test_single_key_still_limits_to_one_session():
    settings = SimpleNamespace(browser_backend="hyperbrowser", max_agents=10,
                               hyperbrowser_api_keys=["a"])
    assert _effective_agent_count(settings, available_accounts=10) == 1


def test_legacy_settings_without_keys_attribute_still_limit_to_one():
    """Old deployments / SimpleNamespace fakes without the new attribute."""
    settings = SimpleNamespace(browser_backend="hyperbrowser", max_agents=10)
    assert _effective_agent_count(settings, available_accounts=10) == 1


def test_local_backend_ignores_the_key_pool():
    settings = SimpleNamespace(browser_backend="local", max_agents=4,
                               hyperbrowser_api_keys=["a", "b", "c"])
    assert _effective_agent_count(settings, available_accounts=10) == 4


# ------------------------------------------------------------ per-key locking
def test_key_locks_are_per_token():
    lock_a = hb_mod._key_lock("tk_a")
    lock_b = hb_mod._key_lock("tk_b")
    assert lock_a is not lock_b
    assert hb_mod._key_lock("tk_a") is lock_a          # stable per key
    with asyncio.Runner() as runner:
        async def check():
            async with lock_a:                          # tk_a held...
                acquired = await asyncio.wait_for(      # ...tk_b still free
                    lock_b.acquire(), timeout=0.1)
                lock_b.release()
                assert acquired
        runner.run(check())


def test_same_key_clients_share_one_lock():
    lock = hb_mod._key_lock("tk_a")
    with asyncio.Runner() as runner:
        async def check():
            async with lock:
                with pytest.raises(asyncio.TimeoutError):   # same token queues
                    await asyncio.wait_for(lock.acquire(), timeout=0.1)
        runner.run(check())


# ------------------------------------------------------------ per-key sessions
def test_different_key_holder_is_not_evicted(monkeypatch):
    """Two tokens = two live sessions; eviction only happens within a token."""
    _fake_sdk(monkeypatch)
    standin_a, standin_b = _Standin(), _Standin()
    hb_mod._ACTIVE_BY_KEY["tk_a"] = standin_a
    hb_mod._ACTIVE_BY_KEY["tk_b"] = standin_b
    b = _client("tk_b")
    with asyncio.Runner() as runner:
        runner.run(b._ensure_session())
    assert not standin_a.torn               # tk_a session untouched (PARALLEL)
    assert hb_mod._ACTIVE_BY_KEY["tk_b"] is b
    assert b._session_id == "s-1" and b._page is not None


def test_same_key_holder_is_evicted(monkeypatch):
    """One session per token: the new holder tears the old one down first."""
    _fake_sdk(monkeypatch)
    standin_a = _Standin()
    hb_mod._ACTIVE_BY_KEY["tk_a"] = standin_a
    c = _client("tk_a")
    with asyncio.Runner() as runner:
        runner.run(c._ensure_session())
    assert standin_a.torn                   # free-plan semantics within tk_a
    assert hb_mod._ACTIVE_BY_KEY["tk_a"] is c


def test_teardown_releases_only_its_own_key_slot(monkeypatch):
    _fake_sdk(monkeypatch)
    c = _client("tk_a")
    with asyncio.Runner() as runner:
        runner.run(c._ensure_session())
        assert hb_mod._ACTIVE_BY_KEY.get("tk_a") is c
        runner.run(c.close())
    assert "tk_a" not in hb_mod._ACTIVE_BY_KEY
