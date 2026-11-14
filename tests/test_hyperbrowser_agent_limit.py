from types import SimpleNamespace

from agentics.orchestrator import _effective_agent_count


def test_hyperbrowser_backend_uses_one_agent_for_single_session():
    settings = SimpleNamespace(browser_backend="hyperbrowser", max_agents=10)
    assert _effective_agent_count(settings, available_accounts=10) == 1


def test_local_backend_respects_configured_agent_limit():
    settings = SimpleNamespace(browser_backend="local", max_agents=4)
    assert _effective_agent_count(settings, available_accounts=10) == 4


def test_agent_count_never_exceeds_available_accounts():
    settings = SimpleNamespace(browser_backend="local", max_agents=10)
    assert _effective_agent_count(settings, available_accounts=2) == 2
