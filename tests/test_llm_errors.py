"""LLM-failure classification — account cooldown policy shared by all backends."""
from __future__ import annotations

import pytest

from agentics.llm.errors import is_auth_failure, is_llm_failure


def _hyperbrowser_exc(name: str, msg: str) -> Exception:
    from agentics.llm.hyperbrowser_client import (
        HyperbrowserAuthError,
        HyperbrowserBackendUnavailable,
        HyperbrowserError,
        HyperbrowserUIError,
    )
    return {
        "HyperbrowserError": HyperbrowserError(msg),
        "HyperbrowserAuthError": HyperbrowserAuthError(msg),
        "HyperbrowserBackendUnavailable": HyperbrowserBackendUnavailable(msg),
        "HyperbrowserUIError": HyperbrowserUIError(msg),
    }[name]


@pytest.mark.parametrize("exc", [
    RuntimeError("qwen chat HTTP 401: bad cookie"),
    RuntimeError("perplexity session-token cookie missing"),
    RuntimeError("account login required (sign in page rendered)"),
    RuntimeError("invalid api key for provider"),
])
def test_plain_runtime_auth_failures_are_auth(exc):
    assert is_auth_failure(exc) is True


@pytest.mark.parametrize("exc", [
    RuntimeError("websocket read timed out (op=60s)"),
    RuntimeError("connection dropped mid-stream"),
    RuntimeError("no usable browser-session account for purpose 'plan'"),
    TimeoutError("composer wait exceeded"),
    # the two false positives the orchestrator's old heuristic produced
    ValueError("4 attempts exhausted before response"),
    RuntimeError("cycle finished in 403ms"),
])
def test_transient_non_auth_failures_are_not_auth(exc):
    assert is_auth_failure(exc) is False


def test_autherror_subclasses_are_auth():
    class PerplexityAuthError(Exception):
        pass

    assert is_auth_failure(PerplexityAuthError("expired")) is True


def test_hyperbrowser_auth_error_is_auth_without_message_markers():
    exc = _hyperbrowser_exc("HyperbrowserAuthError", "totally opaque message")
    assert is_auth_failure(exc) is True


def test_hyperbrowser_outage_and_ui_errors_are_not_auth():
    for name in ("HyperbrowserBackendUnavailable", "HyperbrowserUIError"):
        assert is_auth_failure(_hyperbrowser_exc(name, "provider down")) is False


def test_is_llm_failure_spans_backends():
    assert is_llm_failure(_hyperbrowser_exc("HyperbrowserError", "x")) is True
    assert is_llm_failure(RuntimeError(
        "no usable browser-session account for purpose 'plan'")) is True
    assert is_llm_failure(RuntimeError("github api 403 rate limited")) is False


# ------------------------------------------------------- infra vs provider
def test_is_infra_failure_matches_session_transport_classes():
    """The three transport classes from the live 2026-11-13 storm: a session
    that died mid-ask (playwright target-closed family), a rebuild rejected
    by the per-token cap, and a fresh session whose CDP endpoint 404'd."""
    from agentics.llm.errors import is_infra_failure

    assert is_infra_failure(RuntimeError(
        "Page.evaluate: Target page, context or browser has been closed"))
    assert is_infra_failure(RuntimeError(
        "Maximum number of active sessions (1) reached. Please close some "
        "sessions before creating new ones."))
    assert is_infra_failure(RuntimeError(
        'BrowserType.connect_over_cdp: WebSocket error ... 404 Not Found '
        '{"code":"not_found","message":"session not found"}'))


def test_is_infra_failure_matches_playwright_target_closed_type():
    from agentics.llm.errors import is_infra_failure

    class TargetClosedError(Exception):
        pass

    assert is_infra_failure(TargetClosedError("ws disconnected code=1006"))


def test_is_infra_failure_rejects_model_auth_and_timeout_failures():
    """The exemption stays scoped: answer/UI outages, plain read timeouts
    and auth failures still feed the P20 fail-streak."""
    from agentics.llm.errors import is_infra_failure

    assert not is_infra_failure(RuntimeError(
        "Perplexity produced no answer text after 120s"))
    assert not is_infra_failure(RuntimeError("websocket read timed out (op=60s)"))
    assert not is_infra_failure(RuntimeError(
        "account-03: session-token cookie missing after load"))
