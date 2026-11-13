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
