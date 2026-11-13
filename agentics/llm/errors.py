"""LLM-failure classification shared across browser backends.

Two overlapping error systems exist on purpose:

* the Hyperbrowser backend raises its own hierarchy
  (``HyperbrowserError`` / ``HyperbrowserAuthError`` /
  ``HyperbrowserBackendUnavailable`` / ``HyperbrowserUIError``);
* the local cookie-WebSocket backend (``perplexity_client``) and the Qwen
  client raise plain ``RuntimeError`` subclasses / provider exceptions.

The pool must not cool down an account for transient non-auth failures
(a websocket read timeout is not a dead cookie — the 2026-11-13 incident
took a single-account lab down for 15 minutes exactly because of this).
Only :func:`is_auth_failure` failures deserve an account cooldown.
"""
from __future__ import annotations

import re

# HTTP status codes are matched with word boundaries so "403ms" or a port
# number cannot masquerade as an auth failure.
_HTTP_AUTH_STATUS = re.compile(r"\b(?:401|403)\b")

AUTH_MARKERS: tuple[str, ...] = (
    "unauthorized", "session-token", "sign in", "login",
    "log in", "bad cookie", "invalid api key", "api key",
)

_NO_ACCOUNT_MSG = "no usable browser-session account"


class LLMError(RuntimeError):
    """Marker base for classifier-based checks (backends may reuse it)."""


class LLMRefused(LLMError):
    """The ask never reached the provider — a refusal, not a failure.

    Three sources raise it today: the P20 budget gate is implicit (its
    refusals never call ``ask``), the session host has no usable account
    (empty pool / every account cooling down), and the Hyperbrowser
    frontend circuit is open. The caller already knows the request was
    never attempted, so the P20 fail-streak must NOT count it as a
    provider failure — otherwise the breaker punishes the agent for the
    breaker's own refusals and a dead window inflates into a dead hour
    (live 2026-11-13: llm_usage rows 7-12 recorded 6 "failures" of 0.4ms
    each while the frontend circuit was merely open)."""


def is_llm_refusal(exc: BaseException) -> bool:
    """True when the exception means 'the request never left the machine'.

    Refusals (``LLMRefused``, an open Hyperbrowser frontend circuit) are
    budget/telemetry events, NOT provider failures: they must not feed the
    P20 fail-streak. The type-name check keeps this import-light — the
    Hyperbrowser module is optional and errors.py must not import it at
    module scope."""
    if isinstance(exc, LLMRefused):
        return True
    return type(exc).__name__ == "HyperbrowserBackendUnavailable"


def _hyperbrowser_error_types() -> tuple[type[BaseException], ...]:
    try:
        from agentics.llm.hyperbrowser_client import HyperbrowserError
        return (HyperbrowserError,)
    except ImportError:  # pragma: no cover - optional backend
        return ()


def is_llm_failure(exc: BaseException) -> bool:
    """True when the exception means 'the LLM access path failed' — nodes
    may degrade the cycle instead of crashing."""
    if isinstance(exc, _hyperbrowser_error_types()):
        return True
    if type(exc).__name__.startswith("Hyperbrowser"):
        return True
    if isinstance(exc, LLMError):
        return True
    return _NO_ACCOUNT_MSG in str(exc)


def is_auth_failure(exc: BaseException) -> bool:
    """True when the failure is credential-specific (cookie/token dead) —
    only these deserve an account cooldown."""
    name = type(exc).__name__
    if name == "HyperbrowserAuthError" or name.endswith("AuthError"):
        return True
    msg = str(exc).lower()
    if _HTTP_AUTH_STATUS.search(msg):
        return True
    return any(marker in msg for marker in AUTH_MARKERS)
