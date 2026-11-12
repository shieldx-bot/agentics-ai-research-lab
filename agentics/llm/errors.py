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
