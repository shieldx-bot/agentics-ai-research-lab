"""Static cascade routing (#2 of the advanced-tech roadmap, wave-3): when a
cookie-pool ask FAILS (a refusal = dry pool / open circuit, or a genuine
provider error), the purpose — if the owner allows it — gets ONE retry
against an independent frontier provider (ChatGPT) instead of the cycle
dying for the rest of the cooldown window.

Why (roadmap: "Router/cascade đa mô hình"): FrugalGPT-style cascades cut
cost, and here more importantly they cut LOST CYCLES — the 2026-11-13
live runs showed a dry pool freezing whole research branches for up to a
900s cooldown. A cascade converts "pool refused" into "answered by the
other model family", with hard honesty rules:

* verdict/jury purposes NEVER cascade (verify / review / repair / defend
  / tournament / ideation): the machine adjudication chain must keep its
  provenance, and the second jury must stay independent — a cascade
  answer entering the tournament would correlate the juries the dual-
  model design exists to de-correlate;
* cascade runs ONLY when explicitly configured (``LLM_CASCADE_PROVIDER``,
  default "" = off);
* one attempt per failed ask, never raises, and every attempt is
  journalled (``router_cascade``) so the owner can see which model
  actually answered;
* cascade asks do NOT consume the P20 pool budget (different provider
  account) and are NOT recorded in ``llm_usage`` — that ledger counts
  cookie-pool asks only, same rule as the local router.
"""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("llm.router")

# Verdict + jury purposes must never be answered by a cascade provider.
_NO_CASCADE = frozenset({
    "verify", "review", "repair", "defend", "tournament",
    "tournament_judge2", "ideation",
})

_DEFAULT_PURPOSES = "research,plan,implement"


def cascade_provider(settings, purpose: str) -> str:
    """The configured cascade provider ('' = off / purpose not allowed)."""
    provider = str(getattr(settings, "llm_cascade_provider", "") or
                   "").strip().lower()
    if not provider:
        return ""
    p = (purpose or "").strip().lower()
    if not p or p in _NO_CASCADE:
        return ""
    allowed = {
        s.strip().lower()
        for s in str(getattr(settings, "llm_cascade_purposes",
                             _DEFAULT_PURPOSES) or "").split(",")
        if s.strip()
    }
    return provider if p in allowed else ""


def get_cascade_client(settings):
    """The cascade client (a ChatGPTClient), or None — never raises.

    Shares the second-judge cloud session when the dual jury is already
    configured (one session serving both = fewer Hyperbrowser tokens
    burned); builds a standalone cached session otherwise."""
    provider = str(getattr(settings, "llm_cascade_provider", "") or
                   "").strip().lower()
    if provider != "chatgpt":
        return None
    from agentics.llm.chatgpt_client import ChatGPTClient, get_judge_client

    if str(getattr(settings, "judge_second_provider", "") or ""
           ).strip().lower() == "chatgpt":
        try:
            client = get_judge_client(settings)
            if client is not None:
                return client
        except Exception as exc:  # noqa: BLE001 — fall through to standalone
            log.warning("cascade: judge client unavailable (%s); "
                        "trying a standalone session", exc)
    path = str(getattr(settings, "chatgpt_cookie_path", "") or "").strip()
    if not path:
        log.warning("cascade provider chatgpt needs chatgpt_cookie_path")
        return None
    expanded = Path(path).expanduser()
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    if not expanded.exists():
        log.warning("cascade cookie file missing: %s", expanded)
        return None
    cache = getattr(get_cascade_client, "_cache", None)
    if cache is None:
        cache = get_cascade_client._cache = {}  # type: ignore[attr-defined]
    key = str(expanded.resolve())
    client = cache.get(key)
    if client is None:
        client = ChatGPTClient(expanded, settings)
        cache[key] = client
    return client


async def try_cascade(ctx, prompt: str, mode: str, purpose: str,
                      system: str) -> tuple[str, list[str]] | None:
    """ONE cascade attempt after a pool-side ask failed.

    Returns (answer, citations) or None. Never raises; every attempt is
    journalled so the ledger stays truthful about which model answered.
    """
    settings = getattr(ctx, "settings", None)
    provider = cascade_provider(settings, purpose)
    if not provider:
        return None
    client = get_cascade_client(settings)
    if client is None:
        return None
    memory = getattr(ctx, "memory", None)

    def _journal(outcome: str, why: str = "") -> None:
        if memory is None:
            return
        try:
            memory.journal(ctx.agent_id, "router_cascade",
                           {"purpose": purpose, "provider": provider,
                            "outcome": outcome, "why": why[:200]})
        except Exception:  # noqa: BLE001 — telemetry must never gate harder
            pass

    try:
        resp = await client.ask(prompt, mode=mode, purpose=purpose,
                                system=system)
    except Exception as exc:  # noqa: BLE001 — cascade must never crash a cycle
        log.warning("[%s] %s cascade %s failed: %s",
                    getattr(ctx, "agent_id", "?"), purpose, provider, exc)
        _journal("failed", str(exc))
        return None
    answer = (getattr(resp, "answer", "") or "").strip()
    if not answer:
        _journal("empty")
        return None
    log.info("[%s] %s served by cascade provider %s",
             getattr(ctx, "agent_id", "?"), purpose, provider)
    _journal("ok")
    return answer, list(getattr(resp, "citations", []) or [])
