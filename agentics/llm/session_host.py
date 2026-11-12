"""BrowserSessionHost — dual-provider router over browser-session accounts.

The lab keeps its cookie-based, free-tier access model, but splits jobs by
what each LLM is actually good at:

* **Perplexity** (socket.io, ``academic`` mode) — the *research* node. It
  returns verifiable citation URLs ("giấy trắng mực đen"), which is exactly
  what paper search needs and what Qwen does not reliably provide.
* **Qwen** (chat.qwen.ai native API, ``qwen4.0-max-preview``) — every
  *code* node: ``plan``, ``implement``, ``review`` (and distillation). It is
  the strongest coder of the two, so it writes the plans, the code and the
  review verdicts.

Both providers are browser-session based (cookies exported from a logged-in
browser), so the host owns ONE pool per provider and routes per call:

    host.ask(prompt, mode="academic", purpose="research")   -> Perplexity
    host.ask(prompt, mode="concise",  purpose="implement")  -> Qwen

Graceful degradation: if the primary pool for a purpose has no healthy
account (empty, cooling down, WAF-blocked), the host falls back to the other
provider and logs a warning (citations may be missing on the Qwen path).

One agent thread = one pinned identity: ``view(prefer_index=i)`` pins agent
``i`` to account ``i`` of each pool, falling back to any healthy sibling
account when its own is cooling down — same contract as the old single
``AccountPool``, now across two providers.
"""
from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from agentics.config import Settings
from agentics.llm.perplexity_client import CookieAccount, LLMResponse, PerplexityClient
from agentics.llm.pool import AccountPool
from agentics.llm.qwen_client import QwenAccount, QwenClient

log = logging.getLogger("llm.session_host")

# purpose -> primary provider (first match wins; unknown purpose -> "general")
PURPOSE_PROVIDER: dict[str, str] = {
    "research": "perplexity",     # citations are non-negotiable here
    "plan": "qwen",
    "implement": "qwen",
    "review": "qwen",
    "distill": "qwen",
    "general": "qwen",
}
PROVIDERS = ("perplexity", "qwen")


class BrowserSessionHost:
    """Routes LLM calls by purpose across Perplexity and Qwen session pools.

    Use :meth:`from_settings` for standalone tooling (CLI), or build once with
    explicit pools and hand out :meth:`view` per agent thread (orchestrator).
    """

    def __init__(
        self,
        settings: Settings,
        pplx_pool: AccountPool | None = None,
        qwen_pool: AccountPool | None = None,
        prefer_index: int | None = None,
    ) -> None:
        self.settings = settings
        if pplx_pool is None or qwen_pool is None:
            both = _build_pools(settings)
            pplx_pool = pplx_pool or both[0]
            qwen_pool = qwen_pool or both[1]
        self.pools: dict[str, AccountPool] = {
            "perplexity": pplx_pool,
            "qwen": qwen_pool,
        }
        self.prefer_index = prefer_index
        # client cache: one live client per (provider, account) so Qwen keeps
        # its refreshed access token and Perplexity keeps socket state.
        self._clients: dict[tuple[str, str], Any] = {}
        self._last: tuple[str, CookieAccount | QwenAccount] | None = None

    # ------------------------------------------------------------ factory
    @classmethod
    def from_settings(cls, settings: Settings) -> "BrowserSessionHost":
        """Standalone host (no pinning) — loads both pools from accounts/."""
        return cls(settings, *_build_pools(settings))

    def view(self, prefer_index: int | None = None) -> "BrowserSessionHost":
        """Per-agent view sharing the pools (1 thread = 1 pinned identity)."""
        return BrowserSessionHost(
            self.settings, self.pools["perplexity"], self.pools["qwen"],
            prefer_index=prefer_index,
        )

    # ------------------------------------------------------------ routing
    def provider_for(self, purpose: str) -> str:
        return PURPOSE_PROVIDER.get(purpose, PURPOSE_PROVIDER["general"])

    def _pool(self, provider: str) -> AccountPool:
        return self.pools[provider]

    def _client_for(self, provider: str, account: CookieAccount | QwenAccount):
        key = (provider, account.name)
        if key not in self._clients:
            self._clients[key] = self._pool(provider).client_for(account)
        return self._clients[key]

    def _lease(self, purpose: str) -> tuple[str, Any] | tuple[None, None]:
        """Lease a client for a purpose; falls back to the other provider."""
        primary = self.provider_for(purpose)
        order = [primary] + [p for p in PROVIDERS if p != primary]
        for provider in order:
            pool = self._pool(provider)
            if not len(pool):
                continue
            acc = pool.lease(self.prefer_index)
            if acc is not None:
                return provider, self._client_for(provider, acc)
        return None, None

    # ------------------------------------------------------------ ask
    async def ask(
        self,
        query: str,
        *,
        mode: str = "concise",
        system: str | None = None,
        purpose: str = "general",
    ) -> LLMResponse:
        """One routed call. Raises RuntimeError when no account is usable."""
        provider, client = self._lease(purpose)
        if client is None:
            raise RuntimeError(
                "no usable browser-session account in either pool "
                "(accounts/ has no healthy perplexity or qwen sessions)")
        if provider != self.provider_for(purpose):
            log.warning("purpose '%s' falling back to %s (primary pool exhausted)",
                        purpose, provider)
        try:
            resp = await client.ask(query, mode=mode, system=system, purpose=purpose)
        except Exception as exc:
            acc = getattr(client, "account", None)
            if acc is not None:
                self._pool(provider).mark_failed(acc, str(exc)[:200])
            raise
        acc = getattr(client, "account", None)
        if acc is not None:
            self._pool(provider).mark_ok(acc)
            self._last = (provider, acc)
        return resp

    # ------------------------------------------------------------ state
    @property
    def account(self) -> CookieAccount | QwenAccount | None:
        """Last account that served a call (for ctx.meta / status output)."""
        return self._last[1] if self._last else None

    @property
    def last_provider(self) -> str:
        return self._last[0] if self._last else ""

    def mark_failed(self, error: str, cooldown_s: int = 900) -> None:
        """Cooldown the last-used account (agent-loop failover path)."""
        if self._last is None:
            return
        provider, acc = self._last
        self._pool(provider).mark_failed(acc, error, cooldown_s)

    def __len__(self) -> int:
        """How many agent identities can be pinned (largest pool wins)."""
        return max(len(self.pools["perplexity"]), len(self.pools["qwen"]))

    def status_lines(self) -> list[str]:
        lines = ["-- perplexity (research, citations) --"]
        lines.extend(self._pool("perplexity").status_lines() or ["(no accounts)"])
        lines.append("-- qwen (plan/implement/review, qwen4.0-max-preview) --")
        lines.extend(self._pool("qwen").status_lines() or ["(no accounts)"])
        return lines


def _build_pools(settings: Settings) -> tuple[AccountPool, AccountPool]:
    """One AccountPool per provider; Settings is a dataclass so ``replace``
    flips ``llm_provider`` without touching the shared original."""
    pplx = AccountPool(replace(settings, llm_provider="perplexity"))
    qwen = AccountPool(replace(settings, llm_provider="qwen"))
    return pplx, qwen
