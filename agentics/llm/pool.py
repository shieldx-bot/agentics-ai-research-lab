"""Account pool — one cookie account per agent thread, with health & cooldown."""
from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path

from agentics.config import Settings
from agentics.llm.perplexity_client import CookieAccount, PerplexityClient
from agentics.llm.qwen_client import QwenAccount, QwenClient, load_qwen_account

log = logging.getLogger("llm.pool")

PPLX_DOMAIN = "perplexity.ai"


def load_account_file(path: Path) -> CookieAccount:
    """Load a cookie export (Cookie-Editor JSON array format)."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if isinstance(raw, dict) and "cookies" in raw:
        raw = raw["cookies"]
    cookies: dict[str, str] = {}
    for item in raw:
        name, value = item.get("name"), item.get("value")
        domain = str(item.get("domain", ""))
        if name and value is not None and PPLX_DOMAIN in domain:
            cookies[name] = str(value)
    return CookieAccount(name=path.stem, cookies=cookies, path=str(path))


@dataclass
class AccountHealth:
    ok: bool = True
    last_error: str = ""
    cooldown_until: float = 0.0
    calls: int = 0
    failures: int = 0

    def in_cooldown(self) -> bool:
        return time.time() < self.cooldown_until


class AccountPool:
    """Owns all cookie accounts; assigns one per agent thread.

    Rotates on auth failure: `lease()` returns the healthiest free account,
    `mark_failed()` puts a broken account on cooldown so another agent thread
    can take over its role on the next cycle.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.accounts: list[CookieAccount] = []
        self.health: dict[str, AccountHealth] = {}
        self._load_accounts()

    def _load_accounts(self) -> None:
        provider = (self.settings.llm_provider or "perplexity").lower()
        pattern = "qwen-*.json" if provider == "qwen" else "account-*.json"
        files = sorted(self.settings.accounts_dir.glob(pattern))
        # templates (accounts/*.example.json) are docs, never live sessions
        files = [f for f in files if ".example" not in f.name]
        loader = load_qwen_account if provider == "qwen" else load_account_file
        for f in files:
            try:
                acc = loader(f)
                if getattr(acc, "cookies", None):
                    self.accounts.append(acc)
                    self.health[acc.name] = AccountHealth()
                else:
                    log.warning("account file %s has no %s cookies", f.name, provider)
            except Exception as exc:  # noqa: BLE001
                log.error("cannot parse account file %s: %s", f.name, exc)
        log.info("loaded %d %s cookie accounts", len(self.accounts), provider)

    def __len__(self) -> int:
        return len(self.accounts)

    def lease(self, prefer_index: int | None = None) -> CookieAccount | None:
        """Return a healthy account (round-robin with jitter), or None."""
        candidates = [
            a for a in self.accounts
            if not self.health[a.name].in_cooldown()
        ]
        if not candidates:
            return None
        if prefer_index is not None and prefer_index < len(candidates):
            return candidates[prefer_index]
        return random.choice(candidates)

    def mark_ok(self, account: CookieAccount) -> None:
        h = self.health[account.name]
        h.ok = True
        h.last_error = ""
        h.calls += 1

    def mark_failed(self, account: CookieAccount, error: str, cooldown_s: int = 900) -> None:
        h = self.health[account.name]
        h.ok = False
        h.last_error = error[:200]
        h.failures += 1
        h.cooldown_until = time.time() + cooldown_s
        log.warning("account %s on cooldown %ds: %s", account.name, cooldown_s, h.last_error)

    def client_for(self, account: CookieAccount | QwenAccount):
        if isinstance(account, QwenAccount):
            return QwenClient(account, timeout=self.settings.pplx_timeout)
        return PerplexityClient(
            account,
            ws_url=self.settings.pplx_ws_url,
            origin=self.settings.pplx_origin,
            timeout=self.settings.pplx_timeout,
        )

    def status_lines(self) -> list[str]:
        lines = []
        for acc in self.accounts:
            h = self.health[acc.name]
            state = "cooldown" if h.in_cooldown() else ("ok" if h.ok else "unknown")
            lines.append(
                f"{acc.name:<14} {state:<9} calls={h.calls:<5} fails={h.failures:<4} {h.last_error}"
            )
        return lines
