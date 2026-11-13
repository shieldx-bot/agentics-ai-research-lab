"""P18 — LLM budget & circuit breaker: a runaway agent cannot burn the pool.

Every agent shares one cookie/quota pool (the lab's finite LLM resource).
Policies P1-P17 prevent agents from stepping on each other's CODE; nothing
prevented an agent stuck in a hallucination loop from draining the shared
pool forever. This module adds the two production guardrails:

* **Pool-wide hourly cap** — the budget counts every LLM call recorded in
  `llm_usage` inside a sliding window; past the cap every agent is refused
  until the window slides. The cap is POOL-WIDE because the resource is.
* **Per-agent circuit breaker** — an agent whose calls fail N times in a
  row (provider errors, not slow answers) is opened into a cooldown: it
  must stop asking until the cooldown passes or it gets one real answer.
  This turns "one agent hammering a dead provider" into a bounded event
  instead of an infinite loop.

The policy NEVER raises: a refused ask is a plain (False, reason) tuple,
the caller journals it and the cycle degrades gracefully.
"""
from __future__ import annotations

import time


class BudgetPolicy:
    """Shared-memory backed rate limiter + circuit breaker (P18)."""

    def __init__(
        self, memory,
        max_calls_per_hour: int = 120,
        fail_streak_limit: int = 5,
        cooldown_s: float = 900.0,
        agent_id: str = "",
    ) -> None:
        self.memory = memory
        self.max_calls_per_hour = max(int(max_calls_per_hour), 0)
        self.fail_streak_limit = max(int(fail_streak_limit), 1)
        self.cooldown_s = max(float(cooldown_s), 0.0)
        self.agent_id = agent_id

    # ------------------------------------------------------------ gate
    def can_ask(self, agent: str = "") -> tuple[bool, str]:
        """(allowed, reason) BEFORE an ask. Refusals are policy, not errors."""
        agent = agent or self.agent_id
        if not agent or self.memory is None:
            return True, ""
        # circuit breaker first — cheapest check explains the most
        if self.breaker_open(agent):
            return False, (f"P18 circuit breaker open for {agent} "
                           f"(fail streak >= {self.fail_streak_limit}, "
                           f"cooldown {int(self.cooldown_s)}s)")
        if self.max_calls_per_hour > 0:
            used = self.memory.llm_calls_in_window(3600.0)
            if used >= self.max_calls_per_hour:
                return False, (f"P18 pool budget exhausted: {used} calls in "
                               f"the last hour (cap {self.max_calls_per_hour})")
        return True, ""

    def record(self, agent: str, purpose: str, ok: bool,
               duration_s: float = 0.0) -> None:
        """Record ONE ask AFTER it happened (ok=False = provider failure).
        Crossing the fail-streak limit opens the breaker and journals it."""
        if not agent or self.memory is None:
            return
        self.memory.record_llm_call(agent, purpose, ok, duration_s)
        if ok:
            return
        streak = self.memory.llm_fail_streak(agent)
        if streak >= self.fail_streak_limit and self.cooldown_s > 0:
            # journal once per opening: only when the streak EQUALS the limit
            if streak == self.fail_streak_limit:
                self.memory.journal(agent, "budget_breaker_open", {
                    "agent": agent, "fail_streak": streak,
                    "cooldown_s": int(self.cooldown_s),
                })
                self.memory.add_decision(
                    agent, "budget_breaker_open",
                    f"{streak} consecutive LLM failures — paused {int(self.cooldown_s)}s")

    # ------------------------------------------------------------ state
    def breaker_open(self, agent: str = "") -> bool:
        """True while the agent is in its post-failure cooldown."""
        agent = agent or self.agent_id
        if not agent or self.memory is None or self.cooldown_s <= 0:
            return False
        streak = self.memory.llm_fail_streak(agent)
        if streak < self.fail_streak_limit:
            return False
        last_ok = self.memory.last_llm_ok_ts(agent)
        if last_ok is None:
            return True                      # never succeeded: stay open
        return (time.time() - last_ok) < self.cooldown_s

    def status(self) -> dict:
        """Dashboard/CLI snapshot: pool usage vs cap and open breakers."""
        used = self.memory.llm_calls_in_window(3600.0) if self.memory else 0
        return {
            "calls_last_hour": used,
            "cap_per_hour": self.max_calls_per_hour,
            "pool_exhausted": (self.max_calls_per_hour > 0
                               and used >= self.max_calls_per_hour),
            "fail_streak_limit": self.fail_streak_limit,
            "cooldown_s": int(self.cooldown_s),
        }
