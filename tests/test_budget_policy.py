"""P20 — LLM budget & circuit breaker: a runaway agent is a bounded event.

Covers the shared-memory usage ledger, the pool-wide hourly cap, the
per-agent fail-streak breaker with cooldown and reopen, and the `_llm_ask`
gate (blocked asks must not reach the provider and must record the refusal).
"""

import sys
from types import SimpleNamespace

import pytest

from agentics.agents.roles import Role
from agentics.graph.nodes import _llm_ask
from agentics.memory.shared_memory import SharedMemory
from agentics.policies.budget import BudgetPolicy


@pytest.fixture()
def mem(tmp_path):
    m = SharedMemory(tmp_path / "b.db")
    yield m
    m.close()


def _policy(mem, **kw):
    kw.setdefault("agent_id", "a03")
    return BudgetPolicy(mem, **kw)


# --------------------------------------------------------------- usage ledger
def test_record_and_window_count(mem):
    p = _policy(mem, max_calls_per_hour=10)
    for _ in range(3):
        p.record("a03", "plan", True, 0.5)
    p.record("a03", "review", False, 0.1)
    assert mem.llm_calls_in_window(3600.0) == 4
    assert mem.llm_calls_in_window(3600.0, agent="a03") == 4
    assert mem.llm_calls_in_window(3600.0, agent="a99") == 0


def test_fail_streak_counts_only_consecutive_failures(mem):
    p = _policy(mem)
    p.record("a03", "plan", False)
    p.record("a03", "plan", False)
    p.record("a03", "plan", True)          # a real answer resets the streak
    p.record("a03", "plan", False)
    assert mem.llm_fail_streak("a03") == 1
    assert mem.last_llm_ok_ts("a03") is not None


# ------------------------------------------------------------------- pool cap
def test_pool_cap_blocks_every_agent_when_exhausted(mem):
    p = _policy(mem, max_calls_per_hour=3)
    for i in range(3):
        p.record("a02", "plan", True)
    assert p.can_ask("a02") == (False, "P20 pool budget exhausted: 3 calls in "
                                      "the last hour (cap 3)")
    # the pool is shared: a03 is blocked by a02's usage
    assert p.can_ask("a03")[0] is False


def test_cap_zero_disables_the_cap_not_the_breaker(mem):
    p = _policy(mem, max_calls_per_hour=0, fail_streak_limit=2, cooldown_s=60)
    for _ in range(9):
        p.record("a03", "plan", True)
    assert p.can_ask("a03") == (True, "")


# ------------------------------------------------------------- circuit breaker
def test_breaker_opens_on_fail_streak_and_reopens_after_ok(mem):
    p = _policy(mem, fail_streak_limit=3, cooldown_s=900)
    assert p.breaker_open("a03") is False
    for _ in range(3):
        p.record("a03", "plan", False)
    assert p.breaker_open("a03") is True
    allowed, why = p.can_ask("a03")
    assert allowed is False and "circuit breaker open" in why
    # one real answer reopens it instantly
    p.record("a03", "plan", True)
    assert p.breaker_open("a03") is False


def test_breaker_opening_is_journaled_once(mem):
    p = _policy(mem, fail_streak_limit=2, cooldown_s=60)
    p.record("a03", "plan", False)
    p.record("a03", "plan", False)         # streak == limit -> journal once
    p.record("a03", "plan", False)         # streak > limit -> NOT again
    events = [e["event"] for e in mem.recent_journal(limit=10)]
    assert events.count("budget_breaker_open") == 1


def test_breaker_never_succeeds_without_cooldown(mem):
    p = _policy(mem, fail_streak_limit=1, cooldown_s=0.0)
    p.record("a03", "plan", False)
    assert p.breaker_open("a03") is False  # cooldown 0 = breaker disabled


def test_status_snapshot(mem):
    p = _policy(mem, max_calls_per_hour=5)
    p.record("a02", "plan", True)
    s = p.status()
    assert s["calls_last_hour"] == 1 and s["cap_per_hour"] == 5
    assert s["pool_exhausted"] is False


# --------------------------------------------------------------- _llm_ask gate
def test_llm_ask_blocked_by_budget_never_reaches_provider():
    calls = []

    async def fake_ask(*a, **k):
        calls.append(a)
        return SimpleNamespace(answer="hello", citations=[])

    class Mem:
        def journal(self, *a, **k):
            pass

    budget = BudgetPolicy(None, max_calls_per_hour=0)
    budget.can_ask = lambda agent: (False, "P20 pool budget exhausted (test)")
    budget.record = lambda *a, **k: None
    ctx = SimpleNamespace(llm=SimpleNamespace(ask=fake_ask, account=None),
                          agent_id="a03", role=Role.CODER, settings=SimpleNamespace(
                              research_topic="t"),
                          memory=Mem(), meta={}, budget=budget)
    state = {}
    answer, citations = _run(_llm_ask(ctx, "prompt", purpose="plan", state=state))
    assert answer == "" and citations == []
    assert calls == []                     # the provider was never touched
    assert "P20 budget" in state["error"]


def test_llm_ask_without_budget_attr_still_works():
    """Tests and legacy paths build contexts without a budget — the gate
    must degrade to a plain pass-through (P20 never breaks a healthy path)."""
    calls = []

    async def fake_ask(*a, **k):
        calls.append(a)
        return SimpleNamespace(answer="hi", citations=[])

    ctx = SimpleNamespace(llm=SimpleNamespace(ask=fake_ask, account=None),
                          agent_id="a03", role=Role.CODER,
                          settings=SimpleNamespace(research_topic="t"),
                          meta={})
    answer, _ = _run(_llm_ask(ctx, "p"))
    assert answer == "hi" and len(calls) == 1


def _run(coro):
    import asyncio
    return asyncio.run(coro)
