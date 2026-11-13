"""P20 — LLM budget & circuit breaker: a runaway agent is a bounded event.

Covers the shared-memory usage ledger, the pool-wide hourly cap, the
per-agent fail-streak breaker with cooldown and reopen, and the `_llm_ask`
gate (blocked asks must not reach the provider and must record the refusal).
"""

import sys
import time
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


# ------------------------------------------------- cooldown anchor (live fix)
def _backdate_rows(mem, agent: str, seconds: float) -> None:
    """Shift every llm_usage row of one agent into the past (the ledger
    stamps time.time() internally, so tests backdate via SQL)."""
    with mem._lock, mem._conn:
        mem._conn.execute(
            "UPDATE llm_usage SET ts = ts - ? WHERE agent=?", (seconds, agent))


def test_last_llm_fail_ts_picks_the_newest_failure(mem):
    p = _policy(mem)
    assert mem.last_llm_fail_ts("a03") is None
    p.record("a03", "plan", False)
    _backdate_rows(mem, "a03", 100)        # the old failure is now stale
    p.record("a03", "plan", True)
    p.record("a03", "plan", False)         # a fresh failure
    fresh = mem.last_llm_fail_ts("a03")
    assert fresh is not None and (time.time() - fresh) < 10


def test_breaker_anchors_at_last_failure_not_last_ok(mem):
    """Live 2026-11-13 regression: the breaker anchored its cooldown at the
    agent's last SUCCESS, so one stale ok kept it permanently closed while
    0.4ms refusals piled up — the runaway it exists to bound ran free."""
    p = _policy(mem, fail_streak_limit=3, cooldown_s=900)
    for _ in range(3):
        p.record("a03", "plan", True)
    _backdate_rows(mem, "a03", 3600)       # last ok is an hour stale
    for _ in range(3):                     # fresh consecutive provider fails
        p.record("a03", "plan", False)
    assert p.breaker_open("a03") is True   # was False under the old anchor
    remaining = p.breaker_remaining_s("a03")
    assert 0 < remaining <= 900


def test_breaker_gives_one_real_attempt_per_cooldown_window(mem):
    """While the provider stays dead the ladder is strict: the window passes,
    the agent gets ONE real attempt; if it fails again the window re-arms."""
    p = _policy(mem, fail_streak_limit=2, cooldown_s=900)
    p.record("a03", "plan", False)
    p.record("a03", "plan", False)
    assert p.breaker_open("a03") is True
    _backdate_rows(mem, "a03", 1000)       # cooldown elapsed, streak >= 2
    assert p.breaker_open("a03") is False  # one real attempt allowed
    p.record("a03", "plan", False)         # the attempt failed again
    assert p.breaker_open("a03") is True   # window re-armed from THAT fail
    p.record("a03", "plan", True)          # a real answer closes it outright
    assert p.breaker_open("a03") is False


def test_breaker_remaining_s_zero_cases(mem):
    p = _policy(mem, fail_streak_limit=2, cooldown_s=900)
    assert p.breaker_remaining_s("a03") == 0.0     # no history at all
    p.record("a03", "plan", False)
    assert p.breaker_remaining_s("a03") == 0.0     # streak below the limit
    off = _policy(mem, fail_streak_limit=1, cooldown_s=0.0)
    off.record("a03", "plan", False)
    assert off.breaker_remaining_s("a03") == 0.0   # breaker disabled


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


# ------------------------------------------------------- refusals vs failures
class _JournalMem:
    """Minimal memory stub capturing journal calls (never raises)."""

    def __init__(self):
        self.events: list[tuple] = []

    def journal(self, *a, **k):
        self.events.append(a)


def _refusal_ctx(budget, fake_ask, mem=None):
    return SimpleNamespace(
        llm=SimpleNamespace(ask=fake_ask, account=None),
        agent_id="a03", role=Role.CODER,
        settings=SimpleNamespace(research_topic="t"),
        memory=mem or _JournalMem(), meta={}, budget=budget)


def test_llm_ask_refusal_is_not_recorded_as_failure():
    """An LLMRefused (empty pool / cooling accounts) never reached the
    provider: the fail-streak and the pool budget must stay untouched."""
    from agentics.llm.errors import LLMRefused

    recorded = []

    async def fake_ask(*a, **k):
        raise LLMRefused("no usable browser-session account for purpose 'plan'")

    budget = BudgetPolicy(None, max_calls_per_hour=0)
    budget.record = lambda *a, **k: recorded.append(a)
    mem = _JournalMem()
    state = {}
    answer, citations = _run(_llm_ask(
        _refusal_ctx(budget, fake_ask, mem), "p", purpose="plan", state=state))
    assert answer == "" and citations == []
    assert recorded == []                  # NOT a provider failure
    assert "llm refused" in state["error"]
    assert any(e[1] == "llm_refused" for e in mem.events)


def test_llm_ask_backend_circuit_refusal_not_recorded():
    """The Hyperbrowser frontend circuit refuses in 0.4ms — recording that
    as a failure let a 10-minute outage inflate a 15-minute breaker lock
    (live llm_usage rows 7-12). Refused is refused, whatever the type."""
    from agentics.llm.hyperbrowser_client import HyperbrowserBackendUnavailable

    recorded = []

    async def fake_ask(*a, **k):
        raise HyperbrowserBackendUnavailable(
            "Hyperbrowser/Perplexity frontend circuit open for 300s: challenge")

    budget = BudgetPolicy(None, max_calls_per_hour=0)
    budget.record = lambda *a, **k: recorded.append(a)
    state = {}
    answer, _ = _run(_llm_ask(
        _refusal_ctx(budget, fake_ask), "p", purpose="verify", state=state))
    assert answer == "" and recorded == []
    assert "llm refused" in state["error"]


def test_llm_ask_provider_failure_still_recorded():
    """A real attempt that really failed (network, UI) still feeds the
    fail-streak — refusals must not blur the line the breaker counts."""
    recorded = []

    async def fake_ask(*a, **k):
        raise RuntimeError("websocket read timed out (op=60s)")

    budget = BudgetPolicy(None, max_calls_per_hour=0)
    budget.record = lambda *a, **k: recorded.append(a)
    state = {}
    answer, _ = _run(_llm_ask(
        _refusal_ctx(budget, fake_ask), "p", purpose="plan", state=state))
    assert answer == "" and len(recorded) == 1
    assert recorded[0][:3] == ("a03", "plan", False)
    assert "llm plan call failed" in state["error"]


def _run(coro):
    import asyncio
    return asyncio.run(coro)
