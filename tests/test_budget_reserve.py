"""P20b — science reserve: verify/ideation keep fuel when plumbing drains
the pool, and a blocked science ask waits the short window slide and
retries once instead of dying for the rest of the hour.

Live 2026-11-14 07:08: the previous hour's plumbing pushed llm_usage past
the pool cap, and the saturation-escape hatch itself was refused —
`[a02] verify blocked by budget` — the one path that unblocks the whole
pipeline starved behind the same gate as the calls that starved it.
"""

import asyncio
import sys
import time
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402

from agentics.agents.roles import Role  # noqa: E402
from agentics.graph.nodes import _llm_ask  # noqa: E402
from agentics.memory.shared_memory import SharedMemory  # noqa: E402
from agentics.policies.budget import BudgetPolicy  # noqa: E402

RESERVE_PURPOSES = ("verify", "ideation")


@pytest.fixture()
def mem(tmp_path):
    m = SharedMemory(tmp_path / "b.db")
    yield m
    m.close()


def _policy(mem, **kw):
    kw.setdefault("agent_id", "a02")
    kw.setdefault("reserve_pct", 0.25)
    kw.setdefault("reserve_purposes", RESERVE_PURPOSES)
    return BudgetPolicy(mem, **kw)


def _backdate(mem, ts_offset_s: float, n: int = 1) -> None:
    """Insert n ledger rows dated ts_offset_s in the past (the window is
    what slides, so tests need calls older than `now`)."""
    with mem._lock:
        for _ in range(n):
            mem._conn.execute(
                "INSERT INTO llm_usage (ts, agent, purpose, ok, duration_s) "
                "VALUES (?, 'a02', 'plan', 1, 0.1)",
                (time.time() - ts_offset_s,))


# --------------------------------------------------------------- the reserve
def test_reserved_purpose_spends_the_full_cap(mem):
    p = _policy(mem, max_calls_per_hour=8)
    for _ in range(6):                       # = soft cap (8 * 0.75)
        p.record("a03", "plan", True)
    # plumbing: refused at the soft cap, and the message says so
    allowed, why = p.can_ask("a03", purpose="plan")
    assert allowed is False and "(cap 6)" in why
    # science: still allowed — the reserve slice belongs to it
    allowed, why = p.can_ask("a02", purpose="verify")
    assert allowed is True and why == ""
    allowed, _ = p.can_ask("a02", purpose="ideation")
    assert allowed is True


def test_reserved_purpose_still_hits_the_hard_cap(mem):
    p = _policy(mem, max_calls_per_hour=8)
    for _ in range(8):                       # the WHOLE cap incl. reserve
        p.record("a03", "plan", True)
    assert p.can_ask("a02", purpose="verify")[0] is False


def test_no_reserve_keeps_the_old_behaviour(mem):
    p = BudgetPolicy(mem, max_calls_per_hour=3, agent_id="a02")
    for _ in range(3):
        p.record("a02", "plan", True)
    allowed, why = p.can_ask("a02", purpose="verify")
    assert allowed is False and "(cap 3)" in why


def test_unknown_purpose_is_general_not_reserved(mem):
    p = _policy(mem, max_calls_per_hour=8)
    for _ in range(6):
        p.record("a02", "plan", True)
    assert p.can_ask("a02", purpose="research")[0] is False
    assert p.can_ask("a02", purpose="")[0] is False


def test_status_reports_the_reserve(mem):
    p = _policy(mem, max_calls_per_hour=8)
    st = p.status()
    assert st["reserve_pct"] == 0.25
    assert st["reserve_purposes"] == list(RESERVE_PURPOSES)
    assert st["soft_cap_general"] == 6


# ------------------------------------------------------------- retry hint
def test_retry_in_s_anchors_at_the_oldest_call(mem):
    p = _policy(mem, max_calls_per_hour=8)
    _backdate(mem, 3590.0)                   # exits the window in ~10 s
    for _ in range(5):                       # 6 total = soft cap
        p.record("a02", "plan", True)
    wait = p.retry_in_s("a02", purpose="plan")
    assert 0 < wait <= 3600 - 3589           # ~10 s, not the full hour
    # the science purpose is not blocked at all: nothing to wait for
    assert p.retry_in_s("a02", purpose="verify") == 0.0


def test_retry_in_s_zero_when_under_the_cap(mem):
    p = _policy(mem, max_calls_per_hour=8)
    p.record("a02", "plan", True)
    assert p.retry_in_s("a02", purpose="plan") == 0.0


def test_retry_in_s_counts_calls_that_must_slide(mem):
    p = _policy(mem, max_calls_per_hour=8, reserve_pct=0.5)
    _backdate(mem, 3500.0)                   # 1st oldest
    _backdate(mem, 3400.0)                   # 2nd oldest
    for _ in range(2):
        p.record("a02", "plan", True)        # 4 total; soft cap = 4
    # need = 4-4+1 = 1 -> the -3500 s call is the anchor
    wait = p.retry_in_s("a02", purpose="plan")
    assert 0 < wait <= 3600 - 3499


# ------------------------------------------------ _llm_ask wait-and-retry
class _FakeLLM:
    def __init__(self):
        self.by_purpose = {}

    async def ask(self, prompt, mode="concise", purpose="", system=""):
        self.by_purpose[purpose] = self.by_purpose.get(purpose, 0) + 1
        return SimpleNamespace(answer="answer", citations=[])


def _ctx(mem, llm, wait_cap_s=0.5):
    settings = SimpleNamespace(budget_block_wait_cap_s=wait_cap_s,
                               research_topic="t")
    return SimpleNamespace(
        agent_id="a02", role=Role.RESEARCHER, settings=settings,
        memory=mem, llm=SimpleNamespace(ask=llm.ask, account=None),
        meta={}, budget=_policy(mem, max_calls_per_hour=6),
    )


def test_llm_ask_reserved_purpose_waits_and_gets_through(mem):
    """The escape hatch arrives while the window is nearly drained: after a
    sub-second wait the oldest call slides out and the ask goes through."""
    llm = _FakeLLM()
    ctx = _ctx(mem, llm)
    _backdate(mem, 3600.0 - 0.03)            # exits the window in ~0.03 s
    for _ in range(5):
        ctx.budget.record("a02", "plan", True)   # 6 = hard cap
    assert ctx.budget.can_ask("a02", purpose="verify")[0] is False
    answer, _ = asyncio.run(_llm_ask(ctx, "prompt", purpose="verify"))
    assert answer == "answer"
    # exactly ONE real ask went out (resolve_system_prompt's synthesis ask
    # is separate and expected); the budget-wait path must not re-ask
    assert llm.by_purpose.get("verify") == 1


def test_llm_ask_general_purpose_never_waits(mem):
    llm = _FakeLLM()
    ctx = _ctx(mem, llm)
    for _ in range(5):                       # soft cap = int(6*0.75) = 4
        ctx.budget.record("a02", "plan", True)
    answer, _ = asyncio.run(_llm_ask(ctx, "prompt", purpose="plan"))
    assert answer == ""                     # refused immediately
    assert llm.by_purpose.get("plan", 0) == 0


def test_llm_ask_gives_up_when_the_wait_would_be_an_hour(mem):
    llm = _FakeLLM()
    ctx = _ctx(mem, llm)
    for _ in range(6):
        ctx.budget.record("a02", "plan", True)   # fresh rows: ~3600 s wait
    answer, _ = asyncio.run(_llm_ask(ctx, "prompt", purpose="verify"))
    assert answer == ""                     # no hour-long wait inside a node
    assert llm.by_purpose.get("verify", 0) == 0
