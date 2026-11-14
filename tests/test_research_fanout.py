"""P18b research fan-out — parallel research angles on separate pool identities.

With RESEARCH_FANOUT > 1, node_research asks complementary angles (survey /
benchmarks / open problems) concurrently, each on a different Hyperbrowser
token (or cookie account on the local backend). The merged answer flows into
the SAME finding pipeline; every angle passes the P20 budget gate. All tests
offline — no HTTP, no sessions.
"""

import asyncio
import sys
import time
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402

from agentics.config import Settings  # noqa: E402
from agentics.graph import nodes  # noqa: E402
from agentics.llm.perplexity_client import LLMResponse  # noqa: E402
from agentics.memory.shared_memory import SharedMemory  # noqa: E402

# ---------------------------------------------------------------- fakes
_C1 = "https://arxiv.org/abs/2401.11111"
_C2 = "https://doi.org/10.1/bench"
_C3 = "https://arxiv.org/abs/2401.22222"


class FakeVector:
    available = False

    def __init__(self):
        self.added = []

    def add(self, content, meta):
        self.added.append((content, meta))


class FakePersonal:
    def add_lesson(self, lesson, issue=None):
        pass


class Budget:
    """P20 ledger stand-in: counts asks, gates when exhausted."""

    def __init__(self, left=10):
        self.left = left
        self.asks = []

    def can_ask(self, agent, purpose=""):
        return (self.left > 0, "ok" if self.left > 0 else "exhausted")

    def record(self, agent, purpose, ok, elapsed):
        self.asks.append((purpose, ok))
        self.left -= 1


class AngleHost:
    """Pinned-view stand-in: per-angle answers + real concurrency stats.

    All views share one stats dict, so overlapping asks show up as
    max_inflight >= 2 — the actual proof that the fan-out is parallel.
    """

    def __init__(self, stats=None, fail_angles=False):
        self.stats = stats if stats is not None else {"inflight": 0, "max": 0}
        self.fail_angles = fail_angles
        self.account = SimpleNamespace(name="account-01")
        self.prefer_index = 0

    def view(self, prefer_index=None):
        h = AngleHost(self.stats, self.fail_angles)
        h.prefer_index = prefer_index or 0
        h.account = SimpleNamespace(name=f"account-{(prefer_index or 0) + 1:02d}")
        return h

    async def ask(self, query, *, mode="concise", system=None, purpose=None):
        self.stats["inflight"] += 1
        self.stats["max"] = max(self.stats["max"], self.stats["inflight"])
        try:
            await asyncio.sleep(0.05)
            if "[Angle: benchmarks" in query:
                if self.fail_angles:
                    raise RuntimeError("composer vanished")
                return LLMResponse(answer="benchmark angle answer",
                                   citations=[_C1, _C2])
            if "[Angle: limitations" in query:
                return LLMResponse(answer="open problems angle answer",
                                   citations=[_C3])
            return LLMResponse(answer="primary survey answer", citations=[_C1])
        finally:
            self.stats["inflight"] -= 1

    def mark_failed(self, error, cooldown_s=900):
        pass


class BudgetedHost(AngleHost):
    """Host that skips the extra angles when the budget gate refuses."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.budget = Budget(left=10)

    def view(self, prefer_index=None):
        h = BudgetedHost(self.stats, self.fail_angles)
        h.prefer_index = prefer_index or 0
        h.account = SimpleNamespace(name=f"account-{(prefer_index or 0) + 1:02d}")
        h.budget = self.budget                  # the same ledger everywhere
        return h

    async def ask(self, query, *, mode="concise", system=None, purpose=None):
        if "[Angle:" in query:                       # gate fires per angle
            allowed, why = self.budget.can_ask("a04")
            if not allowed:
                raise RuntimeError(f"budget: {why}")
            self.budget.record("a04", "research", True, 0.01)
        return await super().ask(query, mode=mode, system=system, purpose=purpose)


def _ctx(mem, llm, settings_overrides=None):
    settings = SimpleNamespace(
        research_topic="NFV automation",
        free_tools_enabled=True,
        research_fanout=1,
        browser_backend="hyperbrowser",
        hyperbrowser_api_keys=["tk0", "tk1", "tk2"],
    )
    for k, v in (settings_overrides or {}).items():
        setattr(settings, k, v)
    return SimpleNamespace(
        agent_id="a04", role=__import__("agentics.agents.roles", fromlist=["Role"]).Role.RESEARCHER,
        memory=mem, vector=FakeVector(), llm=llm,
        meta={"personal": FakePersonal()},
        settings=settings,
        heartbeat=lambda status="working": None,
        report=lambda **k: None,
    )


@pytest.fixture(autouse=True)
def _offline_bibtex(monkeypatch):
    """Keep node_research's BibTeX grounding off the network."""

    async def fake_enriched(urls, settings=None, max_lookups=8):
        return "@misc{grounded, title = {Real metadata}}"

    monkeypatch.setattr(nodes.free_tools, "enriched_bibtex", fake_enriched)


# ---------------------------------------------------------------- config
def test_settings_fanout_knob(monkeypatch):
    monkeypatch.setenv("RESEARCH_FANOUT", "3")
    assert Settings().research_fanout == 3
    monkeypatch.delenv("RESEARCH_FANOUT")
    assert Settings().research_fanout == 1


# ---------------------------------------------------------------- view leasing
def test_fanout_views_lease_distinct_tokens(tmp_path):
    host = AngleHost()
    host.prefer_index = 0
    ctx = _ctx(SharedMemory(tmp_path / "m.db"), host)
    views = nodes._fanout_views(ctx, 2)
    assert [v.prefer_index for v in views] == [1, 2]
    assert {v.account.name for v in views} == {"account-02", "account-03"}


def test_fanout_views_wrap_and_never_reuse_own_token(tmp_path):
    host = AngleHost()
    host.prefer_index = 2                      # last token of the pool
    ctx = _ctx(SharedMemory(tmp_path / "m.db"), host)
    views = nodes._fanout_views(ctx, 2)
    assert [v.prefer_index for v in views] == [0, 1]


def test_fanout_views_single_token_means_no_fanout(tmp_path):
    ctx = _ctx(SharedMemory(tmp_path / "m.db"), AngleHost(),
               {"hyperbrowser_api_keys": ["tk0"]})
    assert nodes._fanout_views(ctx, 2) == []


def test_fanout_views_requires_a_viewable_host(tmp_path):
    class BareHost:
        account = SimpleNamespace(name="account-01")

    ctx = _ctx(SharedMemory(tmp_path / "m.db"), BareHost())
    assert nodes._fanout_views(ctx, 2) == []


def test_fanout_views_local_backend_uses_pool_size(tmp_path):
    class PoolHost:
        prefer_index = 0
        account = SimpleNamespace(name="account-01")

        def view(self, prefer_index=None):
            h = AngleHost()
            h.prefer_index = prefer_index
            return h

        class _pool:
            accounts = [object(), object()]     # two cookie accounts

        pools = {"perplexity": _pool()}

    ctx = _ctx(SharedMemory(tmp_path / "m.db"), PoolHost(),
               {"browser_backend": "local", "hyperbrowser_api_keys": []})
    views = nodes._fanout_views(ctx, 3)         # the agent holds 1 account,
    assert len(views) == 1                      # 1 spare -> 1 extra angle max


# ---------------------------------------------------------------- node wiring
def _state():
    return {"issue_title": "Survey NFV", "papers": [], "citations": []}


def test_fanout_runs_angles_in_parallel_and_merges(tmp_path):
    mem = SharedMemory(tmp_path / "m.db")
    host = AngleHost()
    ctx = _ctx(mem, host, {"research_fanout": 3})
    out = asyncio.run(nodes.node_research(ctx, _state()))
    assert host.stats["max"] >= 2               # genuinely concurrent asks
    answer = out["papers"][0]["answer"]
    assert answer.startswith("primary survey answer")
    assert "## Supplementary angle — benchmarks & datasets" in answer
    assert "benchmark angle answer" in answer
    assert "open problems angle answer" in answer
    assert out["citations"] == [_C1, _C2, _C3]  # primary first, deduped
    assert ctx.meta["research_fanout"] == 3
    assert ctx.meta["research_source"] == "llm"
    assert "Real metadata" in out["bibtex"]     # one batched grounding call
    content = mem.list_findings(limit=1)[0]["content"]
    assert "Supplementary angle" in content
    mem.close()


def test_fanout_off_by_default_single_ask(tmp_path):
    mem = SharedMemory(tmp_path / "m.db")
    host = AngleHost()                          # settings.research_fanout = 1
    ctx = _ctx(mem, host)
    out = asyncio.run(nodes.node_research(ctx, _state()))
    assert host.stats["max"] == 1
    assert out["papers"][0]["answer"] == "primary survey answer"
    assert out["citations"] == [_C1]
    assert "research_fanout" not in ctx.meta
    mem.close()


def test_fanout_angle_failure_does_not_break_the_finding(tmp_path):
    mem = SharedMemory(tmp_path / "m.db")
    host = AngleHost(fail_angles=True)          # benchmarks angle raises
    ctx = _ctx(mem, host, {"research_fanout": 3})
    out = asyncio.run(nodes.node_research(ctx, _state()))
    assert out["papers"][0]["answer"].startswith("primary survey answer")
    assert "benchmark" not in out["papers"][0]["answer"]
    assert "open problems angle answer" in out["papers"][0]["answer"]
    assert out["citations"] == [_C1, _C3]
    assert ctx.meta["research_fanout"] == 2
    mem.close()


def test_fanout_respects_the_budget_gate(tmp_path):
    mem = SharedMemory(tmp_path / "m.db")
    host = BudgetedHost()
    ctx = _ctx(mem, host, {"research_fanout": 3})
    host.budget = Budget(left=0)                # gate refuses every angle
    out = asyncio.run(nodes.node_research(ctx, _state()))
    assert out["papers"][0]["answer"] == "primary survey answer"
    assert "research_fanout" not in ctx.meta    # no angle ran, nothing merged
    mem.close()


def test_fanout_budget_exhausted_after_first_angle(tmp_path):
    mem = SharedMemory(tmp_path / "m.db")
    host = BudgetedHost()
    ctx = _ctx(mem, host, {"research_fanout": 3})
    host.budget = Budget(left=1)                # room for exactly one angle
    out = asyncio.run(nodes.node_research(ctx, _state()))
    assert "benchmark angle answer" in out["papers"][0]["answer"]
    assert "open problems" not in out["papers"][0]["answer"]
    mem.close()


def test_primary_failure_still_falls_back_to_free_tools(tmp_path, monkeypatch):
    mem = SharedMemory(tmp_path / "m.db")

    class DeadHost(AngleHost):
        async def ask(self, query, **kw):
            raise RuntimeError("cookie dead")

    ctx = _ctx(mem, DeadHost(), {"research_fanout": 3})

    async def fake_fallback(query, settings=None, max_results=8):
        return nodes.free_tools.FreeResearch(answer="free tools finding",
                                             citations=["https://arxiv.org/abs/x"])

    monkeypatch.setattr(nodes.free_tools, "fallback_research", fake_fallback)
    out = asyncio.run(nodes.node_research(ctx, _state()))
    assert out["papers"][0]["answer"] == "free tools finding"
    assert ctx.meta["research_source"] == "free-tools"
    mem.close()
