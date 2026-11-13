"""Breadth steering + solo-lab fan-out — widening the swarm's source diet.

exploration_coverage (swarm_eval) = distinct citation domains / citation
URLs. Live 2026-11-13 it sat at 76.9% (13 URLs / 10 domains) because
(a) a one-token pool could never fan out — `_fanout_views` returns [] when
capacity < 2 — so every cycle cited one answer's source pool, and
(b) the query never knew which domains the swarm had already cited.

The fix is behavioural, not metric-side: a breadth directive built from
shared memory (same domain normalisation the evaluator uses) and a
sequential same-view angle for solo pools. All tests offline — no HTTP,
no sessions, no LLM.
"""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402

from agentics.config import Settings  # noqa: E402
from agentics.graph import nodes  # noqa: E402
from agentics.llm.perplexity_client import LLMResponse  # noqa: E402
from agentics.memory.shared_memory import SharedMemory  # noqa: E402
from agentics.research import diversity  # noqa: E402
from agentics.swarm_eval import _distinct_domains  # noqa: E402

_C1 = "https://sqlite.org/wal.html"
_C2 = "https://arxiv.org/abs/2401.11111"
_C3 = "https://dl.acm.org/doi/10.1145/3124680.3124719"


# ---------------------------------------------------------------- diet read
class FakeMemory:
    def __init__(self, findings=None, explode=False):
        self._findings = findings or []
        self._explode = explode

    def list_findings(self, limit=50):
        if self._explode:
            raise RuntimeError("memory exploded")
        return self._findings[:limit]


def test_cited_domains_uses_the_evaluators_normalisation():
    urls = ["https://arxiv.org/abs/2401.1", "https://arxiv.org/html/2401.1v2",
            "https://www.sqlite.org/wal.html", "not-a-url", ""]
    assert _distinct_domains(urls) == {"arxiv.org", "sqlite.org"}


def test_cited_domains_reads_findings_and_dedups():
    mem = FakeMemory([
        {"citations": [_C1, _C2]},
        {"citations": [_C1, _C3]},          # sqlite.org repeats — one domain
        {"citations": None},                # finding without citations
    ])
    assert diversity.cited_domains(mem) == sorted(
        {"sqlite.org", "arxiv.org", "dl.acm.org"})


def test_cited_domains_degrades_on_memory_error():
    assert diversity.cited_domains(FakeMemory(explode=True)) == []


def test_breadth_directive_empty_when_no_diet():
    assert diversity.breadth_directive([]) == ""


def test_breadth_directive_names_diet_and_asks_for_outside_sources():
    text = diversity.breadth_directive(["arxiv.org", "sqlite.org"])
    assert "arxiv.org, sqlite.org" in text
    assert "OUTSIDE" in text
    # canonical re-citation stays allowed — honesty over novelty-chasing
    assert "canonical" in text


def test_breadth_directive_caps_the_domain_list():
    many = [f"d{i}.example.com" for i in range(40)]
    text = diversity.breadth_directive(many)
    assert "d15.example.com" in text and "d16.example.com" not in text


# ------------------------------------------------- citation host hygiene
def test_canonical_url_unifies_host_forms_of_one_document():
    from agentics.research.citations import canonical_url
    assert canonical_url("https://www.sqlite.org/wal.html") == \
        canonical_url("https://sqlite.org/wal.html")
    assert canonical_url("https://dx.doi.org/10.1145/x") == \
        canonical_url("https://doi.org/10.1145/x")
    assert canonical_url("https://arxiv.org/abs/1/#sec-2") == \
        canonical_url("https://arxiv.org/abs/1/")
    # genuinely different documents stay different
    assert canonical_url("https://sqlite.org/wal.html") != \
        canonical_url("https://sqlite.org/c3ref/wal_checkpoint_v2.html")


def test_merge_research_dedups_same_document_across_host_forms():
    primary = LLMResponse(answer="primary", citations=["https://sqlite.org/wal.html"])
    extras = [("benchmarks", SimpleNamespace(
        answer="angle body",
        citations=["https://www.sqlite.org/wal.html",      # same doc, www form
                   "https://dblp.org/pid/09/11157"]))]
    merged = nodes._merge_research(primary, extras)
    assert merged.citations == ["https://sqlite.org/wal.html",
                                "https://dblp.org/pid/09/11157"]
    assert "angle body" in merged.answer


# ---------------------------------------------------------------- fan-out solo
class FakeVector:
    available = False

    def __init__(self):
        self.added = []

    def add(self, content, meta):
        self.added.append((content, meta))


class FakePersonal:
    def add_lesson(self, lesson, issue=None):
        pass


class SoloHost:
    """One pinned identity, no spare views — the 1-Hyperbrowser-token lab.

    Records every query so the tests can assert the breadth directive and
    the angle hint actually travelled; `max` mirrors real concurrency."""

    def __init__(self):
        self.stats = {"inflight": 0, "max": 0, "queries": []}
        self.account = SimpleNamespace(name="account-01")
        self.prefer_index = 0

    def view(self, prefer_index=None):
        h = SoloHost()
        h.stats = self.stats
        h.prefer_index = prefer_index or 0
        return h

    async def ask(self, query, *, mode="concise", system=None, purpose=None):
        self.stats["inflight"] += 1
        self.stats["max"] = max(self.stats["max"], self.stats["inflight"])
        try:
            self.stats["queries"].append(query)
            await asyncio.sleep(0.01)
            if "[Angle: benchmarks" in query:
                return LLMResponse(answer="benchmark angle answer",
                                   citations=[_C2, _C3])
            return LLMResponse(answer="primary survey answer",
                               citations=[_C1, _C2])
        finally:
            self.stats["inflight"] -= 1

    def mark_failed(self, error, cooldown_s=900):
        pass


class Budget:
    def __init__(self, left=10):
        self.left = left
        self.asks = []

    def can_ask(self, agent):
        return (self.left > 0, "ok" if self.left > 0 else "exhausted")

    def record(self, agent, purpose, ok, elapsed):
        self.asks.append((purpose, ok))
        self.left -= 1


def _ctx(mem, llm, settings_overrides=None):
    settings = SimpleNamespace(
        research_topic="SQLite WAL tuning",
        free_tools_enabled=True,
        research_fanout=1,
        research_same_view_angle=True,
        browser_backend="hyperbrowser",
        hyperbrowser_api_keys=["tk0"],      # ONE token — the solo lab
    )
    for k, v in (settings_overrides or {}).items():
        setattr(settings, k, v)
    import agentics.agents.roles as roles_mod
    return SimpleNamespace(
        agent_id="a01", role=roles_mod.Role.RESEARCHER,
        memory=mem, vector=FakeVector(), llm=llm,
        meta={"personal": FakePersonal()},
        settings=settings,
        heartbeat=lambda status="working": None,
        report=lambda **k: None,
    )


def _state():
    return {"issue_title": "Survey WAL", "papers": [], "citations": []}


@pytest.fixture(autouse=True)
def _offline_bibtex(monkeypatch):
    async def fake_enriched(urls, settings=None, max_lookups=8):
        return "@misc{grounded, title = {Real metadata}}"

    monkeypatch.setattr(nodes.free_tools, "enriched_bibtex", fake_enriched)


def test_same_view_angle_runs_for_a_solo_pool(tmp_path):
    mem = SharedMemory(tmp_path / "m.db")
    host = SoloHost()
    ctx = _ctx(mem, host, {"research_fanout": 2})
    out = asyncio.run(nodes.node_research(ctx, _state()))
    # sequential, never parallel on one session
    assert host.stats["max"] == 1
    answer = out["papers"][0]["answer"]
    assert answer.startswith("primary survey answer")
    assert "## Supplementary angle — benchmarks & datasets" in answer
    assert out["citations"] == [_C1, _C2, _C3]   # merged, URL-deduped
    assert ctx.meta["research_fanout"] == 2
    assert ctx.meta["research_fanout_same_view"] == 1
    events = [e["event"] for e in mem.journal_events(0.0)]
    assert "research_fanout_same_view" in events
    mem.close()


def test_same_view_angle_respects_the_off_switch(tmp_path):
    mem = SharedMemory(tmp_path / "m.db")
    host = SoloHost()
    ctx = _ctx(mem, host, {"research_fanout": 2,
                           "research_same_view_angle": False})
    out = asyncio.run(nodes.node_research(ctx, _state()))
    assert len(host.stats["queries"]) == 1        # only the primary ask
    assert out["citations"] == [_C1, _C2]
    assert "research_fanout_same_view" not in ctx.meta
    mem.close()


def test_same_view_angle_blocked_by_budget_gates_nothing(tmp_path):
    mem = SharedMemory(tmp_path / "m.db")
    host = SoloHost()
    ctx = _ctx(mem, host, {"research_fanout": 2})
    ctx.budget = Budget(left=1)    # room for the primary only — the gate
    asyncio.run(nodes.node_research(ctx, _state()))   # refuses every angle
    assert len(host.stats["queries"]) == 1      # only the primary ask ran
    assert "research_fanout_same_view" not in ctx.meta
    mem.close()


def test_same_view_angle_failure_stops_the_loop(tmp_path, monkeypatch):
    mem = SharedMemory(tmp_path / "m.db")
    host = SoloHost()
    ctx = _ctx(mem, host, {"research_fanout": 3})   # 2 extra angles wanted

    async def dead_angle(ctx, view, angle, base_query, budget):
        return None                                  # first angle dies

    monkeypatch.setattr(nodes, "_ask_angle", dead_angle)
    extras = asyncio.run(nodes._same_view_angles(ctx, "q", 2, None))
    assert extras == []
    assert "research_fanout_same_view" not in ctx.meta


def test_settings_same_view_knob(monkeypatch):
    monkeypatch.setenv("RESEARCH_SAME_VIEW_ANGLE", "0")
    assert Settings().research_same_view_angle is False
    monkeypatch.delenv("RESEARCH_SAME_VIEW_ANGLE")
    assert Settings().research_same_view_angle is True


# ---------------------------------------------------------------- steering
def test_breadth_directive_travelled_in_the_query_and_journal(tmp_path):
    mem = SharedMemory(tmp_path / "m.db")
    mem.add_finding("a09", "Findings: prior work",
                    "prior content", [_C1, _C2], None)
    host = SoloHost()
    ctx = _ctx(mem, host)
    asyncio.run(nodes.node_research(ctx, _state()))
    primary = host.stats["queries"][0]
    assert "Breadth directive" in primary
    assert "sqlite.org" in primary                  # the existing diet named
    events = [e for e in mem.journal_events(0.0)
              if e["event"] == "research_breadth_steered"]
    assert events and events[-1]["detail"]["known_domains"] >= 2
    mem.close()


def test_breadth_steering_degrades_silently_on_memory_error(tmp_path):
    mem = SharedMemory(tmp_path / "m.db")
    host = SoloHost()
    ctx = _ctx(mem, host)

    class Exploding:
        """Diet read + journal explode; everything else hits real memory."""

        def __init__(self, real):
            self._real = real

        def list_findings(self, limit=50):
            raise RuntimeError("db locked")

        def journal(self, *a, **k):
            raise RuntimeError("journal gone")

        def __getattr__(self, name):
            return getattr(self._real, name)

    ctx.memory = Exploding(mem)
    out = asyncio.run(nodes.node_research(ctx, _state()))
    assert out["papers"][0]["answer"] == "primary survey answer"  # unharmed
    assert "Breadth directive" not in host.stats["queries"][0]
