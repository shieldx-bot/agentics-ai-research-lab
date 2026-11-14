"""P7 saturation rotation — a covered topic must rotate to a fresh angle.

Live 2026-11-13: a lone researcher whose configured topic was already
covered by stored findings printed "topic already covered by 3 findings;
skipping" EVERY cycle forever — the lab looked dead from the owner's seat.
Rotation tries fresh ANGLES of the same domain; an angle already researched
(its label lives in a stored finding title) is skipped, so the ratchet
advances and only a fresh angle reaches the LLM. The vector gate cannot do
this job: broad survey findings sit at distance ~0.4 of EVERY query in
their domain (live calibration), so angle freshness is judged on titles.
All offline — no HTTP, no sessions.
"""
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402

from agentics.graph import nodes  # noqa: E402
from agentics.memory.shared_memory import SharedMemory  # noqa: E402

TOPIC = "Survey NFV"


class RotVector:
    """Vector stand-in: queries in `covered` come back as 3 near hits
    (distance 0.4 — inside DEDUP_MAX_DISTANCE), everything else as far
    hits (1.5 — filtered out by the gate)."""

    available = True

    def __init__(self, covered):
        self.covered = set(covered)
        self.queries: list[str] = []

    def search(self, query, n=5):
        self.queries.append(query)
        if query in self.covered:
            return [{"text": "old finding", "meta": {"title": "old"},
                     "distance": 0.4}] * 3
        return [{"text": "far", "meta": {}, "distance": 1.5}] * 3

    def add(self, text, metadata=None):
        return "doc-1"


class FakePersonal:
    def add_lesson(self, lesson, issue=None):
        pass


def _ctx(tmp_path, vector):
    mem = SharedMemory(tmp_path / "m.db")
    return SimpleNamespace(
        agent_id="a04",
        role=__import__("agentics.agents.roles", fromlist=["Role"]).Role.RESEARCHER,
        memory=mem, vector=vector,
        # non-None: a None host would push node_research into the free-tools
        # HTTP fallback; search_papers itself is monkeypatched offline anyway
        llm=SimpleNamespace(account=SimpleNamespace(name="account-01")),
        meta={"personal": FakePersonal()},
        settings=SimpleNamespace(research_topic="NFV automation",
                                 free_tools_enabled=True),
        heartbeat=lambda status="working": None,
        report=lambda **k: None,
    ), mem


def _seed_angle(mem, label: str) -> None:
    """Store a finding whose title carries the angle label — exactly what a
    previous rotation run leaves behind (finding title = f"Findings:
    {topic}" and the rotated topic embeds the label)."""
    mem.add_finding("a01", title=f"Findings: Survey NFV ({label})",
                    content="prior angle research", citations=[])


def _state():
    return {"issue_title": TOPIC, "papers": [], "citations": []}


@pytest.fixture()
def offline_llm(monkeypatch):
    """Record the research query node_research would send; return a
    well-formed response so the finding pipeline completes offline."""
    seen = {}

    async def fake_search(llm, query, purpose="research"):
        seen["query"] = query
        return SimpleNamespace(ok=True, answer="fresh angle answer",
                               citations=["https://arxiv.org/abs/2401.1"],
                               raw={}, elapsed_s=0.1)

    monkeypatch.setattr(nodes.papers_mod, "search_papers", fake_search)

    async def fake_enriched(urls, settings=None, max_lookups=8):
        return "@misc{grounded, title = {Real metadata}}"

    monkeypatch.setattr(nodes.free_tools, "enriched_bibtex", fake_enriched)
    return seen


def test_saturated_topic_rotates_to_first_fresh_angle(tmp_path, offline_llm):
    """The configured topic is covered and no angle was researched yet ->
    the FIRST angle is researched (the query carries the angle label) and
    the rotation is journalled for the owner."""
    ctx, mem = _ctx(tmp_path, RotVector(covered={TOPIC}))
    out = asyncio.run(nodes.node_research(ctx, _state()))
    assert "(benchmarks & datasets)" in offline_llm["query"]
    assert out["papers"][0]["answer"] == "fresh angle answer"
    events = [e["event"] for e in mem.recent_journal(limit=10)]
    assert "research_topic_rotated" in events


def test_rotation_skips_already_researched_angles(tmp_path, offline_llm):
    """The ratchet: an angle whose label lives in a stored finding title is
    spent — the NEXT angle gets researched."""
    ctx, mem = _ctx(tmp_path, RotVector(covered={TOPIC}))
    _seed_angle(mem, "benchmarks & datasets")
    asyncio.run(nodes.node_research(ctx, _state()))
    assert "(limitations & open problems)" in offline_llm["query"]
    assert "(benchmarks & datasets)" not in offline_llm["query"]


def test_every_angle_researched_rests_honestly(tmp_path, offline_llm):
    """When the topic AND every rotation angle are spent, the cycle rests
    WITHOUT calling the LLM — saturation is visible, not silent."""
    ctx, mem = _ctx(tmp_path, RotVector(covered={TOPIC}))
    for label in nodes._ROTATION_ANGLES:
        _seed_angle(mem, label)
    out = asyncio.run(nodes.node_research(ctx, _state()))
    assert out["summary"].startswith("topic already covered")
    assert out["done"] is True
    assert "query" not in offline_llm        # the LLM was never asked
    events = [e["event"] for e in mem.recent_journal(limit=10)]
    assert "research_topic_rotated" not in events


def test_uncovered_topic_never_rotates(tmp_path, offline_llm):
    """A fresh topic goes to the LLM verbatim — no angle suffix, no journal."""
    ctx, _mem = _ctx(tmp_path, RotVector(covered=set()))
    asyncio.run(nodes.node_research(ctx, _state()))
    assert offline_llm["query"].startswith(TOPIC)
    assert "(benchmarks & datasets)" not in offline_llm["query"]


def test_angle_researched_judges_titles_not_the_topic(tmp_path):
    """Unit: only a finding title carrying the LABEL's own vocabulary marks
    the angle spent — base-topic findings (no label) never do."""
    mem = SharedMemory(tmp_path / "titles.db")
    ctx = SimpleNamespace(memory=mem)
    assert nodes._angle_researched(ctx, "benchmarks & datasets") is False
    mem.add_finding("a01", title="Findings: Survey NFV broad survey",
                    content="x", citations=[])
    assert nodes._angle_researched(ctx, "benchmarks & datasets") is False
    mem.add_finding("a01", title="Findings: Survey NFV (benchmarks & datasets)",
                    content="x", citations=[])
    assert nodes._angle_researched(ctx, "benchmarks & datasets") is True
    mem.close()


def test_dedup_known_fallback_uses_finding_search(tmp_path):
    """Vector offline -> the SAME gate runs over the SQLite keyword search
    (hits >= 2), exactly like the pre-rotation behavior."""
    from agentics.agents.roles import Role

    mem = SharedMemory(tmp_path / "m.db")
    mem.add_finding("a01", title="Survey NFV resource scheduling",
                    content="x", citations=[])
    ctx = SimpleNamespace(memory=mem, vector=RotVector(covered=set()),
                          agent_id="a01", role=Role.RESEARCHER)
    ctx.vector.available = False             # force the fallback branch
    known = asyncio.run(nodes._dedup_known(ctx, "Survey NFV"))
    assert len(known) == 1 and known[0]["hits"] >= 2
