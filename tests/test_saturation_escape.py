"""Saturation escape hatches + award TTL (owner-dashboard round 2026-11-14).

Live findings the owner's dashboard exposed, one fix per test:

1. The gate rejected EVERY gap proposal while the topic was saturated
   (weak=-25 missed the repair band by 5 points) — the lab oscillated
   skip->reject->skip forever and the whole settle/replication/discovery
   pipeline starved on 0 registered hypotheses. Fix: under saturation the
   repair band widens (`idea_repair_margin_saturation`), and when even the
   gap pivot fails the lab PRE-REGISTERS quantitative claims extracted
   from its own stored findings (`_seed_hypotheses_from_findings`).
2. Contract-net re-awarded task #4 to the same coder EVERY scan forever
   (the awarded agent never claimed it). Fix: award TTL — a (task, agent)
   pair re-awarded `coord_award_ttl` consecutive scans without a claim is
   dropped from the allocation and journalled (`award_ttl_expired`).
3. The claim digest printed `claimed ''` when the node ended with
   "no unclaimed tasks; resting" — a lie the owner reads.

All offline — no HTTP, no sessions, no git.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agentics.coordination.allocation import recommend_awards  # noqa: E402
from agentics.graph import nodes  # noqa: E402
from agentics.graph.research_graph import _node_digest  # noqa: E402
from agentics.memory.shared_memory import SharedMemory  # noqa: E402


@pytest.fixture()
def mem(tmp_path: Path):
    m = SharedMemory(tmp_path / "m.db")
    yield m
    m.close()


def _beat(m: SharedMemory, agent: str, role: str) -> None:
    m.heartbeat(agent, role, cycle=1, status="idle")


def _hist(m: SharedMemory, agent: str, category: str, outcome: str) -> None:
    m.add_trace(agent, "coder", 1, outcome, category=category)


def _hyp(m: SharedMemory, claim: str, status: str = "proposed",
         code: str = "") -> int:
    return m.add_hypothesis("a02", claim,
                            {"metric": "m", "op": ">=", "target": 1},
                            status=status, protocol_code=code)


# ------------------------------------------------------------ harness bits

class _S:
    research_topic = "NFV"
    diversity_jaccard_max = 0.8
    ideation_enabled = True
    idea_min_score = 60
    idea_repair_margin = 20
    idea_repair_margin_saturation = 30
    idea_max_children = 2
    idea_max_attempts = 2
    world_context_notes = 12
    verify_protocol_repair_rounds = 0
    settle_require_anchor = False
    discovery_second_judge = False
    coord_award_ttl = 3


class FakeWS:
    def __init__(self, root: Path) -> None:
        self.path = root

    def sync_main(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)


class FakeLLM:
    """Purpose-routed fake: answers are (marker, payload) pairs matched
    against the prompt body; the FIRST match is consumed (pop) so repeated
    markers (judge #1 vs judge #2) can return different payloads."""

    account = SimpleNamespace(name="offline")

    def __init__(self, *answers: tuple[str, str]) -> None:
        self.answers = list(answers)
        self.calls: list[tuple[str, str]] = []

    async def ask(self, prompt, *, mode="concise", system=None,
                  purpose="general"):
        self.calls.append((purpose, prompt))
        for i, (marker, payload) in enumerate(self.answers):
            if marker in prompt:
                del self.answers[i]
                return SimpleNamespace(answer=payload, citations=[], ok=True)
        return SimpleNamespace(answer="", citations=[], ok=False)


def _ctx(m: SharedMemory, tmp_path, llm, vector=None):
    from agentics.agents.roles import Role

    return SimpleNamespace(
        agent_id="a02", role=Role.RESEARCHER, memory=m, llm=llm,
        workspace=FakeWS(tmp_path / "ws"),
        cycle=1, settings=_S(), vector=vector, budget=None,
        heartbeat=lambda *a, **k: None, report=lambda **kw: None,
        meta={"_system_prompt": "sys"},
    )


# ------------------------------------------------- award TTL (fix 3)

def _award_hits(m: SharedMemory) -> int:
    """The (single) award row's TTL counter, read from the table itself."""
    rows = m.task_awards_rows()
    return rows[0]["hits"] if rows else 0


def test_award_ttl_drops_unclaimed_pair(mem):
    mem.create_task("self-evolution: refine agent prompts", category="evolution")
    _beat(mem, "a01", "coder")
    _hist(mem, "a01", "evolution", "merged")

    first = recommend_awards(mem, ttl=3)
    assert len(first) == 1 and "hits" not in first[0]   # public contract intact
    assert _award_hits(mem) == 1
    assert recommend_awards(mem, ttl=3)
    assert _award_hits(mem) == 2                 # consecutive unclaimed scans
    assert recommend_awards(mem, ttl=3)
    assert _award_hits(mem) == 3
    # fourth scan: the pair exhausted its TTL — dropped, table empty
    fourth = recommend_awards(mem, ttl=3)
    assert fourth == []
    assert mem.awards_for("a01") == {}
    events = [e["event"] for e in mem.recent_journal(limit=10)]
    assert "award_ttl_expired" in events
    # a later scan re-awards the pair FRESH (hits restarts at 1): the TTL is
    # a breathing window, not a permanent blacklist
    fifth = recommend_awards(mem, ttl=3)
    assert len(fifth) == 1 and _award_hits(mem) == 1


def test_award_ttl_zero_disables(mem):
    mem.create_task("self-evolution: refine agent prompts", category="evolution")
    _beat(mem, "a01", "coder")
    _hist(mem, "a01", "evolution", "merged")
    for _ in range(6):
        awards = recommend_awards(mem, ttl=0)
        assert len(awards) == 1                   # re-award forever, pre-TTL


def test_award_ttl_counts_per_pair_and_resets_for_new_agent(mem):
    """The hits counter is keyed on the (task, agent) PAIR: the same pair
    re-awarded scan after scan accumulates hits; whenever the award moves
    to a DIFFERENT agent, its counter starts fresh at 1."""
    mem.create_task("self-evolution: refine agent prompts", category="evolution")
    _beat(mem, "a01", "coder")
    _beat(mem, "a02", "coder")
    for _ in range(3):
        _hist(mem, "a01", "evolution", "merged")
    winner = recommend_awards(mem, ttl=3)[0]
    assert _award_hits(mem) == 1
    # identical state -> identical winner (deterministic allocation), the
    # SAME pair accumulates: hits 1 -> 2 -> 3
    winner2 = recommend_awards(mem, ttl=3)[0]
    assert (winner2["task_id"], winner2["agent"]) \
        == (winner["task_id"], winner["agent"])
    assert _award_hits(mem) == 2
    assert recommend_awards(mem, ttl=3)
    assert _award_hits(mem) == 3
    assert recommend_awards(mem, ttl=3) == []     # TTL exhausted
    # after the drop, a fresh scan re-awards with a clean counter
    assert recommend_awards(mem, ttl=3)
    assert _award_hits(mem) == 1


# ----------------------------- gate saturation repair band (fix 2)

# weakest axis N=38 -> margin 38-60 = -22: REJECT at the default band (20),
# REPAIR at the saturation band (30).
BOUNDARY_JUDGE = json.dumps({"non_obviousness": 38, "clarity": 65,
                             "feasibility": 60, "soundness": 6,
                             "notes": "plausible but incremental"})
REPAIRED = json.dumps({
    "claim": "an adversarially-perturbed VNF forecaster with online "
             "conformal recalibration cuts saturation misses",
    "metric": "miss_rate", "op": "<", "target": 0.2})
GOOD_JUDGE = json.dumps({"non_obviousness": 80, "clarity": 75,
                         "feasibility": 70, "soundness": 8,
                         "notes": "sharpened mechanism, measurable effect"})


def _child():
    return {"claim": "an online residual-based conformal VNF resource "
                     "forecaster reduces saturation misses",
            "metric": "miss_rate", "op": "<", "target": 0.25,
            "signature": {"problem": "VNF saturation", "mechanism": "forecasting",
                          "novelty": "conformal trigger", "assumptions": "drift",
                          "effect": "fewer misses"}}


def _events(m: SharedMemory, limit: int = 10) -> list[str]:
    return [e["event"] for e in m.recent_journal(limit=limit, agent="a02")]


def test_gate_rejects_boundary_idea_at_default_band(mem, tmp_path):
    from agentics.graph.nodes import _gate_and_register

    llm = FakeLLM(("ROLE: judge", BOUNDARY_JUDGE))
    ctx = _ctx(mem, tmp_path, llm)
    hyp_id = asyncio.run(_gate_and_register(ctx, _child(), [], []))
    assert hyp_id is None                         # weak=-22 < -20: outright
    assert "idea_rejected" in _events(mem)


def test_gate_repairs_boundary_idea_under_saturation_band(mem, tmp_path):
    from agentics.graph.nodes import _gate_and_register

    llm = FakeLLM(("ROLE: judge", BOUNDARY_JUDGE),
                  ("ROLE: repair", REPAIRED),
                  ("ROLE: judge", GOOD_JUDGE))
    ctx = _ctx(mem, tmp_path, llm)
    hyp_id = asyncio.run(
        _gate_and_register(ctx, _child(), [], [], repair_margin=30.0))
    assert hyp_id is not None                     # one repair round granted
    row = mem.get_hypothesis(hyp_id)
    assert row["status"] == "proposed"
    # the registered claim is the REPAIRED one, not the draft
    assert "conformal recalibration" in row["claim"]
    assert "idea_repaired" in _events(mem)


def test_repaired_idea_still_needs_full_pass(mem, tmp_path):
    """The wider band grants the REPAIR ATTEMPT only: a repaired child the
    re-judge still scores below tau is rejected — the bar for REGISTRATION
    never moves."""
    from agentics.graph.nodes import _gate_and_register

    llm = FakeLLM(("ROLE: judge", BOUNDARY_JUDGE),
                  ("ROLE: repair", REPAIRED),
                  ("ROLE: judge", BOUNDARY_JUDGE))
    ctx = _ctx(mem, tmp_path, llm)
    hyp_id = asyncio.run(
        _gate_and_register(ctx, _child(), [], [], repair_margin=30.0))
    assert hyp_id is None
    assert "idea_repaired" not in _events(mem)
    assert "idea_rejected" in _events(mem)


def test_gap_proposal_uses_the_saturation_band(mem, tmp_path):
    """_propose_gap_hypothesis passes the WIDER margin: a boundary idea the
    default band would kill gets repaired and registered."""
    from agentics.graph.nodes import _propose_gap_hypothesis

    llm = FakeLLM(("ROLE: gap-proposer", json.dumps({"hypotheses": [_child()]})),
                  ("ROLE: judge", BOUNDARY_JUDGE),
                  ("ROLE: repair", REPAIRED),
                  ("ROLE: judge", GOOD_JUDGE))
    ctx = _ctx(mem, tmp_path, llm)
    state: dict = {}
    assert asyncio.run(_propose_gap_hypothesis(ctx, state)) is True
    row = mem.get_hypothesis(1)
    assert row is not None and "conformal recalibration" in row["claim"]
    assert "saturation pivot" in (state.get("summary") or "")


# ------------------------- findings seeding (fix 1: P16bis raw material)

HYP_SPEC = json.dumps({"hypotheses": [{
    "claim": "on synthetic VNF telemetry, lagged cross-resource features "
             "lift CPU-saturation recall over a single-resource baseline",
    "metric": "recall", "op": ">=", "target": 0.85,
    "protocol": "print('AGENTICS_RESULT: {\"recall\": 0.9}')",
    "signature": {"problem": "VNF saturation", "mechanism": "forecasting",
                  "novelty": "cross-resource lags", "assumptions": "stationary",
                  "effect": "earlier prediction"},
}]})

HYP_SPEC_2 = json.dumps({"hypotheses": [{
    "claim": "a queueing-model simulation bounds VNF autoscaler reaction "
             "delay under burst arrivals",
    "metric": "reaction_delay_s", "op": "<", "target": 5.0,
    "protocol": "print('AGENTICS_RESULT: {\"reaction_delay_s\": 3.1}')",
}]})


def test_seed_hypotheses_from_findings_registers_p16bis(mem, tmp_path):
    from agentics.graph.nodes import _seed_hypotheses_from_findings

    mem.add_finding("a03", title="Survey of VNF prediction",
                    content="ml methods for resource forecasting in NFV")
    llm = FakeLLM(("Extract up to 2 QUANTITATIVE", HYP_SPEC))
    ctx = _ctx(mem, tmp_path, llm)
    state = {"issue_number": 1}
    assert asyncio.run(_seed_hypotheses_from_findings(ctx, state)) is True
    row = mem.get_hypothesis(1)
    assert row["status"] == "proposed"            # pre-registered, NOT run
    assert row["protocol_code"]                   # runnable next cycle
    assert row["finding_id"] == 1                 # provenance to the finding
    assert row["criterion"]["target"] == 0.85     # locked before any run
    assert "findings_seeded_hypotheses" in _events(mem)
    assert "pre-registered 1" in state["summary"]


def test_seed_skips_near_duplicate_claims(mem, tmp_path):
    from agentics.graph.nodes import _seed_hypotheses_from_findings

    mem.add_finding("a03", title="Survey of VNF prediction",
                    content="ml methods for resource forecasting in NFV")
    _hyp(mem, "on synthetic VNF telemetry, lagged cross-resource features "
              "lift CPU-saturation recall over a single-resource baseline")
    llm = FakeLLM(("Extract up to 2 QUANTITATIVE", HYP_SPEC))
    ctx = _ctx(mem, tmp_path, llm)
    assert asyncio.run(_seed_hypotheses_from_findings(ctx, {})) is False
    # nothing new registered — the stream is not polluted with clones
    with mem._lock:
        n = mem._conn.execute(
            "SELECT COUNT(*) FROM hypotheses").fetchone()[0]
    assert n == 1


def test_seed_without_findings_is_a_noop(mem, tmp_path):
    from agentics.graph.nodes import _seed_hypotheses_from_findings

    llm = FakeLLM()
    ctx = _ctx(mem, tmp_path, llm)
    assert asyncio.run(_seed_hypotheses_from_findings(ctx, {})) is False
    assert llm.calls == []                        # no LLM call burned


# ------------------- full saturation path in node_research (fixes 1+2)

def test_node_research_saturation_seeds_findings(tmp_path):
    """Topic saturated (offline keyword fallback: 3 findings share the
    topic's words) + registry empty + gap pivot refused (the repaired
    round has no answer to give) -> the findings seeding registers
    hypotheses INSTEAD of the dead 'skipping' line."""
    m = SharedMemory(tmp_path / "node.db")
    for title in ("VNF prediction early survey",
                  "VNF prediction early benchmark",
                  "VNF prediction early methods"):
        m.add_finding("a03", title=title, content="prior angle research")
    llm = FakeLLM(
        # gap pivot: gate puts the boundary idea into repair, and the fake
        # has NO "ROLE: repair" answer -> repair produces nothing -> refused
        ("ROLE: gap-proposer", json.dumps({"hypotheses": [_child()]})),
        ("ROLE: judge", BOUNDARY_JUDGE),
        # findings seeding: one spec per finding, PEM-distinct claims
        ("Extract up to 2 QUANTITATIVE", HYP_SPEC),
        ("Extract up to 2 QUANTITATIVE", HYP_SPEC_2),
    )
    vector = SimpleNamespace(available=False)
    ctx = _ctx(m, tmp_path, llm, vector=vector)
    out = asyncio.run(nodes.node_research(
        ctx, {"issue_title": "VNF prediction early", "issue_body": "",
              "papers": [], "citations": []}))
    assert "pre-registered" in (out.get("summary") or "")
    with m._lock:
        n = m._conn.execute(
            "SELECT COUNT(*) FROM hypotheses WHERE status='proposed'"
        ).fetchone()[0]
    assert n == 2                                 # one per finding
    events = [e["event"] for e in m.recent_journal(limit=15, agent="a02")]
    assert "findings_seeded_hypotheses" in events


# ------------------------------------------------------------- digest fix 4

def test_claim_digest_no_longer_lies_about_resting():
    out = {"summary": "no unclaimed tasks; resting this cycle",
           "action": "claim"}
    digest = _node_digest("claim", out)
    assert "claimed ''" not in digest
    assert "resting this cycle" in digest


def test_claim_digest_still_reports_real_claims():
    out = {"issue_number": 7, "issue_title": "Fix crash", "action": "claim"}
    assert _node_digest("claim", out) == "#7 claimed 'Fix crash'"
