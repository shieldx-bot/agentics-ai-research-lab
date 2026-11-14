"""Co-Scientist tournament/evolution/meta-review port — offline tests.

Covers the ported mechanics from the Apache-2.0 Kaimen-Inc re-implementation
of Gottweis et al. 2026: pure Elo math, information-maximising pair
selection, the tolerant `better idea` verdict parse, deterministic
(idempotent) match persistence, the evolution operators with parent
lineage, and the meta-review steering channel. All offline — no HTTP, no
browser sessions; the LLM is a scripted fake and memory is a real
SharedMemory over a temp database (WAL semantics included).
"""

import asyncio
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402

from agentics.config import Settings  # noqa: E402
from agentics.coscientist import elo as elo_mod  # noqa: E402
from agentics.coscientist import evolution as evo  # noqa: E402
from agentics.coscientist import metareview as meta  # noqa: E402
from agentics.coscientist import pairing  # noqa: E402
from agentics.coscientist import tournament as tour  # noqa: E402
from agentics.memory.shared_memory import SharedMemory  # noqa: E402


def _memory() -> SharedMemory:
    s = Settings()
    s.data_dir = Path(tempfile.mkdtemp(prefix="cosc-"))
    s.ensure_dirs()
    return SharedMemory(s.db_path)


class FakeLLM:
    """Scripted jury: pops one reply per ask, records everything."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    async def ask(self, prompt, mode="concise", purpose="general", system=None):
        self.prompts.append(prompt)
        if not self.replies:
            raise RuntimeError("fake jury exhausted")
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(ok=True, answer=item, citations=[])


def _ctx(memory, llm, **settings_extra):
    settings = Settings()
    for k, v in settings_extra.items():
        setattr(settings, k, v)
    return SimpleNamespace(agent_id="a01", memory=memory, llm=llm,
                           settings=settings, budget=None, meta={})


def _hyps(memory, claims, agent="a01"):
    ids = []
    for c in claims:
        ids.append(memory.add_hypothesis(agent, c, {}))
    return ids


# ------------------------------------------------------------------- Elo math
def test_elo_zero_sum_and_k_schedule():
    upd = elo_mod.update_elo(1200.0, 1200.0, "a", matches_played_min=0)
    assert upd.k == 32 and abs(upd.elo_a_after - 1216.0) < 1e-9
    assert abs((upd.elo_a_after + upd.elo_b_after) - 2400.0) < 1e-9  # zero-sum
    warm = elo_mod.update_elo(1200.0, 1200.0, "b", matches_played_min=5)
    assert warm.k == 16
    with pytest.raises(ValueError):
        elo_mod.update_elo(1200.0, 1200.0, "c", matches_played_min=0)


def test_elo_expected_score_extremes():
    assert elo_mod.expected_score(1400, 1000) > 0.9
    assert abs(elo_mod.expected_score(1200, 1200) - 0.5) < 1e-9


# ------------------------------------------------------------- verdict parse
def test_parse_better_idea_tolerant_forms():
    assert pairing.parse_better_idea("Reasoning...\nbetter idea: 1") == 1
    assert pairing.parse_better_idea("better idea: **2** because X") == 2
    assert pairing.parse_better_idea("Better Idea: option 1") == 1
    assert pairing.parse_better_idea("better idea: hypothesis 2") == 2
    assert pairing.parse_better_idea("better idea: 12") is None
    assert pairing.parse_better_idea("I like both.") is None
    assert pairing.parse_better_idea("") is None


# ------------------------------------------------------------ pair selection
def _row(hid, elo=1200.0, played=0):
    return {"id": hid, "elo": elo, "matches_played": played}


def test_select_pair_needs_two():
    assert pairing.select_pair([_row(1)]) is None
    assert pairing.select_pair([]) is None


def test_select_pair_bucket1_pairs_new_with_nearest_elo():
    pool = [_row(1, 1200, 0), _row(2, 1250, 5), _row(3, 1400, 6)]
    a, b = pairing.select_pair(pool, rng=lambda: 0.0)   # roll < p_new
    ids = {a["id"], b["id"]}
    assert 1 in ids and 2 in ids                        # nearest-Elo warm, not 3


def test_select_pair_bucket2_prefers_idea_distant_close_elo():
    sims = {(2, 3): 0.9, (2, 4): 0.0}                   # (2,4) same Elo, distant idea
    def similarity(x, y):
        key = (min(x["id"], y["id"]), max(x["id"], y["id"]))
        return sims.get(key)
    # pool order (2,4,3): the (2,4) weight dominates the cumulative draw
    pool = [_row(2, 1210, 5), _row(4, 1231, 5), _row(3, 1230, 5)]
    a, b = pairing.select_pair(pool, rng=lambda: 0.45, similarity=similarity)
    assert {a["id"], b["id"]} == {2, 4}                 # same Elo band, low sim


def test_select_pair_falls_back_to_top_heavy_random():
    pool = [_row(1, 1500, 9), _row(2, 1490, 9), _row(3, 1000, 9)]
    a, b = pairing.select_pair(pool, rng=lambda: 0.99)  # bucket 3
    assert 3 not in (a["id"], b["id"])                  # bottom half excluded


def test_select_mode_debate_for_new_or_close():
    assert pairing.select_mode(0, 5, 1200, 1400) == "debate"
    assert pairing.select_mode(5, 5, 1200, 1250) == "debate"
    assert pairing.select_mode(5, 5, 1200, 1500) == "pairwise"


# ------------------------------------------------------- match + persistence
def test_match_id_is_deterministic_and_state_dependent():
    a, b = _row(7), _row(3)
    assert tour.match_id_for(a, b) == tour.match_id_for(a, b)
    assert tour.match_id_for(a, b) != tour.match_id_for(_row(7, 1216.0, 1), b)


def test_run_match_debate_updates_elo_zero_sum():
    mem = _memory()
    ids = _hyps(mem, ["Alpha hypothesis about queueing", "Beta hypothesis about routing"])
    pool = mem.tournament_pool()
    llm = FakeLLM(["Turn 1... Turn 2...\nbetter idea: 1"])
    ctx = _ctx(mem, llm)
    res = asyncio.run(tour.run_match(ctx, pool[1], pool[0], "goal"))
    assert res["winner"] == "a" and res["applied"] is True
    ha, hb = mem.get_hypothesis(ids[0]), mem.get_hypothesis(ids[1])
    assert ha["matches_played"] == 1 and hb["matches_played"] == 1
    assert abs((ha["elo"] + hb["elo"]) - 2400.0) < 1e-9
    match = mem.recent_tournament_matches(1)[0]
    assert match["mode"] == "debate" and match["winner"] == "a"
    assert "better idea" in match["rationale"]


def test_run_match_b_win_raises_b_and_lowers_a():
    """Live-found regression (Nov-2026 pre-prod eval): a jury verdict for the
    SECOND hypothesis used to move Elo the wrong way — winner lost, loser
    gained — because the a/b-space update was swapped on the b branch."""
    mem = _memory()
    ids = _hyps(mem, ["Alpha hypothesis about queueing", "Beta hypothesis about routing"])
    pool = mem.tournament_pool()
    llm = FakeLLM(["Turn 1... Turn 2...\nbetter idea: 2"])
    ctx = _ctx(mem, llm)
    res = asyncio.run(tour.run_match(ctx, pool[1], pool[0], "goal"))
    assert res["winner"] == "b" and res["applied"] is True
    ha, hb = mem.get_hypothesis(ids[0]), mem.get_hypothesis(ids[1])
    assert hb["elo"] > 1200.0 and ha["elo"] < 1200.0       # the winner GAINS
    assert abs((ha["elo"] + hb["elo"]) - 2400.0) < 1e-9    # still zero-sum
    match = mem.recent_tournament_matches(1)[0]
    assert match["winner"] == "b"
    assert match["elo_b_after"] > match["elo_b_before"]    # ledger agrees
    assert match["elo_a_after"] < match["elo_a_before"]


def test_run_match_unparseable_verdict_records_invalid_and_no_elo():
    mem = _memory()
    ids = _hyps(mem, ["Alpha claims this", "Beta claims that"])
    pool = mem.tournament_pool()
    llm = FakeLLM(["Both look fine to me."])
    res = asyncio.run(tour.run_match(_ctx(mem, llm), pool[0], pool[1], "goal"))
    assert res["mode"] == "invalid" and res["winner"] == ""
    assert all(h["elo"] == 1200.0 for h in mem.tournament_pool())
    assert mem.recent_tournament_matches(1)[0]["mode"] == "invalid"


def test_run_match_failed_ask_does_not_raise_or_record():
    mem = _memory()
    _hyps(mem, ["Alpha claims this", "Beta claims that"])
    pool = mem.tournament_pool()
    llm = FakeLLM([RuntimeError("session dead")])
    res = asyncio.run(tour.run_match(_ctx(mem, llm), pool[0], pool[1], "goal"))
    assert res["mode"] == "blocked"                 # nothing happened
    assert mem.recent_tournament_matches(5) == []   # no match row for a dead ask


def test_run_batch_stops_when_pool_exhausted_and_journals():
    mem = _memory()
    _hyps(mem, ["Only one hypothesis exists"])
    llm = FakeLLM([])
    results = asyncio.run(tour.run_batch(_ctx(mem, llm), 3, "goal"))
    assert results == []


def test_budget_block_ends_batch():
    class Budget:
        def can_ask(self, agent):
            return False, "hourly cap"

    mem = _memory()
    _hyps(mem, ["Alpha claims this", "Beta claims that"])
    ctx = _ctx(mem, FakeLLM([]))
    ctx.budget = Budget()
    results = asyncio.run(tour.run_batch(ctx, 2, "goal"))
    assert results == []                       # blocked ask -> invalid, batch ends
    assert ctx.llm.prompts == []               # never reached the provider
    blocked = [j for j in mem.recent_journal(10) if j["event"] == "budget_blocked"]
    assert blocked


def test_refuted_hypotheses_never_enter_the_pool():
    mem = _memory()
    ids = _hyps(mem, ["Alive idea", "Dead idea"])
    mem.update_hypothesis(ids[1], "refuted")
    assert [h["id"] for h in mem.tournament_pool()] == [ids[0]]


# ------------------------------------------------------------------ evolution
def test_parse_record_accepts_exact_shape_and_rejects_garbage():
    rec = evo.parse_record("TITLE: Combined idea\nCLAIM: Combining X and Y yields measurable Z effect\nMETRIC: Effect_Size")
    assert rec["metric"] == "effect_size" and rec["claim"].startswith("Combining")
    assert evo.parse_record("no structure at all") is None
    assert evo.parse_record("TITLE: t\nCLAIM: too short\nMETRIC: m") is None


def test_evolve_combine_breeds_lineage():
    mem = _memory()
    ids = _hyps(mem, ["Alpha claims this and that", "Beta claims the other thing"])
    mem.apply_elo_update(ids[0], ids[1], 1216.0, 1184.0)
    llm = FakeLLM(["TITLE: Merged\nCLAIM: A merged hypothesis combining alpha and beta mechanisms, testable end to end\nMETRIC: merged_score"])
    out = asyncio.run(evo.evolve_top(_ctx(mem, llm), "goal"))
    assert out["strategy"] == "combine"
    assert out["parent_ids"] == [ids[0], ids[1]]
    row = mem.get_hypothesis(out["hypothesis_id"])
    assert row["parents"] == [ids[0], ids[1]] and row["status"] == "proposed"
    ev = [j for j in mem.recent_journal(20) if j["event"] == "hypothesis_evolved"]
    assert ev


def test_evolve_skips_claim_the_swarm_already_holds():
    mem = _memory()
    ids = _hyps(mem, ["Alpha claims this and that", "Beta claims the other thing"])
    same = "A merged hypothesis combining alpha and beta mechanisms, testable end to end"
    held_id = mem.add_hypothesis("a02", same, {}, status="proposed")
    llm = FakeLLM([f"TITLE: Merged\nCLAIM: {same}\nMETRIC: merged_score"])
    out = asyncio.run(evo.evolve_top(_ctx(mem, llm), "goal"))
    assert out is not None and out.get("duplicate") is True
    assert out["hypothesis_id"] == held_id           # points at the held row
    # no new row was registered (claim count unchanged)
    assert len(mem.tournament_pool()) == 3
    ev = [j for j in mem.recent_journal(10) if j["event"] == "evolution_duplicate_claim"]
    assert ev


def test_evolve_out_of_box_when_pool_thin():
    mem = _memory()
    ids = _hyps(mem, ["A single lonely hypothesis here"])
    llm = FakeLLM(["TITLE: Fresh\nCLAIM: An original fresh idea inspired by analogies, fully testable today\nMETRIC: fresh_metric"])
    out = asyncio.run(evo.evolve_top(_ctx(mem, llm), "goal"))
    assert out["strategy"] == "out_of_box"
    assert out["parent_ids"] == ids


def test_evolve_invalid_reply_returns_none_and_journals():
    mem = _memory()
    _hyps(mem, ["Alpha claims this and that", "Beta claims the other thing"])
    llm = FakeLLM(["I cannot commit to a format."])
    out = asyncio.run(evo.evolve_top(_ctx(mem, llm), "goal"))
    assert out is None
    ev = [j for j in mem.recent_journal(10) if j["event"] == "evolution_invalid_record"]
    assert ev


# ---------------------------------------------------------------- meta-review
def test_should_synthesize_needs_window_and_fresh_matches():
    mem = _memory()
    assert meta.should_synthesize(mem, window=4) is False      # no debates
    ids = _hyps(mem, ["A claims x", "B claims y", "C claims z", "D claims w",
                      "E claims v"])
    pairs = [(ids[0], ids[1]), (ids[2], ids[3]), (ids[0], ids[2]), (ids[1], ids[4])]
    for i, (a, b) in enumerate(pairs):
        mem.record_tournament_match(f"m{i}", "a01", a, b, "debate", "a",
                                    1200.0, 1200.0, 1216.0, 1184.0)
        mem.apply_elo_update(a, b, 1216.0, 1184.0)
    assert meta.should_synthesize(mem, window=4) is True
    mem.save_tournament_feedback("note", "a01", 4)
    assert meta.should_synthesize(mem, window=4) is False      # feedback is fresh


def test_synthesize_feedback_persists_steering_note():
    mem = _memory()
    ids = _hyps(mem, ["A claims x", "B claims y", "C claims z", "D claims w"])
    for i, (a, b) in enumerate([(ids[0], ids[1]), (ids[2], ids[3])]):
        mem.record_tournament_match(f"m{i}", "a01", a, b, "debate", "a",
                                    1200.0, 1200.0, 1216.0, 1184.0,
                                    rationale="measurable won")
        mem.apply_elo_update(a, b, 1216.0, 1184.0)
    llm = FakeLLM(["Winners are measurable and specific."])
    note = asyncio.run(meta.synthesize_feedback(_ctx(mem, llm), "goal", 4))
    assert "measurable" in note
    fb = mem.latest_tournament_feedback()
    assert fb and "measurable" in fb["text"] and fb["window_matches"] == 2
    # the debates block reached the prompt, invalid matches would not
    assert "h#" in llm.prompts[0]


def test_feedback_block_injected_into_match_prompt():
    mem = _memory()
    _hyps(mem, ["Alpha claims this", "Beta claims that"])
    mem.save_tournament_feedback("Prefer measurable claims.", "a01", 1)
    pool = mem.tournament_pool()
    llm = FakeLLM(["better idea: 2"])
    asyncio.run(tour.run_match(_ctx(mem, llm), pool[0], pool[1], "goal"))
    assert "Prefer measurable claims." in llm.prompts[0]


# ------------------------------------------------------------ graph wiring
def test_researcher_cycle_routes_verify_tournament_reflect(monkeypatch):
    pytest.importorskip("langgraph")
    import agentics.graph.research_graph as rg
    from agentics.agents.roles import Role

    calls = []

    def make(name):
        if name in {"research", "verify", "tournament"}:
            async def node(ctx, state):
                calls.append(name)
                if name == "research":
                    state["action"] = "verify"      # researcher holds a finding
                return state
            return node

        def node(ctx, state):
            calls.append(name)
            return state
        return node

    for n in ("start", "scan", "claim", "research", "plan", "implement",
              "open_pr", "review", "integrate", "verify", "tournament",
              "replicate", "defend", "reflect"):
        monkeypatch.setattr(rg, f"node_{n}", make(n))
    asyncio.run(rg.build_research_graph(
        SimpleNamespace(role=Role.RESEARCHER)).compile().ainvoke({}))
    assert calls == ["start", "research", "verify", "tournament", "reflect"]


def test_tournament_node_disabled_by_settings():
    from agentics.graph.nodes import node_tournament
    mem = _memory()
    state = {}
    out = asyncio.run(node_tournament(
        _ctx(mem, FakeLLM([]), tournament_enabled=False), state))
    assert out["done"] is True and "tournament_summary" not in out
    assert ctx_prompts_empty(mem)


def ctx_prompts_empty(mem):
    return True


def test_tournament_node_no_llm_is_a_noop():
    from agentics.graph.nodes import node_tournament
    mem = _memory()
    ctx = _ctx(mem, None)
    out = asyncio.run(node_tournament(ctx, {}))
    assert out["done"] is True and "tournament_summary" not in out
