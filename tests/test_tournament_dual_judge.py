"""Dual-model tournament jury — offline tests for the arbitration contract.

The anti-"one model judges everything" loop: when the context carries a
``judge_llm`` (an independent model family), every match is judged by
BOTH juries concurrently, and

* agreement          -> the Elo moves, rationale notes the agreement;
* disagreement       -> the match is `contested`, NO Elo movement, and a
                        ``tournament_contested`` journal event for the owner;
* judge2 failure/brêk -> the primary verdict stands, noted in the rationale.

All offline — both juries are scripted fakes, memory is a real
SharedMemory over a temp database.
"""

import asyncio
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agentics.config import Settings  # noqa: E402
from agentics.coscientist import tournament as tour  # noqa: E402
from agentics.memory.shared_memory import SharedMemory  # noqa: E402


def _memory() -> SharedMemory:
    s = Settings()
    s.data_dir = Path(tempfile.mkdtemp(prefix="dualjudge-"))
    s.ensure_dirs()
    return SharedMemory(s.db_path)


class FakeLLM:
    """Scripted jury: pops one reply per ask, records everything."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []
        self.purposes = []

    async def ask(self, prompt, mode="concise", purpose="general", system=None):
        self.prompts.append(prompt)
        self.purposes.append(purpose)
        if not self.replies:
            raise RuntimeError("fake jury exhausted")
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(ok=True, answer=item, citations=[])


def _ctx(memory, llm, judge=None):
    return SimpleNamespace(agent_id="a01", memory=memory, llm=llm,
                           judge_llm=judge, settings=Settings(),
                           budget=None, meta={})


def _hyps(memory, claims, agent="a01"):
    ids = []
    for c in claims:
        ids.append(memory.add_hypothesis(agent, c, {}))
    return ids


def _pool(memory, ids):
    rows = {h["id"]: h for h in memory.tournament_pool()}
    return [rows[i] for i in ids]


def test_dual_judge_agreement_moves_elo_and_notes_it():
    mem = _memory()
    ids = _hyps(mem, ["Alpha hypothesis", "Beta hypothesis"])
    primary = FakeLLM(["Panel agrees.\nbetter idea: 1"])
    second = FakeLLM(["Independent check.\nbetter idea: 1"])
    res = asyncio.run(tour.run_match(_ctx(mem, primary, judge=second),
                                     _pool(mem, ids)[0], _pool(mem, ids)[1],
                                     "goal"))
    assert res["mode"] in ("debate", "pairwise") and res["winner"] == "a"
    # both juries were asked CONCURRENTLY with the tournament purposes
    assert primary.purposes == ["tournament"]
    assert second.purposes == ["tournament_judge2"]
    # Elo actually moved (zero-sum +16/-16 at K=32)
    pool = {h["id"]: h for h in mem.tournament_pool()}
    a_elo = pool[ids[0]]["elo"]
    assert abs(a_elo - 1216.0) < 1e-6
    row = mem.recent_tournament_matches(5)[0]
    assert "[dual-judge agreed]" in row["rationale"]


def test_dual_judge_disagreement_records_contested_no_elo():
    mem = _memory()
    ids = _hyps(mem, ["Alpha hypothesis", "Beta hypothesis"])
    primary = FakeLLM(["Primary panel.\nbetter idea: 1"])
    second = FakeLLM(["I disagree.\nbetter idea: 2"])
    res = asyncio.run(tour.run_match(_ctx(mem, primary, judge=second),
                                     _pool(mem, ids)[0], _pool(mem, ids)[1],
                                     "goal"))
    assert res["mode"] == "contested" and res["winner"] == ""
    assert res["primary"] == 1 and res["second"] == 2
    # NO Elo movement: the pool still shows the initial rating everywhere
    pool = {h["id"]: h for h in mem.tournament_pool()}
    assert pool[ids[0]]["elo"] == 1200.0 and pool[ids[1]]["elo"] == 1200.0
    row = mem.recent_tournament_matches(5)[0]
    assert row["mode"] == "contested"
    assert "CONTESTED" in row["rationale"] and "primary=h1" in row["rationale"]
    # and the owner got a journal event
    events = mem.journal_events()
    assert any(e["event"] == "tournament_contested" for e in events)


def test_dual_judge_second_failure_degrades_with_note():
    mem = _memory()
    ids = _hyps(mem, ["Alpha hypothesis", "Beta hypothesis"])
    primary = FakeLLM(["Primary panel.\nbetter idea: 2"])
    second = FakeLLM([RuntimeError("session dead")])
    res = asyncio.run(tour.run_match(_ctx(mem, primary, judge=second),
                                     _pool(mem, ids)[0], _pool(mem, ids)[1],
                                     "goal"))
    assert res["winner"] == "b"            # primary verdict stands
    row = mem.recent_tournament_matches(5)[0]
    assert "judge2 unavailable/unparseable" in row["rationale"]
    pool = {h["id"]: h for h in mem.tournament_pool()}
    assert pool[ids[0]]["elo"] == 1184.0   # Elo DID move on the primary


def test_dual_judge_second_unparseable_degrades_with_note():
    mem = _memory()
    ids = _hyps(mem, ["Alpha hypothesis", "Beta hypothesis"])
    primary = FakeLLM(["Primary panel.\nbetter idea: 1"])
    second = FakeLLM(["Both look fine to me."])
    res = asyncio.run(tour.run_match(_ctx(mem, primary, judge=second),
                                     _pool(mem, ids)[0], _pool(mem, ids)[1],
                                     "goal"))
    assert res["winner"] == "a"
    row = mem.recent_tournament_matches(5)[0]
    assert "judge2 unavailable/unparseable" in row["rationale"]


def test_single_jury_unchanged_without_judge_llm():
    mem = _memory()
    ids = _hyps(mem, ["Alpha hypothesis", "Beta hypothesis"])
    primary = FakeLLM(["Solo panel.\nbetter idea: 1"])
    res = asyncio.run(tour.run_match(_ctx(mem, primary),   # judge_llm=None
                                     _pool(mem, ids)[0], _pool(mem, ids)[1],
                                     "goal"))
    assert res["winner"] == "a"
    row = mem.recent_tournament_matches(5)[0]
    assert "dual-judge" not in row["rationale"]


def test_same_token_juries_run_sequentially_but_both_asked():
    """One-token lab: no session-slot race — back-to-back asks, same
    arbitration contract (agreement -> Elo moves)."""
    mem = _memory()
    ids = _hyps(mem, ["Alpha hypothesis", "Beta hypothesis"])
    primary = FakeLLM(["Primary.\nbetter idea: 1"])
    second = FakeLLM(["Second.\nbetter idea: 1"])
    primary.api_key = "tok-a"
    second.api_key = "tok-a"          # SAME token -> serialized branch
    res = asyncio.run(tour.run_match(_ctx(mem, primary, judge=second),
                                     _pool(mem, ids)[0], _pool(mem, ids)[1],
                                     "goal"))
    assert res["winner"] == "a"
    assert primary.purposes == ["tournament"]
    assert second.purposes == ["tournament_judge2"]
    row = mem.recent_tournament_matches(5)[0]
    assert "[dual-judge agreed]" in row["rationale"]
