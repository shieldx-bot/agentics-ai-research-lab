"""Advanced-tech integration round: MAP-Elites QD archive (#2), GraphRAG-lite
(#4) and tournament proximity pairing (#5).

Local-router tests live in test_local_router.py; TTT (PUCT/entropic/reward)
in test_ttt_integration.py.
"""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agentics.memory.shared_memory import SharedMemory  # noqa: E402
from agentics.research import kg_extract  # noqa: E402
from agentics.coscientist import pairing  # noqa: E402


@pytest.fixture()
def mem(tmp_path):
    return SharedMemory(tmp_path / "adv.db")


# ------------------------------------------------- #2 MAP-Elites archive
def test_qd_cell_keeps_the_best_elite(mem):
    mem.qd_record("forecasting", "rejected", None, "weak idea", 0.0)
    mem.qd_record("forecasting", "core", 7, "strong idea", 72.0)
    mem.qd_record("forecasting", "core", 9, "weaker clone", 41.0)
    cells = {c["cell"]: c for c in mem.qd_cells_list()}
    elite = cells["forecasting|core"]
    assert elite["hyp_id"] == 7 and elite["quality"] == 72.0
    # a better newcomer DOES replace the elite
    mem.qd_record("forecasting", "core", 11, "even better", 88.0)
    cells = {c["cell"]: c for c in mem.qd_cells_list()}
    assert cells["forecasting|core"]["hyp_id"] == 11


def test_qd_islands_round_robin_and_underexplored(mem):
    k1 = mem.qd_record("forecasting", "core", 1, "a", 50.0)
    k2 = mem.qd_record("clustering", "core", 2, "b", 40.0)
    assert k1 and k2
    islands = {c["mechanism"]: c["island"] for c in mem.qd_cells_list()}
    assert islands["forecasting"] != islands["clustering"]
    taxonomy = ["forecasting", "clustering", "causal", "optimization"]
    missing = mem.qd_underexplored(taxonomy)
    assert set(missing) == {"causal", "optimization"}


def test_qd_record_never_crashes_on_empty_mechanism(mem):
    key = mem.qd_record("", "", None, "x", 0.0)
    assert key == "unknown|core"


# ----------------------------------------------------- #4 GraphRAG-lite
def test_parse_triples_tolerant():
    ans = 'noise\n[{"src": "LSTM", "rel": "Outperforms", "dst": "ARIMA",\n' \
          '  "kind": "method"}, {"bad": 1}, "junk"]'
    t = kg_extract._parse_triples(ans)
    assert len(t) == 1
    assert t[0] == {"src": "lstm", "rel": "outperforms", "dst": "arima",
                    "kind": "method"}
    assert kg_extract._parse_triples("no json here") == []
    assert kg_extract._parse_triples("") == []


def test_parse_triples_rejects_self_loops_and_empty_names():
    t = kg_extract._parse_triples(
        '[{"src": "x", "rel": "uses", "dst": "x"}, {"src": "", "rel": "a", "dst": "b"}]')
    assert t == []


def _seed_graph(mem):
    mem.kg_entity("lstm", "method")
    mem.kg_entity("arima", "method")
    mem.kg_entity("nfv", "concept")


def test_kg_context_renders_paths(mem):
    _seed_graph(mem)
    assert mem.kg_edge("lstm", "arima", "outperforms")
    block = mem.kg_context(["lstm"])
    assert "--outperforms-->" in block
    assert "lstm" in block and "arima" in block
    # unrelated term -> empty (callers skip the block)
    assert mem.kg_context(["quantum"]) == ""
    assert mem.kg_context([]) == ""


def test_kg_ingest_requires_local_runtime(mem, monkeypatch):
    settings = SimpleNamespace(local_llm_enabled=False, local_llm_url="",
                               local_llm_model="")
    assert asyncio.run(kg_extract.ingest_text(
        mem, "some finding text", settings)) == 0
    # enabled but runtime absent -> still 0, no exception
    settings2 = SimpleNamespace(local_llm_enabled=True,
                                local_llm_url="http://127.0.0.1:11434",
                                local_llm_model="qwen3:1.7b",
                                local_llm_timeout=1.0)
    assert asyncio.run(kg_extract.ingest_text(
        mem, "text", settings2)) == 0


def test_kg_ingest_stores_llm_triples(mem, monkeypatch):
    settings = SimpleNamespace(local_llm_enabled=True,
                               local_llm_url="http://x", local_llm_model="m",
                               local_llm_timeout=1.0)

    class FakeRouter:
        @staticmethod
        async def available(settings=None):
            return True

        @staticmethod
        async def ask(prompt, settings=None, purpose="local", system=""):
            return SimpleNamespace(ok=True, answer=(
                '[{"src": "transformer", "rel": "outperforms", "dst": "lstm",'
                ' "kind": "method"}]'))

    added = asyncio.run(kg_extract.ingest_text(
        mem, "finding body", settings, router=FakeRouter))
    assert added == 1
    block = mem.kg_context(["transformer"])
    assert "transformer" in block and "lstm" in block


# ----------------------------------------- #5 tournament proximity pairing
def test_jaccard_similarity_flags_clones():
    a = {"claim": "gradient boosting improves vnf anomaly detection accuracy",
         "signature": {"mechanism": "forecasting"}}
    clone = {"claim": "gradient boosting improves anomaly detection accuracy",
             "signature": {"mechanism": "forecasting"}}
    other = {"claim": "graph neural networks for network slicing latency",
             "signature": {"mechanism": "gnn"}}
    assert pairing.jaccard_similarity(a, clone) > 0.9
    assert pairing.jaccard_similarity(a, other) < 0.5
    assert pairing.jaccard_similarity({"claim": ""}, a) == 0.0


def test_bucket1_skips_near_identical_opponent():
    new_h = {"id": 1, "elo": 1200.0, "matches_played": 0,
             "claim": "gradient boosting improves vnf anomaly detection",
             "signature": {"mechanism": "forecasting"}}
    clone = {"id": 2, "elo": 1200.0, "matches_played": 9,
             "claim": "gradient boosting improves vnf anomaly detection more",
             "signature": {"mechanism": "forecasting"}}
    distinct = {"id": 3, "elo": 1210.0, "matches_played": 9,
                "claim": "boolean satisfiability bounds on chip layouts",
                "signature": {"mechanism": "optimization"}}
    rng = __import__("random").Random(7)
    pair = pairing.select_pair([new_h, clone, distinct], rng=rng,
                               p_new=1.0, p_close=0.0,
                               similarity=pairing.jaccard_similarity)
    assert pair is not None
    ids = {pair[0]["id"], pair[1]["id"]}
    assert 1 in ids and 2 not in ids       # the clone was skipped


def test_pairing_degrades_without_similarity():
    pool = [{"id": i, "elo": 1200.0 + i, "matches_played": 5} for i in (1, 2)]
    rng = __import__("random").Random(3)
    assert pairing.select_pair(pool, rng=rng) is not None
    assert pairing.select_pair(pool[:1], rng=rng) is None
