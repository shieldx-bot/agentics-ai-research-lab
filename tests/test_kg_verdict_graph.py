"""GraphRAG verdict layer (roadmap #4) — multi-hop context, P16 trust
provenance and semantic contradiction detection. Hermetic: no LLM, plain
token overlap and BFS over the seeded kg_entities/kg_edges tables.
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agentics.memory.shared_memory import SharedMemory  # noqa: E402


@pytest.fixture()
def mem(tmp_path):
    return SharedMemory(tmp_path / "kg.db")


def _seed_graph(mem):
    mem.kg_entity("lstm", kind="method")
    mem.kg_entity("arima", kind="method")
    mem.kg_entity("transformer", kind="method")
    mem.kg_edge("lstm", "arima", "outperforms")
    mem.kg_edge("transformer", "lstm", "outperforms")


# --------------------------------------------------------------- multi-hop
def test_neighborhood_renders_hops(mem):
    _seed_graph(mem)
    block = mem.kg_neighborhood(["lstm"], hops=2)
    lines = block.splitlines()
    assert any(l.startswith("h1:") and "lstm" in l for l in lines)
    assert any(l.startswith("h2:") and "transformer" in l for l in lines)


def test_neighborhood_empty_graph_is_empty_string(mem):
    assert mem.kg_neighborhood(["lstm"]) == ""
    assert mem.kg_neighborhood([]) == ""


def test_neighborhood_seed_terms_from_text(mem):
    _seed_graph(mem)
    terms = mem.kg_seed_terms_from_text("Does the LSTM still beat baselines?")
    assert terms == ["lstm"]
    assert "h1:" in mem.kg_neighborhood_from_text(
        "Does the LSTM still beat baselines?")


# ------------------------------------------------------- P16 trust layer
def test_refuted_settle_marks_edges_untrusted(mem):
    _seed_graph(mem)
    hyp = mem.add_hypothesis("a01", "lstm outperforms arima on mae",
                             {"metric": "mae", "op": "<", "target": 1})
    mem.kg_edge("lstm", "arima", "outperforms", source_hyp=hyp)
    assert mem.kg_trust_report()["untrusted"] == 0
    mem.update_hypothesis(hyp, "refuted", note="machine said no")
    tr = mem.kg_trust_report()
    # 3 edges total: 2 provenance-free stay trusted, the refuted-source dies
    assert tr["edges"] == 3 and tr["untrusted"] == 1 and tr["trusted"] == 2
    # trust filter keeps the refuted-source edge out of the served context:
    # hops=1 from lstm sees transformer->lstm + the provenance-free lstm->arima
    assert mem.kg_neighborhood(["lstm"], hops=1, trust_only=True) \
        .count("outperforms") == 2
    assert mem.kg_neighborhood(["lstm"], hops=1, trust_only=False) \
        .count("outperforms") == 3


def test_supported_settle_keeps_edges_trusted(mem):
    _seed_graph(mem)
    hyp = mem.add_hypothesis("a01", "lstm outperforms arima",
                             {"metric": "mae", "op": "<", "target": 1})
    mem.kg_edge("lstm", "arima", "outperforms", source_hyp=hyp)
    mem.update_hypothesis(hyp, "supported", note="confirmed")
    assert mem.kg_trust_report()["untrusted"] == 0


# --------------------------------------------------- semantic contradictions
def test_contradictions_catch_opposite_verdicts(mem):
    a = mem.add_hypothesis("a01", "dropout improves lstm accuracy on mae",
                           {"metric": "mae", "op": "<", "target": 1})
    b = mem.add_hypothesis("a02", "dropout degrades lstm accuracy on mae badly",
                           {"metric": "rmse", "op": ">", "target": 2})
    mem.update_hypothesis(a, "supported", note="ok")
    mem.update_hypothesis(b, "refuted", note="no")
    pairs = mem.kg_contradictions(threshold=0.4)
    ids = {(p["a"], p["b"]) for p in pairs} | {(p["b"], p["a"]) for p in pairs}
    assert (a, b) in ids


def test_contradictions_ignore_same_verdict_pairs(mem):
    a = mem.add_hypothesis("a01", "dropout improves lstm accuracy on mae",
                           {"metric": "mae", "op": "<", "target": 1})
    b = mem.add_hypothesis("a02", "dropout improves lstm accuracy on mae greatly",
                           {"metric": "rmse", "op": ">", "target": 2})
    mem.update_hypothesis(a, "supported", note="ok")
    mem.update_hypothesis(b, "supported", note="ok")
    assert mem.kg_contradictions(threshold=0.4) == []


def test_contradictions_empty_ledger(mem):
    assert mem.kg_contradictions() == []


# ------------------------------------------------------------------- CLI
def test_cli_kg_query_and_contradictions(monkeypatch, tmp_path):
    from typer.testing import CliRunner
    from agentics import cli
    from agentics.config import Settings

    s = Settings(data_dir=tmp_path / "data")
    s.ensure_dirs()
    mem = SharedMemory(s.db_path)
    _seed_graph(mem)
    a = mem.add_hypothesis("a01", "dropout improves lstm accuracy on mae",
                           {"metric": "mae", "op": "<", "target": 1})
    b = mem.add_hypothesis("a02", "dropout degrades lstm accuracy on mae badly",
                           {"metric": "rmse", "op": ">", "target": 2})
    mem.update_hypothesis(a, "supported", note="ok")
    mem.update_hypothesis(b, "refuted", note="no")
    mem.close()
    monkeypatch.setattr(cli, "_settings", lambda: s)
    runner = CliRunner()
    r1 = runner.invoke(cli.app, ["kg-query", "lstm", "--hops", "2"])
    assert r1.exit_code == 0, r1.output
    assert "h1:" in r1.output and "untrusted" in r1.output
    r2 = runner.invoke(cli.app, ["kg-contradictions", "--threshold", "0.4"])
    assert r2.exit_code == 0, r2.output
    assert "h#a" in r2.output or "Không tìm thấy" not in r2.output


def test_cli_kg_query_empty_graph(monkeypatch, tmp_path):
    from typer.testing import CliRunner
    from agentics import cli
    from agentics.config import Settings

    s = Settings(data_dir=tmp_path / "data")
    s.ensure_dirs()
    monkeypatch.setattr(cli, "_settings", lambda: s)
    result = CliRunner().invoke(cli.app, ["kg-query", "nothing"])
    assert result.exit_code == 0, result.output
    assert "Đồ thị trống" in result.output
