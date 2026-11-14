"""Tests for the 3-paper research-loop upgrade (2026-11-14).

- DAPS (arXiv 2609.00077): mechanism taxonomy + entropy audit, PEM
  near-duplicate gate, CCR coverage reweighting.
- IDEAgent (arXiv 2607.22375): hypothesis signature quintuple storage.
- Kosmos (arXiv 2511.02824): world_notes table + bounded prompt rendering.
"""
from __future__ import annotations

import asyncio
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agentics import diversity as qd  # noqa: E402
from agentics.graph.nodes import _pem_gate  # noqa: E402
from agentics.memory.shared_memory import SharedMemory  # noqa: E402


# ---------------------------------------------------------------- taxonomy

def test_classify_maps_mechanism_vocabulary():
    assert qd.classify("An LSTM forecaster predicts VNF load five minutes ahead") \
        == "forecasting"
    assert qd.classify("A reactive threshold autoscaler adds replicas") \
        == "autoscaling"
    assert qd.classify("Discrete-event simulation of a service-function chain") \
        == "simulation"
    assert qd.classify("Complete gibberish zzz qqq") == "other"


def test_entropy_distinguishes_diverse_from_collapsed():
    uniform = ["cat%d" % i for i in range(10)]
    collapsed = ["forecasting"] * 9 + ["benchmark"]
    assert qd.entropy(uniform) > 2.0
    assert qd.entropy(collapsed) < 0.5
    assert qd.entropy([]) == 0.0
    assert qd.entropy(["x"]) == 0.0


def test_near_duplicate_flags_rewording_not_genuine_novelty():
    priors = [{"id": 1, "claim":
               "a linear regression performance model estimates CPU "
               "utilization from packet rate with RMSE under 5 points"}]
    clone = ("a linear regression performance model can estimate VNF CPU "
             "utilization from packet rate with RMSE below 5 points")
    fresh = ("a GNN topology forecaster reduces migration count under bursty "
             "traffic in a three-VNF chain")
    dup = qd.near_duplicate(clone, priors, threshold=0.8)
    assert dup is not None and dup[0] >= 0.8
    assert dup[1]["id"] == 1
    assert qd.near_duplicate(fresh, priors, threshold=0.8) is None


def test_near_duplicate_signature_axes_strengthen_match():
    priors = [{"id": 7, "claim": "short claim",
               "signature": {"problem": "VNF resource prediction",
                             "mechanism": "linear regression on packet rate"}}]
    cand = "VNF resource prediction with linear regression on packet rate"
    dup = qd.near_duplicate(cand, priors, threshold=0.8)
    assert dup is not None


def test_coverage_order_prefers_rare_categories():
    candidates = [
        {"id": 1, "claim": "another LSTM forecasting baseline"},
        {"id": 2, "claim": "reinforcement learning control policy"},
    ]
    recent = ["forecasting claim %d" % i for i in range(9)]
    ordered = qd.coverage_order(candidates, recent)
    assert ordered[0]["id"] == 2            # optimization: unseen category


def test_audit_payload_shape():
    a = qd.audit(["forecast the load", "autoscale the replicas", "other zzz"])
    assert set(a) == {"entropy_nats", "window", "categories"}
    assert a["window"] == 3


# ---------------------------------------------------------------- memory

@pytest.fixture()
def mem(tmp_path):
    return SharedMemory(tmp_path / "m.db")


def _hyp(mem, claim, status="proposed", code=""):
    return mem.add_hypothesis("a02", claim,
                              {"metric": "m", "op": ">=", "target": 1},
                              status=status, protocol_code=code)


def test_signature_roundtrip(mem):
    hid = _hyp(mem, "some claim")
    sig = {"problem": "p", "mechanism": "m", "novelty": "n",
           "assumptions": "a", "effect": "e"}
    assert mem.set_hypothesis_signature(hid, sig) is True
    rows = mem.recent_hypotheses(limit=5)
    assert rows[0]["id"] == hid and rows[0]["signature"]["mechanism"] == "m"
    assert mem.set_hypothesis_signature(999, sig) is False


def test_recent_hypotheses_covers_all_statuses(mem):
    _hyp(mem, "proposed claim", status="proposed")
    _hyp(mem, "supported claim", status="supported", code="print(1)")
    rows = mem.recent_hypotheses(limit=10)
    statuses = {r["status"] for r in rows}
    assert {"proposed", "supported"} <= statuses


def test_world_notes_add_render_bound(mem):
    for i in range(20):
        mem.world_add("a01", "finding", f"finding {i}", source=f"finding#{i}")
    mem.world_add("a02", "gap", "no time-resolved flux data", source="h#3")
    rendered = mem.world_render(limit=5)
    assert "[FINDING]" in rendered and "[GAP]" in rendered
    # newest first, bounded
    assert "finding 19" in rendered
    lines = [ln for ln in rendered.splitlines() if ln.startswith("- ")]
    assert len(lines) == 5
    # kind filter
    assert all(i["kind"] == "gap" for i in mem.world_items(kind="gap"))


def test_world_notes_truncated_text_and_provenance(mem):
    mem.world_add("a01", "verified", "x" * 1000, source="h#" + "9" * 80)
    item = mem.world_items(limit=1)[0]
    assert len(item["text"]) == 400
    assert len(item["source"]) == 60


# ---------------------------------------------------------------- PEM gate

class _Settings:
    diversity_jaccard_max = 0.8


def test_pem_gate_rejects_near_clone_and_journals(tmp_path):
    mem = SharedMemory(tmp_path / "m.db")
    hid = _hyp(mem, "a linear regression model estimates VNF CPU utilization "
                    "from packet rate with held-out RMSE under 5 points")
    ctx = SimpleNamespace(memory=mem, agent_id="a05",
                          settings=_Settings())
    clone = ("a linear regression performance model can estimate VNF CPU "
             "utilization from packet rate, held-out RMSE under 5 points")
    assert _pem_gate(ctx, clone) is False
    events = mem.recent_journal(limit=5, agent="a05")
    assert any(e["event"] == "hypothesis_near_duplicate" for e in events)


def test_pem_gate_allows_fresh_claim(tmp_path):
    mem = SharedMemory(tmp_path / "m.db")
    _hyp(mem, "a linear regression model estimates VNF CPU utilization "
              "from packet rate with held-out RMSE under 5 points")
    ctx = SimpleNamespace(memory=mem, agent_id="a05", settings=_Settings())
    fresh = ("a topology-aware GNN forecaster cuts VNF migration count in "
             "bursty three-VNF chains")
    assert _pem_gate(ctx, fresh) is True


def test_pem_gate_excludes_self(tmp_path):
    mem = SharedMemory(tmp_path / "m.db")
    hid = _hyp(mem, "a linear regression model estimates VNF CPU utilization "
                    "from packet rate with held-out RMSE under 5 points")
    ctx = SimpleNamespace(memory=mem, agent_id="a05", settings=_Settings())
    same = ("a linear regression model estimates VNF CPU utilization from "
            "packet rate with held-out RMSE under 5 points")
    assert _pem_gate(ctx, same, exclude_id=hid) is True


# --------------------------------------------- deep-dive CCR/inherit hooks

def test_deep_dive_helper_inherits_verdict_for_near_clone(tmp_path):
    """A registry proposal that re-words a settled claim must inherit the
    settled verdict (no wasted protocol run) and be journaled as reused."""
    from agentics.graph.nodes import _test_proposed_hypothesis

    mem = SharedMemory(tmp_path / "m.db")
    settled_id = mem.add_hypothesis(
        "a03", "a linear regression model estimates VNF CPU utilization "
               "from packet rate with held-out RMSE under 5 points",
        {"metric": "performance_model_rmse_pct", "op": "<=", "target": 5.0},
        status="supported", protocol_code="print(1)")
    clone_id = mem.add_hypothesis(
        "a04", "a linear regression performance model can estimate VNF CPU "
               "utilization from packet rate, held-out RMSE under 5 points",
        {"metric": "performance_model_rmse_pct", "op": "<=", "target": 5.0},
        status="proposed", protocol_code="")

    class FakeLLM:
        pass

    ctx = SimpleNamespace(llm=FakeLLM(), memory=mem, agent_id="a04",
                          workspace=None, settings=_Settings())
    state: dict = {}
    assert asyncio.run(_test_proposed_hypothesis(ctx, state)) is True
    hyp = mem.get_hypothesis(clone_id)
    assert hyp["status"] == "supported"        # inherited, not re-run
    assert "inherited" in hyp["verdict_note"]
    assert state["summary"].startswith("deep-dive: registry h#")


def test_deep_dive_helper_false_when_registry_empty(tmp_path):
    from agentics.graph.nodes import _test_proposed_hypothesis
    mem = SharedMemory(tmp_path / "m.db")
    ctx = SimpleNamespace(llm=SimpleNamespace(), memory=mem, agent_id="a02",
                          workspace=None, settings=_Settings())
    assert asyncio.run(_test_proposed_hypothesis(ctx, {})) is False
