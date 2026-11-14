"""Wave-2 #3 — EIG frontier (Bayesian experimental design).

Contract under test:
* closed-form Bernoulli EIG peaks at the 50/50 prior and vanishes on a
  saturated one (a metric answered the same way 20x buys nothing);
* priors come from the experiment ledger per metric; unknown metric gets
  the Laplace prior (maximum uncertainty -> maximum information);
* cost punishes retests (attempts raise the denominator);
* the PUCT frontier accepts eig_weight and reorders accordingly; 0 restores
  pure PUCT.
"""

import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agentics.evolution.eig import (  # noqa: E402
    bernoulli_eig, binary_entropy, eig_score, metric_priors)
from agentics.memory.shared_memory import SharedMemory  # noqa: E402

LN2 = math.log(2.0)


# ------------------------------------------------------------------- math
def test_binary_entropy_basic():
    assert binary_entropy(0.0) == 0.0 and binary_entropy(1.0) == 0.0
    assert abs(binary_entropy(0.5) - LN2) < 1e-9
    assert 0.0 < binary_entropy(0.9) < LN2


def test_eig_peaks_at_uncertain_prior():
    eig_laplace = bernoulli_eig(1, 1)           # Laplace: max uncertainty
    eig_sure = bernoulli_eig(21, 1)             # ledger says ~always yes
    eig_tight_balanced = bernoulli_eig(21, 21)  # well-estimated 50/50
    assert 0.0 < eig_laplace <= 0.2             # ~0.19 nats, never ln2
    assert eig_sure < eig_laplace * 0.5         # sure-things buy little
    assert eig_tight_balanced < eig_laplace * 0.2   # pinned metrics too


def test_eig_never_negative():
    assert bernoulli_eig(1, 1) >= 0.0
    assert bernoulli_eig(2, 5) >= 0.0
    assert bernoulli_eig(50, 2) >= 0.0


# ------------------------------------------------------------------ priors
def test_priors_from_ledger(tmp_path):
    m = SharedMemory(tmp_path / "eig.db")
    hid = m.add_hypothesis("a01", "c1", {"metric": "acc", "op": ">=",
                                         "target": 0.5})
    m.record_experiment("a01", hid, "supported", metric="acc")
    m.record_experiment("a01", hid, "refuted", metric="acc")
    m.record_experiment("a01", hid, "inconclusive", metric="acc")
    m.record_experiment("a01", hid, "supported", metric="latency")
    priors = metric_priors(m)
    assert priors["acc"] == (1, 1)              # inconclusive carries nothing
    assert priors["latency"] == (1, 0)
    assert "unknown_metric" not in priors


# ------------------------------------------------------------------ scoring
def test_eig_score_unknown_metric_uses_laplace(tmp_path):
    m = SharedMemory(tmp_path / "eig2.db")
    h = {"criterion": '{"metric": "never_seen"}', "attempts": 0}
    s = eig_score(m, h)
    assert s["prior_alpha"] == 1.0 and s["prior_beta"] == 1.0
    assert s["eig_score"] > 0.2                 # max uncertainty, max gain


def test_eig_score_cost_punishes_retests(tmp_path):
    m = SharedMemory(tmp_path / "eig3.db")
    fresh = eig_score(m, {"criterion": '{"metric": "acc"}', "attempts": 0})
    retested = eig_score(m, {"criterion": '{"metric": "acc"}',
                             "attempts": 4})
    assert retested["eig_score"] < fresh["eig_score"]


def test_eig_score_saturated_metric_buys_nothing(tmp_path):
    m = SharedMemory(tmp_path / "eig4.db")
    hid = m.add_hypothesis("a01", "c", {"metric": "acc", "op": ">=",
                                        "target": 0.5})
    for _ in range(20):
        m.record_experiment("a01", hid, "supported", metric="acc")
    s = eig_score(m, {"criterion": '{"metric": "acc"}', "attempts": 0})
    assert s["eig_score"] < 0.15


# ------------------------------------------------------- PUCT integration
def _discovery_branch(tmp_path, metric):
    """One discovery root + one proposed branch child with a criterion."""
    m = SharedMemory(tmp_path / f"puct_{metric}.db")
    root = m.add_hypothesis("a01", "root claim", {"metric": metric,
                                                  "op": ">=", "target": 0.5})
    m.mark_discovery(root, 85.0)
    child = m.add_hypothesis("a02", "branch claim", {"metric": metric,
                                                     "op": ">=",
                                                     "target": 0.9},
                             parents=[root], depth=1)
    return m, root, child


def test_puct_frontier_accepts_eig_weight(tmp_path):
    m, root, child = _discovery_branch(tmp_path, "fresh_metric_1")
    frontier = m.discovery_frontier_puct(limit=3, max_depth=4, c=1.0,
                                         eig_weight=0.5)
    assert len(frontier) == 1
    h = frontier[0]
    assert h["id"] == child and h["eig_score"] > 0.0
    assert h["puct_score"] > h["puct_q"] + h["puct_bonus"] - 1e-6


def test_puct_frontier_zero_weight_restores_pure_puct(tmp_path):
    m, root, child = _discovery_branch(tmp_path, "fresh_metric_2")
    with_eig = m.discovery_frontier_puct(limit=3, max_depth=4, c=1.0,
                                         eig_weight=0.0)
    assert with_eig and "eig_score" not in with_eig[0]


def test_puct_frontier_prefers_high_information_metric(tmp_path):
    """Two branches, same PUCT geometry; the unexplored metric branch must
    outrank the saturated-metric branch once EIG is on."""
    m1, r1, c1 = _discovery_branch(tmp_path, "unexplored")
    # a second branch in the SAME memory: build root2/child2 by hand
    m = m1
    root2 = m.add_hypothesis("a01", "root claim two",
                             {"metric": "saturated", "op": ">=",
                              "target": 0.5})
    m.mark_discovery(root2, 85.0)
    child2 = m.add_hypothesis("a02", "branch claim two",
                              {"metric": "saturated", "op": ">=",
                               "target": 0.9},
                              parents=[root2], depth=1)
    hid = m.add_hypothesis("a01", "ledger fill", {"metric": "saturated",
                                                  "op": ">=", "target": 0.5})
    for _ in range(20):
        m.record_experiment("a01", hid, "supported", metric="saturated")
    frontier = m.discovery_frontier_puct(limit=4, max_depth=4, c=1.0,
                                         eig_weight=1.0)
    ids = [h["id"] for h in frontier]
    assert c1 in ids and child2 in ids
    assert ids.index(c1) < ids.index(child2)    # information first
