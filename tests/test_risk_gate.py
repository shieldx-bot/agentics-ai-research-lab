"""Wave-3 #5 — learned risk gate, shadow-only v1 (verify/gate.py).

Contract under test:
* features are computed strictly from PRIOR traces (no leakage);
* below MIN_TRAIN_ROWS the gate is honestly silent (no model);
* on a synthetic separable ledger it trains, ranks risk sensibly, and
  journals `risk_gate_prediction` WITHOUT ever changing a flow;
* "off" mode journals nothing.
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agentics.memory.shared_memory import SharedMemory  # noqa: E402
from agentics.verify import gate as g  # noqa: E402


def _mem(tmp_path):
    return SharedMemory(tmp_path / "gate.db")


def _ctx(memory, cycle=3, **settings_kw):
    base = dict(risk_gate_mode="shadow", risk_gate_soft_threshold=0.25)
    base.update(settings_kw)
    return SimpleNamespace(agent_id="a11", memory=memory, cycle=cycle,
                           settings=SimpleNamespace(**base))


def _seed(tmp_path, n=60, good_ratio=0.6):
    """A separable ledger: agent_0 only ever succeeds, agent_1 only ever
    fails — so their prior-history features genuinely differ."""
    m = _mem(tmp_path)
    for i in range(n):
        good = i < int(n * good_ratio)
        agent = "agent_0" if good else "agent_1"
        for k in range(6):
            m.add_trace(agent, role="coder", cycle=k,
                        outcome="approved" if good else "failed",
                        category="efficiency",
                        attempted="run protocol probe %d" % k,
                        failure="" if good else "assertion failed")
        m.add_trace(agent, role="coder", cycle=6,
                    outcome="merged" if good else "rejected",
                    category="efficiency",
                    attempted="protocol v%d for the convergence claim" % i,
                    failure="" if good else "tier2 tests failed")
    return m


# ------------------------------------------------------------- features
def test_features_no_leakage(tmp_path):
    m = _mem(tmp_path)
    rid = m.add_trace("a11", role="coder", cycle=1, outcome="failed",
                      category="c", attempted="x", failure="boom")
    rid2 = m.add_trace("a11", role="coder", cycle=2, outcome="approved",
                       category="c", attempted="y")
    t0 = time.time() - 1000
    # add_trace stamps wall-clock time; pin both rows so the ordering is
    # exactly what the leakage assertion needs (test-only manipulation).
    with m._conn:
        m._conn.execute("UPDATE traces SET ts=? WHERE id=?", (t0, rid))
        m._conn.execute("UPDATE traces SET ts=? WHERE id=?", (t0 + 1000, rid2))
    f_past = g.features_for(m, "a11", "attempt", 3, before_ts=t0 + 500)
    assert f_past["prior_attempts"] == 1 and f_past["fail_streak"] == 1
    f_all = g.features_for(m, "a11", "attempt", 3)
    assert f_all["prior_attempts"] == 2 and f_all["had_failure"] == 0


def test_features_empty_history_is_honest_zero(tmp_path):
    m = _mem(tmp_path)
    f = g.features_for(m, "nobody", "", 0)
    assert f["prior_attempts"] == 0.0 and f["prior_fail_rate"] == 0.0


# -------------------------------------------------------------- training
def test_too_few_rows_is_silent(tmp_path):
    m = _mem(tmp_path)
    for i in range(g.MIN_TRAIN_ROWS - 5):
        m.add_trace("a11", role="coder", cycle=i, outcome="approved",
                    attempted="run %d" % i)
    assert g.train(m) is None


def test_train_and_rank_risk(tmp_path):
    m = _seed(tmp_path)
    model = g.train(m)
    assert model is not None and model.n >= g.MIN_TRAIN_ROWS
    assert model.auc > 0.6, "a separable ledger must beat random (in-sample)"
    risky = g.features_for(m, "agent_1", "protocol for the claim", 7)
    safe = g.features_for(m, "agent_0", "protocol for the claim", 7)
    # agent_1 ends its history on a failure streak, agent_0 on successes;
    # p is P(settle clean) — the clean agent must rank higher
    assert model.predict(safe) > model.predict(risky)
    assert 0.0 <= model.predict(risky) <= 1.0


# ---------------------------------------------------------------- shadow
def test_shadow_check_journals_never_blocks(tmp_path):
    m = _seed(tmp_path, n=60)
    g._state.update({"model": None, "trained_at": 0.0, "n_train": 0})
    ctx = _ctx(m)
    state = {"plan": "protocol v2 for the claim", "error": ""}
    g.shadow_check(ctx, state)          # must not raise, must not block
    rows = m._conn.execute(
        "SELECT detail FROM journal WHERE event='risk_gate_prediction' "
        "ORDER BY id DESC LIMIT 1").fetchone()
    assert rows, "shadow mode journals the prediction"
    detail = json.loads(rows[0])
    assert 0.0 <= detail["p"] <= 1.0
    assert "auc_insample" in detail and "n_train" in detail
    assert state["plan"] == "protocol v2 for the claim"


def test_off_mode_journals_nothing(tmp_path):
    m = _seed(tmp_path, n=60)
    g._state.update({"model": None, "trained_at": 0.0, "n_train": 0})
    ctx = _ctx(m, risk_gate_mode="off")
    g.shadow_check(ctx, {"plan": "x"})
    rows = m._conn.execute(
        "SELECT COUNT(*) FROM journal WHERE event='risk_gate_prediction'"
    ).fetchone()
    assert rows[0] == 0


def test_soft_mode_is_advisory_journal_only(tmp_path):
    m = _seed(tmp_path, n=60)
    g._state.update({"model": None, "trained_at": 0.0, "n_train": 0})
    # threshold 1.0 ⇒ p < 1.0 always fires the advisory row (sigmoid < 1)
    ctx = _ctx(m, risk_gate_mode="soft", risk_gate_soft_threshold=1.0)
    state = {"plan": "protocol for the claim"}
    g.shadow_check(ctx, state)
    rows = m._conn.execute(
        "SELECT COUNT(*) FROM journal WHERE event='risk_gate_softblock'"
    ).fetchone()
    assert rows[0] >= 1
    # the flow state is untouched — advisory only
    assert state == {"plan": "protocol for the claim"}


def test_review_node_hooks_shadow_gate():
    """The review node calls shadow_check before the panel opens (source
    wiring — the node must keep the try/except so the gate can never
    break a review)."""
    import inspect
    from agentics.graph import nodes
    src = inspect.getsource(nodes)
    assert "from agentics.verify.gate import shadow_check" in src
    assert "shadow_check(ctx, state)" in src
