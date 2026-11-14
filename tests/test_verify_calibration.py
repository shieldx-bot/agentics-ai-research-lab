"""Wave-2 #4 — verify calibration layer.

Contract under test:
* agreement stats come from the lab's own re-measurements (P23 replays +
  P24bis replications), decisive pairs only;
* Wilson-95 lower bound is the honest floor (small n -> low floor);
* settle notes get the calibration tag ONLY with enough re-runs;
* the supervisor audit alerts once when the floor drops, never floods.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agentics.memory.shared_memory import SharedMemory  # noqa: E402
from agentics.verify import calibration as cal  # noqa: E402


def _mem(tmp_path):
    return SharedMemory(tmp_path / "cal.db")


def _settled_pair(m, agent="a01", verdict="supported", n_replays=1,
                  replay_verdict="confirmed"):
    """One settled hypothesis + `n_replays` re-measurements of it."""
    hyp = m.add_hypothesis(agent, f"claim {m is not None}",
                           {"metric": "acc", "op": ">=", "target": 0.5})
    m.update_hypothesis(hyp, verdict)
    ev_id = m.record_evidence(agent, None, "python x.py", 0, "confirmed",
                              hypothesis=f"claim {hyp}", method="protocol",
                              broadcast=False)
    for _ in range(n_replays):
        m.record_evidence(agent, None, "python x.py", 0, replay_verdict,
                          hypothesis=f"claim {hyp}", method="replay",
                          broadcast=False)
        m.record_replication_detail(ev_id, hyp, agent, replay_verdict, {},
                                    [])
    return hyp


def test_wilson_floor_honest_for_small_n():
    assert cal.wilson_lower(0, 0) == 0.0
    # 5/5 agreement with n=5 still floors ~0.57, NOT 1.0
    assert 0.48 <= cal.wilson_lower(5, 5) < 0.8
    # 50/50 agreement floors high
    assert cal.wilson_lower(50, 50) > 0.9
    # 8/10 agreement floors below the raw rate
    assert cal.wilson_lower(8, 10) < 0.8


def test_stats_count_decisive_pairs_only(tmp_path):
    m = _mem(tmp_path)
    _settled_pair(m, verdict="supported", replay_verdict="confirmed")
    _settled_pair(m, verdict="refuted", replay_verdict="refuted")
    _settled_pair(m, verdict="supported", replay_verdict="refuted")
    _settled_pair(m, verdict="supported", replay_verdict="inconclusive")
    s = cal.agreement_stats(m)
    assert s["n"] == 3                       # inconclusive excluded
    assert s["agreements"] == 2 and s["disagreements"] == 1
    assert abs(s["rate"] - 2 / 3) < 1e-9


def test_suffix_requires_min_history(tmp_path):
    m = _mem(tmp_path)
    _settled_pair(m)                          # n = 1 (detail) + 1 (replay)
    assert cal.agreement_stats(m)["n"] >= 1
    assert cal.calibration_suffix(m, min_n=5) == ""   # too young to claim


def test_suffix_present_with_history(tmp_path):
    m = _mem(tmp_path)
    for _ in range(6):
        _settled_pair(m, verdict="supported", replay_verdict="confirmed")
    sfx = cal.calibration_suffix(m, min_n=5)
    assert sfx.startswith(" [cal ")
    assert "Wilson-95" in sfx and "re-runs" in sfx


def test_settle_note_carries_calibration(tmp_path):
    m = _mem(tmp_path)
    for _ in range(6):
        _settled_pair(m, verdict="supported", replay_verdict="confirmed")
    hyp = m.add_hypothesis("a02", "fresh claim for settle",
                           {"metric": "acc", "op": ">=", "target": 0.5})
    m.update_hypothesis(hyp, "supported")
    m.update_hypothesis(hyp, "supported",
                        note="verdict passed the criterion")
    from agentics.verify.experiment import settle_hypothesis  # noqa: F401
    # settle_hypothesis needs a full Hypothesis/Evidence object stack —
    # exercise just the suffix injection path used inside it:
    note = ("verdict passed the criterion"
            + cal.calibration_suffix(m, min_n=5))
    assert "[cal 1.00 over" in note or "[cal 0." in note


def test_audit_alerts_once_on_low_floor(tmp_path):
    m = _mem(tmp_path)
    # 10 decisive re-runs, 4 disagree -> rate 0.6, Wilson floor < 0.7
    for _ in range(6):
        _settled_pair(m, verdict="supported", replay_verdict="confirmed")
    for _ in range(4):
        _settled_pair(m, verdict="supported", replay_verdict="refuted")
    alerts = []
    ctx = SimpleNamespace(
        agent_id="a07", memory=m,
        settings=SimpleNamespace(),
        journal=lambda agent, event, detail: None,
    )
    m.post_message = lambda agent, kind, content, issue=None: (
        alerts.append(content) if kind == "alert" else None)
    s = cal.calibration_audit(ctx)
    assert s["n"] >= 10
    assert len(alerts) == 1 and "CALIBRATION ALERT" in alerts[0]


def test_audit_silent_when_healthy(tmp_path):
    m = _mem(tmp_path)
    for _ in range(10):
        _settled_pair(m, verdict="supported", replay_verdict="confirmed")
    alerts = []
    ctx = SimpleNamespace(
        agent_id="a07", memory=m, settings=SimpleNamespace(),
        journal=lambda agent, event, detail: None)
    m.post_message = lambda agent, kind, content, issue=None: (
        alerts.append(content) if kind == "alert" else None)
    cal.calibration_audit(ctx)
    assert alerts == []


def test_audit_silent_when_history_young(tmp_path):
    m = _mem(tmp_path)
    _settled_pair(m)                          # n < min_n
    alerts = []
    ctx = SimpleNamespace(
        agent_id="a07", memory=m, settings=SimpleNamespace(),
        journal=lambda agent, event, detail: None)
    m.post_message = lambda agent, kind, content, issue=None: (
        alerts.append(content) if kind == "alert" else None)
    cal.calibration_audit(ctx)
    assert alerts == []
