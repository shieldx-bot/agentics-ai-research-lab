"""P23 — the protocol source lives INSIDE memory, not just in a workspace.

P21 retention prunes dead agents' workspaces; without the stored code a
settled claim becomes unverifiable forever (its protocol file is gone).
Storing `protocol_code` at registration time keeps every settled claim
replayable — the precondition for drift detection.
"""
from __future__ import annotations

from pathlib import Path

from agentics.memory.shared_memory import SharedMemory
from agentics.verify.experiment import register_hypothesis
from agentics.verify.hypothesis import parse_hypothesis


def _mem(tmp_path: Path) -> SharedMemory:
    return SharedMemory(tmp_path / "m.db")


def test_add_hypothesis_stores_protocol_code(tmp_path):
    m = _mem(tmp_path)
    code = "import json\nprint('AGENTICS_RESULT:', json.dumps({'m': 1.0}))\n"
    hyp_id = m.add_hypothesis(
        "a02", "metric m reaches 1.0", {"metric": "m", "op": "==", "target": 1.0},
        protocol_path="experiments/verify_h1.py", protocol_code=code)
    hyp = m.get_hypothesis(hyp_id)
    assert hyp["protocol_code"] == code
    assert m.list_hypotheses()[0]["protocol_code"] == code


def test_add_hypothesis_code_is_bounded(tmp_path):
    m = _mem(tmp_path)
    hyp_id = m.add_hypothesis(
        "a02", "huge protocol claim", {"metric": "m", "op": ">=", "target": 0},
        protocol_code="x = 1\n" * 10_000)          # 70k chars > 20k cap
    assert 0 < len(m.get_hypothesis(hyp_id)["protocol_code"]) <= 20_000


def test_register_hypothesis_stores_the_code(tmp_path):
    """The coder-path bridge (experiment.register_hypothesis) stores it too."""
    m = _mem(tmp_path)
    raw = {
        "claim": "the cache halves discovery calls",
        "prediction": "hit rate stays above 50%",
        "criterion": {"metric": "hit_rate", "op": ">=", "target": 0.5},
        "code": "import json\nprint('AGENTICS_RESULT:', json.dumps({'hit_rate': 0.72}))\n",
    }
    hyp, reason = parse_hypothesis(raw)
    assert hyp is not None, reason
    hyp_id = register_hypothesis(m, "a04", hyp)
    row = m.get_hypothesis(hyp_id)
    assert row["protocol_code"] == hyp.code
    assert row["status"] == "testing" and row["protocol_path"] == hyp.script_path


def test_default_is_empty_not_null(tmp_path):
    """Rows registered without code (older paths) read as '', never None."""
    m = _mem(tmp_path)
    hyp_id = m.add_hypothesis("a02", "no code claim",
                              {"metric": "m", "op": ">=", "target": 1})
    assert m.get_hypothesis(hyp_id)["protocol_code"] == ""
