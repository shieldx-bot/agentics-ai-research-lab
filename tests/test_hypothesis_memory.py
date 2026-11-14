"""P16bis — pre-registered hypotheses + verified knowledge (H2/H3).

The hypotheses table keeps the criterion STRUCTURED (metric/op/target)
registered before the protocol runs; `verified_knowledge` is the feed every
agent's prompt sees — settled claims become common ground, refuted claims
stop being repeated. Composes with the predecessor's evidence table: the
protocol outcome is attached onto record_evidence's row, never a second
insert path.
"""
from __future__ import annotations

from pathlib import Path

from agentics.memory.shared_memory import SharedMemory
from agentics.verify_metrics import Criterion, run_protocol


def _mem(tmp_path: Path) -> SharedMemory:
    return SharedMemory(tmp_path / "m.db")


def test_hypothesis_lifecycle_roundtrip(tmp_path):
    m = _mem(tmp_path)
    hyp_id = m.add_hypothesis(
        "a02", "Random Forest reaches >=0.9 precision on the demo set",
        {"metric": "precision", "op": ">=", "target": 0.9},
        task_id=7, finding_id=3, protocol_path="experiments/verify_rf.py",
    )
    hyp = m.get_hypothesis(hyp_id)
    assert hyp["status"] == "proposed"
    assert hyp["criterion"] == {"metric": "precision", "op": ">=", "target": 0.9}
    assert m.update_hypothesis(hyp_id, "testing") is True
    ev_id = m.record_evidence(
        "a02", 7, "python experiments/verify_rf.py", 0, "confirmed",
        hypothesis=hyp["claim"], method="protocol",
        artifact="experiments/verify_rf.py",
    )
    m.attach_protocol_outcome(ev_id, {"precision": 0.93}, [{"exit_code": 0}],
                              hypothesis_id=hyp_id)
    m.update_hypothesis(hyp_id, "supported", evidence_id=ev_id,
                        note="measured precision=0.93 vs precision >= 0.9")
    hyp = m.get_hypothesis(hyp_id)
    assert hyp["status"] == "supported" and hyp["evidence_id"] == ev_id
    assert hyp["attempts"] == 2                       # testing + supported
    row = m.evidence_for_task(7)[0]
    assert row["hypothesis_id"] == hyp_id
    assert row["metrics"] == {"precision": 0.93}
    assert row["runs"][0]["exit_code"] == 0


def test_verified_knowledge_only_shows_settled_claims(tmp_path):
    m = _mem(tmp_path)
    sup = m.add_hypothesis("a02", "claim A holds", {"metric": "m", "op": ">=", "target": 1})
    ref = m.add_hypothesis("a03", "claim B holds", {"metric": "m", "op": ">=", "target": 1})
    m.add_hypothesis("a04", "claim C pending", {"metric": "m", "op": ">=", "target": 1})
    m.update_hypothesis(sup, "supported", note="measured m=1.0")
    m.update_hypothesis(ref, "refuted", note="measured m=0.2")
    knowledge = m.verified_knowledge()
    statuses = {k["claim"]: k["status"] for k in knowledge}
    assert statuses == {"claim A holds": "supported", "claim B holds": "refuted"}
    assert all("pending" not in k["claim"] for k in knowledge)
    assert m.list_hypotheses(status="proposed")[0]["claim"] == "claim C pending"
    assert m.hypothesis_counts() == {"supported": 1, "refuted": 1, "proposed": 1}


def test_list_hypotheses_filters_by_task(tmp_path):
    m = _mem(tmp_path)
    m.add_hypothesis("a02", "task 7 claim", {}, task_id=7)
    m.add_hypothesis("a02", "task 8 claim", {}, task_id=8)
    assert [h["claim"] for h in m.list_hypotheses(task_id=7)] == ["task 7 claim"]


def test_protocol_evidence_composes_with_the_shared_table(tmp_path):
    """run_protocol -> record_evidence -> attach: one insert, no duplication."""
    m = _mem(tmp_path)
    root = tmp_path / "ws"
    (root / "experiments").mkdir(parents=True)

    class FakeWS:
        path = root

        def run_command(self, cmd, timeout=120):
            script = root / cmd[1]
            out = script.read_text().replace("METRIC", "0.93")
            import subprocess
            import sys
            proc = subprocess.run([sys.executable, "-c", out],
                                  capture_output=True, text=True)
            return proc.returncode, proc.stdout + proc.stderr

    script = root / "experiments" / "verify.py"
    script.write_text("print('AGENTICS_RESULT:', '{\"precision\": METRIC}')\n")
    ev = run_protocol(FakeWS(), "experiments/verify.py",
                      Criterion("precision", ">=", 0.9))
    assert ev.verdict == "confirmed"
    hyp_id = m.add_hypothesis("a02", "p >= 0.9",
                              {"metric": "precision", "op": ">=", "target": 0.9})
    ev_id = m.record_evidence(
        "a02", None, ev.cmd, ev.exit_code, ev.verdict,
        hypothesis="p >= 0.9", method=ev.method, output=ev.output,
        duration_s=ev.duration_s, artifact="experiments/verify.py",
    )
    m.attach_protocol_outcome(ev_id, ev.meta["metrics"], ev.meta["runs"],
                              hypothesis_id=hyp_id)
    row = m.latest_evidence()[0]
    assert row["verdict"] == "confirmed" and row["metrics"]["precision"] == 0.93
    assert len(row["runs"]) == 2 and row["hypothesis_id"] == hyp_id


def test_evidence_row_reader_tolerates_legacy_rows(tmp_path):
    """A DB written before the metrics columns still reads cleanly."""
    m = _mem(tmp_path)
    m.record_evidence("a01", None, "python x.py", 0, "confirmed")
    m._lock and m._conn.execute(
        "UPDATE evidence SET metrics='', runs='' WHERE id=1")
    m._conn.commit()
    row = m.latest_evidence()[0]
    assert row["metrics"] == {} and row["runs"] == []


def test_verify_ops_evidence_still_records_without_metrics(tmp_path):
    """The predecessor's run_command path keeps working unchanged."""
    m = _mem(tmp_path)
    ev_id = m.record_evidence("a04", 3, "python -m pytest", 0, "confirmed")
    row = m.evidence_for_task(3)[0]
    assert row["id"] == ev_id and row["method"] == "run_command"
    assert row["metrics"] == {} and row["runs"] == []
