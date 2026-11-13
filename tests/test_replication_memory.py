"""P24bis — independent replication: the shared provenance layer.

The predecessor's P24 ledger (hypotheses.replication +
hypotheses_needing_replication + record_replication) is the canonical
once-per-swarm bookkeeping; this layer stores the PROVENANCE — who
replicated, what they measured, per-run detail — keyed to the author's
evidence row, and keeps `verified_knowledge.replicated` in sync with the
canonical column.

Memory guarantees anchored here:
* `hypotheses_needing_replication` never offers my own claim (the P24
  selector's independence rule is the single gate);
* `record_replication_detail` is first-peer-wins per evidence (race-free);
* the engine writes BOTH ledgers — the tests pin the double-ledger
  invariant (canonical column + provenance row agree);
* recording broadcasts to every agent (messages + journal).
"""
from __future__ import annotations

from pathlib import Path

from agentics.memory.shared_memory import SharedMemory


def _mem(tmp_path: Path) -> SharedMemory:
    return SharedMemory(tmp_path / "m.db")


def _supported_claim(m: SharedMemory, author: str = "researcher-1",
                     code: str = "print('AGENTICS_RESULT: {}')") -> tuple[int, int]:
    """Author registers a hypothesis (protocol source stored, P23), runs a
    confirmed protocol, settles it."""
    hyp = m.add_hypothesis(
        author, "RF reaches >= 0.9 precision on the demo set",
        {"metric": "precision", "op": ">=", "target": 0.9},
        status="testing", protocol_code=code)
    ev = m.record_evidence(
        author, None, "python experiments/verify_h%d.py" % hyp, 0, "confirmed",
        hypothesis="RF reaches >= 0.9 precision on the demo set", method="protocol")
    m.update_hypothesis(hyp, "supported", evidence_id=ev, note="measured 0.93")
    return hyp, ev


def test_p24_selector_rules(tmp_path):
    m = _mem(tmp_path)
    hyp, ev = _supported_claim(m)
    # a peer sees it; the author never sees their own claim (independence)
    peer_view = m.hypotheses_needing_replication("reviewer-1")
    assert [h["id"] for h in peer_view] == [hyp]
    assert peer_view[0]["agent"] == "researcher-1"
    assert peer_view[0]["protocol_code"].startswith("print(")
    assert peer_view[0]["criterion"]["op"] == ">="
    assert m.hypotheses_needing_replication("researcher-1") == []
    # unshareable protocol -> not offered to anyone (no empty-code row)
    _supported_claim(m, code="")
    assert all(h["protocol_code"]
               for h in m.hypotheses_needing_replication("x"))
    # refuted evidence (dead claim) is not worth replicating
    m.set_evidence_verdict(ev, "refuted", note="reviewer disagrees", by="rev")
    m.update_hypothesis(hyp, "refuted", note="downgraded")
    assert m.hypotheses_needing_replication("reviewer-3") == []


def test_record_replication_detail_uniqueness(tmp_path):
    m = _mem(tmp_path)
    hyp, ev = _supported_claim(m)
    assert m.record_replication_detail(
        ev, hyp, "reviewer-1", "confirmed", {"precision": 0.94},
        [{"exit_code": 0}, {"exit_code": 0}], note="agrees", duration_s=1.2)
    # the SAME replicator cannot double-record its run...
    assert m.record_replication_detail(
        ev, hyp, "reviewer-1", "confirmed", {}, []) is False
    # ...but ANOTHER peer's attempt is a legitimate retry (P24 budget) and
    # gets its own provenance row
    assert m.record_replication_detail(
        ev, hyp, "reviewer-2", "refuted", {"precision": 0.4}, []) is True
    row = m.replication_for_evidence(ev)          # newest attempt wins the read
    assert row["replicator"] == "reviewer-2" and row["verdict"] == "refuted"
    assert m.replication_for_evidence(99999) is None


def test_recording_replication_broadcasts(tmp_path):
    m = _mem(tmp_path)
    hyp, ev = _supported_claim(m)
    m.record_replication_detail(ev, hyp, "reviewer-1", "refuted",
                                {"precision": 0.51},
                                [{"exit_code": 0}, {"exit_code": 0}])
    msgs = [x for x in m.recent_messages() if x["agent"] == "reviewer-1"]
    assert any(x["kind"] == "replication"
               and "replicated evidence #%d" % ev in x["content"]
               for x in msgs)
    journal = [j for j in m.recent_journal(agent="reviewer-1")
               if j["event"] == "replication"]
    assert journal, "replication must land in the journal feed"


def test_double_ledger_invariant(tmp_path):
    """Engine contract: record_replication (canonical) AND
    record_replication_detail (provenance) agree on the outcome."""
    m = _mem(tmp_path)
    hyp, ev = _supported_claim(m)
    m.record_replication(hyp, "confirmed")
    m.record_replication_detail(ev, hyp, "reviewer-1", "confirmed",
                                {"precision": 0.95},
                                [{"exit_code": 0}, {"exit_code": 0}])
    vk = m.verified_knowledge()[0]
    assert vk["replicated"] is True and vk["id"] == hyp
    assert m.replication_for_evidence(ev)["verdict"] == "confirmed"
    assert m.get_hypothesis(hyp)["replication"] == "confirmed"
    # a refuted canonical outcome also downgrades the verified flag view
    m2 = _mem(tmp_path / "n.db")
    hyp2, ev2 = _supported_claim(m2)
    m2.record_replication(hyp2, "refuted")
    m2.record_replication_detail(ev2, hyp2, "reviewer-1", "refuted", {}, [])
    assert m2.verified_knowledge()[0]["replicated"] is False


def test_replication_counts_follow_canonical_ledger(tmp_path):
    m = _mem(tmp_path)
    assert m.replication_counts() == {"replicated": 0, "refuted": 0, "pending": 0}
    hyp, ev = _supported_claim(m)
    assert m.replication_counts()["pending"] == 1
    m.record_replication(hyp, "confirmed")
    m.record_replication_detail(ev, hyp, "reviewer-1", "confirmed", {}, [])
    assert m.replication_counts() == {"replicated": 1, "refuted": 0, "pending": 0}


def test_composes_with_replay_sweep(tmp_path):
    """The three layers coexist on one claim: replayable (P23), replicable
    (P24 selector), and the provenance stays queryable (P24bis)."""
    m = _mem(tmp_path)
    hyp, ev = _supported_claim(m)
    m.record_replication(hyp, "confirmed")
    m.record_replication_detail(ev, hyp, "reviewer-1", "confirmed", {}, [])
    sweep = m.settled_hypotheses(require_code=True)
    assert [h["id"] for h in sweep] == [hyp]
    assert m.hypotheses_needing_replication("reviewer-9") == []
    assert sweep[0]["protocol_code"].startswith("print(")
