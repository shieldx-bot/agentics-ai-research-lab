"""P23bis — independent replication: the shared layer.

Composes with the predecessor's P23 replay (protocol_code stored at
registration, settled-claim re-runs over time): this layer adds the
INDEPENDENCE dimension — a claim is "replicated" only when a DIFFERENT
agent re-runs the shareable protocol in its own workspace and the machine
verdict agrees.

Memory guarantees anchored here:
* `pending_replications` never offers my own claim, unshareable protocols,
  non-confirmed evidence, or already-replicated slots;
* `record_replication` is first-peer-wins (race-free without locks);
* `verified_knowledge` distinguishes author-confirmed from replicated;
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
    confirmed protocol, settles it — the replay sweep's own happy path."""
    hyp = m.add_hypothesis(
        author, "RF reaches >= 0.9 precision on the demo set",
        {"metric": "precision", "op": ">=", "target": 0.9},
        status="testing", protocol_code=code)
    ev = m.record_evidence(
        author, None, "python experiments/verify_h%d.py" % hyp, 0, "confirmed",
        hypothesis="RF reaches >= 0.9 precision on the demo set", method="protocol")
    m.update_hypothesis(hyp, "supported", evidence_id=ev, note="measured 0.93")
    return hyp, ev


def test_pending_replications_rules(tmp_path):
    m = _mem(tmp_path)
    hyp, ev = _supported_claim(m)
    # a peer sees it; the author never sees their own claim (independence)
    peer_view = m.pending_replications("reviewer-1")
    assert [p["hypothesis_id"] for p in peer_view] == [hyp]
    assert peer_view[0]["author"] == "researcher-1"
    assert peer_view[0]["protocol_code"].startswith("print(")
    assert peer_view[0]["criterion"]["op"] == ">="
    assert m.pending_replications("researcher-1") == []
    # unshareable protocol -> not offered to anyone
    hyp2, _ev2 = _supported_claim(m, code="")
    assert all(p["hypothesis_id"] != hyp2
               for p in m.pending_replications("reviewer-2"))
    # refuted evidence is not worth replicating
    m.set_evidence_verdict(ev, "refuted", note="reviewer disagrees", by="rev")
    assert m.pending_replications("reviewer-3") == []


def test_record_replication_first_peer_wins(tmp_path):
    m = _mem(tmp_path)
    hyp, ev = _supported_claim(m)
    assert m.record_replication(
        ev, hyp, "reviewer-1", "confirmed", {"precision": 0.94},
        [{"exit_code": 0}, {"exit_code": 0}], note="agrees", duration_s=1.2)
    # a second peer loses the slot — single replication per evidence
    assert m.record_replication(ev, hyp, "reviewer-2", "confirmed", {}, []) is False
    row = m.replication_for_evidence(ev)
    assert row["replicator"] == "reviewer-1" and row["verdict"] == "confirmed"
    assert row["measured"] == {"precision": 0.94} and row["duration_s"] == 1.2
    assert m.replication_for_evidence(99999) is None


def test_recording_replication_broadcasts(tmp_path):
    m = _mem(tmp_path)
    hyp, ev = _supported_claim(m)
    m.record_replication(ev, hyp, "reviewer-1", "refuted",
                         {"precision": 0.51}, [{"exit_code": 0}, {"exit_code": 0}])
    msgs = [x for x in m.recent_messages() if x["agent"] == "reviewer-1"]
    assert any(x["kind"] == "replication"
               and "replicated evidence #%d" % ev in x["content"]
               for x in msgs)
    journal = [j for j in m.recent_journal(agent="reviewer-1")
               if j["event"] == "replication"]
    assert journal, "replication must land in the journal feed"


def test_verified_knowledge_replicated_flag(tmp_path):
    m = _mem(tmp_path)
    hyp, ev = _supported_claim(m)
    assert m.verified_knowledge()[0]["replicated"] is False
    m.record_replication(ev, hyp, "reviewer-1", "confirmed",
                         {"precision": 0.95}, [{"exit_code": 0}, {"exit_code": 0}])
    vk = m.verified_knowledge()[0]
    assert vk["replicated"] is True and vk["id"] == hyp
    # a refuted replication never marks the claim replicated
    m2 = _mem(tmp_path / "n.db")
    hyp2, ev2 = _supported_claim(m2)
    m2.record_replication(ev2, hyp2, "reviewer-1", "refuted", {}, [])
    assert m2.verified_knowledge()[0]["replicated"] is False


def test_replication_counts(tmp_path):
    m = _mem(tmp_path)
    assert m.replication_counts() == {"replicated": 0, "refuted": 0, "pending": 0}
    hyp, ev = _supported_claim(m)
    assert m.replication_counts()["pending"] == 1
    m.record_replication(ev, hyp, "reviewer-1", "confirmed", {}, [])
    assert m.replication_counts() == {"replicated": 1, "refuted": 0, "pending": 0}


def test_composes_with_replay_sweep(tmp_path):
    """The two P23 layers coexist: a replicated claim is still replayable —
    the sweep's input includes it and both views agree on the claim."""
    m = _mem(tmp_path)
    hyp, ev = _supported_claim(m)
    m.record_replication(ev, hyp, "reviewer-1", "confirmed", {}, [])
    sweep = m.settled_hypotheses(require_code=True)
    assert [h["id"] for h in sweep] == [hyp]
    vk = m.verified_knowledge()[0]
    assert vk["replicated"] and sweep[0]["protocol_code"].startswith("print(")
