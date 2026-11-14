"""P24 — replication ledger: an INDEPENDENT agent re-runs a settled claim's
stored protocol (P23 source) in its own workspace, so a settled verdict is
never just one agent's word (H2/H3 self-healing).

This file pins the LEDGER layer only (selection + bookkeeping); the node
wiring is pinned in tests/test_replication.py.
"""
from __future__ import annotations

import pytest

from agentics.memory.shared_memory import SharedMemory

CRITERION = {"metric": "precision", "op": ">=", "target": 0.9}
PASS_PROTOCOL = (
    "import json\n"
    "print('AGENTICS_RESULT:', json.dumps({'precision': 0.92}))\n"
)


@pytest.fixture()
def mem(tmp_path):
    m = SharedMemory(tmp_path / "m.db")
    yield m
    m.close()


def _supported(mem: SharedMemory, agent: str, claim: str,
               protocol: str = PASS_PROTOCOL) -> int:
    """A settled 'supported' claim with its replayable protocol (P23)."""
    hyp_id = mem.add_hypothesis(agent, claim, CRITERION,
                                status="testing", protocol_code=protocol)
    mem.update_hypothesis(hyp_id, "supported", evidence_id=1,
                          note="measured precision=0.92 >= 0.9")
    return hyp_id


def test_selection_rules(mem):
    mine = _supported(mem, "a01", "claim mine")
    other = _supported(mem, "a02", "claim other")
    no_code = mem.add_hypothesis("a03", "claim no code", CRITERION,
                                 status="testing", protocol_code="")
    mem.update_hypothesis(no_code, "supported", evidence_id=1, note="no source")

    # a01 replicates OTHERS' claims, oldest first; never their own
    todo = mem.hypotheses_needing_replication("a01", limit=10)
    assert [h["id"] for h in todo] == [other]

    # refuted / inconclusive claims are not replication fuel
    refuted = _supported(mem, "a04", "claim refuted")
    mem.update_hypothesis(refuted, "refuted", note="dead")
    inconclusive = _supported(mem, "a05", "claim inc")
    mem.update_hypothesis(inconclusive, "inconclusive", note="flaky")
    todo = mem.hypotheses_needing_replication("a01", limit=10)
    assert other in [h["id"] for h in todo]
    assert refuted not in [h["id"] for h in todo]
    assert inconclusive not in [h["id"] for h in todo]


def test_record_replication_roundtrip_and_limits(mem):
    hyp = _supported(mem, "a02", "claim x")
    # first attempt is inconclusive (sandbox hiccup): retry is allowed
    assert mem.record_replication(hyp, "inconclusive")
    row = mem.get_hypothesis(hyp)
    assert row["replication"] == "inconclusive" and row["replication_attempts"] == 1
    assert row["verdict_note"] == "measured precision=0.92 >= 0.9"  # original kept
    assert [h["id"] for h in mem.hypotheses_needing_replication("a01")] == [hyp]

    # a second inconclusive attempt exhausts the retry budget
    mem.record_replication(hyp, "inconclusive")
    assert mem.get_hypothesis(hyp)["replication_attempts"] == 2
    assert mem.hypotheses_needing_replication("a01") == []

    # invalid outcomes are rejected, never silently stored
    with pytest.raises(ValueError):
        mem.record_replication(hyp, "maybe")


def test_refuted_replication_keeps_verdict_note_until_engine_closes(mem):
    """record_replication only flips the bookkeeping — the P23 replay engine
    (replay_hypothesis) closes the ledger: re-settles the claim + journals
    claim_falsified so verified_knowledge self-corrects."""
    hyp = _supported(mem, "a02", "claim poisoned")
    ev_id = mem.get_hypothesis(hyp)["evidence_id"]
    mem.record_replication(hyp, "refuted")
    # bookkeeping flipped, original verdict still standing
    assert mem.get_hypothesis(hyp)["status"] == "supported"

    # the engine's ledger close (P23 drift handling):
    mem.set_evidence_verdict(ev_id, "refuted",
                             "replay: measured 0.5 < 0.9 (replica a05)",
                             by="a05")
    mem.update_hypothesis(hyp, "refuted", note="replay: refuted")
    row = mem.get_hypothesis(hyp)
    assert row["status"] == "refuted" and row["replication"] == "refuted"
    knowledge = {k["id"]: k for k in mem.verified_knowledge()}
    assert knowledge[hyp]["status"] == "refuted"


def test_legacy_row_reader_defaults(mem):
    """Rows written before the migration (fewer columns in the SELECT tuple)
    read with ''/0 defaults, never None — the positional len-guard."""
    hyp = _supported(mem, "a02", "claim legacy")
    full = mem.get_hypothesis(hyp)
    assert full["replication"] == "" and full["replication_attempts"] == 0
    # a 12-column tuple (pre-P24 reader shape) still decodes
    short = (1, 0.0, "a01", None, None, "claim", "{}", "", "supported",
             "", 1, 0)
    row = SharedMemory._hypothesis_row(short)
    assert row["protocol_code"] == "" and row["replication"] == ""
    assert row["replication_attempts"] == 0
