"""P24bis — the replication engine: a peer re-runs the claim for real.

Real SharedMemory, real protocol execution in a FRESH peer workspace (same
contract as node_verify's tests: a plain directory whose run_command
executes workspace scripts). No network, no git, no LLM — the replicator
needs none: the criterion is frozen data and the verdict is machine-made.

Composes with the predecessor's P24 ledger: the engine writes the canonical
bookkeeping (hypotheses.replication) AND the provenance detail
(replications table) — the tests below pin the double-ledger invariant.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from agentics.memory.shared_memory import SharedMemory
from agentics.verify.replication import replicate_candidate, replicate_pending

PROTOCOL = (
    "import json\n"
    "correct = sum(1 for i in range(200) if (i * 7) % 10 < 9)\n"
    "print('simulated eval done')\n"
    "print('AGENTICS_RESULT:', json.dumps({'precision': correct / 200}))\n"
)
WEAKER_PROTOCOL = PROTOCOL.replace("(i * 7) % 10 < 9", "(i * 7) % 10 < 4")
UNPARSABLE_PROTOCOL = "print('no result line here')\n"


class PeerWS:
    """A fresh peer workspace — a plain directory with the runner contract."""

    def __init__(self, root: Path) -> None:
        self.path = root
        self.path.mkdir(parents=True, exist_ok=True)

    def sync_main(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)

    def write_file(self, rel: str, content: str) -> None:
        target = self.path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def run_command(self, cmd, timeout=120):
        script = self.path / cmd[1]
        proc = subprocess.run([sys.executable, str(script)], capture_output=True,
                              text=True, timeout=timeout, cwd=self.path,
                              check=False)
        return proc.returncode, proc.stdout + proc.stderr


class GuardedWS(PeerWS):
    """A workspace whose guard refuses experiments/ writes (P10 in the way)."""

    def write_file(self, rel: str, content: str) -> None:
        raise ValueError("P10 protected path: blocked")


@pytest.fixture()
def mem(tmp_path):
    m = SharedMemory(tmp_path / "m.db")
    yield m
    m.close()


def _supported(mem: SharedMemory, code: str = PROTOCOL,
               author: str = "researcher-1",
               claim: str = "the method reaches >=0.9 precision on the toy set"):
    """The author's happy path, compressed: register -> run -> settle."""
    hyp = mem.add_hypothesis(
        author, claim, {"metric": "precision", "op": ">=", "target": 0.9},
        status="testing", protocol_code=code)
    ev = mem.record_evidence(
        author, 7, "python experiments/verify_h%d.py" % hyp, 0, "confirmed",
        hypothesis=claim, method="protocol",
        output="AGENTICS_RESULT: {\"precision\": 0.905}")
    mem.update_hypothesis(hyp, "supported", evidence_id=ev, note="measured 0.905")
    return hyp, ev


def _candidate(mem: SharedMemory, agent: str) -> dict:
    pending = mem.hypotheses_needing_replication(agent)
    assert pending, "a settled shareable claim must be pending for a peer"
    return pending[0]


def test_confirmed_replication_flags_common_knowledge(mem, tmp_path):
    hyp, ev = _supported(mem)
    peer_ws = PeerWS(tmp_path / "peer-ws")
    report = replicate_candidate(mem, peer_ws, _candidate(mem, "reviewer-1"),
                                 "reviewer-1")
    assert report["verdict"] == "confirmed" and report["replicated"]
    # the swarm sees the claim as independently replicated (H3) — the P24
    # ledger column is the single source of truth
    vk = mem.verified_knowledge()[0]
    assert vk["replicated"] is True and vk["id"] == hyp
    assert mem.get_hypothesis(hyp)["replication"] == "confirmed"
    # the provenance detail row exists: who replicated, what was measured
    detail = mem.replication_for_evidence(ev)
    assert detail["replicator"] == "reviewer-1"
    assert detail["verdict"] == "confirmed"
    assert detail["measured"]["precision"] == pytest.approx(0.9, abs=0.05)
    # the peer's own evidence row exists, clearly method-tagged
    peer_rows = [r for r in mem.latest_evidence(limit=5)
                 if r["method"] == "replication"]
    assert peer_rows and peer_rows[0]["hypothesis_id"] == hyp
    assert "replication_confirmed" in [j["event"] for j in mem.recent_journal()]
    # the protocol was rebuilt INSIDE the peer's workspace, not the author's
    assert (tmp_path / "peer-ws" / "experiments" /
            f"replicate_h{hyp}.py").exists()


def test_refuted_replication_overrides_the_author(mem, tmp_path):
    hyp, ev = _supported(mem)
    peer_ws = PeerWS(tmp_path / "peer-ws")
    cand = _candidate(mem, "reviewer-1")
    cand["protocol_code"] = WEAKER_PROTOCOL        # the peer measures ~0.4
    report = replicate_candidate(mem, peer_ws, cand, "reviewer-1")
    assert report["verdict"] == "refuted" and not report["replicated"]
    # the author's evidence row is re-verdicted by the machine's verdict
    row = [r for r in mem.latest_evidence(limit=5) if r["id"] == ev][0]
    assert row["verdict"] == "refuted" and "reviewer-1" in row["verdict_by"]
    # the hypothesis is downgraded — dead claims stop posing as common ground
    hyp_row = mem.get_hypothesis(hyp)
    assert hyp_row["status"] == "refuted"
    assert hyp_row["replication"] == "refuted"     # both ledgers agree
    assert "replication_refuted" in [j["event"] for j in mem.recent_journal()]
    assert mem.verified_knowledge()[0]["status"] == "refuted"


def test_inconclusive_replication_never_refutes(mem, tmp_path):
    hyp, ev = _supported(mem)
    peer_ws = PeerWS(tmp_path / "peer-ws")
    cand = _candidate(mem, "reviewer-1")
    cand["protocol_code"] = UNPARSABLE_PROTOCOL    # sandbox hiccup class
    report = replicate_candidate(mem, peer_ws, cand, "reviewer-1")
    assert report["verdict"] == "inconclusive" and not report["replicated"]
    # P17 asymmetry: an inconclusive replication never re-verdicts anything
    row = [r for r in mem.latest_evidence(limit=5) if r["id"] == ev][0]
    assert row["verdict"] == "confirmed"
    assert mem.get_hypothesis(hyp)["status"] == "supported"
    assert "replication_inconclusive" in [j["event"] for j in mem.recent_journal()]
    # the P24 retry budget keeps the claim available for another peer
    assert [h["id"] for h in
            mem.hypotheses_needing_replication("reviewer-2")] == [hyp]


def test_refused_self_replication_and_double_confirm_guard(mem, tmp_path):
    hyp, ev = _supported(mem)
    # the author sees NO pending work (own claims are excluded by the P24
    # selector) — build the candidate from a peer's view and try to run it
    # as the author anyway: the engine refuses self-replication in depth.
    author_ws = PeerWS(tmp_path / "author-ws")
    report = replicate_candidate(mem, author_ws, _candidate(mem, "reviewer-1"),
                                 "researcher-1")
    assert not report["replicated"] and "self-replication" in report["note"]
    assert mem.replication_for_evidence(ev) is None   # nothing recorded
    # once a peer's replication is confirmed, no second confirmation runs —
    # even a direct engine call on a stale candidate skips it
    peer_ws = PeerWS(tmp_path / "peer-ws")
    first = replicate_candidate(mem, peer_ws, _candidate(mem, "reviewer-1"),
                                "reviewer-1")
    assert first["replicated"]
    cand_stale = {"id": hyp, "evidence_id": ev, "agent": "researcher-1",
                  "claim": "the method reaches >=0.9 precision on the toy set",
                  "criterion": {"metric": "precision", "op": ">=", "target": 0.9},
                  "protocol_code": PROTOCOL, "replication": "confirmed"}
    late = replicate_candidate(mem, PeerWS(tmp_path / "late-ws"),
                               cand_stale, "reviewer-2")
    assert not late["replicated"] and "already replicated" in late["note"]
    # the canonical bookkeeping was NOT double-written
    assert mem.get_hypothesis(hyp)["replication_attempts"] == 1


def test_guard_blocked_write_is_journaled(mem, tmp_path):
    hyp, ev = _supported(mem)
    report = replicate_candidate(mem, GuardedWS(tmp_path / "guarded-ws"),
                                 _candidate(mem, "reviewer-1"), "reviewer-1")
    assert not report["replicated"] and "write refused" in report["note"]
    assert "replication_blocked" in [j["event"] for j in mem.recent_journal()]
    row = [r for r in mem.latest_evidence(limit=5) if r["id"] == ev][0]
    assert row["verdict"] == "confirmed"            # never a false refutation


def test_replicate_pending_is_bounded_oldest_first(mem, tmp_path):
    hyp_a, _ev_a = _supported(mem, claim="claim A on the toy set")
    hyp_b, _ev_b = _supported(mem, author="researcher-2",
                              claim="claim B on the toy set")
    peer_ws = PeerWS(tmp_path / "peer-ws")
    reports = replicate_pending(mem, peer_ws, "reviewer-1", limit=1)
    assert len(reports) == 1 and reports[0]["replicated"]
    assert reports[0]["hypothesis_id"] == hyp_a      # P24: oldest first
    # one replication per cycle — claim B stays pending for the next one
    assert mem.replication_counts()["pending"] == 1
    second = replicate_pending(mem, peer_ws, "reviewer-1", limit=1)
    assert len(second) == 1 and second[0]["hypothesis_id"] == hyp_b
    # an empty queue is a clean no-op
    assert replicate_pending(mem, peer_ws, "reviewer-1", limit=3) == []
    counts = mem.replication_counts()
    assert counts["replicated"] == 2 and counts["pending"] == 0


def test_retry_budget_exhaustion(mem, tmp_path):
    """After 2 inconclusive attempts the claim is no longer offered — a
    permanently flaky claim cannot burn swarm compute forever (P24)."""
    _supported(mem)
    cand1 = _candidate(mem, "reviewer-1")
    cand1["protocol_code"] = UNPARSABLE_PROTOCOL
    replicate_candidate(mem, PeerWS(tmp_path / "ws1"), cand1, "reviewer-1")
    cand2 = _candidate(mem, "reviewer-2")
    cand2["protocol_code"] = UNPARSABLE_PROTOCOL
    replicate_candidate(mem, PeerWS(tmp_path / "ws2"), cand2, "reviewer-2")
    assert mem.hypotheses_needing_replication("reviewer-3") == []
    counts = mem.replication_counts()
    assert counts["replicated"] == 0 and counts["pending"] == 0
