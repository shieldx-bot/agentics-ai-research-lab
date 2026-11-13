"""P23 — replay (re-verification of settled claims) + contradiction alerts.

Replication is what makes the ledger trustworthy over time: a claim settled
once must survive being re-run from its stored protocol source; and when two
hypotheses with the SAME criterion settle to OPPOSITE verdicts, the swarm has
disagreed with itself on a measurable fact — that must be an alert, not a
silently corrupted knowledge base.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from agentics.memory.shared_memory import SharedMemory
from agentics.verify.replay import replay_hypothesis, replay_settled

GOOD_CODE = (
    "import json\n"
    "correct = sum(1 for i in range(200) if (i * 7) % 10 < 9)\n"
    "print('AGENTICS_RESULT:', json.dumps({'precision': correct / 200}))\n"
)
BAD_CODE = GOOD_CODE.replace("(i * 7) % 10 < 9", "(i * 7) % 10 < 4")
CRASH_CODE = "1 / 0\n"


class FakeWS:
    """Plain-directory workspace; run_command executes workspace .py scripts
    for real (same allowlist contract as RepoWorkspace)."""

    def __init__(self, root: Path) -> None:
        self.path = root

    def write_file(self, rel: str, content: str) -> None:
        target = self.path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def run_command(self, cmd, timeout=120):
        proc = subprocess.run([sys.executable, str(self.path / cmd[1])],
                              capture_output=True, text=True, timeout=timeout,
                              cwd=self.path, check=False)
        return proc.returncode, proc.stdout + proc.stderr


@pytest.fixture()
def mem(tmp_path):
    m = SharedMemory(tmp_path / "m.db")
    yield m
    m.close()


def _register(mem: SharedMemory, claim: str, code: str, status: str,
              metric: str = "precision") -> int:
    return mem.add_hypothesis(
        "a02", claim, {"metric": metric, "op": ">=", "target": 0.9},
        status=status, protocol_code=code)


def _events(mem: SharedMemory) -> list[str]:
    return [j["event"] for j in mem.recent_journal(limit=30)]


def test_replay_reconfirms_a_supported_claim(tmp_path, mem):
    hyp_id = _register(mem, "ensemble reaches >=0.9 precision", GOOD_CODE,
                       status="supported")
    rep = replay_hypothesis(mem, FakeWS(tmp_path / "ws"), hyp_id, agent="owner")
    assert rep["replayed"] and rep["event"] == "claim_reconfirmed"
    assert rep["status"] == rep["was"] == "supported"
    ev = mem.latest_evidence()[0]
    assert ev["method"] == "replay" and ev["verdict"] == "confirmed"
    assert ev["hypothesis_id"] == hyp_id and len(ev["runs"]) == 2
    assert "claim_reconfirmed" in _events(mem)


def test_replay_detects_drift_supported_becomes_refuted(tmp_path, mem):
    """A claim that settled supported but whose protocol actually measures
    0.4: the replay corrects the ledger and alerts the swarm."""
    hyp_id = _register(mem, "baseline reaches >=0.9 precision", BAD_CODE,
                       status="supported")
    rep = replay_hypothesis(mem, FakeWS(tmp_path / "ws"), hyp_id)
    assert rep["event"] == "claim_falsified" and rep["status"] == "refuted"
    assert mem.get_hypothesis(hyp_id)["status"] == "refuted"
    knowledge = {k["claim"]: k["status"] for k in mem.verified_knowledge()}
    assert knowledge["baseline reaches >=0.9 precision"] == "refuted"
    assert "claim_falsified" in _events(mem)


def test_replay_overturns_a_wrong_refutation(tmp_path, mem):
    hyp_id = _register(mem, "upheld claim", GOOD_CODE, status="refuted")
    rep = replay_hypothesis(mem, FakeWS(tmp_path / "ws"), hyp_id)
    assert rep["event"] == "claim_overturned" and rep["status"] == "supported"


def test_replay_of_a_crashing_protocol_is_claim_unstable(tmp_path, mem):
    hyp_id = _register(mem, "unstable claim", CRASH_CODE, status="supported")
    rep = replay_hypothesis(mem, FakeWS(tmp_path / "ws"), hyp_id)
    assert rep["event"] == "claim_unstable"
    assert rep["status"] == "inconclusive"       # inconclusive never fakes success


def test_replay_skips_claims_without_stored_code(tmp_path, mem):
    hyp_id = _register(mem, "legacy claim", "", status="supported")
    rep = replay_hypothesis(mem, FakeWS(tmp_path / "ws"), hyp_id)
    assert not rep["replayed"] and "no stored protocol" in rep["note"]
    assert mem.latest_evidence() == []            # no evidence row fabricated


def test_replay_sweep_never_raises(tmp_path, mem):
    ids = [_register(mem, "claim A", GOOD_CODE, "supported"),
           _register(mem, "claim B", BAD_CODE, "refuted"),
           _register(mem, "claim C", "", "supported"),      # skipped
           _register(mem, "claim D", CRASH_CODE, "supported")]
    rep = replay_settled(mem, FakeWS(tmp_path / "ws"), limit=10)
    assert len(rep) == 3                          # require_code drops claim C
    assert {r["hyp_id"] for r in rep} == set(ids) - {ids[2]}
    assert all("note" in r for r in rep)


# ------------------------------------------------------------ contradiction
def test_contradiction_alerts_when_same_criterion_settles_opposite(tmp_path, mem):
    c = {"metric": "precision", "op": ">=", "target": 0.9}
    first = mem.add_hypothesis("a02", "ensemble is precise", c)
    mem.update_hypothesis(first, "supported")
    second = mem.add_hypothesis("a03", "ensemble is NOT precise", c)
    mem.update_hypothesis(second, "refuted", agent="a03")
    contradictions = [j for j in mem.recent_journal(limit=30)
                      if j["event"] == "claim_contradiction"]
    assert len(contradictions) == 1
    detail = json.loads(contradictions[0]["detail"])
    assert detail["hyp"] == second and detail["against_hyp"] == first
    assert detail["settled_to"] == "refuted" and detail["criterion"]["metric"] == "precision"


def test_no_contradiction_for_different_criteria(tmp_path, mem):
    first = mem.add_hypothesis("a02", "claim on precision",
                               {"metric": "precision", "op": ">=", "target": 0.9})
    mem.update_hypothesis(first, "supported")
    second = mem.add_hypothesis("a03", "claim on recall",
                                {"metric": "recall", "op": ">=", "target": 0.9})
    mem.update_hypothesis(second, "refuted", agent="a03")
    assert [j["event"] for j in mem.recent_journal(limit=30)
            if j["event"] == "claim_contradiction"] == []


def test_same_criterion_same_verdict_is_not_a_contradiction(tmp_path, mem):
    c = {"metric": "precision", "op": ">=", "target": 0.9}
    first = mem.add_hypothesis("a02", "claim one", c)
    mem.update_hypothesis(first, "supported")
    second = mem.add_hypothesis("a03", "claim two", c)
    mem.update_hypothesis(second, "supported", agent="a03")
    assert [j["event"] for j in mem.recent_journal(limit=30)
            if j["event"] == "claim_contradiction"] == []


def test_replayed_flip_of_the_same_row_is_not_a_contradiction(tmp_path, mem):
    """A replay flipping its OWN row must not alert (that's drift, P23)."""
    hyp_id = _register(mem, "self flip", BAD_CODE, status="supported",
                       metric="precision")     # same criterion dict shape
    rep = replay_hypothesis(mem, FakeWS(tmp_path / "ws"), hyp_id)
    assert rep["event"] == "claim_falsified"
    events = _events(mem)
    assert events.count("claim_falsified") == 1
    assert "claim_contradiction" not in events
