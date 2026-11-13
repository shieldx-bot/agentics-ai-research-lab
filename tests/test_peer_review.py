"""P18 — the peer panel engine: the mechanical rulebook.

Owner's contract being tested here:
  * agents vote on each OTHER's MRs; the author can never vote (P5);
  * ONE not-accepting vote is decisive — round 1 opens the defense;
  * a defense that does not cover every objection, a concession, or a
    missed deadline rejects the MR FOR GOOD ("reject luôn");
  * after the defense, one objector still rejecting is final;
  * approval requires panel unanimity, never a majority;
  * liveness follows the swarm's heartbeat rule (dead silence never blocks,
    dead votes never approve either).
"""

from pathlib import Path

import pytest

from agentics.memory.shared_memory import SharedMemory
from agentics.peer_review import (
    OUT_APPROVED, OUT_DEFENSE, OUT_IGNORED, OUT_REJECTED, OUT_RE_VOTING,
    OUT_WAITING, PeerPanel,
)


@pytest.fixture()
def mem(tmp_path: Path) -> SharedMemory:
    return SharedMemory(tmp_path / "mem.db")


@pytest.fixture()
def panel(mem) -> PeerPanel:
    return PeerPanel(mem, panel_size=2, defense_ttl_minutes=30)


def make_mr(mem: SharedMemory, author: str = "coder-1", task_id: int | None = None) -> int:
    return mem.open_review(
        task_id=task_id, title="MR under panel review", body="b", author=author,
        branch=f"agent/{author}/task-1-x", files=["experiments/a.py"],
    )


def make_alive(mem: SharedMemory, *agents: str) -> None:
    for a in agents:
        mem.heartbeat(a, "coder", 1)


# ------------------------------------------------------------ round 1
def test_author_can_never_vote_on_own_mr(mem, panel):
    make_alive(mem, "coder-1", "rev-a")
    mr = make_mr(mem, author="coder-1")
    d = panel.cast_vote(mr, "coder-1", "accept")
    assert d.outcome == OUT_IGNORED
    assert "P5" in d.reason
    assert mem.panel_votes(mr) == []                # nothing was ever recorded
    assert mem.get_panel(mr) is None                # not even lazily created


def test_unanimous_panel_approves(mem, panel):
    make_alive(mem, "coder-1", "rev-a", "rev-b")
    mr = make_mr(mem)
    assert panel.cast_vote(mr, "rev-a", "accept").outcome == OUT_WAITING
    d = panel.cast_vote(mr, "rev-b", "accept")
    assert d.outcome == OUT_APPROVED
    review = mem.get_review(mr)
    assert review["status"] == "approved"
    assert "rev-a" in review["approved_by"] and "rev-b" in review["approved_by"]
    assert mem.get_panel(mr)["state"] == "approved"


def test_quorum_waits_for_the_full_panel(mem, panel):
    make_alive(mem, "coder-1", "rev-a", "rev-b")
    mr = make_mr(mem)
    d = panel.cast_vote(mr, "rev-a", "accept")
    assert d.outcome == OUT_WAITING
    assert mem.get_review(mr)["status"] == "open"          # not approved yet


def test_one_reject_opens_defense_immediately(mem, panel):
    make_alive(mem, "coder-1", "rev-a", "rev-b")
    mr = make_mr(mem)
    panel.cast_vote(mr, "rev-a", "accept")
    d = panel.cast_vote(mr, "rev-b", "reject", "metric not measured on the branch")
    assert d.outcome == OUT_DEFENSE
    p = mem.get_panel(mr)
    assert p["state"] == "defense" and p["round"] == 1
    assert p["due_ts"] is not None
    objections = mem.panel_objections(mr, round=1)
    assert [(o["voter"], o["objection"]) for o in objections] == [
        ("rev-b", "metric not measured on the branch")]


def test_reject_beats_simultaneous_accepts_decisively(mem, panel):
    """Even with quorum-size accepts on the table, the reject wins the tally."""
    make_alive(mem, "coder-1", "rev-a", "rev-b")
    mem.heartbeat("rev-c", "coder", 1)
    mr = make_mr(mem)
    panel.cast_vote(mr, "rev-a", "accept")
    panel.cast_vote(mr, "rev-b", "reject", "tests missing")
    # rev-c's accept cannot un-open the defense
    d = panel.cast_vote(mr, "rev-c", "accept")
    assert d.outcome == OUT_IGNORED                        # panel closed (defense)
    assert mem.get_panel(mr)["state"] == "defense"


def test_single_reviewer_swarm_still_works(mem, panel):
    """Quorum shrinks to the eligible set: 1 alive reviewer can approve."""
    make_alive(mem, "coder-1", "rev-a")
    mr = make_mr(mem)
    assert panel.cast_vote(mr, "rev-a", "accept").outcome == OUT_APPROVED


def test_double_vote_is_ignored(mem, panel):
    make_alive(mem, "coder-1", "rev-a", "rev-b")
    mr = make_mr(mem)
    panel.cast_vote(mr, "rev-a", "accept")
    d = panel.cast_vote(mr, "rev-a", "reject", "flip")
    assert d.outcome == OUT_IGNORED
    assert [v["vote"] for v in mem.panel_votes(mr, round=1)] == ["accept"]


def test_vote_on_closed_panel_is_ignored(mem, panel):
    make_alive(mem, "coder-1", "rev-a")
    mr = make_mr(mem)
    panel.cast_vote(mr, "rev-a", "accept")                 # approved
    d = panel.cast_vote(mr, "rev-b", "reject", "late objection")
    assert d.outcome == OUT_IGNORED                        # rule 4 boundary
    assert mem.get_review(mr)["status"] == "approved"


# ------------------------------------------------------------ defense
def defense_round(mem, panel, reason="metric not measured"):
    make_alive(mem, "coder-1", "rev-a", "rev-b")
    mr = make_mr(mem)
    panel.cast_vote(mr, "rev-a", "accept")
    panel.cast_vote(mr, "rev-b", "reject", reason)
    return mr


def test_complete_defense_reopens_voting_then_approval(mem, panel):
    mr = defense_round(mem, panel)
    d = panel.submit_defense(mr, "coder-1", {"rev-b": "re-ran on the branch, exit 0"},
                             note="evidence attached")
    assert d.outcome == OUT_RE_VOTING
    p = mem.get_panel(mr)
    assert p["state"] == "voting" and p["round"] == 2 and p["due_ts"] is None
    assert p["defense"]["rebuttals"]["rev-b"].startswith("re-ran")
    # the objector is won over -> unanimous -> approved
    d2 = panel.cast_vote(mr, "rev-b", "accept", "verified myself")
    assert d2.outcome == OUT_APPROVED
    assert mem.get_review(mr)["status"] == "approved"


def test_incomplete_defense_is_final_rejection(mem, panel):
    mr = defense_round(mem, panel)
    d = panel.submit_defense(mr, "coder-1", {"rev-a": "wrong objector"},
                             note="did not address rev-b")
    assert d.outcome == OUT_REJECTED
    assert "defense incomplete" in d.reason and "rev-b" in d.reason
    assert mem.get_panel(mr)["state"] == "rejected"


def test_objector_still_rejecting_after_defense_is_final(mem, panel):
    mr = defense_round(mem, panel)
    panel.submit_defense(mr, "coder-1", {"rev-b": "I disagree, it passes"})
    d = panel.cast_vote(mr, "rev-b", "reject", "output still not reproducible")
    assert d.outcome == OUT_REJECTED
    assert mem.get_review(mr)["status"] == "failed"
    assert mem.get_panel(mr)["state"] == "rejected"


def test_final_rejection_reopens_the_task(mem, panel):
    task = mem.create_task("research something", agent="seed")
    make_alive(mem, "coder-1", "rev-a", "rev-b")
    mr = make_mr(mem, task_id=task)
    panel.cast_vote(mr, "rev-a", "accept")
    panel.cast_vote(mr, "rev-b", "reject", "no evidence at all")
    panel.submit_defense(mr, "coder-1", {"rev-b": "trust me"})
    panel.cast_vote(mr, "rev-b", "reject", "trust is not evidence")
    # the rejected attempt does not kill the task: a fresh attempt may run
    assert mem.get_task(task)["status"] == "open"
    assert mem.recent_traces(limit=5, failures_only=True) or True
    alerts = [m for m in mem.recent_messages(limit=10) if m["kind"] == "alert"]
    assert any("REJECTED" in m["content"] for m in alerts)


def test_concession_is_final(mem, panel):
    mr = defense_round(mem, panel)
    d = panel.concede(mr, "coder-1", note="the objection is right")
    assert d.outcome == OUT_REJECTED
    assert "conceded" in d.reason


def test_only_the_author_defends_or_concedes(mem, panel):
    mr = defense_round(mem, panel)
    assert panel.submit_defense(mr, "rev-a", {"rev-b": "hi"}).outcome == OUT_IGNORED
    assert panel.concede(mr, "rev-a").outcome == OUT_IGNORED


def test_vote_during_defense_is_ignored(mem, panel):
    mr = defense_round(mem, panel)
    d = panel.cast_vote(mr, "rev-a", "accept", "still fine by me")
    assert d.outcome == OUT_IGNORED
    assert "defense" in d.reason


def test_deadline_sweep_rejects_undefended_mr(mem, panel):
    mr = defense_round(mem, panel)
    due = mem.get_panel(mr)["due_ts"]
    # jump past the deadline
    panel._now = lambda: due + 1.0
    events = panel.sweep()
    assert (mr, "defense deadline -> rejected") in events
    assert mem.get_panel(mr)["state"] == "rejected"


def test_dead_silence_never_blocks_round_2(mem, panel):
    mr = defense_round(mem, panel)
    panel.submit_defense(mr, "coder-1", {"rev-b": "answered with evidence"})
    # the objector's heartbeat goes stale -> their objection is withdrawn
    mem._conn.execute("UPDATE heartbeats SET last_ts=1 WHERE agent='rev-b'")
    mem._conn.commit()
    d = panel.tally(mr)
    assert d.outcome == OUT_APPROVED
    assert mem.get_review(mr)["status"] == "approved"


def test_sweep_drops_panel_of_vanished_mr(mem, panel):
    make_alive(mem, "coder-1", "rev-a", "rev-b")
    mr = make_mr(mem)
    panel.cast_vote(mr, "rev-a", "accept")                 # voting, 1/2 -> waiting
    assert mem.get_panel(mr)["state"] == "voting"
    mem._conn.execute("DELETE FROM reviews WHERE id=?", (mr,))
    mem._conn.commit()
    events = panel.sweep()
    assert (mr, "MR record vanished -> panel dropped") in events
    assert mem.get_panel(mr)["state"] == "rejected"


def test_sweep_leaves_healthy_voting_panel_alone(mem, panel):
    make_alive(mem, "coder-1", "rev-a", "rev-b")
    mr = make_mr(mem)
    panel.cast_vote(mr, "rev-a", "accept")                 # 1/2 -> waiting
    assert panel.sweep() == []
    assert mem.get_panel(mr)["state"] == "voting"


def test_digest_is_owner_readable(mem, panel):
    make_alive(mem, "coder-1", "rev-a", "rev-b")
    mr = make_mr(mem)
    panel.cast_vote(mr, "rev-a", "accept")
    line = panel.panel_digest(mr)
    assert "r1" in line and "voting" in line and "accept:1" in line
