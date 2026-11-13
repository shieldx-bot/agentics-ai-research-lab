"""P19 — peer panel review: the shared-memory layer.

The panel state machine makes "agents evaluate each other" durable shared
truth: every vote, objection and defense is a row every thread can read.
These tests pin the storage contract the PeerPanel engine (and the dashboard)
builds on.
"""

from pathlib import Path

import pytest

from agentics.memory.shared_memory import SharedMemory


@pytest.fixture()
def mem(tmp_path: Path) -> SharedMemory:
    return SharedMemory(tmp_path / "mem.db")


def test_ensure_panel_is_idempotent_and_defaults_to_voting_round_1(mem):
    p1 = mem.ensure_panel(7, opened_ts=100.0)
    p2 = mem.ensure_panel(7)                      # second call must not reset it
    assert p1["mr_id"] == 7
    assert p1["state"] == "voting"
    assert p1["round"] == 1
    assert p1["opened_ts"] == 100.0               # the explicit ts wins, once
    assert p2["opened_ts"] == 100.0
    assert p1["due_ts"] is None and p1["defense"] == {}


def test_vote_first_vote_sticks_per_round(mem):
    assert mem.record_panel_vote(1, 1, "rev-a", "accept") is True
    # a reviewer cannot silently flip their verdict after the fact
    assert mem.record_panel_vote(1, 1, "rev-a", "reject", "changed my mind") is False
    # a new round is a fresh vote (after a defense)
    assert mem.record_panel_vote(1, 2, "rev-a", "accept") is True
    assert mem.record_panel_vote(1, 1, "rev-b", "reject", "why") is True
    votes_r1 = mem.panel_votes(1, round=1)
    assert [(v["voter"], v["vote"]) for v in votes_r1] == [("rev-a", "accept"),
                                                           ("rev-b", "reject")]
    assert votes_r1[1]["reason"] == "why"
    assert len(mem.panel_votes(1)) == 3           # all rounds, chronological


def test_objection_upsert_keeps_latest_text(mem):
    mem.record_panel_objection(1, 1, "rev-b", "metric not measured")
    mem.record_panel_objection(1, 1, "rev-b", "metric not measured on the MR branch")
    rows = mem.panel_objections(1, round=1)
    assert len(rows) == 1
    assert rows[0]["objection"] == "metric not measured on the MR branch"
    assert rows[0]["voter"] == "rev-b"


def test_set_panel_transitions_state_round_and_deadline(mem):
    mem.ensure_panel(3)
    mem.set_panel(3, state="defense", due_ts=555.0)
    assert mem.get_panel(3)["state"] == "defense"
    assert mem.get_panel(3)["due_ts"] == 555.0
    mem.set_panel(3, round=2, clear_due=True,
                  defense={"rebuttals": {"O1": "measured again, passes"}})
    p = mem.get_panel(3)
    assert p["round"] == 2 and p["due_ts"] is None
    assert p["defense"]["rebuttals"]["O1"].startswith("measured")


def test_panels_in_state_filters_and_orders(mem):
    for mr in (5, 2, 9):
        mem.ensure_panel(mr)
    mem.set_panel(5, state="defense", due_ts=1.0)
    mem.set_panel(9, state="approved")
    voting = mem.panels_in_state("voting")
    assert [p["mr_id"] for p in voting] == [2]    # 5 and 9 left the state
    live = mem.panels_in_state("voting", "defense")
    assert [p["mr_id"] for p in live] == [2, 5]
    assert mem.panels_in_state("rejected") == []


def test_panel_row_survives_reopen(tmp_path: Path):
    """A fresh SharedMemory on the same DB file keeps panel truth (WAL)."""
    db = tmp_path / "mem.db"
    m1 = SharedMemory(db)
    m1.ensure_panel(4)
    m1.record_panel_vote(4, 1, "rev-x", "reject", "no evidence")
    m2 = SharedMemory(db)
    assert m2.get_panel(4)["state"] == "voting"
    assert m2.panel_votes(4)[0]["voter"] == "rev-x"


# ---------------------------------------------- owner visibility (P19 CLI)
def _full_lifecycle(mem) -> tuple[int, int]:
    """One approved-after-defense MR + one rejected-on-incomplete-defense MR."""
    mem.heartbeat("coder-1", "coder", 1)
    mem.heartbeat("rev-a", "coder", 1)
    mem.heartbeat("rev-b", "writer", 1)
    good = mem.open_review(task_id=None, title="good MR", body="b", author="coder-1",
                           branch="agent/coder-1/g", files=["a.py"])
    bad = mem.open_review(task_id=None, title="bad MR", body="b", author="rev-b",
                          branch="agent/rev-b/b", files=["b.py"])
    mem.ensure_panel(good)                        # the engine creates it on first vote
    mem.ensure_panel(bad)
    mem.record_panel_vote(good, 1, "rev-a", "reject", "metric not measured")
    mem.record_panel_objection(good, 1, "rev-a", "metric not measured")
    mem.record_panel_vote(good, 1, "rev-b", "accept", "fine")
    mem.set_panel(good, state="defense", due_ts=999.0)
    mem.set_panel(good, state="voting", round=2, clear_due=True,
                  defense={"rebuttals": {"rev-a": "measured again, passes"},
                           "evidence_ids": [3], "ts": 123.0})
    mem.record_panel_vote(good, 2, "rev-a", "accept", "fair")
    mem.set_panel(good, state="approved")
    mem.set_review_verdict(good, "approved", "panel[rev-a,rev-b]", "P19 unanimous")
    mem.record_panel_vote(bad, 1, "rev-a", "reject", "no evidence")
    mem.record_panel_objection(bad, 1, "rev-a", "no evidence at all")
    mem.set_panel(bad, state="defense", due_ts=999.0)
    mem.set_review_verdict(bad, "failed", "rev-b",
                           "defense incomplete — no rebuttal for ['rev-a']")
    mem.set_panel(bad, state="rejected")
    return good, bad


def test_panels_overview_joins_mr_and_orders_newest_first(mem):
    good, bad = _full_lifecycle(mem)
    rows = mem.panels_overview()
    assert [r["mr_id"] for r in rows] == [bad, good]      # newest (updated) first
    g = rows[1]
    assert g["state"] == "approved" and g["round"] == 2
    assert g["author"] == "coder-1" and g["title"] == "good MR"
    assert g["verdict_comment"] == "P19 unanimous"
    assert mem.panels_overview(state="rejected")[0]["mr_id"] == bad
    assert mem.panels_overview(state="voting") == []
    assert mem.panels_overview(limit=1)[0]["mr_id"] == bad


def test_panel_stats_counts_outcomes_and_classifies_rejections(mem):
    good, bad = _full_lifecycle(mem)
    st = mem.panel_stats()
    assert st["by_state"] == {"approved": 1, "rejected": 1}
    assert st["decided"] == 2
    assert st["approval_rate"] == pytest.approx(0.5)
    # the good MR reached round 2 with a stored defense -> defended & saved
    assert st["defended"] == 1
    assert st["saved_by_defense"] == 1
    assert st["defense_save_rate"] == pytest.approx(1.0)
    assert st["rejected_why"] == {"defense_incomplete": 1}
    assert st["top_objectors"] == [{"voter": "rev-a", "objections": 2}]
    # votes in the DECISIVE round: r2 accept (good) + r1 reject (bad)
    assert st["votes"] == {"accept": 1, "reject": 1}


def test_panel_stats_with_no_panels_is_all_zeros(mem):
    st = mem.panel_stats()
    assert st["by_state"] == {}
    assert st["decided"] == 0
    assert st["approval_rate"] is None
    assert st["defended"] == 0 and st["saved_by_defense"] == 0
    assert st["defense_save_rate"] is None
    assert st["rejected_why"] == {} and st["top_objectors"] == []


def test_panel_stats_since_filters_old_panels(mem):
    _full_lifecycle(mem)
    future = mem.get_panel(2)["opened_ts"] + 1.0
    assert mem.panel_stats(since_ts=future)["by_state"] == {}
    assert mem.panel_stats(since_ts=0.0)["decided"] == 2
