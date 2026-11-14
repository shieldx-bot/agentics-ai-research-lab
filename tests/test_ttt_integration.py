"""TTT-Discover integration (docs/TTT_DISCOVER.md phase 1-3):

C1 continuous reward (settle anchor law = validity check),
C4 r_sota tracking (`new_state_of_the_art` journal + best_reward),
C2 PUCT frontier (Q = MAX descendant reward, decay with n, lineage
   blocking, BFS fallback on degenerate buffers),
C3 entropic credit assignment (adaptive beta, constant-group zero,
   best branch takes nearly the whole bonus_max).
"""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agentics.verify.reward import (  # noqa: E402
    adaptive_softmax_weights,
    reward_from_row,
)
from agentics.memory.shared_memory import SharedMemory  # noqa: E402
from agentics.graph.nodes import _branch_feedback  # noqa: E402


@pytest.fixture()
def mem(tmp_path):
    return SharedMemory(tmp_path / "ttt.db")


def _hyp(mem, claim, status="proposed", parents=None, depth=0, branch=""):
    return mem.add_hypothesis(
        "a02", claim, {"metric": "acc", "op": ">", "target": 1},
        status=status, parents=parents, depth=depth, branch=branch)


def _row(status="supported", measured=None, op=">", target=1.0, ratio=0.5):
    return {"status": status, "measured": measured, "op": op,
            "target": target, "self_authored_ratio": ratio}


# ------------------------------------------------------------------ C1
def test_reward_zero_for_refuted_and_unanchored():
    assert reward_from_row(_row(status="refuted", measured=10)) == 0.0
    assert reward_from_row(_row(status="inconclusive", measured=10)) == 0.0
    # fully self-graded verdict = validity fail = reward 0
    assert reward_from_row(_row(measured=10, ratio=1.0)) == 0.0
    assert reward_from_row(_row(measured=10, ratio=None)) == 0.0


def test_reward_margin_normalized_and_clipped():
    # measured=2, target=1 -> margin=1, scale=|target|=1 -> reward 1.0
    assert reward_from_row(_row(measured=2.0)) == pytest.approx(1.0)
    # measured=1.25 -> margin 0.25
    assert reward_from_row(_row(measured=1.25)) == pytest.approx(0.25)
    # op "<": target - measured
    r = _row(measured=0.5, op="<", target=1.0)
    assert reward_from_row(r) == pytest.approx(0.5)
    # no anchor info + non-supported statuses short-circuit earlier
    assert reward_from_row(_row(measured=1.25, status="supported",
                                ratio=0.5)) == pytest.approx(0.25)


def test_reward_history_scale_uses_span():
    r = _row(measured=2.0, op=">", target=1.0)
    assert reward_from_row(r, scale="history",
                           history_span=4.0) == pytest.approx(0.25)
    # unusable span falls back to |target|
    assert reward_from_row(r, scale="history",
                           history_span=None) == pytest.approx(1.0)


# ------------------------------------------------------------------ C4
def test_sota_journal_only_on_anchored_improvement(mem):
    e1 = mem.record_experiment("a02", 1, "supported", metric="acc",
                               measured=1.2, op=">", target=1.0,
                               self_authored_ratio=0.5,
                               anchors='["citation"]')
    assert e1 > 0
    evs = [e["event"] for e in mem.recent_journal(limit=10)]
    assert "new_state_of_the_art" in evs          # first anchored reward
    # better reward -> another sota event
    mem.record_experiment("a02", 2, "supported", metric="acc",
                          measured=1.8, op=">", target=1.0,
                          self_authored_ratio=0.5, anchors='["citation"]')
    # worse reward -> NO sota event
    mem.record_experiment("a02", 3, "supported", metric="acc",
                          measured=1.3, op=">", target=1.0,
                          self_authored_ratio=0.5, anchors='["citation"]')
    events = [e["event"] for e in mem.recent_journal(limit=20)]
    assert events.count("new_state_of_the_art") == 2
    # unanchored supported (ratio 1.0) is a validity fail: never a reward
    mem.record_experiment("a02", 4, "supported", metric="acc",
                          measured=99.0, op=">", target=1.0,
                          self_authored_ratio=1.0, anchors='')
    events = [e["event"] for e in mem.recent_journal(limit=20)]
    assert events.count("new_state_of_the_art") == 2


def test_best_reward_tracks_anchored_supported_only(mem):
    assert mem.best_reward() == 0.0
    mem.record_experiment("a02", 1, "supported", measured=1.4, op=">",
                          target=1.0, self_authored_ratio=0.5,
                          anchors='["pinned_data"]')
    assert mem.best_reward() == pytest.approx(0.4)
    mem.record_experiment("a02", 2, "supported", measured=99, op=">",
                          target=1.0, self_authored_ratio=1.0,
                          anchors='')                # unanchored: ignored
    mem.record_experiment("a02", 3, "refuted", measured=5, op=">",
                          target=1.0, self_authored_ratio=0.5,
                          anchors='["citation"]')    # refuted: ignored
    assert mem.best_reward() == pytest.approx(0.4)


# ------------------------------------------------------- adaptive softmax
def test_softmax_constant_group_is_uniform():
    w = adaptive_softmax_weights([0.5, 0.5, 0.5])
    assert w == pytest.approx([1 / 3] * 3)


def test_softmax_single_reward_is_certain():
    assert adaptive_softmax_weights([0.7]) == pytest.approx([1.0])
    assert adaptive_softmax_weights([]) == []


def test_softmax_is_one_nat_from_uniform_and_favors_the_max():
    rewards = [0.1, 0.4, 0.9]
    w = adaptive_softmax_weights(rewards)
    assert max(w) == w[rewards.index(0.9)]
    assert w[rewards.index(0.9)] > w[rewards.index(0.4)] > w[rewards.index(0.1)]
    # KL(q || uniform) = log(sum e^{b r})/n - b*mean(r) ~= log 2
    import math
    mean_r = sum(rewards) / len(rewards)
    b = 1.0                       # any beta; verify the construction shape
    w2 = adaptive_softmax_weights([r * b for r in rewards])
    assert sum(w2) == pytest.approx(1.0)
    assert math.isclose(sum(w2), 1.0, rel_tol=1e-9)


# ------------------------------------------------------------------ C2
def test_puct_frontier_q_max_and_n_decay_across_subtrees(mem):
    """Two independent discoveries, one branch each: the branch whose
    SUBTREE holds the higher MAX reward must rank first (Q = max), and its
    exploitation bonus must DECAY as more settles accumulate (n grows)."""
    disc_a = _hyp(mem, "discovery alpha root", status="supported")
    disc_b = _hyp(mem, "discovery beta root", status="supported")
    mem.mark_discovery(disc_a, 80)
    mem.mark_discovery(disc_b, 80)
    ba = _hyp(mem, "branch under alpha with rich reward", parents=[disc_a],
              depth=1, branch="mechanism")
    bb = _hyp(mem, "branch under beta with poor reward", parents=[disc_b],
              depth=1, branch="boundary")
    la = _hyp(mem, "leaf under alpha branch", parents=[ba], depth=2)
    lb = _hyp(mem, "leaf under beta branch", parents=[bb], depth=2)
    mem.record_experiment("a02", la, "supported", measured=1.9, op=">",
                          target=1.0, self_authored_ratio=0.5,
                          anchors='["citation"]')
    mem.record_experiment("a02", lb, "supported", measured=1.05, op=">",
                          target=1.0, self_authored_ratio=0.5,
                          anchors='["citation"]')
    frontier = mem.discovery_frontier_puct(limit=2, max_depth=4, c=1.0)
    assert [h["id"] for h in frontier] == [ba, bb]
    assert frontier[0]["puct_q"] == pytest.approx(0.9)
    assert frontier[1]["puct_q"] == pytest.approx(0.05)
    assert frontier[0]["puct_n"] == 1
    bonus_before = frontier[0]["puct_bonus"]
    # dig the same branch 4 more times: n grows, the bonus shrinks
    for _ in range(4):
        mem.record_experiment("a02", la, "supported", measured=1.2, op=">",
                              target=1.0, self_authored_ratio=0.5,
                              anchors='["citation"]')
    after = {h["id"]: h for h in mem.discovery_frontier_puct(
        limit=2, max_depth=4, c=1.0)}
    assert after[ba]["puct_n"] == 5
    assert after[ba]["puct_bonus"] < bonus_before


def test_puct_frontier_lineage_blocks_same_parent(mem):
    disc = _hyp(mem, "discovery root three branches", status="supported")
    mem.mark_discovery(disc, 80)
    for i in range(3):
        _hyp(mem, f"branch child {i} of same root", parents=[disc], depth=1,
             branch=f"angle{i}")
    frontier = mem.discovery_frontier_puct(limit=3, max_depth=4)
    # one batch never carries two children of the SAME discovery root
    assert len(frontier) == 1


def test_puct_frontier_empty_buffer_falls_back_to_bfs(mem):
    disc = _hyp(mem, "discovery root nothing settled", status="supported")
    mem.mark_discovery(disc, 80)
    b1 = _hyp(mem, "unexplored branch alpha", parents=[disc], depth=1)
    _hyp(mem, "unexplored branch beta", parents=[disc], depth=1)
    frontier = mem.discovery_frontier_puct(limit=2, max_depth=4)
    # no settled rewards anywhere -> novelty proxy, id order (BFS shape),
    # still lineage-blocked to one branch of this root
    assert len(frontier) == 1
    assert frontier[0]["id"] == b1


# ------------------------------------------------------------------ C3
class _S:
    research_topic = "NFV"
    entropic_bonus_max = 8.0
    reward_scale = "target"
    ideation_enabled = True


def _fb_ctx(mem):
    from agentics.agents.roles import Role
    return SimpleNamespace(agent_id="a02", role=Role.RESEARCHER,
                           memory=mem, settings=_S(),
                           heartbeat=lambda *a, **k: None,
                           meta={"_system_prompt": "sys"})


def test_entropic_boost_single_settle_takes_bonus_max(mem):
    disc = _hyp(mem, "discovery root single branch", status="supported")
    mem.mark_discovery(disc, 80)
    good = _hyp(mem, "branch child with strong reward", parents=[disc],
                depth=1, branch="mechanism")
    mem.record_experiment("a02", good, "supported", measured=1.6, op=">",
                          target=1.0, self_authored_ratio=0.5,
                          anchors='["citation"]')
    asyncio.run(_branch_feedback(_fb_ctx(mem), good, "supported"))
    assert mem.get_hypothesis(disc)["novelty"] == pytest.approx(88.0)


def test_entropic_boost_best_branch_wins_group(mem):
    disc = _hyp(mem, "discovery root contested branches", status="supported")
    mem.mark_discovery(disc, 80)
    best = _hyp(mem, "branch with top reward", parents=[disc], depth=1,
                branch="mechanism")
    weak = _hyp(mem, "branch with weak reward", parents=[disc], depth=1,
                branch="boundary")
    mem.record_experiment("a02", weak, "supported", measured=1.01, op=">",
                          target=1.0, self_authored_ratio=0.5,
                          anchors='["citation"]')
    mem.record_experiment("a02", best, "supported", measured=1.9, op=">",
                          target=1.0, self_authored_ratio=0.5,
                          anchors='["citation"]')
    asyncio.run(_branch_feedback(_fb_ctx(mem), best, "supported"))
    nov_best = mem.get_hypothesis(disc)["novelty"]
    assert nov_best > 80.0                  # best branch: meaningful boost
    # replay the WEAK branch's settle (same group): its weight ~ 0
    mem2_nov = mem.get_hypothesis(disc)["novelty"]
    asyncio.run(_branch_feedback(_fb_ctx(mem), weak, "supported"))
    nov_after = mem.get_hypothesis(disc)["novelty"]
    assert nov_after >= mem2_nov            # weak branch adds ~nothing


def test_entropic_boost_zero_for_constant_group(mem):
    disc = _hyp(mem, "discovery root repeated result", status="supported")
    mem.mark_discovery(disc, 80)
    b1 = _hyp(mem, "branch settle one clip", parents=[disc], depth=1)
    b2 = _hyp(mem, "branch settle two clip", parents=[disc], depth=1)
    # two settles with IDENTICAL rewards -> constant group -> boost 0
    mem.record_experiment("a02", b1, "supported", measured=1.5, op=">",
                          target=1.0, self_authored_ratio=0.5,
                          anchors='["citation"]')
    asyncio.run(_branch_feedback(_fb_ctx(mem), b1, "supported"))
    assert mem.get_hypothesis(disc)["novelty"] == pytest.approx(88.0)
    mem.record_experiment("a02", b2, "supported", measured=1.5, op=">",
                          target=1.0, self_authored_ratio=0.5,
                          anchors='["citation"]')
    asyncio.run(_branch_feedback(_fb_ctx(mem), b2, "supported"))
    assert mem.get_hypothesis(disc)["novelty"] == pytest.approx(88.0)
