"""Performance–Novelty scoring: paper-faithful layer + lab adaptation."""

import math

import pytest

from agentics.evolution.scoring import (
    IssueInfo,
    build_profile,
    categorize_issue,
    cosine_distance,
    novelty,
    performance_novelty_scores,
    rank_issues,
    score_issue,
    select_parent_group,
)


# ------------------------------------------------------------ Algorithm 1
def test_cosine_distance_basics():
    assert cosine_distance([1, 1, 0], [1, 1, 0]) == pytest.approx(0.0, abs=1e-6)
    assert cosine_distance([1, 0], [0, 1]) == pytest.approx(1.0)
    # paper quirk: two all-zero vectors are "maximally novel" (cold start)
    assert cosine_distance([0, 0], [0, 0]) == pytest.approx(1.0)


def test_scores_match_the_formula_computed_independently():
    vecs = {
        "a": [1, 1, 0, 0, 1],
        "b": [1, 0, 0, 0, 1],
        "c": [0, 1, 1, 0, 0],
        "d": [0, 0, 1, 1, 0],
        "e": [1, 1, 1, 1, 1],
    }
    perf = {"a": 0.9, "b": 0.5, "c": 0.4, "d": 0.7, "e": 0.8}
    m = 2
    got = performance_novelty_scores(vecs, perf, m)

    def dist(x, y):
        dot = sum(p * q for p, q in zip(x, y))
        return 1 - dot / (
            math.sqrt(sum(p * p for p in x)) * math.sqrt(sum(q * q for q in y)) + 1e-8
        )

    for i in vecs:
        nearest = sorted(dist(vecs[i], vecs[j]) for j in vecs if j != i)[:m]
        expected = perf[i] * math.sqrt(sum(nearest) / m)
        assert got[i] == pytest.approx(expected)


def test_select_parent_group_prefers_novel_when_performance_is_equal():
    vecs = {"x1": [1, 1, 0, 0], "x2": [1, 1, 0, 0], "y": [0, 0, 1, 1]}
    perf = {k: 0.5 for k in vecs}
    # M=1: x1/x2 are clones (novelty ~0), y is far from both -> y first
    assert select_parent_group(vecs, perf, k=1, m=1) == ["y"]
    assert select_parent_group(vecs, perf, k=2, m=1)[0] == "y"


def test_select_parent_group_is_deterministic_and_handles_small_archives():
    assert select_parent_group({"only": [1, 0]}, {"only": 0.4}, k=2, m=4) == ["only"]
    assert novelty("only", {"only": [1, 0]}) == 1.0  # no peers -> neutral-high, not 0
    vecs = {"b": [1, 0], "a": [1, 0]}
    assert select_parent_group(vecs, {"a": 0.5, "b": 0.5}, k=2) == ["a", "b"]  # id tie-break


# ------------------------------------------------------------ categorisation
def test_categorize_issue_label_beats_keywords():
    assert categorize_issue("Fix the bug in docs", labels=["kind/research"]) == "research"
    assert categorize_issue("anything", labels=["documentation"]) == "docs"
    assert categorize_issue("Fix crash when claims expire") == "bugfix"
    assert categorize_issue("Survey SOTA for VNF placement") == "research"
    assert categorize_issue("Write README section") == "docs"
    assert categorize_issue("self-evolution: refine prompts") == "evolution"
    assert categorize_issue("Implement a thing") == "other"


# ------------------------------------------------------------ profile + ranking
def _issue(n, title, ts, labels=()):
    return IssueInfo(number=n, title=title, labels=tuple(labels), created_ts=ts)


def test_new_agent_keeps_fifo_behaviour():
    prof = build_profile([])
    issues = [_issue(3, "Write docs", 30), _issue(1, "Fix bug", 10), _issue(2, "Survey papers", 20)]
    order = [i.number for i, _ in rank_issues(issues, prof)]
    assert order == [1, 2, 3]  # all tie -> oldest first, like before


def test_saturated_specialist_is_steered_to_a_new_area_but_not_zeroed():
    prof = build_profile([("bugfix", "merged")] * 5)  # only ever fixes bugs, always merged
    older_bug, newer_docs = _issue(1, "Fix crash", 10), _issue(2, "Write docs", 20)
    ranked = rank_issues([older_bug, newer_docs], prof)
    assert [i.number for i, _ in ranked] == [2, 1]  # exploration wins over FIFO
    assert score_issue(prof, older_bug) > 0  # specialty still competitive


def test_unseen_category_is_not_punished():
    prof = build_profile([("code", "merged"), ("code", "failed")])
    assert prof.alpha("docs") == pytest.approx(prof.overall_rate())  # unknown != bad


def test_failing_in_a_category_lowers_its_rank():
    prof = build_profile([("research", "failed")] * 4 + [("docs", "merged")] * 4)
    research, docs = _issue(1, "Survey SOTA papers", 10), _issue(2, "Write docs", 20)
    assert prof.alpha("research") < prof.alpha("docs")
    # equal share of history -> same novelty -> performance decides
    assert rank_issues([research, docs], prof)[0][0].number == 2


def test_priority_label_outranks():
    prof = build_profile([])
    issues = [_issue(1, "Fix bug", 10), _issue(2, "Fix bug too", 20, ["P0"])]
    assert rank_issues(issues, prof)[0][0].number == 2


def test_success_vector_is_binary_over_categories():
    prof = build_profile([("code", "merged"), ("docs", "failed")])
    vec = prof.success_vector(("research", "code", "docs"))
    assert vec == [0, 1, 0]
