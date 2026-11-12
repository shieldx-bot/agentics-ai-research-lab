"""Performance–Novelty scoring.

Two layers, kept deliberately separate:

1. ``cosine_distance`` / ``novelty`` / ``performance_novelty_scores`` /
   ``select_parent_group`` — a faithful implementation of Algorithm 1 in
   *Group-Evolving Agents* (Weng et al., 2026): score(i) = alpha_i * sqrt(nov(i)),
   nov(i) = mean cosine distance to the M nearest neighbours of a binary
   task-success vector. Generic: works on any set of "individuals"
   (agents, prompt variants...).

2. ``rank_issues`` — a lab-specific *adaptation* (NOT in the paper). The lab has
   no fixed benchmark, so "task-success vector" is replaced by per-category
   merge history, and the score for (agent, issue) keeps the paper's shape
   (performance x sqrt(novelty)) with a floor so an agent's own specialty is
   never scored zero. See docs/EVOLUTION.md for the limits of this proxy.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

EPS = 1e-8

CATEGORIES: tuple[str, ...] = (
    "research", "code", "docs", "experiment", "bugfix", "infra", "evolution", "other",
)


# ======================================================================
# Layer 1 — Algorithm 1 (paper-faithful)
# ======================================================================
def cosine_distance(a: Sequence[float], b: Sequence[float], eps: float = EPS) -> float:
    """d(i,j) = 1 - z_i.z_j / (|z_i||z_j| + eps)   (Eq. 1).

    Quirk inherited from the paper's formula: two all-zero vectors have
    distance 1 (maximally novel) because the numerator is 0 and eps keeps the
    denominator positive. For a cold-start archive this acts as exploration.
    """
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return max(0.0, 1.0 - dot / (na * nb + eps))


def novelty(
    target: str, vectors: Mapping[str, Sequence[float]], m: int = 4,
) -> float:
    """nov(i): mean distance to the M most similar *other* individuals (Eq. 2).

    With fewer than M peers uses all of them; a lone individual has no
    neighbours, so it gets the neutral-high value 1.0 instead of 0 (which would
    zero its score).
    """
    dists = sorted(
        cosine_distance(vectors[target], v) for k, v in vectors.items() if k != target
    )
    if not dists:
        return 1.0
    nearest = dists[: max(1, min(m, len(dists)))]
    return sum(nearest) / len(nearest)


def performance_novelty_scores(
    vectors: Mapping[str, Sequence[float]],
    performance: Mapping[str, float],
    m: int = 4,
) -> dict[str, float]:
    """score(i) = alpha_i * sqrt(nov(i))   (Eq. 3)."""
    return {
        i: float(performance.get(i, 0.0)) * math.sqrt(novelty(i, vectors, m))
        for i in vectors
    }


def select_parent_group(
    vectors: Mapping[str, Sequence[float]],
    performance: Mapping[str, float],
    k: int = 2,
    m: int = 4,
) -> list[str]:
    """Top-K individuals by Performance–Novelty score (deterministic tie-break by id)."""
    scores = performance_novelty_scores(vectors, performance, m)
    ranked = sorted(scores, key=lambda i: (-scores[i], i))
    return ranked[:k]


# ======================================================================
# Layer 2 — lab adaptation: (agent, issue) ranking
# ======================================================================
_LABEL_TO_CATEGORY = {
    "bug": "bugfix", "documentation": "docs", "enhancement": "code",
    "research": "research", "literature": "research", "experiment": "experiment",
    "infra": "infra", "ci": "infra",
}
_KEYWORD_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("evolution", re.compile(r"self[- ]evolution", re.I)),
    ("bugfix", re.compile(r"\b(bug|fix|crash|regression|traceback)\b", re.I)),
    ("docs", re.compile(r"\b(docs?|documentation|readme|report|write-?up)\b", re.I)),
    ("research", re.compile(
        r"\b(survey|literature|papers?|state[- ]of[- ]the[- ]art|sota)\b", re.I)),
    ("experiment", re.compile(
        r"\b(experiment|benchmark|evaluate|ablation|baseline)\b", re.I)),
    ("infra", re.compile(r"\b(ci|workflows?|docker|deploy|pipeline|config)\b", re.I)),
)
_PRIORITY = {
    "p0": 3.0, "priority/critical": 3.0,
    "p1": 2.0, "priority/high": 2.0,
    "p3": 0.5, "priority/low": 0.5,
}


def categorize_issue(title: str, body: str = "", labels: Iterable[str] = ()) -> str:
    """Explicit ``kind/<category>`` label wins, then well-known labels, then keywords."""
    lowered = [lb.lower() for lb in labels]
    for lb in lowered:
        if lb.startswith("kind/") and lb[5:] in CATEGORIES:
            return lb[5:]
    for lb in lowered:
        if lb in _LABEL_TO_CATEGORY:
            return _LABEL_TO_CATEGORY[lb]
    text = f"{title}\n{body[:500]}"
    for cat, pattern in _KEYWORD_RULES:
        if pattern.search(text):
            return cat
    return "other"


def priority_from_labels(labels: Iterable[str]) -> float:
    return max((_PRIORITY.get(lb.lower(), 1.0) for lb in labels), default=1.0)


@dataclass
class IssueInfo:
    number: int
    title: str
    body: str = ""
    labels: tuple[str, ...] = ()
    created_ts: float = 0.0

    @property
    def category(self) -> str:
        return categorize_issue(self.title, self.body, self.labels)

    @property
    def priority(self) -> float:
        return priority_from_labels(self.labels)


@dataclass
class AgentProfile:
    """What an agent has actually delivered, per issue category."""
    attempts: dict[str, int] = field(default_factory=dict)
    successes: dict[str, int] = field(default_factory=dict)

    @property
    def total_attempts(self) -> int:
        return sum(self.attempts.values())

    def overall_rate(self) -> float:
        """Laplace-smoothed overall merge rate (0.5 for a brand-new agent)."""
        return (sum(self.successes.values()) + 1) / (self.total_attempts + 2)

    def alpha(self, category: str, strength: float = 2.0) -> float:
        """Performance in `category`, shrunk toward the agent's overall rate so an
        unseen category is not punished (unknown != bad)."""
        prior = self.overall_rate()
        return (self.successes.get(category, 0) + strength * prior) / (
            self.attempts.get(category, 0) + strength
        )

    def novelty(self, category: str) -> float:
        """1 - share of this agent's history spent in `category` (1.0 if no history)."""
        total = self.total_attempts
        if total == 0:
            return 1.0
        return 1.0 - self.attempts.get(category, 0) / total

    def success_vector(self, categories: Sequence[str] = CATEGORIES) -> list[int]:
        """z_i in {0,1}^D — Algorithm 1's representation, over categories."""
        return [1 if self.successes.get(c, 0) > 0 else 0 for c in categories]


def build_profile(outcomes: Iterable[tuple[str, str]]) -> AgentProfile:
    """outcomes: (category, outcome) with outcome in merged | rejected | failed."""
    prof = AgentProfile()
    for category, outcome in outcomes:
        prof.attempts[category] = prof.attempts.get(category, 0) + 1
        if outcome == "merged":
            prof.successes[category] = prof.successes.get(category, 0) + 1
    return prof


def score_issue(
    profile: AgentProfile, issue: IssueInfo, novelty_floor: float = 0.25,
) -> float:
    """alpha(agent, category) * sqrt(floor + novelty) * priority.

    `novelty_floor` keeps an agent's saturated specialty competitive: with 0.25
    the most-novel category is favoured by at most ~2.2x, a mild bias like the
    paper's sqrt, instead of an exploration-only rule.
    """
    cat = issue.category
    return (
        profile.alpha(cat)
        * math.sqrt(novelty_floor + profile.novelty(cat))
        * issue.priority
    )


def rank_issues(
    issues: Sequence[IssueInfo], profile: AgentProfile, novelty_floor: float = 0.25,
) -> list[tuple[IssueInfo, float]]:
    """Best first; ties fall back to oldest issue (== the previous FIFO behaviour)."""
    scored = [(iss, score_issue(profile, iss, novelty_floor)) for iss in issues]
    scored.sort(key=lambda t: (-round(t[1], 9), t[0].created_ts, t[0].number))
    return scored
