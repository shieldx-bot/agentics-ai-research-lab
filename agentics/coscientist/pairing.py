"""Tournament pair selection, match-mode choice and verdict parsing.

Ported from the Apache-2.0 Kaimen-Inc/Co-Scientist re-implementation
(co_scientist/agents/ranking.py, pure parts). The reference selects pairs
in three buckets that maximise information per match:

  bucket 1 — a NEW hypothesis (< 3 matches) against its nearest-Elo
             seasoned opponent: fresh ideas must calibrate fast;
  bucket 2 — a close-Elo pair inside the warm set, sampled weighted by
             idea-distance (vector dissimilarity when the lab's vector
             memory is available): debates between similar ratings but
             different approaches carry the most signal;
  bucket 3 — an occasional top-heavy random pull keeps the leaderboard
             connected (a champion that never defends is not a champion).

Debate mode (a structured multi-turn deliberation prompt) is used for
inexperienced or closely-matched pairs; plain pairwise comparison for
clearly-separated veterans.
"""
from __future__ import annotations

import math
import random
import re
from typing import Literal

ELO_INITIAL = 1200.0

PairMode = Literal["pairwise", "debate"]

# debate while either side is this inexperienced, or ratings are this close
DEBATE_WHEN_MATCHES_LT = 3
DEBATE_WHEN_ELO_DELTA_LT = 200.0


def select_mode(a_matches: int, b_matches: int, elo_a: float,
                elo_b: float) -> PairMode:
    """Debate for new/close matchups, plain pairwise for separated veterans."""
    if min(a_matches, b_matches) < DEBATE_WHEN_MATCHES_LT:
        return "debate"
    if abs(elo_a - elo_b) < DEBATE_WHEN_ELO_DELTA_LT:
        return "debate"
    return "pairwise"


def select_pair(
    pool: list[dict],
    *,
    rng: random.Random | None = None,
    p_new: float = 0.4,
    p_close: float = 0.4,
    similarity=None,                     # callable(a, b) -> float | None
) -> tuple[dict, dict] | None:
    """Pick the next matchup from ``pool`` (list of hypothesis rows).

    Each row needs: ``id``, ``elo`` (or None -> ELO_INITIAL),
    ``matches_played``. Returns None when a meaningful pair cannot be
    formed (fewer than 2 hypotheses). Mirrors the reference's three
    buckets; the similarity callback is optional and every failure of
    the vector tier degrades to unweighted choice, never an error.
    """
    if len(pool) < 2:
        return None
    rng = rng or random.random
    roll = rng() if callable(rng) else rng.random()

    def elo(h: dict) -> float:
        return float(h.get("elo") if h.get("elo") is not None else ELO_INITIAL)

    def matches(h: dict) -> int:
        return int(h.get("matches_played") or 0)

    new_hyps = [h for h in pool if matches(h) < DEBATE_WHEN_MATCHES_LT]
    warm = [h for h in pool if matches(h) >= DEBATE_WHEN_MATCHES_LT]

    # bucket 1: new entrant vs nearest-Elo seasoned opponent
    if roll < p_new and new_hyps and warm:
        a = rng_choice(rng, new_hyps)
        b = min(warm, key=lambda h: abs(elo(h) - elo(a)))
        return a, b

    # bucket 2: close-Elo pair in the warm set, weighted by idea distance
    if roll < p_new + p_close and len(warm) >= 2:
        pair = _sample_close_elo(warm, elo, similarity, rng)
        if pair is not None:
            return pair

    # bucket 3: random top-heavy pull (keeps the leaderboard connected)
    ranked = sorted(pool, key=lambda h: -elo(h))
    top = ranked[: max(2, len(ranked) // 2)]
    if len(top) >= 2:
        a = rng_choice(rng, top)
        b = rng_choice(rng, [h for h in top if h is not a])
        if b is not None:
            return a, b
    return None


def _sample_close_elo(pool, elo_of, similarity, rng):
    """Among pairs with |Δelo| < 200, sample weighted by exp(-Δelo/200)*(1-sim)."""
    pairs, weights = [], []
    for i, a in enumerate(pool):
        for b in pool[i + 1:]:
            d_elo = abs(elo_of(a) - elo_of(b))
            if d_elo > 200.0:
                continue
            sim = None
            if similarity is not None:
                try:
                    sim = similarity(a, b)
                except Exception:          # noqa: BLE001 — vector tier must never gate
                    sim = None
            w_sim = 1.0 - (sim if sim is not None else 0.0)
            w = math.exp(-d_elo / 200.0) * max(w_sim, 0.05)
            pairs.append((a, b))
            weights.append(w)
            if len(pairs) >= 20:           # cap, as the reference does
                break
        if len(pairs) >= 20:
            break
    if not pairs:
        return None
    total = sum(weights)
    if total <= 0:
        return rng_choice(rng, pairs)
    r = rng() if callable(rng) else rng.random()
    cum = 0.0
    for w, pair in zip(weights, pairs):
        cum += w
        if cum >= r:
            return pair
    return pairs[-1]


def rng_choice(rng, seq):
    """random.choice via an injected rng (callable returning [0,1) or Random)."""
    seq = list(seq)
    if not seq:
        return None
    r = rng() if callable(rng) else rng.random()
    return seq[min(int(r * len(seq)), len(seq) - 1)]


_VERDICT_DIGIT_RE = re.compile(r"^[\W_]*\**\s*([12])\b")


def parse_better_idea(text: str) -> int | None:
    """Find the trailing ``better idea: 1|2`` marker (case-insensitive).

    Accepts the marker on any line, bold, or 'option/hypothesis 1'
    phrasings — the same tolerant parse as the reference — and still
    rejects ambiguous answers like 'better idea: 12'.
    """
    if not text:
        return None
    for line in reversed(text.strip().splitlines()):
        low = line.strip().lower()
        if "better idea" in low and ":" in low:
            tail = low.split(":", 1)[1].strip()
            m = _VERDICT_DIGIT_RE.match(tail)
            if m:
                return int(m.group(1))
            for keyword in ("option", "hypothesis", "hyp"):
                if tail.startswith(keyword):
                    rest = tail[len(keyword):].lstrip()
                    m2 = _VERDICT_DIGIT_RE.match(rest)
                    if m2:
                        return int(m2.group(1))
    return None


# ------------------------------------------------------------------- prompts
PAIRWISE_PROMPT = """You are an expert evaluator comparing two research hypotheses for a self-improving AI research lab.

Goal of the lab: {goal}

Hypothesis 1:
<HYPOTHESIS_TEXT id="{id1}">
{text1}
</HYPOTHESIS_TEXT>

Hypothesis 2:
<HYPOTHESIS_TEXT id="{id2}">
{text2}
</HYPOTHESIS_TEXT>

{feedback_block}Evaluate which hypothesis is the stronger RESEARCH DIRECTION: correctness plausibility, novelty, testability with a small self-contained python simulation, and utility to the lab's goal. The lab verifies hypotheses by RUNNING code, so a hypothesis that names a measurable metric is worth more than a vague vision.

Be decisive. End your response with the exact line:
better idea: <1 or 2>
"""

DEBATE_PROMPT = """You are simulating a panel of domain experts in a structured debate evaluating two competing research hypotheses. Only one can be pursued; identify the optimal choice.

Goal of the lab: {goal}

Hypothesis 1:
<HYPOTHESIS_TEXT id="{id1}">
{text1}
</HYPOTHESIS_TEXT>

Hypothesis 2:
<HYPOTHESIS_TEXT id="{id2}">
{text2}
</HYPOTHESIS_TEXT>

{feedback_block}Debate procedure (3-5 turns, in one response, labelled Turn 1, Turn 2, ...):
- Turn 1: concise summary of both hypotheses.
- Later turns: critically evaluate each against the goal — potential correctness, utility, specificity, novelty, and how cleanly a small python protocol could MEASURE it. Identify weaknesses and flaws in both.
- Final turn: conclusive judgment with the rationale.

Then indicate the superior hypothesis by writing the exact line:
better idea: <1 or 2>
"""

EVOLVE_COMBINE_PROMPT = """You are an expert in scientific synthesis working for a self-improving AI research lab.

Goal of the lab: {goal}

Combine the best parts of the two hypotheses below into ONE new, stronger hypothesis. The result must (a) preserve what works in each, (b) explicitly resolve any contradictions between them, and (c) be MORE SPECIFIC and testable than either parent — the lab verifies hypotheses by running small python protocols that print AGENTICS_RESULT: {{"<metric>": <number>}}.

Hypothesis A:
<HYPOTHESIS_TEXT id="{id1}">
{text1}
</HYPOTHESIS_TEXT>

Hypothesis B:
<HYPOTHESIS_TEXT id="{id2}">
{text2}
</HYPOTHESIS_TEXT>

{feedback_block}Reply with the new hypothesis ONLY, in this exact shape (no preamble):
TITLE: <one line>
CLAIM: <the full hypothesis statement, falsifiable and measurable>
METRIC: <a short snake_case metric name the protocol would measure>
"""

EVOLVE_OOB_PROMPT = """You are an expert researcher generating a NOVEL hypothesis for a self-improving AI research lab, inspired by — but not aggregating — the ideas below.

Goal of the lab: {goal}

Inspiration (use analogy and cross-domain leaps, not direct replication):
{inspirations}

{feedback_block}Think out-of-the-box: propose ONE original, specific, falsifiable hypothesis the lab can test by running a small self-contained python simulation that prints AGENTICS_RESULT: {{"<metric>": <number>}}. Avoid recombining the inspirations verbatim; find the principle BEHIND them.

Reply with the new hypothesis ONLY, in this exact shape (no preamble):
TITLE: <one line>
CLAIM: <the full hypothesis statement, falsifiable and measurable>
METRIC: <a short snake_case metric name the protocol would measure>
"""

META_REVIEW_PROMPT = """You are the meta-reviewer of an AI research lab running an Elo tournament over research hypotheses.

Goal of the lab: {goal}

Recent tournament debates (what won and what lost, with the jury's rationale):
{debates}

Instructions:
- Identify RECURRING patterns: what do winning hypotheses share, what do losing ones share?
- Produce actionable guidance for generating the next generation of hypotheses.
- Do not evaluate individual hypotheses; synthesise the pattern.

Reply with a compact steering note (max 120 words) the lab will inject into future generation/debate prompts. No preamble, no headings.
"""
