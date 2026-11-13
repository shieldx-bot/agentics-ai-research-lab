"""QD ideation closed loop — IDEAgent qualification gate + Kosmos insight
harvest, wired onto the deep-dive registry (2026-11-14, "integrate deeper"
round; deepens commit 4011788's world-model + diversity gates).

Three mechanisms, one per source paper:

- Kosmos (arXiv 2511.02824) CLOSES THE LOOP: valuable findings scale with
  cycles only when experiment results feed the next generation. Two ported
  loops:
    * INSIGHT HARVEST — after a deep-dive hypothesis settles, one LLM call
      proposes 0-2 FOLLOW-UP hypotheses grounded in the measured result
      (children of the settled row, lineage kept via `parents`).
    * GAP PROPOSAL — the world model is a TASK PROPOSER, not just context:
      when the topic is saturated AND the registry is empty, the agent
      queries the world model and proposes ONE gap-closing hypothesis
      instead of idling (autoresearch's NEVER-STOP clause).
- IDEAgent (arXiv 2607.22375) QUALIFIES ideas BEFORE they cost a protocol
  run: a judge scores non-obviousness/clarity/feasibility (0-100) and
  soundness (0-9); the qualification gate accepts at tau>=60, routes
  near-misses (within delta=20) through ONE repair round, and archives
  failures as COMPACT rejected patterns that later seed prompts receive
  (the R archive) — full rejected ideas would bloat context.
- autoresearch (karpathy/autoresearch) supplies the loop DISCIPLINE:
  every settle lands in an `experiment_ledger` row (keep/discard semantics:
  supported=keep, refuted=inconclusive=discard, crash=inconclusive with
  guard note), near-miss `inconclusive` hypotheses get a bounded RETRY with
  the previous failure note in the prompt ("if you cannot get things to
  work, attempt a fix; if the idea is fundamentally broken, move on"),
  and the simplicity criterion: the repair prompt demands the SIMPLEST
  corrected protocol, not a more elaborate one.

Everything mechanical (gate math, parsers) is dependency-free and pure so
it runs in tests without an LLM. LLM orchestration lives in graph nodes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

# IDEAgent primary search thresholds: tau_N = tau_S = tau_C = 60, diversity
# handled separately by the PEM word-Jaccard gate (diversity_jaccard_max).
DEFAULT_TAU = 60
# Near-miss margin: a candidate that misses ANY threshold by <= delta is
# eligible for exactly one repair round (IDEAgent Stage II).
DEFAULT_REPAIR_MARGIN = 20
_SOUNDNESS_SCALE = 9.0

_ALLOWED_OPS = (">=", "<=", ">", "<")


@dataclass(frozen=True)
class Judgement:
    """One candidate's evaluator outcome (IDEAgent Quality+Soundness)."""
    non_obviousness: float   # 0-100: genuine conceptual leap?
    clarity: float           # 0-100: causal steps described?
    feasibility: float       # 0-100: plausible implementation path?
    soundness: float         # 0-9:   logical/mathematical consistency
    notes: str = ""

    def soundness_100(self) -> float:
        return self.soundness / _SOUNDNESS_SCALE * 100.0

    def quality(self) -> float:
        """IDEAgent Q(I) = 0.7*N + 0.2*S + 0.1*C (feasibility excluded —
        an ambitious, sound mechanism stays valuable even if hard)."""
        return (0.7 * self.non_obviousness
                + 0.2 * self.soundness_100()
                + 0.1 * self.clarity)

    def worst_margin(self, tau: float) -> float:
        """How far the WEAKEST gated score sits below `tau` (>=0 = pass)."""
        return min(self.non_obviousness, self.soundness_100(),
                   self.clarity) - tau


def qualification(j: Judgement, tau: float = DEFAULT_TAU,
                  repair_margin: float = DEFAULT_REPAIR_MARGIN) -> str:
    """Route a judged candidate: 'accept' | 'repair' | 'reject'.

    IDEAgent Stage II: gate = N>=tau AND S>=tau AND C>=tau; a near-miss —
    every gated score within `repair_margin` below tau — is repaired once
    instead of discarded; anything weaker is rejected outright (repair is
    instructed to preserve the mechanism, so it cannot rescue a dead one).
    """
    margin = j.worst_margin(tau)
    if margin >= 0:
        return "accept"
    if margin >= -repair_margin:
        return "repair"
    return "reject"


# ------------------------------------------------------------------ prompts

JUDGE_PROMPT = (
    "ROLE: judge. Score this research hypothesis as a strict reviewer.\n\n"
    "Hypothesis claim:\n{claim}\n\n"
    "Proposed machine criterion (how it would be tested):\n{criterion}\n\n"
    "Score on these axes:\n"
    "- non_obviousness 0-100: does the central value addition require a "
    "genuine conceptual leap beyond routine extensions, direct combinations "
    "or ablations of known work?\n"
    "- clarity 0-100: are the causal steps from mechanism to expected "
    "effect well described?\n"
    "- feasibility 0-100: does a plausible implementation and evaluation "
    "path exist with python + pip only?\n"
    "- soundness 0-9: logical and mathematical consistency of the argument "
    "and assumptions (0-3 = broken logic).\n"
    'Reply ONLY JSON: {{"non_obviousness": 0, "clarity": 0, "feasibility": 0, '
    '"soundness": 0, "notes": "one-sentence critic feedback"}}'
)

REPAIR_PROMPT = (
    "ROLE: repair. Your hypothesis draft scored below the qualification "
    "gate. Fix it WITHOUT changing the central mechanism.\n\n"
    "Draft claim:\n{claim}\n\n"
    "Draft criterion: {criterion}\n\n"
    "Critic feedback: {feedback}\n\n"
    "Weak scores: {weak}\n\n"
    "Rewrite the claim to fix the weak axes: sharpen the mechanism, state "
    "the assumptions, and make the expected effect MEASURABLE by a python "
    "protocol. Keep it SIMPLE — the simplest protocol that measures the "
    "effect wins.\n"
    'Reply ONLY JSON: {{"claim": "...", "metric": "<name>", "op": ">=", '
    '"target": 0.9, "tolerance": 0.05, "signature": {{"problem": "...", '
    '"mechanism": "...", "novelty": "...", "assumptions": "...", '
    '"effect": "..."}}}}'
)

INSIGHT_PROMPT = (
    "ROLE: ideator. A swarm experiment just SETTLED. Propose up to "
    "{n_children} follow-up hypotheses that the measured result makes "
    "worth testing next — the loop must CONTINUE from results, not restart "
    "from the literature.\n\n"
    "Settled claim: {claim}\n"
    "Verdict: {verdict}\n"
    "Measured: {measured}\n\n"
    "Swarm world model (verified facts are common ground, refuted ones are "
    "dead ends, gaps are open questions):\n{world}\n\n"
    "Rejected patterns (do NOT re-propose these mechanisms):\n{rejected}\n\n"
    "Rules: each child must be TESTABLE with python + pip only, carry a "
    "machine criterion, and differ from the parent and from every rejected "
    "pattern. If the result genuinely opens no testable question, reply "
    "with an empty list.\n"
    'Reply ONLY JSON: {{"hypotheses": [{{"claim": "...", "metric": "<name>", '
    '"op": ">=", "target": 0.9, "tolerance": 0.05, "signature": '
    '{{"problem": "...", "mechanism": "...", "novelty": "...", '
    '"assumptions": "...", "effect": "..."}}}}]}}'
)

GAP_PROMPT = (
    "ROLE: gap-proposer. The literature topic is saturated and the "
    "hypothesis registry is EMPTY — the loop must not idle. Read the "
    "swarm's world model below and propose EXACTLY ONE hypothesis that "
    "closes the most valuable open GAP or QUESTION.\n\n"
    "World model:\n{world}\n\n"
    "Recent claims (avoid re-wordings of these):\n{recent}\n\n"
    "Rejected patterns (do NOT re-propose these mechanisms):\n{rejected}\n\n"
    "Rules: TESTABLE with python + pip only (a runnable protocol must "
    "exist), machine criterion required, mechanism must be new relative to "
    "the rejected patterns.\n"
    'Reply ONLY JSON: {{"hypotheses": [{{"claim": "...", "metric": '
    '"<name>", "op": ">=", "target": 0.9, "tolerance": 0.05, "signature": '
    '{{"problem": "...", "mechanism": "...", "novelty": "...", '
    '"assumptions": "...", "effect": "..."}}}}]}}'
)


# ----------------------------------------------------------------- parsers

def parse_children(answer: str, max_children: int = 2) -> list[dict]:
    """Extract valid follow-up hypothesis specs from an ideation reply.

    Each surviving child: claim (>=12 chars), machine criterion
    {metric, op in >=/<=/>/<, target number}, optional signature quintuple.
    Invalid entries are dropped (never raise — a malformed ideation reply
    must cost the swarm nothing)."""
    try:
        start = answer.find("{")
        end = answer.rfind("}")
        if start < 0 or end <= start:
            return []
        raw = json.loads(answer[start:end + 1])
    except (ValueError, TypeError):
        return []
    hyps = raw.get("hypotheses") if isinstance(raw, dict) else None
    if not isinstance(hyps, list):
        return []
    out: list[dict] = []
    for h in hyps[: max_children * 4]:        # bounded scan, cap applied after
        child = _child_from_dict(h)
        if child is None:
            continue
        out.append(child)
        if len(out) >= max_children:
            break
    return out


def _child_from_dict(h) -> dict | None:
    """Validate ONE hypothesis spec dict; None when it breaks the contract."""
    if not isinstance(h, dict):
        return None
    claim = str(h.get("claim") or "").strip()
    crit = _criterion_of(h)
    if len(claim) < 12 or crit is None:
        return None
    child = {"claim": claim, "criterion": crit}
    sig = h.get("signature")
    if isinstance(sig, dict) and sig:
        child["signature"] = sig
    return child


def normalize_child(child) -> dict | None:
    """Accept either the parsed {claim, criterion} shape or the raw wire
    shape {claim, metric, op, target, [tolerance, signature]} and return
    the parsed shape (None when the contract is broken)."""
    if isinstance(child, dict) and isinstance(child.get("criterion"), dict):
        return child
    return _child_from_dict(child)


def _criterion_of(h: dict) -> dict | None:
    metric = str(h.get("metric") or "").strip()
    op = str(h.get("op") or "").strip()
    target = h.get("target")
    if not metric or op not in _ALLOWED_OPS:
        return None
    if not isinstance(target, (int, float)) or isinstance(target, bool):
        return None
    crit: dict = {"metric": metric, "op": op, "target": float(target)}
    tol = h.get("tolerance")
    if isinstance(tol, (int, float)) and not isinstance(tol, bool) and tol > 0:
        crit["tolerance"] = float(tol)
    return crit


def parse_scores(answer: str) -> Judgement | None:
    """Parse a judge reply into a Judgement; None when unparsable."""
    try:
        start = answer.find("{")
        end = answer.rfind("}")
        if start < 0 or end <= start:
            return None
        raw = json.loads(answer[start:end + 1])
    except (ValueError, TypeError):
        return None
    if not isinstance(raw, dict):
        return None

    def _score(key: str, hi: float) -> float | None:
        v = raw.get(key)
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return None
        return max(0.0, min(hi, float(v)))

    n = _score("non_obviousness", 100)
    c = _score("clarity", 100)
    f = _score("feasibility", 100)
    s = _score("soundness", 9)
    if None in (n, c, s):
        return None
    return Judgement(
        non_obviousness=n, clarity=c if c is not None else 0.0,
        feasibility=f if f is not None else 0.0, soundness=s,
        notes=str(raw.get("notes") or "")[:300],
    )


def parse_repair(answer: str) -> dict | None:
    """Parse a repaired child — the bare child JSON or the wrapped
    {"hypotheses": [...]} form both parse (models drift between them)."""
    kids = parse_children(answer, max_children=1)
    if kids:
        return kids[0]
    try:
        start = answer.find("{")
        end = answer.rfind("}")
        if start < 0 or end <= start:
            return None
        raw = json.loads(answer[start:end + 1])
    except (ValueError, TypeError):
        return None
    return _child_from_dict(raw)


__all__ = [
    "DEFAULT_REPAIR_MARGIN", "DEFAULT_TAU", "GAP_PROMPT", "INSIGHT_PROMPT",
    "JUDGE_PROMPT", "Judgement", "REPAIR_PROMPT", "normalize_child",
    "qualification", "parse_children", "parse_repair", "parse_scores",
]
