"""Research-diversity instrumentation — DAPS / IDEAgent mechanics, adapted.

Three papers drive this module:

- "Beneath the Diff" (DAPS, arXiv 2609.00077): code-level autonomous research
  loops suffer ALGORITHMIC MODE COLLAPSE — surface variation (different files
  edited, different wording) stays high while the MECHANISM of each proposal
  concentrates onto a few recurring patterns, and in-loop gains stop
  transferring to held-out checks. Mitigations used here:
    * mechanism taxonomy + Shannon-entropy monitoring (the collapse alarm),
    * Persistent Edit Memory  -> near_duplicate() gate (PEM),
    * Category-Coverage Reweighting -> coverage_order() (CCR).
- IDEAgent (arXiv 2607.22375): ideation as Quality-Diversity search — every
  idea gets a compact structured SIGNATURE (quintuple), compared against
  active/historical/rejected archives so near-clones are rejected instead of
  counted as separate discoveries.
- Kosmos (arXiv 2511.02824): valuable findings scale with cycles ONLY while
  the loop stays diverse; a structured world model keeps 200+ rollouts
  coherent. The world_notes table in shared memory is that world model.

Everything here is dependency-free and mechanical (no LLM calls): the monitor
and gates must be cheap enough to run inside every supervisor/researcher
cycle. The frozen taxonomy is domain-oriented (ML/NFV research mechanisms,
mirroring the lab's live claim stream) with an `other` bucket absorbing the
long tail, exactly like DAPS's 9-category instrument.
"""
from __future__ import annotations

import math
import re

# Frozen before any live use (DAPS §3.2 derivation discipline): keyword
# inventory over the lab's research-mechanism vocabulary. The classifier is
# rule-based so the audit is deterministic and auditable in the journal.
_MECHANISM_KEYWORDS: dict[str, tuple[str, ...]] = {
    "forecasting": ("forecast", "predict", "prediction", "time series",
                    "time-series", "lstm", "gnn", "graph neural", "regress",
                    "arima", "prophet", "early warning", "early failure",
                    "prognostic"),
    "autoscaling": ("autoscal", "scaling polic", "scale action", "replica",
                    "resource alloc", "elastic", "horizontal scale",
                    "provisioning", "capacity"),
    "anomaly": ("anomaly", "fault detection", "failure detection", "outlier",
                "root cause", "drift detection", "degradation"),
    "scheduling": ("schedul", "queueing", "queue", "placement", "bin pack",
                   "priority", "load balanc"),
    "benchmark": ("benchmark", "dataset", "evaluation protocol", "testbed",
                  "metrics design", "reproducibility"),
    "simulation": ("simulation", "discrete-event", "simulator",
                   "synthetic data", "digital twin"),
    "optimization": ("optimiz", "hyperparameter", "reinforcement",
                     "control policy", "bandit", "bayesian", "search strateg"),
    "theory": ("survey", "review", "taxonomy", "state of the art",
               "formaliz", "bound", "complexity"),
    "security": ("security", "adversar", "privacy", "attack", "intrusion"),
    "efficiency": ("energy", "latency", "throughput", "overhead", "cost",
                   "migration"),
}

_WORD_RE = re.compile(r"[a-z][a-z\-]{2,}")

# Stopwords for the Jaccard near-duplicate gate (PEM): metric scaffolding
# words that appear in EVERY hypothesis and must not mask real similarity.
_STOPWORDS = frozenset({
    "the", "a", "an", "of", "in", "on", "for", "and", "or", "to", "with",
    "at", "by", "from", "than", "at least", "is", "are", "will", "can",
    "that", "this", "when", "while", "per", "each", "versus", "against",
    "within", "over", "under", "below", "above", "into", "its", "their", "our", "not", "no",
    "less", "more", "same", "such", "between", "after", "before", "during",
    "based", "using", "use", "used", "new", "synthetic", "measured",
    "compared", "relative", "increase", "decrease", "reduce", "reduces",
    "improve", "improves", "achieve", "achieves", "mean", "test",
})

# Light suffix stripping (no stemmer dependency): merges estimate/estimates,
# regression/regress, prediction/predict, scaling/scaled — the morphological
# variants LLM re-wordings lean on to dodge exact-match dedup.
_SUFFIXES = ("ations", "ation", "ions", "ion", "ments", "ment", "ings",
             "ing", "ies", "ers", "ed", "es", "s", "e")


def _stem(word: str) -> str:
    for suf in _SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 5:
            return word[: len(word) - len(suf)]
    return word


def classify(text: str) -> str:
    """Map a claim / finding title to ONE mechanism category (or 'other')."""
    t = (text or "").lower()
    best, best_hits = "other", 0
    for cat, keys in _MECHANISM_KEYWORDS.items():
        hits = sum(1 for k in keys if k in t)
        if hits > best_hits:
            best, best_hits = cat, hits
    return best


def entropy(labels: list[str]) -> float:
    """Shannon entropy (nats) of the category distribution.

    Healthy research streams measure ~1.5-2.2 nats over a 10-category
    taxonomy; DAPS observed collapse as a ~0.6 nat drop while SURFACE
    diversity stayed flat — which is why we monitor MECHANISM entropy, not
    wording variety."""
    n = len(labels)
    if n == 0:
        return 0.0
    counts: dict[str, int] = {}
    for lb in labels:
        counts[lb] = counts.get(lb, 0) + 1
    h = 0.0
    for c in counts.values():
        p = c / n
        h -= p * math.log(p)
    return h


def _words(text: str) -> set[str]:
    return {_stem(w) for w in _WORD_RE.findall((text or "").lower())
            if w not in _STOPWORDS}


def jaccard(a: str, b: str) -> float:
    """Word-Jaccard similarity over non-stopword tokens."""
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return 0.0
    inter = len(wa & wb)
    union = len(wa | wb)
    return inter / union if union else 0.0


def signature_text(sig: dict) -> str:
    """IDEAgent quintuple -> one string for comparisons. Only the mechanism
    and problem axes matter for diversity (IDEAgent §5 Diversity Axis:
    failure mode / causal diagnosis / intervention carry the weight)."""
    if not isinstance(sig, dict):
        return ""
    return " ".join(str(sig.get(k, "") or "") for k in
                    ("problem", "mechanism"))


def near_duplicate(claim: str, priors: list, threshold: float = 0.8,
                   exclude_ids: set | None = None) -> tuple[float, object] | None:
    """PEM gate: find the closest prior hypothesis to `claim`.

    `priors` is a list of dicts with at least {id, claim}; a signature dict
    (IDEAgent quintuple) is compared through its problem+mechanism axes when
    present. Returns (score, prior_row) when similarity >= threshold and the
    prior's id is not in `exclude_ids`, else None."""
    exclude = exclude_ids or set()
    best, best_row, best_score = None, None, 0.0
    for p in priors:
        if not isinstance(p, dict):
            continue
        pid = p.get("id")
        if pid is not None and pid in exclude:
            continue
        # IDEAgent: the signature quintuple is the idea's identity — a
        # candidate matching the problem+mechanism axes IS a near-clone even
        # when the wording differs, so the score is the best of both texts.
        texts = [p.get("claim") or ""]
        sig = p.get("signature")
        if isinstance(sig, dict) and sig.get("mechanism"):
            texts.append(signature_text(sig))
        score = max(jaccard(claim, t) for t in texts if t)
        if score > best_score:
            best, best_row, best_score = score, p, score
    if best is not None and best >= threshold:
        return best_score, best_row
    return None


def coverage_order(candidates: list[dict], recent_claims: list[str],
                   text_key: str = "claim") -> list[dict]:
    """CCR: order candidates RAREST-category-first against a recent window.

    DAPS insight: the proposer's prior is sharply peaked toward a few
    reliable edit types and acceptance pressure amplifies the skew. Re-weight
    by under-representation: candidates whose mechanism category is rare in
    the recent stream go first; ties keep registry age order (oldest id)."""
    recent_cats: dict[str, int] = {}
    for c in recent_claims:
        cat = classify(c)
        recent_cats[cat] = recent_cats.get(cat, 0) + 1
    def _key(item: dict):
        cat = classify(str(item.get(text_key) or ""))
        return (recent_cats.get(cat, 0), item.get("id") or 0)
    return sorted(candidates, key=_key)


def audit(candidates_claims: list[str]) -> dict:
    """Diversity audit payload for the journal: mechanism entropy + the
    categories actually seen. Cheap enough for every supervisor cycle."""
    cats = [classify(c) for c in candidates_claims]
    return {
        "entropy_nats": round(entropy(cats), 3),
        "window": len(cats),
        "categories": sorted(set(cats)),
    }
