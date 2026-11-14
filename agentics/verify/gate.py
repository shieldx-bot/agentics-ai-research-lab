"""Learned risk gate (#5 of the advanced-tech roadmap, wave-3) —
SHADOW-ONLY v1.

The lab's ledger IS a labelled dataset: ``traces`` carries machine-truth
outcomes (approved/merged vs rejected/failed). A tiny logistic gate over
numeric features estimates P(the current cycle settles clean) from the
agent's PRIOR trace history, so the owner can see the risk BEFORE the
panel opens — and so the roadmap's soft-block phase (force one rework)
will have an honest, measured basis later.

Honest boundaries (straight from the roadmap):
* shadow mode never blocks anything — it journals
  ``risk_gate_prediction`` (p = P(settle clean); LOW p = HIGH risk) and
  nothing else. The soft mode is advisory journaling only
  (``risk_gate_softblock`` when p drops under the threshold) until the
  shadow AUC proves itself on real data; it NEVER changes a flow;
* the gate never decides verdicts, never kills claims — P16 + P19 stay
  the only adjudication;
* no leakage: features for a training row are computed strictly from
  traces recorded BEFORE that row's timestamp;
* small and auditable: 6 numeric features, pure-python logistic
  regression (no new dependency), weights + AUC + n journalled on
  (re)train; below MIN_TRAIN_ROWS the gate is silent — no model is
  better than a fake one.
"""
from __future__ import annotations

import logging
import math
import time

log = logging.getLogger("verify.gate")

_LABEL_OK = frozenset({"approved", "merged"})
_LABEL_BAD = frozenset({"rejected", "failed"})
MIN_TRAIN_ROWS = 40
RETRAIN_S = 900.0                      # honest retrain cadence: 15 minutes
FEATURES = ("prior_attempts", "prior_fail_rate", "fail_streak",
            "attempted_chars", "had_failure", "cycle_idx")

# process-wide cache: {model, trained_at, n_train}
_state: dict = {"model": None, "trained_at": 0.0, "n_train": 0}


# ------------------------------------------------------------- features
def features_for(memory, agent: str, attempted: str, cycle: int,
                 before_ts: float | None = None) -> dict:
    """Numeric features for one decision point. ``before_ts`` keeps
    training rows leakage-free (only PRIOR traces count)."""
    outcomes = memory.trace_outcomes(agent, before_ts=before_ts, limit=100)
    prior = [o for _, o in outcomes]
    n = len(prior)
    fails = sum(1 for o in prior if o in _LABEL_BAD)
    streak = 0
    for o in reversed(prior):
        if o in _LABEL_BAD:
            streak += 1
        else:
            break
    return {
        "prior_attempts": float(n),
        "prior_fail_rate": (fails / n) if n else 0.0,
        "fail_streak": float(streak),
        "attempted_chars": float(len(attempted or "")),
        "had_failure": 1.0 if streak > 0 else 0.0,
        "cycle_idx": float(cycle or 0),
    }


# ------------------------------------------------------------ logistic
def _auc(scores: list[float], labels: list[int]) -> float:
    """Rank-based AUC (Mann-Whitney U)."""
    pos = sorted(s for s, l in zip(scores, labels) if l == 1)
    neg = sorted(s for s, l in zip(scores, labels) if l == 0)
    if not pos or not neg:
        return 0.5
    wins = ties = 0
    for p in pos:
        lo, hi = 0, len(neg)
        while lo < hi:                    # first neg >= p
            mid = (lo + hi) // 2
            if neg[mid] < p:
                lo = mid + 1
            else:
                hi = mid
        less = lo
        eq = hi
        while eq < len(neg) and neg[eq] == p:
            eq += 1
        wins += less
        ties += eq - less
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


class RiskGate:
    """A standardized logistic model over the 6 gate features.

    ``predict`` returns P(the cycle settles clean) — LOW p = HIGH risk."""

    def __init__(self, weights: list[float], bias: float,
                 means: list[float], stds: list[float],
                 auc: float, n: int) -> None:
        self.weights, self.bias = weights, bias
        self.means, self.stds = means, stds
        self.auc, self.n = auc, n

    def predict(self, feats: dict) -> float:
        z = self.bias
        for i, key in enumerate(FEATURES):
            std = self.stds[i] or 1.0
            z += self.weights[i] * ((float(feats.get(key, 0.0))
                                     - self.means[i]) / std)
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def _fit(X: list[list[float]], y: list[int]) -> tuple[RiskGate, float]:
    n_f = len(FEATURES)
    means = [sum(col) / len(col) for col in zip(*X)]
    stds = []
    for i, col in enumerate(zip(*X)):
        var = sum((v - means[i]) ** 2 for v in col) / max(1, len(col) - 1)
        stds.append(math.sqrt(var) or 1.0)
    Z = [[(x[i] - means[i]) / stds[i] for i in range(n_f)] for x in X]
    w = [0.0] * n_f
    b = 0.0
    lr, l2 = 0.1, 1e-3
    for _ in range(300):
        gw = [0.0] * n_f
        gb = 0.0
        for row, label in zip(Z, y):
            z = b + sum(wi * xi for wi, xi in zip(w, row))
            p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
            err = p - label
            gb += err
            for i in range(n_f):
                gw[i] += err * row[i]
        m = len(y)
        for i in range(n_f):
            w[i] -= lr * (gw[i] / m + l2 * w[i])
        b -= lr * (gb / m)
    model = RiskGate(w, b, means, stds, 0.0, len(y))
    # In-sample scores on the SAME standardized rows (reported honestly as
    # "auc_insample" in every journal payload — never presented as held-out).
    scores = [1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, b + sum(
        wi * xi for wi, xi in zip(w, row)))))) for row in Z]
    model.auc = _auc(scores, y)
    return model, model.auc


def _fit_with_scores(X: list[list[float]], y: list[int]) -> tuple[RiskGate, float]:
    """Fit on standardized rows, then AUC on the SAME rows (in-sample —
    reported honestly as such)."""
    return _fit(X, y)


# -------------------------------------------------------------- training
def train(memory, min_rows: int = MIN_TRAIN_ROWS) -> RiskGate | None:
    """Train from settled trace rows (features strictly prior). None when
    the ledger is still too small or single-class — honest silence."""
    try:
        rows = memory.terminal_traces(limit=500)
        if len(rows) < min_rows:
            return None
        X, y = [], []
        for r in reversed(rows):          # oldest first
            f = features_for(memory, r["agent"], r["attempted"], r["cycle"],
                             before_ts=r["ts"])
            X.append([f[k] for k in FEATURES])
            y.append(1 if r["outcome"] in _LABEL_OK else 0)
        if len(set(y)) < 2:
            return None
        model, _ = _fit_with_scores(X, y)
        return model
    except Exception as exc:  # noqa: BLE001 — a broken gate must stay silent
        log.debug("risk gate training skipped: %s", exc)
        return None


def _get_model(memory, force: bool = False) -> RiskGate | None:
    now = time.monotonic()
    if force or _state["model"] is None \
            or now - _state["trained_at"] > RETRAIN_S:
        model = train(memory)
        _state["model"] = model
        _state["trained_at"] = now
        _state["n_train"] = model.n if model else 0
        if model is not None:
            log.info("risk gate trained: n=%d in-sample AUC=%.3f",
                     model.n, model.auc)
    return _state["model"]


# ---------------------------------------------------------------- shadow
def shadow_check(ctx, state: dict | None) -> None:
    """Called when a panel opens (review node) — journals the prediction.

    NEVER raises, NEVER blocks: shadow mode only writes
    ``risk_gate_prediction``; the advisory ``risk_gate_softblock`` row is
    the only trace of the soft mode, and no code path changes because of
    it. Too-few-rows ⇒ silent (the gate does not pretend to know).
    """
    settings = getattr(ctx, "settings", None)
    mode = str(getattr(settings, "risk_gate_mode", "shadow") or
               "shadow").strip().lower()
    if mode == "off":
        return
    memory = getattr(ctx, "memory", None)
    if memory is None:
        return
    try:
        st = state or {}
        attempted = str(st.get("plan") or st.get("attempted")
                        or st.get("approach") or "")
        feats = features_for(memory, getattr(ctx, "agent_id", "?"),
                             attempted, int(getattr(ctx, "cycle", 0) or 0))
        model = _get_model(memory)
        if model is None:
            return
        p = model.predict(feats)
        payload = {"p": round(p, 4),
                   **{k: round(v, 3) for k, v in feats.items()},
                   "auc_insample": round(model.auc, 3), "n_train": model.n}
        memory.journal(getattr(ctx, "agent_id", "?"),
                       "risk_gate_prediction", payload)
        if mode == "soft" and p < float(
                getattr(settings, "risk_gate_soft_threshold", 0.25) or 0.25):
            memory.journal(getattr(ctx, "agent_id", "?"),
                           "risk_gate_softblock",
                           {"p": round(p, 4), "note": "advisory only"})
    except Exception as exc:  # noqa: BLE001 — the gate must never break a cycle
        log.debug("risk gate shadow skipped: %s", exc)
