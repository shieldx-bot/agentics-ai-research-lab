"""EIG frontier (wave-2 #3): rank candidate experiments by EXPECTED
INFORMATION GAIN per unit cost — Bayesian experimental design with the
lab's own settle ledger as the prior source, no gradients anywhere.

Why (2026-11-14): PUCT chooses WHICH DISCOVERY BRANCH to dig into; EIG
answers the orthogonal question — of the claims inside a branch, which is
WORTH testing first when the pool is at 90/90 and every cookie call must
buy information. A claim whose metric historically settles ~50/50 carries
up to ln2 nats of entropy to resolve; a metric the ledger has answered the
same way twenty times carries none — running it again is a "sure-thing"
buy of nothing.

Model (Beta-Bernoulli, closed form):
* prior per metric: alpha = 1 + #supported settles of that metric,
  beta = 1 + #refuted settles (experiment_ledger, autoresearch);
* EIG = mutual information between the metric's true rate and ONE more
  Bernoulli observation — the Beta-entropy closed form with digamma:
  I = H(Beta(a,b)) - q*H(Beta(a+1,b)) - (1-q)*H(Beta(a,b+1)) [nats];
  maximum at maximum prior uncertainty (Laplace), O(1/n) once the ledger
  has pinned the metric down — sure-things AND tightly-known metrics buy
  almost nothing, which is exactly the ranking a cookie-starved lab needs;
* cost = 1 + 0.5*attempts (a retested claim is not free);
* score = (EIG / ln2) / cost, clipped to [0, 1] — consumed by
  discovery_frontier_puct as an additive term (eig_weight).

Contract: never raises; unknown metrics get the Laplace prior (1,1) —
maximum uncertainty, maximum information, which is exactly what an
unexplored direction deserves.
"""
from __future__ import annotations

import json
import logging
import math

log = logging.getLogger("evolution.eig")

_LN2 = math.log(2.0)


def _digamma(x: float) -> float:
    """psi(x) — recurrence to >= 6, then the standard asymptotic series."""
    result = 0.0
    while x < 6.0:
        result -= 1.0 / x
        x += 1.0
    inv = 1.0 / x
    inv2 = inv * inv
    return result + (math.log(x) - 0.5 * inv
                     - inv2 * (1.0 / 12.0
                               - inv2 * (1.0 / 120.0
                                         - inv2 * (1.0 / 252.0))))


def _log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def beta_entropy(a: float, b: float) -> float:
    """Differential entropy of Beta(a, b) (closed form, digamma). May be
    negative — only DIFFERENCES are meaningful (the EIG below)."""
    return (_log_beta(a, b)
            - (a - 1.0) * _digamma(a)
            - (b - 1.0) * _digamma(b)
            + (a + b - 2.0) * _digamma(a + b))


def binary_entropy(p: float) -> float:
    """H(p) in nats; 0 outside (0,1) where entropy is degenerate."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))


def metric_priors(memory, limit: int = 400) -> dict[str, tuple[int, int]]:
    """{metric: (supported_count, refuted_count)} from the experiment
    ledger, newest `limit` rows."""
    counts: dict[str, tuple[int, int]] = {}
    try:
        with memory._lock:  # noqa: SLF001 — same-package read helper
            rows = memory._conn.execute(
                "SELECT metric, status FROM experiment_ledger "
                "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        for metric, status in rows:
            m = str(metric or "").strip()
            if not m:
                continue
            sup, ref = counts.get(m, (0, 0))
            if status == "supported":
                sup += 1
            elif status == "refuted":
                ref += 1
            counts[m] = (sup, ref)
    except Exception as exc:  # noqa: BLE001 — priors are sugar
        log.debug("ledger priors unavailable: %s", exc)
    return counts


def bernoulli_eig(alpha: float, beta: float) -> float:
    """Mutual information between the metric's true rate and ONE more
    Bernoulli observation, under a Beta(alpha, beta) prior (closed form):

        I = H(Beta(a,b)) - q*H(Beta(a+1,b)) - (1-q)*H(Beta(a,b+1))

    Maximum (~0.19 nats) at maximum prior uncertainty (Laplace); O(1/n)
    once the ledger has pinned the metric down — a sure-thing metric and a
    tightly-estimated metric both buy almost nothing. Clamped to [0, ln2]
    against floating-point noise."""
    a, b = max(1.0, float(alpha)), max(1.0, float(beta))
    n = a + b
    q = a / n
    eig = (beta_entropy(a, b)
           - q * beta_entropy(a + 1.0, b)
           - (1.0 - q) * beta_entropy(a, b + 1.0))
    return max(0.0, min(_LN2, eig))


def eig_score(memory, hyp: dict,
              priors: dict[str, tuple[int, int]] | None = None) -> dict:
    """Normalized information-gain-per-cost for one proposed hypothesis.

    Returns {"eig_nats", "eig_score", "prior_alpha", "prior_beta"} —
    score in [0,1] = (EIG/ln2) / cost. Unknown metric -> Laplace (1,1)."""
    try:
        crit = json.loads(hyp.get("criterion") or "{}")
        metric = str(crit.get("metric") or "").strip()
    except Exception:  # noqa: BLE001
        metric = ""
    if priors is None:
        priors = metric_priors(memory)
    sup, ref = priors.get(metric, (0, 0))
    alpha, beta = 1.0 + sup, 1.0 + ref
    eig = max(0.0, bernoulli_eig(alpha, beta))
    cost = 1.0 + 0.5 * float(hyp.get("attempts") or 0)
    score = min(1.0, (eig / _LN2) / max(0.5, cost))
    return {"eig_nats": round(eig, 4), "eig_score": round(score, 4),
            "prior_alpha": alpha, "prior_beta": beta}


__all__ = ["binary_entropy", "metric_priors", "bernoulli_eig", "eig_score"]
