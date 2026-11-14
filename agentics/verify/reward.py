"""C1/C4 (TTT_DISCOVER.md) — continuous reward R(s) + r_sota tracking.

The paper defines a discovery as an event where R(s) > r_sota. The lab's
reward flows ONLY from anchored, machine-verdicted settles — the settle
anchor law IS the paper's validity check:

    R(s) = 0                              status != supported
    R(s) = 0                              unanchored (self_authored_ratio = 1)
    R(s) = clip(margin / scale, 0, 1)     anchored-supported

with margin normalized by |target| (reward_scale="target") or by the
metric's observed span ("history"). Pure math, no torch, no provider.
"""
from __future__ import annotations

import json


def reward_from_row(row: dict, scale: str = "target",
                    history_span: float | None = None) -> float:
    """Continuous reward for ONE experiment_ledger row.

    `row` needs: status, measured, op, target, self_authored_ratio.
    `history_span` is the caller-provided |max-min| of the metric's past
    values (used only when scale == "history"; falls back to |target| when
    the span is unusable). Returns a float in [0, 1]; never raises."""
    try:
        status = str(row.get("status") or "")
        if status != "supported":
            return 0.0
        ratio = row.get("self_authored_ratio")
        if ratio is None or float(ratio) >= 1.0:
            return 0.0                     # no provenance / validity fail
        measured = row.get("measured")
        target = row.get("target")
        op = str(row.get("op") or "").strip()
        if measured is None or op in ("", "==", "="):
            return 1.0 if op in ("==", "=") and measured is not None else 0.0
        measured, target = float(measured), float(target) if target is not None else None
        if op == ">":
            margin = measured - target
        elif op == "<":
            margin = target - measured
        else:
            return 0.0
        denom: float | None = abs(target) if target not in (None, 0.0) else None
        if scale == "history" and history_span and history_span > 0:
            denom = float(history_span)
        if not denom or denom <= 0:
            denom = max(abs(measured), 1e-9)
        return max(0.0, min(1.0, margin / denom))
    except (TypeError, ValueError):
        return 0.0


def adaptive_softmax_weights(rewards: list[float]) -> list[float]:
    """Entropic credit assignment (train.py::compute_advantages, entropic
    adaptive-beta branch): weights w_i = softmax(beta * r_i) with beta
    binary-searched so KL(q || uniform) = log 2 — the distribution stays
    exactly one nat away from uniform regardless of reward scale.

    remove_constant_reward_groups: when every reward is identical the
    group carries NO information and returns uniform weights (which the
    caller must treat as boost = 0). Never raises."""
    n = len(rewards)
    if n == 0:
        return []
    if n == 1:
        return [1.0]
    rmin, rmax = min(rewards), max(rewards)
    if rmax - rmin <= 1e-12:
        return [1.0 / n] * n             # constant group -> uniform
    # KL(softmax(br) || uniform) = log(sum_i e^{b r_i}) - b*mean(r)  (nats)
    # monotone non-decreasing in b, >= 0 at b=0 -> binary search the root
    lo, hi = 0.0, 1.0
    target = _log(2.0)

    def kl(b: float) -> float:
        m = sum(_exp(b * r) for r in rewards) / n
        return _log(m) - b * (sum(rewards) / n)

    while kl(hi) < target:
        hi *= 2.0
        if hi > 1e6:
            break
    for _ in range(40):
        mid = (lo + hi) / 2.0
        if kl(mid) < target:
            lo = mid
        else:
            hi = mid
    beta = (lo + hi) / 2.0
    exps = [_exp(beta * (r - rmax)) for r in rewards]   # shift for stability
    total = sum(exps)
    if total <= 0:
        return [1.0 / n] * n
    return [e / total for e in exps]


def _log(x: float) -> float:
    import math
    return math.log(max(x, 1e-300))


def _exp(x: float) -> float:
    import math
    try:
        return math.exp(max(min(x, 700.0), -700.0))
    except OverflowError:
        return math.inf


def reward_stats(rows: list[dict], scale: str = "target") -> dict:
    """Aggregates for the PUCT frontier / dashboard: (best, mean, span) over
    the rows' metric values — the history-scale denominator lives here."""
    vals = [float(r.get("measured")) for r in rows
            if r.get("measured") is not None]
    span = (max(vals) - min(vals)) if len(vals) >= 2 else None
    rewards = [reward_from_row(r, scale=scale, history_span=span) for r in rows]
    live = [r for r in rewards if r > 0.0]
    return {
        "rewards": rewards,
        "best": max(rewards) if rewards else 0.0,
        "mean": (sum(live) / len(live)) if live else 0.0,
        "span": span,
    }


def anchors_of(row: dict) -> list[str]:
    try:
        val = json.loads(row.get("anchors") or "[]")
        return val if isinstance(val, list) else []
    except ValueError:
        return []
