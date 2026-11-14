"""Verify calibration layer (wave-2 #4): every settle verdict carries a
CONFIDENCE BOUND measured from the lab's own replication/replay history —
not a model opinion.

Why (2026-11-14): a `supported` verdict propagates everywhere at once —
PUCT Q, Elo prior, GraphRAG, verified_knowledge prompts — so a wrongly
confident lab compounds one bad measurement through every consumer. The
P23 replay sweep and the P24bis replication pipeline already produce the
calibration set for free: each replay/replication is an independent
re-measurement of a settled claim, and its agreement with the original
verdict IS the lab's empirical reliability.

Statistics (no gradient needed):
* agreement rate r = agreeing re-measurements / total decisive ones
  (inconclusive re-runs are excluded — they carry no signal);
* Wilson lower bound at 95% (z=1.96) — a small sample can never claim high
  confidence, which is exactly the behavior a settle note should have.

Contract:
* never raises — calibration is sugar on the verdict, not a gate;
* n < min_n -> no suffix / no alarm (an empty history must not fake
  confidence);
* alarms fire from the supervisor sweep only, so one bad hour costs one
  alert, not a flood.
"""
from __future__ import annotations

import logging
import math

log = logging.getLogger("verify.calibration")

_Z95 = 1.96
_MAP = {("confirmed", "supported"), ("refuted", "refuted")}


def agreement_stats(memory, limit: int = 60) -> dict:
    """Agreement between settled verdicts and their independent
    re-measurements (P23 replays + P24bis replications), newest-first.

    Returns {n, agreements, disagreements, rate, wilson_lo}: `n` counts
    only DECISIVE pairs (re-run reached confirmed/refuted while the claim
    is supported/refuted)."""
    agreements = disagreements = 0
    try:
        rows = []
        with memory._lock:  # noqa: SLF001 — same-package read helper
            rows += memory._conn.execute(
                "SELECT r.verdict, h.status FROM replications r "
                "JOIN hypotheses h ON h.id = r.hypothesis_id "
                "ORDER BY r.id DESC LIMIT ?", (limit,)).fetchall()
            rows += memory._conn.execute(
                "SELECT e.verdict, h.status FROM evidence e "
                "JOIN hypotheses h ON h.id = e.hypothesis_id "
                "WHERE e.method='replay' "
                "ORDER BY e.id DESC LIMIT ?", (limit,)).fetchall()
        for verdict, status in rows:
            if (str(verdict), str(status)) in _MAP:
                agreements += 1
            elif str(verdict) in ("confirmed", "refuted"):
                disagreements += 1
            # inconclusive / non-decisive: no signal, excluded
    except Exception as exc:  # noqa: BLE001 — calibration must never break
        log.debug("calibration read failed: %s", exc)
        return {"n": 0, "agreements": 0, "disagreements": 0,
                "rate": 0.0, "wilson_lo": 0.0}
    n = agreements + disagreements
    rate = agreements / n if n else 0.0
    return {"n": n, "agreements": agreements,
            "disagreements": disagreements, "rate": rate,
            "wilson_lo": wilson_lower(agreements, n)}


def wilson_lower(successes: int, n: int, z: float = _Z95) -> float:
    """Wilson score interval LOWER bound — the honest confidence floor."""
    if n <= 0:
        return 0.0
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = p + z2 / (2.0 * n)
    spread = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return max(0.0, min(1.0, (centre - spread) / denom))


def calibration_suffix(memory, min_n: int = 5) -> str:
    """The settle-note tag: `[cal 0.90 over 20 re-runs, Wilson-95 >= 0.72]`.

    Empty string when the lab has fewer than `min_n` decisive re-runs —
    an immature history must NOT decorate verdicts with fake confidence."""
    try:
        s = agreement_stats(memory)
    except Exception:  # noqa: BLE001
        return ""
    if s["n"] < int(min_n):
        return ""
    return (f" [cal {s['rate']:.2f} over {s['n']} re-runs, "
            f"Wilson-95 >= {s['wilson_lo']:.2f}]")


def calibration_audit(ctx, min_n: int = 10,
                      warn_lo: float = 0.7) -> dict:
    """Supervisor-sweep hook: journal the stats every sweep, alert the
    owner when the honest floor drops below `warn_lo`."""
    memory = ctx.memory
    s = agreement_stats(memory)
    try:
        memory.journal(ctx.agent_id, "verify_calibration", {
            "n": s["n"], "rate": round(s["rate"], 3),
            "wilson_lo": round(s["wilson_lo"], 3)})
    except Exception:  # noqa: BLE001
        pass
    if s["n"] >= int(min_n) and s["wilson_lo"] < float(warn_lo):
        try:
            memory.post_message(
                ctx.agent_id, "alert",
                f"VERIFY CALIBRATION ALERT: re-measurement agreement "
                f"{s['rate']:.2f} (Wilson-95 >= {s['wilson_lo']:.2f}, "
                f"n={s['n']}) — settled verdicts are NOT reliably "
                f"reproducible; PUCT Q / Elo / GraphRAG consume them. "
                f"Check replay drift and protocol determinism.")
            memory.journal(ctx.agent_id, "verify_calibration_alert", {
                "n": s["n"], "wilson_lo": round(s["wilson_lo"], 3)})
        except Exception:  # noqa: BLE001
            pass
    return s


__all__ = ["agreement_stats", "wilson_lower", "calibration_suffix",
           "calibration_audit"]
