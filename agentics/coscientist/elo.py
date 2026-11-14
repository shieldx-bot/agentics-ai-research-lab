"""Pure Elo math — no DB, no I/O.

Ported from the Apache-2.0 Kaimen-Inc/Co-Scientist re-implementation
(co_scientist/orchestrator/elo.py), kept API-compatible so the lab's
tournament behaves like the reference system:

* K factor is scheduled by experience: new entrants (few matches) move
  fast (K=32), seasoned hypotheses move slowly (K=16) — quick, honest
  convergence for fresh ideas without letting one early upset swing a
  veteran permanently;
* updates are ZERO-SUM: one hypothesis's gain is exactly the other's
  loss, so the pool's total rating mass is conserved.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EloUpdate:
    elo_a_after: float
    elo_b_after: float
    expected_a: float
    k: int


def k_factor(matches_played: int, *, new_threshold: int = 3,
             k_new: int = 32, k_warm: int = 16) -> int:
    """Higher K for new entrants -> faster convergence; lower K once seasoned.

    (The reference default is 5; the lab's tournaments are small and
    budgeted, so the switch happens one match earlier — documented here
    rather than silently diverging.)"""
    return k_new if matches_played < new_threshold else k_warm


def expected_score(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def update_elo(
    elo_a: float,
    elo_b: float,
    winner: str,           # "a" | "b"
    matches_played_min: int,
    *,
    k_new: int = 32,
    k_warm: int = 16,
) -> EloUpdate:
    """Standard Elo update. K is decided by the *less experienced* player."""
    if winner not in ("a", "b"):
        raise ValueError(f"winner must be 'a' or 'b', got {winner!r}")
    k = k_factor(matches_played_min, k_new=k_new, k_warm=k_warm)
    e_a = expected_score(elo_a, elo_b)
    s_a = 1.0 if winner == "a" else 0.0
    delta = k * (s_a - e_a)
    return EloUpdate(
        elo_a_after=elo_a + delta,
        elo_b_after=elo_b - delta,        # zero-sum
        expected_a=e_a,
        k=k,
    )
