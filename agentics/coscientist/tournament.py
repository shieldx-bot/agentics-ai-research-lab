"""The Elo tournament — simulated scientific debates between hypotheses.

Ported from the Apache-2.0 Kaimen-Inc/Co-Scientist re-implementation
(co_scientist/agents/ranking.py), adapted to the lab's LLM edge:

* one match = ONE Perplexity ask (the reference's API edge is cheap and
  fast; ours is a free browser session, so a "tournament batch" is
  bounded by config and every ask passes the P20 budget gate);
* the verdict parse, Elo math, pair selection and the crash-idempotency
  contract (deterministic match_id -> INSERT OR IGNORE -> only the row
  owner applies the delta) behave exactly like the reference;
* refuted hypotheses never enter the pool: a claim the lab FALSIFIED by
  running code is a dead idea, no matter how elegant the debate finds it.
"""
from __future__ import annotations

import hashlib
import logging
import time

from agentics.coscientist import pairing
from agentics.coscientist.elo import update_elo

log = logging.getLogger("coscientist.tournament")

_JURY_SYSTEM = (
    "You are the ranking jury of a self-improving AI research lab. You "
    "compare research hypotheses through a simulated panel debate and "
    "always end with the exact verdict line you are asked for. Be honest "
    "and decisive; never invent facts."
)


def match_id_for(hyp_a: dict, hyp_b: dict) -> str:
    """Deterministic match id — hash of the PRE-match rating state.

    A crash between the debate ask and the persistence replays with the
    same pre-state (the state only changes when the delta is applied),
    so the retry INSERT OR IGNOREs the same row and never double-applies
    the Elo delta. Same contract as the reference's task-id-derived
    round_id, without needing a task queue."""
    lo, hi = sorted((int(hyp_a["id"]), int(hyp_b["id"])))
    pre = "|".join((
        str(lo), str(hi),
        f"{float(hyp_a.get('elo') or pairing.ELO_INITIAL):.6f}",
        f"{float(hyp_b.get('elo') or pairing.ELO_INITIAL):.6f}",
        str(int(hyp_a.get("matches_played") or 0)),
        str(int(hyp_b.get("matches_played") or 0)),
    ))
    return "t" + hashlib.sha1(pre.encode()).hexdigest()[:16]


def feedback_block(memory) -> str:
    """The meta-review steering note, or '' before the first synthesis."""
    try:
        fb = memory.latest_tournament_feedback()
    except Exception:          # noqa: BLE001 — feedback must never gate matches
        return ""
    if not fb or not (fb.get("text") or "").strip():
        return ""
    return ("Steering from the meta-review of previous debates "
            "(patterns to build on or avoid):\n"
            f"{fb['text'].strip()[:800]}\n\n")


async def _jury_ask(ctx, prompt: str) -> tuple[str, bool, bool]:
    """One budget-gated jury ask. Returns (answer, ok, blocked_or_failed).

    ``blocked_or_failed`` marks asks that never produced a jury opinion —
    a P20 budget block or a provider failure. Those must not be recorded
    as matches (nothing happened) and must END a batch: retrying the same
    ask within the same seconds only hammers a dead/budgeted session."""
    if ctx.llm is None:
        return "", False, True
    budget = getattr(ctx, "budget", None)
    started = time.monotonic()
    if budget is not None:
        allowed, why = budget.can_ask(ctx.agent_id)
        if not allowed:
            log.info("[%s] tournament ask blocked by budget: %s",
                     ctx.agent_id, why)
            try:
                ctx.memory.journal(ctx.agent_id, "budget_blocked",
                                   {"purpose": "tournament", "why": why[:200]})
            except Exception:  # noqa: BLE001
                pass
            return "", False, True
    try:
        resp = await ctx.llm.ask(prompt, mode="concise", purpose="tournament",
                                 system=_JURY_SYSTEM)
        ok = bool(resp and resp.ok)
        answer = (resp.answer or "") if resp else ""
    except Exception as exc:  # noqa: BLE001 — one match must not kill a cycle
        log.warning("[%s] tournament ask failed: %s", ctx.agent_id, exc)
        answer, ok, blocked_or_failed = "", False, True
    else:
        blocked_or_failed = not ok
    if budget is not None:
        budget.record(ctx.agent_id, "tournament", ok, time.monotonic() - started)
    return answer, ok, blocked_or_failed


async def run_match(ctx, hyp_a: dict, hyp_b: dict, goal: str) -> dict:
    """One tournament match: select mode -> jury ask -> parse -> Elo.

    Failure-isolated by contract: any unparseable/blocked debate is
    recorded as an ``invalid`` match (no Elo movement) and reported back,
    never raised."""
    # anchor on the lower id so prompt ordering is stable across retries
    a, b = (hyp_a, hyp_b) if hyp_a["id"] <= hyp_b["id"] else (hyp_b, hyp_a)
    elo_a = float(a.get("elo") or pairing.ELO_INITIAL)
    elo_b = float(b.get("elo") or pairing.ELO_INITIAL)
    mode = pairing.select_mode(int(a.get("matches_played") or 0),
                               int(b.get("matches_played") or 0),
                               elo_a, elo_b)
    template = pairing.DEBATE_PROMPT if mode == "debate" else pairing.PAIRWISE_PROMPT
    prompt = template.format(
        goal=goal[:600],
        id1=f"h{a['id']}", text1=(a.get("claim") or "")[:1200],
        id2=f"h{b['id']}", text2=(b.get("claim") or "")[:1200],
        feedback_block=feedback_block(ctx.memory),
    )
    answer, ok, stopped = await _jury_ask(ctx, prompt)
    mid = match_id_for(a, b)
    if stopped:
        # nothing happened: no match row, and the batch should not hammer on
        return {"match_id": mid, "mode": "blocked", "winner": "",
                "a": a["id"], "b": b["id"], "stop": True}
    choice = pairing.parse_better_idea(answer)
    if choice is None:
        ctx.memory.record_tournament_match(
            mid, ctx.agent_id, a["id"], b["id"], "invalid", "",
            elo_a, elo_b, None, None, rationale=answer[:600])
        try:
            ctx.memory.journal(ctx.agent_id, "tournament_invalid_verdict",
                               {"match": mid, "a": a["id"], "b": b["id"]})
        except Exception:  # noqa: BLE001
            pass
        return {"match_id": mid, "mode": "invalid", "winner": "",
                "a": a["id"], "b": b["id"]}

    # "1" == the anchor (a), "2" == the opponent (b)
    winner = "a" if choice == 1 else "b"
    upd = update_elo(elo_a, elo_b, winner,
                     matches_played_min=min(int(a.get("matches_played") or 0),
                                            int(b.get("matches_played") or 0)))
    elo_a_after = upd.elo_a_after if winner == "a" else upd.elo_b_after
    elo_b_after = upd.elo_b_after if winner == "a" else upd.elo_a_after
    applied = ctx.memory.record_tournament_match(
        mid, ctx.agent_id, a["id"], b["id"], mode, winner,
        elo_a, elo_b, elo_a_after, elo_b_after, rationale=answer[:1200])
    if applied:
        ctx.memory.apply_elo_update(a["id"], b["id"], elo_a_after, elo_b_after)
        try:
            ctx.memory.journal(ctx.agent_id, "tournament_match", {
                "match": mid, "mode": mode, "winner": winner,
                "a": a["id"], "b": b["id"],
                "elo": [round(elo_a, 1), round(elo_b, 1)],
                "elo_after": [round(elo_a_after, 1), round(elo_b_after, 1)],
            })
        except Exception:  # noqa: BLE001
            pass
    else:
        log.info("match %s already recorded — Elo delta not re-applied", mid)
    return {"match_id": mid, "mode": mode, "winner": winner,
            "a": a["id"], "b": b["id"],
            "elo_after": [round(elo_a_after, 1), round(elo_b_after, 1)],
            "applied": applied}


async def run_batch(ctx, n_matches: int, goal: str) -> list[dict]:
    """A bounded sequence of matches. A budget-blocked or failed ask ends
    the batch (the swarm retries next cycle — same-session retries only
    hammer a dead edge); an unparseable VERDICT is a real outcome and the
    batch continues. Every result is journalled."""
    results: list[dict] = []
    for _ in range(max(0, int(n_matches))):
        pool = ctx.memory.tournament_pool()
        pair = pairing.select_pair(pool)
        if pair is None:
            break
        res = await run_match(ctx, pair[0], pair[1], goal)
        if res.get("stop"):
            break
        results.append(res)
    if results:
        try:
            ctx.memory.journal(ctx.agent_id, "tournament_batch", {
                "matches": len(results),
                "winners": [r.get("winner") for r in results],
                "invalid": sum(1 for r in results if r["mode"] == "invalid"),
            })
        except Exception:  # noqa: BLE001
            pass
    return results
