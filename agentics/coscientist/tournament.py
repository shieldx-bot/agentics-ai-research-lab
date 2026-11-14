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
  running code is a dead idea, no matter how elegant the debate finds it;
* DUAL-JUDGE (anti "one model judges everything"): when the context
  carries a ``judge_llm`` (an independent model family, e.g. ChatGPT —
  see agentics/llm/chatgpt_client.py), every match is judged by BOTH
  juries CONCURRENTLY. Agreement -> Elo moves. Disagreement -> the match
  is recorded as ``contested`` with NO Elo movement and a journal event
  for the owner — the honest handling of a correlated-error-prone signal.
  A judge that fails or answers unparseably degrades to the primary
  verdict with an explicit note in the rationale, never silently.
"""
from __future__ import annotations

import asyncio
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


async def _jury_ask(ctx, prompt: str, llm=None,
                    purpose: str = "tournament") -> tuple[str, bool, bool]:
    """One budget-gated jury ask. Returns (answer, ok, blocked_or_failed).

    ``blocked_or_failed`` marks asks that never produced a jury opinion —
    a P20 budget block or a provider failure. Those must not be recorded
    as matches (nothing happened) and must END a batch: retrying the same
    ask within the same seconds only hammers a dead/budgeted session.
    ``llm`` overrides the context's client (the second judge uses the SAME
    P20 budget gate — it spends the lab's shared session slots too)."""
    if llm is None:
        llm = getattr(ctx, "llm", None)
    if llm is None:
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
        resp = await llm.ask(prompt, mode="concise", purpose=purpose,
                             system=_JURY_SYSTEM)
        ok = bool(resp and resp.ok)
        answer = (resp.answer or "") if resp else ""
    except Exception as exc:  # noqa: BLE001 — one match must not kill a cycle
        log.warning("[%s] tournament ask failed: %s", ctx.agent_id, exc)
        answer, ok, blocked_or_failed = "", False, True
    else:
        blocked_or_failed = not ok
    if budget is not None:
        budget.record(ctx.agent_id, purpose, ok, time.monotonic() - started)
    return answer, ok, blocked_or_failed


async def _dual_jury(ctx, prompt: str) -> tuple[str, bool, bool, str, bool]:
    """Judge one match with the primary jury AND the independent judge.

    Two model families looking at the same prompt at the same time is the
    whole point — so the asks run CONCURRENTLY when the Hyperbrowser pool
    can host both cloud sessions (>= 2 tokens; the judge client is built
    on its own token by get_judge_client). On the ONE-token lab the
    single session slot cannot hold both sessions — racing them would
    make each client sweep the other off the slot mid-ask — so the two
    asks run back-to-back instead. The second ask never raises: a judge
    outage degrades to the primary verdict (ok2=False), it never blocks
    a match by itself."""
    judge2 = getattr(ctx, "judge_llm", None)
    if judge2 is None:
        answer, ok, stopped = await _jury_ask(ctx, prompt)
        return answer, ok, stopped, "", False
    primary_key = getattr(getattr(ctx, "llm", None), "api_key", "") or ""
    judge_key = getattr(judge2, "api_key", "") or ""
    if not primary_key or not judge_key or primary_key == judge_key:
        # one session slot for two juries: serialize honestly
        answer, ok, stopped = await _jury_ask(ctx, prompt)
        if stopped:
            return answer, ok, stopped, "", False
        answer2, ok2, _ = await _jury_ask(
            ctx, prompt, llm=judge2, purpose="tournament_judge2")
        return answer, ok, stopped, answer2, ok2
    second = asyncio.ensure_future(
        _jury_ask(ctx, prompt, llm=judge2, purpose="tournament_judge2"))
    answer, ok, stopped = await _jury_ask(ctx, prompt)
    try:
        answer2, ok2, _stopped2 = await second
    except Exception as exc:  # noqa: BLE001 — judge outage must not kill the match
        log.warning("[%s] second judge crashed: %s", ctx.agent_id, exc)
        answer2, ok2 = "", False
    return answer, ok, stopped, answer2, ok2


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
    answer, ok, stopped, answer2, ok2 = await _dual_jury(ctx, prompt)
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

    # DUAL-JUDGE arbitration: an independent model family that DISAGREES
    # with the primary jury means the debate signal is not trustworthy —
    # record the match as contested with NO Elo movement. A judge that
    # failed/unparseable degrades to the primary verdict, noted openly.
    v2 = pairing.parse_better_idea(answer2) if ok2 else None
    if v2 is not None and v2 != choice:
        rationale = (f"dual-judge CONTESTED: primary=h{choice}, "
                     f"second=h{v2}; primary rationale: {answer[:500]}")
        ctx.memory.record_tournament_match(
            mid, ctx.agent_id, a["id"], b["id"], "contested", "",
            elo_a, elo_b, None, None, rationale=rationale[:1200])
        try:
            ctx.memory.journal(ctx.agent_id, "tournament_contested", {
                "match": mid, "a": a["id"], "b": b["id"],
                "primary": choice, "second": v2,
            })
        except Exception:  # noqa: BLE001
            pass
        return {"match_id": mid, "mode": "contested", "winner": "",
                "a": a["id"], "b": b["id"],
                "primary": choice, "second": v2}

    # "1" == the anchor (a), "2" == the opponent (b)
    winner = "a" if choice == 1 else "b"
    note = ""
    if getattr(ctx, "judge_llm", None) is not None:
        note = ("dual-judge agreed" if (ok2 and v2 == choice)
                else "judge2 unavailable/unparseable; primary verdict stands")
    upd = update_elo(elo_a, elo_b, winner,
                     matches_played_min=min(int(a.get("matches_played") or 0),
                                            int(b.get("matches_played") or 0)))
    # update_elo already works in (a, b) space: a_after is a's new rating and
    # b_after is b's new rating no matter who won. An earlier version swapped
    # the two on the b branch (porting artefact, found live in the Nov-2026
    # pre-production eval): every b-win then moved Elo the WRONG way — the
    # winner lost points and the loser gained — corrupting half of the
    # tournament's quality signal. Keep the direct assignment.
    elo_a_after = upd.elo_a_after
    elo_b_after = upd.elo_b_after
    applied = ctx.memory.record_tournament_match(
        mid, ctx.agent_id, a["id"], b["id"], mode, winner,
        elo_a, elo_b, elo_a_after, elo_b_after,
        rationale=(f"[{note}] {answer}" if note else answer)[:1200])
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
        # Proximity-aware pairing (#5): lexical similarity keeps clone-vs-
        # clone debates out of the queue (bucket 1 skips near-identical
        # opponents; bucket 2 weights pairs by 1 - similarity). Degrades to
        # unweighted selection on any per-pair failure, never an error.
        pair = pairing.select_pair(pool, similarity=pairing.jaccard_similarity)
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
                "contested": sum(1 for r in results if r["mode"] == "contested"),
            })
        except Exception:  # noqa: BLE001
            pass
    return results
