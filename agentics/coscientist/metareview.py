"""Meta-review — distil win/loss patterns into steering feedback.

Ported from the Apache-2.0 Kaimen-Inc/Co-Scientist re-implementation
(co_scientist/agents/metareview.py, system-feedback path). The reference
injects this feedback into future Generation and Evolution prompts; here
it steers the tournament jury and the evolution engine the same way,
closing the self-improvement loop the paper describes:

    debate -> Elo -> evolution -> debate ... with meta-review feedback
    flowing backwards through every stage.
"""
from __future__ import annotations

import logging

from agentics.coscientist.tournament import _jury_ask

log = logging.getLogger("coscientist.metareview")

_META_SYSTEM = (
    "You are the meta-reviewer of an AI research lab's hypothesis "
    "tournament. You synthesise debate patterns into compact, actionable "
    "steering notes. No preamble, no headings."
)


def should_synthesize(memory, window: int = 4) -> bool:
    """True when `window` fresh debates accumulated since the last synthesis."""
    try:
        recent = memory.recent_tournament_matches(limit=int(window))
        if len(recent) < int(window):
            return False
        fb = memory.latest_tournament_feedback()
        newest_match_ts = max(r["ts"] for r in recent)
        return fb is None or newest_match_ts > (fb.get("ts") or 0.0)
    except Exception:          # noqa: BLE001 — a routing read must never crash
        return False


def _debates_block(matches: list[dict]) -> str:
    lines = []
    for m in matches:
        if m.get("mode") == "invalid" or not m.get("winner"):
            continue
        a_after = (m.get("elo_a_after") if m["winner"] == "a"
                   else m.get("elo_b_after"))
        b_after = (m.get("elo_b_after") if m["winner"] == "a"
                   else m.get("elo_a_after"))
        lines.append(
            f"h#{m['hyp_a']} vs h#{m['hyp_b']} [{m['mode']}] -> winner "
            f"h#{m['hyp_a'] if m['winner'] == 'a' else m['hyp_b']} "
            f"(Elo {a_after:.0f} vs {b_after:.0f})\n"
            f"  jury: {(m.get('rationale') or '')[:400]}")
    return "\n".join(lines) if lines else "(no decisive debates yet)"


async def synthesize_feedback(ctx, goal: str, window: int = 4) -> str:
    """One budget-gated ask that turns the latest debates into a steering
    note, persisted for every future jury/evolution prompt. Returns the
    note (empty string on any failure — the loop keeps running unsteered)."""
    matches = ctx.memory.recent_tournament_matches(limit=int(window))
    block = _debates_block(matches)
    prompt = (
        f"Goal of the lab: {goal[:600]}\n\n"
        f"Recent tournament debates:\n{block[:4000]}\n\n"
        "Synthesise the recurring win/loss patterns into a compact steering "
        "note (max 120 words) for future hypothesis generation and debates."
    )
    answer, ok, _stopped = await _jury_ask(ctx, prompt)
    text = (answer or "").strip() if ok else ""
    if not text:
        return ""
    ev_id = ctx.memory.save_tournament_feedback(
        text, ctx.agent_id, len(matches))
    try:
        ctx.memory.journal(ctx.agent_id, "tournament_meta_review", {
            "feedback_id": ev_id, "window": len(matches),
            "text": text[:300]})
    except Exception:  # noqa: BLE001
        pass
    return text
