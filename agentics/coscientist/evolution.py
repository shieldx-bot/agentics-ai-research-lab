"""Evolution — breed the next generation from top-Elo hypotheses.

Ported from the Apache-2.0 Kaimen-Inc/Co-Scientist re-implementation
(co_scientist/agents/evolution.py), two operators that fit the lab's
budget edge:

* ``combine``   — merge the most idea-distant pair of the top-k into one
                  stronger, MORE testable hypothesis (the paper's ablation
                  credits iterative evolution most);
* ``out_of_box`` — one original hypothesis inspired by (not aggregating)
                  the top-k.

Every bred hypothesis enters memory as a normal ``proposed`` row with
``parents`` lineage, so it rides the SAME lifecycle as everything else:
the tournament ranks it, and only P16 code verification ever settles it.
Dedup is honest and cheap: a claim the swarm already holds (settled OR
merely proposed) is not re-bred — the ask is skipped, not discarded.
"""
from __future__ import annotations

import logging
import re

from agentics.coscientist import pairing
from agentics.coscientist.tournament import _jury_ask, feedback_block

log = logging.getLogger("coscientist.evolution")

_EVO_SYSTEM = (
    "You are the evolution engine of a self-improving AI research lab. "
    "You breed new research hypotheses from the strongest existing ones. "
    "Reply in the exact format requested; no preamble."
)

_RECORD_RE = re.compile(
    r"^\s*TITLE\s*:\s*(?P<title>.+?)\s*\n"
    r"\s*CLAIM\s*:\s*(?P<claim>.+?)\s*\n"
    r"\s*METRIC\s*:\s*(?P<metric>.+?)\s*$",
    re.DOTALL | re.IGNORECASE,
)


def parse_record(text: str) -> dict | None:
    """Parse the TITLE/CLAIM/METRIC record; None when the reply broke the
    format (a malformed record is never guessed into a hypothesis)."""
    if not text:
        return None
    m = _RECORD_RE.search(text.strip())
    if not m:
        return None
    title = (m.group("title") or "").strip().splitlines()
    claim = " ".join((m.group("claim") or "").split())
    metric = re.sub(r"[^a-z0-9_]", "_",
                    (m.group("metric") or "").strip().lower()).strip("_")
    if not claim or len(claim) < 20:
        return None
    return {"title": (title[0] if title else "")[:200],
            "claim": claim[:500], "metric": metric[:60]}


def _prompt_body(memory, ask_prompt: str) -> str:
    return ask_prompt


async def evolve_top(ctx, goal: str, top_k: int = 3,
                     strategy: str = "auto") -> dict | None:
    """One evolution ask over the current top-Elo pool.

    Returns the result dict {strategy, parent_ids, hypothesis_id} or None
    (pool too small, ask blocked, unparseable reply, duplicate claim) —
    failure-isolated like every tournament operator."""
    top = ctx.memory.top_by_elo(limit=max(2, int(top_k)))
    if not top:
        return None
    fb = feedback_block(ctx.memory)

    if strategy == "auto":
        strategy = "combine" if len(top) >= 2 else "out_of_box"

    if strategy == "combine" and len(top) >= 2:
        # most idea-distant pair of the top set — the reference's pick.
        # Without an embedder (the lab's vector tier is optional) the pair
        # is the two highest-rated, exactly like the reference degrades.
        a, b = top[0], top[1]
        prompt = pairing.EVOLVE_COMBINE_PROMPT.format(
            goal=goal[:600], id1=f"h{a['id']}", text1=(a.get("claim") or "")[:1200],
            id2=f"h{b['id']}", text2=(b.get("claim") or "")[:1200],
            feedback_block=fb)
        parents = [a["id"], b["id"]]
    elif strategy == "out_of_box":
        inspirations = "\n\n".join(
            f"<HYPOTHESIS_TEXT id=\"h{h['id']}\">\n{(h.get('claim') or '')[:800]}\n"
            "</HYPOTHESIS_TEXT>" for h in top[:5])
        prompt = pairing.EVOLVE_OOB_PROMPT.format(
            goal=goal[:600], inspirations=inspirations, feedback_block=fb)
        parents = [h["id"] for h in top[:5]]
    else:
        return None

    answer, ok, _stopped = await _jury_ask(ctx, prompt)
    if not ok:
        return None
    record = parse_record(answer)
    if record is None:
        try:
            ctx.memory.journal(ctx.agent_id, "evolution_invalid_record",
                               {"strategy": strategy, "reply": answer[:300]})
        except Exception:  # noqa: BLE001
            pass
        return None
    # H3-style dedup: a claim the swarm already holds (ANY status — settled,
    # proposed or under test) is not re-bred; the ask already happened, so
    # return the held row's identity instead of registering a duplicate.
    existing = ctx.memory.hypothesis_by_claim(record["claim"])
    if existing is not None:
        try:
            ctx.memory.journal(ctx.agent_id, "evolution_duplicate_claim", {
                "strategy": strategy, "held": existing["id"],
                "status": existing["status"]})
        except Exception:  # noqa: BLE001
            pass
        return {"strategy": strategy, "parent_ids": parents,
                "hypothesis_id": existing["id"], "title": record["title"],
                "claim": record["claim"], "duplicate": True}
    new_id = ctx.memory.add_hypothesis(
        ctx.agent_id, record["claim"], {}, task_id=None, finding_id=None,
        status="proposed", parents=parents)
    try:
        ctx.memory.journal(ctx.agent_id, "hypothesis_evolved", {
            "strategy": strategy, "parents": parents, "new": new_id,
            "metric_hint": record["metric"], "title": record["title"][:120],
        })
        ctx.memory.post_message(
            ctx.agent_id, "broadcast",
            f"[EVOLUTION] h#{new_id} bred via {strategy} from "
            f"{[f'h#{p}' for p in parents]}: {record['claim'][:160]}")
    except Exception:  # noqa: BLE001 — telemetry must never gate
        pass
    return {"strategy": strategy, "parent_ids": parents,
            "hypothesis_id": new_id, "title": record["title"],
            "claim": record["claim"]}
