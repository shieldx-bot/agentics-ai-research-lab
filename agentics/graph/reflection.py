"""Reflection tree (wave-2 #5, Generative-Agents style): periodically the
supervisor SYNTHESIZES the world model's raw notes into higher-level
insights, on the LOCAL router — zero cookie cost.

Why (2026-11-14): world_notes is a flat ledger where only the newest 12
notes reach a prompt; a 7-agent swarm running hundreds of cycles loses the
big patterns ("every protocol that used metric X failed because Y") under
the routine verified/refuted stream. Reflection compresses the stream into
a small tree: level-1 insights synthesize raw notes, level-2 synthesize
level-1s, depth capped at 3. Ideation seeds from insights instead of raw
findings, so the swarm proposes from patterns, not anecdotes.

Honesty contract (the anti-hallucination spine):
* every insight MUST cite the note ids it was synthesized FROM (parents);
  citations are validated against real note ids — a fabricated provenance
  is dropped, never stored;
* insight text is stored verbatim-bounded (400 chars) with the reflector's
  agent id and a `reflect#<cycle>` source tag — no hidden authorship;
* a later refuted settle that contradicts an insight's parents lets the
  NEXT reflection supersede it (status via kind='insight' replaced by the
  newest — the bounded 400-note window does the forgetting naturally);
* never raises, never touches the cookie pool, depth capped, one
  reflection per reflect_every_cycles supervisor cycles.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re

from agentics.llm import local_router

log = logging.getLogger("graph.reflection")

_REFLECT_PROMPT = (
    "You are the reflection pass of a research lab's shared world model.\n"
    "Recent structured notes (id: text):\n{notes}\n\n"
    "Synthesize at most {max_insights} HIGHER-LEVEL insights that these "
    "notes jointly imply (a repeated failure pattern, an emerging theme, a "
    "contradiction worth resolving). Do NOT restate a single note.\n"
    "Reply ONLY JSON: {{\"insights\": [{{\"text\": \"one insight\", "
    "\"parents\": [<note ids it is based on>]}}]}}"
)

_ID_RE = re.compile(r"\d+")


def _parse_insights(answer: str) -> dict:
    """Tolerant JSON read: strip code fences, take the outermost object.
    Never raises — garbage in, {} out (then the caller stores nothing)."""
    text = (answer or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text).rstrip("`").strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {}


def _recent_notes(memory, limit: int = 14) -> list[dict]:
    """Raw notes (level 0) plus level-1 insights available for synthesis.
    Only kinds that carry empirical weight feed a reflection."""
    keep = ("verified", "refuted", "gap", "insight")
    items = [i for i in memory.world_items(limit=limit * 2)
             if i["kind"] in keep and i["level"] <= 1]
    return items[:limit]


def _parents_valid(parents, known_ids: set[int]) -> list[int]:
    """Provenance gate: parents must be REAL note ids, >= 1 of them, all
    cited within the offered window. Fabrications are dropped wholesale —
    an insight without honest provenance has no value to the swarm."""
    if not isinstance(parents, list):
        return []
    ids: list[int] = []
    for p in parents:
        try:
            pid = int(p)
        except (TypeError, ValueError):
            continue
        if pid in known_ids and pid not in ids:
            ids.append(pid)
    return ids


async def reflect_once(ctx) -> int:
    """Run ONE reflection pass. Returns the number of insights stored."""
    settings = getattr(ctx, "settings", None)
    memory = getattr(ctx, "memory", None)
    if memory is None or not local_router.allowed_for(settings, "reflect"):
        return 0
    notes = _recent_notes(memory)
    min_notes = int(getattr(settings, "reflect_min_notes", 5) or 5)
    if len(notes) < min_notes:
        return 0
    notes_text = "\n".join(
        f"{i['id']} [{i['kind']}] {i['text'][:180]}" for i in notes)
    known_ids = {i["id"] for i in notes}
    resp = await local_router.ask(
        _REFLECT_PROMPT.format(notes=notes_text, max_insights=2),
        settings=settings, purpose="reflect")
    if resp is None or not resp.ok:
        return 0
    parsed = _parse_insights(resp.answer)
    insights = parsed.get("insights") if isinstance(parsed, dict) else None
    if not isinstance(insights, list):
        return 0
    stored = 0
    for ins in insights[:2]:
        if not isinstance(ins, dict):
            continue
        text = str(ins.get("text", "")).strip()
        parents = _parents_valid(ins.get("parents"), known_ids)
        if not text or not parents:
            continue  # fabricated or unsupported provenance -> dropped
        level = 1 + max(
            (n["level"] for n in notes if n["id"] in parents), default=0)
        if level > 3:
            continue
        memory.world_add(ctx.agent_id, "insight", text[:400],
                         source=f"reflect#{getattr(ctx, 'cycle', 0)}",
                         level=min(3, level), parents=parents)
        stored += 1
    if stored:
        try:
            memory.journal(ctx.agent_id, "reflection_stored",
                           {"insights": stored})
        except Exception:  # noqa: BLE001
            pass
        log.info("reflection stored %d insight(s) (level<=3, provenance ok)",
                 stored)
    return stored


def maybe_reflect(ctx) -> None:
    """Sync entry for node_scan (LangGraph runs sync nodes in executor
    threads, no running loop there): every `reflect_every_cycles` cycles,
    run one reflection pass on the local router."""
    settings = getattr(ctx, "settings", None)
    every_n = int(getattr(settings, "reflect_every_cycles", 6) or 6)
    if every_n <= 0 or (getattr(ctx, "cycle", 0) % every_n) != 0:
        return
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(reflect_once(ctx))
        return
    # a loop IS running (safety path): schedule instead of nesting
    asyncio.get_running_loop().create_task(reflect_once(ctx))


__all__ = ["reflect_once", "maybe_reflect", "_parents_valid"]
