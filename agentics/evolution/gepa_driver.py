"""GEPA-style reflective prompt regeneration (#1 of the advanced-tech
roadmap, wave-3).

DSPy-lite (wave-2 #2) MUTATES the champion on a fixed template. GEPA's
leverage is the REFLECTION: read the actual failure traces, explain WHY
the mission keeps failing, THEN rewrite against that explanation. This
driver adds that two-step reflection for (role, category) pairs whose
recent experience actually contains failures; healthy champions keep
serving through the plain mutation path.

Honest boundaries (straight from the roadmap):
* GEPA is only a GENERATOR — the lab stays the judge: candidates enter
  through the ordinary ``store_prompt_variant`` arm pool, get scored on
  exit-code truth by the existing feedback loop, and get promoted by the
  existing ``promote_prompt_variant`` machinery. Nothing else changes;
* generation runs on the LOCAL router (purpose ``prompt_evolve``) —
  zero cookie cost; a missing local model degrades to "no new arm";
* the P16 verifier prompt is never touched (criteria stay frozen).
"""
from __future__ import annotations

import logging

from agentics.llm import local_router

log = logging.getLogger("evolution.gepa")

_CRITIQUE_PROMPT = (
    "You are auditing a research agent's operating instructions.\n"
    "Mission under review:\n---\n{mission}\n---\n"
    "Recent cycle outcomes recorded under it (machine/exit-code truth, "
    "not opinions):\n{experience}\n"
    "Step 1 — in 2-4 sentences, name the CONCRETE failure patterns in "
    "those outcomes that this mission causes or fails to prevent.\n"
    "Step 2 — write ONE improved mission that fixes exactly those "
    "patterns: keep everything that works, stay under {max_chars} "
    "characters, same role ({role}).\n"
    "Reply EXACTLY in this format:\n"
    "CRITIQUE: <your 2-4 sentences>\n"
    "MISSION: <the improved mission text>"
)

_FAILURE_MARKERS = ("fail", "error", "rework", "refus", "✗", "exception")


def _has_failures(experience: str) -> bool:
    low = (experience or "").lower()
    return any(marker in low for marker in _FAILURE_MARKERS)


def parse_critique_mission(answer: str) -> tuple[str, str]:
    """Split 'CRITIQUE: ... MISSION: ...'; ('', '') when the shape is odd.

    Tolerant to case and to the MISSION line wrapping multiple lines
    (everything after the last MISSION: marker is the mission)."""
    if not answer:
        return "", ""
    text = answer.strip()
    critique, mission = "", ""
    lower = text.lower()
    mi = lower.rfind("mission:")
    if mi >= 0:
        mission = text[mi + len("mission:"):].strip()
        ci = lower.find("critique:")
        if 0 <= ci < mi:
            critique = text[ci + len("critique:"):mi].strip()
    return critique, mission


async def regenerate(ctx, role: str, category: str, champion: str) -> int:
    """Reflection-driven regeneration for a failing (role, category).

    Returns the new variant id (0 = nothing stored — no failures in the
    recent experience, local model unavailable, or an unusable answer).
    Never raises.
    """
    settings = getattr(ctx, "settings", None)
    memory = getattr(ctx, "memory", None)
    if memory is None or not bool(getattr(settings, "gepa_reflection_enabled",
                                          True)):
        return 0
    if not local_router.allowed_for(settings, "prompt_evolve"):
        return 0
    try:
        experience = memory.role_experience(role, category, limit=8)
        if not _has_failures(experience):
            return 0
        prompt = _CRITIQUE_PROMPT.format(
            mission=champion[:3000],
            experience=experience[:1500] or "(no history yet)",
            max_chars=max(600, len(champion) + 200),
            role=role)
        resp = await local_router.ask(prompt, settings=settings,
                                      purpose="prompt_evolve")
        if resp is None or not getattr(resp, "ok", False):
            return 0
        critique, mission = parse_critique_mission(resp.answer or "")
        if not mission or len(mission) < 200 \
                or mission == champion[:len(mission)]:
            return 0
        pool_size = int(getattr(settings, "prompt_variant_pool", 3) or 3)
        vid = memory.store_prompt_variant(role, category, mission,
                                          origin="gepa",
                                          pool_size=pool_size)
        if vid:
            log.info("[%s] GEPA arm v#%s born for %s/%s (reflective)",
                     getattr(ctx, "agent_id", "?"), vid, role, category)
            try:
                ctx.memory.journal(ctx.agent_id, "gepa_reflection",
                                   {"variant": vid, "role": role,
                                    "category": category,
                                    "critique": critique[:400]})
            except Exception:  # noqa: BLE001
                pass
        return int(vid or 0)
    except Exception as exc:  # noqa: BLE001 — reflection is sugar, never fatal
        log.debug("gepa regeneration skipped: %s", exc)
        return 0
