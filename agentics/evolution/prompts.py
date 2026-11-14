"""DSPy-lite prompt evolution (wave-2 #2): the role missions EVOLVE on real
cycle outcomes instead of only re-synthesizing after failures.

Why (design, 2026-11-14): the existing role_prompts machinery is a single
champion per (role, category) with uses/successes counters and a
rework-after-N-failures TTL — selection without VARIATION. DSPy/OpenEvolve
show the leverage is in the population: keep 3 mutation arms per role, serve
them on exploration cycles, keep the score card on EXIT-CODE outcomes (a
cycle settles without error = a win; never an LLM's opinion), and promote
the arm that beats the champion's win-rate by a margin. Mutation itself
runs on the LOCAL router (purpose `prompt_evolve`) — zero cookie cost, and
the science that follows still runs pool-side.

Contract:
* never raises — a broken evolution is worse than a static prompt, so every
  entry point degrades to the champion mission;
* selection signal is machine truth only (bump_prompt_feedback's success
  flag, which reflect derives from state["error"]);
* the pool is bounded (oldest arms retire) and promotion reuses
  store_role_prompt, so TTL/rework/audit machinery stays untouched.
"""
from __future__ import annotations

import logging
import random

from agentics.llm import local_router

log = logging.getLogger("evolution.prompts")

_MUTATE_PROMPT = (
    "You are improving the operating instructions of a research agent.\n"
    "Current mission (the incumbent):\n---\n{mission}\n---\n"
    "Recent outcomes of cycles run under it:\n{experience}\n"
    "Write ONE improved variant of the mission: keep everything that works, "
    "fix the failure patterns listed above, stay under {max_chars} "
    "characters, same role ({role}), no preamble.\n"
    "Reply ONLY with the new mission text."
)


def _flag(settings, name: str, default):
    return getattr(settings, name, default)


async def maybe_variant(ctx, role: str, category: str,
                        champion: str) -> tuple[str, int]:
    """Serve the champion OR an armed challenger for this cycle.

    Returns (mission, variant_id). variant_id == 0 means the champion is
    serving. With probability `prompt_explore_p` a challenger is Thompson-
    sampled (Beta(w+1, l+1) per arm); otherwise the champion serves. When
    the pool is underfilled, ONE mutation is requested from the local
    router first (best-effort; local down -> champion serves)."""
    variant_id = 0
    settings = getattr(ctx, "settings", None)
    memory = getattr(ctx, "memory", None)
    if memory is None or not bool(_flag(settings, "prompt_evolution_enabled",
                                        True)):
        return champion, 0
    try:
        pool = memory.prompt_variants(role, category)
        pool_size = getattr(settings, "prompt_variant_pool", 3)
        pool_size = 3 if pool_size is None else int(pool_size)
        if len(pool) < pool_size:
            await _mutate_into_pool(ctx, role, category, champion, pool_size)
            pool = memory.prompt_variants(role, category)
        if not pool:
            return champion, 0
        explore_p = getattr(settings, "prompt_explore_p", 0.25)
        explore_p = 0.25 if explore_p is None else float(explore_p)
        if random.random() >= explore_p:
            return champion, 0
        arm = max(pool, key=lambda v: random.betavariate(
            v["wins"] + 1.0, v["losses"] + 1.0))
        log.info("[%s] prompt variant v#%s serves %s/%s (Thompson sample)",
                 getattr(ctx, "agent_id", "?"), arm["id"], role, category)
        try:
            ctx.memory.journal(ctx.agent_id, "prompt_variant_served",
                               {"variant": arm["id"], "role": role,
                                "category": category})
        except Exception:  # noqa: BLE001
            pass
        return arm["mission"], int(arm["id"])
    except Exception as exc:  # noqa: BLE001 — degrade to champion
        log.warning("prompt evolution variant pick failed: %s", exc)
        return champion, 0


async def _mutate_into_pool(ctx, role: str, category: str, champion: str,
                            pool_size: int) -> None:
    """One local-router mutation of the champion into the challenger pool.

    Never raises; never uses the cookie pool."""
    settings = getattr(ctx, "settings", None)
    if not local_router.allowed_for(settings, "prompt_evolve"):
        return
    # Wave-3 #1 — GEPA reflection first: when the champion's recent
    # experience shows failures, a critique-then-rewrite arm beats a blind
    # mutation. The plain mutation below stays the fallback (and the only
    # path for healthy champions with nothing to reflect on).
    try:
        from agentics.evolution import gepa_driver
        if await gepa_driver.regenerate(ctx, role, category, champion):
            return
    except Exception:  # noqa: BLE001 — reflection is sugar
        pass
    try:
        experience = ctx.memory.role_experience(role, category, limit=6)
        prompt = _MUTATE_PROMPT.format(
            mission=champion[:3000],
            experience=experience[:1200] or "(no history yet)",
            max_chars=max(600, len(champion) + 200),
            role=role)
        resp = await local_router.ask(
            prompt, settings=settings, purpose="prompt_evolve")
        if resp is None or not resp.ok:
            return
        mission = (resp.answer or "").strip()
        if len(mission) < 200 or mission == champion[:len(mission)]:
            return
        vid = ctx.memory.store_prompt_variant(role, category, mission,
                                              origin="mutate",
                                              pool_size=pool_size)
        if vid:
            ctx.memory.journal(ctx.agent_id, "prompt_variant_born",
                               {"variant": vid, "role": role,
                                "category": category})
    except Exception as exc:  # noqa: BLE001 — mutation is sugar
        log.debug("prompt mutation skipped: %s", exc)


def note_variant_feedback(ctx, state: dict) -> None:
    """Feed the cycle outcome into the variant that served this cycle
    (called from bump_prompt_feedback, which already derived `success`).

    Also runs the promotion check: an arm with enough samples that beats
    the champion by `prompt_variant_promote_edge` becomes the new champion
    through the ordinary store_role_prompt path."""
    meta = getattr(ctx, "meta", None)
    vid = int(meta.get("_prompt_variant_id") or 0) if isinstance(meta, dict) \
        else 0
    if not vid:
        return
    settings = getattr(ctx, "settings", None)
    memory = getattr(ctx, "memory", None)
    if memory is None:
        return
    success = not state.get("error")
    try:
        memory.record_prompt_variant_outcome(vid, success)
        if isinstance(meta, dict):
            meta["_prompt_variant_id"] = 0
        promoted = memory.promote_prompt_variant(
            vid,
            min_uses=int(_flag(settings, "prompt_variant_promote_min", 10)
                         or 10),
            edge=float(_flag(settings, "prompt_variant_promote_edge", 0.15)
                       or 0.15))
        if promoted:
            log.info("prompt variant v#%s PROMOTED to champion (%s)",
                     vid, "win" if success else "sample")
            try:
                ctx.memory.journal(ctx.agent_id, "prompt_variant_promoted",
                                   {"variant": vid})
            except Exception:  # noqa: BLE001
                pass
    except Exception as exc:  # noqa: BLE001 — feedback must not kill reflect
        log.debug("variant feedback failed: %s", exc)
