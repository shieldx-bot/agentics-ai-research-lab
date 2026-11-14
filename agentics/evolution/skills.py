"""Voyager-style skill library (wave-2 #1): protocols the lab SETTLED as
supported are distilled into reusable skills, and the coder's next protocol
starts from them instead of re-inventing from scratch.

Why (2026-11-14): every verified protocol is a one-off — the lab paid
cookie calls to find a correct measurement approach and then let it rot in
experiment_ledger. This module is the compounding memory: the supervisor
sweep distills recent supported settles into the `skill_library` table
(description via the LOCAL router, purpose skill_distill — zero cookie
cost; code is the settled protocol verbatim, no LLM rewriting), and the
protocol-writing / implement prompts get the token-overlap top-3 skills
injected ("adapt, don't reinvent").

Honesty contract:
* skills come ONLY from machine-verified settles (status='supported');
* the distilled description never replaces the code — the settled protocol
  source is stored verbatim and rendered with the skill;
* retrieval is plain token overlap (no embedding dependency), never
  raises, bumps `uses` so the dashboard shows which skills actually
  compound;
* everything degrades to the old prompt when the library is empty or the
  feature is off.
"""
from __future__ import annotations

import asyncio
import logging
import re

from agentics.llm import local_router

log = logging.getLogger("evolution.skills")

_WORD_RE = re.compile(r"[a-z0-9_]{3,}")
_STOP = frozenset({
    "the", "and", "for", "with", "that", "this", "than", "from", "are",
    "was", "were", "has", "have", "had", "not", "but", "its", "into",
    "when", "which", "will", "can", "may", "should", "must", "then",
    "their", "them", "they", "been", "being", "over", "under", "more",
    "less", "also", "each", "any", "all", "one", "two", "how", "why",
})

_DISTILL_PROMPT = (
    "Distill one REUSABLE skill from this research protocol that the lab "
    "empirically verified (exit-code settle).\n\n"
    "Claim it proved:\n{claim}\n"
    "Protocol source (read-only reference, do NOT rewrite it):\n"
    "{code}\n\n"
    "Reply ONLY JSON: {{\"name\": \"short snake_case skill name\", "
    "\"description\": \"what the skill does and WHEN to reuse it "
    "(<= 3 sentences)\"}}"
)


def relevant_skills(memory, claim_text: str, limit: int = 3) -> list[dict]:
    """Top skills by token overlap between the claim and the skill's
    name/description/metric. Bumps `uses` on the returned skills."""
    words = set(w for w in _WORD_RE.findall((claim_text or "").lower())
                if w not in _STOP)
    if not words:
        return []
    scored = []
    for skill in memory.library_skills(limit=60):
        hay = set(_WORD_RE.findall(
            f"{skill['name']} {skill['description']} "
            f"{skill['metric']}".lower())) - _STOP
        if not hay:
            continue
        overlap = len(words & hay)
        if overlap > 0:
            scored.append((overlap, len(words & hay) / len(hay), skill))
    scored.sort(key=lambda t: (-t[0], -t[1], -t[2]["uses"]))
    top = [s[2] for s in scored[:limit]]
    if top:
        try:
            memory.bump_skill_uses([s["id"] for s in top])
        except Exception:  # noqa: BLE001 — sugar
            pass
    return top


def skills_block(memory, claim_text: str, limit: int = 3) -> str:
    """Prompt section: reusable skills for this claim ('' when none)."""
    try:
        skills = relevant_skills(memory, claim_text, limit=limit)
    except Exception:  # noqa: BLE001 — retrieval must never break a prompt
        return ""
    if not skills:
        return ""
    lines = ["Reusable skills the lab empirically verified "
             "(adapt their approach; the full settled protocol is stored "
             "in the skill library):"]
    for s in skills:
        lines.append(f"- {s['name']}: {s['description'][:220]} "
                     f"(metric: {s['metric'] or 'n/a'}, "
                     f"reused {s['uses']}x)")
    return "\n".join(lines)[:1200]


def _parse_skill(answer: str) -> dict:
    """Tolerant JSON read of the distillation reply (never raises)."""
    text = (answer or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text).rstrip("`").strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        import json
        parsed = json.loads(text[start:end + 1])
        return parsed if isinstance(parsed, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _pending_settles(memory, limit: int = 2) -> list[dict]:
    """Supported hypotheses with protocol code and NO distilled skill yet."""
    try:
        with memory._lock:  # noqa: SLF001 — same-package sweep helper
            rows = memory._conn.execute(
                "SELECT id, claim, protocol_code FROM hypotheses "
                "WHERE status='supported' AND protocol_code != '' "
                "ORDER BY id DESC LIMIT 25").fetchall()
        pending = []
        for hid, claim, code in rows:
            if memory.skill_for_hyp(hid) is None:
                pending.append({"id": hid, "claim": claim, "code": code})
            if len(pending) >= limit:
                break
        return pending
    except Exception:  # noqa: BLE001
        return []


async def distill_pending(ctx, limit: int = 2) -> int:
    """Distill up to `limit` not-yet-distilled supported settles via the
    LOCAL router. Returns how many skills were stored."""
    settings = getattr(ctx, "settings", None)
    memory = getattr(ctx, "memory", None)
    if memory is None or not local_router.allowed_for(settings,
                                                      "skill_distill"):
        return 0
    stored = 0
    for hyp in _pending_settles(memory, limit=limit):
        try:
            resp = await local_router.ask(
                _DISTILL_PROMPT.format(claim=hyp["claim"][:800],
                                       code=hyp["code"][:4000]),
                settings=settings, purpose="skill_distill")
            if resp is None or not resp.ok:
                continue
            parsed = _parse_skill(resp.answer)
            name = str(parsed.get("name", "")).strip()
            desc = str(parsed.get("description", "")).strip()
            if not name:
                continue
            metric_row = memory.get_hypothesis(hyp["id"]) or {}
            try:
                crit = metric_row.get("criterion") or "{}"
                import json as _json
                metric = str(_json.loads(crit).get("metric", ""))[:60]
            except Exception:  # noqa: BLE001
                metric = ""
            sid = memory.add_skill(name, desc, code=hyp["code"],
                                   metric=metric, source_hyp=hyp["id"])
            if sid:
                stored += 1
                try:
                    memory.journal(ctx.agent_id, "skill_distilled",
                                   {"skill": sid, "hyp": hyp["id"],
                                    "name": name[:60]})
                except Exception:  # noqa: BLE001
                    pass
        except Exception as exc:  # noqa: BLE001 — distill is sugar
            log.debug("skill distill skipped: %s", exc)
    if stored:
        log.info("skill library: distilled %d new skill(s)", stored)
    return stored


def maybe_distill(ctx) -> None:
    """Sync entry for node_scan (executor threads have no running loop):
    every `skill_distill_every_cycles` cycles, distill pending settles."""
    settings = getattr(ctx, "settings", None)
    if not bool(getattr(settings, "skill_library_enabled", True)):
        return
    every_n = int(getattr(settings, "skill_distill_every_cycles", 4) or 4)
    if every_n <= 0 or (getattr(ctx, "cycle", 0) % every_n) != 0:
        return
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(distill_pending(ctx))
        return
    asyncio.get_running_loop().create_task(distill_pending(ctx))


__all__ = ["relevant_skills", "skills_block", "distill_pending",
           "maybe_distill"]
