"""GEPA-style reflective prompt evolution (roadmap #1) on top of the wave-2
DSPy-lite pool (`agentics/evolution/prompts.py`).

Wave-2 gave the lab SELECTION (champion vs mutation arms, Thompson sampling,
exit-code win/loss). GEPA adds the missing REFLECTION + VARIATION half:

1. reflect   — collect the champion, every armed arm's win-rate and the real
               failure lines of recent cycles (role_experience);
2. pareto    — parents are the non-dominated front on (win-rate up, mission
               length down): no arm is a parent just because it is old;
3. generate  — crossover the pareto parents (sentence interleave) + inject
               the dominant failure lesson; when the LOCAL router is allowed
               for `prompt_evolve` a reflective variant is requested from it,
               otherwise the deterministic generator alone produces the batch
               — the lab never blocks and never spends a cookie;
4. insert    — candidates enter the EXISTING challenger pool via
               store_prompt_variant(origin='gepa'), so selection, retirement,
               promotion and TTL machinery stay exactly as they are.

The lab is still the judge: a candidate only becomes a champion through
real cycle outcomes (P11 lifecycle), never through this module's opinion.
Every entry point degrades gracefully — a broken GEPA is worse than a
static prompt.
"""
from __future__ import annotations

import logging
import random
import re

from agentics.llm import local_router

log = logging.getLogger("evolution.gepa")

_REFLECT_PROMPT = (
    "You are the GEPA reflective optimizer for one agent role.\n"
    "Champion mission (incumbent):\n---\n{champion}\n---\n"
    "Pareto-parent missions (best win-rate per length):\n{parents}\n"
    "Real failure lines from recent cycles:\n{failures}\n"
    "Arm record (win-rate, length): {arms}\n"
    "Write ONE improved mission that keeps what works on the pareto front, "
    "directly attacks the failure lines, stays under {max_chars} chars, "
    "same role ({role}). Reply ONLY with the new mission text."
)


def pareto_front(items: list[dict], *, score_key: str = "score",
                 cost_key: str = "cost") -> list[dict]:
    """Non-dominated front: score higher-better, cost lower-better.

    An item is dominated when some other item has score >= its score AND
    cost <= its cost (with at least one strict). Ties on both axes keep the
    FIRST occurrence (stable input order), so callers with equal parents are
    deterministic. Pure function — heavily unit-tested."""
    best: list[dict] = []
    for i, a in enumerate(items):
        dominated = False
        for j, b in enumerate(items):
            if i == j:
                continue
            if (b[score_key] >= a[score_key] and b[cost_key] <= a[cost_key]
                    and (b[score_key] > a[score_key]
                         or b[cost_key] < a[cost_key])):
                dominated = True
                break
        if not dominated:
            best.append(a)
    # a same-score same-cost duplicate later in the list is still dominated
    # by its twin — keep only the first of each (score, cost) cell
    seen: set[tuple[float, float]] = set()
    out: list[dict] = []
    for it in best:
        key = (float(it[score_key]), float(it[cost_key]))
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", (text or "").strip())
    return [p.strip() for p in parts if len(p.strip()) >= 20]


def crossover(a: str, b: str, rng: random.Random) -> str:
    """Deterministic crossover: interleave parents' sentences, a-side first.

    Duplicates collapse; the result keeps each parent's operational
    sentences so a crossover cannot invent content — it can only RECOMBINE
    what the lab already served."""
    sa, sb = _sentences(a), _sentences(b)
    if not sa:
        return b
    if not sb:
        return a
    picked: list[str] = []
    seen: set[str] = set()
    for pair in zip(sa, sb):
        for s in pair:
            if s not in seen:
                seen.add(s)
                picked.append(s)
    longer = sa if len(sa) > len(sb) else sb
    for s in longer[min(len(sa), len(sb)):]:
        if s not in seen:
            seen.add(s)
            picked.append(s)
    rng.shuffle(picked)
    return " ".join(picked)


def _failure_lessons(reflection: dict, max_lessons: int = 2) -> list[str]:
    """Top repeated failure signatures from real cycle history — the lesson
    lines the generator injects into every child mission."""
    counts: dict[str, int] = {}
    for line in reflection.get("failures", []):
        sig = (line or "").strip().lower()[:120]
        if len(sig) < 8:
            continue
        counts[sig] = counts.get(sig, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [f"Avoid repeating this failure: {sig}." for sig, _n
            in ranked[:max(max_lessons, 0)]]


def collect_reflection(memory, role: str, category: str,
                       limit: int = 6) -> dict:
    """Everything the reflection needs, from the lab's own ledgers only."""
    champion = ""
    try:
        row = memory.get_role_prompt(role, category)
        champion = (row or {}).get("mission") or ""
    except Exception:  # noqa: BLE001
        champion = ""
    arms = []
    try:
        for v in memory.prompt_variants(role, category):
            uses = int(v.get("uses") or 0)
            wins = int(v.get("wins") or 0)
            arms.append({"id": v.get("id"), "mission": v.get("mission") or "",
                         "uses": uses, "wins": wins,
                         "score": (wins / uses) if uses else 0.0,
                         "cost": len(v.get("mission") or "")})
    except Exception:  # noqa: BLE001
        arms = []
    failures: list[str] = []
    try:
        experience = memory.role_experience(role, category, limit=limit) or ""
        failures = [ln.strip("- ").strip() for ln in experience.splitlines()
                    if ln.strip()]
    except Exception:  # noqa: BLE001
        failures = []
    return {"role": role, "category": category, "champion": champion,
            "arms": arms, "failures": failures}


def offline_candidates(reflection: dict, n: int = 3,
                       seed: int = 0) -> list[str]:
    """Deterministic GEPA batch WITHOUT any LLM: pareto parents ->
    crossover -> failure-lesson injection. Honest fallback, hermetic tests.
    With a single parent the variants are sentence REORDERINGS (nothing is
    invented) plus lesson placement — distinct, but still only recombination."""
    champion = reflection.get("champion") or ""
    pool: list[dict] = [{"mission": champion, "score": 1.0,
                         "cost": len(champion)}] if champion else []
    pool += [{"mission": a["mission"], "score": a["score"], "cost": a["cost"]}
             for a in reflection.get("arms", []) if a.get("mission")]
    parents = pareto_front(pool)
    lessons = _failure_lessons(reflection)
    rng = random.Random(seed)
    out: list[str] = []
    k = len(parents)
    for i in range(max(0, int(n))):
        if k == 0:
            break
        if k == 1:
            sents = _sentences(parents[0]["mission"])
            if not sents:
                sents = [parents[0]["mission"]]
            if i % 2:                       # alternate: reversed vs original
                sents = list(reversed(sents))
            text = " ".join(sents)
        else:
            a = parents[i % k]["mission"]
            b = parents[(i + 1) % k]["mission"]
            text = crossover(a, b, rng)
        if lessons:
            # alternate lesson placement so single-parent batches still differ
            text = (text + " " + " ".join(lessons) if i % 2 == 0
                    else " ".join(lessons) + " " + text)
        text = text.strip()
        if len(text) >= 80 and text not in out:
            out.append(text)
    return out


async def reflective_candidates(memory, settings, role: str, category: str,
                                n: int = 3, seed: int = 0) -> list[str]:
    """GEPA batch: LOCAL-router reflection when allowed, deterministic
    generator otherwise. Never raises; empty list when there is nothing
    honest to breed from (no champion AND no arms)."""
    if not getattr(settings, "gepa_enabled", True):
        return []
    reflection = collect_reflection(memory, role, category)
    if not reflection["champion"] and not reflection["arms"]:
        return []
    texts: list[str] = []
    if local_router.allowed_for(settings, "prompt_evolve"):
        try:
            parents = pareto_front(
                [{"mission": reflection["champion"], "score": 1.0,
                  "cost": len(reflection["champion"])}]
                + [{"mission": a["mission"], "score": a["score"],
                    "cost": a["cost"]}
                   for a in reflection["arms"] if a.get("mission")])
            prompt = _REFLECT_PROMPT.format(
                champion=reflection["champion"][:3000],
                parents="\n".join(f"- {p['mission'][:400]}"
                                  for p in parents) or "(none)",
                failures="\n".join(reflection["failures"][:8])
                or "(no failures recorded)",
                arms=", ".join(f"{a['score']:.0%}({a['cost']}c)"
                               for a in reflection["arms"]) or "(none)",
                max_chars=max(600, len(reflection["champion"]) + 200),
                role=role)
            resp = await local_router.ask(prompt, settings=settings,
                                          purpose="prompt_evolve")
            if resp is not None and getattr(resp, "ok", False):
                mission = (getattr(resp, "answer", "") or "").strip()
                if len(mission) >= 120:
                    texts.append(mission)
        except Exception as exc:  # noqa: BLE001 — reflection is sugar
            log.debug("gepa reflective ask skipped: %s", exc)
    for text in offline_candidates(reflection, n=n, seed=seed):
        if text not in texts:
            texts.append(text)
    return texts[:max(1, int(n))]


def evolve(memory, settings, role: str, category: str, texts: list[str]) -> list[int]:
    """Insert a GEPA batch into the existing challenger pool (origin='gepa').

    Returns the inserted variant ids; the pool's own size cap retires the
    oldest arms, so GEPA cannot bloat it. The champion is never touched."""
    ids: list[int] = []
    pool_size = int(getattr(settings, "prompt_variant_pool", 3) or 3)
    for text in texts:
        try:
            vid = memory.store_prompt_variant(role, category, text,
                                              origin="gepa",
                                              pool_size=pool_size)
        except Exception:  # noqa: BLE001
            vid = 0
        if vid:
            ids.append(vid)
    return ids
