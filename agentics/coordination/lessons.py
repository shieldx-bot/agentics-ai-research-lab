"""Swarm-wide lessons — failure signatures aggregated across ALL agents.

Gap closed: the planner already saw (a) its OWN lessons (personal memory)
and (b) raw trace rows from other agents (``trace_context``, 3-4 rows).
What no prompt ever showed is the PATTERN: the same failure signature
repeated by three different agents is no longer "a trace", it is proof the
approach never worked — exactly the (xN) aggregation the personal-lesson
renderer has always had, but over the whole swarm.

Closest prior art: Voyager's lesson loop ("store sequences that work and
remember what does not"), Reflexion's verbal feedback across attempts, and
GEA's group experience sharing — the lab's traces were written for exactly
this but only ever read back row-by-row.

Mechanical, bounded, empty-safe: normalized signature = the first ~110
characters of the failure note (or the attempted approach when no note was
recorded), whitespace-collapsed; ``xN`` = how many distinct agents' attempts
ended in that signature; ordered by repeats, then alphabetically for
determinism.
"""
from __future__ import annotations

from agentics.memory.shared_memory import SharedMemory

_SIG_MAX = 110


def swarm_lessons(
    memory: SharedMemory,
    category: str | None,
    exclude_agent: str | None = None,
    limit: int = 4,
    max_chars: int = 800,
) -> str:
    """Aggregated failure patterns for ``category`` ("" when none).

    ``exclude_agent`` mirrors ``trace_context``: the planner's OWN failures
    arrive separately (personal lessons) and would be double-counted here.
    """
    if not category:
        return ""
    fails = memory.recent_traces(limit=30, category=category,
                                 failures_only=True, exclude_agent=exclude_agent)
    if not fails:
        return ""
    counts: dict[str, dict] = {}
    for t in fails:                     # recent_traces: oldest -> newest
        sig = " ".join(str(t.get("failure") or t.get("attempted") or "").split())
        if not sig:
            continue
        sig = sig[:_SIG_MAX]
        slot = counts.setdefault(sig, {"n": 0, "last_agent": t["agent"]})
        slot["n"] += 1
        slot["last_agent"] = t["agent"]
    if not counts:
        return ""
    top = sorted(counts.items(), key=lambda kv: (-kv[1]["n"], kv[0]))[:limit]
    lines = [f"- {sig} (x{v['n']}; last: {v['last_agent']})"
             for sig, v in top]
    text = "\n".join(lines)
    return text if len(text) <= max_chars else text[: max_chars - 1] + "…"
