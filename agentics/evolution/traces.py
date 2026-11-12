"""Turn the shared `traces` + `tool_registry` tables into prompt context.

GEA's key mechanism is that every agent reflects on *the group's* traces
(what was tried, what failed, how it ended), not only on its own. This module
is the lab's equivalent of Algorithm 2's "aggregate and share" step.
"""
from __future__ import annotations

import re
from collections import Counter

from agentics.memory.shared_memory import SharedMemory

# Files an evolution PR may touch. Everything else (policies, stop switch,
# GitHub/LLM plumbing) is out of bounds for self-modification.
EVOLUTION_ALLOWED_PATHS: tuple[str, ...] = (
    "agentics/agents/roles.py",
    "agentics/evolution/",
    "tests/",
    "docs/",
)

def evolution_path_allowed(path: str) -> bool:
    """Exact file match, or prefix match for directory entries (ending in '/')."""
    return any(path == a or (a.endswith("/") and path.startswith(a))
               for a in EVOLUTION_ALLOWED_PATHS)


_TOOL_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{2,40}$")


def valid_tool_name(name: str) -> bool:
    return bool(_TOOL_NAME_RE.match(name or ""))


def _clip(text: str, n: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _fmt_trace(t: dict) -> str:
    line = f"- [{t['agent']}|{t['category']}|{t['outcome']}] tried: {_clip(t['attempted'], 160)}"
    if t["failure"]:
        line += f" | failed: {_clip(t['failure'], 200)}"
    return line


def trace_context(
    memory: SharedMemory, agent_id: str, issue: int | None, category: str,
    max_chars: int = 2400,
) -> str:
    """Other agents' relevant traces: same-issue attempts, failures to avoid,
    and approaches that got merged in this category."""
    seen: set[int] = set()
    sections: list[tuple[str, list[dict]]] = []

    def take(title: str, rows: list[dict], limit: int) -> None:
        fresh = [r for r in rows if r["id"] not in seen][-limit:]
        seen.update(r["id"] for r in fresh)
        if fresh:
            sections.append((title, fresh))

    if issue is not None:
        take("Earlier attempts on THIS issue (by other agents)",
             memory.recent_traces(limit=5, issue=issue, exclude_agent=agent_id), 3)
    take("Failures in similar work — do NOT repeat these",
         memory.recent_traces(limit=8, category=category, failures_only=True,
                              exclude_agent=agent_id), 4)
    merged = [t for t in memory.recent_traces(limit=15, category=category,
                                              exclude_agent=agent_id)
              if t["outcome"] == "merged"]
    take("Approaches that were merged in similar work", merged, 3)

    out: list[str] = []
    for title, rows in sections:
        out.append(f"{title}:")
        out.extend(_fmt_trace(r) for r in rows)
    text = "\n".join(out) or "(no traces yet)"
    return text if len(text) <= max_chars else text[: max_chars - 1] + "…"


def tool_context(memory: SharedMemory, query: str, limit: int = 8, max_chars: int = 1500) -> str:
    tools = memory.list_tools(query, limit=limit)
    if not tools:
        return "(registry empty)"
    text = "\n".join(
        f"- {t['name']} ({t['path']}): {_clip(t['description'], 140)} [used {t['uses']}x]"
        for t in tools
    )
    return text if len(text) <= max_chars else text[: max_chars - 1] + "…"


def build_evolution_issue(memory: SharedMemory, window: int = 40) -> tuple[str, str]:
    """(title, body) for the periodic self-evolution issue. Evidence-first: the
    issue embeds failure traces so the Coder works from data, not vibes."""
    fails = memory.recent_traces(limit=window, failures_only=True)
    stats = memory.trace_stats()
    cats = Counter(t["category"] for t in fails)
    rate = "n/a" if stats["merge_rate"] is None else f"{stats['merge_rate']:.0%}"
    lines = [
        "Periodic **self-evolution** pass (GEA-style). Goal: improve the agents' own "
        "prompts using the group's failure traces.",
        "",
        "## Evidence",
        f"- Outcomes so far: `{stats['by_outcome']}` (merge rate: {rate})",
        f"- Failure traces considered: {len(fails)}; by category: {dict(cats)}",
        "",
        "### Recent failures",
    ]
    lines += [_fmt_trace(t) for t in fails[-15:]] or ["- (none recorded yet — do nothing)"]
    lines += [
        "",
        "## Rules (enforced by the pipeline)",
        "- You may only change: " + ", ".join(f"`{p}`" for p in EVOLUTION_ALLOWED_PATHS) + ".",
        "- Each prompt change must cite the trace ids / failure pattern it addresses.",
        "- Do not touch policies, stop mechanism, GitHub or LLM plumbing.",
        "- If the evidence is too thin to justify a change, close this issue with a comment.",
    ]
    return "self-evolution: refine agent prompts from failure traces", "\n".join(lines)
