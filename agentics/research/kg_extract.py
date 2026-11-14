"""GraphRAG-lite (#4): knowledge-graph ingestion + path-based context.

Findings/settles are mined into (entity -rel-> entity) triples stored in
`kg_entities`/`kg_edges` (SharedMemory). Research/ideation prompts then get
a PATH block over the graph instead of only flat findings — better context
per LLM call, which matters when every call is budget-capped (P20).

Extraction runs on the LOCAL router (purpose `kg_extract`) when configured:
zero cookie spend. Without a local runtime the lab simply has no graph —
an honest degradation, never an invented triple. Every function degrades
gracefully and never raises: the KG is context sugar, not a pipeline stage.
"""
from __future__ import annotations

import json
import logging
import re

log = logging.getLogger("research.kg")

_EXTRACT_PROMPT = """Extract research knowledge triples from the text below.
Rules: at most 8 triples; entity names lowercase, short (1-4 words);
relation is one of: improves | uses | measures | outperforms | causes | limits.
Reply with ONLY a JSON array like:
[{{"src": "lstm", "rel": "outperforms", "dst": "arima", "kind": "method"}}]

Text:
{text}
"""


def _parse_triples(answer: str) -> list[dict]:
    """Tolerant JSON-array parse; anything malformed yields []."""
    if not answer:
        return []
    m = re.search(r"\[.*\]", answer, re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except ValueError:
        return []
    out = []
    if not isinstance(data, list):
        return []
    for t in data[:8]:
        if not isinstance(t, dict):
            continue
        src = str(t.get("src") or "").strip().lower()[:80]
        dst = str(t.get("dst") or "").strip().lower()[:80]
        rel = str(t.get("rel") or "related").strip().lower()[:40]
        kind = str(t.get("kind") or "concept").strip().lower()[:40]
        if src and dst and src != dst:
            out.append({"src": src, "rel": rel, "dst": dst, "kind": kind})
    return out


async def ingest_text(memory, text: str, settings,
                      source_hyp: int | None = None,
                      router=None) -> int:
    """Extract triples from ONE text (a finding/answer) via the local
    router and store them. Returns the number of edges added (0 when the
    local runtime is unavailable or nothing parsed). Never raises.
    `router` is injectable for tests; defaults to agentics.llm.local_router."""
    try:
        if not getattr(settings, "local_llm_enabled", False):
            return 0
        if not (text or "").strip():
            return 0
        if router is None:
            from agentics.llm import local_router as router
        if not await router.available(settings):
            return 0
        resp = await router.ask(
            _EXTRACT_PROMPT.format(text=text[:4000]),
            settings=settings, purpose="kg_extract")
        if resp is None or not resp.ok:
            return 0
        triples = _parse_triples(resp.answer)
        added = 0
        for t in triples:
            memory.kg_entity(t["src"], t["kind"])
            memory.kg_entity(t["dst"], t["kind"])
            if memory.kg_edge(t["src"], t["dst"], t["rel"], source_hyp):
                added += 1
        return added
    except Exception as exc:  # noqa: BLE001 — KG sugar must never break research
        log.warning("kg ingest failed: %s", exc)
        return 0


def context_block(memory, topic: str, limit: int = 10) -> str:
    """The prompt block for a research topic, or '' (callers skip it)."""
    try:
        terms = [w for w in re.split(r"\W+", (topic or "").lower())
                 if len(w) > 3][:6]
        if not terms:
            return ""
        block = memory.kg_context(terms, limit=limit)
        if not block:
            return ""
        return ("Knowledge graph paths already known to the lab "
                "(grounded in earlier findings):\n" + block)
    except Exception:  # noqa: BLE001 — context sugar must never break research
        return ""


def mechanism_hint(memory, taxonomy: list[str], limit: int = 5) -> str:
    """MAP-Elites steering: mechanisms with NO archived elite yet."""
    try:
        missing = memory.qd_underexplored(taxonomy, limit=limit)
        if not missing:
            return ""
        return ("Mechanism categories the archive has NO good candidate "
                "for yet (prefer pivoting toward one of these): "
                + ", ".join(missing))
    except Exception:  # noqa: BLE001
        return ""


__all__ = ["ingest_text", "context_block", "mechanism_hint", "_parse_triples"]
