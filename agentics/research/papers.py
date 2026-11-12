"""Paper search via Perplexity academic mode + dedup through vector memory."""
from __future__ import annotations

import json
import logging

from agentics.llm.perplexity_client import LLMResponse, PerplexityClient
from agentics.memory.vector_store import VectorStore

log = logging.getLogger("research.papers")

SEARCH_SYSTEM = (
    "You are a meticulous academic research assistant. Prefer peer-reviewed "
    "sources (arXiv, IEEE, ACM, Springer, journals). Always surface concrete "
    "facts: method names, datasets, metrics, numbers. Never invent a paper."
)


async def search_papers(
    client: PerplexityClient,
    query: str,
    max_papers: int = 8,
    purpose: str | None = "research",
) -> LLMResponse:
    """Ask Perplexity in academic mode; returns answer + citation URLs.

    ``purpose="research"`` routes the call to Perplexity through the
    BrowserSessionHost (citations required); pass ``purpose=None`` when the
    client is a bare PerplexityClient without routing.
    """
    prompt = (
        f"Survey the most relevant recent research on: {query}. "
        f"List up to {max_papers} works. For each: authors/title/year, venue, "
        "key contribution, and results with numbers if available. "
        "Finish with a short synthesis of the state of the art and open gaps."
    )
    kw = {"purpose": purpose} if purpose else {}
    resp = await client.ask(prompt, mode="academic", system=SEARCH_SYSTEM, **kw)
    if not resp.ok:
        log.warning("paper search returned empty answer for: %s", query[:80])
    return resp


# Cosine/L2 distance threshold for the MiniLM embeddings Chroma uses by default:
# pairs below this are considered "same topic" for the P7 duplicate gate.
DEDUP_MAX_DISTANCE = 0.85


async def dedup_query(
    vector: VectorStore, query: str, n: int = 5, max_distance: float = DEDUP_MAX_DISTANCE,
) -> list[dict]:
    """Check whether the swarm already researched this topic (P7).

    Only results within `max_distance` of the query count as duplicates —
    similarity matters, not raw result count.
    """
    if not vector.available:
        return []
    return [r for r in vector.search(query, n=n) if r.get("distance", 1e9) <= max_distance]


async def summarize_for_memory(resp: LLMResponse) -> str:
    """Compact a research answer into a memory-friendly digest."""
    return f"{resp.answer[:1500]}\n\nCitations: {json.dumps(resp.citations[:10])}"
