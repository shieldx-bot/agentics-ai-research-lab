"""Free, keyless research tools — official APIs that return REAL metadata.

Philosophy (same as P16): a finding must rest on real, checkable evidence —
not on an LLM's word. Every source here is the publisher's own official API,
so the metadata (authors, title, year, abstract, citation counts) is fetched
ground truth, never generated. They complement the cookie-based LLM pools:

* ``fallback_research`` lets ``node_research`` still produce a real,
  citation-backed finding when every LLM cookie is dead — the cycle keeps
  running instead of degrading to an error state.
* ``enriched_bibtex`` replaces the scrape-the-<title>-tag BibTeX with real
  publisher metadata (authors, year, venue, eprint/DOI) for arXiv and DOI
  citation URLs; unknown URLs keep the legacy behaviour.

Sources (all keyless, free; no account, no token):

| Source      | Endpoint                          | Politeness                     |
| ----------- | --------------------------------- | ------------------------------ |
| arXiv       | export.arxiv.org/api/query        | 3 s between calls, shared lock |
| OpenAlex    | api.openalex.org/works            | mailto polite-pool parameter   |
| Crossref    | api.crossref.org/works, doi.org   | mailto polite-pool parameter   |
| HuggingFace | huggingface.co/api/datasets       | plain GET                      |
| Wikipedia   | en.wikipedia.org/api/rest_v1      | plain GET                      |
| DuckDuckGo  | html.duckduckgo.com/html/         | best-effort, graceful failure  |

Transport is ``aiohttp`` (already a core dependency) with a per-request
timeout; every function degrades gracefully (returns ``[]``/``None``) so a
source outage can never break a research cycle. A small process-wide TTL
cache keeps repeated topics cheap and the rate limits happy.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict

import aiohttp

from agentics.research import citations as cite_fmt

log = logging.getLogger("research.free_tools")

# ---------------------------------------------------------------- settings
_TIMEOUT_S = 15.0
_ARXIV_MIN_INTERVAL_S = 3.0          # arXiv asks for 1 call / 3 s
_CACHE_TTL_S = 600.0                 # process-wide result cache
_MAX_LOOKUPS = 8                     # per-call enrichment cap

_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV_ID_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})(v\d+)?", re.I)
_DOI_RE = re.compile(r"doi\.org/(10\.\S+?)(?:\s|$|[)\]}>,\"])", re.I)

_USER_AGENT = "agentics-ai-research-lab (+https://github.com/shieldx-bot/agentics-ai-research-lab)"


def _mailto(settings=None) -> str:
    """Polite-pool contact (OpenAlex/Crossref): FREE_TOOLS_MAILTO or none."""
    return (getattr(settings, "free_tools_mailto", "") or "").strip()


def _timeout(settings=None) -> float:
    return float(getattr(settings, "free_tools_timeout", _TIMEOUT_S) or _TIMEOUT_S)


def _enabled(settings=None) -> bool:
    return bool(getattr(settings, "free_tools_enabled", True))


# ---------------------------------------------------------------- cache
_cache: dict[tuple, tuple[float, object]] = {}


def clear_cache() -> None:
    """Reset the process-wide result cache (used between tests/runs)."""
    _cache.clear()


def _cache_get(key: tuple):
    hit = _cache.get(key)
    if hit and (time.monotonic() - hit[0]) < _CACHE_TTL_S:
        return hit[1]
    if hit:
        _cache.pop(key, None)
    return None


def _cache_put(key: tuple, value: object) -> None:
    if len(_cache) > 256:                      # bounded: drop the oldest
        oldest = min(_cache.items(), key=lambda kv: kv[1][0])
        _cache.pop(oldest[0], None)
    _cache[key] = (time.monotonic(), value)


# ---------------------------------------------------------------- transport
_arxiv_lock = asyncio.Lock()
_arxiv_last: float = 0.0


async def _request(url: str, *, params: dict | None = None, headers: dict | None = None,
                   method: str = "GET", data: dict | None = None,
                   settings=None) -> str | None:
    """One polite HTTP call; returns body text or None on any failure."""
    timeout = aiohttp.ClientTimeout(total=_timeout(settings))
    hdrs = {"User-Agent": _USER_AGENT, **(headers or {})}
    try:
        async with aiohttp.ClientSession(timeout=timeout, headers=hdrs) as sess:
            async with sess.request(method, url, params=params, data=data) as resp:
                if resp.status != 200:
                    log.warning("free_tools %s %s -> HTTP %s", method, url.split("?")[0], resp.status)
                    return None
                return await resp.text()
    except Exception as exc:  # noqa: BLE001 — a source outage must never break a cycle
        log.warning("free_tools %s failed: %s", url.split("?")[0], exc)
        return None


async def _get_json(url: str, *, params: dict | None = None, settings=None) -> dict | list | None:
    import json
    body = await _request(url, params=params, settings=settings)
    if body is None:
        return None
    try:
        return json.loads(body)
    except ValueError:
        log.warning("free_tools %s returned non-JSON body", url.split("?")[0])
        return None


async def _arxiv_pace() -> None:
    """Process-wide politeness: at most one arXiv call every 3 seconds."""
    global _arxiv_last
    async with _arxiv_lock:
        wait = _ARXIV_MIN_INTERVAL_S - (time.monotonic() - _arxiv_last)
        if wait > 0:
            await asyncio.sleep(wait)
        _arxiv_last = time.monotonic()


# ---------------------------------------------------------------- data model
@dataclass
class PaperHit:
    """One real paper, exactly as an official API reported it."""
    source: str                     # arxiv | openalex | crossref
    title: str
    url: str
    year: int | None = None
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    venue: str = ""
    doi: str = ""
    cited_by: int | None = None

    def authors_short(self) -> str:
        if not self.authors:
            return "(authors not reported)"
        return f"{self.authors[0]} et al." if len(self.authors) > 1 else self.authors[0]

    def as_row(self) -> dict:
        return asdict(self)


@dataclass
class FreeResearch:
    """LLMResponse-compatible research result built from real metadata."""
    answer: str
    citations: list[str] = field(default_factory=list)
    bibtex: str = ""
    source: str = "free-tools"

    @property
    def ok(self) -> bool:
        return bool(self.answer.strip())


# ---------------------------------------------------------------- arXiv
def _arxiv_query_variants(query: str) -> list[str]:
    """Phrase query first, then space-joined terms (arXiv treats ' ' as AND)."""
    q = " ".join(query.split())
    variants = [f'all:"{q}"']
    words = [w for w in re.split(r"\W+", q) if len(w) > 2][:8]
    if len(words) > 1:
        variants.append(" AND ".join(f"all:{w}" for w in words))
    return variants


def _parse_arxiv_feed(xml_text: str) -> list[PaperHit]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        log.warning("arXiv feed parse failed: %s", exc)
        return []
    hits: list[PaperHit] = []
    for entry in root.findall(f"{_ATOM}entry"):
        title = " ".join((entry.findtext(f"{_ATOM}title") or "").split())
        if not title:
            continue
        eid = (entry.findtext(f"{_ATOM}id") or "").strip()
        url = f"https://arxiv.org/abs/{eid.rsplit('/abs/', 1)[-1]}" if eid else eid
        published = (entry.findtext(f"{_ATOM}published") or "")[:4]
        hit = PaperHit(
            source="arxiv",
            title=title,
            url=url,
            year=int(published) if published.isdigit() else None,
            authors=[a.findtext(f"{_ATOM}name") or "" for a in entry.findall(f"{_ATOM}author")],
            abstract=" ".join((entry.findtext(f"{_ATOM}summary") or "").split())[:1200],
            venue="arXiv",
            doi=(entry.findtext("{http://arxiv.org/schemas/atom}doi") or "").strip(),
        )
        cat = entry.find("{http://arxiv.org/schemas/atom}primary_category")
        if cat is not None:
            hit.venue = f"arXiv [{cat.get('term', '')}]"
        hits.append(hit)
    return hits


async def arxiv_search(query: str, max_results: int = 8, settings=None) -> list[PaperHit]:
    """Search arXiv via its official Atom API — real titles/authors/abstracts."""
    if not _enabled(settings) or not query.strip():
        return []
    key = ("arxiv", query, max_results)
    cached = _cache_get(key)
    if cached is not None:
        return cached
    base = "https://export.arxiv.org/api/query"
    for variant in _arxiv_query_variants(query):
        await _arxiv_pace()
        body = await _request(
            base,
            params={
                "search_query": variant,
                "start": 0,
                "max_results": max(1, min(int(max_results), 20)),
                "sortBy": "relevance",
            },
            settings=settings,
        )
        hits = _parse_arxiv_feed(body) if body else []
        if hits:
            _cache_put(key, hits)
            return hits[:max_results]
    _cache_put(key, [])
    return []


def arxiv_id_from_url(url: str) -> str:
    m = _ARXIV_ID_RE.search(url)
    return f"{m.group(1)}{m.group(2) or ''}" if m else ""


def doi_from_url(url: str) -> str:
    m = _DOI_RE.search(url)
    return m.group(1).rstrip(".,;") if m else ""


# ---------------------------------------------------------------- OpenAlex
def _abstract_from_inverted(inv: dict | None) -> str:
    """OpenAlex ships abstracts as {word: [positions]} — rebuild the text."""
    if not inv:
        return ""
    words: list[str] = []
    for word, positions in inv.items():
        for pos in positions:
            words.append((pos, word))
    return " ".join(w for _, w in sorted(words))[:1200]


async def openalex_search(query: str, max_results: int = 8, settings=None) -> list[PaperHit]:
    """Search the OpenAlex scholarly graph — DOIs, venues, citation counts."""
    if not _enabled(settings) or not query.strip():
        return []
    key = ("openalex", query, max_results)
    cached = _cache_get(key)
    if cached is not None:
        return cached
    params: dict = {"search": query, "per-page": max(1, min(int(max_results), 25))}
    mail = _mailto(settings)
    if mail:
        params["mailto"] = mail
    data = await _get_json("https://api.openalex.org/works", params=params, settings=settings)
    hits: list[PaperHit] = []
    for w in (data or {}).get("results", [])[:max_results]:
        title = (w.get("display_name") or "").strip()
        if not title:
            continue
        doi = (w.get("doi") or "").replace("https://doi.org/", "")
        src = ((w.get("primary_location") or {}).get("source") or {}).get("display_name") or ""
        hits.append(PaperHit(
            source="openalex",
            title=title,
            url=(w.get("doi") or w.get("id") or ""),
            year=w.get("publication_year"),
            authors=[a.get("author", {}).get("display_name", "")
                     for a in (w.get("authorships") or []) if a.get("author")],
            abstract=_abstract_from_inverted(w.get("abstract_inverted_index")),
            venue=src or "OpenAlex",
            doi=doi,
            cited_by=w.get("cited_by_count"),
        ))
    _cache_put(key, hits)
    return hits


# ---------------------------------------------------------------- Crossref
async def crossref_works(query: str, max_results: int = 8, settings=None) -> list[PaperHit]:
    """Search Crossref — publisher-deposited DOI metadata."""
    if not _enabled(settings) or not query.strip():
        return []
    key = ("crossref", query, max_results)
    cached = _cache_get(key)
    if cached is not None:
        return cached
    params: dict = {
        "query": query,
        "rows": max(1, min(int(max_results), 20)),
        "select": "DOI,title,author,issued,container-title,is-referenced-by-count,URL,abstract",
    }
    mail = _mailto(settings)
    if mail:
        params["mailto"] = mail
    data = await _get_json("https://api.crossref.org/works", params=params, settings=settings)
    hits: list[PaperHit] = []
    for w in ((data or {}).get("message") or {}).get("items", [])[:max_results]:
        title = " ".join((w.get("title") or [""])[0].split())
        if not title:
            continue
        year = None
        issued = (w.get("issued") or {}).get("date-parts") or [[None]]
        if issued and issued[0] and issued[0][0]:
            year = issued[0][0]
        hits.append(PaperHit(
            source="crossref",
            title=title,
            url=f"https://doi.org/{w.get('DOI', '')}" if w.get("DOI") else (w.get("URL") or ""),
            year=year,
            authors=[f"{a.get('family', '')}, {a.get('given', '')}".strip(", ")
                     for a in (w.get("author") or [])],
            abstract=" ".join(re.sub(r"<[^>]+>", " ", w.get("abstract") or "").split())[:1200],
            venue=" ".join((w.get("container-title") or [""])[0].split()) or "Crossref",
            doi=(w.get("DOI") or "").strip(),
            cited_by=w.get("is-referenced-by-count"),
        ))
    _cache_put(key, hits)
    return hits


async def crossref_bibtex(doi: str, settings=None) -> str:
    """DOI -> BibTeX via Crossref content negotiation (real metadata)."""
    doi = (doi or "").strip()
    if not _enabled(settings) or not doi:
        return ""
    body = await _request(
        f"https://doi.org/{doi}",
        headers={"Accept": "application/x-bibtex"},
        settings=settings,
    )
    return (body or "").strip()


# ---------------------------------------------------------------- BibTeX enrichment
def _bibtex_entry(key: str, hit: PaperHit, eprint: str = "") -> str:
    fields = [f"  title = {{{hit.title}}},"]
    if hit.authors:
        fields.append("  author = {" + " and ".join(hit.authors[:10]) + "},")
    if hit.year:
        fields.append(f"  year = {{{hit.year}}},")
    if eprint:
        fields.append(f"  eprint = {{{eprint}}},\n  archivePrefix = {{arXiv}},")
    if hit.doi:
        fields.append(f"  doi = {{{hit.doi}}},")
    if hit.venue:
        fields.append(f"  howpublished = {{{hit.venue}}},")
    fields.append(f"  url = {{{hit.url}}},")
    return f"@misc{{{key},\n" + "\n".join(fields) + "\n}"


async def enriched_bibtex(urls: list[str], settings=None, max_lookups: int = _MAX_LOOKUPS) -> str:
    """BibTeX from REAL publisher metadata for arXiv/DOI citation URLs.

    arXiv URL -> arXiv API (authors/title/year/eprint);
    DOI URL   -> Crossref (authors/year/journal/DOI);
    anything else -> the legacy behaviour (deterministic key + <title> scrape)
    executed in a thread so the event loop never blocks. Unknown metadata is
    omitted, never invented — the same rule the original citations module set.
    """
    cites = cite_fmt.parse_citations(list(urls))
    if not cites:
        return ""
    entries: list[str] = []
    legacy: list[cite_fmt.Citation] = []
    for c in cites[:max(0, int(max_lookups))]:
        hit: PaperHit | None = None
        eprint = ""
        arxiv_id = arxiv_id_from_url(c.url)
        doi = doi_from_url(c.url)
        if arxiv_id:
            await _arxiv_pace()
            body = await _request(
                "https://export.arxiv.org/api/query",
                params={"id_list": arxiv_id, "max_results": 1},
                settings=settings,
            )
            parsed = _parse_arxiv_feed(body) if body else []
            if parsed:
                hit = parsed[0]
                eprint = arxiv_id
        elif doi:
            data = await _get_json(f"https://api.crossref.org/works/{doi}", settings=settings)
            msg = (data or {}).get("message") or {}
            if msg.get("title"):
                year = None
                issued = (msg.get("issued") or {}).get("date-parts") or [[None]]
                if issued and issued[0] and issued[0][0]:
                    year = issued[0][0]
                hit = PaperHit(
                    source="crossref",
                    title=" ".join(msg["title"][0].split()),
                    url=c.url,
                    year=year,
                    authors=[f"{a.get('family', '')}, {a.get('given', '')}".strip(", ")
                             for a in (msg.get("author") or [])],
                    venue=" ".join((msg.get("container-title") or [""])[0].split()) or "DOI",
                    doi=doi,
                )
        if hit:
            entries.append(_bibtex_entry(c.key, hit, eprint=eprint))
        else:
            legacy.append(c)
    if legacy:
        rest = await asyncio.to_thread(cite_fmt.to_bibtex, legacy, True)
        if rest:
            entries.append(rest)
    return "\n\n".join(entries)


# ---------------------------------------------------------------- general tools
async def hf_datasets_search(query: str, max_results: int = 5, settings=None) -> list[dict]:
    """Find public datasets on Hugging Face — useful when a verification
    protocol needs real data instead of synthetic numbers."""
    if not _enabled(settings) or not query.strip():
        return []
    key = ("hf-datasets", query, max_results)
    cached = _cache_get(key)
    if cached is not None:
        return cached
    data = await _get_json(
        "https://huggingface.co/api/datasets",
        params={"search": query, "limit": max(1, min(int(max_results), 20)),
                "sort": "downloads", "direction": -1},
        settings=settings,
    )
    out = []
    for d in (data or [])[:max_results] if isinstance(data, list) else []:
        out.append({
            "id": d.get("id", ""),
            "downloads": d.get("downloads"),
            "likes": d.get("likes"),
            "last_modified": d.get("lastModified", ""),
            "url": f"https://huggingface.co/datasets/{d.get('id', '')}",
        })
    _cache_put(key, out)
    return out


async def wikipedia_summary(term: str, settings=None) -> dict | None:
    """One-paragraph definition of a term — terminology grounding, not a claim source."""
    if not _enabled(settings) or not term.strip():
        return None
    key = ("wikipedia", term)
    cached = _cache_get(key)
    if cached is not None:
        return cached or None
    title = urllib.parse.quote(term.strip().replace(" ", "_"))
    data = await _get_json(
        f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}", settings=settings,
    )
    out = None
    if isinstance(data, dict) and data.get("extract"):
        out = {
            "term": data.get("title", term),
            "extract": data["extract"][:1000],
            "url": (data.get("content_urls") or {}).get("desktop", {}).get("page", ""),
        }
    _cache_put(key, out or ())
    return out


async def ddg_web_search(query: str, max_results: int = 6, settings=None) -> list[dict]:
    """DuckDuckGo HTML results — a keyless general web search, best-effort.

    The markup is unofficial and may change; any failure returns [] (the
    scholarly sources above remain the primary evidence path)."""
    if not _enabled(settings) or not query.strip():
        return []
    key = ("ddg", query, max_results)
    cached = _cache_get(key)
    if cached is not None:
        return cached
    body = await _request(
        "https://html.duckduckgo.com/html/",
        method="POST",
        data={"q": query, "kl": "wt-wt"},
        settings=settings,
    )
    out: list[dict] = []
    if body:
        raw = re.findall(
            r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
            body, re.S,
        )[: max(1, int(max_results))]
        for href, title in raw:
            if "uddg=" in href:
                q = urllib.parse.parse_qs(urllib.parse.urlparse(
                    "https://" + href.lstrip("/")).query)
                href = (q.get("uddg") or [href])[0]
            title = re.sub(r"<[^>]+>", "", title)
            out.append({"title": " ".join(title.split()), "url": href})
        snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', body, re.S)
        for i, s in enumerate(snippets[:len(out)]):
            out[i]["snippet"] = " ".join(re.sub(r"<[^>]+>", "", s).split())[:300]
    _cache_put(key, out)
    return out


# ---------------------------------------------------------------- composition
def _dedup(hits: list[PaperHit]) -> list[PaperHit]:
    seen: set[str] = set()
    out: list[PaperHit] = []
    for h in hits:
        k = (h.doi.lower() or re.sub(r"[^a-z0-9]", "", h.title.lower())[:64])
        if k in seen or not k:
            continue
        seen.add(k)
        out.append(h)
    return out


def _answer_text(query: str, hits: list[PaperHit]) -> str:
    lines = [
        f"Ground-truth literature on '{query}' fetched from keyless official APIs "
        "(arXiv, OpenAlex, Crossref) — metadata below is real and verifiable, "
        "not generated by a language model.",
        "",
    ]
    for i, h in enumerate(hits, 1):
        cited = f", cited by {h.cited_by}" if h.cited_by is not None else ""
        lines.append(f"{i}. {h.title} — {h.authors_short()}"
                     f"{f' ({h.year})' if h.year else ''} [{h.venue}{cited}]")
        if h.abstract:
            lines.append(f"   Abstract: {h.abstract[:400]}")
        lines.append(f"   {h.url}")
    lines += [
        "",
        "Synthesis note: list reflects API relevance ranking; verify claims "
        "against the linked sources before relying on them.",
    ]
    return "\n".join(lines)


async def search_all(query: str, max_results: int = 8, settings=None) -> list[PaperHit]:
    """arXiv + OpenAlex in parallel, Crossref only when they came up short."""
    arxiv, openalex = await asyncio.gather(
        arxiv_search(query, max_results, settings),
        openalex_search(query, max_results, settings),
    )
    merged = _dedup(arxiv + openalex)
    if len(merged) < 3:
        crossref = await crossref_works(query, max_results, settings)
        merged = _dedup(merged + crossref)
    return merged[:max_results]


async def fallback_research(query: str, settings=None, max_results: int = 8) -> FreeResearch:
    """Real-metadata research result when the LLM pools are unavailable.

    Shape-compatible with ``LLMResponse`` (``.answer``/``.citations``/``.ok``)
    so ``node_research`` can consume it unchanged. Returns ``ok=False`` when
    every source came up empty — callers keep their existing error handling.
    """
    hits = await search_all(query, max_results=max_results, settings=settings)
    if not hits:
        return FreeResearch(answer="", citations=[])
    citations = [h.url for h in hits if h.url]
    bibtex = await enriched_bibtex(citations, settings=settings)
    return FreeResearch(answer=_answer_text(query, hits), citations=citations, bibtex=bibtex)
