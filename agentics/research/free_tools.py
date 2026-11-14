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

| Source           | Endpoint                                 | Politeness                     |
| ---------------- | ---------------------------------------- | ------------------------------ |
| arXiv            | export.arxiv.org/api/query               | 3 s between calls, shared lock |
| OpenAlex         | api.openalex.org/works                   | mailto polite-pool parameter   |
| Crossref         | api.crossref.org/works, doi.org          | mailto polite-pool parameter   |
| SemanticScholar  | api.semanticscholar.org/graph/v1/paper   | anonymous pool, 429 retried    |
| EuropePMC        | ebi.ac.uk/europepmc/webservices/rest     | plain GET, generous budget     |
| HuggingFace      | huggingface.co/api/{datasets,models}     | plain GET                      |
| Wikipedia        | en.wikipedia.org/api/rest_v1             | plain GET                      |
| DuckDuckGo       | html.duckduckgo.com/html/                | best-effort, graceful failure  |
| GitHub (repos)   | api.github.com/search/repositories       | 1 call / 6.5 s, shared lock    |
| PyPI             | pypi.org/pypi/<name>/json                | plain GET                      |

Transport is ``aiohttp`` (already a core dependency) with a per-request
timeout; every function degrades gracefully (returns ``[]``/``None``) so a
source outage can never break a research cycle. A small process-wide TTL
cache keeps repeated topics cheap and the rate limits happy.
"""
from __future__ import annotations

import asyncio
import logging
import random
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
# Per-HOST politeness for the shared-anonymous-budget APIs. Live 07:08:
# 7 agents fired S2/OpenAlex/Crossref searches in parallel and all three
# answered 429 within the first minute — the swarm looks like ONE client
# to them, so the pace must be swarm-wide, not per-agent.
_MIN_INTERVALS: dict = {
    "api.semanticscholar.org": 1.0,   # anonymous pool, 1 rps-ish
    "api.openalex.org": 1.0,          # polite pool / datacenter IPs
    "api.crossref.org": 1.0,          # + mailto when provided
}
_pacer_last: dict = {}

_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV_ID_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})(v\d+)?", re.I)
_DOI_RE = re.compile(r"doi\.org/(10\.\S+?)(?:\s|$|[)\]}>,\"])", re.I)
# arXiv's own DataCite DOI prefix: api.crossref.org will NEVER hold these
# (10.48550 is registered with DataCite) — route them to the arXiv API.
_ARXIV_DOI_RE = re.compile(r"^10\.48550/arxiv\.([0-9]{4}\.[0-9]{4,5})(v\d+)?$",
                           re.I)

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
    """One polite HTTP call; returns body text or None on any failure.

    A 429/503 (shared-IP budget exhaustion — common for OpenAlex on
    datacenter hosts) is retried ONCE after the server's Retry-After delay
    (bounded to 10 s, jittered so parallel agents do not re-fire in
    lock-step); everything else fails fast to the graceful path. Calls to
    hosts in ``_MIN_INTERVALS`` are swarm-wide paced before firing."""
    await _pace_host(url)
    timeout = aiohttp.ClientTimeout(total=_timeout(settings))
    hdrs = {"User-Agent": _USER_AGENT, **(headers or {})}
    for attempt in (0, 1):
        retry_delay: float | None = None
        try:
            async with aiohttp.ClientSession(timeout=timeout, headers=hdrs) as sess:
                async with sess.request(method, url, params=params, data=data) as resp:
                    if resp.status == 200:
                        return await resp.text()
                    if resp.status in (429, 503) and attempt == 0:
                        retry_delay = min(_retry_after(resp) or 2.0, 10.0)
                        retry_delay *= random.uniform(1.0, 1.5)
                    else:
                        log.warning("free_tools %s %s -> HTTP %s",
                                    method, url.split("?")[0], resp.status)
                        return None
        except Exception as exc:  # noqa: BLE001 — a source outage must never break a cycle
            log.warning("free_tools %s failed: %s", url.split("?")[0], exc)
            return None
        if retry_delay is not None:      # sleep OUTSIDE the session context
            log.warning("free_tools %s -> HTTP 429/503, retrying in %.0fs",
                        url.split("?")[0], retry_delay)
            await asyncio.sleep(retry_delay)
    return None


async def _pace_host(url: str) -> None:
    """Swarm-wide politeness: at most one call per _MIN_INTERVALS window
    per host (single event loop, all agents included). Hosts not listed
    are unpaced."""
    host = urllib.parse.urlsplit(url).netloc.lower()
    interval = _MIN_INTERVALS.get(host, 0.0)
    if interval <= 0:
        return
    wait = interval - (time.monotonic() - _pacer_last.get(host, 0.0))
    if wait > 0:
        await asyncio.sleep(wait)
    _pacer_last[host] = time.monotonic()


def _retry_after(resp: "aiohttp.ClientResponse") -> float | None:
    raw = resp.headers.get("Retry-After")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
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


# ---------------------------------------------------------------- Semantic Scholar
async def semantic_scholar_search(query: str, max_results: int = 8, settings=None) -> list[PaperHit]:
    """Search Semantic Scholar's public graph — real citation counts, TLDR.

    OpenAlex is documented to 429 on shared datacenter IPs (FREE_TOOLS.md
    §5); Semantic Scholar runs a separate anonymous pool, so it fills the
    same scholarly role from an independent budget. DOI/arXiv ids from
    ``externalIds`` keep the citation URLs resolvable for BibTeX grounding.
    """
    if not _enabled(settings) or not query.strip():
        return []
    key = ("s2", query, max_results)
    cached = _cache_get(key)
    if cached is not None:
        return cached
    data = await _get_json(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        params={
            "query": query,
            "limit": max(1, min(int(max_results), 20)),
            "fields": "title,abstract,year,authors,url,venue,citationCount,externalIds",
        },
        settings=settings,
    )
    hits: list[PaperHit] = []
    for p in ((data or {}).get("data") or [])[: max(1, int(max_results))]:
        title = " ".join((p.get("title") or "").split())
        if not title:
            continue
        ext = p.get("externalIds") or {}
        doi = (ext.get("DOI") or "").strip()
        arx = (ext.get("ArXiv") or "").strip()
        url = f"https://doi.org/{doi}" if doi else (
            f"https://arxiv.org/abs/{arx}" if arx else (p.get("url") or ""))
        hits.append(PaperHit(
            source="semanticscholar",
            title=title,
            url=url,
            year=p.get("year"),
            authors=[(a.get("name") or "") for a in (p.get("authors") or [])],
            abstract=" ".join((p.get("abstract") or "").split())[:1200],
            venue=(p.get("venue") or "").strip() or "Semantic Scholar",
            doi=doi,
            cited_by=p.get("citationCount"),
        ))
    _cache_put(key, hits)
    return hits


# ---------------------------------------------------------------- Europe PMC
async def europepmc_search(query: str, max_results: int = 8, settings=None) -> list[PaperHit]:
    """Search Europe PMC — full abstracts + preprints from a third budget.

    Covers biomed/CS/math literature OpenAlex and Crossref index thinly;
    ``authorString`` is real metadata, never invented. Second-tier filler
    in ``search_all`` so a finding still exists when the other pools are
    exhausted.
    """
    if not _enabled(settings) or not query.strip():
        return []
    key = ("europepmc", query, max_results)
    cached = _cache_get(key)
    if cached is not None:
        return cached
    data = await _get_json(
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        params={
            "query": query,
            "format": "json",
            "pageSize": max(1, min(int(max_results), 25)),
        },
        settings=settings,
    )
    hits: list[PaperHit] = []
    results = (((data or {}).get("resultList") or {}).get("result") or [])[: max(1, int(max_results))]
    for r in results:
        title = " ".join((r.get("title") or "").split())
        if not title:
            continue
        doi = (r.get("doi") or "").strip()
        src = (r.get("source") or "").strip()
        aid = (r.get("id") or "").strip()
        url = f"https://doi.org/{doi}" if doi else (
            f"https://europepmc.org/article/{src}/{aid}" if src and aid else "")
        pub_year = (r.get("pubYear") or "").strip()
        hits.append(PaperHit(
            source="europepmc",
            title=title,
            url=url,
            year=int(pub_year) if pub_year.isdigit() else None,
            authors=[a.strip() for a in (r.get("authorString") or "").split(",") if a.strip()][:12],
            abstract=" ".join(re.sub(r"<[^>]+>", " ", r.get("abstractText") or "").split())[:1200],
            venue=(r.get("journalTitle") or "").strip() or "Europe PMC",
            doi=doi,
            cited_by=r.get("citedByCount"),
        ))
    _cache_put(key, hits)
    return hits


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

    arXiv URL -> arXiv API (authors/title/year/eprint) — all ids batched
    into ONE ``id_list`` call (the polite way: 1 request, not N×3 s);
    DOI URL   -> Crossref (authors/year/journal/DOI), fetched in parallel;
    anything else -> the legacy behaviour (deterministic key + <title>
    scrape) executed in a thread so the event loop never blocks. Entries
    keep the citation order. Unknown metadata is omitted, never invented —
    the same rule the original citations module set.
    """
    cites = cite_fmt.parse_citations(list(urls))
    if not cites:
        return ""
    selected = cites[: max(0, int(max_lookups))]
    arxiv_ids: dict[str, str] = {}      # arXiv id -> citation key
    dois: dict[str, str] = {}           # doi -> citation key
    legacy_keys: list[str] = []
    for c in selected:
        aid, doi = arxiv_id_from_url(c.url), doi_from_url(c.url)
        if aid and not doi:
            arxiv_ids.setdefault(aid, c.key)
        elif doi:
            m = _ARXIV_DOI_RE.match(doi)
            if m:
                # 10.48550/arxiv.NNNN.NNNNN -> the arXiv batch below; asking
                # Crossref for it is a guaranteed 404 (DataCite, not Crossref)
                arxiv_ids.setdefault(m.group(1) + (m.group(2) or ""), c.key)
            else:
                dois.setdefault(doi, c.key)
        else:
            legacy_keys.append(c.key)

    results: dict[str, str] = {}
    _legacy_urls: dict[str, str] = {}

    def _to_legacy(key: str, url: str) -> None:
        legacy_keys.append(key)  # keep deterministic order; resolved at the end
        if url:                   # remember the original URL shape for the key
            _legacy_urls[key] = url

    if arxiv_ids:
        await _arxiv_pace()
        body = await _request(
            "https://export.arxiv.org/api/query",
            params={"id_list": ",".join(arxiv_ids), "max_results": len(arxiv_ids)},
            settings=settings,
        )
        parsed = _parse_arxiv_feed(body) if body else []
        by_id = {h.url.rsplit("/abs/", 1)[-1]: h for h in parsed}
        for aid, key in arxiv_ids.items():
            hit = by_id.get(aid) or by_id.get(aid.split("v")[0])
            if hit is None:  # id_list echoes versions; match base id too
                hit = next((h for k, h in by_id.items() if k.split("v")[0] == aid.split("v")[0]), None)
            if hit:
                if not hit.doi:
                    # the citation URL carried the DataCite DOI — keep it
                    hit.doi = f"10.48550/arxiv.{aid.split('v')[0]}"
                results[key] = _bibtex_entry(key, hit, eprint=aid)
            else:
                _to_legacy(key, f"https://arxiv.org/abs/{aid}")
    if dois:
        fetches = await asyncio.gather(
            *[_get_json(f"https://api.crossref.org/works/{doi}", settings=settings)
              for doi in dois]
        )
        for doi, data in zip(dois, fetches):
            msg = ((data or {}).get("message") or {}) if isinstance(data, dict) else {}
            if not msg.get("title"):
                _to_legacy(dois[doi], f"https://doi.org/{doi}")
                continue
            year = None
            issued = (msg.get("issued") or {}).get("date-parts") or [[None]]
            if issued and issued[0] and issued[0][0]:
                year = issued[0][0]
            hit = PaperHit(
                source="crossref",
                title=" ".join(msg["title"][0].split()),
                url=f"https://doi.org/{doi}",
                year=year,
                authors=[f"{a.get('family', '')}, {a.get('given', '')}".strip(", ")
                         for a in (msg.get("author") or [])],
                venue=" ".join((msg.get("container-title") or [""])[0].split()) or "DOI",
                doi=doi,
            )
            results[dois[doi]] = _bibtex_entry(dois[doi], hit)
    if legacy_keys:
        # preserve the original URL per key (parse_citations re-derives keys)
        legacy_cites = []
        for key in legacy_keys:
            url = _legacy_urls.get(key)
            if url:
                legacy_cites.append(cite_fmt.Citation(url=url, key=key))
            else:  # plain URL passed in without arXiv/DOI — fall back to input
                legacy_cites.append(next(c for c in selected if c.key == key))
        rest = await asyncio.to_thread(cite_fmt.to_bibtex, legacy_cites, True)
        if rest:
            chunks = rest.split("\n\n")
            for c, chunk in zip(legacy_cites, chunks):
                results[c.key] = chunk
    return "\n\n".join(results[c.key] for c in selected if c.key in results)


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


# ---------------------------------------------------------------- verification toolkit
# Anonymous GitHub search allows ~10 calls/min process-wide: pace it like arXiv.
_gh_lock = asyncio.Lock()
_gh_last: float = 0.0
_GH_MIN_INTERVAL_S = 6.5


async def _github_pace() -> None:
    """Process-wide politeness: at most one GitHub search every ~6.5 s."""
    global _gh_last
    async with _gh_lock:
        wait = _GH_MIN_INTERVAL_S - (time.monotonic() - _gh_last)
        if wait > 0:
            await asyncio.sleep(wait)
        _gh_last = time.monotonic()


async def github_repo_search(query: str, max_results: int = 5, settings=None) -> list[dict]:
    """Search public GitHub repos — 'does this idea already have code?'

    Directly serves hypothesis verification (H1/H2): a reference
    implementation an agent can clone into the workspace sandbox is the
    fastest path from idea to a runnable protocol. Keyless (10 req/min
    anonymous budget), real stargazer/language/license metadata.
    """
    if not _enabled(settings) or not query.strip():
        return []
    key = ("github", query, max_results)
    cached = _cache_get(key)
    if cached is not None:
        return cached
    await _github_pace()
    data = await _get_json(
        "https://api.github.com/search/repositories",
        params={"q": query, "per_page": max(1, min(int(max_results), 20))},
        settings=settings,
    )
    out: list[dict] = []
    for it in ((data or {}).get("items") or [])[: max(1, int(max_results))]:
        full = (it.get("full_name") or "").strip()
        if not full:
            continue
        out.append({
            "repo": full,
            "url": it.get("html_url") or f"https://github.com/{full}",
            "description": " ".join((it.get("description") or "").split())[:300],
            "stars": it.get("stargazers_count"),
            "language": it.get("language") or "",
            "license": ((it.get("license") or {}) or {}).get("spdx_id") or "",
            "pushed_at": (it.get("pushed_at") or "")[:10],
        })
    _cache_put(key, out)
    return out


async def pypi_package_info(package: str, settings=None) -> dict | None:
    """Real PyPI metadata for one package — exists? latest version? python?

    Verification protocols must pin REAL dependency versions (P16
    reproducibility): this checks a package actually exists before an
    agent writes ``pip install ...`` into a ``verify_cmds`` protocol, and
    returns the latest published version to pin. Unknown/typo names
    return None — never an invented version.
    """
    name = urllib.parse.quote((package or "").strip().strip("/"))
    if not _enabled(settings) or not name:
        return None
    key = ("pypi", name)
    cached = _cache_get(key)
    if cached is not None:
        return cached or None
    data = await _get_json(f"https://pypi.org/pypi/{name}/json", settings=settings)
    out = None
    if isinstance(data, dict) and (data.get("info") or {}).get("name"):
        info = data["info"]
        out = {
            "name": info.get("name", ""),
            "version": info.get("version", ""),
            "summary": " ".join((info.get("summary") or "").split())[:300],
            "requires_python": info.get("requires_python") or "",
            "home_page": info.get("home_page")
                         or ((info.get("project_urls") or {}).get("Homepage") or ""),
            "license": (info.get("license") or "")[:120],
        }
    _cache_put(key, out or ())
    return out


async def hf_models_search(query: str, max_results: int = 5, settings=None) -> list[dict]:
    """Find public models on Hugging Face — real pretrained weights for a
    verification protocol (complements ``hf_datasets_search``)."""
    if not _enabled(settings) or not query.strip():
        return []
    key = ("hf-models", query, max_results)
    cached = _cache_get(key)
    if cached is not None:
        return cached
    data = await _get_json(
        "https://huggingface.co/api/models",
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
            "pipeline_tag": d.get("pipeline_tag") or "",
            "last_modified": d.get("lastModified", ""),
            "url": f"https://huggingface.co/{d.get('id', '')}",
        })
    _cache_put(key, out)
    return out


async def verification_toolkit(query: str, settings=None) -> dict | None:
    """Repos + datasets + models that make a hypothesis checkable NOW.

    Bundles the three tool-discovery sources into one cached call so a
    finding ships with runnable code paths and real data instead of only
    a literature list. Every element degrades independently; the whole
    bundle returns None when nothing is found.
    """
    if not _enabled(settings) or not query.strip():
        return None
    key = ("toolkit", query)
    cached = _cache_get(key)
    if cached is not None:
        return cached or None
    repos, datasets, models = await asyncio.gather(
        github_repo_search(query, 3, settings=settings),
        hf_datasets_search(query, 3, settings=settings),
        hf_models_search(query, 3, settings=settings),
    )
    out = None
    if repos or datasets or models:
        out = {"repos": repos, "datasets": datasets, "models": models}
    _cache_put(key, out or ())
    return out


def _toolkit_text(toolkit: dict) -> str:
    lines = ["Reference implementations & data (real metadata — clone/download "
             "only after checking the license and availability):"]
    for r in toolkit.get("repos") or []:
        stars = f", stars {r['stars']}" if r.get("stars") is not None else ""
        lang = r.get("language") or "unknown lang"
        lic = f", {r['license']}" if r.get("license") else ""
        line = f"- [repo] {r['repo']} ({lang}{stars}{lic}) — {r['url']}"
        if r.get("description"):
            line += f": {r['description']}"
        lines.append(line)
    for d in toolkit.get("datasets") or []:
        downloads = d.get("downloads") if d.get("downloads") is not None else 0
        lines.append(f"- [dataset] {d['id']} ({downloads} downloads) — {d['url']}")
    for m in toolkit.get("models") or []:
        downloads = m.get("downloads") if m.get("downloads") is not None else 0
        tag = f", {m['pipeline_tag']}" if m.get("pipeline_tag") else ""
        lines.append(f"- [model] {m['id']} ({downloads} downloads{tag}) — {m['url']}")
    return "\n".join(lines)


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
    sources = ", ".join(sorted({h.source for h in hits})) or "official APIs"
    lines = [
        f"Ground-truth literature on '{query}' fetched from keyless official APIs "
        f"({sources}) — metadata below is real and verifiable, "
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
    """arXiv + OpenAlex + Semantic Scholar in parallel; Crossref and Europe
    PMC only when the first three came up short (independent rate budgets,
    so one exhausted pool never empties a finding)."""
    arxiv, openalex, s2 = await asyncio.gather(
        arxiv_search(query, max_results, settings),
        openalex_search(query, max_results, settings),
        semantic_scholar_search(query, max_results, settings),
    )
    merged = _dedup(arxiv + openalex + s2)
    if len(merged) < 3:
        crossref = await crossref_works(query, max_results, settings)
        merged = _dedup(merged + crossref)
    if len(merged) < 3:
        epmc = await europepmc_search(query, max_results, settings)
        merged = _dedup(merged + epmc)
    return merged[:max_results]


async def fallback_research(query: str, settings=None, max_results: int = 8) -> FreeResearch:
    """Real-metadata research result when the LLM pools are unavailable.

    Shape-compatible with ``LLMResponse`` (``.answer``/``.citations``/``.ok``)
    so ``node_research`` can consume it unchanged. Returns ``ok=False`` when
    every source came up empty — callers keep their existing error handling.
    The literature list is enriched with the verification toolkit (repos,
    datasets, models): the finding tells downstream plan/verify nodes what
    code and data actually exist, so a hypothesis can go straight to a
    runnable protocol.
    """
    hits, toolkit = await asyncio.gather(
        search_all(query, max_results=max_results, settings=settings),
        verification_toolkit(query, settings=settings),
    )
    if not hits:
        return FreeResearch(answer="", citations=[])
    citations = [h.url for h in hits if h.url]
    bibtex = await enriched_bibtex(citations, settings=settings)
    answer = _answer_text(query, hits)
    if toolkit:
        answer += "\n\n" + _toolkit_text(toolkit)
    return FreeResearch(answer=answer, citations=citations, bibtex=bibtex)
