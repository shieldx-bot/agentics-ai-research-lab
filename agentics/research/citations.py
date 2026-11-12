"""Citation formatting — BibTeX & APA from Perplexity citation URLs.

Perplexity returns raw URLs; we derive deterministic identifiers and ask-free
BibTeX entries (no invented metadata: unknown fields are omitted, not faked).
"""
from __future__ import annotations

import re
import urllib.parse
import urllib.request
from dataclasses import dataclass

_ARXIV_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})(v\d+)?", re.I)
_DOI_RE = re.compile(r"doi\.org/(10\.\S+)", re.I)


@dataclass
class Citation:
    url: str
    key: str
    title: str = ""
    venue: str = ""


def _key_from_url(url: str) -> str:
    m = _ARXIV_RE.search(url)
    if m:
        return f"arxiv{m.group(1).replace('.', '')}"
    m = _DOI_RE.search(url)
    if m:
        return "doi" + re.sub(r"[^a-zA-Z0-9]", "", m.group(1))[:20]
    host = urllib.parse.urlparse(url).netloc.replace("www.", "")
    slug = re.sub(r"[^a-zA-Z0-9]", "", urllib.parse.urlparse(url).path)[-24:]
    return f"{re.sub(r'[^a-zA-Z0-9]', '', host)}{slug}"


def parse_citations(urls: list[str]) -> list[Citation]:
    out: list[Citation] = []
    for url in urls:
        c = Citation(url=url, key=_key_from_url(url))
        m = _ARXIV_RE.search(url)
        if m:
            c.venue = "arXiv"
        out.append(c)
    return out


def fetch_title(url: str, timeout: int = 8) -> str:
    """Best-effort <title> extraction; returns '' on any failure."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            html = resp.read(65536).decode("utf-8", errors="replace")
        m = re.search(r"<title[^>]*>(.*?)</title>", html, re.S | re.I)
        if m:
            return re.sub(r"\s+", " ", m.group(1)).strip()
    except Exception:  # noqa: BLE001
        pass
    return ""


def to_bibtex(citations: list[Citation], enrich_titles: bool = True) -> str:
    entries: list[str] = []
    for c in citations:
        if enrich_titles and not c.title:
            c.title = fetch_title(c.url)
        host = urllib.parse.urlparse(c.url).netloc.replace("www.", "")
        year = ""
        m = re.search(r"(20[0-2]\d)", c.url)
        if m:
            year = m.group(1)
        fields = [
            f"  title = {{{c.title or '(untitled)'}}},",
            f"  url = {{{c.url}}},",
            f"  accessed = {{{host}}},",
        ]
        if year:
            fields.append(f"  year = {{{year}}},")
        if c.venue:
            fields.append(f"  howpublished = {{{c.venue}}},")
        entries.append(f"@misc{{{c.key},\n" + "\n".join(fields) + "\n}")
    return "\n\n".join(entries)


def to_apa(citations: list[Citation], enrich_titles: bool = True) -> str:
    lines: list[str] = []
    for i, c in enumerate(citations, 1):
        if enrich_titles and not c.title:
            c.title = fetch_title(c.url)
        host = urllib.parse.urlparse(c.url).netloc.replace("www.", "")
        title = c.title or c.url
        lines.append(f"[{i}] {title}. {host}. {c.url}")
    return "\n".join(lines)
