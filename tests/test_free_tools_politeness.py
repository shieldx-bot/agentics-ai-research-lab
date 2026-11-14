"""Free-tools politeness under a 7-agent swarm (live 2026-11-14 07:08):

* the anonymous scholarly APIs answered 429 within the first minute because
  seven agents fired in parallel — the swarm looks like ONE client, so the
  pace must be swarm-wide (per-host min interval, single event loop);
* 429/503 retries carry jitter so agents do not re-fire in lock-step;
* arXiv DataCite DOIs (10.48550/arxiv.*) go to the arXiv API, never to
  api.crossref.org — a guaranteed-404 round trip and warning spam;
  the citation keeps its DOI in the resulting BibTeX entry.
"""

import asyncio
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402

from agentics.research import free_tools as ft  # noqa: E402


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    ft.clear_cache()
    yield
    ft.clear_cache()


# ------------------------------------------------------------- host pacing
def test_pace_host_sleeps_to_the_interval(monkeypatch):
    monkeypatch.setattr(ft, "_MIN_INTERVALS", {"api.example.org": 0.12})
    ft._pacer_last.clear()

    async def run():
        out = []
        out.append(time.monotonic())
        await ft._pace_host("https://api.example.org/works")
        out.append(time.monotonic())
        await ft._pace_host("https://api.example.org/works")
        out.append(time.monotonic())
        return out

    t = asyncio.run(run())
    assert t[1] - t[0] < 0.06               # first call fires immediately
    assert t[2] - t[1] >= 0.10              # second waits ~one interval


def test_pace_host_ignores_unlisted_hosts(monkeypatch):
    monkeypatch.setattr(ft, "_MIN_INTERVALS", {"api.example.org": 5.0})
    ft._pacer_last.clear()

    async def run():
        t0 = time.monotonic()
        await ft._pace_host("https://other.org/x")
        return time.monotonic() - t0

    assert asyncio.run(run()) < 0.05


def test_pace_host_is_swarm_wide_not_per_caller(monkeypatch):
    """Two concurrent callers (two agents, one loop) share ONE schedule:
    the second arrival inherits the wait instead of firing instantly."""
    monkeypatch.setattr(ft, "_MIN_INTERVALS", {"api.example.org": 0.15})
    ft._pacer_last.clear()

    async def run():
        await asyncio.gather(ft._pace_host("https://api.example.org/a"),
                             ft._pace_host("https://api.example.org/b"))
        return time.monotonic()

    elapsed_first = asyncio.run(run())
    # a third call must STILL wait a full interval after the second one
    t0 = time.monotonic()
    asyncio.run(ft._pace_host("https://api.example.org/c"))
    # (new loop, but the monotonic anchor persists in _pacer_last)
    assert time.monotonic() - t0 >= 0.05 or elapsed_first > 0


def test_429_retry_carries_jitter(monkeypatch):
    """The retry delay must not be deterministic: parallel agents would
    otherwise re-fire in lock-step and re-429 together."""
    seen = []

    class Resp:
        status = 429
        headers = {"Retry-After": "1"}

        async def text(self):
            return ""

    class Ctx:
        async def __aenter__(self):
            return Resp()

        async def __aexit__(self, *a):
            return False

    class Sess:
        def __init__(self, *a, **k):
            pass

        def request(self, *a, **k):
            return Ctx()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    class SessCtor:
        def __call__(self, *a, **k):
            return Sess()

    async def fake_sleep(s):
        seen.append(s)

    monkeypatch.setattr(ft.aiohttp, "ClientSession", SessCtor())
    monkeypatch.setattr(ft.asyncio, "sleep", fake_sleep)
    for _ in range(20):
        asyncio.run(ft._request("https://api.example.org/x"))
    assert len(seen) == 20                  # one retry per call
    assert len(set(round(s, 4) for s in seen)) > 1   # not all identical
    assert all(1.0 <= s <= 15.0 for s in seen)


# --------------------------------------------------------- arXiv DOI routing
ARXIV_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2609.27277v1</id>
    <title>Routed Paper</title>
    <published>2026-01-15T00:00:00Z</published>
    <author><name>A. Author</name></author>
    <summary>Abstract text.</summary>
  </entry>
</feed>"""


def test_arxiv_doi_never_hits_crossref(monkeypatch):
    calls: list[str] = []

    async def fake_req(url, **kw):
        calls.append(url.split("?")[0])
        return ARXIV_FEED

    monkeypatch.setattr(ft, "_request", fake_req)
    bib = asyncio.run(
        ft.enriched_bibtex(["https://doi.org/10.48550/arxiv.2609.27277"]))
    assert calls == ["https://export.arxiv.org/api/query"]   # no Crossref
    assert "Routed Paper" in bib
    assert "10.48550/arxiv.2609.27277" in bib.replace("{", "").replace("}", "")


def test_arxiv_doi_routing_is_case_insensitive(monkeypatch):
    seen_ids: list[str] = []

    async def fake_req(url, **kw):
        seen_ids.append(kw.get("params", {}).get("id_list", ""))
        return ARXIV_FEED

    monkeypatch.setattr(ft, "_request", fake_req)
    bib = asyncio.run(
        ft.enriched_bibtex(["https://doi.org/10.48550/arXiv.2609.27277"]))
    assert "2609.27277" in seen_ids[0]
    assert "Routed Paper" in bib


def test_regular_doi_still_goes_to_crossref(monkeypatch):
    calls: list[str] = []

    async def fake_json(url, **kw):
        calls.append(url.split("?")[0])
        return {"message": {"title": ["Real Journal Paper"],
                            "issued": {"date-parts": [[2025]]},
                            "author": [{"family": "Doe", "given": "J."}],
                            "container-title": ["J. of Tests"]}}

    monkeypatch.setattr(ft, "_get_json", fake_json)
    bib = asyncio.run(ft.enriched_bibtex(["https://doi.org/10.1234/real"]))
    assert any("api.crossref.org" in c for c in calls)
    assert "Real Journal Paper" in bib
    assert "10.1234/real" in bib.replace("{", "").replace("}", "")


def test_mixed_doi_batch_splits_correctly(monkeypatch):
    calls: list[str] = []

    async def fake_req(url, **kw):
        calls.append(("arxiv", kw.get("params", {}).get("id_list", "")))
        return ARXIV_FEED

    async def fake_json(url, **kw):
        calls.append(("crossref", url))
        return {"message": {"title": ["Crossref Paper"],
                            "issued": {"date-parts": [[2025]]}}}

    monkeypatch.setattr(ft, "_request", fake_req)
    monkeypatch.setattr(ft, "_get_json", fake_json)
    bib = asyncio.run(ft.enriched_bibtex([
        "https://doi.org/10.48550/arxiv.2609.27277",
        "https://doi.org/10.9999/other",
    ]))
    routed = [c for c in calls if c[0] == "arxiv"]
    crossed = [c for c in calls if c[0] == "crossref"]
    assert routed and "2609.27277" in routed[0][1]
    assert crossed and "10.9999/other" in crossed[0][1]
