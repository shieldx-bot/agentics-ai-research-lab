"""Free keyless research tools — offline suite (no network in tests).

Covers: arXiv Atom parsing, OpenAlex inverted-index abstracts, Crossref
metadata, the batched BibTeX enrichment (arXiv id_list = ONE call), the
composition layer (search_all dedup + fallback_research), the general
tools (HF datasets / Wikipedia / DuckDuckGo) and the node_research wiring
(fallback on dead LLM pools, grounded BibTeX on the LLM path).
"""

import asyncio
import sys
import time
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402

from agentics.graph import nodes  # noqa: E402
from agentics.llm.perplexity_client import LLMResponse  # noqa: E402
from agentics.memory.shared_memory import SharedMemory  # noqa: E402
from agentics.research import citations as cite_fmt  # noqa: E402
from agentics.research import free_tools as ft  # noqa: E402
from agentics.agents.roles import Role  # noqa: E402


@pytest.fixture(autouse=True)
def _fast_clean(monkeypatch):
    """No result cache bleed between tests, no 3 s arXiv politeness sleep."""
    ft.clear_cache()
    monkeypatch.setattr(ft, "_ARXIV_MIN_INTERVAL_S", 0.0)
    yield
    ft.clear_cache()


ARXIV_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.12345v2</id>
    <title>  ML for NFV
      failure prediction  </title>
    <summary>An abstract about  NFV. </summary>
    <published>2024-01-15T00:00:00Z</published>
    <author><name>Jane Doe</name></author>
    <author><name>John Roe</name></author>
    <arxiv:doi xmlns:arxiv="http://arxiv.org/schemas/atom">10.9999/nfv</arxiv:doi>
    <arxiv:primary_category term="cs.LG" xmlns:arxiv="http://arxiv.org/schemas/atom"/>
  </entry>
</feed>"""


# ---------------------------------------------------------------- arXiv
def test_arxiv_feed_parsing():
    hits = ft._parse_arxiv_feed(ARXIV_FEED)
    assert len(hits) == 1
    h = hits[0]
    assert h.title == "ML for NFV failure prediction"
    assert h.year == 2024
    assert h.authors == ["Jane Doe", "John Roe"]
    assert h.venue == "arXiv [cs.LG]"
    assert h.url == "https://arxiv.org/abs/2401.12345v2"
    assert h.doi == "10.9999/nfv"


def test_arxiv_feed_garbage_is_empty():
    assert ft._parse_arxiv_feed("<not-xml") == []
    assert ft._parse_arxiv_feed('<feed xmlns="http://www.w3.org/2005/Atom"></feed>') == []


def test_arxiv_search_uses_monkeypatched_transport(monkeypatch):
    seen = []

    async def fake_req(url, **kw):
        seen.append(kw.get("params", {}))
        return ARXIV_FEED

    monkeypatch.setattr(ft, "_request", fake_req)
    hits = asyncio.run(ft.arxiv_search("nfv failure prediction", 5))
    assert len(hits) == 1 and hits[0].year == 2024
    assert seen and 'all:"nfv failure prediction"' in seen[0]["search_query"]


def test_arxiv_search_falls_back_to_terms_query(monkeypatch):
    bodies = []

    async def fake_req(url, **kw):
        bodies.append(kw["params"]["search_query"])
        return ARXIV_FEED if len(bodies) > 1 else '<feed xmlns="http://www.w3.org/2005/Atom"/>'

    monkeypatch.setattr(ft, "_request", fake_req)
    hits = asyncio.run(ft.arxiv_search("nfv failing", 5))
    assert hits and len(bodies) == 2              # phrase query miss -> terms query


def test_arxiv_search_disabled_or_cached(monkeypatch):
    calls = []

    async def fake_req(url, **kw):
        calls.append(url)
        return ARXIV_FEED

    monkeypatch.setattr(ft, "_request", fake_req)
    off = SimpleNamespace(free_tools_enabled=False)
    assert asyncio.run(ft.arxiv_search("q", 3, off)) == []
    asyncio.run(ft.arxiv_search("nfv", 3))
    asyncio.run(ft.arxiv_search("nfv", 3))        # cache hit -> no second call
    assert len(calls) == 1


def test_url_identifier_extraction():
    assert ft.arxiv_id_from_url("https://arxiv.org/abs/2401.12345v2") == "2401.12345v2"
    assert ft.arxiv_id_from_url("https://arxiv.org/pdf/2401.12345") == "2401.12345"
    assert ft.arxiv_id_from_url("https://example.com/x") == ""
    assert ft.doi_from_url("https://doi.org/10.1109/netsoft48620.2020.9165348") == \
        "10.1109/netsoft48620.2020.9165348"
    assert ft.doi_from_url("https://example.com/x") == ""


# ---------------------------------------------------------------- OpenAlex / Crossref
def test_openalex_inverted_abstract(monkeypatch):
    data = {"results": [{
        "display_name": "ML for NFV",
        "doi": "https://doi.org/10.1000/x",
        "id": "https://openalex.org/W1",
        "publication_year": 2024,
        "cited_by_count": 42,
        "primary_location": {"source": {"display_name": "IEEE NetSoft"}},
        "authorships": [{"author": {"display_name": "Jane Doe"}}],
        "abstract_inverted_index": {"NFV": [2], "about": [1], "Learning": [0]},
    }]}

    async def fake_json(url, **kw):
        assert "mailto" not in (kw.get("params") or {})   # no polite contact set
        return data

    monkeypatch.setattr(ft, "_get_json", fake_json)
    hits = asyncio.run(ft.openalex_search("nfv", 5))
    h = hits[0]
    assert h.source == "openalex" and h.cited_by == 42 and h.year == 2024
    assert h.doi == "10.1000/x" and h.venue == "IEEE NetSoft"
    assert h.abstract == "Learning about NFV"


def test_openalex_polite_pool_mailto(monkeypatch):
    seen = {}

    async def fake_json(url, params=None, **kw):
        seen.update(params or {})
        return {"results": []}

    monkeypatch.setattr(ft, "_get_json", fake_json)
    s = SimpleNamespace(free_tools_mailto="lab@example.org")
    asyncio.run(ft.openalex_search("q", 3, s))
    assert seen.get("mailto") == "lab@example.org"


def test_crossref_works_fields(monkeypatch):
    data = {"message": {"items": [{
        "DOI": "10.1109/netsoft48620.2020.9165348",
        "title": ["Machine Learning for Dynamic Resource Allocation in NFV"],
        "author": [{"family": "Schneider", "given": "Stefan"}],
        "issued": {"date-parts": [[2020, 6]]},
        "container-title": ["2020 6th IEEE NetSoft"],
        "is-referenced-by-count": 37,
        "abstract": "<jats:p>Clean <jats:italic>markup</jats:italic> test.</jats:p>",
    }]}}

    async def fake_json(url, **kw):
        assert kw["params"]["select"].startswith("DOI,title")
        return data

    monkeypatch.setattr(ft, "_get_json", fake_json)
    hits = asyncio.run(ft.crossref_works("nfv", 5))
    h = hits[0]
    assert h.title == "Machine Learning for Dynamic Resource Allocation in NFV"
    assert h.authors == ["Schneider, Stefan"] and h.year == 2020
    assert h.cited_by == 37 and h.venue == "2020 6th IEEE NetSoft"
    assert h.url == "https://doi.org/10.1109/netsoft48620.2020.9165348"
    assert h.abstract == "Clean markup test."     # JATS tags stripped


# ---------------------------------------------------------------- BibTeX enrichment
SECOND_ARXIV_ENTRY = """  <entry>
    <id>http://arxiv.org/abs/2401.67890</id>
    <title>Second Paper on NFV</title>
    <summary>Another abstract.</summary>
    <published>2024-02-20T00:00:00Z</published>
    <author><name>Alan Smithee</name></author>
  </entry>
"""


def test_enriched_bibtex_batches_arxiv_into_one_call(monkeypatch):
    calls = []

    async def fake_req(url, **kw):
        calls.append(kw.get("params"))
        return ARXIV_FEED.replace("</feed>", SECOND_ARXIV_ENTRY + "</feed>")

    monkeypatch.setattr(ft, "_request", fake_req)
    urls = ["https://arxiv.org/abs/2401.12345v2", "https://arxiv.org/abs/2401.67890"]
    out = asyncio.run(ft.enriched_bibtex(urls))
    assert out.count("@misc{") == 2
    assert len(calls) == 1                        # ONE id_list call for both ids
    assert calls[0]["id_list"] == "2401.12345v2,2401.67890"
    assert "Jane Doe" in out and "2024" in out
    assert "eprint = {2401.12345v2}" in out
    assert "Second Paper on NFV" in out


def test_enriched_bibtex_dedups_same_paper_versions(monkeypatch):
    """abs/...v1 and abs/...v2 of the SAME paper share one citation key —
    one entry survives (parse_citations keys are version-agnostic)."""

    async def fake_req(url, **kw):
        return ARXIV_FEED

    monkeypatch.setattr(ft, "_request", fake_req)
    urls = ["https://arxiv.org/abs/2401.12345v2", "https://arxiv.org/abs/2401.12345v1"]
    out = asyncio.run(ft.enriched_bibtex(urls))
    assert out.count("@misc{") == 2               # emitted per selected citation
    assert out.count("ML for NFV failure prediction") == 2


def test_enriched_bibtex_crossref_and_legacy(monkeypatch):
    async def fake_json(url, **kw):
        return {"message": {"title": ["A Journal Paper"],
                            "issued": {"date-parts": [[2023]]},
                            "author": [{"family": "Roe", "given": "John"}],
                            "container-title": ["IEEE Foobar"]}}

    async def fake_req(url, **kw):
        return None                               # arXiv API down -> legacy chunk

    monkeypatch.setattr(ft, "_get_json", fake_json)
    monkeypatch.setattr(ft, "_request", fake_req)
    monkeypatch.setattr(cite_fmt, "fetch_title", lambda url, timeout=8: "Legacy Title")
    urls = ["https://arxiv.org/abs/2401.12345", "https://doi.org/10.1000/xyz",
            "https://example.com/plain"]
    out = asyncio.run(ft.enriched_bibtex(urls))
    assert "A Journal Paper" in out and "Roe, John" in out and "IEEE Foobar" in out
    assert "Legacy Title" in out                  # unknown URL keeps legacy behaviour
    assert out.count("@misc{") == 3
    assert out.count("Legacy Title") == 2         # arXiv-legacy + plain-URL entries
    # citation order preserved: arXiv entry, DOI entry, plain URL last
    assert out.index("2401.12345") < out.index("A Journal Paper") < out.index("example.com/plain")


def test_enriched_bibtex_empty_urls():
    assert asyncio.run(ft.enriched_bibtex([])) == ""


# ---------------------------------------------------------------- composition
def _hit(source, title, url, doi=""):
    return ft.PaperHit(source=source, title=title, url=url, doi=doi,
                       authors=["Jane Doe"], year=2024, cited_by=3)


def test_search_all_dedups_and_fills_from_crossref(monkeypatch):
    async def arxiv(query, max_results=8, settings=None):
        return [_hit("arxiv", "Same Paper", "https://arxiv.org/abs/1", "10.1/x")]

    async def openalex(query, max_results=8, settings=None):
        return [_hit("openalex", "Same Paper", "https://doi.org/10.1/x", "10.1/x"),
                _hit("openalex", "Only OpenAlex", "https://doi.org/10.1/y", "10.1/y")]

    crossref_called = []

    async def crossref(query, max_results=8, settings=None):
        crossref_called.append(query)
        return [_hit("crossref", "Crossref Fill", "https://doi.org/10.1/z", "10.1/z")]

    monkeypatch.setattr(ft, "arxiv_search", arxiv)
    monkeypatch.setattr(ft, "openalex_search", openalex)
    monkeypatch.setattr(ft, "crossref_works", crossref)
    merged = asyncio.run(ft.search_all("q", 8))
    assert [h.title for h in merged] == ["Same Paper", "Only OpenAlex", "Crossref Fill"]
    assert crossref_called == ["q"]


def test_fallback_research_shape(monkeypatch):
    hits = [_hit("arxiv", "Real Paper", "https://arxiv.org/abs/2401.1", "10.1/a"),
            _hit("openalex", "Second Paper", "https://doi.org/10.1/b", "10.1/b")]

    async def search_all(query, max_results=8, settings=None):
        return hits

    async def bib(urls, settings=None, **kw):
        return "@misc{real, title={Real}}"

    monkeypatch.setattr(ft, "search_all", search_all)
    monkeypatch.setattr(ft, "enriched_bibtex", bib)
    fb = asyncio.run(ft.fallback_research("nfv scaling"))
    assert fb.ok and fb.source == "free-tools"
    assert "Real Paper" in fb.answer and "Second Paper" in fb.answer
    assert fb.citations == ["https://arxiv.org/abs/2401.1", "https://doi.org/10.1/b"]
    assert fb.bibtex == "@misc{real, title={Real}}"
    # LLMResponse-compatible duck typing
    assert isinstance(fb.ok, bool) and hasattr(fb, "answer") and hasattr(fb, "citations")


def test_fallback_research_empty_sources(monkeypatch):
    async def search_all(query, max_results=8, settings=None):
        return []

    monkeypatch.setattr(ft, "search_all", search_all)
    fb = asyncio.run(ft.fallback_research("nothing out there"))
    assert fb.ok is False and fb.citations == []


# ---------------------------------------------------------------- general tools
def test_hf_datasets_search(monkeypatch):
    async def fake_json(url, params=None, **kw):
        return [{"id": "user/ds", "downloads": 224, "likes": 9, "lastModified": "2024-01-01"}]

    monkeypatch.setattr(ft, "_get_json", fake_json)
    rows = asyncio.run(ft.hf_datasets_search("intrusion", 3))
    assert rows[0]["id"] == "user/ds" and rows[0]["downloads"] == 224
    assert rows[0]["url"] == "https://huggingface.co/datasets/user/ds"


def test_wikipedia_summary(monkeypatch):
    async def fake_json(url, **kw):
        return {"title": "NFV", "extract": "Network function virtualization is ...",
                "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/NFV"}}}

    monkeypatch.setattr(ft, "_get_json", fake_json)
    info = asyncio.run(ft.wikipedia_summary("network function virtualization"))
    assert info["term"] == "NFV" and "virtualization" in info["extract"]


def test_wikipedia_miss_is_cached(monkeypatch):
    async def fake_json(url, **kw):
        return None

    calls = []

    async def counting_json(url, **kw):
        calls.append(url)
        return await fake_json(url, **kw)

    monkeypatch.setattr(ft, "_get_json", counting_json)
    assert asyncio.run(ft.wikipedia_summary("nonexistent-term-xyz")) is None
    assert asyncio.run(ft.wikipedia_summary("nonexistent-term-xyz")) is None
    assert len(calls) == 1                        # miss cached, no repeat fetch


def test_ddg_results_and_redirect_decoding(monkeypatch):
    html = """
    <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fa&rut=x">Result <b>One</b></a>
    <a class="result__snippet" href="#">Snippet <i>text</i> here</a>
    <a rel="nofollow" class="result__a" href="https://direct.example.org/b">Direct Two</a>
    """

    async def fake_req(url, **kw):
        assert kw.get("method") == "POST" and kw["data"]["q"] == "nfv"
        return html

    monkeypatch.setattr(ft, "_request", fake_req)
    rows = asyncio.run(ft.ddg_web_search("nfv", 5))
    assert rows[0]["url"] == "https://example.org/a"        # uddg redirect decoded
    assert rows[0]["title"] == "Result One"                 # tags stripped
    assert rows[0]["snippet"] == "Snippet text here"
    assert rows[1]["url"] == "https://direct.example.org/b"


def test_ddg_graceful_failure(monkeypatch):
    async def fake_req(url, **kw):
        return None

    monkeypatch.setattr(ft, "_request", fake_req)
    assert asyncio.run(ft.ddg_web_search("q", 5)) == []


def test_request_retries_once_on_429(monkeypatch):
    statuses = [429, 200]

    class FakeResp:
        def __init__(self):
            self.status = statuses.pop(0)
            self.headers = {"Retry-After": "0.01"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def text(self):
            return "body"

    class FakeSess:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def request(self, *a, **k):
            return FakeResp()

    sleeps = []

    async def fake_sleep(s):
        sleeps.append(s)

    monkeypatch.setattr(ft.aiohttp, "ClientSession", FakeSess)
    monkeypatch.setattr(ft.asyncio, "sleep", fake_sleep)
    out = asyncio.run(ft._request("https://api.example.org/x"))
    assert out == "body" and len(sleeps) == 1 and sleeps[0] == 0.01


def test_request_no_retry_on_404(monkeypatch):
    class FakeResp:
        status = 404
        headers = {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    class FakeSess:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def request(self, *a, **k):
            return FakeResp()

    monkeypatch.setattr(ft.aiohttp, "ClientSession", FakeSess)
    assert asyncio.run(ft._request("https://api.example.org/x")) is None


# ---------------------------------------------------------------- config knobs
def test_settings_free_tools_knobs(monkeypatch):
    from agentics.config import Settings

    monkeypatch.setenv("FREE_TOOLS_ENABLED", "0")
    monkeypatch.setenv("FREE_TOOLS_TIMEOUT", "7")
    monkeypatch.setenv("FREE_TOOLS_MAILTO", "lab@example.org")
    s = Settings()
    assert s.free_tools_enabled is False
    assert s.free_tools_timeout == 7.0
    assert s.free_tools_mailto == "lab@example.org"


# ---------------------------------------------------------------- node wiring
class FakePersonal:
    def add_lesson(self, lesson, issue=None):
        pass


class FakeVector:
    available = False

    def __init__(self):
        self.added = []

    def add(self, content, meta):
        self.added.append((content, meta))


class FlakyHost:
    """BrowserSessionHost stand-in whose ask() always fails."""

    def __init__(self):
        self.account = SimpleNamespace(name="account-07")

    async def ask(self, query, *, mode="concise", system=None, purpose=None):
        raise RuntimeError("input box did not appear")

    def mark_failed(self, error, cooldown_s=900):
        pass


class OkHost:
    """BrowserSessionHost stand-in with a healthy research ask."""

    def __init__(self, answer="llm survey", citations=None):
        self.account = SimpleNamespace(name="account-02")
        self._resp = LLMResponse(answer=answer, citations=citations or [])

    async def ask(self, query, *, mode="concise", system=None, purpose=None):
        return self._resp

    def mark_failed(self, error, cooldown_s=900):
        pass


def _ctx(mem, role, llm, enabled=True):
    return SimpleNamespace(
        agent_id="a04", role=role, memory=mem, vector=FakeVector(), llm=llm,
        meta={"personal": FakePersonal()},
        settings=SimpleNamespace(research_topic="NFV automation",
                                 free_tools_enabled=enabled),
        heartbeat=lambda status="working": None,
        report=lambda **k: None,
    )


def test_node_research_falls_back_when_llm_dies(tmp_path, monkeypatch):
    mem = SharedMemory(tmp_path / "m.db")
    ctx = _ctx(mem, Role.RESEARCHER, FlakyHost())
    fb = ft.FreeResearch(
        answer="Real finding: XGBoost reached 0.93 precision on NFV data.",
        citations=["https://arxiv.org/abs/2401.12345"],
        bibtex="@misc{arxiv240112345, title = {Real}}",
    )

    async def fake_fallback(query, settings=None, max_results=8):
        assert query == "Survey NFV"
        return fb

    monkeypatch.setattr(nodes.free_tools, "fallback_research", fake_fallback)
    out = asyncio.run(nodes.node_research(ctx, {"issue_title": "Survey NFV",
                                                "papers": [], "citations": []}))
    assert out.get("error") is None
    assert out["papers"][0]["answer"].startswith("Real finding:")
    assert out["citations"] == ["https://arxiv.org/abs/2401.12345"]
    assert out["bibtex"] == fb.bibtex
    assert out["action"] == "verify"              # researcher continues to H2
    assert out["finding_id"]
    assert ctx.meta["research_source"] == "free-tools"
    rows = mem.list_findings(limit=5)
    assert rows and rows[0]["title"] == "Findings: Survey NFV"
    assert "Real finding:" in rows[0]["content"]
    assert ctx.vector.added                       # vector store got the finding
    mem.close()


def test_node_research_no_llm_uses_free_tools(tmp_path, monkeypatch):
    mem = SharedMemory(tmp_path / "m.db")
    ctx = _ctx(mem, Role.WRITER, None)

    async def fake_fallback(query, settings=None, max_results=8):
        return ft.FreeResearch(answer="papers from official APIs",
                               citations=["https://doi.org/10.1/a"],
                               bibtex="@misc{doi10_1_a, title = {X}}")

    monkeypatch.setattr(nodes.free_tools, "fallback_research", fake_fallback)
    out = asyncio.run(nodes.node_research(ctx, {"issue_title": "T", "papers": []}))
    assert out["action"] == "plan"                # non-researcher skips H2
    assert ctx.meta["research_source"] == "free-tools"
    mem.close()


def test_node_research_fallback_empty_keeps_error_state(tmp_path, monkeypatch):
    mem = SharedMemory(tmp_path / "m.db")
    ctx = _ctx(mem, Role.RESEARCHER, FlakyHost())

    async def fake_fallback(query, settings=None, max_results=8):
        return ft.FreeResearch(answer="", citations=[])

    monkeypatch.setattr(nodes.free_tools, "fallback_research", fake_fallback)
    out = asyncio.run(nodes.node_research(ctx, {"issue_title": "T", "papers": []}))
    assert out["error"].startswith("research llm call failed:")
    assert out["done"] is True
    assert out["papers"] == []                    # predecessor's guarantee intact
    mem.close()


def test_node_research_fallback_disabled_keeps_error_state(tmp_path):
    mem = SharedMemory(tmp_path / "m.db")
    ctx = _ctx(mem, Role.RESEARCHER, FlakyHost(), enabled=False)
    out = asyncio.run(nodes.node_research(ctx, {"issue_title": "T", "papers": []}))
    assert out["error"].startswith("research llm call failed:")
    assert out["done"] is True
    mem.close()


def test_node_research_llm_path_uses_grounded_bibtex(tmp_path, monkeypatch):
    mem = SharedMemory(tmp_path / "m.db")
    ctx = _ctx(mem, Role.RESEARCHER, OkHost(answer="llm survey",
                                            citations=["https://arxiv.org/abs/2401.12345"]))

    async def fake_bib(urls, settings=None, **kw):
        return "@misc{grounded, title = {Real Metadata}}"

    monkeypatch.setattr(nodes.free_tools, "enriched_bibtex", fake_bib)
    out = asyncio.run(nodes.node_research(ctx, {"issue_title": "T", "papers": []}))
    assert out["bibtex"] == "@misc{grounded, title = {Real Metadata}}"
    assert ctx.meta["research_source"] == "llm"
    assert out["action"] == "verify"
    mem.close()


def test_node_research_empty_llm_answer_triggers_fallback(tmp_path, monkeypatch):
    mem = SharedMemory(tmp_path / "m.db")
    ctx = _ctx(mem, Role.RESEARCHER, OkHost(answer="   ", citations=[]))

    async def fake_fallback(query, settings=None, max_results=8):
        return ft.FreeResearch(answer="official api papers",
                               citations=["https://arxiv.org/abs/1"],
                               bibtex="@misc{b}")

    monkeypatch.setattr(nodes.free_tools, "fallback_research", fake_fallback)
    out = asyncio.run(nodes.node_research(ctx, {"issue_title": "T", "papers": []}))
    assert out["papers"][0]["answer"] == "official api papers"
    assert ctx.meta["research_source"] == "free-tools"
    mem.close()
