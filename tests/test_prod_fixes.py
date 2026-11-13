"""Production-fix regression tests (2026-11-13 incident):

1. GitHub 403/500 storms   — API pacing, GithubRetry config, scan cache,
   claim-label retry.
2. Playwright timeout on the plan node — challenge fail-fast, diagnostic
   input-box errors (never a raw Playwright TimeoutError).
3. Cycle resilience — LLM failures degrade to state["error"] + reflect
   instead of killing the LangGraph task.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentics.agents.roles import Role
from agentics.config import Settings
from agentics.github_ops import issue_ops
from agentics.github_ops.client import GitHubClient, make_paced
from agentics.graph import nodes
from agentics.llm.hyperbrowser_client import HyperbrowserClient
from agentics.llm.perplexity_client import CookieAccount
from agentics.memory.shared_memory import SharedMemory


# ============================================================ 1. GitHub ops
def test_make_paced_enforces_min_interval():
    calls: list[float] = []

    def orig(*a, **k):
        calls.append(time.monotonic())
        return "ok"

    paced = make_paced(orig, min_interval=0.06)
    paced()
    paced()
    assert len(calls) == 2
    assert calls[1] - calls[0] >= 0.055          # jitter may add, never subtract


def test_github_client_installs_pacing_and_retry():
    s = Settings()
    s.github_token = "dummy-token-for-pacing-test"
    s.gh_min_api_interval = 1.0
    client = GitHubClient(s)
    # pacing wrappers shadow the requester methods
    assert getattr(client.gh.requester.requestJsonAndCheck, "__name__", "") == "paced"
    assert getattr(client.gh.requester.requestJson, "__name__", "") == "paced"
    # retry is a configured GithubRetry (6 attempts, capped waits)
    retry = client.gh.requester._Requester__retry
    assert retry.total == 6
    assert retry.secondary_rate_wait == 60
    assert retry.max_rate_limit_wait == 900


def test_github_client_pacing_can_be_disabled():
    s = Settings()
    s.github_token = "dummy-token-for-pacing-test"
    s.gh_min_api_interval = 0
    client = GitHubClient(s)
    assert getattr(client.gh.requester.requestJsonAndCheck, "__name__", "") != "paced"


class _FakeIssue:
    def __init__(self, number, title, labels=()):
        self.number = number
        self.title = title
        self.body = ""
        self.pull_request = None
        self.labels = [SimpleNamespace(name=n) for n in labels]
        self.added: list[str] = []

    def add_to_labels(self, name):
        self.added.append(name)


class _FakeRepo:
    """Duck-typed repo WITH a stable identity — the only kind the shared
    scan cache caches (see issue_ops.scan_open_issues: id()-keying poisons
    the cache when CPython reuses a dead fake's address)."""

    def __init__(self, issues, repo_id=42001):
        self._issues = issues
        self.id = repo_id
        self.get_issues_calls = 0

    def get_issues(self, **kw):
        self.get_issues_calls += 1
        return list(self._issues)


def test_scan_open_issues_cached_across_calls():
    repo = _FakeRepo([_FakeIssue(1, "Fix a", labels=())])
    first = issue_ops.scan_open_issues(repo)
    second = issue_ops.scan_open_issues(repo)
    assert repo.get_issues_calls == 1            # one API call, two scans
    assert [i.number for i in first] == [i.number for i in second] == [1]


def test_scan_cache_distinguishes_repos_and_limits():
    repo_a = _FakeRepo([_FakeIssue(1, "A")], repo_id=42002)
    repo_b = _FakeRepo([_FakeIssue(2, "B")], repo_id=42003)
    assert issue_ops.scan_open_issues(repo_a)[0].number == 1
    assert issue_ops.scan_open_issues(repo_b)[0].number == 2
    assert repo_a.get_issues_calls == 1 and repo_b.get_issues_calls == 1


def test_scan_cache_skips_identity_less_repos():
    """Their c90ded0 contract: a duck-typed repo WITHOUT a stable id must NOT
    poison the shared cache — every scan goes to the (fake) API."""
    class NoIdRepo(_FakeRepo):
        def __init__(self, issues):
            super().__init__(issues)
            del self.id
    repo = NoIdRepo([_FakeIssue(1, "A")])
    issue_ops.scan_open_issues(repo)
    issue_ops.scan_open_issues(repo)
    assert repo.get_issues_calls == 2


def test_scan_cache_refresh_after_clear():
    repo = _FakeRepo([_FakeIssue(1, "A")])
    issue_ops.scan_open_issues(repo)
    issue_ops.clear_scan_cache()
    issue_ops.scan_open_issues(repo)
    assert repo.get_issues_calls == 2


def test_claim_issue_retries_label_on_transient_5xx(monkeypatch):
    monkeypatch.setattr(issue_ops, "CLAIM_LABEL_RETRY_DELAYS", (0.0, 0.0))
    issue = _FakeIssue(19, "Fix claim race")
    attempts = {"n": 0}

    def add_label(name):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("too many 500 error responses")
        issue.added.append(name)

    issue.add_to_labels = add_label
    issue.create_comment = lambda body: None

    repo = SimpleNamespace(get_issue=lambda n: issue)
    ok = issue_ops.claim_issue(repo, 19, "a04", ttl_minutes=45)
    assert ok is True and attempts["n"] == 3
    assert issue.added == ["claimed/a04"]


def test_claim_issue_gives_up_after_retries(monkeypatch):
    monkeypatch.setattr(issue_ops, "CLAIM_LABEL_RETRY_DELAYS", (0.0, 0.0))
    issue = _FakeIssue(19, "Fix claim race")
    attempts = {"n": 0}

    def add_label(name):
        attempts["n"] += 1
        raise RuntimeError("github is down (500)")

    issue.add_to_labels = add_label
    repo = SimpleNamespace(get_issue=lambda n: issue)
    assert issue_ops.claim_issue(repo, 19, "a04", ttl_minutes=45) is False
    assert attempts["n"] == 3                    # 1 initial + 2 retries
    assert not issue.added


def test_claim_issue_double_claim_rejected_via_fresh_labels():
    held = _FakeIssue(5, "Taken", labels=("claimed/a02",))
    repo = SimpleNamespace(get_issue=lambda n: held)
    assert issue_ops.claim_issue(repo, 5, "a04", ttl_minutes=45) is False


# ==================================================== 2. Hyperbrowser client
def _hb_client(tmp_path, page, ctx=None):
    s = Settings()
    s.hb_input_timeout = 5
    client = HyperbrowserClient(
        CookieAccount(name="account-07", cookies=[
            {"name": "session-token", "value": "pplx-x", "domain": ".perplexity.ai"}]),
        s,
    )
    client._page = page
    if ctx is None:
        async def _cookies(_url):
            return [{"name": "session-token"}]
        ctx = SimpleNamespace(cookies=_cookies)
    client._ctx = ctx
    return client


class _FakeLocator:
    def __init__(self, fail_times: int = 0):
        self.fail_times = fail_times

    @property
    def first(self):
        return self

    async def wait_for(self, state="visible", timeout=None):
        if self.fail_times > 0:
            self.fail_times -= 1
            raise TimeoutError(f"Locator.wait_for: Timeout {timeout}ms exceeded.")

    async def click(self, timeout=None): ...
    async def focus(self): ...
    async def press(self, key): ...


class _FakeKeyboard:
    def __init__(self):
        self.pressed: list[str] = []
        self.inserted: list[str] = []

    async def press(self, key):
        self.pressed.append(key)

    async def insert_text(self, text):
        self.inserted.append(text)


class _FakePage:
    """Offline stand-in for the Playwright page used by HyperbrowserClient.

    ``evaluate`` returns ``_eval`` — remote code expects a dict from its
    diagnostics JS, so challenge/auth tests set it to a dict; default is a
    body string for generic probes.
    """

    def __init__(self, title="Perplexity", body="Ask anything", fail_times=0,
                 eval_value=None):
        self._title, self._body, self._fail = title, body, fail_times
        self._eval = eval_value
        self.keyboard = _FakeKeyboard()
        self.url = "https://www.perplexity.ai/"
        self.goto_calls: list[str] = []

    async def goto(self, url, timeout=None, wait_until=None):
        self.goto_calls.append(url)

    async def title(self):
        return self._title

    async def evaluate(self, script):
        if self._eval is not None:
            return self._eval
        return self._body

    def locator(self, sel):
        # each new locator consumes one failure slot (like a real page whose
        # overlay disappears after the first dismissal)
        fails = 0
        if self._fail > 0:
            fails, self._fail = self._fail, self._fail - 1
        return _FakeLocator(fail_times=fails)


def _diag(title="Perplexity", body="", url="https://www.perplexity.ai/"):
    """Diagnostics dict shaped like the remote _page_diagnostics JS result."""
    return {"url": url, "title": title, "body": body}


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture()
def _fast_hb(monkeypatch):
    """Zero the UI settle delays so browser tests run instantly."""
    from agentics.llm import hyperbrowser_client as hb

    async def _nosleep(_s):
        return None

    monkeypatch.setattr(hb.asyncio, "sleep", _nosleep)
    return hb


def test_goto_home_fails_fast_on_challenge_page(_fast_hb):
    from agentics.llm.hyperbrowser_client import HyperbrowserBackendUnavailable

    page = _FakePage(eval_value=_diag(title="Just a moment...",
                                      body="Checking your browser before accessing"))
    client = _hb_client(Path("/tmp"), page)
    with pytest.raises(HyperbrowserBackendUnavailable, match="verification page"):
        _run(client._goto_home())


def test_goto_home_passes_healthy_page(_fast_hb):
    page = _FakePage(eval_value=_diag(title="Perplexity — Ask anything",
                                      body="Welcome back, researcher"))
    client = _hb_client(Path("/tmp"), page)
    _run(client._goto_home())                    # must not raise
    assert page.goto_calls == ["https://www.perplexity.ai"]


def test_composer_timeout_raises_typed_ui_error(_fast_hb):
    from agentics.llm.hyperbrowser_client import HyperbrowserUIError

    page = _FakePage(eval_value=_diag(title="Perplexity", body=""),
                     fail_times=99)              # composer never appears
    client = _hb_client(Path("/tmp"), page)
    with pytest.raises(HyperbrowserUIError, match="composer unavailable") as info:
        _run(client._submit("hello"))
    # the diagnostic carries the page context and honours HB_INPUT_TIMEOUT
    assert "5s" in str(info.value)
    assert not isinstance(info.value, TimeoutError)


def test_composer_signin_wall_raises_auth_error(_fast_hb):
    from agentics.llm.hyperbrowser_client import HyperbrowserAuthError

    page = _FakePage(eval_value=_diag(title="Perplexity",
                                      body="You must sign in to continue"),
                     fail_times=99)
    client = _hb_client(Path("/tmp"), page)
    with pytest.raises(HyperbrowserAuthError, match="sign-in page"):
        _run(client._submit("hello"))


# ==================================================== 3. Cycle resilience
class FakePersonal:
    def add_lesson(self, lesson, issue=None):
        self.lessons = getattr(self, "lessons", []) + [lesson]


class FlakyHost:
    """BrowserSessionHost stand-in whose ask() always fails."""

    def __init__(self, error="RuntimeError: input box did not appear"):
        self.error = error
        self.account = SimpleNamespace(name="account-07")

    async def ask(self, query, *, mode="concise", system=None, purpose=None):
        raise RuntimeError(self.error)

    def mark_failed(self, error, cooldown_s=900):
        pass


def _mem_ctx(tmp_path, role, llm):
    mem = SharedMemory(tmp_path / "m.db")
    ctx = SimpleNamespace(
        agent_id="a04",
        role=role,
        memory=mem,
        vector=SimpleNamespace(available=False),
        llm=llm,
        meta={"personal": FakePersonal()},
        # free tools off: these tests pin the LLM-only behaviour — the
        # fallback path (enabled) has its own suite in test_free_tools.py
        settings=SimpleNamespace(research_topic="NFV automation",
                                 free_tools_enabled=False),
        heartbeat=lambda status="working": None,
        report=lambda **k: None,
    )
    return ctx, mem


def test_node_plan_llm_failure_degrades_to_error_state(tmp_path):
    ctx, _ = _mem_ctx(tmp_path, Role.CODER, FlakyHost())
    state = {"issue_number": 70, "issue_title": "Fix plan node", "issue_body": "",
             "papers": [{}], "category": "bugfix"}
    out = asyncio.run(nodes.node_plan(ctx, state))
    assert out["error"].startswith("llm plan call failed:")
    assert out["done"] is True
    assert "plan" not in out                     # nothing half-parsed


def test_node_research_llm_failure_degrades_to_error_state(tmp_path):
    ctx, _ = _mem_ctx(tmp_path, Role.RESEARCHER, FlakyHost())
    state = {"issue_title": "Survey NFV", "papers": [], "citations": []}
    out = asyncio.run(nodes.node_research(ctx, state))
    assert out["error"].startswith("research llm call failed:")
    assert out["done"] is True
    assert out["papers"] == []                   # no finding stored


def test_node_review_llm_failure_skips_verdict(tmp_path):
    """Local-mode MR flow: reviewer's LLM dies -> no verdict, no enqueue,
    the MR stays open for the next cycle (degrade, don't crash)."""
    ctx, mem = _mem_ctx(tmp_path, Role.REVIEWER, FlakyHost())
    task_id = mem.create_task("work item", body="", category="code", agent="seed")
    mr_id = mem.open_review(
        task_id=task_id, title="[a02] work (#%d)" % task_id, body="",
        author="a02", branch="agent/a02/task-%d-x" % task_id,
        files=["docs/x.md"], verification="tier-1 passed")
    ctx.merge_queue = SimpleNamespace(
        enqueue=lambda *a, **k: pytest.fail("must not enqueue on failed review"))
    state = asyncio.run(nodes.node_review(ctx, {}))
    assert state["error"].startswith("llm review call failed:")
    assert state["done"] is True
    assert "review_verdict" not in state         # no verdict without the LLM
    assert mem.get_review(mr_id)["status"] == "open"
