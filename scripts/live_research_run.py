"""Live research run — the real RESEARCHER cycle over the configured browser backend.

Runs the actual graph nodes (node_start -> node_research -> node_plan) with
only GitHub/workspace edges stubbed: the LLM edge is fully live (the
Perplexity cookie pool) and memory is a real SharedMemory.

Backends (LIVE_BACKEND, default: hyperbrowser when HB_API_KEY is set,
else the free local cookie WebSocket):
    local        raw cookie WebSocket from THIS machine — free, no key,
                 needs a residential IP (Perplexity 403-blocks datacenters)
    hyperbrowser stealth cloud browser (residential proxy) — works from
                 datacenter IPs, needs HB_API_KEY / HB_API_KEYS

Env:
    HB_API_KEY   Hyperbrowser API key (enables the hyperbrowser backend)
    LIVE_BACKEND backend override: "local" | "hyperbrowser" (optional)
    LIVE_TOPIC   research topic override (optional)
    LIVE_REPORT  report path override (default: data/live_run_report.json)

Usage:
    HB_API_KEY=... python scripts/live_research_run.py   # cloud backend
    python scripts/live_research_run.py                  # local cookies
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from agentics.agents.roles import Role  # noqa: E402
from agentics.config import Settings  # noqa: E402
from agentics.graph import nodes  # noqa: E402
from agentics.llm.session_host import BrowserSessionHost  # noqa: E402
from agentics.memory.shared_memory import SharedMemory  # noqa: E402
from agentics.memory.vector_store import VectorStore  # noqa: E402

DEFAULT_TOPIC = (
    "machine learning algorithms for early failure prediction of network "
    "functions in NFV (Network Function Virtualization) environments"
)


class FakePersonal:
    def add_lesson(self, lesson, issue=None):
        pass


def build_ctx(settings: Settings, host: BrowserSessionHost, mem: SharedMemory):
    """AgentContext-shaped namespace with GitHub/workspace edges stubbed."""
    from types import SimpleNamespace

    def _report(node: str = "", status: str = "working", task_id: int | None = None,
                task_title: str | None = None, intent: dict | None = None,
                event: str = "", detail: str | dict = "", **_) -> None:
        # same contract as AgentContext.report (agents/base.py): live state +
        # optional journal entry, so the dashboard shows the run while it goes
        mem.report("live-r01", Role.RESEARCHER.value, node=node, status=status,
                   cycle=0, task_id=task_id, task_title=task_title, intent=intent)
        if event:
            mem.journal("live-r01", event, detail)

    ctx = SimpleNamespace(
        agent_id="live-r01",
        role=Role.RESEARCHER,
        settings=settings,
        memory=mem,
        vector=VectorStore(settings.chroma_path),
        llm=host,
        gh=SimpleNamespace(repo=None),
        workspace=None,
        claims=None,
        ownership=None,
        merge_queue=None,
        cycle=0,
        meta={"personal": FakePersonal()},
        heartbeat=lambda status="working": None,
    )
    ctx.report = _report
    return ctx


async def main() -> None:
    settings = Settings()
    hb_key = os.environ.get("HB_API_KEY", "") or settings.hyperbrowser_api_key
    backend = os.environ.get("LIVE_BACKEND", "").lower() or (
        "hyperbrowser" if hb_key else "local")
    if backend not in ("local", "hyperbrowser"):
        print(f"ERROR: unknown LIVE_BACKEND {backend!r} (local | hyperbrowser)")
        sys.exit(2)
    settings.browser_backend = backend
    if backend == "hyperbrowser":
        if not hb_key:
            print("ERROR: LIVE_BACKEND=hyperbrowser needs HB_API_KEY "
                  "(or LIVE_BACKEND=local for the free cookie WebSocket)")
            sys.exit(2)
        settings.hyperbrowser_api_key = hb_key
    settings.research_topic = os.environ.get("LIVE_TOPIC", DEFAULT_TOPIC)
    settings.ensure_dirs()

    scratch = Path(tempfile.mkdtemp(prefix="live-run-"))
    mem = SharedMemory(scratch / "memory.db")
    host = BrowserSessionHost.from_settings(settings)
    ctx = build_ctx(settings, host, mem)
    report = {"topic": settings.research_topic, "backend": backend}

    t0 = time.monotonic()
    state = {}
    print(f"[{time.monotonic() - t0:6.1f}s] node_start …")
    state = nodes.node_start(ctx, state)
    print(f"[{time.monotonic() - t0:6.1f}s] node_research (topic: {settings.research_topic[:70]}…)")
    state = await nodes.node_research(ctx, state)
    answer = state.get("papers", [{}])[0].get("answer", "")
    report["research"] = {
        "done": state.get("done"), "error": state.get("error", ""),
        "citations": state.get("citations", []),
        "answer": answer,
        "answer_chars": len(answer),
        "bibtex": state.get("bibtex", ""),
    }
    print(f"[{time.monotonic() - t0:6.1f}s] research done: "
          f"{report['research']['answer_chars']} chars, "
          f"{len(report['research']['citations'])} citations")
    print("---- research answer (first 1500 chars) ----")
    print(answer[:1500])
    print("---- citations ----")
    for c in report["research"]["citations"]:
        print(" -", c)

    # feed research into the plan node like the CODER path does
    state = dict(state)
    state["issue_number"] = 0
    state["issue_title"] = f"Prototype: {settings.research_topic}"
    state["issue_body"] = "Build a small reproducible prototype from the research findings."
    state["category"] = "feature"
    print(f"[{time.monotonic() - t0:6.1f}s] node_plan …")
    state = await nodes.node_plan(ctx, state)
    plan_raw = state.get("plan", "")
    try:
        plan = json.loads(plan_raw)
    except (json.JSONDecodeError, TypeError):
        plan = None
    report["plan"] = {"parsed": plan is not None, "error": state.get("error", ""),
                      "plan": plan}
    print(f"[{time.monotonic() - t0:6.1f}s] plan done: parsed={plan is not None}")
    if plan:
        print(json.dumps(plan, indent=1, ensure_ascii=False)[:2500])

    # persisted memory evidence
    findings = mem.list_findings(limit=5)
    report["memory"] = {"findings": len(findings),
                        "messages": len(mem.recent_messages(limit=20))}
    print(f"[{time.monotonic() - t0:6.1f}s] memory: {len(findings)} finding(s) persisted")

    for client in list(host._clients.values()):
        await client.close()
    mem.close()

    out = Path(os.environ.get(
        "LIVE_REPORT", str(REPO / "data" / "live_run_report.json")))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"report saved: {out} | total wall time: {time.monotonic() - t0:.0f}s")


if __name__ == "__main__":
    asyncio.run(main())
