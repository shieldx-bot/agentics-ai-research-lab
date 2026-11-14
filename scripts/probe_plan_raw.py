"""Capture the RAW planner answer before JSON parsing (plan-parse diagnosis).

Same shape as live_research_run.py (real LLM edge, stubbed edges) but wraps
nodes.parse_json_block so the exact model output is saved to /tmp/plan_raw.txt.

Usage: HB_API_KEY=... python scripts/probe_plan_raw.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from agentics.agents.roles import Role  # noqa: E402
from agentics.config import Settings  # noqa: E402
from agentics.graph import nodes  # noqa: E402
from agentics.llm.session_host import BrowserSessionHost  # noqa: E402
from agentics.memory.shared_memory import SharedMemory  # noqa: E402
from agentics.memory.vector_store import VectorStore  # noqa: E402

from scripts.live_research_run import DEFAULT_TOPIC, FakePersonal, build_ctx  # noqa: E402


async def main() -> None:
    settings = Settings()
    settings.browser_backend = "hyperbrowser"
    settings.hyperbrowser_api_key = os.environ["HB_API_KEY"]
    settings.research_topic = os.environ.get("LIVE_TOPIC", DEFAULT_TOPIC)
    settings.ensure_dirs()

    scratch = Path(tempfile.mkdtemp(prefix="plan-probe-"))
    mem = SharedMemory(scratch / "memory.db")
    host = BrowserSessionHost.from_settings(settings)
    ctx = build_ctx(settings, host, mem)

    # wrap the parser to capture the raw answer it receives
    raw = {"text": ""}

    def _spy(text):
        raw["text"] = text
        return _orig(text)

    _orig = nodes.parse_json_block
    nodes.parse_json_block = _spy

    state: dict = {}
    state = nodes.node_start(ctx, state)
    state = await nodes.node_research(ctx, state)
    state = dict(state)
    state["issue_number"] = 0
    state["issue_title"] = f"Prototype: {settings.research_topic}"
    state["issue_body"] = "Build a small reproducible prototype from the research findings."
    state["category"] = "feature"
    state = await nodes.node_plan(ctx, state)
    print("plan error:", repr(state.get("error", "")))
    Path("/tmp/plan_raw.txt").write_text(raw["text"], encoding="utf-8")
    print("raw answer saved to /tmp/plan_raw.txt, len =", len(raw["text"]))

    for client in list(host._clients.values()):
        await client.close()
    mem.close()


if __name__ == "__main__":
    asyncio.run(main())
