"""Live lab demo — the REAL orchestrator loop over Hyperbrowser, bounded.

Runs the actual multi-agent swarm (real LangGraph cycle, real SQLite memory,
real bare repo + merges, real Perplexity-via-Hyperbrowser LLM edge) for a
fixed number of minutes, then stops it gracefully through the owner's STOP
mechanism and prints the owner's state view.

This is the production entrypoint's little brother: nothing is stubbed — the
only difference from `agentics run` is the time box and the role override.

Usage:
    HB_API_KEY=… python scripts/live_lab_demo.py [--minutes 8] [--agents 1]
                            [--role researcher] [--topic "…"]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from agentics.config import get_settings  # noqa: E402
from agentics.memory.shared_memory import SharedMemory  # noqa: E402
from agentics.stop import clear_stop, request_stop  # noqa: E402


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=int, default=8)
    ap.add_argument("--shutdown-wait", type=int, default=120,
                    help="seconds to wait for the graceful stop before a hard cancel")
    ap.add_argument("--agents", type=int, default=1)
    ap.add_argument("--role", default="researcher",
                    help="role for agent 1 (demo override of lab.yaml role_map)")
    ap.add_argument("--force-role", action="store_true",
                    help="bypass assign_roles' supervisor guarantee — with a "
                         "1-agent swarm the machinery would otherwise force "
                         "agent 1 into the supervisor role")
    ap.add_argument("--topic", default=None)
    args = ap.parse_args()

    import agentics.orchestrator as orch_mod
    from agentics.agents.roles import Role
    from agentics.orchestrator import Orchestrator

    settings = get_settings()
    settings.max_agents = args.agents
    if args.topic:
        settings.research_topic = args.topic
    clear_stop()

    if args.role:
        orch_mod._role_map = lambda s: {1: args.role}  # noqa: SLF001 — demo override
    if args.force_role:
        chosen = Role(args.role)
        orch_mod.assign_roles = lambda n, rm: [chosen] * max(1, n)  # noqa: SLF001

    orch = Orchestrator(settings)
    print(f"=== live lab demo: {args.agents} agent(s) as {args.role!r}, "
          f"time box {args.minutes} min, topic: {settings.research_topic[:70]}…")

    runner = asyncio.create_task(orch.run_forever())
    try:
        await asyncio.sleep(args.minutes * 60)
    finally:
        print("=== time box elapsed — requesting graceful stop (owner STOP) ===")
        request_stop()
        try:
            await asyncio.wait_for(runner, timeout=args.shutdown_wait)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            runner.cancel()
            await asyncio.gather(runner, return_exceptions=True)

    # the orchestrator closed its memory on shutdown — reopen for the report
    mem = SharedMemory(settings.db_path)
    print("\n=== agents (owner view) ===")
    for a in mem.active_agents(stale_after_s=86400):
        intent = (a.get("intent") or {}).get("goal", "-")
        print(f"  {a['agent']} {a['role']:<11} cycles={a['cycle']:<3} "
              f"status={a['status']:<10} node={a['node'] or '-':<10} "
              f"intent: {str(intent)[:60]}")
    print("\n=== tasks ===")
    for k, v in sorted(mem.task_counts().items()):
        print(f"  {k}: {v}")
    print("\n=== findings (shared memory) ===")
    for f in mem.list_findings(limit=5):
        print(f"  finding #{f['id']} by {f['agent']}: {f['title'][:70]}")
    print("\n=== hypotheses & verdicts (P16 — from real runs) ===")
    for h in mem.list_hypotheses(limit=8):
        c = h.get("criterion") or {}
        print(f"  h#{h['id']} [{h['status']}] {h['claim'][:64]!r} "
              f"criterion: {c.get('metric')} {c.get('op')} {c.get('target')}")
    print("\n=== evidence (exit-code verdicts, shared swarm-wide) ===")
    for e in mem.latest_evidence(limit=8):
        print(f"  [{e['verdict']}] exit={e['exit_code']} by {e['agent']}: "
              f"{e['cmd'][:70]}")
    print("\n=== journal timeline (last 30) ===")
    for ev in mem.recent_journal(limit=30):
        detail = ev.get("detail")
        detail = (json.dumps(detail, ensure_ascii=False)[:70]
                  if isinstance(detail, dict) else str(detail)[:70])
        print(f"  [{ev['agent']}] {ev['event']:<22} {detail}")
    mem.close()

    repo_path = orch.local.repo.path
    log = subprocess.run(
        ["git", "-C", str(repo_path), "log", "--oneline", "-6", "main"],
        capture_output=True, text=True).stdout.strip()
    print(f"\n=== main history on the local bare repo ===\n{log or '(no commits yet)'}")
    print("\n=== demo complete ===")


if __name__ == "__main__":
    asyncio.run(main())
