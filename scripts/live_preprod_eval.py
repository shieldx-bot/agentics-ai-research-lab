"""Pre-production live eval — FULL orchestrator loop on the REAL credentials.

The production candidate: every LLM edge goes through the real Hyperbrowser
token pool (secrets.yaml), every verdict is a real exit code, every hypothesis
that survives lands in the real bare repo. Nothing is stubbed — the only
differences from `agentics run` are the time box and the single-agent role
override needed to fit the eval inside one sandbox call.

After the time box the script prints the honest evaluation bundle:
  - owner state view (agents, tasks, findings, hypotheses, evidence)
  - Co-Scientist tournament standings (Elo, matches, meta-review feedback)
  - fan-out events (parallel + same-view) from the journal
  - swarm_eval report restricted to THIS run's window (since_ts marker)
  - P16 verdict decided-rate inside the window (the honest number)
  - citation distinct-domain ratio inside the window
Everything is also dumped to JSON for the written report.

Usage:
    python scripts/live_preprod_eval.py [--minutes 7] [--agents 1]
        [--role researcher] [--topic "…"] [--out report.json]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from agentics.config import get_settings  # noqa: E402
from agentics.memory.shared_memory import SharedMemory  # noqa: E402
from agentics.stop import clear_stop, request_stop  # noqa: E402


def window_report(mem: SharedMemory, since_ts: float) -> dict:
    """Honest in-window numbers: swarm_eval + verdict/citation details."""
    from agentics.swarm_eval import (
        _distinct_domains,
        evaluate_swarm,
        percent_rows,
    )

    rep = evaluate_swarm(mem, since_ts=since_ts)
    rows = percent_rows(rep)

    # P16 verdict outcome split inside the window (raw, no smoothing)
    ev = [e for e in mem.latest_evidence(limit=500) if e.get("ts", 0) >= since_ts]
    verdicts: dict[str, int] = {}
    for e in ev:
        verdicts[e["verdict"]] = verdicts.get(e["verdict"], 0) + 1
    decided = verdicts.get("confirmed", 0) + verdicts.get("refuted", 0)

    # citation distinct-domain ratio inside the window — reuses the EXACT
    # swarm_eval helper so the detail number is the metric's own math
    findings = [f for f in mem.list_findings(limit=2000) if f.get("ts", 0) >= since_ts]
    cites = [c for f in findings for c in (f.get("citations") or [])]
    hosts = _distinct_domains(cites)

    # fan-out activity inside the window
    evs = mem.journal_events(since_ts)
    fanout = [e for e in evs if str(e.get("event", "")).startswith("research_fanout")]

    # tournament activity inside the window
    matches = [m for m in mem.recent_tournament_matches(limit=200)
               if m.get("ts", 0) >= since_ts]

    return {
        "window_since_ts": since_ts,
        "swarm_eval_rows": rows,
        "verdicts": verdicts,
        "decided_total": decided,
        "evidence_total": len(ev),
        "citations_total": len(cites),
        "distinct_hosts": sorted(hosts),
        "distinct_host_count": len(hosts),
        "fanout_events": [{"event": e.get("event"), "detail": e.get("detail")}
                          for e in fanout],
        "tournament_matches": matches,
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=int, default=7)
    ap.add_argument("--shutdown-wait", type=int, default=60)
    ap.add_argument("--agents", type=int, default=1)
    ap.add_argument("--role", default="researcher")
    ap.add_argument("--force-role", action="store_true")
    ap.add_argument("--topic", default=None)
    ap.add_argument("--out", default=None, help="JSON report path")
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
        orch_mod._role_map = lambda s: {1: args.role}  # noqa: SLF001 — eval override
    if args.force_role:
        chosen = Role(args.role)
        orch_mod.assign_roles = lambda n, rm: [chosen] * max(1, n)  # noqa: SLF001

    # window marker BEFORE the swarm touches anything — clean metric window
    mem0 = SharedMemory(settings.db_path)
    row = mem0._conn.execute("SELECT MAX(rowid) FROM journal").fetchone()
    since_ts = time.time()
    print(f"=== pre-prod eval window: since_ts={since_ts:.2f} "
          f"(journal rowid<= {row[0]}) ===")
    print(f"=== topic: {settings.research_topic} ===")
    print(f"=== HB token pool: {len(settings.hyperbrowser_api_keys)} key(s) ===")
    mem0.close()

    orch = Orchestrator(settings)
    t0 = time.monotonic()
    runner = asyncio.create_task(orch.run_forever())
    try:
        await asyncio.sleep(args.minutes * 60)
    finally:
        print(f"=== time box elapsed ({time.monotonic() - t0:.0f}s) — graceful stop ===")
        request_stop()
        try:
            await asyncio.wait_for(runner, timeout=args.shutdown_wait)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            runner.cancel()
            await asyncio.gather(runner, return_exceptions=True)

    # orchestrator closed its memory on shutdown — reopen for the report
    mem = SharedMemory(settings.db_path)
    out = {
        "topic": settings.research_topic,
        "minutes": args.minutes,
        "agents": args.agents,
        "role": args.role,
        "hb_pool_size": len(settings.hyperbrowser_api_keys),
        "window_since_ts": since_ts,
        "wall_seconds": time.monotonic() - t0,
        "agents_view": mem.active_agents(stale_after_s=86400),
        "task_counts": mem.task_counts(),
        "findings": [dict(f) for f in mem.list_findings(limit=20)],
        "hypotheses": [dict(h) for h in mem.list_hypotheses(limit=20)],
        "evidence": [dict(e) for e in mem.latest_evidence(limit=50)],
        "tournament": {
            "pool": [dict(h) for h in mem.tournament_pool(limit=20)],
            "top_by_elo": [dict(h) for h in mem.top_by_elo(limit=5)],
            "matches": [dict(m) for m in mem.recent_tournament_matches(limit=50)],
            "feedback": ([dict(mem.latest_tournament_feedback())
                         if mem.latest_tournament_feedback() else {}]),
        },
        "journal": [dict(e) for e in mem.recent_journal(limit=60)],
        "window": window_report(mem, since_ts),
    }
    mem.close()

    repo_path = orch.local.repo.path
    out["bare_repo_log"] = subprocess.run(
        ["git", "-C", str(repo_path), "log", "--oneline", "-8", "main"],
        capture_output=True, text=True).stdout.strip()

    payload = json.dumps(out, indent=2, ensure_ascii=False, default=str)
    if args.out:
        Path(args.out).write_text(payload)
        print(f"=== JSON report saved: {args.out} ===")

    print("\n=== agents (owner view) ===")
    for a in out["agents_view"]:
        intent = (a.get("intent") or {}).get("goal", "-")
        print(f"  {a['agent']} {a['role']:<11} cycles={a['cycle']:<3} "
              f"status={a['status']:<10} node={a['node'] or '-':<10} "
              f"intent: {str(intent)[:60]}")
    print("\n=== tasks ===")
    for k, v in sorted(out["task_counts"].items()):
        print(f"  {k}: {v}")
    print("\n=== findings ===")
    for f in out["findings"]:
        print(f"  #{f['id']} by {f['agent']}: {str(f.get('title'))[:70]}")
    print("\n=== hypotheses ===")
    for h in out["hypotheses"]:
        print(f"  h#{h['id']} [{h['status']}] elo={h.get('elo')} "
              f"played={h.get('matches_played')} :: {str(h.get('claim'))[:72]!r}")
    print("\n=== evidence (exit-code verdicts) ===")
    for e in out["evidence"]:
        print(f"  [{e['verdict']}] exit={e['exit_code']} by {e['agent']}: "
              f"{str(e.get('cmd'))[:70]}")
    print("\n=== tournament standings (Elo) ===")
    for h in out["tournament"]["top_by_elo"]:
        print(f"  h#{h['id']} elo={h.get('elo')} played={h.get('matches_played')} "
              f":: {str(h.get('claim'))[:60]!r}")
    print(f"  matches in window: {len(out['window']['tournament_matches'])}")
    print("\n=== fan-out events ===")
    for f in out["window"]["fanout_events"]:
        print(f"  {f['event']}: {json.dumps(f['detail'], ensure_ascii=False)[:90]}")
    print("\n=== swarm_eval (in-window) ===")
    for r in out["window"]["swarm_eval_rows"]:
        imp = "*" if r.get("important") else " "
        print(f"  {imp} {r['group']:<18} {r['metric']:<28} "
              f"raw={str(r['raw']):<12} pct={r['pct']}")
    w = out["window"]
    print("\n=== honest headline numbers ===")
    print(f"  evidence rows: {w['evidence_total']}, verdict split: {w['verdicts']}, "
          f"decided: {w['decided_total']}")
    print(f"  citations: {w['citations_total']}, distinct hosts: "
          f"{w['distinct_host_count']}")
    print(f"\n=== main history on the local bare repo ===\n{out['bare_repo_log'] or '(none)'}")
    print("\n=== eval complete ===")


if __name__ == "__main__":
    asyncio.run(main())
