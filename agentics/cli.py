"""Agentics CLI — run / stop / status / dashboard / task / test-llm.

Local-only mode: the lab runs entirely on this machine — no GITHUB_TOKEN
needed, nothing leaves the box except the LLM provider traffic.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from agentics import __version__
from agentics.config import get_settings, load_yaml_overrides, PROJECT_ROOT
from agentics.logging_setup import setup_logging
from agentics.stop import request_stop, should_stop

app = typer.Typer(help="Agentics AI Research Lab — multi-agent research swarm", no_args_is_help=True)
console = Console()


def _settings():
    return load_yaml_overrides(get_settings())


@app.command()
def run(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="DEBUG logging"),
) -> None:
    """Start the infinite multi-agent research loop (Ctrl+C to stop)."""
    settings = _settings()
    setup_logging("DEBUG" if verbose else "INFO", log_dir=PROJECT_ROOT / "logs")
    from agentics.orchestrator import Orchestrator
    from agentics.stop import install_signal_handlers

    orch = Orchestrator(settings)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    install_signal_handlers(loop)
    console.print(f"[bold green]Agentics v{__version__}[/] — {len(orch.host)} browser-session "
                  f"identities (perplexity: {len(orch.host.pools['perplexity'])}, "
                  f"qwen: {len(orch.host.pools['qwen'])}) — [bold]local-only mode[/] "
                  f"(origin: {settings.local_repo_path})")
    console.print("Owner visibility: [bold]python -m agentics.cli dashboard[/] "
                  "→ http://127.0.0.1:8080")
    if should_stop():
        from agentics.stop import clear_stop
        clear_stop()
    try:
        loop.run_until_complete(orch.run_forever())
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()
        console.print("[bold]Lab stopped.[/]")


@app.command()
def stop() -> None:
    """Owner: request a graceful stop of all agent threads (STOP file)."""
    request_stop()
    console.print("[bold yellow]STOP file created[/] — agents finish current node, "
                  "release claims and exit.")


@app.command()
def status() -> None:
    """Show agents (node + intent), tasks, claims from local memory (works offline)."""
    settings = _settings()
    setup_logging("ERROR")
    from agentics.memory.shared_memory import SharedMemory
    from agentics.llm.session_host import BrowserSessionHost

    settings.ensure_dirs()
    memory = SharedMemory(settings.db_path)
    host = BrowserSessionHost.from_settings(settings)
    table = Table(title="Agents (live state)")
    for col in ("agent", "role", "cycle", "status", "node", "task", "intent (goal)"):
        table.add_column(col, overflow="fold")
    import time as _t
    for a in memory.active_agents(stale_after_s=86400):
        goal = (a["intent"] or {}).get("goal", "")
        table.add_row(
            a["agent"], a["role"], str(a["cycle"]), a["status"], a["node"] or "-",
            f"#{a['task_id']} {a['task_title'][:36]}" if a.get("task_id") else "-",
            goal[:60],
        )
    console.print(table)
    counts = memory.task_counts()
    if counts:
        console.print("[bold]Task counts:[/] " + ", ".join(
            f"{k}={v}" for k, v in sorted(counts.items())))
    claims = memory.active_claims()
    if claims:
        console.print("[bold]Active claims:[/]")
        for task_id, c in claims.items():
            console.print(f"  task #{task_id} -> {c.agent} ({c.branch})")
    console.print(f"[bold]Browser-session identities loaded:[/] {len(host)} "
                  f"(perplexity: {len(host.pools['perplexity'])}, qwen: {len(host.pools['qwen'])})")
    memory.close()


@app.command("dashboard")
def dashboard(
    port: int = typer.Option(None, "--port", help="HTTP port (default DASHBOARD_PORT or 8080)"),
    host: str = typer.Option("127.0.0.1", "--host"),
) -> None:
    """Owner: open the live web dashboard (agents, intent, tasks, MRs, journal)."""
    settings = _settings()
    setup_logging("ERROR")
    try:
        import uvicorn
    except ImportError:
        console.print("[red]Dashboard cần fastapi + uvicorn:[/] pip install fastapi uvicorn")
        raise typer.Exit(1)
    from agentics.dashboard.app import create_app
    from agentics.memory.shared_memory import SharedMemory

    settings.ensure_dirs()
    memory = SharedMemory(settings.db_path)
    real_port = port or settings.dashboard_port
    console.print(f"[bold]Dashboard:[/] http://{host}:{real_port} — Ctrl+C để thoát")
    uvicorn.run(create_app(memory, settings), host=host, port=real_port, log_level="warning")


@app.command("task")
def task(
    title: list[str] = typer.Argument(None, help="task title (owner adds work)"),
    body: str = typer.Option("", "--body", "-b", help="task description"),
    category: str = typer.Option("research", "--category", "-c",
                                 help="research | code | docs | bugfix | other"),
    list_all: bool = typer.Option(False, "--list", "-l", help="list recent tasks"),
) -> None:
    """Owner: add a task to the local backlog (or --list to view it)."""
    settings = _settings()
    setup_logging("ERROR")
    from agentics.memory.shared_memory import SharedMemory

    settings.ensure_dirs()
    memory = SharedMemory(settings.db_path)
    if list_all or not title:
        table = Table(title="Tasks")
        for col in ("#", "status", "category", "claimed_by", "title"):
            table.add_column(col, overflow="fold")
        for t in memory.list_tasks(limit=25):
            table.add_row(str(t["id"]), t["status"], t["category"],
                          t["claimed_by"] or "-", t["title"])
        console.print(table)
        memory.close()
        return
    full_title = " ".join(title).strip()
    task_id = memory.create_task(full_title, body=body, category=category, agent="owner")
    console.print(f"[bold green]task #{task_id} created[/] — agents will pick it up "
                  "in their next claim cycle.")
    memory.close()


@app.command("allow-protected")
def allow_protected(
    task_id: int = typer.Argument(..., help="task id allowed to touch agent code"),
    revoke: bool = typer.Option(False, "--revoke", help="remove the exemption instead"),
) -> None:
    """Owner: grant (or --revoke) the ONLY path for a task to modify lab code.

    Protected paths (agentics/, configs/, accounts/, scripts/, .github/) are
    hard-blocked for agents. With an exemption, a task still may never touch
    accounts/ or configs/secrets.yaml.
    """
    settings = _settings()
    setup_logging("ERROR")
    from agentics.memory.shared_memory import SharedMemory

    settings.ensure_dirs()
    memory = SharedMemory(settings.db_path)
    if memory.get_task(task_id) is None:
        console.print(f"[red]task #{task_id} does not exist[/]")
        raise typer.Exit(1)
    memory.set_task_exempt(task_id, not revoke, agent="owner")
    state = "EXEMPT (may touch agent code, except credentials)" if not revoke else "PROTECTED again"
    console.print(f"[bold]task #{task_id}[/] → {state}")
    memory.close()


@app.command("events")
def events(
    tail: int = typer.Option(40, "--tail", "-n", help="how many events to show"),
    agent: str = typer.Option("", "--agent", "-a", help="only one agent's events"),
) -> None:
    """Owner: read the activity journal (what/why each agent did) in a terminal.

    The same feed the dashboard shows — useful over SSH when no browser is at
    hand. Oldest first, so it reads like a story; add --follow-like polling by
    re-running with a bigger --tail.
    """
    settings = _settings()
    setup_logging("ERROR")
    from agentics.memory.shared_memory import SharedMemory

    settings.ensure_dirs()
    memory = SharedMemory(settings.db_path)
    rows = memory.recent_journal(limit=max(1, tail), agent=agent or None)
    if not rows:
        console.print("[dim]journal trống — chưa có agent nào chạy.[/]")
        memory.close()
        return
    table = Table(title=f"Journal ({len(rows)} sự kiện, mới nhất cuối)")
    for col in ("time", "agent", "event", "detail"):
        table.add_column(col, overflow="fold")
    import time as _t
    for r in rows:
        table.add_row(
            _t.strftime("%H:%M:%S", _t.localtime(r["ts"])),
            r["agent"], r["event"], r["detail"],
        )
    console.print(table)
    memory.close()


@app.command("explain")
def explain(
    agent_id: str = typer.Argument(..., help="agent id, e.g. a03"),
) -> None:
    """Owner: WHY-trace one agent — current intent, claims, MRs, last actions.

    Answers "a03 đang làm gì, vì sao?" without opening the dashboard: live
    intent from the heartbeat, held claims, MRs authored, and the agent's
    latest journal entries in order.
    """
    settings = _settings()
    setup_logging("ERROR")
    from agentics.memory.shared_memory import SharedMemory

    settings.ensure_dirs()
    memory = SharedMemory(settings.db_path)
    live = {a["agent"]: a for a in memory.active_agents(stale_after_s=86400)}
    a = live.get(agent_id)
    if a:
        intent = a.get("intent") or {}
        console.print(f"[bold]{agent_id}[/] role={a['role']} node={a['node'] or '-'} "
                      f"status={a['status']} cycle={a['cycle']}")
        if a.get("task_id"):
            console.print(f"  task #{a['task_id']}: {a['task_title']}")
        console.print(f"  [bold]mục đích:[/] {intent.get('goal', '(chưa công bố)')}")
        if intent.get("why"):
            console.print(f"  [bold]vì sao:[/] {intent['why']}")
        for s in (intent.get("steps") or [])[:8]:
            console.print(f"    - {s}")
    else:
        console.print(f"[dim]{agent_id} không có heartbeat gần đây.[/]")
    claims = [c for tid, c in memory.active_claims().items() if c.agent == agent_id]
    for c in claims:
        task = memory.get_task(c.issue)
        title = task["title"][:60] if task else "?"
        console.print(f"  claim: task #{c.issue} ({title}) — branch {c.branch}")
    mrs = memory.reviews_by_author(agent_id)
    for mr in mrs[-8:]:
        console.print(f"  MR #{mr['id']} [{mr['status']}] task #{mr['task_id']} "
                      f"— {(mr.get('verdict_comment') or '')[:80]}")
    journal = memory.recent_journal(limit=15, agent=agent_id)
    if journal:
        console.print("[bold]hành động gần nhất (cũ nhất trước):[/]")
        import time as _t
        for r in reversed(journal):
            console.print(f"  {_t.strftime('%H:%M:%S', _t.localtime(r['ts']))} "
                          f"[cyan]{r['event']}[/] {r['detail']}")
    memory.close()


@app.command()
def verify(
    refresh: bool = typer.Option(False, "--refresh",
                                 help="accept the current tree as the new baseline"),
) -> None:
    """Owner: has the RUNNING lab's own code been changed out-of-band?

    Complements the P10 protected-paths guard: its three layers stop agent
    changesets into the engine; this detects ANY drift of the runtime tree
    the lab itself runs from (docs/LOCAL_MODE.md, hypothesis-1 layer 4).
    """
    settings = _settings()
    setup_logging("ERROR")
    from agentics.sandbox.integrity import IntegrityGuard, verify as verify_tree

    guard = IntegrityGuard(PROJECT_ROOT, settings.integrity_path)
    if refresh:
        count = guard.refresh_baseline()
        console.print(f"[bold green]baseline refreshed[/] — {count} runtime file(s) hashed.")
        return
    baseline = guard.load()
    if baseline is None:
        guard.ensure_baseline()
        console.print("[bold green]baseline created[/] (first run) — runtime tree hashed.")
        return
    drift = verify_tree(PROJECT_ROOT, baseline)
    if not drift:
        console.print("[bold green]OK[/] — the running lab matches its integrity baseline.")
        return
    console.print(f"[bold red]DRIFT:[/] {len(drift)} runtime file(s) differ from baseline:")
    for d in drift[:20]:
        console.print(f"  {d['kind']:<8} {d['path']}")
    console.print("[dim]If these changes are yours (or reviewed), accept them with "
                  "`agentics verify --refresh`.[/]")
    raise typer.Exit(1)


@app.command()
def metrics(
    days: int = typer.Option(0, "--days", help="only the last N days (0 = all time)"),
) -> None:
    """How is the lab itself evolving? Outcomes of agent work from the shared traces (offline)."""
    import time as _t
    settings = _settings()
    setup_logging("ERROR")
    from agentics.memory.shared_memory import SharedMemory

    settings.ensure_dirs()
    memory = SharedMemory(settings.db_path)
    since = _t.time() - days * 86400 if days > 0 else 0.0
    st = memory.trace_stats(since)
    rate = "n/a" if st["merge_rate"] is None else f"{st['merge_rate']:.0%}"
    console.print(f"[bold]Outcomes:[/] {st['by_outcome']}   [bold]merge rate:[/] {rate} "
                  f"(of {st['judged']} judged: merged / rejected / failed)")
    table = Table(title="By category")
    for col in ("category", "merged", "rejected", "failed", "yield", "pending/approved"):
        table.add_column(col)
    for cat, o in sorted(st["by_category"].items()):
        table.add_row(cat, str(o.get("merged", 0)), str(o.get("rejected", 0)),
                      str(o.get("failed", 0)), str(o.get("yield", 0)),
                      str(o.get("pending", 0) + o.get("approved", 0)))
    console.print(table)
    memory.close()


@app.command("memory-stats")
def memory_stats() -> None:
    """LifeMem memory health: traces, workflow clusters, distilled skills."""
    settings = _settings()
    setup_logging("ERROR")
    from agentics.memory.shared_memory import SharedMemory

    settings.ensure_dirs()
    memory = SharedMemory(settings.db_path)
    st = memory.memory_stats()
    console.print(
        f"[bold]Traces:[/] {st['traces']} (clustered: {st['clustered']})   "
        f"[bold]Clusters:[/] {st['clusters']}   "
        f"[bold]Skills:[/] {st['active_skills']} active / "
        f"{st['superseded_skills']} superseded"
    )
    if st["top_skills"]:
        table = Table(title="Most-reused skills")
        for col in ("uses", "category", "skill"):
            table.add_column(col)
        for s in st["top_skills"]:
            table.add_row(str(s["uses"]), s["category"], s["text"])
        console.print(table)
    memory.close()


@app.command()
def test_llm(
    account: int = typer.Option(1, "--account", help="account index (1-based)"),
    provider: str = typer.Option("perplexity", "--provider", help="perplexity | qwen"),
    query: str = typer.Option("What is network function virtualization? One paragraph.",
                              "--query"),
) -> None:
    """Smoke-test one browser-session account (Perplexity or Qwen)."""
    settings = _settings()
    setup_logging("INFO")
    from agentics.llm.session_host import BrowserSessionHost

    host = BrowserSessionHost.from_settings(settings)
    prov = provider.lower()
    if prov not in host.pools:
        console.print(f"[red]unknown provider '{provider}' (use perplexity|qwen)[/]")
        raise typer.Exit(1)
    pool = host.pools[prov]
    if not pool.accounts:
        console.print(f"[red]No {prov} accounts found in accounts/[/]")
        raise typer.Exit(1)
    acc = pool.accounts[min(account, len(pool.accounts)) - 1]
    console.print(f"Testing [bold]{acc.name}[/] ({prov}, "
                  f"{len(acc.cookies)} cookies, model={getattr(acc, 'model', '-')})…")
    client = host._client_for(prov, acc)

    async def _run():
        try:
            resp = await client.ask(query, mode="academic")
            console.print(f"elapsed: {resp.elapsed_s}s | citations: {len(resp.citations)}")
            console.print(resp.answer[:1200])
            if resp.citations:
                console.print("[dim]citations:[/]")
                for c in resp.citations[:5]:
                    console.print(f"  - {c}")
        finally:
            closer = getattr(client, "close", None)   # hyperbrowser clients
            if closer is not None:
                try:
                    await closer()                    # stop the cloud session
                except Exception:  # noqa: BLE001
                    pass

    asyncio.run(_run())
    pool.mark_ok(acc)


@app.command()
def assign_roles() -> None:
    """Print the role assignment for current accounts (dry-run)."""
    settings = _settings()
    from agentics.llm.session_host import BrowserSessionHost
    from agentics.orchestrator import _role_map
    from agentics.agents.roles import assign_roles

    host = BrowserSessionHost.from_settings(settings)
    roles = assign_roles(min(len(host), settings.max_agents), _role_map(settings))
    for i, role in zip(range(len(host)), roles):
        acc = host.pools["perplexity"].accounts[i] if i < len(host.pools["perplexity"]) \
            else host.pools["qwen"].accounts[i]
        console.print(f"{acc.name:<14} -> [bold cyan]{role.value}[/]")


if __name__ == "__main__":
    app()
