"""Agentics CLI — run / stop / status / metrics / test-llm / assign-roles."""
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
    console.print(f"[bold green]Agentics v{__version__}[/] — {len(orch.pool)} accounts, "
                  f"repo {settings.github_repo}")
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
    """Show agents, accounts, claims from local memory (works offline)."""
    settings = _settings()
    setup_logging("ERROR")
    from agentics.orchestrator import Orchestrator

    orch = Orchestrator.__new__(Orchestrator)  # avoid GitHub connection
    from agentics.memory.shared_memory import SharedMemory
    from agentics.llm.pool import AccountPool
    from agentics.memory.vector_store import VectorStore

    settings.ensure_dirs()
    memory = SharedMemory(settings.db_path)
    pool = AccountPool(settings)
    table = Table(title="Agents (heartbeat)")
    for col in ("agent", "role", "cycle", "status", "last seen"):
        table.add_column(col)
    import time as _t
    for a in memory.active_agents(stale_after_s=86400):
        table.add_row(a["agent"], a["role"], str(a["cycle"]), a["status"],
                      _t.strftime("%H:%M:%S", _t.localtime(a["last_ts"])))
    console.print(table)
    claims = memory.active_claims()
    if claims:
        console.print("[bold]Active claims:[/]")
        for issue_no, c in claims.items():
            console.print(f"  issue #{issue_no} -> {c.agent} ({c.branch})")
    console.print(f"[bold]Accounts loaded:[/] {len(pool)}")
    for line in pool.status_lines():
        console.print(f"  {line}")
    memory.close()


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
    query: str = typer.Option("What is network function virtualization? One paragraph.",
                              "--query"),
) -> None:
    """Smoke-test one cookie account against Perplexity."""
    settings = _settings()
    setup_logging("INFO")
    from agentics.llm.pool import AccountPool

    pool = AccountPool(settings)
    if not pool.accounts:
        console.print("[red]No accounts found in accounts/[/]")
        raise typer.Exit(1)
    acc = pool.accounts[min(account, len(pool.accounts)) - 1]
    console.print(f"Testing [bold]{acc.name}[/] ({len(acc.cookies)} cookies)…")
    client = pool.client_for(acc)

    async def _run():
        resp = await client.ask(query, mode="academic")
        console.print(f"elapsed: {resp.elapsed_s}s | citations: {len(resp.citations)}")
        console.print(resp.answer[:1200])
        if resp.citations:
            console.print("[dim]citations:[/]")
            for c in resp.citations[:5]:
                console.print(f"  - {c}")

    asyncio.run(_run())
    pool.mark_ok(acc)


@app.command()
def assign_roles() -> None:
    """Print the role assignment for current accounts (dry-run)."""
    settings = _settings()
    from agentics.llm.pool import AccountPool
    from agentics.orchestrator import _role_map
    from agentics.agents.roles import assign_roles

    pool = AccountPool(settings)
    roles = assign_roles(min(len(pool), settings.max_agents), _role_map(settings))
    for acc, role in zip(pool.accounts, roles):
        console.print(f"{acc.name:<14} -> [bold cyan]{role.value}[/]")


if __name__ == "__main__":
    app()
