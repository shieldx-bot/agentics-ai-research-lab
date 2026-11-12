"""Agentics CLI — run / stop / status / test-llm / assign-roles."""
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
