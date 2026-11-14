"""Agentics CLI — run / stop / status / dashboard / task / journal / test-llm.

Local-only mode: the lab runs entirely on this machine — no GITHUB_TOKEN
needed, nothing leaves the box except the LLM provider traffic.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

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


@app.command("repair")
def repair() -> None:
    """Owner: run the maintenance pass NOW instead of waiting for a scan cycle.

    Calls the two existing local-mode services (no new machinery): the
    TaskSource's orphan repair (tasks stuck in claimed/in_review with no
    lease/MR go back to the open pool) and the ClaimPolicy's stale-lease
    sweep (leases of dead/expired agents are released)."""
    settings = _settings()
    setup_logging("ERROR")
    from agentics.memory.shared_memory import SharedMemory
    from agentics.local_ops import LocalClaimPolicy, TaskSource

    settings.ensure_dirs()
    memory = SharedMemory(settings.db_path)
    try:
        swept = LocalClaimPolicy(memory, ttl_minutes=30).sweep_stale()
        repaired = TaskSource(memory, settings).repair_orphan_tasks()
        if swept:
            console.print(f"[bold]Swept stale leases[/] -> tasks re-opened: "
                          + ", ".join(f"#{t}" for t in swept))
        else:
            console.print("[bold]Swept stale leases:[/] none")
        if repaired:
            console.print(f"[bold]Repaired orphan tasks[/] -> re-opened: "
                          + ", ".join(f"#{t}" for t in repaired))
        else:
            console.print("[bold]Repaired orphan tasks:[/] none")
        memory.journal("owner", "manual_maintenance", {
            "swept": swept, "repaired": repaired})
    finally:
        memory.close()


@app.command("reset-memory")
def reset_memory_cmd(
    yes: bool = typer.Option(False, "--yes", "-y",
                             help="bỏ qua câu hỏi xác nhận"),
    force: bool = typer.Option(False, "--force",
                               help="cho phép reset dù còn agent có heartbeat mới"),
    keep_repo: bool = typer.Option(False, "--keep-repo",
                                   help="giữ data/lab.git (branch/MR của topic cũ)"),
    dry_run: bool = typer.Option(False, "--dry-run",
                                 help="chỉ liệt kê thứ sẽ bị xoá, không xoá thật"),
) -> None:
    """Owner: xoá TOÀN BỘ memory của lab khi đổi research topic mới.

    Xoá: shared memory DB (tasks/findings/panels/journal/role prompts),
    vector memory (recall finding), personal memory (bài học mỗi agent),
    workspace các agent và data/lab.git (branch/MR cũ) — trừ khi --keep-repo.
    Giữ nguyên: code lab, baseline integrity, accounts/, configs/, .env, logs.
    Lab khởi động lại sẽ tự seed lab.git và tạo schema mới."""
    setup_logging("ERROR")
    from agentics.memory.reset import ResetBlocked, reset_memory, running_agents

    settings = _settings()
    try:
        preview = reset_memory(settings, keep_repo=keep_repo, dry_run=True)
    except ResetBlocked as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1)

    if preview.removed:
        table = Table(title="Memory sẽ được xoá (đổi topic mới)" +
                          (" — DRY RUN" if dry_run else ""))
        for col in ("thành phần", "dung lượng"):
            table.add_column(col)
        for label, size in preview.removed:
            human = f"{size / 1024:.1f} KB" if size >= 1024 else f"{size} B"
            table.add_row(label, human)
        console.print(table)
    else:
        console.print("[yellow]Không có memory nào để xoá (lab còn nguyên).[/]")

    if dry_run:
        console.print("[dim]dry-run: chưa xoá gì. Bỏ --dry-run để xoá thật.[/]")
        return

    live = running_agents(settings)
    if live and not force:
        names = ", ".join(f"{a['agent']}({a['role']})" for a in live)
        console.print(f"[red]Có {len(live)} agent đang chạy ({names}) — "
                      f"chạy `agentics stop` trước, hoặc dùng --force.[/]")
        raise typer.Exit(1)
    if not yes and not typer.confirm("Xoá toàn bộ memory ở trên?"):
        console.print("[dim]Đã huỷ, không xoá gì.[/]")
        return

    try:
        report = reset_memory(settings, keep_repo=keep_repo)
    except ResetBlocked as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1)
    for label, _size in report.removed:
        console.print(f"  [red]xoá[/] {label}")
    for kept in report.kept:
        console.print(f"  [green]giữ[/] {kept}")
    console.print(f"[bold green]Memory đã reset[/] — giải phóng "
                  f"{report.freed_human}. Đổi RESEARCH_TOPIC trong .env/"
                  f"configs/lab.yaml rồi chạy lại `python -m agentics.cli run` "
                  f"— lab tự seed lại lab.git và schema mới.")


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
    follow: bool = typer.Option(False, "--follow", "-f", help="stream new entries live (like tail -f)"),
) -> None:
    """Owner: read the activity journal (what/why each agent did) in a terminal.

    The same feed the dashboard shows — useful over SSH when no browser is at
    hand. Oldest first, so it reads like a story; --follow streams new entries
    as they land (Ctrl+C to stop).

        python -m agentics.cli events -f           # live stream
        python -m agentics.cli events -a a02 -n 5  # one agent, last 5
    """
    import time as _t

    settings = _settings()
    setup_logging("ERROR")
    from agentics.memory.shared_memory import SharedMemory

    settings.ensure_dirs()
    memory = SharedMemory(settings.db_path)

    def _row(r: dict) -> str:
        return (
            _t.strftime("%H:%M:%S", _t.localtime(r["ts"])),
            r["agent"], r["event"], r["detail"],
        )

    try:
        rows = memory.recent_journal(limit=max(1, tail), agent=agent or None)
        if not rows and not follow:
            console.print("[dim]journal trống — chưa có agent nào chạy.[/]")
            return
        table = Table(title=f"Journal ({len(rows)} sự kiện, mới nhất cuối)")
        for col in ("time", "agent", "event", "detail"):
            table.add_column(col, overflow="fold")
        for r in rows:
            table.add_row(*_row(r))
        console.print(table)
        if not follow:
            return
        last_id = rows[-1]["id"] if rows else 0
        console.print("[dim]— streaming new journal entries, Ctrl+C to stop —[/]")
        while True:
            fresh = [r for r in memory.recent_journal(limit=200, agent=agent or None)
                     if r["id"] > last_id]
            for r in sorted(fresh, key=lambda x: x["id"]):
                ts, ag, ev, detail = _row(r)
                console.print(
                    f"[dim]{ts}[/] [magenta]{ag:<10}[/] [cyan]{ev:<16}[/] {detail}"
                )
                last_id = r["id"]
            _t.sleep(1.0)
    except KeyboardInterrupt:
        console.print("[dim]journal closed[/]")
    finally:
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


@app.command("swarm-eval")
def swarm_eval(
    days: int = typer.Option(0, "--days", help="chỉ đánh giá N ngày gần nhất (0 = toàn bộ)"),
    as_json: bool = typer.Option(False, "--json", help="in toàn bộ report dạng JSON"),
    as_percent: bool = typer.Option(False, "--percent", help="bảng % tổng hợp — mọi chỉ số quy về phần trăm cho dễ nắm"),
    no_save: bool = typer.Option(False, "--no-save", help="không lưu snapshot vào DB"),
) -> None:
    """Đánh giá bầy đàn AS A SWARM — quỹ đạo, cộng tác, hội tụ, chịu lỗi, chi phí.

    Tổng hợp 5 nhóm chỉ số swarm-intelligence từ chính vết tích của lab
    (journal, tasks, MR, panel P19, ledger LLM): hiệu quả phối hợp & nhiệm
    vụ, hành vi hội tụ, quỹ đạo & chuyển giao, khả năng chịu lỗi & mở rộng
    (độ co giãn S qua các snapshot), quản trị & chi phí. Không dùng LLM,
    chỉ đọc — mỗi lần chạy lưu 1 snapshot để so sánh giữa các quy mô đàn."""
    import time as _t
    settings = _settings()
    setup_logging("ERROR")
    from agentics.memory.shared_memory import SharedMemory
    from agentics.swarm_eval import (attach_scalability, evaluate_swarm,
                                     percent_rows, save_snapshot)

    settings.ensure_dirs()
    memory = SharedMemory(settings.db_path)
    try:
        since = _t.time() - days * 86400 if days > 0 else 0.0
        report = evaluate_swarm(memory, since_ts=since)
        report = attach_scalability(memory, report)
        if not no_save:
            snap_id = save_snapshot(memory, report)
            report["snapshot_id"] = snap_id
        if as_json:
            if as_percent:
                report["percent_rows"] = percent_rows(report)
            console.print_json(json.dumps(report, ensure_ascii=False))
            return
        if as_percent:
            _print_percent_rows(percent_rows(report))
            return
        _print_swarm_eval(report, memory)
    finally:
        memory.close()


@app.command("gepa-evolve")
def gepa_evolve(
    role: str = typer.Argument(..., help="role cần tiến hoá mission (vd: coder)"),
    category: str = typer.Argument("code", help="category của (role, category)"),
    n: int = typer.Option(3, "--n", help="số ứng viên sinh ra mỗi batch"),
    seed: int = typer.Option(0, "--seed", help="seed cho generator xác định"),
    dry_run: bool = typer.Option(False, "--dry-run", help="chỉ in, không chèn vào pool"),
) -> None:
    """GEPA — tối ưu prompt phản chiếu: pareto cha + crossover + bài học thất bại.

    Sinh ứng viên mission cho (role, category) từ chính vết tích thật của lab
    (win-rate của các arm, failure lines của chu kỳ). Ưu tiên LOCAL router
    (purpose prompt_evolve) — không tốn cookie; không có local runtime thì
    dùng generator xác định. Lab vẫn là judge: ứng viên chỉ vào challenger
    pool (origin='gepa'), thăng chức champion vẫn do P11 quyết trên exit-code."""
    import asyncio as _asyncio

    from agentics.evolution import gepa as _gepa
    from agentics.memory.shared_memory import SharedMemory

    settings = _settings()
    setup_logging("ERROR")
    settings.ensure_dirs()
    memory = SharedMemory(settings.db_path)
    try:
        texts = _asyncio.run(_gepa.reflective_candidates(
            memory, settings, role, category, n=n, seed=seed))
        if not texts:
            console.print("[yellow]Không có gì để lai tạo: chưa có champion "
                          "lẫn arm nào cho (role, category) này.[/]")
            return
        table = Table(title=f"GEPA candidates — {role}/{category} (seed={seed})")
        table.add_column("#", justify="right")
        table.add_column("độ dài", justify="right")
        table.add_column("mở đầu mission", overflow="fold")
        for i, t in enumerate(texts, 1):
            table.add_row(str(i), str(len(t)), t[:140])
        console.print(table)
        if dry_run:
            console.print("[dim]--dry-run: không chèn gì vào pool.[/]")
            return
        ids = _gepa.evolve(memory, settings, role, category, texts)
        pool = memory.prompt_variants(role, category)
        gepa_arms = sum(1 for v in pool if v.get("origin") == "gepa")
        console.print(f"[green]Đã chèn {len(ids)} ứng viên[/] "
                      f"(origin='gepa') — pool hiện {len(pool)} arm, "
                      f"{gepa_arms} từ GEPA. Champion không đổi; promotion "
                      f"vẫn do exit-code quyết (P11).")
        try:
            memory.journal("owner", "gepa_candidates",
                           {"role": role, "category": category,
                            "inserted": ids, "seed": seed})
        except Exception:  # noqa: BLE001
            pass
    finally:
        memory.close()


@app.command("kg-query")
def kg_query(
    term: str = typer.Argument(..., help="từ khoá seed (khớp entity trong đồ thị)"),
    hops: int = typer.Option(2, "--hops", help="số hop BFS (tối đa 2)"),
    term_multiple: list[str] = typer.Option([], "--term", help="thêm seed khác"),
) -> None:
    """GraphRAG-lite — lái đồ thị tri thức: neighbourhood đa hop từ seed.

    Đồ thị được khai thác từ findings/settles (kg_extract) và bị lọc TRỰC TIẾP
    bởi máy: cạnh sinh từ giả thuyết bị P16 kết luận refuted bị đánh dấu
    untrusted và (mặc định) không được phục vụ nữa."""
    settings = _settings()
    setup_logging("ERROR")
    from agentics.memory.shared_memory import SharedMemory
    settings.ensure_dirs()
    memory = SharedMemory(settings.db_path)
    try:
        terms = [term] + list(term_multiple)
        block = memory.kg_neighborhood(terms, hops=hops, limit=24)
        if not block:
            console.print("[yellow]Đồ thị trống hoặc không có entity nào khớp "
                          "seed — chưa có gì để lái.[/]")
            return
        console.print(block)
        tr = memory.kg_trust_report()
        console.print(f"[dim]đồ thị: {tr['edges']} cạnh "
                      f"({tr['trusted']} trusted / {tr['untrusted']} untrusted "
                      f"— untrusted = nguồn P16 bị refuted)[/]")
    finally:
        memory.close()


@app.command("kg-contradictions")
def kg_contradictions(
    threshold: float = typer.Option(0.55, "--threshold",
                                    help="Jaccard tối thiểu giữa 2 claim"),
) -> None:
    """Phát hiện mâu thuẫn: cặp claim đã chốt với phán quyết NGƯỢC nhau.

    Hai lớp: (1) cặp criterion trùng khớp — alarm đã journal sẵn lúc chốt;
    (2) cặp claim khác criterion nhưng chồng từ ≥ threshold — alarm NGỮ NGHĨA
    chỉ lớp này bắt được. Kèm báo cáo vệ sinh đồ thị (cạnh mất tín nhiệm)."""
    settings = _settings()
    setup_logging("ERROR")
    from agentics.memory.shared_memory import SharedMemory
    settings.ensure_dirs()
    memory = SharedMemory(settings.db_path)
    try:
        pairs = memory.kg_contradictions(threshold=threshold)
        if pairs:
            t = Table(title="Mâu thuẫn ngữ nghĩa (opposite verdicts, overlap cao)")
            t.add_column("hypothesis", justify="right")
            t.add_column("phán quyết")
            t.add_column("đối lập")
            t.add_column("phán quyết")
            t.add_column("jaccard", justify="right")
            for p in pairs:
                t.add_row(f"h#{p['a']}", p["a_status"], f"h#{p['b']}",
                          p["b_status"], f"{p['jaccard']:.2f}")
            console.print(t)
            for p in pairs[:6]:
                console.print(f"[dim]h#{p['a']}: {p['a_claim']}[/]\n"
                              f"[dim]h#{p['b']}: {p['b_claim']}[/]")
        else:
            console.print("[green]Không tìm thấy cặp mâu thuẫn ngữ nghĩa nào.[/]")
        tr = memory.kg_trust_report()
        console.print(f"[dim]vệ sinh đồ thị: {tr['trusted']} cạnh trusted, "
                      f"{tr['untrusted']} untrusted (nguồn P16 refuted) — "
                      f"không còn được phục vụ vào context propose.[/]")
    finally:
        memory.close()


@app.command("standings")
def standings(
    limit: int = typer.Option(15, "--limit", help="số hypothesis hiển thị"),
) -> None:
    """Bảng xếp hạng Elo của giải đấu giả thuyết (Co-Scientist tournament).

    Chỉ đọc — không tốn LLM. Elo là chiều CHẤT LƯỢNG ý tưởng (do các trận
    debate simulated cộng dồn); status là chiều CHÂN LÝ (do P16 chạy code
    quyết định). Hai chiều này tách bạch có chủ đích."""
    settings = _settings()
    setup_logging("ERROR")
    from agentics.memory.shared_memory import SharedMemory

    settings.ensure_dirs()
    memory = SharedMemory(settings.db_path)
    try:
        pool = memory.tournament_pool()
        matches = memory.recent_tournament_matches(limit=10)
        if not pool:
            console.print("[yellow]Chưa có hypothesis nào trong pool — "
                          "để lab chạy vài chu kỳ research/verify đã.[/]")
            return
        table = Table(title=f"Tournament standings — {len(pool)} hypothesis "
                            f"(đang xếp hạng, bỏ giả thuyết đã bị bác bỏ)")
        for col in ("#", "Elo", "trận", "status", "claim", "nguồn"):
            table.add_column(col)
        ranked = sorted(pool, key=lambda h: -(h.get("elo") or 1200))[:limit]
        for i, h in enumerate(ranked, 1):
            when = datetime.fromtimestamp(h["ts"], tz=timezone.utc).strftime(
                "%m-%d %H:%M")
            table.add_row(
                f"h#{h['id']}", f"{h.get('elo') or 1200:.0f}",
                str(h.get("matches_played") or 0), h["status"],
                (h["claim"] or "")[:60], f"{h['agent']}@{when}")
        console.print(table)
        decided = [m for m in matches if m.get("winner")]
        console.print(f"[dim]{len(matches)} trận gần nhất "
                      f"({len(decided)} có kết quả) — chi tiết: "
                      "journal event 'tournament_match'.[/]")
    finally:
        memory.close()


@app.command("tournament")
def tournament(
    matches: int = typer.Option(2, "--matches", help="số trận debate chạy"),
    evolve: int = typer.Option(0, "--evolve", help="số phép tiến hóa chạy"),
    no_meta: bool = typer.Option(False, "--no-meta", help="không tổng hợp feedback"),
) -> None:
    """Chạy trực tiếp một batch giải đấu (Co-Scientist) qua LLM thật.

    Dùng để gom trận debate ngoài chu kỳ lab (owner chạy tay): chọn cặp,
    tranh luận qua Perplexity, cập nhật Elo zero-sum, có thể tiến hóa top
    hypotheses thành thế hệ mới. Mọi ask qua P20 budget gate."""
    import asyncio

    settings = _settings()
    setup_logging("INFO")
    from agentics.memory.shared_memory import SharedMemory

    settings.ensure_dirs()
    memory = SharedMemory(settings.db_path)

    async def _run() -> None:
        from agentics.llm.session_host import BrowserSessionHost

        host = BrowserSessionHost.from_settings(settings)
        ctx = SimpleNamespace(
            agent_id="owner-cli", memory=memory, llm=host, settings=settings,
            budget=None, meta={})
        try:
            from agentics.coscientist import evolution as evo_mod
            from agentics.coscientist import metareview as meta_mod
            from agentics.coscientist import tournament as tour_mod

            goal = settings.research_topic
            results = await tour_mod.run_batch(ctx, matches, goal) if matches else []
            for r in results:
                if r.get("mode") == "blocked":
                    console.print("  [yellow]batch dừng: ask bị chặn/lỗi "
                                  "(budget hoặc session)[/]")
                elif r.get("mode") == "invalid":
                    console.print(f"  [yellow]h#{r['a']} vs h#{r['b']}: "
                                  "không parse được kết luận[/]")
                else:
                    e = r.get("elo_after") or [None, None]
                    console.print(
                        f"  h#{r['a']} vs h#{r['b']} [{r['mode']}] → "
                        f"h#{r['a'] if r['winner'] == 'a' else r['b']} thắng "
                        f"(Elo → {e[0]} / {e[1]})")
            if evolve:
                out = await evo_mod.evolve_top(ctx, goal)
                if out:
                    console.print(f"  tiến hóa: h#{out['hypothesis_id']} "
                                  f"({out['strategy']}) từ {out['parent_ids']}")
            if not no_meta and meta_mod.should_synthesize(
                    memory, settings.meta_review_window):
                note = await meta_mod.synthesize_feedback(
                    ctx, goal, settings.meta_review_window)
                if note:
                    console.print(f"  meta-review: {note[:200]}")
        finally:
            for client in list(getattr(host, "_clients", {}).values()):
                try:
                    await client.close()
                except Exception:  # noqa: BLE001
                    pass

    try:
        asyncio.run(_run())
    finally:
        memory.close()


def _fmt(v, pct: bool = False) -> str:
    """One metric cell — honest 'n/a' when the lab lacks the data."""
    if v is None:
        return "[dim]n/a (chưa đủ dữ liệu)[/]"
    if pct:
        return f"{v:.0%}"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def _print_swarm_eval(report: dict, memory) -> None:
    """Rich rendering: one table per metric group, snapshot footer."""
    g = report["groups"]
    n = report["agents_active"]
    console.print(f"[bold]Swarm evaluation[/] — {n} agent hoạt động "
                  f"({report['generated'][:16]}Z)")
    if n == 0:
        console.print("[yellow]Không có hoạt động agent nào trong cửa sổ "
                      "này — hãy để lab chạy vài chu kỳ rồi đánh giá lại.[/]")

    t1 = Table(title="1. Hiệu quả phối hợp & nhiệm vụ (Coordination & Task)")
    t1.add_column("chỉ số", overflow="fold")
    t1.add_column("giá trị")
    c = g["coordination_task"]
    t1.add_row("Tỷ lệ hoàn thành (merge/judged)", _fmt(c["task_completion_rate"], pct=True))
    t1.add_row("Hiệu quả phối hợp (panel có >=2 reviewer độc lập)",
               _fmt(c["coordination_effectiveness"], pct=True))
    t1.add_row(f"  panel đã quyết / có cross-review",
               f"{c['panels_decided']} / {c['panels_with_cross_review']}")
    t1.add_row("Độ bao phủ tìm kiếm (domain citation khác nhau)",
               _fmt(c["exploration_coverage"], pct=True))
    t1.add_row("  findings / domains", f"{c['findings']} / {c['citation_domains']}")
    t1.add_row("Độ đa dạng giải pháp (tác giả có kết quả / agent)",
               _fmt(c["solution_diversity"], pct=True))
    t1.add_row("  số nhóm công việc (category)", str(c["category_spread"]))
    t1.add_row("Tỷ lệ đóng nhiệm vụ (backlog)", _fmt(c["task_close_rate"], pct=True))
    t1.add_row("  task đang mở", str(c["tasks_open"]))
    t1.add_row("Cân bằng khối lượng (1 − share agent top)",
               _fmt(c["workload_balance"], pct=True))
    console.print(t1)

    t2 = Table(title="2. Động lực học & hội tụ (Convergence & Dynamics)")
    t2.add_column("chỉ số", overflow="fold")
    t2.add_column("giá trị")
    d = g["convergence_dynamics"]
    t2.add_row("Hội tụ đúng (tỷ lệ panel approve)", _fmt(d["convergence_correctness"], pct=True))
    t2.add_row("Số vòng vote trung bình tới phán quyết", _fmt(d["mean_panel_rounds"]))
    t2.add_row("Chu kỳ trung bình của kết quả đầu ra", _fmt(d["mean_trace_cycle"]))
    t2.add_row("Tỷ lệ defense cứu được MR", _fmt(d["defense_save_rate"], pct=True))
    t2.add_row("  panel có bảo vệ (round 2)", str(d["panels_defended"]))
    t2.add_row("Giả thuyết đã quyết (P24)",
               _fmt(d["hypothesis_resolution_rate"], pct=True))
    t2.add_row("  tổng giả thuyết", str(d["hypotheses_total"]))
    console.print(t2)

    t3 = Table(title="3. Quỹ đạo & luồng thông tin (Trajectory & Handoff)")
    t3.add_column("chỉ số", overflow="fold")
    t3.add_column("giá trị")
    h = g["trajectory_handoff"]
    t3.add_row("Tỷ lệ chu kỳ sinh sản lượng (vs kẹt vòng lặp)",
               _fmt(h["productive_vs_looping"], pct=True))
    t3.add_row("  chu kỳ sinh sản lượng / tổng",
               f"{h['productive_cycles']} / {h['total_cycles']}")
    t3.add_row("Độ dài quỹ đạo (node/chu kỳ)", _fmt(h["trajectory_length_nodes"]))
    t3.add_row("Thời lượng chu kỳ trung bình", _fmt(h["mean_cycle_duration_s"]))
    t3.add_row("Ổn định claim (reconfirm/claim)", _fmt(h["claim_stability"]))
    t3.add_row("Ma sát claim (lock conflict/unstable/replay)", str(h["claim_friction_events"]))
    t3.add_row("Tỷ lệ chu kỳ lỗi", _fmt(h["cycle_error_share"], pct=True))
    console.print(t3)

    t4 = Table(title="4. Chịu lỗi & mở rộng (Resilience & Scalability)")
    t4.add_column("chỉ số", overflow="fold")
    t4.add_column("giá trị")
    r = g["resilience_scalability"]
    repl = ", ".join(f"{k}={n}" for k, n in (r["replication_counts"] or {}).items())
    t4.add_row("LLM ok-rate", _fmt(r["llm_ok_rate"], pct=True))
    t4.add_row("Budget breaker mở", str(r["budget_breaker_opens"]))
    t4.add_row("Phục hồi sau lỗi chu kỳ", _fmt(r["error_recovery_rate"], pct=True))
    t4.add_row("plan_retry_recovered / rework_start",
               f"{r['plan_retry_recovered']} / {r['rework_starts']}")
    t4.add_row("Evidence thông kiểm chứng",
               _fmt(r["evidence_pass_rate"], pct=True))
    t4.add_row("  evidence đã chạy", str(r["evidence_total"]))
    t4.add_row("Role-prompt sống qua chu kỳ (P11)",
               _fmt(r["role_prompt_success_rate"], pct=True))
    t4.add_row("Replication (P24)", repl or _fmt(None))
    t4.add_row("Độ co giãn quy mô S = %ΔP/%ΔN", _fmt(r.get("scalability_S")))
    note = r.get("scalability_note") or ""
    if note:
        t4.add_row("  ghi chú S", f"[dim]{note}[/]")
    console.print(t4)

    t5 = Table(title="5. Quản trị & chi phí (Governance & Cost)")
    t5.add_column("chỉ số", overflow="fold")
    t5.add_column("giá trị")
    v = g["governance_cost"]
    why = ", ".join(f"{k}={n}" for k, n in (v["rejected_why"] or {}).items())
    mean_dur = (f"{v['llm_mean_duration_s']:.1f}s"
                if v["llm_mean_duration_s"] else _fmt(None))
    per_merged_calls = _fmt(v["llm_calls_per_merged"])
    per_merged_secs = (f"{v['llm_seconds_per_merged']:.0f}s"
                       if v["llm_seconds_per_merged"] else _fmt(None))
    t5.add_row("Quality gap (objectors: rejected − approved)", _fmt(v["quality_gap_objectors"]))
    t5.add_row("Incoherence (bị peer bắt lỗi thực chất)",
               _fmt(v["incoherence_rate"], pct=True))
    t5.add_row("Lý do reject", why or _fmt(None))
    t5.add_row("LLM calls / mean duration", f"{v['llm_calls']} / {mean_dur}")
    t5.add_row("Đa dạng mục đích LLM (số purpose)", str(v["llm_purpose_spread"]))
    t5.add_row("Mức tập trung purpose hàng đầu", _fmt(v["llm_top_purpose_share"], pct=True))
    t5.add_row("Chi phí mỗi MR merge (calls / giây)",
               f"{per_merged_calls} / {per_merged_secs}")
    t5.add_row("  đơn vị đo", f"[dim]{v['cost_unit_note']}[/]")
    console.print(t5)

    snaps = memory.list_eval_snapshots(limit=3)
    if report.get("snapshot_id") is not None:
        foot = f"snapshot #{report['snapshot_id']} đã lưu — " \
               f"{len(snaps)} snapshot gần nhất: "
        foot += ", ".join(f"#{s['id']}({s['agent_count']} agent)"
                          for s in snaps)
        console.print(f"[dim]{foot}[/]")
    else:
        console.print(f"[dim]không lưu snapshot (--no-save) — "
                      f"{len(snaps)} snapshot cũ trong DB[/]")
    console.print("[dim]So sánh theo thời gian: python -m agentics.cli swarm-eval "
                  "--days 7 --json | jq .groups[/]")


_GROUP_NAMES = {1: "1 phối hợp", 2: "2 hội tụ", 3: "3 quỹ đạo",
                4: "4 chịu lỗi", 5: "5 quản trị"}


def _pct_cell(row: dict) -> str:
    """Colour the percentage by health direction; '—' when not convertible."""
    pct, good = row.get("pct"), row.get("good_high", True)
    if pct is None:
        return "[dim]—[/]"
    ok = pct >= 70 if good else pct <= 30
    warn = (pct >= 40 if good else pct <= 60) and not ok
    colour = "green" if ok else ("yellow" if warn else "red")
    return f"[{colour}]{pct:.0f}%[/]"


def _print_percent_rows(rows: list[dict]) -> None:
    """One unified table: every metric normalised to a percentage share."""
    t = Table(title="Bảng % tổng hợp — quy đổi mọi chỉ số swarm về phần trăm")
    t.add_column("nhóm", style="cyan", no_wrap=True)
    t.add_column("chỉ số", overflow="fold")
    t.add_column("giá trị gốc", overflow="fold")
    t.add_column("%", justify="right", no_wrap=True)
    t.add_column("cách đọc", overflow="fold")
    for row in rows:
        t.add_row(_GROUP_NAMES.get(row["group"], str(row["group"])),
                  row["metric"], row["raw"], _pct_cell(row), row["hint"])
    console.print(t)
    console.print("[dim]Quy ước: % = tỉ lệ phần trăm theo mẫu số tự nhiên của "
                  "từng chỉ số; '—' = không có mẫu số trung thực để quy đổi. "
                  "Màu: xanh ≥ ngưỡng khỏe, vàng = trung gian, đỏ = cần nhìn "
                  "lại (theo hướng tốt của từng chỉ số).[/]")


@app.command("memory-stats")
def memory_stats() -> None:
    """Memory health: LifeMem traces/clusters/skills + bounded tables + personal lessons."""
    settings = _settings()
    setup_logging("ERROR")
    from agentics.memory.personal_memory import PersonalMemory
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
    console.print(
        f"[bold]Findings:[/] {st['findings']}   [bold]Messages:[/] {st['messages']}   "
        f"[bold]Decisions:[/] {st['decisions']}   [bold]Evidence:[/] {st['evidence']}   "
        f"[bold]Hypotheses:[/] {st['hypotheses']}"
    )
    # v3 sync-trim health: vector docs should track the findings count — a
    # persistently larger number means the tier drifted (ghosts accumulating).
    from agentics.memory.vector_store import VectorStore

    vector = VectorStore(settings.chroma_path)
    if vector.available:
        drift = vector.count() - st["findings"]
        note = "in sync" if drift == 0 else f"drift {drift:+d} (ghosts trimmed lazily)"
        console.print(f"[bold]Vector docs:[/] {vector.count()} ({note})")
    memory.close()
    if st["top_skills"]:
        table = Table(title="Most-reused skills")
        for col in ("uses", "category", "skill"):
            table.add_column(col)
        for s in st["top_skills"]:
            table.add_row(str(s["uses"]), s["category"], s["text"])
        console.print(table)

    # v2: personal memory health — deduped lessons per agent (the same DB
    # the graph opens at data/personal/<agent>.db, read-only here)
    personal_dir = settings.data_dir / "personal"
    rows = []
    if personal_dir.is_dir():
        for db in sorted(personal_dir.glob("*.db")):
            try:
                pm = PersonalMemory(db.stem, personal_dir)
                stats = pm.lesson_stats()
                pm.close()
            except Exception:  # noqa: BLE001, S112 — corrupt notebook skipped, keep reporting
                continue
            rows.append((db.stem, stats))
    if rows:
        table = Table(title="Personal memory (deduped lessons)")
        for col in ("agent", "lessons", "recurrences", "recurring"):
            table.add_column(col)
        for agent, s in rows:
            table.add_row(agent, str(s["lessons"]), str(s["recurrences"]),
                          str(s["recurring"]))
        console.print(table)


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


@app.command("evidence")
def evidence(
    task_id: int = typer.Argument(None, help="chỉ hiện bằng chứng của 1 task (bỏ trống = tất cả)"),
    search: str = typer.Option("", "--search", "-s", help="tìm theo giả thuyết/lệnh/kết quả"),
    limit: int = typer.Option(30, "--limit", "-l", help="số dòng tối đa"),
) -> None:
    """Owner: mọi kết quả kiểm chứng thực nghiệm (P16) các agent đã ghi.

    Đây là phần trả lời cho giả thuyết 3: kết quả kiểm chứng của một agent
    được lưu chung và MỌI agent (cùng owner) tra cứu lại được."""
    settings = _settings()
    setup_logging("ERROR")
    from agentics.memory.shared_memory import SharedMemory

    mem = SharedMemory(settings.db_path)
    try:
        counts = mem.evidence_counts()
        console.print("[bold]Bằng chứng kiểm chứng (P16)[/]",
                      " — ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
                      or "(chưa có)")
        rows = (mem.evidence_search(search, limit=limit) if search
                else mem.evidence_for_task(task_id) if task_id is not None
                else mem.latest_evidence(limit=limit))
        table = Table(title="Empirical evidence" + (f" — task #{task_id}" if task_id is not None else ""))
        table.add_column("ID", justify="right")
        table.add_column("Task", justify="right")
        table.add_column("Agent")
        table.add_column("Verdict")
        table.add_column("Lệnh kiểm chứng", overflow="fold")
        table.add_column("Exit", justify="right")
        table.add_column("Ghi chú", overflow="fold")
        for e in rows:
            color = {"confirmed": "green", "refuted": "red"}.get(e["verdict"], "yellow")
            table.add_row(
                str(e["id"]),
                str(e["task_id"] if e["task_id"] is not None else "-"),
                e["agent"], f"[{color}]{e['verdict']}[/]", e["cmd"],
                str(e["exit_code"]),
                e["note"] or (f"bởi {e['verdict_by']}" if e["verdict_by"] else ""),
            )
        console.print(table)
    finally:
        mem.close()


@app.command()
def audit(
    event: str = typer.Option("", "--event", "-e", help="lọc theo loại sự kiện (merge/mr_reject/task_exempt_set/...)"),
    limit: int = typer.Option(50, "--limit", "-l", help="số dòng tối đa"),
) -> None:
    """Owner: nhật ký kiểm toán bất biến (P21) — merge, từ chối merge, miễn
    protected path, re-verdict bằng chứng. Bảng này KHÔNG BAO GIỜ bị trim:
    đối chiếu sau sự cố vẫn đầy đủ dù journal vận hành đã cắt gọn."""
    settings = _settings()
    setup_logging("ERROR")
    from agentics.memory.shared_memory import SharedMemory

    mem = SharedMemory(settings.db_path)
    try:
        rows = mem.audit_recent(limit=limit)
        if event:
            rows = [r for r in rows if r["event"] == event]
        total = mem.audit_count()
        console.print(f"[bold]Audit trail (P21)[/] — {total} dòng bất biến, "
                      f"hiện {len(rows)} mới nhất")
        table = Table(title="Immutable audit log")
        table.add_column("ID", justify="right")
        table.add_column("Thời điểm", justify="right")
        table.add_column("Agent")
        table.add_column("Sự kiện")
        table.add_column("Chi tiết", overflow="fold")
        import json as _json
        import time as _time
        for r in reversed(rows):
            try:
                detail = _json.dumps(_json.loads(r["detail"]), ensure_ascii=False)
            except Exception:  # noqa: BLE001
                detail = r["detail"]
            table.add_row(
                str(r["id"]), str(int(r["ts"])), r["agent"], r["event"], detail,
            )
        console.print(table)
    finally:
        mem.close()


@app.command()
def panel(
    mr_id: int = typer.Option(0, "--mr", "-m",
                              help="chi tiết 1 MR: mọi phiếu, objection, bản bảo vệ"),
    state: str = typer.Option("", "--state", "-s",
                              help="chỉ hiện panel ở 1 trạng thái (voting/defense/approved/rejected)"),
    stats: bool = typer.Option(False, "--stats", help="số liệu cơ chế P19 (tỉ lệ duyệt, cứu được bởi bảo vệ...)"),
    limit: int = typer.Option(20, "--limit", "-l", help="số panel tối đa"),
) -> None:
    """Owner: bảng bỏ phiếu đánh giá lẫn nhau (P19) — ai vote gì, phản đối ra sao,
    đồng hồ bảo vệ còn bao lâu, và cơ chế có đang cải thiện kết quả chung không.

    Đối tác terminal của cột panel trên dashboard; trạng thái panel là nguồn
    sự thật chung của swarm."""
    settings = _settings()
    setup_logging("ERROR")
    from agentics.memory.shared_memory import SharedMemory

    mem = SharedMemory(settings.db_path)
    try:
        if stats:
            st = mem.panel_stats()
            by_state = st["by_state"]
            if by_state:
                console.print("[bold]Peer panel (P19)[/] — tổng:",
                              " — ".join(f"{k}: {v}" for k, v in sorted(by_state.items())))
            else:
                console.print("[bold]Peer panel (P19)[/] — (chưa có panel nào)")
                return
            decided = st["decided"]
            approved = by_state.get("approved", 0)
            console.print(f"  [bold]tỉ lệ duyệt:[/] {st['approval_rate']:.0%} "
                          f"({approved}/{decided} panel đã kết luận)")
            if st["defended"]:
                save = st["defense_save_rate"]
                console.print(f"  [bold]đã vào vòng bảo vệ:[/] {st['defended']}"
                              + (f" — bảo vệ cứu được {save:.0%}"
                                 if save is not None else ""))
            if st["rejected_why"]:
                console.print("  [bold]lý do reject:[/]",
                              ", ".join(f"{k}: {v}" for k, v in st["rejected_why"].items()))
            if st["top_objectors"]:
                console.print("  [bold]người phản đối nhiều nhất:[/]",
                              ", ".join(f"{r['voter']} ({r['objections']})"
                                        for r in st["top_objectors"]))
            if st["votes"]:
                console.print("  [bold]phiếu (vòng quyết định):[/]",
                              " — ".join(f"{k}: {v}" for k, v in sorted(st["votes"].items())))
            return
        if mr_id:
            p = mem.get_panel(mr_id)
            if p is None:
                console.print(f"[yellow]MR #{mr_id} chưa có panel.[/]")
                return
            review = mem.get_review(mr_id)
            if review is not None:
                console.print(f"[bold]MR #{mr_id}[/] — {review['title']} "
                              f"(tác giả {review['author']})")
            else:
                console.print(f"[bold]MR #{mr_id}[/] (bản ghi MR đã mất)")
            console.print(f"  trạng thái: [bold]{p['state']}[/], vòng {p['round']}")
            if p["state"] == "defense" and p["due_ts"]:
                import time as _t
                left = max(0, int(p["due_ts"] - _t.time()))
                console.print(f"  đồng hồ bảo vệ: còn [bold]{left // 60}p{left % 60:02d}s[/]")
            for rnd in range(1, p["round"] + 1):
                votes = mem.panel_votes(mr_id, round=rnd)
                if votes:
                    console.print(f"  [bold]phiếu vòng {rnd}:[/]")
                    for v in votes:
                        color = "green" if v["vote"] == "accept" else "red"
                        console.print(f"    [{color}]{v['vote']:>6}[/] {v['voter']}"
                                      f" — {v['reason'][:160]}")
                objs = mem.panel_objections(mr_id, round=rnd)
                if objs:
                    console.print(f"  [bold]objection vòng {rnd}:[/]")
                    for o in objs:
                        console.print(f"    [red]•[/] {o['voter']} — {o['objection'][:200]}")
            defense = p.get("defense") or {}
            if defense.get("rebuttals"):
                console.print("  [bold]bản bảo vệ đã nộp:[/]")
                for k, text in list(defense["rebuttals"].items())[:6]:
                    console.print(f"    → {k}: {text[:200]}")
                if defense.get("evidence_ids"):
                    console.print(f"    bằng chứng kèm theo: {defense['evidence_ids']}")
            return
        rows = mem.panels_overview(limit=limit, state=state)
        if not rows:
            console.print("(chưa có panel nào — MR mới sẽ tự mở panel khi reviewer vote)")
            return
        table = Table(title="Peer panels (P19)" + (f" — state={state}" if state else ""))
        table.add_column("MR", justify="right")
        table.add_column("Trạng thái")
        table.add_column("Vòng", justify="right")
        table.add_column("Tác giả")
        table.add_column("Task", justify="right")
        table.add_column("Đồng hồ bảo vệ", justify="right")
        table.add_column("Tiêu đề / kết luận", overflow="fold")
        import time as _t
        now = _t.time()
        for r in rows:
            color = {"approved": "green", "rejected": "red",
                     "defense": "cyan", "voting": "yellow"}.get(r["state"], "white")
            clock = ""
            if r["state"] == "defense" and r["due_ts"]:
                clock = f"{max(0, int(r['due_ts'] - now)) // 60}p"
            tail = (r["verdict_comment"] or r["title"] or "")[:120]
            table.add_row(
                str(r["mr_id"]), f"[{color}]{r['state']}[/]", str(r["round"]),
                r["author"] or "?", str(r["task_id"] if r["task_id"] is not None else "-"),
                clock, tail,
            )
        console.print(table)
    finally:
        mem.close()


@app.command()
def hypotheses(
    status: str = typer.Option("", "--status", "-s",
                               help="chỉ hiện 1 trạng thái (supported/refuted/...)"),
    limit: int = typer.Option(30, "--limit", "-l", help="số dòng tối đa"),
) -> None:
    """Owner: các giả thuyết đã đăng ký + kết luận máy chấm (P16bis).

    Mỗi dòng là một claim được pre-register tiêu chí định lượng TRƯỚC khi
    chạy protocol; verdict là kết quả so sánh số đo với tiêu chí — không
    phải lời kết luận của LLM. Đây là phần trả lời trực tiếp cho giả thuyết 2."""
    settings = _settings()
    setup_logging("ERROR")
    from agentics.memory.shared_memory import SharedMemory

    mem = SharedMemory(settings.db_path)
    try:
        counts = mem.hypothesis_counts()
        console.print("[bold]Giả thuyết (P16bis)[/]",
                      " — ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
                      or "(chưa có)")
        rows = mem.list_hypotheses(limit=limit, status=status or None)
        table = Table(title="Pre-registered hypotheses"
                      + (f" — status {status}" if status else ""))
        table.add_column("ID", justify="right")
        table.add_column("Task", justify="right")
        table.add_column("Agent")
        table.add_column("Trạng thái")
        table.add_column("Giả thuyết", overflow="fold")
        table.add_column("Tiêu chí")
        table.add_column("Kết quả đo", overflow="fold")
        for h in rows:
            color = {"supported": "green", "refuted": "red"}.get(h["status"], "yellow")
            c = h["criterion"]
            crit = f"{c.get('metric', '-')} {c.get('op', '')} {c.get('target', '')}" \
                if c else "-"
            table.add_row(
                str(h["id"]),
                str(h["task_id"] if h["task_id"] is not None else "-"),
                h["agent"], f"[{color}]{h['status']}[/]",
                h["claim"], crit, h["verdict_note"] or "-",
            )
        console.print(table)
        settled = mem.verified_knowledge(limit=5)
        if settled:
            console.print("[bold]Tri thức đã xác thực (mọi agent đều thấy)[/]")
            for k in settled:
                color = "green" if k["status"] == "supported" else "red"
                console.print(f"  [{color}]{k['status'].upper()}[/] {k['claim'][:120]}"
                              f" — {k['note'][:110]}")
    finally:
        mem.close()


@app.command()
def approve(
    hyp_id: int = typer.Argument(..., help="id của hypothesis cần chấp thuận"),
    note: str = typer.Option("", "--note", "-n", help="lý do chấp thuận (ghi vào audit trail)"),
) -> None:
    """Owner: chấp thuận một claim (neo ngoài tầm với của agent).

    Settle anchor law: một verdict chỉ được ghi `supported` khi có ít nhất
    một đầu vào mà agent KHÔNG chấp bút. Chấp thuận của owner là đầu vào đó
    trong local-only mode: self_authored_ratio giảm xuống dưới 1 và claim
    unanchored được mở đường thành tri thức chung. Mọi quyết định được ghi
    journal (event `owner_approved`) để audit."""
    settings = _settings()
    setup_logging("ERROR")
    from agentics.memory.shared_memory import SharedMemory

    mem = SharedMemory(settings.db_path)
    try:
        row = mem.get_hypothesis(hyp_id)
        if row is None:
            console.print(f"[red]Không tìm thấy hypothesis #{hyp_id}[/]")
            raise typer.Exit(1)
        if mem.owner_approve(hyp_id, note=note):
            console.print(f"[green]Đã chấp thuận h#{hyp_id}[/] — {row['claim'][:120]}")
            console.print(f"  anchors hiện có: {mem.settle_anchor_state(hyp_id)}")
        else:
            console.print("[yellow]Claim này đã được chấp thuận trước đó.[/]")
    finally:
        mem.close()


@app.command("verify-all")
def verify_all(
    limit: int = typer.Option(20, "--limit", "-l", help="số claim mới nhất được chạy lại"),
    timeout: int = typer.Option(120, "--timeout", "-t", help="giới hạn giây mỗi protocol"),
) -> None:
    """Owner: tái chạy các claim đã settled từ protocol đã lưu (P23 replay).

    Như một researcher thật lặp lại thí nghiệm cũ: claim giữ nguyên kết luận
    -> claim_reconfirmed; bị lật ngược -> claim_falsified / claim_overturned
    (tri thức chung được sửa tự động); không chạy được nữa -> claim_unstable.
    Mọi kết quả ghi vào bảng evidence dùng chung (method='replay')."""
    settings = _settings()
    setup_logging("ERROR")
    from agentics.config import PROJECT_ROOT
    from agentics.github_ops.repo_workspace import RepoWorkspace
    from agentics.local_ops import LocalRepo
    from agentics.memory.shared_memory import SharedMemory
    from agentics.verify.replay import replay_settled

    mem = SharedMemory(settings.db_path)
    try:
        pending = mem.settled_hypotheses(limit=limit, require_code=True)
        if not pending:
            console.print("[bold]P23 replay[/]: chưa có claim settled nào "
                          "kèm protocol đã lưu — không có gì để tái chạy.")
            return
        console.print(f"[bold]P23 replay[/]: tái chạy {len(pending)} claim "
                      "settled mới nhất trong workspace owner-replay…")
        LocalRepo(settings).ensure(src=PROJECT_ROOT)
        ws = RepoWorkspace("owner-replay", settings)
        ws.sync_main()
        reports = replay_settled(mem, ws, limit=limit, agent="owner",
                                 timeout=timeout)
        table = Table(title="Re-verification (P23) — settled claims, re-run")
        table.add_column("Claim", justify="right")
        table.add_column("Kết luận cũ")
        table.add_column("Kết luận mới")
        table.add_column("Sự kiện")
        table.add_column("Ghi chú", overflow="fold")
        flipped = 0
        for r in reports:
            old = r.get("was", "-")
            new = r.get("status", "-")
            if r.get("replayed") and old != new:
                flipped += 1
            color = {"claim_reconfirmed": "green", "claim_falsified": "red",
                     "claim_overturned": "yellow"}.get(r.get("event", ""), "dim")
            table.add_row(
                f"h#{r['hyp_id']}", str(old),
                f"[{color}]{new}[/]" if r.get("replayed") else "-",
                r.get("event", "-") or "-",
                (r.get("note") or "")[:160],
            )
        console.print(table)
        console.print(f"[bold]{flipped}[/] claim bị lật kết luận sau khi tái "
                      "chạy — tri thức chung đã được sửa (xem journal "
                      "claim_falsified / claim_overturned).")
    finally:
        mem.close()


@app.command()
def replay(
    limit: int = typer.Option(5, "--limit", "-l", help="số claim replay trong lần quét này"),
    hyp_id: int = typer.Option(0, "--id", help="replay đúng 1 giả thuyết theo id"),
) -> None:
    """Owner: chạy lại (replicate) các claim đã settled từ protocol đã lưu (P23).

    Mỗi replay là 2 lần chạy thật của protocol trong workspace riêng, machine
    verdict so với tiêu chí đã pre-register: reconfirm giữ kiến thức, drift
    (supported->refuted) sửa verified_knowledge, overturn đảo kết luận, unstable
    đưa về inconclusive. Kết quả ghi vào evidence chung (method='replay') —
    mọi agent đều thấy. Đây là phần trả lời trực tiếp cho giả thuyết 2 + 3."""
    settings = _settings()
    setup_logging("ERROR")
    settings.ensure_dirs()
    from agentics.github_ops.repo_workspace import RepoWorkspace
    from agentics.memory.shared_memory import SharedMemory
    from agentics.verify.replay import replay_hypothesis, replay_settled

    memory = SharedMemory(settings.db_path)
    try:
        if hyp_id > 0:
            report = [replay_hypothesis(memory, RepoWorkspace(
                "owner-cli", settings, guard=None), hyp_id, agent="owner")]
        else:
            report = replay_settled(
                memory, RepoWorkspace("owner-cli", settings, guard=None),
                limit=max(1, limit), agent="owner")
        if not report:
            console.print("[yellow]Không có claim nào replay được[/] "
                          "(chưa có claim settled mang protocol source).")
            return
        table = Table(title=f"P23 replay — {len(report)} claim(s)")
        table.add_column("ID", justify="right")
        table.add_column("Claim", overflow="fold")
        table.add_column("Trước", justify="center")
        table.add_column("Sau", justify="center")
        table.add_column("Sự kiện")
        for r in report:
            color = {"claim_reconfirmed": "green", "claim_falsified": "red",
                     "claim_overturned": "cyan", "claim_unstable": "yellow"}.get(
                r.get("event", ""), "dim")
            table.add_row(
                str(r.get("hyp_id")),
                (r.get("note") or r.get("claim", ""))[:100],
                r.get("was", "-"), r.get("status", "-"),
                f"[{color}]{r.get('event', '-')}[/]" if r.get("replayed")
                else f"[dim]skipped: {r.get('note', '')[:40]}[/]",
            )
        console.print(table)
    finally:
        memory.close()


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


# ---------------------------------------------------------------- owner TUI
def _fmt_age(seconds: float) -> str:
    s = int(max(0.0, seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def _render_watch(settings, memory, now: float) -> None:
    import time as _t

    agents = memory.active_agents(stale_after_s=86400)
    console.print(f"[bold]Agentics watch[/] — local lab  "
                  f"{_t.strftime('%H:%M:%S', _t.localtime(now))}  "
                  "[dim](Ctrl+C to exit)[/]")
    table = Table(title="Agents — what is each AI doing and why")
    for col in ("agent", "role", "cycle", "status", "age", "node", "task",
                "intent (goal — why)"):
        table.add_column(col, overflow="fold")
    for a in agents:
        intent = a.get("intent") or {}
        why = intent.get("why") or intent.get("goal") or ""
        goal = intent.get("goal") or ""
        if goal and why and why != goal:
            purpose = f"{goal[:52]} — [dim]{why[:60]}[/]"
        else:
            purpose = goal[:70] or "[dim](no live intent)[/]"
        table.add_row(
            a["agent"], a["role"], str(a["cycle"]), a["status"],
            _fmt_age(now - a["last_ts"]), a.get("node") or "-",
            f"#{a['task_id']} {a.get('task_title', '')[:24]}" if a.get("task_id") else "-",
            purpose,
        )
    console.print(table)

    claims = memory.active_claims()
    if claims:
        line = ", ".join(f"#{t}→{c.agent}" for t, c in claims.items())
        console.print(f"[bold]Active claims:[/] {line}")
    queue = memory.mq_pending()
    if queue:
        line = ", ".join(f"MR#{q['pr_number']}({q['author']}, try {q['attempts']})"
                         for q in queue)
        console.print(f"[bold]Merge queue:[/] {line}")
    counts = memory.task_counts()
    if counts:
        console.print("[bold]Backlog:[/] " + ", ".join(
            f"{k}={v}" for k, v in sorted(counts.items())))
    if not settings.auto_merge and queue:
        console.print("[yellow]AUTO_MERGE off — approved MRs wait for "
                      "`agentics mr merge <#>`[/]")

    journal = memory.recent_journal(limit=10)
    if journal:
        feed = Table(title="Live activity (journal)")
        for col in ("time", "agent", "event", "detail"):
            feed.add_column(col, overflow="fold")
        for e in reversed(journal):
            feed.add_row(
                _t.strftime("%H:%M:%S", _t.localtime(e["ts"])), e["agent"],
                e["event"], str(e["detail"])[:80],
            )
        console.print(feed)


@app.command()
def watch(
    interval: float = typer.Option(2.0, "--interval", "-i", help="poll seconds"),
    once: bool = typer.Option(False, "--once", help="render one frame and exit"),
) -> None:
    """Owner terminal view: what every agent is doing and why (offline, SQLite).

    Complements `agentics dashboard` (web) for SSH / terminal-only access —
    same shared-memory tables: live intents, claims, merge queue, backlog,
    and the activity journal. `--once` renders a single frame for scripts."""
    import time as _t
    settings = _settings()
    setup_logging("ERROR")
    settings.ensure_dirs()
    from agentics.memory.shared_memory import SharedMemory

    memory = SharedMemory(settings.db_path)
    try:
        if once:
            _render_watch(settings, memory, _t.time())
            return
        while True:
            console.clear()
            _render_watch(settings, memory, _t.time())
            _t.sleep(max(0.5, interval))
    except KeyboardInterrupt:
        pass
    finally:
        memory.close()


# ---------------------------------------------------------------- owner MR gate
mr_app = typer.Typer(help="Owner: local MR pipeline — list / approve / merge")
app.add_typer(mr_app, name="mr")


@mr_app.command("list")
def mr_list(
    limit: int = typer.Option(20, "--limit"),
) -> None:
    """Owner: review the MR pipeline (author, status, approver, task)."""
    settings = _settings()
    setup_logging("ERROR")
    settings.ensure_dirs()
    from agentics.memory.shared_memory import SharedMemory

    memory = SharedMemory(settings.db_path)
    try:
        mrs = memory.recent_reviews(limit=limit)
        if not mrs:
            console.print("[yellow]no MRs yet — agents open them after implementing a task[/]")
            return
        table = Table(title=f"Local MRs ({len(mrs)})")
        for col in ("MR#", "status", "author", "reviewer", "task", "title"):
            table.add_column(col, overflow="fold")
        for m in mrs:
            table.add_row(
                str(m["id"]), m["status"], m["author"], m.get("reviewer") or "-",
                f"#{m['task_id']}" if m.get("task_id") else "-",
                m["title"][:56],
            )
        console.print(table)
        queue = memory.mq_pending()
        if settings.auto_merge:
            console.print("[dim]AUTO_MERGE=on: the Integrator agent merges approved MRs.[/]")
        elif queue:
            ready = [m for m in mrs if m["status"] == "approved"]
            if ready:
                console.print(f"[yellow]{len(ready)} approved MR(s) awaiting YOUR "
                              "merge — `agentics mr merge <#>`[/]")
    finally:
        memory.close()


@mr_app.command("approve")
def mr_approve(
    mr_id: int = typer.Argument(...),
    note: str = typer.Option("", "--note", help="visible review comment"),
) -> None:
    """Owner: approve an MR yourself (counts as a cross-review, enqueues it)."""
    settings = _settings()
    setup_logging("ERROR")
    settings.ensure_dirs()
    from agentics.memory.shared_memory import SharedMemory

    memory = SharedMemory(settings.db_path)
    try:
        review = memory.get_review(mr_id)
        if review is None:
            console.print(f"[red]MR #{mr_id} not found[/]")
            raise typer.Exit(1)
        if review.get("author") == "owner":
            console.print("[red]owner cannot approve their own MR (P5)[/]")
            raise typer.Exit(1)
        memory.set_review_verdict(mr_id, "approved", "owner", note or "owner approval")
        memory.mq_enqueue(mr_id, review["author"])
        memory.journal("owner", "review", {
            "mr": mr_id, "verdict": "approve", "comment": note[:200],
            "task": review.get("task_id"),
        })
        console.print(f"[green]MR #{mr_id} approved by owner[/] — "
                      "queued; merge with `agentics mr merge "
                      f"{mr_id}` (AUTO_MERGE={settings.auto_merge})")
    finally:
        memory.close()


@mr_app.command("merge")
def mr_merge(
    mr_id: int = typer.Argument(...),
) -> None:
    """Owner: merge an approved MR into local main.

    Mirrors the Integrator's drain path exactly: P10 guard (the task must
    carry the owner exemption for protected paths), tier-2 verification on
    main+MR, then the local merge — plus every memory side effect."""
    settings = _settings()
    setup_logging("ERROR")
    settings.ensure_dirs()
    from agentics.github_ops.repo_workspace import RepoWorkspace
    from agentics.memory import lifemem
    from agentics.memory.shared_memory import SharedMemory
    from agentics.policies.guard import ProtectedPathsGuard

    memory = SharedMemory(settings.db_path)
    try:
        review = memory.get_review(mr_id)
        if review is None:
            console.print(f"[red]MR #{mr_id} not found[/]")
            raise typer.Exit(1)
        if review["status"] not in ("approved", "open", "changes_requested"):
            console.print(f"[red]MR #{mr_id} is already {review['status']}[/]")
            raise typer.Exit(1)
        if review["status"] != "approved":
            console.print(f"[red]MR #{mr_id} is {review['status']} — approve it "
                          "first (`agentics mr approve`)[/]")
            raise typer.Exit(1)

        # P10: same gate the Integrator runs — `allow-protected` exempts a task.
        violations = ProtectedPathsGuard().check_task(
            memory, review.get("task_id"), review.get("files", []))
        if violations:
            console.print(f"[red]P10 protected paths in this MR: {violations[:5]}[/] "
                          "— flag the task with `agentics allow-protected "
                          f"{review.get('task_id')}` if this is intended")
            raise typer.Exit(1)

        workspace = RepoWorkspace("owner-cli", settings, guard=None)
        ok, note_text = workspace.verify_merge(review["branch"])
        if not ok:
            console.print(f"[red]tier-2 verification failed: {note_text}[/]")
            memory.mq_mark(mr_id, "failed")
            memory.set_review_verdict(mr_id, "failed", "owner", note_text[:300])
            raise typer.Exit(1)
        merged_ok = workspace.merge_branch_to_main(
            review["branch"],
            f"merge MR #{mr_id} ({review['author']}, owner-approved) — "
            f"task #{review.get('task_id')}",
        )
        if not merged_ok:
            console.print("[red]merge failed (conflict with main) — MR left queued[/]")
            memory.mq_mark(mr_id, "queued", bump_attempt=True)
            raise typer.Exit(1)

        # identical bookkeeping to LocalMergeQueue.drain's success path
        memory.mq_mark(mr_id, "merged")
        memory.set_review_verdict(mr_id, "merged", "owner")
        trace_ids = memory.set_trace_outcome(mr_id, "merged")
        memory.activate_tools(mr_id)
        try:
            lifemem.on_merge(memory, trace_ids)
        except Exception:  # noqa: BLE001
            pass
        if review.get("task_id"):
            memory.set_task_status(review["task_id"], "merged")
        memory.add_decision("owner", f"merge MR #{mr_id} by {review['author']}",
                            "owner-approved interactive merge")
        memory.journal("owner", "merge", {
            "mr": mr_id, "author": review["author"], "task": review.get("task_id"),
        })
        console.print(f"[green]MR #{mr_id} merged into local main[/] "
                      "(task status → merged, traces + tools updated)")
    finally:
        memory.close()


@app.command("papers")
def papers(
    query: str = typer.Argument(None, help="Search phrase (scholarly / datasets / models / repos / web)"),
    source: str = typer.Option("all", "--source", "-s",
                               help="Scholarly source: all | arxiv | openalex | crossref | semanticscholar | europepmc"),
    max_results: int = typer.Option(8, "--max", help="Rows per source"),
    datasets: bool = typer.Option(False, "--datasets", help="Search Hugging Face datasets instead"),
    models: bool = typer.Option(False, "--models", help="Search Hugging Face models instead"),
    github: bool = typer.Option(False, "--github", help="Search GitHub repos instead (reference implementations)"),
    pypi: str = typer.Option("", "--pypi", help="One PyPI package: existence + latest real version to pin"),
    web: bool = typer.Option(False, "--web", help="DuckDuckGo web search instead"),
    define: bool = typer.Option(False, "--define", help="Wikipedia summary of the term instead"),
    doi: str = typer.Option("", "--doi", help="Resolve one DOI to real-metadata BibTeX"),
) -> None:
    """Search the FREE keyless research tools directly (no LLM cookies).

    Same sources the agents fall back to when the LLM pools are dead —
    arXiv / OpenAlex / Crossref / Semantic Scholar / Europe PMC return real
    publisher metadata (authors, year, abstract, citation counts), never
    LLM-generated text. GitHub/PyPI/HuggingFace ground the verification
    side: reference implementations to clone, datasets/models to feed a
    protocol, real dependency versions to pin. Quick way to sanity-check
    the evidence path before starting a run.
    """
    setup_logging("ERROR")
    settings = _settings()
    if not getattr(settings, "free_tools_enabled", True):
        console.print("[red]Free tools are disabled[/] — set FREE_TOOLS_ENABLED=1")
        raise typer.Exit(1)
    if not any((query, datasets, models, github, pypi, web, define, doi)):
        console.print("Nothing to search — give QUERY or one of "
                      "--datasets/--models/--github/--pypi/--web/--define/--doi")
        raise typer.Exit(1)
    import asyncio as _aio
    from agentics.research import free_tools as ft

    async def _run() -> None:
        if doi:
            bib = await ft.crossref_bibtex(doi, settings=settings)
            console.print(bib or f"[yellow]no BibTeX found for DOI {doi}[/]")
            return
        if pypi:
            info = await ft.pypi_package_info(pypi, settings=settings)
            if not info:
                console.print(f"[yellow]no PyPI package named '{pypi}'[/]")
                return
            console.print(f"[bold]{info['name']}=={info['version']}[/] "
                          f"(requires Python {info['requires_python'] or '?'})")
            if info["summary"]:
                console.print(info["summary"])
            console.print(f"[dim]pin: {info['name']}=={info['version']}{(' — ' + info['home_page']) if info['home_page'] else ''}[/]")
            return
        if define:
            info = await ft.wikipedia_summary(query or "", settings=settings)
            if not info:
                console.print("[yellow]no Wikipedia summary found[/]")
                return
            console.print(f"[bold]{info['term']}[/] — {info['extract']}")
            console.print(f"[dim]{info['url']}[/]")
            return
        if datasets:
            rows = await ft.hf_datasets_search(query or "", max_results, settings=settings)
            table = Table(title=f"Hugging Face datasets: {query}")
            for col in ("dataset", "downloads", "likes", "url"):
                table.add_column(col, overflow="fold")
            for r in rows:
                table.add_row(r["id"], str(r["downloads"] or 0), str(r["likes"] or 0), r["url"])
            console.print(table if rows else "[yellow]no datasets found[/]")
            return
        if models:
            rows = await ft.hf_models_search(query or "", max_results, settings=settings)
            table = Table(title=f"Hugging Face models: {query}")
            for col in ("model", "downloads", "likes", "task", "url"):
                table.add_column(col, overflow="fold")
            for r in rows:
                table.add_row(r["id"], str(r["downloads"] or 0), str(r["likes"] or 0),
                              r["pipeline_tag"] or "-", r["url"])
            console.print(table if rows else "[yellow]no models found[/]")
            return
        if github:
            rows = await ft.github_repo_search(query or "", max_results, settings=settings)
            table = Table(title=f"GitHub repos: {query} (reference implementations)")
            for col in ("repo", "stars", "language", "license", "last push", "url"):
                table.add_column(col, overflow="fold")
            for r in rows:
                table.add_row(f"{r['repo']}", str(r["stars"] if r["stars"] is not None else "-"),
                              r["language"] or "-", r["license"] or "-", r["pushed_at"] or "-", r["url"])
            console.print(table if rows else "[yellow]no repos found[/]")
            for r in rows:
                if r["description"]:
                    console.print(f"[dim]{r['repo']}: {r['description']}[/]")
            return
        if web:
            rows = await ft.ddg_web_search(query or "", max_results, settings=settings)
            for i, r in enumerate(rows, 1):
                console.print(f"[bold]{i}. {r['title']}[/]\n  {r['url']}\n  "
                              f"[dim]{r.get('snippet', '')[:200]}[/]")
            if not rows:
                console.print("[yellow]no web results (DuckDuckGo is best-effort)[/]")
            return
        if source == "arxiv":
            hits = await ft.arxiv_search(query or "", max_results, settings=settings)
        elif source == "openalex":
            hits = await ft.openalex_search(query or "", max_results, settings=settings)
        elif source == "crossref":
            hits = await ft.crossref_works(query or "", max_results, settings=settings)
        elif source == "semanticscholar":
            hits = await ft.semantic_scholar_search(query or "", max_results, settings=settings)
        elif source == "europepmc":
            hits = await ft.europepmc_search(query or "", max_results, settings=settings)
        else:
            hits = await ft.search_all(query or "", max_results, settings=settings)
        if not hits:
            console.print("[yellow]no papers found[/]")
            return
        table = Table(title=f"Papers: {query} (source: {source}, keyless)")
        for col in ("title", "authors", "year", "venue", "cited", "url"):
            table.add_column(col, overflow="fold")
        for h in hits:
            table.add_row(h.title[:70], h.authors_short()[:28], str(h.year or "-"),
                          h.venue[:28], str(h.cited_by if h.cited_by is not None else "-"),
                          h.url)
        console.print(table)
        bibtex = await ft.enriched_bibtex([h.url for h in hits], settings=settings)
        if bibtex:
            console.print(f"\n[bold]BibTeX (real metadata):[/]\n{bibtex}")

    _aio.run(_run())


if __name__ == "__main__":
    app()
