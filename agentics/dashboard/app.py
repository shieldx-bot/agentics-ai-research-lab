"""FastAPI dashboard — the owner's answer to "các AI đang làm gì, mục đích gì?".

Single page, zero external assets (works fully offline), polls /api/state
every 2.5 s. Shows, per agent: current node, status, current task, and the
PUBLISHED INTENT (goal / why / steps / expected files) — plus the task board,
the MR pipeline, the merge queue, memory health and the activity journal.

Owner control: POST /stop writes the same STOP file as `agentics stop`
(graceful: agents finish the node, release claims, exit) — disable with
DASHBOARD_ALLOW_STOP=0 for non-localhost deployments.

Run: python -m agentics.cli dashboard          (default http://127.0.0.1:8080)
"""
from __future__ import annotations

import logging
import time

from agentics.config import Settings
from agentics.memory.shared_memory import SharedMemory

log = logging.getLogger("dashboard")


def _state(memory: SharedMemory, settings: Settings) -> dict:
    agents = memory.active_agents(stale_after_s=86400)
    now = time.time()
    for a in agents:
        a["age_s"] = max(0, int(now - a["last_ts"]))
    reviews = memory.recent_reviews(limit=12)
    # P19 — attach the peer-panel state so the owner sees WHO voted, WHAT the
    # round is and how long the author has left to defend.
    for r in reviews:
        panel = memory.get_panel(r["id"])
        if panel:
            counts: dict[str, int] = {}
            for v in memory.panel_votes(r["id"], round=panel["round"]):
                counts[v["vote"]] = counts.get(v["vote"], 0) + 1
            panel["votes"] = counts
            if panel["state"] == "defense" and panel["due_ts"]:
                panel["defense_left_s"] = max(0, int(panel["due_ts"] - now))
        r["panel"] = panel
    return {
        "ts": now,
        "mode": "local-only (no GitHub)",
        "research_topic": settings.research_topic,
        "agents": agents,
        "tasks": {
            "counts": memory.task_counts(),
            "recent": memory.list_tasks(limit=12),
        },
        "claims": [
            {"task": c.issue, "agent": c.agent, "branch": c.branch,
             "expires_in_s": max(0, int(c.expires_at - now))}
            for c in memory.active_claims().values()
        ],
        "merge_queue": memory.mq_pending(),
        "reviews": reviews,
        "locks": memory.active_file_locks(),
        "evidence": {
            "counts": memory.evidence_counts(),
            "recent": memory.latest_evidence(limit=12),
        },
        "hypotheses": {
            "counts": memory.hypothesis_counts(),
            "recent": memory.list_hypotheses(limit=12),
        },
        # P19 — immutable trail (compliance) + P18 budget dial: how much of
        # the shared LLM pool the swarm has burned this hour.
        "audit": {"total": memory.audit_count(),
                  "recent": memory.audit_recent(limit=8)},
        "budget": {
            "calls_last_hour": memory.llm_calls_in_window(3600.0),
            "cap_per_hour": settings.llm_max_calls_per_hour,
        },
        "alerts": [
            e for e in memory.recent_journal(limit=120)
            if e.get("event") in ("integrity_drift", "scope_violation",
                                  "lock_conflict", "task_repair",
                                  "merge_failed", "p16_inconclusive",
                                  "p16_hypothesis_invalid",
                                  "panel_rejected", "panel_sweep")
        ][:12],
        "memory": memory.memory_stats(),
        "activity": memory.recent_journal(limit=40),
    }


_HTML = """<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<title>Agentics Lab — Dashboard owner</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { --bg:#0d1117; --card:#161b22; --edge:#30363d; --fg:#e6edf3;
          --dim:#8b949e; --acc:#58a6ff; --ok:#3fb950; --warn:#d29922;
          --bad:#f85149; --purple:#bc8cff; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:14px/1.45 -apple-system,"Segoe UI",Roboto,"Noto Sans",sans-serif; }
  header { padding:14px 20px; border-bottom:1px solid var(--edge);
           display:flex; align-items:baseline; gap:14px; flex-wrap:wrap; }
  header h1 { font-size:17px; margin:0; }
  header .sub { color:var(--dim); font-size:12px; }
  #stopbtn { margin-left:auto; font-size:12px; padding:4px 12px; cursor:pointer;
             color:var(--bad); background:var(--panel); border:1px solid var(--edge);
             border-radius:6px; }
  #stopbtn:hover { border-color:var(--bad); }
  #stopbtn:disabled { opacity:.5; cursor:default; }
  main { padding:16px 20px 40px; display:grid; gap:16px;
         grid-template-columns:repeat(auto-fit,minmax(340px,1fr)); }
  .card { background:var(--card); border:1px solid var(--edge); border-radius:10px;
          padding:12px 14px; overflow:auto; }
  .card h2 { margin:0 0 10px; font-size:13px; text-transform:uppercase;
             letter-spacing:.06em; color:var(--dim); }
  .agent { border:1px solid var(--edge); border-radius:8px; padding:10px;
           margin-bottom:10px; background:#10151c; }
  .agent .top { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
  .agent b { font-size:15px; }
  .role { color:var(--purple); font-size:12px; }
  .node { color:var(--acc); font-size:12px; }
  .st { font-size:11px; padding:1px 8px; border-radius:99px; border:1px solid var(--edge); }
  .st.working { color:var(--ok); border-color:var(--ok); }
  .st.cycle-end, .st.idle { color:var(--dim); }
  .st.stopped, .st.error { color:var(--bad); border-color:var(--bad); }
  .task { color:var(--warn); font-size:12px; margin-top:4px; }
  .intent { margin-top:6px; font-size:12.5px; color:var(--fg); background:#0b0f14;
            border-left:3px solid var(--acc); padding:6px 8px; border-radius:0 6px 6px 0; }
  .intent .why { color:var(--dim); }
  .intent ul { margin:4px 0 0; padding-left:18px; }
  table { width:100%; border-collapse:collapse; font-size:12.5px; }
  th { text-align:left; color:var(--dim); font-weight:600; padding:4px 6px;
       border-bottom:1px solid var(--edge); }
  td { padding:4px 6px; border-bottom:1px solid #1c2430; vertical-align:top; }
  .pill { font-size:11px; padding:0 7px; border-radius:99px; border:1px solid var(--edge); }
  .pill.open { color:var(--acc); } .pill.claimed,.pill.in_review { color:var(--warn); }
  .pill.merged,.pill.approved { color:var(--ok); }
  .pill.failed,.pill.changes_requested,.pill.closed { color:var(--bad); }
  .acts { max-height:320px; overflow:auto; font-size:12.5px; }
  .acts .row { padding:3px 0; border-bottom:1px solid #1c2430; display:flex; gap:8px; }
  .acts .ev { color:var(--acc); min-width:96px; }
  .acts .ag { color:var(--purple); min-width:34px; }
  .acts .tm { color:var(--dim); min-width:64px; }
  .acts .dt { color:var(--dim); word-break:break-word; }
  .kv { font-size:12.5px; color:var(--dim); }
  .kv b { color:var(--fg); }
  .err { color:var(--bad); padding:20px; }
  footer { color:var(--dim); font-size:11px; padding:0 20px 20px; }
</style>
</head>
<body>
<header>
  <h1>Agentics Lab — mắt owner</h1>
  <span class="sub" id="meta">đang tải…</span>
  <button id="stopbtn" title="Dừng mềm: agents hoàn tất node hiện tại, nhả claim và thoát">⏹ Dừng lab</button>
</header>
<main>
  <div class="card" style="grid-column:1/-1"><h2>Agents — đang làm gì, vì sao</h2>
    <div id="agents"></div></div>
  <div class="card"><h2>Tasks (hàng đợi việc)</h2>
    <div class="kv" id="taskcounts"></div>
    <table id="tasks"><thead><tr><th>#</th><th>Trạng thái</th><th>Tiêu đề</th><th>Agent</th></tr></thead><tbody></tbody></table></div>
  <div class="card"><h2>MR pipeline (review &amp; merge)</h2>
    <table id="reviews"><thead><tr><th>MR</th><th>Trạng thái</th><th>Tác giả → Reviewer</th><th>Task</th><th>Panel P19 (đánh giá lẫn nhau)</th></tr></thead><tbody></tbody></table></div>
  <div class="card"><h2>Claims &amp; Merge queue</h2>
    <div class="kv" id="claims"></div>
    <div class="kv" id="mq" style="margin-top:8px"></div></div>
  <div class="card"><h2>File locks (P13 — ai đang giữ file nào)</h2>
    <table id="locks"><thead><tr><th>File</th><th>Agent</th><th>Hết hạn</th></tr></thead><tbody></tbody></table></div>
  <div class="card" style="grid-column:1/-1"><h2>Kiểm chứng thực nghiệm (P16 — bằng chứng chạy thật, không phải lời nói LLM)</h2>
    <div class="kv" id="evcounts"></div>
    <table id="evidence" style="margin-top:8px"><thead><tr><th>ID</th><th>Task</th><th>Agent</th><th>Verdict</th><th>Lệnh kiểm chứng</th><th>Exit</th></tr></thead><tbody></tbody></table></div>
  <div class="card" style="grid-column:1/-1"><h2>Giả thuyết đã đăng ký (P16bis — tiêu chí chốt TRƯỚC khi chạy, máy chấm sau khi đo)</h2>
    <div class="kv" id="hypcounts"></div>
    <table id="hypotheses" style="margin-top:8px"><thead><tr><th>ID</th><th>Trạng thái</th><th>Giả thuyết</th><th>Tiêu chí</th><th>Đo được</th></tr></thead><tbody></tbody></table></div>
  <div class="card"><h2>Cảnh báo hệ thống (guard P10/P13/P14 · sửa chữa task · P16 nghi ngờ)</h2>
    <div class="acts" id="alerts"></div></div>
  <div class="card"><h2>Bộ nhớ (trí nhớ chung)</h2>
    <div class="kv" id="memory"></div></div>
  <div class="card" style="grid-column:1/-1"><h2>Nhật ký hoạt động (journal)</h2>
    <div class="acts" id="acts"></div></div>
</main>
<footer>Local-only mode — không dữ liệu nào rời khỏi máy này. Làm mới mỗi 2.5s.</footer>
<script>
const esc = s => String(s ?? "").replace(/[&<>"']/g, c =>
  ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const hhmmss = ts => new Date(ts*1000).toLocaleTimeString();
function render(s){
  document.getElementById("meta").textContent =
    `chế độ ${s.mode} — ${new Date(s.ts*1000).toLocaleTimeString()} — chủ đề: ${esc(s.research_topic)}`;
  const ag = document.getElementById("agents"); ag.innerHTML = "";
  if(!s.agents.length) ag.innerHTML = '<div class="kv">chưa có agent nào chạy — khởi động: python -m agentics.cli run</div>';
  for(const a of s.agents){
    const it = a.intent || {};
    const steps = (it.steps||[]).length
      ? '<ul>'+ (it.steps||[]).slice(0,8).map(x=>'<li>'+esc(x)+'</li>').join('') +'</ul>' : '';
    const files = (it.expected_files||[]).length ? ' · files: '+esc((it.expected_files||[]).join(", ")) : '';
    ag.insertAdjacentHTML("beforeend", `
      <div class="agent">
        <div class="top"><b>${esc(a.agent)}</b>
          <span class="role">${esc(a.role)}</span>
          <span class="node">node: ${esc(a.node||"-")}</span>
          <span class="st ${esc(a.status)}">${esc(a.status)}</span>
          <span class="sub" style="color:var(--dim)">cycle ${a.cycle} · ${a.age_s}s trước</span></div>
        ${a.task_id ? `<div class="task">task #${a.task_id}: ${esc(a.task_title)}</div>` : ""}
        <div class="intent"><b>Mục đích:</b> ${esc(it.goal||"(chưa công bố)")}
          ${it.why?`<span class="why">— ${esc(it.why)}</span>`:""}${files}${steps}</div>
      </div>`);
  }
  const tc = s.tasks.counts || {};
  document.getElementById("taskcounts").innerHTML =
    Object.entries(tc).map(([k,v])=>`<span class="pill ${esc(k)}">${esc(k)}: <b>${v}</b></span>`).join(" ");
  document.querySelector("#tasks tbody").innerHTML = (s.tasks.recent||[]).map(t=>`
    <tr><td>${t.id}</td><td><span class="pill ${esc(t.status)}">${esc(t.status)}</span></td>
    <td>${esc(t.title)}${t.exempt?' <span class="pill open">owner-exempt</span>':''}</td>
    <td>${esc(t.claimed_by||"-")}</td></tr>`).join("") || '<tr><td colspan="4" class="kv">trống</td></tr>';
  document.querySelector("#reviews tbody").innerHTML = (s.reviews||[]).map(r=>{
    const p = r.panel;
    const panelCell = p
      ? `<span class="pill ${esc(p.state)}">${esc(p.state)} r${p.round}</span> `
        + Object.entries(p.votes||{}).map(([k,v])=>`${esc(k)}:${v}`).join(" ")
        + (p.defense_left_s !== undefined ? ` · bảo vệ còn ${Math.round(p.defense_left_s/60)}p` : "")
      : '<span class="kv">trước P19</span>';
    return `<tr><td>#${r.id}</td><td><span class="pill ${esc(r.status)}">${esc(r.status)}</span></td>
    <td>${esc(r.author)} → ${esc(r.reviewer||"chờ")}</td><td>${r.task_id??'-'}</td>
    <td>${panelCell}</td></tr>`;
  }).join("") || '<tr><td colspan="5" class="kv">chưa có MR nào</td></tr>';
  document.getElementById("claims").innerHTML = (s.claims||[]).length
    ? (s.claims||[]).map(c=>`task #${c.task} → <b>${esc(c.agent)}</b> (${esc(c.branch)}) · còn ${c.expires_in_s}s`).join("<br>")
    : "không có claim nào đang giữ";
  document.getElementById("mq").innerHTML = (s.merge_queue||[]).length
    ? "Merge queue: " + (s.merge_queue||[]).map(m=>`MR #${m.pr_number} (${esc(m.author)})`).join(", ")
    : "Merge queue: trống";
  document.querySelector("#locks tbody").innerHTML = (s.locks||[]).map(l=>`
    <tr><td>${esc(l.path)}</td><td><b>${esc(l.agent)}</b></td>
    <td class="kv">còn ${Math.max(0, Math.round(l.expires_at - s.ts))}s</td></tr>`).join("")
    || '<tr><td colspan="3" class="kv">không có khóa nào</td></tr>';
  const ev = s.evidence || {counts:{}, recent:[]};
  const vc = ev.counts || {};
  document.getElementById("evcounts").innerHTML = Object.keys(vc).length
    ? Object.entries(vc).map(([k,v])=>`<span class="pill ${k==='confirmed'?'merged':k==='refuted'?'failed':'claimed'}">${esc(k)}: <b>${v}</b></span>`).join(" ")
    : '<span class="kv">chưa có bằng chứng nào — agent khai báo "verify_cmds" trong kế hoạch để chạy kiểm chứng thật (P16)</span>';
  document.querySelector("#evidence tbody").innerHTML = (ev.recent||[]).map(e=>`
    <tr><td>#${e.id}</td><td>${e.task_id??'-'}</td><td>${esc(e.agent)}</td>
    <td><span class="pill ${e.verdict==='confirmed'?'merged':e.verdict==='refuted'?'failed':'claimed'}">${esc(e.verdict)}</span>${e.verdict_by?` <span class="kv">(${esc(e.verdict_by)})</span>`:''}</td>
    <td class="dt">${esc(e.cmd)}</td><td>${e.exit_code}</td></tr>`).join("")
    || '<tr><td colspan="6" class="kv">trống</td></tr>';
  const hyp = s.hypotheses || {counts:{}, recent:[]};
  const hc = hyp.counts || {};
  document.getElementById("hypcounts").innerHTML = Object.keys(hc).length
    ? Object.entries(hc).map(([k,v])=>`<span class="pill ${k==='supported'?'merged':k==='refuted'?'failed':k==='proposed'?'open':'claimed'}">${esc(k)}: <b>${v}</b></span>`).join(" ")
    : '<span class="kv">chưa có giả thuyết nào — researcher sẽ đăng ký giả thuyết + tiêu chí định lượng trước khi chạy protocol (P16bis)</span>';
  document.querySelector("#hypotheses tbody").innerHTML = (hyp.recent||[]).map(h=>{
    const c = h.criterion || {};
    const crit = c.metric ? `${esc(c.metric)} ${esc(c.op)} ${c.target}` : '-';
    const measured = (h.verdict_note||'').replace(/^measured /,'');
    return `<tr><td>#${h.id}</td>
    <td><span class="pill ${h.status==='supported'?'merged':h.status==='refuted'?'failed':h.status==='proposed'?'open':'claimed'}">${esc(h.status)}</span></td>
    <td class="dt">${esc(h.claim)}</td><td class="dt">${crit}</td>
    <td class="dt">${esc(measured||'-')}</td></tr>`;}).join("")
    || '<tr><td colspan="5" class="kv">trống</td></tr>';
  const alerts = s.alerts||[];
  document.getElementById("alerts").innerHTML = alerts.length
    ? alerts.slice().reverse().map(e=>`
      <div class="row"><span class="tm">${hhmmss(e.ts)}</span>
      <span class="ev" style="color:var(--bad)">${esc(e.event)}</span>
      <span class="dt">${esc(typeof e.detail==="string"?e.detail:JSON.stringify(e.detail))}</span></div>`).join("")
    : '<div class="kv">không có cảnh báo nào — mọi lớp guard còn nguyên vẹn</div>';
  const m = s.memory || {};
  document.getElementById("memory").innerHTML =
    `traces: <b>${m.traces??0}</b> (clustered ${m.clustered??0}) · clusters: <b>${m.clusters??0}</b> · ` +
    `skills: <b>${m.active_skills??0}</b> active / ${m.superseded_skills??0} superseded`;
  document.getElementById("acts").innerHTML = (s.activity||[]).slice().reverse().map(e=>`
    <div class="row"><span class="tm">${hhmmss(e.ts)}</span><span class="ag">${esc(e.agent)}</span>
    <span class="ev">${esc(e.event)}</span><span class="dt">${esc(e.detail)}</span></div>`).join("")
    || "chưa có sự kiện";
}
async function tick(){
  try{
    const r = await fetch("/api/state");
    render(await r.json());
  }catch(e){
    document.getElementById("meta").innerHTML =
      '<span class="err">mất kết nối tới dashboard API</span>';
  }
}
document.getElementById("stopbtn").addEventListener("click", async () => {
  const btn = document.getElementById("stopbtn");
  if (!confirm("Dừng toàn bộ lab? Agents sẽ hoàn tất node hiện tại, nhả claim và thoát.")) return;
  btn.disabled = true;
  try {
    const r = await fetch("/stop", { method: "POST" });
    const j = await r.json().catch(() => ({}));
    document.getElementById("meta").textContent =
      r.ok ? (j.note || "đã yêu cầu dừng") : (j.error || `dừng thất bại (${r.status})`);
  } catch(e) {
    document.getElementById("meta").innerHTML =
      '<span class="err">dừng thất bại: ' + esc(String(e)) + '</span>';
  } finally {
    btn.disabled = false;
  }
});
tick(); setInterval(tick, 2500);
</script>
</body>
</html>"""


def create_app(memory: SharedMemory, settings: Settings):
    """Build the dashboard FastAPI app over an open SharedMemory."""
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse

    app = FastAPI(title="Agentics Lab Dashboard", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _HTML

    @app.get("/api/state")
    def state():
        try:
            return JSONResponse(_state(memory, settings))
        except Exception as exc:  # noqa: BLE001
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.post("/stop")
    def stop():
        """Owner control: graceful stop — cùng cơ chế STOP file với CLI.

        Writes the same STOP file as `python -m agentics.cli stop`: every agent
        finishes its current node, releases its claims and exits between
        cycles — nothing is killed hard. Disable for non-localhost deployments
        with DASHBOARD_ALLOW_STOP=0."""
        import os

        if os.environ.get("DASHBOARD_ALLOW_STOP", "1").strip().lower() in {
                "0", "false", "no", "off"}:
            return JSONResponse(
                {"error": "endpoint dừng bị vô hiệu (DASHBOARD_ALLOW_STOP=0)"},
                status_code=403)
        from agentics.stop import request_stop

        request_stop()
        log.info("STOP requested from the dashboard")
        return JSONResponse({
            "stopped": True,
            "note": "Đã ghi file STOP — agents hoàn tất node hiện tại, "
                    "nhả claim và dừng giữa chu kỳ.",
        })

    return app
