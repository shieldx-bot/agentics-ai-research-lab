"""Owner chat — hỏi đáp với lab bằng tiếng Việt, qua model LOCAL.

Design contract (mirrors the dashboard's local-only guarantee):

* **Zero cookie cost** — the question and the memory digest go ONLY to the
  local router (OmniRoute tier — any backend the gateway routes). The chat
  never touches the cookie pool, never
  passes P20 (it is owner plumbing, not swarm work) and never routes to a
  science purpose.
* **Read-only** — the chat consumes SharedMemory but writes nothing: no
  journal rows, no tasks, no claims. An owner question cannot perturb the
  swarm it is observing.
* **Grounded, not vibes** — every prompt embeds a FRESH digest built from
  shared memory (agents, budget, tasks, evidence, hypotheses, journal).
  The system prompt forbids inventing numbers: what is not in the digest
  must be answered with "không có dữ liệu".
* **Graceful degradation** — the local tier absent/slow/dead degrades to a raw
  digest summary (``source="digest"``), so the endpoint always answers the
  owner something truthful instead of an error page.
* **Agent focus** — a question naming an agent ("a03 đang làm gì?") filters
  that agent's journal slice and widens its task/evidence window, so a
  1.7B local model gets the relevant slice, not the firehose.
"""
from __future__ import annotations

import json
import re
import time
from collections import deque

from agentics.config import Settings
from agentics.memory.shared_memory import SharedMemory

_ALERT_EVENTS = ("integrity_drift", "scope_violation", "lock_conflict",
                 "task_repair", "merge_failed", "p16_inconclusive",
                 "p16_hypothesis_invalid", "panel_rejected",
                 "claim_contradiction", "claim_falsified",
                 "claim_overturned", "claim_unstable", "claim_replay_blocked")

_AGENT_RE = re.compile(r"\ba0?[1-9]\d?\b", re.IGNORECASE)

_DIGEST_CAP = 5000        # keep the whole prompt inside qwen3's 4096 ctx
_HISTORY_TURNS = 3        # recent Q/A pairs carried into the prompt
_HISTORY_CLIP = 400       # per-turn clip, chars

_SYSTEM_PROMPT = (
    "Bạn là trợ lý dashboard của chủ sở hữu (owner) Agentics Lab. CHỈ dùng "
    "dữ liệu trong phần DỮ LIỆU LAB để trả lời — KHÔNG bịa số liệu, không "
    "đoán; điều không có trong dữ liệu thì trả lời rõ 'không có dữ liệu về "
    "điều đó'. Trả lời NGẮN gọn bằng tiếng Việt, nêu số liệu cụ thể khi có."
)


def detect_agent(question: str) -> str:
    """'a3 đang làm gì' / 'trạng thái A03' → 'a03' (zero-padded). '' nếu
    câu hỏi không nhắm vào agent cụ thể."""
    match = _AGENT_RE.search(question or "")
    if not match:
        return ""
    return f"a{int(match.group(0)[1:]):02d}"


def _hhmm(ts: float) -> str:
    return time.strftime("%H:%M", time.localtime(ts))


def _clip(value, limit: int) -> str:
    return str(value)[:limit]


def _detail(e: dict) -> str:
    detail = e.get("detail")
    if not isinstance(detail, str):
        detail = json.dumps(detail, ensure_ascii=False)
    return detail[:110]


def build_context(memory: SharedMemory, settings: Settings,
                  question: str = "") -> tuple[str, str]:
    """Fresh digest of shared memory for the LLM prompt. Returns
    ``(digest, focus_agent)``. Read-only, never raises past a missing
    section (each section degrades independently)."""
    focus = detect_agent(question)
    parts: list[str] = []

    agents = memory.active_agents(stale_after_s=86400)
    if agents:
        lines = []
        for a in agents:
            it = a.get("intent") or {}
            line = (f"- {a['agent']} ({a.get('role', '')}) "
                    f"node={a.get('node', '-')} status={a.get('status', '')} "
                    f"cycle={a.get('cycle')}")
            if a.get("task_id"):
                line += f" task=#{a['task_id']} {_clip(a.get('task_title'), 80)}"
            goal = _clip(it.get("goal"), 120)
            if goal:
                line += f" | goal: {goal}"
            lines.append(line)
        parts.append("AGENTS (đang chạy):\n" + "\n".join(lines))

    parts.append(
        f"BUDGET: {memory.llm_calls_in_window(3600.0)} LLM pool calls trong "
        f"1 giờ qua (cap {settings.llm_max_calls_per_hour})")

    tasks = memory.list_tasks(limit=20 if focus else 10)
    tlines = [f"- #{t['id']} [{t['status']}] {_clip(t['title'], 70)} "
              f"→ {t.get('claimed_by') or '-'}" for t in tasks]
    parts.append("TASKS " + json.dumps(memory.task_counts(), ensure_ascii=False)
                 + (":\n" + "\n".join(tlines) if tlines else ": (trống)"))

    evidence = memory.latest_evidence(limit=8 if focus else 4)
    elines = [f"- #{e['id']} task#{e.get('task_id')} {e['agent']} "
              f"verdict={e['verdict']} exit={e.get('exit_code')} "
              f"cmd={_clip(e.get('cmd'), 60)}" for e in evidence]
    parts.append("EVIDENCE (P16) " + json.dumps(memory.evidence_counts(),
                                                ensure_ascii=False)
                 + (":\n" + "\n".join(elines) if elines else ": (trống)"))

    hyps = memory.list_hypotheses(limit=6)
    hlines = [f"- #{h['id']} [{h['status']}] {_clip(h.get('claim'), 110)}"
              for h in hyps]
    parts.append("HYPOTHESES (P16bis) "
                 + json.dumps(memory.hypothesis_counts(), ensure_ascii=False)
                 + (":\n" + "\n".join(hlines) if hlines else ": (trống)"))
    parts.append("REPLICATIONS (P24bis) "
                 + json.dumps(memory.replication_counts(), ensure_ascii=False))

    alerts = [e for e in memory.recent_journal(limit=200)
              if e.get("event") in _ALERT_EVENTS][:5]
    if alerts:
        parts.append("ALERTS:\n" + "\n".join(
            f"- {_hhmm(e['ts'])} {e['agent']} {e['event']} {_detail(e)}"
            for e in alerts))

    if focus:
        events = [e for e in memory.recent_journal(limit=400)
                  if e.get("agent") == focus][:40]
        head = f"JOURNAL của {focus} (40 sự kiện gần nhất):"
    else:
        events = memory.recent_journal(limit=25)
        head = "JOURNAL (25 sự kiện gần nhất):"
    if events:
        parts.append(head + "\n" + "\n".join(
            f"- {_hhmm(e['ts'])} {e['agent']} {e['event']} {_detail(e)}"
            for e in events))

    m = memory.memory_stats()
    parts.append(
        f"MEMORY: traces={m.get('traces', 0)} clusters={m.get('clusters', 0)} "
        f"skills_active={m.get('active_skills', 0)} "
        f"skills_superseded={m.get('superseded_skills', 0)}")

    return "\n\n".join(parts)[:_DIGEST_CAP], focus


def _fallback_answer(digest: str) -> str:
    """No-LLM answer: the digest itself, honestly labelled. Never fails."""
    return ("[Chế độ không LLM — model cục bộ (OmniRoute) không khả dụng; "
            "đây là dữ liệu thô từ shared memory]\n\n" + digest[:2500])


def _history_block(history: deque) -> str:
    if not history:
        return ""
    lines = ["HỎI-ĐÁP GẦN ĐÂY (ngữ cảnh, câu mới nằm dưới cùng):"]
    for q, a in list(history)[-_HISTORY_TURNS:]:
        lines.append(f"- H: {_clip(q, _HISTORY_CLIP)}")
        lines.append(f"  Đ: {_clip(a, _HISTORY_CLIP)}")
    return "\n".join(lines) + "\n\n"


class ChatSession:
    """Per-process owner chat state: a bounded Q/A ring so follow-up
    questions ("còn a04 thì sao?") keep context. Server-side only — the
    browser re-renders from its own transcript on refresh."""

    def __init__(self, maxlen: int = 8):
        self.turns: deque = deque(maxlen=maxlen)

    async def ask(self, memory: SharedMemory, settings: Settings,
                  question: str) -> dict:
        started = time.monotonic()
        digest, focus = build_context(memory, settings, question)

        answer = None
        if str(getattr(settings, "local_llm_model", "") or "").strip():
            prompt = (
                f"DỮ LIỆU LAB (mới nhất):\n{digest}\n\n"
                + _history_block(self.turns)
                + f"CÂU HỎI: {question}\nTrả lời:")
            try:
                from agentics.llm import local_router
                resp = await local_router.ask(prompt, settings=settings,
                                              purpose="owner_chat")
                if resp is not None and getattr(resp, "ok", False):
                    answer = str(resp.answer)
            except Exception:  # noqa: BLE001 — chat must never 500 the owner
                answer = None

        if answer:
            source = "local"
        else:
            answer = _fallback_answer(digest)
            source = "digest"

        self.turns.append((question, answer))
        return {
            "answer": answer,
            "source": source,
            "focused": focus,
            "elapsed_s": round(time.monotonic() - started, 2),
        }
