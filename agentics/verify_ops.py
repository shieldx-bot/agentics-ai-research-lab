"""P16 — empirical evidence: agents verify hypotheses by RUNNING real code.

Motivation (owner): "các AI agent có thể chạy code hoặc tự kiểm chứng để đánh
giá ý tưởng thay vì chờ vào kết luận của LLM" — this module is the bridge
between an agent's hypothesis and a measurable, shareable verdict.

Design:
* the agent's plan declares `verify_cmds` (a list of commands the checker
  should run to confirm the hypothesis, e.g. a pytest suite or a python
  snippet printing a metric);
* RepoWorkspace.run_command already provides the safety rails (allowlisted
  interpreters, scrubbed environment, bounded lifetime, cwd = workspace);
* ExperimentRunner adds what run_command alone does not: a structured,
  bounded evidence record with a verdict derived from the REAL exit code —
  confirmed (0) / refuted (non-zero) / inconclusive (timeout 124, blocked
  126) — instead of an LLM's self-assessment.
* records land in SharedMemory's `evidence` table and are broadcast to every
  agent, so the verdict is shared knowledge, not private opinion.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

# run_command exit codes with a special meaning (see repo_workspace.run_command)
EXIT_TIMEOUT = 124
EXIT_BLOCKED = 126

_VERDICT_BY_EXIT = {
    0: "confirmed",
    EXIT_TIMEOUT: "inconclusive",
    EXIT_BLOCKED: "inconclusive",
}

OUTPUT_TAIL_CHARS = 1500


@dataclass
class Evidence:
    """One structured result of running a real command against a hypothesis."""
    cmd: str
    exit_code: int
    verdict: str
    output: str = ""
    duration_s: float = 0.0
    ok: bool = False
    method: str = "run_command"
    meta: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        """Flat dict for SharedMemory.record_evidence (output already bounded)."""
        return {
            "method": self.method,
            "cmd": self.cmd,
            "exit_code": self.exit_code,
            "verdict": self.verdict,
            "output": self.output[-OUTPUT_TAIL_CHARS:],
            "duration_s": round(self.duration_s, 3),
        }


def _verdict_for(exit_code: int) -> str:
    return _VERDICT_BY_EXIT.get(exit_code, "refuted" if exit_code != 0 else "confirmed")


def run_experiment(workspace, cmd: list[str], timeout: int = 300) -> Evidence:
    """Run one verification command in the agent workspace; return evidence.

    Verdict comes from the process exit code, NOT from any model:
      * 0    -> confirmed   (the check the plan declared actually passed)
      * 124  -> inconclusive (timed out)
      * 126  -> inconclusive (command rejected by the guard — e.g. protected
                path; the agent must re-plan, never "believes" success)
      * else -> refuted (the hypothesis did not hold under test)
    """
    if isinstance(cmd, str):
        cmd = ["/bin/sh", "-c", cmd]
    started = time.monotonic()
    exit_code, output = workspace.run_command([str(c) for c in cmd], timeout=timeout)
    duration = time.monotonic() - started
    return Evidence(
        cmd=" ".join(str(c) for c in cmd),
        exit_code=int(exit_code),
        verdict=_verdict_for(int(exit_code)),
        output=output,
        duration_s=duration,
        ok=int(exit_code) == 0,
    )


def run_experiments(workspace, cmds: list[list[str]], timeout: int = 300) -> list[Evidence]:
    """Run every declared verification command; never raises.

    One blocked/timeout experiment does not abort the rest — the verdict set
    is what the reviewer (and every other agent) will see."""
    out: list[Evidence] = []
    for cmd in cmds or []:
        try:
            out.append(run_experiment(workspace, cmd, timeout=timeout))
        except Exception as exc:  # noqa: BLE001 — evidence must survive runner bugs
            out.append(Evidence(
                cmd=" ".join(str(c) for c in cmd), exit_code=-1,
                verdict="inconclusive", output=f"runner error: {exc}",
            ))
    return out


def summarize(evidence: list[Evidence]) -> str:
    """One-line digest for state['verification'] / journal / dashboard."""
    if not evidence:
        return ""
    counts: dict[str, int] = {}
    for ev in evidence:
        counts[ev.verdict] = counts.get(ev.verdict, 0) + 1
    breakdown = ", ".join(f"{v}:{n}" for v, n in sorted(counts.items()))
    worst = "confirmed" if counts.get("confirmed") == len(evidence) else \
        "refuted" if counts.get("refuted") else "inconclusive"
    first_cmd = evidence[0].cmd
    return f"P16 {len(evidence)} check(s) [{breakdown}] -> {worst} (e.g. `{first_cmd[:80]}`)"
