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

import shlex
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

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


# interpreter aliases a plan JSON may use for the same runtime we run in
_PYTHON_ALIASES = {"python", "python3", Path(sys.executable).name, sys.executable}


def coerce_cmd(cmd: str | list[str]) -> tuple[list[str] | None, str]:
    """Normalize a declared verification command to an allowlisted argv.

    Plans come from LLM JSON, so `verify_cmds` entries are STRINGS like
    ``"python -m pytest tests/x.py"`` or ``"python experiments/check.py"``.
    RepoWorkspace.run_command permits ONLY the active interpreter (no
    shells), so a string must be parsed into argv — never wrapped in
    ``/bin/sh -c`` (which the allowlist would reject, silently turning
    every declared check into `inconclusive`).

    Returns (argv, "") on success or (None, reason) when the command can
    never satisfy the sandbox contract (rejected without running)."""
    if isinstance(cmd, str):
        try:
            argv = shlex.split(cmd)
        except ValueError as exc:
            return None, f"unparsable command: {exc}"
    elif isinstance(cmd, (list, tuple)):
        argv = [str(a) for a in cmd]
    else:
        return None, "command must be text or a list of arguments"
    if not argv:
        return None, "empty command"
    head = argv[0]
    if head == "pytest":                       # "pytest tests/x.py" -> -m form
        return [sys.executable, "-m", "pytest", *argv[1:]], ""
    if head in _PYTHON_ALIASES:                # same runtime, exact path
        return [sys.executable, *argv[1:]], ""
    return None, (
        "verify_cmds must invoke python or pytest (the workspace allowlist "
        f"runs no shells and no other binaries); got {head!r}"
    )


def run_experiment(workspace, cmd: list[str] | str, timeout: int = 300) -> Evidence:
    """Run one verification command in the agent workspace; return evidence.

    Verdict comes from the process exit code, NOT from any model:
      * 0    -> confirmed   (the check the plan declared actually passed)
      * 124  -> inconclusive (timed out)
      * 126  -> inconclusive (command rejected by the guard — e.g. protected
                path; the agent must re-plan, never "believes" success)
      * else -> refuted (the hypothesis did not hold under test)

    String commands are parsed to argv (see coerce_cmd) — a shell is never
    involved, so a declared check either really runs or is recorded as
    inconclusive with the reason."""
    argv, reason = coerce_cmd(cmd)
    if argv is None:
        return Evidence(
            cmd=str(cmd)[:400], exit_code=EXIT_BLOCKED, verdict="inconclusive",
            output=f"command rejected: {reason}",
        )
    started = time.monotonic()
    exit_code, output = workspace.run_command(argv, timeout=timeout)
    duration = time.monotonic() - started
    return Evidence(
        cmd=" ".join(argv),
        exit_code=int(exit_code),
        verdict=_verdict_for(int(exit_code)),
        output=output,
        duration_s=duration,
        ok=int(exit_code) == 0,
    )


def run_experiments(workspace, cmds: list, timeout: int = 300) -> list[Evidence]:
    """Run every declared verification command; never raises.

    Entries may be argv lists or plan-JSON strings ("python -m pytest ...");
    see coerce_cmd. One blocked/timeout experiment does not abort the rest —
    the verdict set is what the reviewer (and every other agent) will see."""
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
