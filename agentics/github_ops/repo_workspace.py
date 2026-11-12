"""Local git workspace per agent — clone once, sync per cycle, push to own branch.

Local-only mode: "origin" is the shared bare repo at data/lab.git (seeded from
this checkout by the orchestrator) — no credentials, no network. All git ops
via subprocess so credentials never enter the repo config.
"""
from __future__ import annotations

import base64
import logging
import os
import re
import signal
import subprocess
import sys
from pathlib import Path

from agentics.config import Settings
from agentics.policies.guard import INTEGRATOR_MERGE_ENV, ProtectedPathsGuard

log = logging.getLogger("gh.workspace")


class RepoWorkspace:
    """A local clone owned by exactly one agent. All git ops via subprocess so
    credentials never enter the repo config."""

    def __init__(
        self, agent_id: str, settings: Settings,
        guard: ProtectedPathsGuard | None = None,
    ) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", agent_id):
            raise ValueError("agent_id must be a simple 1-64 character identifier")
        self.agent_id = agent_id
        self.settings = settings
        self.guard = guard
        self.path = settings.workspaces_dir / agent_id
        # Local-only mode: origin is the on-disk bare repo. No token, no network.
        self._remote = str(settings.local_repo_path)

    # ------------------------------------------------------------ plumbing
    def _git_env(self) -> dict[str, str]:
        """Return a per-process Git auth environment; never persist credentials."""
        allowed = {"PATH", "SYSTEMROOT", "WINDIR", "TMP", "TEMP", "LANG", "LC_ALL"}
        env = {key: value for key, value in os.environ.items() if key in allowed}
        home = self.settings.data_dir / ".git-home" / self.agent_id
        home.mkdir(parents=True, exist_ok=True)
        env["HOME"] = str(home)
        env["GIT_TERMINAL_PROMPT"] = "0"
        token = self.settings.github_token
        if token:
            try:
                count = max(0, int(env.get("GIT_CONFIG_COUNT", "0")))
            except ValueError:
                count = 0
            env["GIT_CONFIG_COUNT"] = str(count + 1)
            env[f"GIT_CONFIG_KEY_{count}"] = "http.extraheader"
            basic = base64.b64encode(f"x-access-token:{token}".encode()).decode("ascii")
            env[f"GIT_CONFIG_VALUE_{count}"] = f"AUTHORIZATION: basic {basic}"
        return env

    def _git(self, *args: str, check: bool = True,
             extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
        # Only network operations receive the credential. Local commands (especially
        # commit hooks) must not inherit a token they do not need.
        network_ops = {"fetch", "push", "pull", "ls-remote"}
        env = self._git_env() if args and args[0] in network_ops else self._command_environment()
        if extra_env:
            env = {**env, **extra_env}
        proc = subprocess.run(
            ["git", *args], cwd=self.path, capture_output=True, text=True, env=env,
            check=False,
        )
        if check and proc.returncode != 0:
            raise RuntimeError(f"git {' '.join(args[:3])}… failed: {proc.stderr[:400]}")
        return proc

    def ensure_cloned(self) -> None:
        if (self.path / ".git").exists():
            self._git("remote", "set-url", "origin", self._remote)
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        log.info("[%s] cloning workspace from %s", self.agent_id, self._remote)
        subprocess.run(
            ["git", "clone", str(self._remote), str(self.path)],
            check=True, capture_output=True, text=True, env=self._git_env(),
        )
        self._git("config", "user.name", f"agentics[{self.agent_id}]")
        self._git("config", "user.email", f"agentics+{self.agent_id}@users.noreply.github.com")

    def sync_main(self, branch: str = "main") -> None:
        self.ensure_cloned()
        self._git("fetch", "origin", branch)
        self._git("checkout", branch)
        self._git("reset", "--hard", f"origin/{branch}")

    def checkout_new_branch(self, branch: str) -> None:
        self._git("checkout", "-B", branch)

    def checkout_branch_from_origin(self, branch: str) -> None:
        """Rework: resume a previously pushed branch instead of resetting to main.

        The branch restarts from its origin head, so the already-reviewed work
        is kept and the author only adds the fixes the review asked for. A
        branch that was never pushed (or was deleted) falls back to a fresh one.
        """
        self.ensure_cloned()
        fetch = self._git("fetch", "origin", branch, check=False)
        verified = (
            self._git("rev-parse", "--verify", f"origin/{branch}", check=False)
            if fetch.returncode == 0 else None
        )
        if verified is not None and verified.returncode == 0:
            self._git("checkout", "-B", branch, f"origin/{branch}")
        else:
            self._git("checkout", "-B", branch)

    # ------------------------------------------------------------ content
    def _resolve_workspace_path(self, rel_path: str) -> Path:
        """Resolve a relative path and reject traversal or symlink escapes."""
        candidate = Path(rel_path)
        if candidate.is_absolute():
            raise ValueError("workspace paths must be relative")
        root = self.path.resolve()
        target = (root / candidate).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError("path escapes the agent workspace") from exc
        return target

    def write_file(self, rel_path: str, content: str) -> None:
        # P10 layer 1: the lab's own code cannot even be written into a
        # workspace (regardless of what the plan says).
        if self.guard is not None and self.guard.is_protected(rel_path):
            raise ValueError(
                f"P10 protected path: '{rel_path}' is lab code — agents must "
                "not modify it (owner can exempt a task: agentics.cli allow-protected)"
            )
        target = self._resolve_workspace_path(rel_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Re-resolve after mkdir to catch a symlink introduced in a parent path.
        target = self._resolve_workspace_path(rel_path)
        target.write_text(content, encoding="utf-8")

    def read_file(self, rel_path: str) -> str | None:
        target = self._resolve_workspace_path(rel_path)
        if not target.exists():
            return None
        return target.read_text(encoding="utf-8", errors="replace")

    def _command_environment(self) -> dict[str, str]:
        """Minimal environment for tests: never forward GitHub/LLM/session secrets."""
        allowed = {"PATH", "SYSTEMROOT", "WINDIR", "TMP", "TEMP", "LANG", "LC_ALL"}
        env = {key: value for key, value in os.environ.items() if key in allowed}
        home = self.path / ".agent-home"
        home.mkdir(parents=True, exist_ok=True)
        env.update({
            "HOME": str(home),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        })
        return env

    def _validate_command(self, cmd: list[str]) -> None:
        """Permit only this Python runtime, approved test modules, or workspace scripts."""
        if not cmd or any(not isinstance(arg, str) or "\x00" in arg for arg in cmd):
            raise ValueError("command must be a non-empty list of text arguments")
        if len(cmd) > 128 or any(len(arg) > 4096 for arg in cmd):
            raise ValueError("command exceeds argument limits")
        executable = Path(cmd[0])
        try:
            same_python = executable.resolve() == Path(sys.executable).resolve()
        except OSError:
            same_python = False
        if not same_python:
            raise ValueError("only the active Python interpreter may be executed")
        if len(cmd) >= 3 and cmd[1] == "-m" and cmd[2] in {"pytest", "py_compile"}:
            args = cmd[3:]
            if any(arg == "-c" or arg.startswith("--") and "=" in arg and
                   any(secret in arg.lower() for secret in ("token", "password", "cookie"))
                   for arg in args):
                raise ValueError("unsafe test-runner argument")
            for arg in args:
                path_arg = arg.split("::", 1)[0]
                if ("/" in path_arg or path_arg.endswith(".py")) and not path_arg.startswith("-"):
                    self._resolve_workspace_path(path_arg)
            return
        if len(cmd) >= 2 and cmd[1].endswith(".py"):
            self._resolve_workspace_path(cmd[1])
            return
        raise ValueError("allowed commands: python -m pytest, python -m py_compile, or workspace .py scripts")

    def run_command(self, cmd: list[str], timeout: int = 120) -> tuple[int, str]:
        """Run approved Python checks with a scrubbed environment and bounded lifetime.

        This is a host-process guardrail, not an OS/container security boundary.
        Use a rootless container sandbox for untrusted code in production.
        """
        if not isinstance(timeout, int) or timeout < 1 or timeout > 3600:
            return 126, "timeout must be an integer between 1 and 3600 seconds"
        try:
            self._validate_command(cmd)
        except (ValueError, OSError) as exc:
            return 126, f"command rejected: {exc}"
        proc = subprocess.Popen(
            cmd, cwd=self.path, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env=self._command_environment(),
            start_new_session=(os.name == "posix"),
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            return proc.returncode, (stdout + stderr)[-4000:]
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                proc.kill()
            stdout, stderr = proc.communicate()
            return 124, ((stdout or "") + (stderr or "") + "\ntimeout")[-4000:]
        except Exception:
            proc.kill()
            proc.communicate()
            raise

    def changed_files(self) -> list[str]:
        """Working-tree changes, one entry per file (untracked dirs are expanded)."""
        proc = self._git("status", "--porcelain", "--untracked-files=all", check=False)
        files = []
        for line in proc.stdout.splitlines():
            if len(line) < 4:
                continue
            # "XY path" — X/Y may be a space (" M path"), so never strip before slicing
            part = line[3:].split(" -> ")[-1].strip().strip('"')
            files.append(part)
        return files

    # ------------------------------------------------------------ push
    def commit_and_push(self, branch: str, message: str,
                        allow_protected: bool = False) -> bool:
        """Commit the working tree and push the agent branch.

        ``allow_protected=True`` marks a push sanctioned by the OWNER (a task
        carrying `agentics.cli allow-protected`): the P10 origin hook then lets
        the branch through even if it touches protected paths. The flag travels
        as an env var on this one push only — agent code cannot forge it
        (git subprocesses run with a scrubbed environment)."""
        self._git("add", "-A")
        proc = self._git("status", "--porcelain", check=False)
        if not proc.stdout.strip():
            log.info("[%s] nothing to commit", self.agent_id)
            return False
        self._git("commit", "-m", message)
        # P4 hygiene: rebase onto latest main before pushing
        self._git("fetch", "origin", "main")
        rebase = self._git("rebase", "origin/main", check=False)
        if rebase.returncode != 0:
            self._git("rebase", "--abort", check=False)
            raise RuntimeError("rebase conflict — caller must yield (P3/P4)")
        push = self._git("push", "-u", "origin", branch, check=False,
                         extra_env={INTEGRATOR_MERGE_ENV: "1"} if allow_protected else None)
        if push.returncode != 0:
            raise RuntimeError(f"push failed: {push.stderr[:400]}")
        log.info("[%s] pushed branch %s", self.agent_id, branch)
        return True

    def merge_branch_to_main(self, branch: str, message: str) -> bool:
        """Local replacement for GitHub's rebase-merge (P4, Integrator only).

        Merges origin/<branch> into a throw-away branch based on origin/main,
        then pushes that as a fast-forward of main on the shared bare repo.
        Returns False when the merge is not possible (conflict)."""
        self.sync_main()
        try:
            self._git("fetch", "origin", branch)
            self._git("checkout", "-B", "merge/tmp", "origin/main")
            merge = self._git("merge", "--no-edit", "--no-ff", f"origin/{branch}",
                              "-m", message, check=False)
            if merge.returncode != 0:
                self._git("merge", "--abort", check=False)
                return False
            push = self._git("push", "origin", "merge/tmp:main", check=False,
                             extra_env={INTEGRATOR_MERGE_ENV: "1"})
            if push.returncode != 0:
                # someone moved main between fetch and push -> not a fast-forward
                log.warning("[%s] push merge/tmp:main failed: %s",
                            self.agent_id, push.stderr[:200])
                return False
            return True
        except RuntimeError as exc:
            log.warning("[%s] local merge failed: %s", self.agent_id, exc)
            return False
        finally:
            self.sync_main()
            self._git("branch", "-D", "merge/tmp", check=False)

    def diff_files_vs_main(self) -> list[str]:
        proc = self._git("diff", "--name-only", "origin/main...HEAD", check=False)
        return [l.strip() for l in proc.stdout.splitlines() if l.strip()]

    # ------------------------------------------------------------ verification
    def verify_merge(
        self, branch: str, cmd: list[str] | None = None, timeout: int = 300,
    ) -> tuple[bool, str]:
        """Tier-2 gate: merge origin/<branch> into a throw-away copy of origin/main
        and run the full test command there. Always leaves the workspace on a
        clean `main`. Returns (ok, note)."""
        self.sync_main()
        try:
            self._git("fetch", "origin", branch)
            self._git("checkout", "-B", "verify/tmp", "origin/main")
            merge = self._git("merge", "--no-edit", "--no-ff", f"origin/{branch}", check=False)
            if merge.returncode != 0:
                self._git("merge", "--abort", check=False)
                return False, f"merge conflict with main: {(merge.stdout + merge.stderr)[-300:]}"
            code, out = self.run_command(
                cmd or [sys.executable, "-m", "pytest", "-q", "-x"], timeout=timeout)
            if code in (0, 5):          # 5 = no tests collected
                return True, ""
            return False, f"tests failed on main+PR (exit {code}): {out[-400:]}"
        except RuntimeError as exc:
            return False, str(exc)
        finally:
            self.sync_main()
            self._git("branch", "-D", "verify/tmp", check=False)
