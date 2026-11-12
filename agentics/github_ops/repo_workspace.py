"""Local git workspace per agent — clone once, sync per cycle, push to own branch."""
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

log = logging.getLogger("gh.workspace")


class RepoWorkspace:
    """A local clone owned by exactly one agent. All git ops via subprocess so
    credentials never enter the repo config."""

    def __init__(self, agent_id: str, settings: Settings) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", agent_id):
            raise ValueError("agent_id must be a simple 1-64 character identifier")
        self.agent_id = agent_id
        self.settings = settings
        self.path = settings.workspaces_dir / agent_id
        # Keep credentials out of the remote URL and .git/config. Git receives
        # the authorization header through a child-process environment variable.
        self._remote = f"https://github.com/{settings.github_repo}.git"

    # ------------------------------------------------------------ plumbing
    def _git_env(self) -> dict[str, str]:
        """Return a per-process Git auth environment; never persist credentials."""
        allowed = {"PATH", "SYSTEMROOT", "WINDIR", "TMP", "TEMP", "LANG", "LC_ALL"}
        env = {key: value for key, value in os.environ.items() if key in allowed}
        env["HOME"] = str(self.path / ".agent-home")
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

    def _git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        # Only network operations receive the credential. Local commands (especially
        # commit hooks) must not inherit a token they do not need.
        network_ops = {"fetch", "push", "pull", "ls-remote"}
        env = self._git_env() if args and args[0] in network_ops else self._command_environment()
        proc = subprocess.run(
            ["git", *args], cwd=self.path, capture_output=True, text=True, env=env,
        )
        if check and proc.returncode != 0:
            raise RuntimeError(f"git {' '.join(args[:3])}… failed: {proc.stderr[:400]}")
        return proc

    def ensure_cloned(self) -> None:
        if (self.path / ".git").exists():
            # Repair legacy clones that may have stored credentials in origin.
            self._git("remote", "set-url", "origin", self._remote)
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        log.info("[%s] cloning repo…", self.agent_id)
        subprocess.run(
            ["git", "clone", "--filter=blob:none", self._remote, str(self.path)],
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
    def commit_and_push(self, branch: str, message: str) -> bool:
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
        push = self._git("push", "-u", "origin", branch, check=False)
        if push.returncode != 0:
            raise RuntimeError(f"push failed: {push.stderr[:400]}")
        log.info("[%s] pushed branch %s", self.agent_id, branch)
        return True

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
