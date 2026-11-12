"""Local git workspace per agent — clone once, sync per cycle, push to own branch."""
from __future__ import annotations

import base64
import logging
import os
import subprocess
import sys
from pathlib import Path

from agentics.config import Settings

log = logging.getLogger("gh.workspace")


class RepoWorkspace:
    """A local clone owned by exactly one agent. All git ops via subprocess so
    credentials never enter the repo config."""

    def __init__(self, agent_id: str, settings: Settings) -> None:
        self.agent_id = agent_id
        self.settings = settings
        self.path = settings.workspaces_dir / agent_id
        # Keep credentials out of the remote URL and .git/config. Authentication
        # is injected only into the child process environment for each Git call.
        self._remote = f"https://github.com/{settings.github_repo}.git"

    def _git_env(self) -> dict[str, str]:
        env = os.environ.copy()
        token = self.settings.github_token
        if token:
            auth = base64.b64encode(f"x-access-token:{token}".encode()).decode()
            try:
                count = int(env.get("GIT_CONFIG_COUNT", "0"))
            except ValueError:
                count = 0
            env[f"GIT_CONFIG_KEY_{count}"] = "http.https://github.com/.extraheader"
            env[f"GIT_CONFIG_VALUE_{count}"] = f"AUTHORIZATION: basic {auth}"
            env["GIT_CONFIG_COUNT"] = str(count + 1)
        return env

    # ------------------------------------------------------------ plumbing
    def _git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        proc = subprocess.run(
            ["git", *args], cwd=self.path, capture_output=True, text=True,
            env=self._git_env(),
        )
        if check and proc.returncode != 0:
            raise RuntimeError(f"git {' '.join(args[:3])}… failed: {proc.stderr[:400]}")
        return proc

    def ensure_cloned(self) -> None:
        if (self.path / ".git").exists():
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
    def write_file(self, rel_path: str, content: str) -> None:
        target = self.path / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def read_file(self, rel_path: str) -> str | None:
        target = self.path / rel_path
        if not target.exists():
            return None
        return target.read_text(encoding="utf-8", errors="replace")

    def _command_allowed(self, cmd: list[str]) -> bool:
        """Allow only the exact Python validation commands used by this project."""
        if not cmd or Path(cmd[0]).resolve() != Path(sys.executable).resolve():
            return False
        if cmd[1:3] == ["-m", "py_compile"] and len(cmd) > 3:
            root = self.path.resolve()
            for raw_path in cmd[3:]:
                candidate = Path(raw_path)
                candidate = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
                if not candidate.is_relative_to(root) or candidate.suffix != ".py":
                    return False
            return True
        return cmd[1:3] == ["-m", "pytest"] and set(cmd[3:]).issubset({"-q", "-x"})

    def run_command(self, cmd: list[str], timeout: int = 120) -> tuple[int, str]:
        """Run only approved Python validation commands (not an OS sandbox)."""
        if not self._command_allowed(cmd):
            return 126, "command rejected by workspace allowlist"
        try:
            proc = subprocess.run(
                cmd, cwd=self.path, capture_output=True, text=True,
                timeout=max(1, min(int(timeout), 300)),
            )
            return proc.returncode, (proc.stdout + proc.stderr)[-4000:]
        except subprocess.TimeoutExpired:
            return 124, "timeout"
        except FileNotFoundError:
            return 127, f"command not found: {cmd[0]}"

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
