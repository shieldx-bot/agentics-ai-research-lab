"""Local git workspace per agent — clone once, sync per cycle, push to own branch."""
from __future__ import annotations

import logging
import subprocess
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
        self._remote = (
            f"https://x-access-token:{settings.github_token}@github.com/"
            f"{settings.github_repo}.git"
        )

    # ------------------------------------------------------------ plumbing
    def _git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        proc = subprocess.run(
            ["git", *args], cwd=self.path, capture_output=True, text=True
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
            check=True, capture_output=True, text=True,
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

    def run_command(self, cmd: list[str], timeout: int = 120) -> tuple[int, str]:
        """Run a whitelisted command inside the workspace (e.g. tests)."""
        try:
            proc = subprocess.run(
                cmd, cwd=self.path, capture_output=True, text=True, timeout=timeout
            )
            return proc.returncode, (proc.stdout + proc.stderr)[-4000:]
        except subprocess.TimeoutExpired:
            return 124, "timeout"
        except FileNotFoundError:
            return 127, f"command not found: {cmd[0]}"

    def changed_files(self) -> list[str]:
        proc = self._git("status", "--porcelain", check=False)
        files = []
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            # "XY path" or "XY \"path with space\""
            part = line[3:].strip().strip('"')
            files.append(part.removeprefix("-> ").strip())
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
