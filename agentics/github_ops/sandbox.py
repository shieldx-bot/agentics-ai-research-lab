"""Fail-closed Docker runner for untrusted repository code.

The Docker daemon MUST be rootless. Only the agent workspace is bind-mounted;
the container receives no host environment, credentials, Docker socket, or network.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Sequence

from agentics.config import Settings


class SandboxError(RuntimeError):
    """Raised when the configured isolation boundary cannot be guaranteed."""


class DockerSandboxRunner:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @staticmethod
    def _docker_environment() -> dict[str, str]:
        # Keep only CLI/runtime settings needed to reach the user's rootless daemon.
        allowed = {
            "PATH", "SYSTEMROOT", "WINDIR", "TMP", "TEMP", "LANG", "LC_ALL",
            "HOME", "XDG_RUNTIME_DIR", "DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG",
        }
        return {key: value for key, value in os.environ.items() if key in allowed}

    def _preflight(self) -> tuple[str, dict[str, str]]:
        if os.name != "posix":
            raise SandboxError("rootless Docker sandbox currently requires a POSIX host")
        docker = shutil.which("docker")
        if not docker:
            raise SandboxError("Docker CLI not found; refusing host execution")
        image = self.settings.sandbox_image.strip()
        if not image or any(ch.isspace() for ch in image):
            raise SandboxError("SANDBOX_IMAGE must be a non-empty image reference")
        env = self._docker_environment()
        try:
            info = subprocess.run(
                [docker, "info", "--format", "{{json .SecurityOptions}}"],
                capture_output=True, text=True, env=env, timeout=10, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SandboxError(f"cannot inspect Docker daemon: {exc}") from exc
        if info.returncode != 0:
            raise SandboxError(f"Docker daemon unavailable: {info.stderr[-500:]}")
        if self.settings.sandbox_require_rootless and "rootless" not in info.stdout.lower():
            raise SandboxError("Docker daemon is not rootless; refusing to run untrusted code")
        try:
            inspected = subprocess.run(
                [docker, "image", "inspect", image],
                capture_output=True, text=True, env=env, timeout=10, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SandboxError(f"cannot inspect sandbox image: {exc}") from exc
        if inspected.returncode != 0:
            raise SandboxError(
                f"sandbox image '{image}' is not available locally; build it before running agents"
            )
        return docker, env

    def build_command(
        self, docker: str, workspace: Path, command: Sequence[str], cidfile: Path,
    ) -> list[str]:
        root = workspace.resolve()
        if "," in str(root):
            raise SandboxError("workspace path cannot contain commas for Docker bind mount")
        cpu_millis = self.settings.sandbox_cpu_millis
        memory_mb = self.settings.sandbox_memory_mb
        pids = self.settings.sandbox_pids_limit
        tmpfs_mb = self.settings.sandbox_tmpfs_mb
        if not 100 <= cpu_millis <= 8000:
            raise SandboxError("SANDBOX_CPU_MILLIS must be between 100 and 8000")
        if not 128 <= memory_mb <= 32768:
            raise SandboxError("SANDBOX_MEMORY_MB must be between 128 and 32768")
        if not 16 <= pids <= 1024:
            raise SandboxError("SANDBOX_PIDS_LIMIT must be between 16 and 1024")
        if not 16 <= tmpfs_mb <= 2048:
            raise SandboxError("SANDBOX_TMPFS_MB must be between 16 and 2048")
        return [
            docker, "run", "--rm", "--init", "--cidfile", str(cidfile),
            "--network=none", "--read-only",
            "--cap-drop=ALL", "--security-opt=no-new-privileges:true",
            f"--pids-limit={pids}", f"--memory={memory_mb}m",
            f"--cpus={cpu_millis / 1000:.3f}",
            "--ipc=private", "--ulimit=nofile=256:256",
            f"--tmpfs=/tmp:rw,noexec,nosuid,nodev,size={tmpfs_mb}m",
            "--tmpfs=/home/agent:rw,noexec,nosuid,nodev,size=16m",
            # In a rootless daemon, container UID 0 maps to the invoking host user.
            # All Linux capabilities are dropped and privilege escalation is disabled.
            "--user=0:0",
            "--env=HOME=/home/agent", "--env=PYTHONNOUSERSITE=1",
            "--env=PYTHONDONTWRITEBYTECODE=1", "--env=PYTHONPATH=/workspace",
            "--mount", f"type=bind,source={root},target=/workspace",
            "--workdir=/workspace", self.settings.sandbox_image,
            "python", *command[1:],
        ]

    @staticmethod
    def _tail(stream, limit: int = 4000) -> str:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        stream.seek(max(0, size - limit))
        return stream.read()

    def _stop_container(self, docker: str, env: dict[str, str], cidfile: Path) -> None:
        try:
            container_id = cidfile.read_text(encoding="utf-8").strip()
        except OSError:
            return
        if not container_id:
            return
        try:
            subprocess.run(
                [docker, "stop", "--time", "1", container_id],
                capture_output=True, text=True, env=env, timeout=5, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass

    def run(
        self, workspace: Path, command: Sequence[str], timeout: int,
    ) -> tuple[int, str]:
        docker, env = self._preflight()
        cid_dir = self.settings.data_dir / "sandbox"
        cid_dir.mkdir(parents=True, exist_ok=True)
        cidfile = cid_dir / f"{uuid.uuid4().hex}.cid"
        docker_cmd = self.build_command(docker, workspace, command, cidfile)
        with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stdout, tempfile.TemporaryFile(
            mode="w+t", encoding="utf-8"
        ) as stderr:
            try:
                proc = subprocess.Popen(
                    docker_cmd, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                    text=True, env=env, start_new_session=True,
                )
            except OSError as exc:
                raise SandboxError(f"cannot start Docker sandbox: {exc}") from exc
            try:
                proc.wait(timeout=timeout)
                return proc.returncode, self._tail(stdout) + self._tail(stderr)
            except subprocess.TimeoutExpired:
                self._stop_container(docker, env, cidfile)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(proc.pid, 9)
                    except ProcessLookupError:
                        pass
                    proc.kill()
                    proc.wait(timeout=5)
                return 124, (self._tail(stdout) + self._tail(stderr) + "\ntimeout")[-4000:]
            finally:
                try:
                    cidfile.unlink(missing_ok=True)
                except OSError:
                    pass
