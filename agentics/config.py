"""Central configuration: env vars + configs/lab.yaml."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    # GitHub
    github_token: str = field(default_factory=lambda: _env("GITHUB_TOKEN"))
    github_repo: str = field(
        default_factory=lambda: _env("GITHUB_REPO", "shieldx-bot/agentics-ai-research-lab")
    )
    # Agents
    max_agents: int = field(default_factory=lambda: _env_int("MAX_AGENTS", 10))
    cycle_pause_seconds: int = field(
        default_factory=lambda: _env_int("CYCLE_PAUSE_SECONDS", 60)
    )
    auto_merge: bool = field(default_factory=lambda: _env_bool("AUTO_MERGE", True))
    claim_ttl_minutes: int = field(
        default_factory=lambda: _env_int("CLAIM_TTL_MINUTES", 45)
    )
    # Self-evolution: Supervisor opens an evidence-based "self-evolution" issue every
    # N of its cycles (0 disables). See docs/EVOLUTION.md.
    evolution_every_n_cycles: int = field(
        default_factory=lambda: _env_int("EVOLUTION_EVERY_N_CYCLES", 10)
    )
    # Untrusted code execution: fail closed when Docker sandbox is unavailable.
    sandbox_enabled: bool = field(default_factory=lambda: _env_bool("SANDBOX_ENABLED", True))
    sandbox_image: str = field(
        default_factory=lambda: _env("SANDBOX_IMAGE", "agentics-sandbox:py311")
    )
    sandbox_require_rootless: bool = field(
        default_factory=lambda: _env_bool("SANDBOX_REQUIRE_ROOTLESS", True)
    )
    sandbox_cpu_millis: int = field(default_factory=lambda: _env_int("SANDBOX_CPU_MILLIS", 1000))
    sandbox_memory_mb: int = field(default_factory=lambda: _env_int("SANDBOX_MEMORY_MB", 1024))
    sandbox_pids_limit: int = field(default_factory=lambda: _env_int("SANDBOX_PIDS_LIMIT", 128))
    sandbox_tmpfs_mb: int = field(default_factory=lambda: _env_int("SANDBOX_TMPFS_MB", 128))
    # Research domain
    research_topic: str = field(
        default_factory=lambda: _env(
            "RESEARCH_TOPIC",
            "AI/ML for Network Function Virtualization (NFV) and 5G core automation",
        )
    )
    # Paths
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    accounts_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "accounts")
    # Perplexity
    pplx_ws_url: str = field(
        default_factory=lambda: _env(
            "PPLX_WS_URL", "wss://www.perplexity.ai/socket.io/?EIO=4&transport=websocket"
        )
    )
    pplx_origin: str = field(
        default_factory=lambda: _env("PPLX_ORIGIN", "https://www.perplexity.ai")
    )
    pplx_timeout: int = field(default_factory=lambda: _env_int("PPLX_TIMEOUT", 180))

    @property
    def owner(self) -> str:
        return self.github_repo.split("/")[0]

    @property
    def repo_name(self) -> str:
        return self.github_repo.split("/")[-1]

    @property
    def stop_file(self) -> Path:
        return PROJECT_ROOT / "STOP"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "shared_memory.db"

    @property
    def chroma_path(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def workspaces_dir(self) -> Path:
        return self.data_dir / "workspaces"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.chroma_path, self.workspaces_dir):
            d.mkdir(parents=True, exist_ok=True)


def load_yaml_overrides(settings: Settings) -> Settings:
    """Merge configs/lab.yaml over env-derived settings (yaml wins where present)."""
    cfg_path = PROJECT_ROOT / "configs" / "lab.yaml"
    if not cfg_path.exists():
        return settings
    with open(cfg_path, "r", encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}
    for key, value in data.items():
        if hasattr(settings, key) and value is not None:
            setattr(settings, key, value)
    return settings


def get_settings() -> Settings:
    """Build settings from env, then apply YAML overrides."""
    return load_yaml_overrides(Settings())
