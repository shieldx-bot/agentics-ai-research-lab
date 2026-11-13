"""Central configuration: .env file + env vars + configs/lab.yaml."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


_ENV_MTIME: float | None = None


def _load_dotenv() -> None:
    """Load ${PROJECT_ROOT}/.env into os.environ (idempotent, mtime-cached).

    Precedence: real environment variables ALWAYS win over .env values, so
    exporting GITHUB_TOKEN=... in the shell still overrides the file.
    Called from _env/_env_int/_env_bool so it always runs BEFORE the field
    defaults of ``Settings`` are evaluated (dataclass default factories run
    before ``__post_init__``).
    """
    global _ENV_MTIME
    env_file = PROJECT_ROOT / ".env"
    if not env_file.is_file():
        return
    try:
        mtime = env_file.stat().st_mtime
    except OSError:
        return
    if _ENV_MTIME is not None and mtime == _ENV_MTIME:
        return
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(env_file, override=False)
    except ImportError:
        # Fallback: minimal KEY=VALUE parser (no dependency required).
        for raw in env_file.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    _ENV_MTIME = mtime


def _env(key: str, default: str = "") -> str:
    _load_dotenv()
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    _load_dotenv()
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _env_bool(key: str, default: bool) -> bool:
    _load_dotenv()
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_float(key: str, default: float) -> float:
    _load_dotenv()
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


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
    auto_merge: bool = field(default_factory=lambda: _env_bool("AUTO_MERGE", False))
    claim_ttl_minutes: int = field(
        default_factory=lambda: _env_int("CLAIM_TTL_MINUTES", 45)
    )
    # LLM backend mode (see docs/LLM_PROVIDERS.md):
    # "perplexity" (default) = Perplexity-only: EVERY node uses the Perplexity
    #   cookie pool (citations everywhere); Qwen is never called.
    # "qwen" = hybrid: research -> Perplexity (citations); plan/implement/review
    #   -> Qwen qwen4.0-max-preview, with cross-provider fallback.
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "perplexity"))
    qwen_model: str = field(default_factory=lambda: _env("QWEN_MODEL", "qwen4.0-max-preview"))
    # Self-evolution: Supervisor opens an evidence-based "self-evolution" issue every
    # N of its cycles (0 disables). See docs/EVOLUTION.md.
    evolution_every_n_cycles: int = field(
        default_factory=lambda: _env_int("EVOLUTION_EVERY_N_CYCLES", 10)
    )
    # Research domain
    research_topic: str = field(
        default_factory=lambda: _env(
            "RESEARCH_TOPIC",
            "AI/ML for Network Function Virtualization (NFV) and 5G core automation",
        )
    )
    # GitHub rate-limit safety: process-wide minimum interval between API
    # calls (all agent threads share one PAT) and TTL for the shared issue
    # scan cache. 0 disables pacing (not recommended for >2 agents).
    gh_min_api_interval: float = field(
        default_factory=lambda: _env_float("GH_MIN_API_INTERVAL", 1.0)
    )
    gh_scan_cache_ttl: float = field(
        default_factory=lambda: _env_float("GH_SCAN_CACHE_TTL", 60)
    )
    # Paths
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    accounts_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "accounts")
    # P10 — the lab's own code that agents must never modify. Owner can extend
    # via configs/lab.yaml (protected_paths:) or ALLOW_PROTECTED_PATHS=1 env
    # to grant a run-wide override (see agentics/policies/guard.py).
    protected_paths: list[str] = field(default_factory=lambda: [
        p for p in _env("PROTECTED_PATHS",
                        "agentics/,configs/,accounts/,scripts/,.github/").split(",")
        if p.strip()
    ])
    # P14 — optional whitelist ring ("cấm tuyệt đối"): when non-empty, ANY
    # file outside these roots is refused for every agent (owner exemption
    # per task still lifts it; credentials are never writable). Empty = off.
    # Default lives in configs/lab.yaml as the union of the ownership surfaces.
    writable_roots: list[str] = field(default_factory=lambda: [
        p for p in _env("WRITABLE_ROOTS", "").split(",") if p.strip()
    ])
    # P13 — per-file lock TTL in the swarm (anti "hai agent đè tay nhau").
    file_lock_ttl_minutes: int = field(
        default_factory=lambda: _env_int("FILE_LOCK_TTL_MINUTES", 30))
    # P18 — peer panel review (đánh giá lẫn nhau): how many cross-reviewers
    # must unanimously accept before an MR passes, and how long the author has
    # to defend their point once ONE reject vote opens the defense round.
    peer_panel_size: int = field(
        default_factory=lambda: _env_int("PEER_PANEL_SIZE", 2))
    peer_defense_ttl_minutes: int = field(
        default_factory=lambda: _env_int("PEER_DEFENSE_TTL_MINUTES", 30))
    # P10 layer 4 — integrity self-healing sweep interval (0 = off).
    integrity_interval_s: int = field(
        default_factory=lambda: _env_int("INTEGRITY_INTERVAL_S", 20))
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
    # Browser backend for the Perplexity pool (see docs/LLM_PROVIDERS.md):
    # "local" (default) = raw cookie WebSocket from THIS machine — works only
    #   from residential IPs; Perplexity 403-blocks datacenter IPs.
    # "hyperbrowser" = browser runs on Hyperbrowser cloud (stealth + US
    #   residential proxy) via Playwright CDP; works from any VPS.
    browser_backend: str = field(default_factory=lambda: _env("BROWSER_BACKEND", "local"))
    hyperbrowser_api_key: str = field(default_factory=lambda: _env("HB_API_KEY"))
    hyperbrowser_proxy_country: str = field(
        default_factory=lambda: _env("HB_PROXY_COUNTRY", "US"))
    hyperbrowser_timeout: int = field(
        default_factory=lambda: _env_int("HB_TIMEOUT", 900))
    # How long the cloud browser waits for the Perplexity composer to become
    # visible before failing an ask with a diagnostic error (challenge page,
    # logged-out wall, or a UI markup change).
    hb_input_timeout: int = field(default_factory=lambda: _env_int("HB_INPUT_TIMEOUT", 45))

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

    @property
    def local_repo_path(self) -> Path:
        """The bare repo that plays the role of "origin" in local-only mode."""
        return self.data_dir / "lab.git"

    @property
    def integrity_path(self) -> Path:
        """Baseline of the runtime tree's hashes (`agentics verify`, layer 4)."""
        return self.data_dir / "integrity" / "baseline.json"

    @property
    def dashboard_port(self) -> int:
        return _env_int("DASHBOARD_PORT", 8080)

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.chroma_path, self.workspaces_dir):
            d.mkdir(parents=True, exist_ok=True)


def load_yaml_overrides(settings: Settings) -> Settings:
    """Merge configs/lab.yaml over env-derived settings (yaml wins where present).

    configs/secrets.yaml (gitignored) is merged last and wins over everything
    else — it holds per-deployment credentials such as ``hyperbrowser_api_key``.
    """
    for name in ("lab.yaml", "secrets.yaml"):
        cfg_path = PROJECT_ROOT / "configs" / name
        if not cfg_path.exists():
            continue
        with open(cfg_path, "r", encoding="utf-8") as fh:
            data: dict[str, Any] = yaml.safe_load(fh) or {}
        for key, value in data.items():
            if hasattr(settings, key) and value is not None:
                setattr(settings, key, value)
    return settings


def get_settings() -> Settings:
    """Build settings from env, then apply YAML overrides."""
    return load_yaml_overrides(Settings())
