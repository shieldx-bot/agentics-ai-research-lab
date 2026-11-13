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


def _hb_api_keys() -> list[str]:
    """Hyperbrowser token pool: HB_API_KEYS (comma-separated) wins, then the
    single HB_API_KEY. Empty list = no tokens (session creation will fail
    with the SDK's own clear error, same as before)."""
    raw = _env("HB_API_KEYS", "")
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    if not keys:
        single = _env("HB_API_KEY", "")
        keys = [single] if single else []
    return keys


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
    # P19 — peer panel review (đánh giá lẫn nhau): how many cross-reviewers
    # must unanimously accept before an MR passes, and how long the author has
    # to defend their point once ONE reject vote opens the defense round.
    peer_panel_size: int = field(
        default_factory=lambda: _env_int("PEER_PANEL_SIZE", 2))
    peer_defense_ttl_minutes: int = field(
        default_factory=lambda: _env_int("PEER_DEFENSE_TTL_MINUTES", 30))
    # P20 — LLM budget: pool-wide hourly cap (0 = off) + per-agent circuit
    # breaker after N consecutive provider failures. Defaults bound a
    # runaway agent without throttling a healthy swarm of 10.
    llm_max_calls_per_hour: int = field(
        default_factory=lambda: _env_int("LLM_MAX_CALLS_PER_HOUR", 120))
    llm_fail_streak_limit: int = field(
        default_factory=lambda: _env_int("LLM_FAIL_STREAK_LIMIT", 5))
    llm_breaker_cooldown_s: int = field(
        default_factory=lambda: _env_int("LLM_BREAKER_COOLDOWN_S", 900))
    # P22 — retention: workspaces of agents inactive longer than this AND
    # with a clean git tree get pruned by the supervisor sweep (0 = off).
    workspace_retention_minutes: int = field(
        default_factory=lambda: _env_int("WORKSPACE_RETENTION_MINUTES", 240))
    # P10 layer 4 — integrity self-healing sweep interval (0 = off).
    integrity_interval_s: int = field(
        default_factory=lambda: _env_int("INTEGRITY_INTERVAL_S", 20))
    # P23 — replay: settled claims are periodically re-run from their stored
    # protocol source by the supervisor sweep. replay_sweep_size caps how many
    # claims one sweep replays (0 = off); replay_every_n_cycles gates how
    # often the sweep runs. Every outcome lands in shared evidence
    # (method='replay') so the whole swarm sees replication results (H3).
    replay_sweep_size: int = field(
        default_factory=lambda: _env_int("REPLAY_SWEEP_SIZE", 2))
    replay_every_n_cycles: int = field(
        default_factory=lambda: _env_int("REPLAY_EVERY_N_CYCLES", 3))
    # P23 flood control: a claim replayed again within cooldown_s (or beyond
    # replay_max_per_hyp total replays) adds no information — the sweep skips
    # it. Live 2026-11-14: two replayable claims were re-run every sweep by
    # both supervisors -> hundreds of duplicate evidence rows.
    replay_cooldown_s: int = field(
        default_factory=lambda: _env_int("REPLAY_COOLDOWN_S", 1800))
    replay_max_per_hyp: int = field(
        default_factory=lambda: _env_int("REPLAY_MAX_PER_HYP", 10))
    # P24bis — independent replication: reviewers re-run settled claims from
    # their stored protocols in their own workspaces. replication_limit caps
    # how many claims ONE reviewer cycle replicates (0 = off). No LLM is
    # involved: the criterion is frozen data and the verdict is machine-made.
    replication_limit: int = field(
        default_factory=lambda: _env_int("REPLICATION_LIMIT", 1))
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
    # HB_API_KEYS: comma-separated Hyperbrowser tokens. Each token owns ONE
    # cloud session, so K tokens scale the swarm to K truly parallel agents
    # (per-token session locks replace the old global single-session lock).
    # Falls back to [hyperbrowser_api_key]; secrets.yaml may override with a
    # YAML list.
    hyperbrowser_api_keys: list[str] = field(default_factory=lambda: _hb_api_keys())
    hyperbrowser_proxy_country: str = field(
        default_factory=lambda: _env("HB_PROXY_COUNTRY", "US"))
    hyperbrowser_timeout: int = field(
        default_factory=lambda: _env_int("HB_TIMEOUT", 900))
    # How long the cloud browser waits for the Perplexity composer to become
    # visible before failing an ask with a diagnostic error (challenge page,
    # logged-out wall, or a UI markup change).
    hb_input_timeout: int = field(default_factory=lambda: _env_int("HB_INPUT_TIMEOUT", 45))
    # --- Second-judge provider (dual-model de-correlation) -------------------
    # When "chatgpt", tournament debate asks run a SECOND, independent model
    # family (ChatGPT via its own cookie session on the same Hyperbrowser
    # edge) alongside the primary Perplexity jury. Both verdicts must agree
    # before Elo moves; disagreement records a `contested` match (no Elo
    # movement) for the owner. "" keeps the single-jury behaviour.
    judge_second_provider: str = field(
        default_factory=lambda: _env("JUDGE_SECOND_PROVIDER", ""))
    chatgpt_cookie_path: str = field(
        default_factory=lambda: _env("CHATGPT_COOKIE_PATH", ""))
    # Free-plan Hyperbrowser sessions carry a hard duration cap (~10 min;
    # live 2026-11-13: six sessions aged 9m37s-10m06s all died mid-ask inside
    # one 30-second window, wrecking every in-flight call). Recycle each
    # cloud session BEFORE the cap — between asks, while healthy — instead
    # of losing everything at once. The client jitters this ±10-15% so a
    # swarm never recycles in sync. 0 disables the guard.
    hb_session_recycle_s: int = field(
        default_factory=lambda: _env_int("HB_SESSION_RECYCLE_S", 480))

    # --- Adaptive planning & role prompts (LLM-authored) ---------------------
    # Role missions are WRITTEN by the LLM (shared swarm-wide via the
    # role_prompts table, reworked after repeated failures, conditioned on
    # recent traces); plans are PROPOSED by the LLM (approach-first) on top
    # of the minimal machine contract. False = the static pre-adaptive lab.
    adaptive_prompts: bool = field(
        default_factory=lambda: _env_bool("ADAPTIVE_PROMPTS", True))
    role_prompt_ttl_hours: float = field(
        default_factory=lambda: _env_float("ROLE_PROMPT_TTL_HOURS", 24.0))
    role_prompt_max_chars: int = field(
        default_factory=lambda: _env_int("ROLE_PROMPT_MAX_CHARS", 1600))
    role_prompt_rework_failures: int = field(
        default_factory=lambda: _env_int("ROLE_PROMPT_REWORK_FAILURES", 2))
    # --- Coordination pack A2X (docs/COORDINATION.md) ------------------------
    # Contract-Net-lite: rank bonus a claim adds to a task the supervisor's
    # scan awarded to THIS agent (0 disables the allocation signal entirely;
    # the awards table is still maintained). ~0.75 flips a ranking the way a
    # priority label would, without overriding the agent's own profile.
    coord_award_bonus: float = field(
        default_factory=lambda: _env_float("COORD_AWARD_BONUS", 0.75))
    # Free keyless research tools (docs/FREE_TOOLS.md): arXiv / OpenAlex /
    # Crossref / HuggingFace datasets / Wikipedia / DuckDuckGo. They ground
    # research in real publisher metadata and keep node_research alive when
    # every LLM cookie is dead. No account or token required for any source.
    free_tools_enabled: bool = field(default_factory=lambda: _env_bool("FREE_TOOLS_ENABLED", True))
    free_tools_timeout: float = field(
        default_factory=lambda: _env_float("FREE_TOOLS_TIMEOUT", 15.0)
    )
    # Optional polite-pool contact for OpenAlex/Crossref (a plain email).
    # Not required, but it puts the calls in the friendlier rate-limit tier.
    free_tools_mailto: str = field(default_factory=lambda: _env("FREE_TOOLS_MAILTO"))
    # P18b research fan-out: when > 1, node_research asks complementary
    # angles (survey / benchmarks / open problems) IN PARALLEL — each extra
    # angle runs on a different Hyperbrowser token of the pool (or a
    # different cookie account on the local backend). Results merge into one
    # finding; every angle still passes the P20 budget gate and the fan-out
    # is journalled. 1 keeps the single-ask behaviour.
    research_fanout: int = field(default_factory=lambda: _env_int("RESEARCH_FANOUT", 1))
    # Solo-lab unblock: when the pool has fewer identities than angles
    # (e.g. one Hyperbrowser token), the extra angle still runs — SEQUENTIALLY
    # on the agent's own pinned view, never parallel on the same session.
    # Off restores the strict single-ask behaviour for 1-identity pools.
    research_same_view_angle: bool = field(
        default_factory=lambda: _env_bool("RESEARCH_SAME_VIEW_ANGLE", True))
    # --- Co-Scientist tournament (agentics/coscientist, ported from the
    # Apache-2.0 Kaimen-Inc re-implementation of Gottweis et al. 2026) ---
    # The Elo tournament over hypotheses: a QUALITY dimension on top of the
    # P16 truth dimension. Every match and evolution ask passes the P20
    # budget gate; 0 matches/cycle or enabled=False disables the loop.
    tournament_enabled: bool = field(
        default_factory=lambda: _env_bool("TOURNAMENT_ENABLED", True))
    tournament_matches_per_cycle: int = field(
        default_factory=lambda: _env_int("TOURNAMENT_MATCHES_PER_CYCLE", 2))
    evolution_per_cycle: int = field(
        default_factory=lambda: _env_int("EVOLUTION_PER_CYCLE", 1))
    evolution_top_k: int = field(
        default_factory=lambda: _env_int("EVOLUTION_TOP_K", 3))
    # Meta-review feedback synthesis: every N decisive debates the newest
    # win/loss patterns are distilled into a steering note (one more ask).
    meta_review_window: int = field(
        default_factory=lambda: _env_int("META_REVIEW_WINDOW", 4))
    # P16-R (Reflection on failed protocols): when a hypothesis protocol is
    # inconclusive, the author gets N bounded self-repair rounds — the
    # machine failure note is fed back and the protocol is rewritten ONCE,
    # then re-adjudicated by the same frozen criterion. 0 = off.
    verify_protocol_repair_rounds: int = field(
        default_factory=lambda: _env_int("VERIFY_PROTOCOL_REPAIR_ROUNDS", 1))

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
