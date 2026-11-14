"""Central configuration: env vars (+ optional legacy .env) and ONE config
file — configs/secrets.yaml (gitignored, the only file an operator edits).
configs/lab.yaml is a deprecated alias still read for backward compat."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Supervisor dispatch: keep at least this many unclaimed tasks in the local
# backlog so free coders/writers never idle behind an empty or thin queue.
# 0 disables the top-up (scan-only supervisor). Used as the Settings default
# AND as the fallback for stripped-down test settings namespaces.
DEFAULT_BACKLOG_WATERMARK = 2


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


def _env_json(key: str, default: dict) -> dict:
    """A dict-valued setting via one JSON env var (MCP_SERVERS etc.)."""
    _load_dotenv()
    raw = os.environ.get(key, "")
    if not raw:
        return default
    try:
        val = json.loads(raw)
        return val if isinstance(val, dict) else default
    except (ValueError, TypeError):
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
    # Supervisor backlog watermark: when a scan sees fewer unclaimed tasks
    # than this, the supervisor tops the backlog up to the watermark using
    # its existing sources (failure investigation, survey rotation). Replaces
    # the old "only when COMPLETELY empty, one task per cycle" rule, which
    # starved parallel coders. 0 disables.
    backlog_watermark: int = field(
        default_factory=lambda: _env_int("BACKLOG_WATERMARK", DEFAULT_BACKLOG_WATERMARK)
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
    # Owner dashboard bind port (python -m agentics.cli dashboard) — env
    # DASHBOARD_PORT still wins over this default; configs/secrets.yaml can
    # pin it per deployment (dashboard_port:).
    dashboard_port: int = field(
        default_factory=lambda: _env_int("DASHBOARD_PORT", 8080))
    # P10 — the lab's own code that agents must never modify. Owner can extend
    # via configs/secrets.yaml (protected_paths:) or ALLOW_PROTECTED_PATHS=1 env
    # to grant a run-wide override (see agentics/policies/guard.py).
    protected_paths: list[str] = field(default_factory=lambda: [
        p for p in _env("PROTECTED_PATHS",
                        "agentics/,configs/,accounts/,scripts/,.github/").split(",")
        if p.strip()
    ])
    # P14 — optional whitelist ring ("cấm tuyệt đối"): when non-empty, ANY
    # file outside these roots is refused for every agent (owner exemption
    # per task still lifts it; credentials are never writable). Empty = off.
    # Default = the union of the ownership surfaces (was configs/lab.yaml).
    writable_roots: list[str] = field(default_factory=lambda: [
        p for p in _env("WRITABLE_ROOTS",
                        "research_notes/,experiments/,benchmarks/,tests/,"
                        "docs/,reports/,README.md").split(",")
        if p.strip()
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
    # P20b — science reserve: the last slice of the pool cap that ONLY the
    # reserve purposes may spend, so plumbing (plan/review/research fan-out)
    # can never starve the verify/ideation escape hatch the way the live
    # 07:08 run starved it (previous hour drained the pool, the saturation
    # pivot itself was refused). 0 = off; purposes are a comma-separated tag
    # list matched against the ask's `purpose`.
    budget_reserve_pct: float = field(
        default_factory=lambda: _env_float("BUDGET_RESERVE_PCT", 0.25))
    budget_reserve_purposes: str = field(
        default_factory=lambda: _env("BUDGET_RESERVE_PURPOSES",
                                     "verify,ideation"))
    # Max seconds a blocked reserve-purpose ask may sleep inside _llm_ask
    # and retry ONCE (the window slides continuously; waiting beats dying
    # for the whole hour). 0 = never wait.
    budget_block_wait_cap_s: float = field(
        default_factory=lambda: _env_float("BUDGET_BLOCK_WAIT_CAP_S", 45.0))
    # TTT_DISCOVER.md C2/C3 config — PUCT frontier + entropic credit.
    puct_c: float = field(default_factory=lambda: _env_float("PUCT_C", 1.0))
    entropic_bonus_max: float = field(
        default_factory=lambda: _env_float("ENTROPIC_BONUS_MAX", 8.0))
    reward_scale: str = field(
        default_factory=lambda: _env("REWARD_SCALE", "target"))
    # Wave-2 #3 — EIG frontier: expected-information-gain-per-cost refines
    # the PUCT order (Beta-Bernoulli priors from the experiment ledger;
    # closed-form entropy reduction). 0 = pure PUCT order.
    eig_weight: float = field(
        default_factory=lambda: _env_float("EIG_WEIGHT", 0.5))
    # Local small-model router (OmniRoute / vLLM / LM Studio via the
    # OpenAI-compatible chat-completions API; native Ollama kept as a legacy
    # backend) — the FREE edge for cheap purposes so the cookie pool is
    # spent only on science. Disabled by default: no surprise network calls
    # to localhost.
    local_llm_enabled: bool = field(
        default_factory=lambda: _env_bool("LOCAL_LLM_ENABLED", False))
    # OmniRoute (self-hosted gateway, `npm i -g omniroute` -> :20128/v1)
    # serves an OpenAI-compatible endpoint that routes to NVIDIA NIM,
    # vLLM, LM Studio, llama.cpp and Ollama alike. `auto` decides by the
    # URL shape: a base ending in /v1 speaks OpenAI chat completions, a
    # bare host:port speaks Ollama native /api/chat.
    local_llm_url: str = field(
        default_factory=lambda: _env("LOCAL_LLM_URL",
                                     "http://127.0.0.1:20128/v1"))
    # auto | omniroute | ollama — pin the protocol instead of the URL
    # heuristic (omniroute/openai both mean the OpenAI-compatible path).
    local_llm_backend: str = field(
        default_factory=lambda: _env("LOCAL_LLM_BACKEND", "auto"))
    # NVIDIA Nemotron 3 Ultra 550B A55B (hybrid Mamba-Transformer MoE, 1M
    # context, function calling) is the intended local-tier brain behind
    # OmniRoute/NIM; set LOCAL_LLM_MODEL to any tag the gateway lists
    # (`GET /v1/models`), e.g. a small qwen3 tag for CPU-only Ollama.
    local_llm_model: str = field(
        default_factory=lambda: _env("LOCAL_LLM_MODEL",
                                     "nvidia/nemotron-3-ultra-550b-a55b"))
    # Vision tier (Nemotron 3.5 Super VL): reserved for image-bearing
    # purposes — the graph has no image inputs yet, so nothing routes here
    # until a vision purpose lands; the knob just pins the model id.
    local_llm_vl_model: str = field(
        default_factory=lambda: _env("LOCAL_LLM_VL_MODEL"))
    # Optional bearer token for the local gateway itself (LOCAL_LLM_API_KEY,
    # mirrored in configs/secrets.yaml). OmniRoute ships an "API Keys" page —
    # required for LAN/tunnel access (the lab on another machine talking to
    # 192.168.x.x:20128 or a Cloud tunnel), optional for plain localhost.
    # Empty = NO Authorization header at all, exactly what an unauthenticated
    # localhost gateway expects.
    local_llm_api_key: str = field(
        default_factory=lambda: _env("LOCAL_LLM_API_KEY"))
    # CPU-realistic defaults (live 2026-11-14 08:51: the first /api/chat
    # also LOADS the model into RAM and qwen3 would think/generate for
    # minutes — 20s timed out with an empty error string every cycle).
    local_llm_timeout: float = field(
        default_factory=lambda: _env_float("LOCAL_LLM_TIMEOUT", 120.0))
    # Bounded generation: plumbing answers (triples, plans, prompts) never
    # need 2048 tokens — cap it so CPU inference finishes in seconds.
    local_llm_num_predict: int = field(
        default_factory=lambda: _env_int("LOCAL_LLM_NUM_PREDICT", 512))
    # Max CONCURRENT local completions (default 1 = serialized). Live
    # 2026-11-14 21:25: two same-second lab asks hit the OmniRoute gateway
    # and BOTH came back HTTP 502, while a single dashboard-chat completion
    # kept working — the upstream serves one request at a time, so the lab
    # must queue instead of burst. Raise only if your backend provably
    # handles parallel completions (probe --ask --concurrency 3).
    local_llm_max_concurrent: int = field(
        default_factory=lambda: _env_int("LOCAL_LLM_MAX_CONCURRENT", 1))
    # Hard context cap per request (live 2026-11-14 10:35 + 13:43: the
    # kernel OOM-killed llama-server TWICE). llama.cpp's prompt cache
    # grows ~per DISTINCT prompt, and each slot's size scales with the
    # context window — the 10:35 journal showed ~238 MiB slots x ~30 LRU
    # entries ≈ 7 GB RSS → global OOM (upstream cap ~8 GiB, NOT
    # configurable: ollama#17351). Pinning num_ctx to 4096 shrinks every
    # slot to a fraction of that and matches the lab's prompt budget
    # (digests are capped at 5000 chars, the chat prompt design assumes
    # "qwen3's 4096 ctx"). Raise only with free RAM to spare.
    local_llm_num_ctx: int = field(
        default_factory=lambda: _env_int("LOCAL_LLM_NUM_CTX", 4096))
    # qwen3-family thinking models: <think> blocks are stripped from the
    # answer regardless; this ALSO asks the runtime to skip thinking
    # entirely (Ollama >= 0.9; older runtimes just ignore the field).
    local_llm_think: bool = field(
        default_factory=lambda: _env_bool("LOCAL_LLM_THINK", False))
    # Request-level keep_alive (overrides any OLLAMA_KEEP_ALIVE server
    # default). Default "5m": the model unloads and frees model+prompt
    # cache on 5 minutes of idle — a GPU reload costs only seconds, while
    # the OOM that killed llama-server twice on 2026-11-14 (10:35 and
    # again before the 13:43:34 restart) costs the whole local edge for
    # hours. "30m"/"1h" keeps the model warm instead on RAM-roomy hosts;
    # see local_llm_num_ctx for the other half of the cache fix.
    local_llm_keep_alive: str = field(
        default_factory=lambda: _env("LOCAL_LLM_KEEP_ALIVE", "5m"))
    # Comma-separated purposes that may route to the local model. Science
    # purposes (research/verify/ideation/implement/review/repair/defend/
    # tournament) can NEVER be listed here — the router refuses them even
    # if someone puts them in the env. plan/role_synthesis are scaffolding:
    # allowed as a P20c budget FALLBACK so a dry pool degrades to a local
    # plan instead of an idle agent (2026-11-14 08:35 plan-starvation).
    local_llm_purposes: str = field(
        default_factory=lambda: _env("LOCAL_LLM_PURPOSES",
                                     "kg_extract,plan,role_synthesis,"
                                     "prompt_evolve,reflect,skill_distill"))
    # GraphRAG-lite: mine findings into the knowledge graph when the local
    # router is configured; ingest is a no-op without it.
    kg_enabled: bool = field(default_factory=lambda: _env_bool("KG_ENABLED", True))
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
    # QD diversity (DAPS / IDEAgent): a new hypothesis whose word-Jaccard
    # with a recent one exceeds diversity_jaccard_max is rejected as a
    # near-clone (PEM); the supervisor's diversity_audit alerts when
    # mechanism entropy over the last 20 hypotheses drops below
    # diversity_min_entropy nats (algorithmic mode collapse alarm).
    diversity_jaccard_max: float = field(
        default_factory=lambda: _env_float("DIVERSITY_JACCARD_MAX", 0.8))
    diversity_min_entropy: float = field(
        default_factory=lambda: _env_float("DIVERSITY_MIN_ENTROPY", 1.0))
    # Kosmos world model: how many world_notes rows the agent snapshots
    # carry into prompts (bounded — the model is a summary, not a ledger).
    world_context_notes: int = field(
        default_factory=lambda: _env_int("WORLD_CONTEXT_NOTES", 12))
    # P24bis — independent replication: reviewers re-run settled claims from
    # their stored protocols in their own workspaces. replication_limit caps
    # how many claims ONE reviewer cycle replicates (0 = off). No LLM is
    # involved: the criterion is frozen data and the verdict is machine-made.
    replication_limit: int = field(
        default_factory=lambda: _env_int("REPLICATION_LIMIT", 1))
    # QD ideation closed loop (IDEAgent gate + Kosmos harvest, 2026-11-14):
    # AGENTICS_IDEATION=0 turns the whole loop off (no ideation LLM calls);
    # idea_min_score is the IDEAgent qualification gate tau (0-100);
    # idea_repair_margin routes near-misses into ONE repair round;
    # idea_max_children bounds how many follow-up hypotheses one settled
    # experiment may spawn; idea_max_attempts bounds deep-dive RETRIES of
    # inconclusive protocols (autoresearch: fix-and-rerun, then move on).
    ideation_enabled: bool = field(
        default_factory=lambda: _env("AGENTICS_IDEATION", "1")
        .strip().lower() not in ("0", "false", "no", "off"))
    idea_min_score: int = field(
        default_factory=lambda: _env_int("IDEA_MIN_SCORE", 60))
    idea_repair_margin: int = field(
        default_factory=lambda: _env_int("IDEA_REPAIR_MARGIN", 20))
    # Saturation escape hatch (owner dashboard 2026-11-14: with the topic
    # saturated the gate rejected EVERY gap proposal outright — weak=-25
    # missed the repair band by 5 points and the lab oscillated
    # skip->reject->skip forever). When the world model proposes a
    # gap-closing hypothesis under saturation, the repair band widens to
    # this margin so a near-miss gets its ONE repair round instead of
    # dying. The re-judge after repair still must FULLY pass the gate
    # (qualification with repair_margin=0), so this cannot lower the bar
    # for what gets registered — it only grants the repair attempt.
    idea_repair_margin_saturation: int = field(
        default_factory=lambda: _env_int("IDEA_REPAIR_MARGIN_SATURATION", 30))
    idea_max_children: int = field(
        default_factory=lambda: _env_int("IDEA_MAX_CHILDREN", 2))
    idea_max_attempts: int = field(
        default_factory=lambda: _env_int("IDEA_MAX_ATTEMPTS", 2))
    # Discovery branching (2026-11-14 novelty round): when the ideator grades
    # a settled result's novelty >= discovery_novelty_min (0-100) the result
    # is marked a DISCOVERY and its follow-ups become angled branches the
    # deep-dive backlog prioritizes (frontier). branch_max_depth caps the
    # branch level below the discovery root — past it the harvest stops
    # spawning children and the frontier stops boosting the subtree.
    discovery_novelty_min: int = field(
        default_factory=lambda: _env_int("DISCOVERY_NOVELTY_MIN", 70))
    branch_max_depth: int = field(
        default_factory=lambda: _env_int("BRANCH_MAX_DEPTH", 4))
    # Discovery double-judge gate: mark_discovery fires only when a SECOND,
    # INDEPENDENT account re-grades the result as surprising AND its answer
    # carries at least one external citation URL (an input the authoring
    # agent did not author). Single-account setups therefore never mark
    # discoveries — that is the honest behavior, not a bug (set 0 to
    # restore the single-judge behavior).
    discovery_second_judge: bool = field(
        default_factory=lambda: _env_bool("DISCOVERY_SECOND_JUDGE", True))
    # Settle anchor law (verdict provenance): a verdict may be recorded
    # `supported` ONLY when self_authored_ratio < 1 — at least one input of
    # the verdict (external citation, independent replication, owner
    # approval, pinned-hash data) must be something the agent did NOT
    # author. An anchorless supported is downgraded to `unanchored` and
    # waits for independent replication (P24 promotes it when a peer's
    # re-run agrees). The law guards OPTIMISM, not pessimism: refuted and
    # inconclusive verdicts pass unchanged.
    settle_require_anchor: bool = field(
        default_factory=lambda: _env_bool("SETTLE_REQUIRE_ANCHOR", True))
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
    # "hyperbrowser" (default) = browser runs on Hyperbrowser cloud (stealth
    #   + US residential proxy) via Playwright CDP; works from any VPS.
    # "local" = raw cookie WebSocket from THIS machine — works only
    #   from residential IPs; Perplexity 403-blocks datacenter IPs.
    browser_backend: str = field(default_factory=lambda: _env("BROWSER_BACKEND", "hyperbrowser"))
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
        default_factory=lambda: _env("JUDGE_SECOND_PROVIDER", "chatgpt"))
    chatgpt_cookie_path: str = field(
        default_factory=lambda: _env("CHATGPT_COOKIE_PATH",
                                     "accounts/chatgpt-01.json"))
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
    # Wave-2 #2 — DSPy-lite prompt evolution: the role mission is not just
    # re-synthesized after failures, it EVOLVES. Challenger variants (arms)
    # are served on exploration cycles, scored on real exit-code outcomes,
    # and promoted when they beat the champion's win-rate by a margin.
    # Mutation runs on the LOCAL router (purpose prompt_evolve) — no cookie
    # cost. Requires adaptive_prompts.
    prompt_evolution_enabled: bool = field(
        default_factory=lambda: _env_bool("PROMPT_EVOLUTION", True))
    prompt_variant_pool: int = field(
        default_factory=lambda: _env_int("PROMPT_VARIANT_POOL", 3))
    prompt_explore_p: float = field(
        default_factory=lambda: _env_float("PROMPT_EXPLORE_P", 0.25))
    prompt_variant_promote_min: int = field(
        default_factory=lambda: _env_int("PROMPT_VARIANT_PROMOTE_MIN", 10))
    prompt_variant_promote_edge: float = field(
        default_factory=lambda: _env_float("PROMPT_VARIANT_PROMOTE_EDGE",
                                           0.15))
    # --- Wave-3 advanced add-ons (roadmap #1/#2/#3/#5) ----------------------
    # (#1) GEPA-style reflective regeneration: when the champion mission's
    # recent experience shows failures, the local model first explains WHY
    # (from real traces) and then rewrites the mission against that
    # explanation. Generator only — the lab stays the judge.
    gepa_reflection_enabled: bool = field(
        default_factory=lambda: _env_bool("GEPA_REFLECTION", True))
    # (#2) static cascade: on a pool-side failure (refusal/provider error),
    # allowed purposes get ONE retry against the configured frontier
    # provider. Verdict/jury purposes can NEVER cascade; "" = off.
    llm_cascade_provider: str = field(
        default_factory=lambda: _env("LLM_CASCADE_PROVIDER", ""))
    llm_cascade_purposes: str = field(
        default_factory=lambda: _env("LLM_CASCADE_PURPOSES",
                                     "research,plan,implement"))
    # (#3) MCP bridge: OWNER-APPROVED stdio servers (JSON env, or the
    # tools.mcp_servers: key in lab.yaml). Empty = strict no-op.
    mcp_servers: dict = field(default_factory=lambda: _env_json("MCP_SERVERS",
                                                                {}))
    mcp_timeout_s: float = field(
        default_factory=lambda: _env_float("MCP_TIMEOUT_S", 30.0))
    # (#5) learned risk gate: shadow-only v1 — journals predictions, never
    # blocks. "off" disables even the journaling.
    risk_gate_mode: str = field(
        default_factory=lambda: _env("RISK_GATE_MODE", "shadow"))
    risk_gate_soft_threshold: float = field(
        default_factory=lambda: _env_float("RISK_GATE_SOFT_THRESHOLD", 0.25))
    # Wave-2 #5 — reflection tree (Generative-Agents style): every
    # reflect_every_cycles supervisor cycles, the world model's recent
    # notes are synthesized into 1-2 higher-level INSIGHT notes on the
    # LOCAL router (purpose reflect) — insights carry provenance (parent
    # note ids, validated) and feed ideation seeding. 0 = off.
    reflect_every_cycles: int = field(
        default_factory=lambda: _env_int("REFLECT_EVERY_CYCLES", 6))
    reflect_min_notes: int = field(
        default_factory=lambda: _env_int("REFLECT_MIN_NOTES", 5))
    # Wave-2 #1 — Voyager-style skill library: supported settles are
    # distilled into reusable skills (description via the LOCAL router,
    # purpose skill_distill; code stored verbatim). Retrieval injects the
    # token-overlap top-3 into protocol-writing/implement prompts.
    skill_library_enabled: bool = field(
        default_factory=lambda: _env_bool("SKILL_LIBRARY", True))
    skill_distill_every_cycles: int = field(
        default_factory=lambda: _env_int("SKILL_DISTILL_EVERY_CYCLES", 4))
    skill_max: int = field(default_factory=lambda: _env_int("SKILL_MAX", 120))
    # --- Coordination pack A2X (docs/COORDINATION.md) ------------------------
    # Contract-Net-lite: rank bonus a claim adds to a task the supervisor's
    # scan awarded to THIS agent (0 disables the allocation signal entirely;
    # the awards table is still maintained). ~0.75 flips a ranking the way a
    # priority label would, without overriding the agent's own profile.
    coord_award_bonus: float = field(
        default_factory=lambda: _env_float("COORD_AWARD_BONUS", 0.75))
    # Award TTL (owner dashboard 2026-11-14: task #4 was re-awarded to the
    # same coder EVERY scan forever because the awarded agent never claimed
    # it — the market signal became a stuck record). An (task, agent) pair
    # that survives this many consecutive scans without being claimed is
    # dropped from the allocation: the manager stops nagging, the task stays
    # open for the ordinary profile race. 0 disables the TTL (re-award
    # forever, pre-TTL behavior).
    coord_award_ttl: int = field(
        default_factory=lambda: _env_int("COORD_AWARD_TTL", 3))
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
    research_fanout: int = field(default_factory=lambda: _env_int("RESEARCH_FANOUT", 3))
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

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.chroma_path, self.workspaces_dir):
            d.mkdir(parents=True, exist_ok=True)


def load_yaml_overrides(settings: Settings) -> Settings:
    """Apply YAML config over env-derived settings (yaml wins where present).

    THE config file is ``configs/secrets.yaml`` (gitignored) — it holds the
    per-deployment values AND the credentials and wins over everything else.
    ``configs/lab.yaml`` is a deprecated alias, still read FIRST for backward
    compatibility with older checkouts; secrets.yaml overrides it key by key.
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
