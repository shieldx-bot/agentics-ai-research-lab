# LLM Providers — Perplexity (cookie) vs Qwen (cookie)

The lab talks to two free web frontends through browser credentials. Both are
unofficial APIs and can change or break at any time; use your own accounts.

> **Research without any LLM cookie:** since the free-tools commit, a dead
> cookie pool no longer kills the research node — `node_research` falls back
> to keyless official APIs (arXiv / OpenAlex / Crossref) that return real
> publisher metadata, and every finding's BibTeX is grounded in that metadata.
> See [FREE_TOOLS.md](FREE_TOOLS.md).

## Perplexity (`LLM_PROVIDER=perplexity`, default)

* Transport: cookie WebSocket (socket.io v4), `agentics/llm/perplexity_client.py`.
* Strengths: **built-in web search with citations** (`mode="academic"` biases
  to scholarly sources) — ideal for the *research* node and the citation
  pipeline (`agentics/research/`).
* Weaknesses: smaller effective model; no file/tool use; rate-limited per
  cookie session.

## Qwen (`LLM_PROVIDER=qwen`) — `agentics/llm/qwen_client.py`

* Transport: native web API of chat.qwen.ai, verified live on 2026-11-12:

| Fact | Value |
| --- | --- |
| Models (`GET /api/models`) | `qwen3.7-plus`, `qwen3.8-max`, `qwen4.0-max-preview` |
| Qwen4-Max-Preview | flagship Qwen4 preview — strongest reasoning/coding |
| Access token TTL | **~15 minutes** (localStorage `token`, JWT) |
| Refresh token TTL | **~30 days** (cookie `refresh_token`, JWT) |
| Refresh | `POST /api/v2/auths/refresh` `{"refresh_token": ...}` → `data.token` |
| Chat | `POST /api/v2/chat/completions` (stream, `incremental_output`, SSE frames `{data:{phase:"answer",content:...}}`) |

* The client auto-refreshes the access token (margin 120 s) and maps
  `mode="academic"/"research"` to Qwen's `auto_search` feature flag.
* **WAF caveat (verified live):** Alibaba's baxia WAF rejects raw HTTP calls
  that lack browser-grade anti-bot tokens (`RGV587_ERROR / FAIL_SYS_USER_VALIDATE`).
  Cookie exports from a *live logged-in* session usually carry `tfstk` and pass;
  stale exports do not. If blocked: re-export cookies from the logged-in
  browser, or drive chat.qwen.ai through a real browser (Playwright /
  agent-browser) — a browser session was confirmed working end-to-end.

### Setup

```bash
cp accounts/qwen-account.example.json accounts/qwen-account-01.json
# ... paste real cookies + refresh_token (see the template's _comment)
echo "LLM_PROVIDER=qwen" >> .env
python -m agentics.cli test-llm          # uses the first qwen-* account
python -m agentics.cli run
```

## Modes — `LLM_PROVIDER` decides (default: perplexity-only)

`agentics/llm/session_host.py` owns the cookie pools and routes per call.

### `LLM_PROVIDER=perplexity` (default) — Perplexity-only

* Every purpose (`research`, `plan`, `implement`, `review`, `distill`,
  `general`) goes to the Perplexity cookie pool (`mode="academic"` for
  research, `"concise"` for code tasks).
* Qwen is **never called**: no routing to it, no cross-provider fallback.
  Its pool is still loaded for CLI diagnostics and shown as `DISABLED`
  in `agentics status`.
* Number of agent threads = number of Perplexity accounts.

### `LLM_PROVIDER=qwen` — hybrid (both pools)

| Purpose (graph node) | Primary provider | Why |
| --- | --- | --- |
| `research` | **Perplexity** (`academic`) | citations + scholarly sources, feeds BibTeX — "giấy trắng mực đen" |
| `plan` | **Qwen4-Max-Preview** | strongest code reasoning; JSON plans |
| `implement` | **Qwen4-Max-Preview** | strongest code reasoning; full-file outputs |
| `review` | **Qwen4-Max-Preview** | verdict quality needs judgement, not search |
| `distill` / `general` | Qwen4-Max-Preview | LifeMem skill distillation, fallbacks |

* Cross-provider fallback: primary pool empty / all cooling down -> the other
  provider serves the call (logged as a warning).
* One agent thread = one pinned identity: `host.view(prefer_index=i)` pins
  agent `i` to account `i` of each pool; a cooled-down account is skipped and
  a healthy sibling is leased instead.
* Failure handling: `ask()` marks the serving account failed on any exception;
  the orchestrator's auth-error path calls `mark_failed()` on the agent's view.
  The host re-leases automatically at the next call.

### Live verification notes (2026-11-12, datacenter sandbox)

* In-page `fetch()` from a real Chrome session carrying injected cookies gets
  **HTTP 200 past the baxia WAF** (raw aiohttp from the same IP gets the WAF
  HTML challenge page) — but the SSE body then hangs while an
  `Access Verification` noCaptcha **jigsaw slider** is shown in the page.
* The slider reappears on every navigation from a datacenter IP and rejects
  synthetic drag attempts (5 calibrated attempts, pixel-measured gaps, varied
  human-like trajectories — all rotated to a new image). Home-residential
  sessions typically do not see it.
* The exported `refresh_token` was already rotated server-side
  (`{"code":"not found"}` on refresh) — exports go stale within hours if the
  browser session rotates tokens.

**Operational conclusion (why the default is perplexity-only):** Qwen needs a
residential machine with cookies exported *right after* solving the slider in
a live browser (must include `tfstk`); Perplexity works unattended. Switch to
`LLM_PROVIDER=qwen` only when you can maintain a fresh Qwen session.

### Setup

```bash
# Perplexity cookies (the only pool used in the default mode)
cp accounts/account-01.json.example accounts/account-01.json   # repeat per cookie

python -m agentics.cli test-llm                    # smoke-test perplexity
python -m agentics.cli test-llm --provider qwen    # diagnostics only
python -m agentics.cli status                      # shows pools + mode
python -m agentics.cli run
```

## Hyperbrowser cloud-browser validation (`scripts/hyperbrowser_smoke.py`)

Raw WebSocket access to Perplexity is blocked from datacenter IPs
(`wss://www.perplexity.ai/socket.io` → HTTP 403 on a bare VPS). To validate the
cookie pool (or harvest/refresh cookies) from such an environment, drive a real
browser session on Hyperbrowser instead — verified live on 2026-11-12: **all 10
accounts logged in, submitted a real query via the web UI and captured real
answers with citations (~60 s round-trip each)**.

```bash
export HB_API_KEY=...            # Hyperbrowser API key
python scripts/hyperbrowser_smoke.py --accounts 1-3 --wait 60
```

* Creates one stealth session per account (US residential proxy), connects
  Playwright over CDP, injects the account cookies, detects login via the
  `session-token` cookie (httpOnly — read through CDP, not `document.cookie`),
  submits a query in the Lexical editor (`keyboard.insert_text`, not `type()`),
  extracts the answer text and screenshots.
* Requires `pip install hyperbrowser playwright` and `playwright install` for
  the local driver. Free Hyperbrowser plans run 1 session at a time.

### Running the lab itself through Hyperbrowser (`BROWSER_BACKEND=hyperbrowser`)

Set in the environment (or `configs/lab.yaml`):

Via `configs/lab.yaml` (committed, no secrets):

```yaml
browser_backend: hyperbrowser
hyperbrowser_proxy_country: US
hyperbrowser_timeout: 900
```

and the API key in `configs/secrets.yaml` (gitignored; copy
`configs/secrets.example.yaml`):

```yaml
hyperbrowser_api_key: "hb_..."
```

Equivalent env vars: `BROWSER_BACKEND`, `HB_API_KEY`, `HB_PROXY_COUNTRY`,
`HB_TIMEOUT`. YAML wins over env; `secrets.yaml` wins over everything.

With this backend the Perplexity pool's clients become
`agentics.llm.hyperbrowser_client.HyperbrowserClient` — the graph nodes are
unchanged; only the transport under `BrowserSessionHost` swaps. Verified live
2026-11-12 (full `research -> plan` cycle, real topic "ML for early failure
prediction of NFV network functions"):

| Node | Time | Result |
| --- | --- | --- |
| `node_research` | ~67 s | 6.9k-char academic survey, 7 verifiable citations (IMC'18, IEEE NetSoft'20, CNSM, ITU Journal…), concrete metrics (RF 93% precision / 99% AUC, XGBoost, SVR…) |
| `node_plan` | ~48 s | valid JSON plan: 14 files + 7 concrete steps grounded in the surveyed papers |

Scaling in cloud-browser mode (P18 token pool):

* every Hyperbrowser token owns ONE cloud session; asks on the same token
  are serialized (free-plan semantics), asks on DIFFERENT tokens run fully
  parallel.
* `HB_API_KEYS=tk1,tk2,tk3` (or a `hyperbrowser_api_keys` list in
  `configs/secrets.yaml`) gives the swarm 3 concurrent sessions — the
  orchestrator sizes the process to 3 parallel agents automatically
  (still capped by `MAX_AGENTS` and the cookie-account pool). Agent *i* is
  pinned to token *i* (round-robin if agents > tokens).
* a single `HB_API_KEY` keeps the old free-plan behavior: one session, one
  agent per process.
* each `ask()` navigates a fresh Perplexity thread (~60–140 s round trip);
  with K tokens the lab sustains K concurrent asks.
* offline unit tests never create sessions; live check:
  `HB_API_KEYS=tk1,tk2,tk3 python scripts/live_research_run.py`

Research fan-out (P18b, `research_fanout` in `configs/lab.yaml` or
`RESEARCH_FANOUT`):

* when > 1, `node_research` asks complementary angles — benchmarks &
  datasets, limitations & open problems — IN PARALLEL, each on a different
  pool identity (a different Hyperbrowser token with the P18 pool, or a
  different cookie account on the local backend). The angle views wrap
  round-robin from the agent's own token and are capped at pool size − 1,
  so an angle never queues behind an ask already running on the same
  identity.
* all answers merge into ONE finding (survey first, angles under their own
  headings, citations deduped) feeding one batched BibTeX grounding call;
  the P20 budget gate and usage ledger apply to every angle; the fan-out
  is journalled per agent (`research_fanout` event, meta `research_fanout`).
* `1` disables the fan-out (legacy single-ask behavior — the safe default;
  `configs/lab.yaml` ships with `research_fanout: 2` = survey + 1 angle).

### Failure handling

* A missing composer is not automatically treated as a bad cookie. The client
  records bounded page metadata (URL/title/body excerpt), distinguishes an
  expired login from a provider verification/challenge page, and tears down
  the failed session.
* One in-place recovery precedes the escalation: a missing composer triggers
  a home reload + one more wait before any circuit opens, so a slow SPA
  hydration no longer opens the shared circuit breaker that used to stall
  every agent for minutes (the 25 s-timeout incident class, 2026-11-12).
* A submit that silently produces NO answer text after the full answer wait
  window raises a UI outage (same no-cooldown treatment) instead of returning
  an empty answer — downstream nodes used to crash parsing `""` as JSON.
* A verification page opens a process-wide circuit breaker (10 minutes); a
  generic composer/frontend failure opens it for 3 minutes. While open, agents
  fail fast instead of creating sessions for every account.
* Provider-wide UI/challenge failures do **not** put individual cookie
  accounts on cooldown. Only account-specific authentication failures do.
  The client does not automate or attempt to bypass CAPTCHA/security
  verification; resolve any required verification through the normal
  provider-supported user flow, then restart/retry.
* Agent issue discovery is cached process-wide for 45 seconds. This coalesces
  synchronized scans; the actual claim operation still re-reads the issue and
  checks its current labels before acquiring the claim.
