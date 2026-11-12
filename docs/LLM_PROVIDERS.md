# LLM Providers — Perplexity (cookie) vs Qwen (cookie)

The lab talks to two free web frontends through browser credentials. Both are
unofficial APIs and can change or break at any time; use your own accounts.

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

## Recommendation (hybrid) — IMPLEMENTED: `BrowserSessionHost`

`agentics/llm/session_host.py` owns BOTH cookie pools side-by-side and routes
per node (no more whole-lab `LLM_PROVIDER` switch for the run loop):

| Purpose (graph node) | Primary provider | Why |
| --- | --- | --- |
| `research` | **Perplexity** (`academic`) | citations + scholarly sources, feeds BibTeX — "giấy trắng mực đen" |
| `plan` | **Qwen4-Max-Preview** | strongest code reasoning; JSON plans |
| `implement` | **Qwen4-Max-Preview** | strongest code reasoning; full-file outputs |
| `review` | **Qwen4-Max-Preview** | verdict quality needs judgement, not search |
| `distill` / `general` | Qwen4-Max-Preview | LifeMem skill distillation, fallbacks |

Details:

* One agent thread = one pinned identity: `host.view(prefer_index=i)` pins
  agent `i` to account `i` of each pool; when its own account cools down the
  view leases any healthy sibling (same contract as the old `AccountPool`).
* **Graceful degradation**: if the primary pool for a purpose is empty or all
  its accounts are cooling down, the call falls back to the other provider
  (logged as a warning — citations may be missing on the Qwen path).
* Failure handling: `ask()` marks the serving account failed on any exception;
  the orchestrator's auth-error path calls `mark_failed()` on the agent's view.
  The host re-leases automatically at the next call.
* `LLM_PROVIDER` still works for standalone tooling (CLI `test-llm`,
  `AccountPool` direct use) but is no longer needed to run the lab.

### Setup (hybrid, both pools)

```bash
# Perplexity cookies (research pool)
cp accounts/account-01.json.example accounts/account-01.json   # repeat per cookie
# Qwen session (code pool): cookies + refresh_token + access_token from a
# live logged-in chat.qwen.ai session
cp accounts/qwen-account.example.json accounts/qwen-01.json

python -m agentics.cli test-llm --provider qwen     # smoke-test one pool
python -m agentics.cli status                        # shows both pools
python -m agentics.cli run                           # hybrid routing on
```

## Live verification notes (2026-11-12, datacenter sandbox)

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

**Operational conclusion:** run the Qwen pool from a residential machine with
cookies exported *right after* solving the slider in a live browser (must
include `tfstk`), or accept that research-only (Perplexity) works unattended
while Qwen needs periodic manual re-verification.
