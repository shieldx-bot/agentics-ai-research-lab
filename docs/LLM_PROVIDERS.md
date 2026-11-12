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
