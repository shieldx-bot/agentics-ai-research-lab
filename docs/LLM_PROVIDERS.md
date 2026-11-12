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

## Recommendation (hybrid)

| Node | Best provider | Why |
| --- | --- | --- |
| `research` | **Perplexity** (`academic`) | citations + scholarly sources, feeds BibTeX |
| `plan` / `implement` / `review` | **Qwen4-Max-Preview** | strongest code reasoning; long outputs |
| self-evolution analysis | Qwen | needs quality judgement, not search |

Running both pools side-by-side per role is future work (track it as an issue:
"role-based provider routing"); for now `LLM_PROVIDER` switches the whole lab.
