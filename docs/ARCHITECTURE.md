# ARCHITECTURE — Agentics AI Research Lab

## 1. Thành phần

```
agentics/
├── config.py            # Cấu hình: env + configs/lab.yaml
├── stop.py              # Stop controller (file STOP + SIGINT)
├── orchestrator.py      # asyncio: 1 task = 1 account cookie = 1 agent thread
├── cli.py               # Typer CLI: run / stop / status / test-llm
├── llm/
│   ├── perplexity_client.py   # WebSocket socket.io client cho Perplexity (cookie auth)
│   └── pool.py                # AccountPool: gán account↔agent, rotation, health check
├── memory/
│   ├── shared_memory.py       # SQLite WAL: messages, findings, decisions, claims, heartbeats
│   ├── personal_memory.py     # SQLite riêng mỗi agent: lessons, state
│   └── vector_store.py        # ChromaDB: semantic search findings/papers
├── github_ops/
│   ├── client.py              # PyGithub wrapper
│   ├── issue_ops.py           # scan, claim, release, close issue
│   ├── branch_ops.py          # tạo/đặt tên nhánh chuẩn
│   ├── pr_ops.py              # mở PR, auto-merge, update branch
│   └── repo_workspace.py      # git clone/pull/push cục bộ cho từng agent
├── policies/
│   ├── claims.py              # Claim registry + TTL
│   ├── ownership.py           # Ownership matrix, conflict check trên diff
│   └── merge_queue.py         # Hàng đợi merge tuần tự
├── research/
│   ├── papers.py              # Search paper qua Perplexity academic mode
│   └── citations.py           # Trích dẫn chuẩn: BibTeX, APA
├── agents/
│   ├── roles.py               # Role enum + prompt templates
│   └── base.py                # AgentContext: tài nguyên được gán cho 1 agent
└── graph/
    ├── state.py               # ResearchState TypedDict
    ├── nodes.py               # Node functions của LangGraph
    └── research_graph.py      # StateGraph: chu kỳ nghiên cứu
```

## 2. Luồng chạy

1. `orchestrator.py` đọc toàn bộ `accounts/*.json`, dựng `AccountPool`.
2. Với mỗi account, spawn 1 asyncio task. Mỗi task:
   - có `AgentContext` (agent_id, role, llm client riêng, workspace git riêng,
     personal memory riêng);
   - chạy `ResearchGraph` theo chu kỳ vô tận, giữa các chu kỳ kiểm tra STOP.
3. **Supervisor** (thường account 01) quét issue còn `open` + không có label
   `claimed/*`; nếu ít việc, tạo issue nghiên cứu mới từ hướng đi trong memory.
4. Agent bất kỳ ở chu kỳ kế chọn issue: xin claim qua `policies.claims`
   (ghi GitHub label + ghi shared memory — 2 lớp khoá).
5. **Researcher/Coder/Writer** xử lý theo role; finding/paper đưa vào shared memory
   + Chroma để agent khác tái sử dụng.
6. PR mở với prefix `[agent-id]`; **Reviewer khác** review; approved → **Integrator**
   merge tuần tự. Không ai merge PR của chính mình.
7. Sau merge: node `reflect` ghi lesson vào personal memory + broadcast tin nhắn
   vào shared memory cho các agent khác ở chu kỳ sau.

## 3. LLM — Perplexity qua cookie

- `PerplexityClient` kết nối `wss://www.perplexity.ai/socket.io/?EIO=4&transport=websocket`
  bằng header Cookie của account, emit event `perplexity_ask`, gom streaming messages
  cho tới khi có câu trả lời cuối kèm `citations`.
- `mode="academic"` ưu tiên nguồn học thuật → dùng cho search paper; citations được
  `research/citations.py` chuyển thành BibTeX/APA chuẩn.
- Endpoint nội bộ của Perplexity **có thể đổi bất cứ lúc nào**; client được thiết kế
  để sửa 1 chỗ (`ASK_EVENT` / URL) khi cần. `test-llm` CLI để kiểm tra nhanh.
- Cookie chết/429 → pool đánh dấu `cooldown`, agent tạm nghỉ và báo cáo vào shared
  memory; Supervisor có thể tái phân vai.

## 4. Memory model

| Store | Vai trò | Ai ghi | Ai đọc |
|---|---|---|---|
| Shared memory (SQLite WAL) | messages, findings, decisions, claims, heartbeats | mọi agent | mọi agent |
| Vector store (ChromaDB) | embedding findings/papers để semantic search | researcher/coder/writer | mọi agent |
| Personal memory (SQLite/agent) | lessons học được, trạng thái riêng | agent đó | agent đó |

Cải tiến memory: node `reflect` cuối mỗi chu kỳ nén kết quả thành 1–3 câu "lesson",
ghi vào cả personal lẫn shared memory → kiến thức tích luỹ qua các chu kỳ vô tận.

## 5. Vòng lặp vô tận & dừng

- Mỗi chu kỳ: graph chạy trọn vòng; pause `CYCLE_PAUSE_SECONDS` + jitter ngẫu nhiên.
- Dừng: file `STOP` ở thư mục gốc (tạo bằng `agentics stop` hoặc `touch STOP`) →
  mọi agent kết thúc node hiện tại rồi dừng; trạng thái claim được giải phóng.
- `SIGINT` (Ctrl+C) được chuyển thành graceful shutdown tương tự.
