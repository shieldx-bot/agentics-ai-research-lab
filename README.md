# Agentics AI Research Lab

> **Multi-agent AI research framework where AI agents talk to each other through shared memory,
> and work directly on a GitHub repository — scanning issues, claiming them, opening branches,
> cross-reviewing, merging and pushing to `main` — exactly like a team of human researchers.**

[![python](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org)
[![langgraph](https://img.shields.io/badge/core-LangGraph-1C3C3C)](https://github.com/langchain-ai/langgraph)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Tiếng Việt ở dưới · [English quickstart](#english-quickstart)

---

## 1. Triết lý thiết kế

Agentics AI Research Lab được xây dựng theo nguyên lý **"đàn kiến nghiên cứu" (research swarm)**:

- **Mỗi luồng agent = một tài khoản cookie riêng.** Hệ thống chạy đa luồng
  (`asyncio`), mỗi luồng nạp một file cookie trong `accounts/` — tức là một
  "nhân cách" LLM độc lập với quota riêng, không giành nhau tài nguyên.
- **Agent tự chọn vai (role) trong mã nguồn** — Supervisor, Researcher, Coder,
  Writer, Reviewer, Integrator — qua cấu hình hoặc tự phân vai khi khởi động.
- **Agents làm việc trực tiếp trên repo GitHub như một chuyên gia GitHub thật:**
  tự quét issue → nhận issue (claim) → tạo nhánh → code / chạy lệnh / viết tài liệu →
  review chéo nhau → mở PR → merge → push `main`.
- **Trao đổi qua bộ nhớ chung + cá nhân.** Mọi finding, quyết định, bài báo trích dẫn
  đều được ghi vào Shared Memory (SQLite WAL) và Vector Store (ChromaDB) để mọi agent
  tra cứu; mỗi agent có thêm Personal Memory ghi lại bài học riêng.
- **Vòng lặp nghiên cứu vô tận.** Agents chạy chu kỳ liên tục cho đến khi owner
  tạo file `STOP` (hoặc bấm `Ctrl+C`) — lúc đó mọi luồng dừng an toàn giữa chu kỳ.
- **Policy chống xung đột là công dân hạng nhất**, không phải remediation:
  claim-based issue locking, bảng sở hữu file (ownership matrix), hàng đợi merge
  tuần tự, rebase-before-push, cross-review bắt buộc.

**Miền nghiên cứu mặc định:** AI/ML cho NFV (Network Function Virtualization) —
kế thừa định hướng của repo gốc `ai-nfv-research-lab` — nhưng bạn có thể đổi chủ đề
nghiên cứu chỉ bằng một file cấu hình.

---

## 2. Kiến trúc tổng quan

```
┌─────────────────────────────── Orchestrator (asyncio) ───────────────────────────────┐
│                                                                                       │
│  ┌─ Thread 1 (account-01) ─┐  ┌─ Thread 2 (account-02) ─┐  ...  ┌─ Thread N ──────┐  │
│  │  Role: Supervisor       │  │  Role: Researcher       │       │  Role: Coder    │  │
│  │  ┌───────────────────┐  │  │  ┌───────────────────┐  │       │                 │  │
│  │  │ LangGraph         │  │  │  │ LangGraph         │  │       │  LangGraph      │  │
│  │  │ ResearchCycle     │  │  │  │ ResearchCycle     │  │       │  ResearchCycle  │  │
│  │  └───────────────────┘  │  │  └───────────────────┘  │       │                 │  │
│  └────────────┬────────────┘  └────────────┬────────────┘       └────────┬────────┘  │
│               │                            │                             │           │
│        ┌──────┴────────────────────────────┴─────────────────────────────┴──────┐    │
│        │                    Shared Memory (SQLite WAL) + ChromaDB               │    │
│        │        messages · findings · decisions · claims · heartbeats           │    │
│        └───────────────────────────────────────────────────────────────────────┘    │
│                                                                                       │
│   LLM Pool: Perplexity (cookie xoay vòng, mode "academic" cho search paper)           │
│   hoặc Qwen chat.qwen.ai (LLM_PROVIDER=qwen, ví dụ qwen4.0-max-preview)              │
│   GitHub Ops: PyGithub — issues · branches · PRs · auto-merge                         │
│   Stop Controller: file STOP + SIGINT → graceful shutdown giữa chu kỳ                 │
└───────────────────────────────────────────────────────────────────────────────────────┘
```

Chi tiết đầy đủ: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) ·
Chính sách chống xung đột: [docs/POLICIES.md](docs/POLICIES.md)

---

## 3. Cài đặt nhanh

```bash
git clone https://github.com/shieldx-bot/agentics-ai-research-lab.git
cd agentics-ai-research-lab

python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) Token GitHub (chỉ cần quyền repo) — KHÔNG commit token vào repo
cp .env.example .env
#    sửa .env: GITHUB_TOKEN=ghp_xxx  GITHUB_REPO=shieldx-bot/agentics-ai-research-lab

# 2) Cookie accounts — mỗi file là 1 agent thread
#    export cookie từ trình duyệt (extension "Cookie-Editor" → Export JSON)
#    rồi đặt vào accounts/account-01.json, account-02.json, ...
python scripts/setup_accounts.py            # chuẩn hoá + kiểm tra độ tươi cookie

# 3) Kiểm tra 1 account có "sống" không
python -m agentics.cli test-llm --account 1

# 4) KHỞI CHẠY ĐÀN AGENT (vòng lặp vô tận)
python -m agentics.cli run

# 5) DỪNG an toàn (owner quyết định)
python -m agentics.cli stop          # hoặc: touch STOP
```

Xem trạng thái trực tiếp: `python -m agentics.cli status`  
Đo tiến hóa của lab (tỉ lệ merge theo loại việc): `python -m agentics.cli metrics` — xem [docs/EVOLUTION.md](docs/EVOLUTION.md)  
Sức khỏe bộ nhớ LifeMem (trace, cụm workflow, skill đã chưng cất): `python -m agentics.cli memory-stats` — xem [docs/MEMORY.md](docs/MEMORY.md)

---

## 4. Vai trò (roles) & luồng công việc

| Role | Nhiệm vụ chính |
|---|---|
| **Supervisor** | Quét issue, đánh giá, tạo issue nghiên cứu mới, phân bổ priority |
| **Researcher** | Search paper (Perplexity academic mode), trích dẫn chuẩn BibTeX/APA, ghi finding vào shared memory |
| **Coder** | Tạo nhánh, viết code, chạy lệnh/thử nghiệm trong workspace cục bộ |
| **Writer** | Viết/sửa tài liệu, tóm tắt kết quả nghiên cứu thành report |
| **Reviewer** | Cross-review PR của agent khác (bắt buộc 1 review thông qua trước khi merge) |
| **Integrator** | Duyệt hàng đợi merge, rebase, merge vào `main` — **tuần tự, một mình** |

Luồng một chu kỳ nghiên cứu (LangGraph `ResearchCycle`):

```
scan → claim → research → plan → implement → self_review
     → cross_review → open_pr → integrate → reflect (memory) → lặp lại
```

---

## 5. Chống xung đột giữa các agents (tóm tắt)

1. **Issue claim locking** — issue chỉ có 1 chủ: label `claimed/<agent>` + TTL,
   claim hết hạn tự được nhặt lại.
2. **Nhánh riêng cho từng agent** — `agent/<agent-id>/issue-<n>-<slug>`, không agent
   nào push vào nhánh của agent khác.
3. **Ownership matrix** — mỗi role "sở hữu" một vùng thư mục; diff đụng vùng người
   khác → agent tự buông (yield) và đổi việc.
4. **Merge queue tuần tự** — chỉ Integrator được merge, từng PR một, rebase trước merge.
5. **Heartbeat & stale detection** — agent chết > 2 chu kỳ → claim của nó được giải phóng.
6. **Cross-review bắt buộc** — không merge PR do chính mình viết.

Chi tiết: [docs/POLICIES.md](docs/POLICIES.md)

---

## 6. Bảo mật — ĐỌC KỸ

- **KHÔNG BAO GIỜ** commit `GITHUB_TOKEN`, file cookie, hay nội dung `.env` lên repo.
  Toàn bộ nằm trong `.gitignore` từ commit đầu tiên.
- Cookie Perplexity = phiên đăng nhập thật. Chỉ dùng cookie của tài khoản **của bạn**,
  và lưu ý việc dùng cookie để gọi API có thể vi phạm Điều khoản dịch vụ của
  Perplexity — bạn tự chịu trách nhiệm khi dùng. Mã nguồn này chỉ dành cho mục đích
  nghiên cứu cá nhân / học thuật.
- Nếu token đã từng dán vào chỗ công khai (chat, issue…), **hãy thu hồi và tạo token
  mới ngay** tại https://github.com/settings/tokens.

## 7. Đóng góp

Dự án được thiết kế để **nhiều người phát triển cùng lúc mà ít xung đột**:
xem [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) — mỗi contributor/agent làm việc trên
nhánh riêng, module ownership rõ ràng, PR phải qua cross-review.

## English quickstart

```bash
pip install -r requirements.txt
cp .env.example .env          # fill GITHUB_TOKEN + GITHUB_REPO
python scripts/setup_accounts.py
python -m agentics.cli run    # infinite research loop
python -m agentics.cli stop   # owner presses the brakes
```

## License

MIT — see [LICENSE](LICENSE).
