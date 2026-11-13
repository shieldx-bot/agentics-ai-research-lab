# Agentics AI Research Lab

> **Multi-agent AI research framework where AI agents talk to each other through shared memory,
> and work directly on a GitHub repository — scanning issues, claiming them, opening branches,
> cross-reviewing, merging and pushing to `main` — exactly like a team of human researchers.**

[![python](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org)
[![langgraph](https://img.shields.io/badge/core-LangGraph-1C3C3C)](https://github.com/langchain-ai/langgraph)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Tiếng Việt ở dưới · [English quickstart](#english-quickstart)

---


## Cloud browser backend (datacenter deployments)

Perplexity 403-blocks raw cookie WebSocket access from datacenter IPs. Set `browser_backend: hyperbrowser` in `configs/lab.yaml` and put your key in `configs/secrets.yaml` (`hyperbrowser_api_key: "hb_..."`) — every node then runs through a stealth cloud browser automatically (see docs/LLM_PROVIDERS.md).

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
│   LLM: BrowserSessionHost — MẶC ĐỊNH chỉ dùng Perplexity (LLM_PROVIDER=perplexity)     │
│   · mọi node (research/plan/implement/review) đều qua pool cookie Perplexity           │
│   · research dùng mode "academic" -> citation giấy trắng mực đen cho BibTeX            │
│   · Qwen chat.qwen.ai bị tắt hoàn toàn; bật lại hybrid: LLM_PROVIDER=qwen              │
│   GitHub Ops: PyGithub — issues · branches · PRs · auto-merge                         │
│   Stop Controller: file STOP + SIGINT → graceful shutdown giữa chu kỳ                 │
└───────────────────────────────────────────────────────────────────────────────────────┘
```

Chi tiết đầy đủ: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) ·
Chính sách chống xung đột: [docs/POLICIES.md](docs/POLICIES.md)

---

## 3. Cài đặt nhanh

> **Chế độ LOCAL-ONLY (nhánh `update_interactive_local`):** lab chạy hoàn toàn
> trên máy của bạn — không cần `GITHUB_TOKEN`, không gọi GitHub. "origin" của
> agent là bare repo tự seed tại `data/lab.git`; tasks/MR/review/merge đều
> nằm trong SQLite. Owner theo dõi qua web dashboard (`agentics dashboard`),
> terminal live view (`agentics watch`) hoặc stream sự kiện (`agentics events -f`).
> Chi tiết kiến trúc + nhật ký tích
> hợp: [docs/LOCAL_MODE.md](docs/LOCAL_MODE.md).

```bash
git clone https://github.com/shieldx-bot/agentics-ai-research-lab.git
cd agentics-ai-research-lab

python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) Không cần token gì cả — chỉ còn cookie LLM
cp .env.example .env             # tuỳ chỉnh RESEARCH_TOPIC / MAX_AGENTS
#    LƯU Ý: agentics/config.py TỰ NẠP .env khi khởi động (không cần `source .env`
#    hay export tay). Biến môi trường thật trong shell vẫn luôn thắng file .env.

# 2) Cookie accounts — mỗi file là 1 agent thread
#    export cookie từ trình duyệt (extension "Cookie-Editor" → Export JSON)
#    rồi đặt vào accounts/account-01.json, account-02.json, ...
python scripts/setup_accounts.py            # chuẩn hoá + kiểm tra độ tươi cookie

# 3) Kiểm tra 1 account có "sống" không
python -m agentics.cli test-llm --account 1

# 4) KHỞI CHẠY ĐÀN AGENT (vòng lặp vô tận, local 100%)
python -m agentics.cli run

# 5) MẮT OWNER — xem agents đang làm gì, mục đích gì, realtime
python -m agentics.cli dashboard     # → http://127.0.0.1:8080
python -m agentics.cli watch        # terminal live view (SSH-friendly)

# 6) Thêm việc cho đàn agent / sửa backlog
python -m agentics.cli task "Nghiên cứu thuật toán X" --category research
python -m agentics.cli task --list

# 6b) CỔNG MERGE (khi AUTO_MERGE=false — mặc định): owner giữ chốt
python -m agentics.cli mr list          # pipeline MR: ai làm, ai duyệt
python -m agentics.cli mr approve 3     # owner duyệt thay/ thêm cross-review
python -m agentics.cli mr merge 3       # owner merge (P10 + tier-2 vẫn chạy)

# 7) DỪNG an toàn (owner quyết định)
python -m agentics.cli stop          # hoặc: touch STOP
```

Xem trạng thái trực tiếp: `python -m agentics.cli status`  
Đo tiến hóa của lab (tỉ lệ merge theo loại việc): `python -m agentics.cli metrics` — xem [docs/EVOLUTION.md](docs/EVOLUTION.md)  
Sức khỏe bộ nhớ LifeMem (trace, cụm workflow, skill đã chưng cất): `python -m agentics.cli memory-stats` — xem [docs/MEMORY.md](docs/MEMORY.md)

**P10 — code lab là của owner:** mọi diff đụng `agentics/`, `configs/`,
`accounts/`, `scripts/`, `.github/` bị chặn ở 3 tầng (ghi file, diff trước
push, cửa merge). Exception DUY NHẤT là owner chạy
`python -m agentics.cli allow-protected <task_id>` (coi `accounts/` và
`configs/secrets.yaml` không bao giờ được miễn).

**Các lớp bổ sung trên nhánh này (P13/P14 + integrity):** danh sách file dự
kiến ở bước plan được **khóa toàn cục** (P13 — agent khác đụng trúng bị dừng
trước khi đốt chu kỳ); diff ở bước implement phải nằm trong **phạm vi intent
đã tuyên bố** (P14 — lệch phạm vi bị yield + báo dashboard); và một
**IntegrityWatcher** quét định kỳ (~20s) tự hoàn tác mọi thay đổi vùng cấm
bị script lạc ghi đè trong workspace — `INTEGRITY_INTERVAL_S=0` để tắt.
Chi tiết: `docs/POLICIES.md` (P10 bổ sung, P13, P14).

**P16/P16bis — agent tự kiểm chứng ý tưởng bằng code chạy thật:** một ý tưởng
không được đánh giá bằng lời kết luận của LLM mà bằng **bằng chứng thực
nghiệm** — plan khai báo `verify_cmds`/`hypotheses` (chốt tiêu chí định lượng
TRƯỚC khi chạy, như pre-registration), protocol được viết và chạy 2 lần trong
workspace có guard, máy so sánh số đo với tiêu chí để phán
`confirmed/refuted/inconclusive` (protocol không tái lập được → inconclusive,
không bao giờ đếm là thành công), và mọi verdict được broadcast cho cả đàn +
inject vào prompt của từng agent (`verified_knowledge` — claim bị phản chứng
sẽ không ai lặp lại). Owner xem bằng `agentics evidence` /
`agentics hypotheses` và 2 panel trên dashboard. Chi tiết:
[docs/VERIFICATION.md](docs/VERIFICATION.md). Reviewer **tự chạy lại check của tác giả trên nhánh thật + full test suite (tier-2) trước khi hỏi LLM** — MR không qua được test bị từ chối với 0 token LLM.

**Free research tools — nguồn bằng chứng thật, không cần API key:** node
research không còn phụ thuộc mạng sống/chết của cookie LLM. Khi LLM fail,
finding được dựng từ metadata thật của **arXiv / OpenAlex / Crossref /
Semantic Scholar / Europe PMC** (tác giả, năm, abstract, số trích dẫn —
fetch từ API chính thức, không phải LLM đẻ ra) và BibTeX được làm giàu bằng
metadata nhà xuất bản thay vì scrape thẻ title. `search_all` chạy 3 nguồn
song song trên 3 budget rate-limit độc lập rồi lấp chỗ trống theo tầng —
OpenAlex bị 429 trên IP datacenter không còn làm rỗng finding. Kèm
**verification toolkit**: HuggingFace datasets + models (dữ liệu và weights
thật cho protocol), GitHub repo search (ý tưởng đã có code chưa? repo tham
chiếu để clone vào sandbox) và PyPI package lookup (version thật để pin vào
`verify_cmds` — tái lập P16) — tất cả keyless, degrade graceful, tôn trọng
rate limit. Owner dùng trực tiếp:

```bash
python -m agentics.cli papers "ML for NFV"                       # bảng bài báo + BibTeX thật
python -m agentics.cli papers --doi 10.1109/netsoft48620.2020.9165348
python -m agentics.cli papers "intrusion detection" --datasets
python -m agentics.cli papers "nfv scheduler" --github            # repo tham chiếu
python -m agentics.cli papers --pypi scikit-learn                 # version thật để pin
```

Chi tiết: [docs/FREE_TOOLS.md](docs/FREE_TOOLS.md).


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
cp .env.example .env          # optional tuning; no GITHUB_TOKEN needed (local-only)
python scripts/setup_accounts.py
python -m agentics.cli run        # infinite research loop — 100% local
python -m agentics.cli dashboard  # owner's live view (http://127.0.0.1:8080)
python -m agentics.cli stop       # owner presses the brakes
```

## License

MIT — see [LICENSE](LICENSE).
