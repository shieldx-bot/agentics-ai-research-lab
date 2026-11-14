# Agentics AI Research Lab

> **Multi-agent AI research lab where agents talk to each other through shared memory,
> research like a swarm, and settle every claim by RUNNING CODE — ranked by an
> Elo tournament (Co-Scientist), self-repaired on failed experiments, and
> evaluated as a swarm before production.**

[![python](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org)
[![langgraph](https://img.shields.io/badge/core-LangGraph-1C3C3C)](https://github.com/langchain-ai/langgraph)
[![tests](https://img.shields.io/badge/tests-1179%20passing-brightgreen)](#8-đánh-giá-bầy-đàn--pre-production)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Tiếng Việt ở dưới · [English quickstart](#english-quickstart)

---

## Có gì mới (2026-11)

- **Lớp bác bỏ P25 — verified nghĩa là SỐNG SÓT qua toàn bộ phân phối**
  (2026-11-14): một base run đạt criterion chỉ là "verified" yếu. Giờ mỗi
  giả thuyết có thể khai báo trường `falsification` — lớp input X (cases)
  + các quan hệ metamorphic (shuffle/double/negate/subset_half/duplicate/
  noise) mà phép đo phải chịu đứng. Runner sinh ca, chạy qua đường thực thi
  hiện có, chấm verdict kiểu Popper: MỘT counterexample → `refuted`; một
  quan hệ bị vi phạm → `refuted`; ca hỏng → `inconclusive`, không bao giờ
  `refuted` do đoán. Kèm **mutation quality gate** ("test yếu thì bằng chứng
  vô giá trị"): lab tự phá protocol bằng 4 mutation operator, mutant sống
  sót quá nhiều → `confirmed` bị hạ xuống `inconclusive` (kèm điểm mutation
  score trong ledger). Chi tiết: `docs/VERIFICATION.md` §14. Bổ trợ GEPA +
  GraphRAG: verdict `refuted` của P25 chính là tín hiệu làm `kg_edges.trusted`
  mất tín nhiệm và là bài học thất bại máy-đọc-được cho reflective evolution.
- **Wave 3 — tầng generator/cascade/tooling** (2026-11-14): đóng nốt roadmap
  TOP 5 — **GEPA reflection trong chu kỳ** (`agentics/evolution/gepa_driver.py`:
  mission hỏng lặp lại bị "chẩn đoán từ trace thật" rồi viết lại NGAY trong
  chu kỳ, generator-only: lab vẫn là judge — bổ trợ cho `gepa.py` chạy batch
  offline qua CLI `agentics gepa-evolve`), **cascade tĩnh** (`llm/router.py`:
  ask pool chết → 1 lần thử lại trên provider frontier độc lập; purpose phán
  quyết KHÔNG BAO GIỜ cascade), **MCP stdio bridge** (`tools/mcp_client.py`:
  server owner-approved vào bảng `mcp_tools` riêng — `tool_registry` vẫn chỉ
  nghĩa "code đã merge qua PR"), **risk gate học được ở chế độ shadow**
  (`verify/gate.py`: logistic 6 feature dự đoán P(settle sạch) trước khi panel
  mở — chỉ ghi nhật ký, không chặn; AUC in-sample ghi rõ là in-sample). Chi
  tiết + KPI: [ADVANCED_STACK.md §6](docs/ADVANCED_STACK.md).
- **GEPA — tối ưu prompt phản chiếu** (`agentics/evolution/gepa.py`): trên nền
  pool challenger wave-2, sinh ứng viên mission từ pareto-front (win-rate ×
  độ dài) + crossover + bài học thất bại thật; LOCAL router khi có, generator
  xác định khi không. Lab vẫn là judge (P11, exit-code). CLI:
  `agentics gepa-evolve <role> <category>`.
- **GraphRAG verdict layer**: cạnh đồ thị mất tín nhiệm ngay khi hypothesis
  nguồn bị P16 kết luận refuted (`kg_edges.trusted`); neighbourhood đa hop
  (`h1:`/`h2:`, lọc trust) chèn vào JUDGE prompt lúc propose; phát hiện
  mâu thuẫn NGỮ NGHĨA giữa các claim đã chốt ngược phán quyết. CLI:
  `agentics kg-query`, `agentics kg-contradictions`.
- **Wave 2 — tiến hoá & hiệu chuẩn** (2026-11-14, `e00bc29..7319550`): skill
  library kiểu Voyager, tiến hoá prompt DSPy-lite, tầng hiệu chuẩn verify
  (Wilson floor), cây phản tư, frontier EIG (information-gain-per-cost).
- **Wave 1 — TOP 5 advanced stack** (2026-11-14, `04ebaac`): frontier PUCT +
  entropic credit, MAP-Elites + islands, local small-model router (Ollama),
  GraphRAG-lite (`kg_entities`/`kg_edges`), Hypothesis Arena (Elo + dual-judge
  ChatGPT độc lập).
- **Luật neo verdict — `self_authored_ratio`** (2026-11-14, vòng "verdict
  provenance"): trục kiểm duyệt KHÔNG phải "thuật toán vs AI" — mà là *ai cầm
  bút viết thang điểm, và thang điểm có neo vào thứ nằm ngoài tầm với của
  agent hay không*. Mỗi settle giờ đo **tỉ lệ đầu vào do agent chấp bút**
  (protocol / target / data) và ghi vào ledger NGAY CẠNH verdict
  (`self_authored_ratio` + danh sách anchor). **Luật settle:** verdict chỉ
  được ghi `supported` khi ratio < 1 — tức phải có ít nhất một đầu vào ngoài
  tầm với: citation URL từ nguồn ngoài (`citation`), replicate độc lập bởi
  account khác (`replication` — P24), chấp thuận của owner trong local-only
  mode (`python -m agentics.cli approve <id>`), dataset khóa hash mà protocol
  khai báo (`# AGENTICS_ANCHOR: pinned_data <file> <sha256>`), hoặc CI
  (`ci_verdict`, dành chỗ). Verdict "supported" thiếu neo bị hạ xuống trạng
  thái mới **`unanchored`** (đã đo, đạt chuẩn, nhưng chưa tin được) — chỉ
  replicate xác nhận / owner chấp thuận mới nâng lên `supported`
  (`replication_promoted`). Cổng discovery nâng cấp tương ứng:
  `mark_discovery` chỉ khi **hai judge độc lập đồng thuận** (tài khoản thứ
  hai, không phải identity của tác giả) VÀ câu trả lời của judge thứ hai có
  ít nhất một citation URL ngoài. `SETTLE_REQUIRE_ANCHOR=0` /
  `DISCOVERY_SECOND_JUDGE=0` tắt (legacy).
- **Vòng nghiên cứu QD closed-loop** (dựa trên Kosmos / IDEAgent / autoresearch,
  2026-11-14): world model dùng chung (`world_notes` — verified / refuted / gap),
  cổng ý tưởng IDEAgent (judge + repair một vòng + lưu mẫu bị từ chối), insight
  harvest — mỗi kết quả settled tự đề xuất follow-up có dòng dõi (`parents`),
  experiment ledger keep/discard, và gap proposal khi registry cạn.
- **Nhận diện PHÁT HIỆN MỚI + chia nhánh đào sâu** (2026-11-14, vòng "branch on
  discovery"): sau MỌI kết quả settled (cả verify lẫn deep-dive), ideator chấm
  **novelty 0-100** so với world model; vượt ngưỡng `DISCOVERY_NOVELTY_MIN`
  (mặc định 70) kết quả được đánh dấu **discovery** trong registry, world model
  ghi chú `DISCOVERY(...)` (được PIN ở đầu context, không bao giờ chìm dưới
  note thường), và các follow-up trở thành **nhánh có góc tấn công** (mechanism
  / generalize / boundary / amplify) ở `depth+1`. Backlog deep-dive ưu tiên
  **cây nhánh của discovery** trước (`discovery_frontier`, BFS nông trước sâu).
  Feedback bandit-lite: nhánh settled `supported` +5 novelty
  (`discovery_confirmed`), nhánh `refuted` ghi `branch_dead_end` kèm góc chết —
  vòng lặp học được hướng đào nào có lời. `BRANCH_MAX_DEPTH` chặn một phát
  hiện độc chiếm registry.
- **Vòng lặp Co-Scientist** (`agentics/coscientist/`) — port hành vi trung thực
  từ vòng tự cải thiện của Co-Scientist (Gottweis et al., Nature 655, 2026; theo
  bản tái hiện Apache-2.0 tại [Kaimen-Inc/Co-Scientist](https://github.com/Kaimen-Inc/Co-Scientist)):
  giải đấu **Elo** xếp hạng giả thuyết bằng tranh luận khoa học mô phỏng,
  **evolution** lai tạo thế hệ tiếp theo từ top-Elo (combine / out-of-the-box,
  có genealogy), **meta-review** chưng cất mẫu thắng/thua thành phản hồi.
  Elo là vòng CHẤT LƯỢNG — chỉ code chạy thật (P16) mới kết liễu một claim.
- **P16-R — tự sửa protocol thất bại** (`agentics/verify/repair.py`): protocol
  kết luận `inconclusive` được đọc lại lỗi máy, tác giả viết lại script đúng
  MỘT vòng có giới hạn, tiêu chí pre-registered bị đóng băng (không dịch
  vạch đích), phán quyết vẫn 100% của máy.
- **Điều phối Contract-Net** (`agentics/coordination/`): supervisor trở thành
  manager — announce / bid / award phân việc cho agent phù hợp nhất theo lịch
  sử merge thật; **swarm lessons** tổng hợp chữ ký thất bại XUYÊN CẢ ĐÀN
  (cùng lỗi lặp lại 3 agent = bằng chứng hướng đi chết, không còn là "một trace").
- **Nghiên cứu chịu lỗi + đa dạng**: breadth steering ăn diet domain từ shared
  memory (bắt buộc ≥3 nguồn ngoài), fan-out góc nhìn SONG SONG thật trên pool
  token Hyperbrowser, dedup trích dẫn canonical URL, và free keyless tools
  (arXiv / OpenAlex / Crossref / Semantic Scholar / Europe PMC / GitHub / PyPI /
  HuggingFace) khi LLM pool chết.
- **Đánh giá bầy đàn**: `agentics swarm-eval` (quỹ đạo, cộng tác, hội tụ, chịu
  lỗi, chi phí) + harness pre-production `scripts/live_preprod_eval.py` —
  số liệu trung thực, kể cả khi xấu (xem [mục 8](#8-đánh-giá-bầy-đàn--pre-production)).

---

## 1. Triết lý thiết kế

Agentics AI Research Lab được xây dựng theo nguyên lý **"đàn kiến nghiên cứu" (research swarm)**:

- **Mỗi luồng agent = một session LLM riêng.** Hệ thống chạy đa luồng
  (`asyncio`); pool token Hyperbrowser càng nhiều thì càng chạy được nhiều
  cloud session song song thật (fan-out nghiên cứu chạy trên các token KHÁC,
  không giành một session).
- **Agent tự chọn vai (role) hoặc được phân vai thích nghi** — Supervisor,
  Researcher, Coder, Writer, Reviewer, Integrator. Bịn một agent đơn độc tự
  thành researcher; hai agent thành supervisor + researcher; đàn lớn round-robin
  với supervisor/integrator luôn đảm bảo (xem `role_map` trong `configs/lab.yaml`).
- **Agents làm việc trên repo như một chuyên gia GitHub thật** (chế độ GitHub)
  hoặc **hoàn toàn local** (mặc định trên nhánh này): quét việc → claim → tạo
  nhánh → code / chạy lệnh → review chéo → merge vào bare repo `data/lab.git`.
- **Trao đổi qua bộ nhớ chung + cá nhân.** Mọi finding, quyết định, bài báo
  trích dẫn, bằng chứng thực nghiệm đều ghi vào Shared Memory (SQLite WAL) và
  Vector Store (ChromaDB); mỗi agent có Personal Memory ghi bài học riêng; và
  giờ có thêm **swarm lessons** — bài học của CẢ ĐÀN.
- **Vòng lặp nghiên cứu vô tận.** Agents chạy chu kỳ liên tục cho đến khi owner
  tạo file `STOP` (hoặc bấm `Ctrl+C`) — mọi luồng dừng an toàn giữa chu kỳ.
- **Hai vòng lặp riêng biệt, không trộn lẫn:**
  - **Vòng SỰ THẬT (P16/P16-R/P23/P24):** một giả thuyết chỉ được kết liễu bởi
    code chạy thật — exit code + metric đo được, tái lập 2 lần, replicate độc lập.
  - **Vòng CHẤT LƯỢNG (Co-Scientist):** Elo tournament + evolution + meta-review
    xếp hạng và lai tạo ý tưởng. Elo KHÔNG BAO GIỜ kết liễu một claim; giả thuyết
    bị code phản chứng không được vào bể đấu dù tranh luận thấy nó đẹp thế nào.

**Miền nghiên cứu mặc định:** AI/ML cho NFV (Network Function Virtualization) —
đổi chủ đề chỉ bằng một dòng `research_topic` trong `configs/lab.yaml`.

---

## 2. Kiến trúc tổng quan

```
┌──────────────────────────── Orchestrator (asyncio) ──────────────────────────────┐
│                                                                                   │
│  ┌─ Thread 1 (token-01) ─┐  ┌─ Thread 2 (token-02) ─┐  ...  ┌─ Thread N ──────┐ │
│  │  LangGraph ResearchCycle│  │  LangGraph ResearchCycle│       │  ResearchCycle │ │
│  │  scan→claim→research→   │  │                          │       │                 │ │
│  │  plan→implement→verify→ │  │                          │       │                 │ │
│  │  review→integrate→reflect│ │                          │       │                 │ │
│  └───────────┬─────────────┘  └───────────┬──────────────┘       └───────┬─────────┘ │
│              │                            │                              │           │
│   ┌──────────┴────────────────────────────┴──────────────────────────────┴────────┐ │
│   │          Shared Memory (SQLite WAL) + ChromaDB + bare repo (data/lab.git)     │ │
│   │     messages · findings · decisions · claims · evidence · Elo standings       │ │
│   └───────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                   │
│  RESEARCH NODE: fan-out song song (góc benchmarks / open problems / production)   │
│    → mỗi góc chạy trên token pool KHÁC; breadth steering ép ≥3 nguồn ngoài diet   │
│    → LLM chết? free keyless tools (arXiv/OpenAlex/Crossref/...) dựng finding thật │
│                                                                                   │
│  TRUTH LOOP (P16/P16bis/P16-R): plan khai báo verify_cmds + hypothesis có tiêu    │
│    chí định lượng TRƯỚC khi chạy → protocol chạy 2 lần trong workspace có guard   │
│    → máy so metric vs tiêu chí → confirmed / refuted / inconclusive               │
│    → inconclusive: P16-R đọc lỗi máy, viết lại script 1 vòng, tiêu chí đóng băng  │
│    → settled: P23 replay + P24 replication độc lập; P21 audit bất biến            │
│                                                                                   │
│  QUALITY LOOP (Co-Scientist): Elo tournament (debate 1 ask/match, K theo kinh     │
│    nghiệm, zero-sum) → evolution (combine / out-of-box từ top-k) → meta-review    │
│    (mẫu thắng/thua → phản hồi vòng sau) — thất bại cô lập, qua cổng P20           │
│                                                                                   │
│  COORDINATION: Contract-Net-lite (announce/bid/award) · reviewer dossier ·        │
│    swarm lessons · P18 peer panel + defense · P13 file locks · P10 protected      │
│                                                                                   │
│  LLM: Perplexity qua Hyperbrowser cloud browser (pool N token, stealth + proxy)   │
│  Budget: P20 cap giờ + circuit breaker · Stop: file STOP + SIGINT graceful        │
└───────────────────────────────────────────────────────────────────────────────────┘
```

Chi tiết đầy đủ: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) ·
Chính sách chống xung đột: [docs/POLICIES.md](docs/POLICIES.md) ·
Co-Scientist: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) + `agentics/coscientist/__init__.py` ·
Điều phối: [docs/COORDINATION.md](docs/COORDINATION.md) ·
Production checklist: [docs/PRODUCTION.md](docs/PRODUCTION.md)

---

## 3. Cài đặt nhanh

> **Chế độ LOCAL-ONLY (nhánh mặc định):** lab chạy hoàn toàn trên máy của bạn —
> không cần `GITHUB_TOKEN`, không gọi GitHub. "origin" của agent là bare repo
> tự seed tại `data/lab.git`; tasks/MR/review/merge đều nằm trong SQLite.
> Owner theo dõi qua web dashboard, terminal live view hoặc stream sự kiện.
> Chi tiết: [docs/LOCAL_MODE.md](docs/LOCAL_MODE.md).

```bash
git clone https://github.com/shieldx-bot/agentics-ai-research-lab.git
cd agentics-ai-research-lab

python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) Cấu hình chạy — KHÔNG cần token GitHub
cp .env.example .env             # tuỳ chỉnh RESEARCH_TOPIC / MAX_AGENTS
#    agentics/config.py TỰ NẠP .env khi khởi động (không cần `source .env`).
#    Biến môi trường thật trong shell vẫn luôn thắng file .env.

# 2) LLM backend — Perplexity qua Hyperbrowser cloud browser
#    (bắt buộc trên IP datacenter; browser_backend: hyperbrowser trong lab.yaml)
cp configs/secrets.example.yaml configs/secrets.yaml   # gitignored, chmod 600
#    điền pool token: hyperbrowser_api_keys: [hb_..., hb_...]
#    1 token = 1 cloud session; K token = K session song song thật
#    (tương đương env: HB_API_KEYS="tk1,tk2,tk3")
#    LLM cookie local: đặt cookie accounts/account-01.json, account-02.json...

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
python -m agentics.cli mr approve 3     # owner duyệt thay/thêm cross-review
python -m agentics.cli mr merge 3       # owner merge (P10 + tier-2 vẫn chạy)

# 7) DỪNG an toàn (owner quyết định)
python -m agentics.cli stop          # hoặc: touch STOP
```

Xem trạng thái: `python -m agentics.cli status`  
Đo tiến hóa của lab: `python -m agentics.cli metrics` — [docs/EVOLUTION.md](docs/EVOLUTION.md)  
Sức khỏe bộ nhớ LifeMem: `python -m agentics.cli memory-stats` — [docs/MEMORY.md](docs/MEMORY.md)

### Các khóa cấu hình mới quan trọng (`configs/lab.yaml`)

| Khóa | Mặc định | Ý nghĩa |
|---|---|---|
| `research_fanout` | `3` | Số góc nhìn nghiên cứu chạy SONG SONG trên pool token (1 = tắt) |
| `research_same_view_angle` | `true` | Pool 1 token vẫn chạy thêm góc tuần tự trên chính session của agent |
| `tournament_enabled` | `true` | Bật vòng Co-Scientist (Elo + evolution + meta-review) |
| `tournament_matches_per_cycle` | `2` | Số trận tranh luận mỗi chu kỳ researcher (chặn budget) |
| `evolution_per_cycle` / `evolution_top_k` | `1` / `3` | Số ask lai tạo / bể top-Elo rút để lai |
| `verify_protocol_repair_rounds` | `1` | Số vòng P16-R tự sửa protocol (0 = tắt) |
| `free_tools_enabled` | `true` | Fallback API học thuật keyless khi LLM pool chết |

---

## 4. Vòng SỰ THẬT — kiểm chứng bằng code chạy thật

**P16/P16bis — agent tự kiểm chứng ý tưởng bằng code chạy thật:** một ý tưởng
không được đánh giá bằng lời kết luận của LLM mà bằng **bằng chứng thực nghiệm**
— plan khai báo `verify_cmds`/`hypotheses` (chốt tiêu chí định lượng TRƯỚC khi
chạy, như pre-registration), protocol được viết và chạy 2 lần trong workspace
có guard, máy so sánh số đo với tiêu chí để phán
`confirmed/refuted/inconclusive` (protocol không tái lập được → inconclusive,
không bao giờ đếm là thành công), và mọi verdict được broadcast cho cả đàn +
inject vào prompt từng agent (`verified_knowledge`).

**P16-R — Reflexion cho protocol thất bại** (`agentics/verify/repair.py`):
khi protocol ra `inconclusive` vì script tự viết của agent vỡ contract
`AGENTICS_RESULT` (lỗi máy cơ học, không phải ý tưởng sai), tác giả nhận lại
chính protocol + tiêu chí ĐÓNG BĂNG + ghi chú lỗi máy, viết lại script đúng
một vòng; phán quyết chạy lại bằng đúng máy chấm cũ. Sửa script — không bao
giờ sửa logic chấm: đó là ranh giới giữa "tự cải thiện" và "lách metric".

**P23/P24 — replay & replication:** claim đã settled được chạy lại từ protocol
đã lưu (`verify-all` / `replay`) và reviewer lặp lại độc lập mỗi chu kỳ. Owner
xem bằng `agentics evidence` / `agentics hypotheses`; Reviewer **tự chạy lại
check của tác giả trên nhánh thật + full test suite (tier-2) trước khi hỏi
LLM** — MR không qua được test bị từ chối với 0 token LLM.

Chi tiết: [docs/VERIFICATION.md](docs/VERIFICATION.md).

## 5. Vòng CHẤT LƯỢNG — giải đấu Co-Scientist

Port trung thực từ vòng tự cải thiện mà ablation của paper Co-Scientist ghi
điểm cao nhất (`agentics/coscientist/`):

| Thành phần | File | Vai trò |
|---|---|---|
| Elo math | `coscientist/elo.py` | Zero-sum, K theo kinh nghiệm (new entrants K=32, kỳ cựu K=16) |
| Tournament | `coscientist/tournament.py` | 1 trận = 1 ask tranh luận; cặp chọn theo information-per-match; crash-idempotent (match_id xác định → INSERT OR IGNORE) |
| Pairing | `coscientist/pairing.py` | Tân binh, cặp Elo gần nhưng ý tưởng xa, thỉnh thoảng top-heavy |
| Evolution | `coscientist/evolution.py` | `combine` (ghép cặp ý tưởng xa nhất trong top-k thành giả thuyết testable hơn) + `out_of_box`; con có `parents`, đi cùng vòng đời propose→P16 |
| Meta-review | `coscientist/metareview.py` | Chưng cất mẫu thắng/thua thành phản hồi cho vòng sau |

**Ba ranh giới trung thực giữ nguyên khi port:**
1. **Refuted không vào bể đấu** — claim bị code falsified là ý tưởng chết, dù
   tranh luận thấy đẹp.
2. **Elo không kết liễu claim** — tournament thêm chiều CHẤT LƯỢNG trên vòng
   SỰ THẬT P16, không thay thế.
3. **Mọi ask qua cổng P20** và toàn bộ node thất bại-cô-lập: giải đấu có thể
   hỏng mà không bao giờ làm vỡ chu kỳ nghiên cứu.

Owner xem bảng xếp hạng: `agentics standings`; đấu thử một batch qua LLM thật:
`agentics tournament`.

## 6. Chống xung đột giữa các agents (tóm tắt)

1. **Claim locking + TTL** — việc chỉ có 1 chủ; claim hết hạn tự được nhặt lại.
2. **Nhánh riêng cho từng agent** — không agent nào push vào nhánh agent khác.
3. **Ownership matrix + P14 whitelist ring** — mỗi role một vùng ghi; diff lệch
   phạm vi intent bị yield + báo dashboard.
4. **P13 file locks** — file dự kiến ở bước plan bị khóa toàn cục, TTL tự hết.
5. **Merge queue tuần tự** — chỉ Integrator merge, từng PR một, rebase trước merge.
6. **Cross-review + P18 panel bắt buộc** — không merge PR do chính mình viết;
   đồng thuận 2 reviewer, tác giả có cửa defense có hạn.
7. **Heartbeat & stale detection** — agent chết > 2 chu kỳ → claim được giải phóng.
8. **P10 protected paths (4 tầng)** — `agentics/ configs/ accounts/ scripts/
   .github/` chặn ở ghi file, diff trước push, cửa merge + IntegrityWatcher tự
   hoàn tác (~20s). Exception DUY NHẤT: `agentics allow-protected <task_id>`
   (`accounts/` và `configs/secrets.yaml` không bao giờ được miễn).

Chi tiết: [docs/POLICIES.md](docs/POLICIES.md)

## 7. Điều phối A2X — Contract-Net, dossier, swarm lessons

- **Contract-Net-lite** (`coordination/allocation.py`): supervisor quét backlog,
  agent sống bid theo Performance–Novelty score trên lịch sử merge của chính
  nó, award greedy trên ma trận điểm toàn cục — mỗi task một chủ, mỗi agent
  một task, tie-break deterministic, ghi `task_awards` và viết lại mỗi scan.
- **Reviewer dossier** (`coordination/dossier.py`): lời mời review kèm lý lịch
  người review — review có căn cứ, không review theo mood.
- **Swarm lessons** (`coordination/lessons.py`): chữ ký thất bại chuẩn hóa +
  tổng hợp XUYÊN CẢ ĐÀN — planner thấy PATTERN ("hướng này 3 agent đều chết")
  thay vì đọc từng trace row rời rạc.

Chi tiết: [docs/COORDINATION.md](docs/COORDINATION.md)

## 8. Đánh giá bầy đàn + pre-production

**`agentics swarm-eval`** — điểm bầy đàn AS A SWARM qua 5 nhóm chỉ số: hoàn
thành nhiệm vụ & phối hợp, hội tụ đúng, quỹ đạo chu kỳ, chịu lỗi & chi phí,
chất lượng phản biện (incoherence, quality gap). Bản đánh giá đầy đủ đầu tiên
đã chạy **trước production** bằng `scripts/live_preprod_eval.py` (topic thật,
7 phút, 1 agent researcher, pool 7 token Hyperbrowser, cửa sổ metrics sạch
qua marker DB):

- Kết quả thô: 8 findings · 7 hypotheses · 6 evidence · 12 trích dẫn phân bổ
  6 host khác nhau · tournament 5 giả thuyết / 4 trận / 1 meta-feedback ·
  LLM ok-rate 100%.
- Số liệu trung thực (chưa đẹp): giả thuyết quyết 3/7 (42.9%), evidence thông
  kiểm chứng máy 1/6 (16.7%), độ bao phủ tìm kiếm 50%, replication 0/1 —
  các con số này được ghi nhận NGUYÊN VẸN, không làm đẹp.
- Đánh giá thực đã bắt và sửa 2 bug thật: truyền Elo b-win bị đảo chiều
  (`848abb6`) và protocol chạy không-marker bị ghi `refuted` thay vì
  `inconclusive` (`b359557`) — đúng mục đích của việc đánh giá trước production.

## 9. Free research tools — nguồn bằng chứng thật, không cần API key

Node research không phụ thuộc mạng sống/chết của cookie LLM. Khi LLM fail,
finding được dựng từ metadata thật của **arXiv / OpenAlex / Crossref /
Semantic Scholar / Europe PMC** (tác giả, năm, abstract, số trích dẫn — fetch
từ API chính thức, không phải LLM đẻ ra) và BibTeX được làm giàu bằng metadata
nhà xuất bản. `search_all` chạy 3 nguồn song song trên 3 budget rate-limit
độc lập. Kèm **verification toolkit**: HuggingFace datasets + models, GitHub
repo search, PyPI package lookup (version thật để pin vào `verify_cmds`) —
tất cả keyless, degrade graceful, tôn trọng rate limit. Owner dùng trực tiếp:

```bash
python -m agentics.cli papers "ML for NFV"                       # bảng bài báo + BibTeX thật
python -m agentics.cli papers --doi 10.1109/netsoft48620.2020.9165348
python -m agentics.cli papers "intrusion detection" --datasets
python -m agentics.cli papers "nfv scheduler" --github            # repo tham chiếu
python -m agentics.cli papers --pypi scikit-learn                 # version thật để pin
```

Chi tiết: [docs/FREE_TOOLS.md](docs/FREE_TOOLS.md).

## 10. Vai trò (roles) & luồng công việc

| Role | Nhiệm vụ chính |
|---|---|
| **Supervisor** | Quét việc, announce Contract-Net, award, dọn claim chết, tổng hợp swarm lessons |
| **Researcher** | Fan-out đa góc, breadth steering, trích dẫn chuẩn BibTeX/APA, đăng ký hypothesis + protocol |
| **Coder** | Tạo nhánh, viết code, chạy lệnh/thử nghiệm trong workspace cục bộ |
| **Writer** | Viết/sửa tài liệu, tóm tắt kết quả nghiên cứu thành report |
| **Reviewer** | Cross-review, tự chạy tier-2 test, replicate độc lập (P24), bỏ phiếu panel |
| **Integrator** | Duyệt hàng đợi merge, rebase, merge — **tuần tự, một mình** |

```
scan → claim → research → plan → implement → verify (P16/P16-R)
     → self_review → cross_review → integrate → reflect → lặp lại
              ↑_______________________________________________|
              (Co-Scientist tournament + meta-review xếp hạng & lai tạo giữa các chu kỳ)
```

## 11. Bảo mật — ĐỌC KỸ

- **KHÔNG BAO GIỜ** commit token, file cookie, hay nội dung `.env` /
  `configs/secrets.yaml` lên repo. Toàn bộ nằm trong `.gitignore` từ commit đầu tiên.
- `accounts/` và `configs/secrets.yaml` **không bao giờ được miễn** bởi bất kỳ
  exemption nào (P10) — credentials không bao giờ là bề mặt ghi của agent.
- Cookie Perplexity = phiên đăng nhập thật. Chỉ dùng cookie của tài khoản
  **của bạn**; việc dùng cookie để gọi API có thể vi phạm Điều khoản dịch vụ
  của Perplexity — bạn tự chịu trách nhiệm. Mã nguồn dành cho mục đích nghiên
  cứu cá nhân / học thuật.
- Nếu token từng dán vào chỗ công khai (chat, issue…), **thu hồi và tạo token
  mới ngay**. `scripts/hb_session_cleanup.py` quét dọn session Hyperbrowser
  trên TOÀN BỘ pool sau mỗi phiên chạy (không để session sống sót treo billing).
- Mã không tin cậy chạy trong **rootless container** (P2) — xem
  [docs/PRODUCTION.md](docs/PRODUCTION.md).

## 12. Tài liệu

| Tài liệu | Nội dung |
|---|---|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Kiến trúc chi tiết + vòng Co-Scientist |
| [POLICIES.md](docs/POLICIES.md) | Toàn bộ policy P1–P24 |
| [VERIFICATION.md](docs/VERIFICATION.md) | P16/P16bis/P16-R/P23/P24 — vòng sự thật |
| [COORDINATION.md](docs/COORDINATION.md) | Contract-Net, dossier, swarm lessons |
| [LOCAL_MODE.md](docs/LOCAL_MODE.md) | Chế độ local-only (bare repo, không GitHub) |
| [PRODUCTION.md](docs/PRODUCTION.md) | Checklist chạy thật: env, quan sát, backup |
| [FREE_TOOLS.md](docs/FREE_TOOLS.md) | API học thuật keyless + verification toolkit |
| [EVOLUTION.md](docs/EVOLUTION.md) | Đo tiến hóa lab qua traces |
| [MEMORY.md](docs/MEMORY.md) | LifeMem: trace, cluster, skill chưng cất |
| [LLM_PROVIDERS.md](docs/LLM_PROVIDERS.md) | Perplexity + Hyperbrowser cloud browser |
| [CONTRIBUTING.md](docs/CONTRIBUTING.md) | Phát triển nhiều người cùng lúc, ít xung đột |
| [ADVANCED_TECH_ROADMAP.md](docs/ADVANCED_TECH_ROADMAP.md) | TOP 5 công nghệ tiên tiến dự kiến đưa vào (GEPA, router, MCP, GraphRAG, verifier học được) |
| [ADVANCED_TECH_ROADMAP_WAVE3.md](docs/ADVANCED_TECH_ROADMAP_WAVE3.md) | TOP 5 wave 3 (semantic cache, hybrid rerank, MCP, RLVR gate, fulltext + citation grounding) — wave-1/2 đã tích hợp, xem [ADVANCED_STACK.md](docs/ADVANCED_STACK.md) |
| [TECH_RADAR_WAVE3.md](docs/TECH_RADAR_WAVE3.md) | Wave 3 bổ trợ — hardening & trust: sandbox cấp OS, prompt-injection defense, thống kê nghiệm, OTel tracing, chaos harness (4/5 chưa có ở bất kỳ proposal nào) |

## English quickstart

```bash
pip install -r requirements.txt
cp .env.example .env                              # optional tuning; no GITHUB_TOKEN needed
cp configs/secrets.example.yaml configs/secrets.yaml   # put your Hyperbrowser pool here
python -m agentics.cli run        # infinite research loop — 100% local
python -m agentics.cli dashboard  # owner's live view (http://127.0.0.1:8080)
python -m agentics.cli standings  # Elo table of the hypothesis tournament
python -m agentics.cli stop       # owner presses the brakes
```

Two loops, never mixed: the **truth loop** settles a hypothesis only by
running code (P16 exit-code verdicts, frozen pre-registered criteria, bounded
P16-R self-repair, P23 replay + P24 independent replication); the **quality
loop** (Co-Scientist Elo tournament, evolution, meta-review) ranks and breeds
ideas — it can never settle a claim.

## License

MIT — see [LICENSE](LICENSE).
