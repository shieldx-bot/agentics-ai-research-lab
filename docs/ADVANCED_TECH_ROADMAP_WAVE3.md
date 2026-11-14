# Lộ trình công nghệ tiên tiến — TOP 5 wave 3

> Tiếp tục nguyên tắc owner: **không đụng core** (graph/orchestrator — badge
> 873 test xanh). Wave-1 (PUCT + entropic credit, MAP-Elites + islands, local
> router, GraphRAG-lite, Elo Arena — xem `docs/ADVANCED_STACK.md` §1–2) và
> wave-2 (skill library, DSPy-lite prompt evolution, EIG frontier, verify
> calibration, reflection tree — §5) đã tích hợp ngày 2026-11-14. Wave-3 này
> chọn **5 lớp add-on CHƯA có trong codebase** (đã soát: không có `mcp`,
> `rerank`, `fts5`, semantic cache, `verify/gate.py`, fulltext PDF trong
> `agentics/`). Mỗi mục: cái gì, vì sao hợp lab, kế hoạch gắn vào điểm móc
> sẵn có, ranh giới trung thực.

## Bảng so sánh nhanh

| # | Công nghệ | Tác động | Công sức | Gắn vào | Thứ tự gợi ý |
|---|---|---|---|---|---|
| 1 | Semantic cache cho pool asks | Pool cookie ↓ 30–70% tiềm năng, hết cơn "121/120 calls" | Nhỏ | `llm/pool.py` + bảng `llm_cache` | **Làm đầu tiên** |
| 2 | Hybrid retrieval + cross-encoder rerank | Recall/novelty precision ↑ cho mọi consumer | Nhỏ–vừa | `memory/vector_store.py` + FTS5 | Song song #1 |
| 3 | MCP client — mở khoá hệ sinh thái tool | Tool mới = khai báo config, không viết code | Vừa | `tools/mcp_client.py` (mới) + `tool_registry` | Sau #1 |
| 4 | Gate rủi ro học từ ledger (RLVR-lite) | `mean_panel_rounds` ↓ (G2) | Vừa | `verify/gate.py` (mới) + traces/panels | Shadow song song |
| 5 | Fulltext ingestion + citation grounding | Evidence đọc cả paper, chống citation ảo | Lớn (chia 3 pha) | `research/fulltext.py` (mới) + chroma | Sau #3 |

---

## 1. Semantic cache cho pool asks (GPTCache-style, tự viết nhẹ)

**Cái gì.** Semantic cache lưu answer theo khoá prompt; câu hỏi trùng → trả
từ cache, không đụng pool. GPTCache (Bang, NLP-OSS 2023) đo tăng tốc 2–10x
khi cache hit; các khảo sát 2026 ghi nhận mức giảm 30–70% chi phí inference
cho các ứng dụng LLM có query lặp.

**Vì sao hợp lab.** Pool cookie là tài nguyên khan hiếm nhất của lab — chính
là nguồn của sự cố 07:08 2026-11-14 (121/120 calls, science bị starve) mà
local router (wave-1 #3) chỉ cứu được nhóm plumbing. Các purpose
`research`/`ideation` hỏi câu survey tương tự nhau qua nhiều cycle khi topic
chưa đổi; wave-1 roadmap từng đo: purpose "research" chiếm 42% tổng call, mean
79.6s/call. Cache tấn công đúng phần này mà không cần model gì thêm.

**Kế hoạch.**
1. Bảng `llm_cache`: `prompt_hash` (SHA-256 của purpose+mode+prompt chuẩn
   hoá) → `response`, `ts`, `hit_count`; `agentics/llm/pool.py` hỏi cache
   trước khi route, ghi lại sau khi có answer; journal event `cache_hit`.
2. Bound: TTL 24h, max rows (roll như `llm_usage`), response truncate; chỉ
   cache answer từng `ok=1`.
3. Cờ `LLM_CACHE` (mặc định off) → shadow 1 tuần (chỉ đếm "would-hit") →
   bật khi hit-rate > 15% và không có regression ở G1/G5.

**Ranh giới trung thực.** KHÔNG cache purpose phán quyết — `verify`, `review`,
`defend`, `tournament`, `repair`, `implement` (cùng blacklist triết lý của
`local_router.allowed_for()`: verdict phải tươi). Pha 1 chỉ exact-hash (không
false-hit); near-dup bằng similarity chỉ xét sau khi có bằng chứng hit-rate
quá thấp — similarity dễ trả câu GẦN đúng cho purpose research.

## 2. Hybrid retrieval + cross-encoder rerank

**Cái gì.** Retrieval 2 giai đoạn: recall rộng = hợp nhất BM25 (SQLite FTS5)
+ vector chroma; sau đó rerank top-k bằng cross-encoder
(`BAAI/bge-reranker-v2-m3` — 568 tham số M, đa ngữ, chuẩn cộng đồng; các
benchmark 2026 cho thấy thêm tầng rerank sau dense retrieval tăng đều chất
lượng top-k).

**Vì sao hợp lab.** `vector_store.py` hiện chỉ có chroma vector phẳng với
embedding MiniLM mặc định: (a) mất truy vấn từ khoá chính xác — tên
method/dataset/metric là đặc thù domain research mà BM25 mạnh hơn hẳn dense;
(b) không có rerank nên top-5 hay lẫn bản "nearly-relevant". Mọi consumer
đang ăn từ retrieval này: `kg_context` (GraphRAG wave-1), paper dedup P7,
novelty check, skill retrieval top-3 (wave-2) — nâng một tầng này thì cả
stack hưởng.

**Kế hoạch.**
1. Index FTS5 `findings_fts` (đồng bộ khi ghi bảng gốc); hybrid =
   union(BM25 top-20, chroma top-20) → rerank → top-5. Fallback: nếu build
   SQLite không có FTS5 thì dùng keyword search sẵn có (`polices.py`).
2. Knob `RERANK` (mặc định off → shadow 1 tuần log thứ tự cũ/mới); rerank
   async, chỉ top-20, không block cycle.
3. Knob `EMBED_MODEL` (mặc định giữ MiniLM; `bge-m3` opt-in, re-embed lazy
   vào collection mới — không ép migration).

**Ranh giới.** Rerank 568M trên CPU tốn ~giây/cặp — chỉ dùng cho retrieval
vào prompt, không đưa vào hot path phán quyết; đổi embedding model phải
re-embed toàn collection cũ hoặc chấp nhận 2 collection song song一段 thời
gian.

## 3. MCP client — chuẩn hoá lớp tool, mở khoá hệ sinh thái

**Cái gì.** Model Context Protocol đã là chuẩn de facto kết nối agent–tool:
spec hiện hành 2026-07-28, transport Streamable HTTP cho server chạy remote,
primitive Tasks (SEP-1686) cho giao tiếp agent–agent, được Linux Foundation
(Agentic AI) host và vận hành theo Working Groups. Cộng đồng phát hành hàng
nghìn MCP server: arXiv, GitHub, database, filesystem, browser…

**Vì sao hợp lab.** Lab vẫn tự bọc từng tool (`free_tools.py`: arXiv,
OpenAlex, Crossref, Semantic Scholar, Europe PMC, PyPI, Wikipedia, GitHub —
mỗi tool tự viết retry/rate-limit/parse) + `tool_registry` SQL tự quản. MCP
thay lớp glue: một client MCP → khai báo capability vào `tool_registry`,
agent dùng như tool thường; xoá tool = xoá cấu hình, không phải xoá code.
Đây là mục #3 của roadmap wave-1 **chưa từng được tích hợp** (wave-1 thực tế
landed PUCT/MAP-Elites/local-router/GraphRAG/Arena) — nguyên văn kế hoạch
vẫn đúng, giờ bổ sung dẫn chứng 2026.

**Kế hoạch.**
1. `agentics/tools/mcp_client.py`: client gọn stdio + Streamable HTTP
   (không cần SDK nặng); server khả dụng khai báo trong
   `configs/lab.yaml: tools.mcp_servers:` do owner duyệt từng capability.
2. Bridge: khi MCP server lên → `register_tool()` vào `tool_registry`
   (P21 dùng lại như cũ); `free_tools` giữ làm fallback keyless khi MCP chết.
3. Bước kế tiếp: một MCP server tự viết (bọc `evidence`, `hypotheses`,
   `swarm-eval`) = lab mở API chuẩn cho agent ngoài — nửa đường tới A2A.

**Ranh giới.** Tool qua MCP vẫn phải qua cổng P20 và sandbox như tool nội;
không tin schema server lạ — capability phải được owner duyệt trong config.

## 4. Gate rủi ro học từ chính ledger (RLVR-lite)

**Cái gì.** Không cần RL đầy đủ: distill cặp (đầu vào → phán quyết máy) mà
lab đã gom thành một gate học được — model nhỏ (logistic/GBM sklearn) dự
đoán xác suất MR phải rework TRƯỚC khi mở panel. Đây là mục #5 roadmap
wave-1 mà wave-2 chỉ làm được nửa calibration (floor confidence của verdict,
chưa phải predictor cho panel rounds).

**Vì sao hợp lab.** Ledger đã gom đủ bảng supervised có nhãn trung thực:
`traces` (attempted/failure → outcome), `mr_panels` + `panel_votes` (nội
dung MR → vote), `evidence` (exit_code/duration → verdict),
`trace_clusters`, `experiment_ledger`, `swarm_eval_snapshots` — phán quyết
của code chạy thật (P16), không phải opinion. KPI thẳng: G2
`mean_panel_rounds` — mỗi rework tránh được là một vòng panel tiết kiệm.

**Kế hoạch.**
1. `agentics/verify/gate.py`: train offline từ ledger (sklearn — có sẵn
   trong env, cần pin vào `requirements.txt`); features: số attempt,
   failure-signature (trace_clusters), domain/category, độ dài protocol,
   skill match; label = merged/rework. Export trọng số vào DB, không ship
   model file trong repo.
2. Shadow ≥ 2 tuần (journal `gate_prediction`, không chặn) → đối chiếu
   AUC ≥ 0.7 → mới bật "chặn mềm": dự đoán rủi ro cao → yêu cầu 1 rework
   tự-fix trước khi mở panel, không cấm.
3. Nâng cấp sau: gate làm prior cho Contract-Net award
   (`coordination/allocation.py` đang dùng lịch sử merge thật — gate tinh
   hoá chính tín hiệu đó).

**Ranh giới.** Gate không bao giờ kết liễu claim hay thay panel — chỉ sắp
thứ tự/đẩy rework; phán quyết cuối vẫn 100% P16 + P19. Chống self-confirm:
train lại chỉ từ dữ liệu đã chốt, không nuốt dự đoán của chính mình (cùng
nguyên tắc calibration wave-2).

## 5. Fulltext ingestion + citation grounding (3 pha)

**Cái gì.** (a) Lazy-ingest fulltext của đúng paper được cite (arXiv PDF,
Europe PMC full text — đều keyless, nguồn đã có trong `free_tools`) → chunk
có provenance → embed vào collection riêng → lab đọc CẢ PAPER thay vì
abstract; (b) citation grounding: kiểm tra trích dẫn có tồn tại và snippet có
support claim — vì LLM-based verification vẫn hallucinate ~20% citation
(Misra et al., AAAI 2026).

**Vì sao hợp lab.** Evidence hiện tại là survey qua Perplexity academic mode
(`papers.py`) + metadata free_tools — chưa lần nào lab đọc section method của
paper nó cite. GraphRAG (kg_extract) chỉ mining được trên text đã có: corpus
fulltext làm graph sâu và multip-hop thật hơn hẳn. Grounding bổ nghĩa cho
bảng `evidence` + `citations.py` sẵn có.

**Kế hoạch.**
1. Pha 1 — `agentics/research/fulltext.py`: lazy (chỉ paper được cite trong
   finding mới fetch), bounded (tối đa N paper/ngày, disk quota); pdfplumber
   (đã có trong env, cần pin `requirements.txt`) extract thô → chunk
   800–1200 token kèm provenance (title/section) → chroma collection
   `papers_fulltext`; kg_extract chạy tiếp trên corpus này.
2. Pha 2 — `citations.py` thêm `grounding_check(citation)`: snippet
   retrieval từ collection + entailment score; fail → journal
   `unverified_citation` (flag cho panel, không auto-reject).
3. Pha 3 (tuỳ chọn) — GROBID service khi có Docker/Java: TEI structured
   (section/figure/reference) chính xác hơn pdfplumber thô; chỉ là backend
   thay thế, không đổi schema.

**Ranh giới.** Không ingest cả kho arXiv — lazy + bounded là điều kiện sống
còn cho sandbox; pdfplumber thô đủ cho pha 1; grounding chỉ flag — phán
đoán vẫn 100% P16/P19.

---

## Điều gì cố tình KHÔNG đưa vào top 5 (wave 3)

- **A2A protocol đầy đủ**: vẫn 1 máy, 1 key — chưa có bài toán interop đa tổ
  chức; MCP (#3) là nửa đường đúng (server MCP tự viết sau này chính là
  endpoint chuẩn cho agent ngoài).
- **Tracing UI (Langfuse/OTel)**: journal + bộ KPI queries
  (`ADVANCED_STACK.md` §3) đã phủ ops; thứ lab thiếu là capability, không
  phải thêm một UI observe.
- **Tree-search orchestration (LATS/MCTS đầy đủ)**: PUCT + EIG đã là directed
  search; LATS trùng cơ chế và tốn call gấp nhiều lần.
- **Context compression (LLMLingua kiểu)**: handoff đã ngắn; đợi G3
  `trajectory_length_nodes` tăng bất thường mới đáng.
- **Deep-research API làm nguồn evidence**: đốt pool cookie và không tái lập
  được — reproducibility của evidence là giá trị cốt lõi của lab.

## Thứ tự triển khai gợi ý

1. **Tuần 1**: #1 cache (shadow đo would-hit) + #2 FTS5 index (rerank
   shadow log thứ tự cũ/mới).
2. **Tuần 2**: #3 MCP client + bridge `tool_registry`; bật #1/#2 nếu metric
   tốt.
3. **Tuần 3–4**: #4 gate train lần đầu (shadow); #5 pha 1 fulltext lazy.
4. Scoreboard vẫn là `agentics swarm-eval --percent`: G5
   `llm_seconds_per_merged` ↓, G2 `mean_panel_rounds` ↓, G4
   `role_prompt_success_rate` giữ, G1 `task_completion_rate` không phá.
   Metric mới cần thêm: cache hit-rate, rerank lift@5, gate AUC, % finding
   có grounded citation.

## Dẫn chứng

- GPTCache — Bang, *An Open-Source Semantic Cache for LLM Applications*,
  NLP-OSS @ EMNLP 2023: <https://aclanthology.org/2023.nlposs-1.24>
- bge-reranker-v2-m3 (BAAI, cross-encoder đa ngữ 568M):
  <https://huggingface.co/BAAI/bge-reranker-v2-m3>
- MCP 2026 roadmap (transport, Tasks/SEP-1686, governance):
  <https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap> ·
  enterprise adoption (Linux Foundation hosting):
  <https://www.cdata.com/blog/2026-year-enterprise-ready-mcp-adoption>
- GROBID (PDF → TEI, cascade của sequence-labeling models):
  <https://grobid.readthedocs.io>
- Citation hallucination ~20% khi LLM-verify — Misra et al., AAAI 2026:
  <https://ojs.aaai.org/index.php/AAAI/article/view/42257>
