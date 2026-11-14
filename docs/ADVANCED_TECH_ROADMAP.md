# Lộ trình công nghệ tiên tiến — TOP 5 cần đưa vào

> **Trạng thái 2026-11-14: #1 (GEPA) và #4 (GraphRAG verdict layer) ĐÃ TRIỂN KHAI**
> — `agentics/evolution/gepa.py` + CLI `gepa-evolve`; `kg_edges.trusted` +
> `kg_neighborhood`/`kg_contradictions` + CLI `kg-query`/`kg-contradictions`;
> propose-time KG context trong `_gate_and_register`. Chi tiết: EVOLUTION.md §6,
> MEMORY.md (GraphRAG verdict layer).
> Còn lại của TOP 5: #2 Router đã có (`agentics/llm/local_router.py` +
> `allowed_for`); #3 MCP chưa làm — chuyển thành mục #3 của wave-3; #5
> Verifier mới landed nửa calibration (`agentics/verify/calibration.py` —
> floor confidence), predictor cho panel rounds chuyển thành mục #4 của
> wave-3 (`docs/ADVANCED_TECH_ROADMAP_WAVE3.md`).
>
> **Audit 2026-11-14** (kiểm chứng độc lập trước khi triển khai tiếp):
> trích dẫn học thuật đã tra web xác nhận THẬT — GEPA (arXiv:2507.19457),
> RouteLLM/FrugalGPT, survey routing arXiv:2603.04445 (TMLR 2026),
> AgentV-RL arXiv:2604.16004 (ACL 2026), MemGraphRAG arXiv:2606.00610
> (KDD 2026), khảo sát Stacklok 09/2026; hai số chưa đối chiếu được chính
> xác đã đánh dấu trong thân bài ("ICLR 2026" của GEPA, "~40%" Stacklok).
> Con trỏ `polices.py` là file không tồn tại — fallback thật là
> keyword-overlap ranking trong `shared_memory` (comment
> `vector_store.py` đã sửa). Suite hiện tại: xem badge README.
> Nguyên tắc của owner: **không đụng vào core** (graph/orchestrator/P16 đã bền
> qua bộ test đầy đủ, xem badge README) — mọi thứ dưới đây là **lớp add-on** gắn vào các điểm móc sẵn
> có (ledger `llm_usage`, `role_prompts`, `findings`/`evidence`/`hypotheses`,
> `tool_registry`, pool token). Mỗi mục ghi rõ: cái gì, vì sao hợp lab này,
> kế hoạch gắn vào đâu, và ranh giới trung thực.

## Bảng so sánh nhanh

| # | Công nghệ | Tác động | Công sức | Gắn vào | Thứ tự gợi ý |
|---|---|---|---|---|---|
| 1 | GEPA — tối ưu prompt phản chiếu | Chất lượng đầu ra ↑ trực tiếp | Trung bình | `evolution/` + `role_prompts` (P11) | **Làm đầu tiên** |
| 2 | Router/cascade đa mô hình | Chi phí ↓ 60–80% tiềm năng | Trung bình | `llm/pool.py` + ledger | Song song #1 |
| 3 | MCP — chuẩn hoá lớp tool | Mở khoá cả hệ sinh thái tool | Trung bình–lớn | `research/free_tools.py` + `tool_registry` | Sau #1 |
| 4 | GraphRAG-lite — đồ thị tri thức | Multi-hop, bắt mâu thuẫn | Trung bình | bảng mới `kg_edges` + chroma cũ | Sau #2 |
| 5 | Verifier học được từ ledger (RLVR-style) | Gate rủi ro trước panel | Lớn dần | `verify/` + traces/panels | Dữ liệu đang gom |

---

## 1. GEPA — tối ưu prompt bằng phản chiếu (nâng cấp P11)

**Cái gì.** GEPA (Genetic-Pareto, Agrawal et al. 2025 — arXiv:2507.19457,
đã kiểm chứng có thật; dòng "trình bày ICLR 2026" chưa đối chiếu được
riêng lẻ) là optimizer **không gradient**: chạy pipeline, đọc trace, dùng
chính LLM phản chiếu "vì sao thất bại", tiến hoá prompt theo front Pareto
(chất lượng/chi phí). Paper cho thấy vượt RL trên nhiều task agent; đã có
sẵn trong DSPy (`dspy.GEPA`) và repo độc lập `gepa-ai/gepa`.

**Vì sao hợp lab.** Lab đã tự làm việc này bằng tay: P11 `role_prompts`
(`store_role_prompt` / `bump_role_prompt` với decay failure, supersede khi
hỏng 2 lần) + meta-review Co-Scientist chưng cất thắng/thua. Đó đúng là "reflective
evolution" thủ công — GEPA thay heuristic (decay ±1, ngưỡng 2) bằng tối ưu có
nghiệm thức, và dữ liệu feedback đã nằm sẵn trong `traces`, `panel_votes`,
`metareview`.

**Kế hoạch.**
1. Thêm `agentics/evolution/gepa_driver.py`: export trace (role, category,
   attempted, failure, outcome) thành dataset reflection; gọi GEPA để sinh
   mission candidate mới cho `(role, category)` đang superseded.
2. Giữ nguyên lifecycle P11: candidate vẫn đi qua `store_role_prompt`
   (status=active), vẫn bị `bump_role_prompt` phạt/supersede — GEPA chỉ là
   **generator**, lab vẫn là **judge**.
3. Đo A/B qua `swarm-eval` G4 `role_prompt_success_rate` (đã có % sẵn).

**Ranh giới trung thực.** Không tối ưu prompt của P16 verifier (tiêu chí
pre-registered đóng băng); không chạy GEPA trong chu kỳ thường (tốn call —
chạy định kỳ offline như meta-review).

## 2. Router/cascade đa mô hình — chi phí theo mục đích

**Cái gì.** Dynamic model routing & cascading: câu khó → model frontier, câu
dễ → model rẻ; RouteLLM (Ong et al. 2025), FrugalGPT, xRouter (router RL
tool-calling). Tổng quan 2026 (arXiv 2603.04445) ghi nhận mức giảm chi phí
60–80% với loss chất lượng tối thiểu.

**Vì sao hợp lab.** Pool đã là **đa mô hình thật** (ChatGPT / Perplexity /
Qwen + dual-judge), và ledger `llm_usage` đã ghi `purpose` (research / plan /
implement / verify / review) + `ok` + `duration_s` — đúng đủ để học router
theo purpose mà không cần thêm telemetry. swarm-eval G5 đã đo
`llm_calls_per_merged` / `llm_seconds_per_merged` → KPI của router có sẵn.
Số thật hiện tại: 24 calls, top purpose "research" chiếm 42%, mean 79.6s/call.

**Kế hoạch.**
1. `agentics/llm/router.py`: bảng ánh xạ `purpose → tier`
   (verify/review → model nhanh; research/tournament → frontier), fallback
   cascade khi model tier thấp self-check thấy thiếu (câu trả lời quá ngắn,
   không citation).
2. Giai đoạn 2: bandit đơn giản (epsilon-greedy) chọn model theo lịch sử
   ok-rate × thời lượng của từng purpose trong ledger — không cần GPU.
3. Gắn cờ `configs/lab.yaml` (`llm.routing: static|bandit|off`), mặc định off.

**Ranh giới.** P16 verify + panel P19 **không bao giờ** hạ tier (phán quyết
phải là model mạnh nhất có sẵn); router chỉ tối ưu chi phí, không đổi định nghĩa
kết quả.

## 3. MCP — chuẩn hoá lớp tool, mở khoá hệ sinh thái

**Cái gì.** Model Context Protocol đã thành **chuẩn de facto** kết nối agent–tool
(khảo sát Stacklok "State of MCP in Software 2026" / "State of AI
Architecture 2026" là có thật — 100 lãnh đạo kỹ thuật và 520 lãnh đạo
AI respectively, 09/2026; con số "~40% doanh nghiệp" ban đầu chưa đối chiếu
được chính xác với báo cáo, đã làm mềm theo audit 2026-11-14), cộng đồng
phát hành hàng nghìn MCP server:
GitHub, arXiv, database, filesystem, browser…

**Vì sao hợp lab.** Lab đang tự bọc từng tool: `research/free_tools.py`
(arXiv/OpenAlex/Crossref/Semantic Scholar/Europe PMC/PyPI…), `tool_registry`
SQL tự quản, `github_ops` tự gọi REST. Mỗi tool tự bọc = mỗi tool tự viết
retry/rate-limit/parse. MCP thay lớp glue đó: một client MCP → khai báo
capability vào `tool_registry`, agent dùng như tool thường; xoá tool = xoá
cấu hình, không phải xoá code.

**Kế hoạch.**
1. `agentics/tools/mcp_client.py`: client stdio/SSE gọn (không cần SDK nặng);
   server MCP nào khả dụng qua `configs/lab.yaml` (`tools.mcp_servers:`).
2. Bridge: khi MCP server lên, `register_tool()` các capability vào
   `tool_registry` (P21 dùng lại như cũ); `free_tools` giữ làm **fallback**
   keyless khi MCP chết.
3. Lợi ích kế tiếp: một MCP server tự viết (gồm `evidence`, `hypotheses`,
   `swarm-eval`) = lab mở API chuẩn cho bất kỳ agent ngoài nào — đường tới A2A.

**Ranh giới.** Tool qua MCP vẫn phải qua cổng P20 và sandbox như tool nội;
không tin schema server lạ — capability phải được owner duyệt trong config.

## 4. GraphRAG-lite — đồ thị tri thức trên bộ nhớ lab

**Cái gì.** Consensus 2026: memory agent tốt nhất là **lai** — vector cho
semantic recall rộng, knowledge graph cho multi-hop + toàn cục (MemGraphRAG,
arXiv 2606.00610; các hệ như Cognee). Lab hiện chỉ có chroma (vector phẳng
trên `findings`) — mất quan hệ "hỗ trợ / bác bỏ / mâu thuẫn / kế thừa".

**Vì sao hợp lab.** Lab đã có sẵn các node tự nhiên của đồ thị: `findings`,
`hypotheses` (claim_hash, genealogy parents trong evolution), `evidence`
(verdict confirmed/refuted), `lessons`. Thiếu duy nhất **cạnh**. Đồ thị nhẹ
trên SQLite (bảng `kg_edges`: src_type/src_id → rel → dst_type/dst_id,
không cần Neo4j) cho: (a) multi-hop retrieval trước khi propose hypothesis —
đỡ đề xuất lại claim đã bị refute; (b) **phát hiện mâu thuẫn**: finding mới
chống lại evidence refuted cũ → cắn câu hỏi; (c) ngữ liệu đa hop cho
meta-review.

**Kế hoạch.**
1. `agentics/memory/kg.py`: bảng `kg_edges` + writer hook vào các sự kiện sẵn
   có (`hypothesis_verified`, `evidence`, finding add) — ghi cạnh cùng lúc
   ghi bảng gốc, không migration dữ liệu cũ.
2. `kg_context(claim)`: 1–2 hop neighbourhood → chèn vào prompt propose
   (điểm móc: `settled_hypothesis_for_claim` hiện chỉ match hash — đồ thị bổ
   sung match ngữ nghĩa).
3. CLI `agentics kg-query <term>` in neighbourhood để owner soi đồ thị.

**Ranh giới.** Không để LLM tự viết cạnh tùy ý — cạnh chỉ sinh từ sự kiện có
verdict (P16/P24/Elo); vector chroma vẫn giữ vai trò recall rộng.

## 5. Verifier học được từ chính ledger của lab (RLVR-style)

**Cái gì.** Xu hướng 2026: RLVR — reward từ verifier kiểm chứng được; nhỏ
nhẹ tinh (`AgentV-RL`, arXiv 2604.16004) cho thấy verifier agentic + reward
model quy mô nhỏ hoạt động tốt. Không cần RL đầy đủ: lab có thể **distill
chính dữ liệu phán quyết của mình** thành một gate học được.

**Vì sao hợp lab.** Ledger đang gom đúng cặp (đầu vào → phán quyết máy):
traces (attempted/failure → outcome), panels (nội dung → vote), evidence
(exit_code/duration → verdict). Đây là bộ supervised label **trung thực** vì
phán quyết là của code chạy thật (P16), không phải opinion. Ứng dụng đầu
tiên khả thi không GPU: **risk gate** — model logistic nhỏ (features: số
attempt, failure signature, domain, độ dài protocol…) dự đoán xác suất MR
merge; thấp → buộc rework TRƯỚC khi mở panel (tiết kiệm rounds — G2
`mean_panel_rounds` hôm nay mất công đo đấy).

**Kế hoạch.**
1. `agentics/verify/gate.py`: train offline từ `swarm_eval_snapshots` +
   traces/panels (sklearn logistic / GBM nhỏ); export trọng số vào DB.
2. Chế độ shadow (chỉ ghi dự đoán, không chặn) → chạy ≥2 tuần → đối chiếu
   AUC → mới bật chế độ chặn mềm (yêu cầu 1 rework, không cấm).
3. Nâng cấp sau: dùng gate làm **prior** cho Contract-Net award
   (`coordination/allocation.py` đang dùng lịch sử merge thật — gate tinh
   hoá chính tín hiệu đó).

**Ranh giới.** Gate không bao giờ kết liễu claim hay thay panel — chỉ sắp
thứ tự/đẩy rework; phán quyết cuối vẫn 100% P16 + P19. Bảo hiểm chống
self-confirm: gate được train lại chỉ từ dữ liệu đã chốt, không tự nuốt dự
đoán của chính nó.

---

## Điều gì cố tình KHÔNG đưa vào top 5

- **A2A protocol đầy đủ**: lab hiện 1 máy, 1 key — chưa có bài toán interop
  đa tổ chức; MCP mục #3 là nửa đường (server MCP tự viết sau này chính là
  endpoint A2A).
- **Context compression (LLMLingua kiểu)**: handoff ngắn, journal đã trim;
  đợi metric G3 `trajectory_length_nodes` tăng bất thường mới đáng.
- **Tree-search orchestration (LATS/MCTS đầy đủ)**: giải đấu Elo + evolution
  đã là search có hướng; LATS sẽ trùng cơ chế và tốn call gấp nhiều lần.

## Thứ tự triển khai gợi ý

1. **Tuần 1**: GEPA driver (điểm móc P11 đã sẵn) + router static theo purpose.
2. **Tuần 2**: MCP client + bridge `tool_registry`; bật shadow router.
3. **Tuần 3–4**: `kg_edges` + writer hooks + `kg_context`; gate shadow train
   lần đầu từ snapshot.
4. **Đo lường mỗi bước**: chính `agentics swarm-eval --percent` là scoreboard
   — mục tiêu: G5 `llm_seconds_per_merged` ↓, G2 `mean_panel_rounds` ↓, G4
   `role_prompt_success_rate` ↑, không phá G1 `task_completion_rate`.
