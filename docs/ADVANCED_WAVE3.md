# ADVANCED WAVE 3 — Đề xuất TOP 5 công nghệ tiên tiến tiếp theo

Trả lời yêu cầu của owner (2026-11-14): *"nên ứng dụng thêm các công nghệ
tiên tiến khác vào để cải thiện, thay vì phải cải tiến phần core"*.

Nguyên tắc chọn: **(1)** không đụng vào core graph / core policy (P1–P24) —
mỗi đề xuất là một module mớicomposed từ cơ chế sẵn có; **(2)** tín hiệu
chọn lọc chỉ dùng EXIT-CODE / dữ liệu SQL thật (đúng luật chung của wave 2);
**(3)** LLM gọi ít nhất có thể — ưu tiên 0-cookie, chỉ dùng local router;
**(4)** mọi consumer đều degrade về hành vi hiện tại khi feature off.

Đã có sẵn (không đề xuất lại): wave 1 (PUCT, MAP-Elites, local router,
GraphRAG-lite, Hypothesis Arena), wave 2 (skill library, prompt evolution,
EIG frontier, calibration, reflection tree), P16–P24 (evidence runner,
replay, replication, panel, audit, retention), world model, QD ideation,
Co-Scientist tournament, GEA traces, A2X coordination, swarm-eval.

---

## TOP 5 — xếp theo đòn bẩy (leverage) giảm dần

| # | Công nghệ | Chi phí LLM | Chạm core? | Compose với |
|---|-----------|-------------|-----------|-------------|
| 1 | Thống kê thí nghiệm: two-arm randomized + effect size | **0** | Không | P16 runner, P23 replay, calibration |
| 2 | Truth maintenance — lan truyền tín nhiệm trên đồ thị claim | **0** | Không | P23 replay, P24 replication, kg_edges |
| 3 | Literature watchdog — đầu dò tài liệu mới theo kg_entities | **0** (API free) | Không | free_tools, PEM gate, world model |
| 4 | Learned success prior — meta-model học từ settle ledger | **0** (CPU) | Không | PUCT P, EIG, frontier |
| 5 | Paper assembler — lắp báo cáo nghiên cứu từ bằng chứng SQL | local only | Không | verified_knowledge, evidence, Elo, citations |

---

## 1. Statistical experiment design — phán quyết thành thống kê thật

**Vấn đề còn lại.** P16 phán quyết bằng exit-code + reproduction gate 2 lần
chạy; điểm số là **point estimate** — một protocol may mắn chạy đúng 2 lần
vẫn `supported` dù phương sai lớn. Mọi consumer phía dưới (PUCT Q, Elo,
calibration Wilson) đều ăn con số đó.

**Công nghệ.** Lớp thiết kế thí nghiệm chuẩn: mỗi claim được chạy **N seed**
(N khả chỉnh, mặc định 5); report mean / std / CI 95% (bootstrap hoặc
Welch); với claim dạng cải tiến, runner tự chia **2 arm** — `baseline`
(protocol hiện có từ skill library) và `treatment` (protocol đề xuất) — và
ra effect size Cohen's d + p-value Welch's t-test. Quy tắc tăng mẫu tuần
tự: nếu |d| lớn nhưng CI chắn 0 → tăng thêm N seed thay vì kết luận.

**Điểm chạm.** Module mới `agentics/verify/stats.py` (thuần Python, không
thêm dependency); `verify/experiment.py` gọi vào sau `run_experiment`;
ledger thêm cột `n_runs, mean, std, ci_lo, ci_hi, effect_d, p_value`;
settle note thêm dòng `[n=5 mean=… ±…, d=…, p=…]`.

**KPI.** Tỉ lệ verdict đảo chiều sau khi có CI (kỳ vọng: 5–15% claim
`supported` cũ sẽ tụt thành `inconclusive` — đó là sự trung thực mới);
calibration floor (wave 2) phải **tăng**, không giảm.

**Degrade.** Không seed được (protocol không nhận tham số) → chạy 1 lần như
hiện tại, note ghi `unseeded`.

## 2. Truth maintenance — khi một claim sập, mọi claim phụ thuộc phải biết

**Vấn đề còn lại.** P23 replay + P24 replication phát hiện mâu thuẫn và
alert cho **chính claim đó**. Nhưng knowledge phụ thuộc (claim B "dựa trên"
claim A qua `uses`/`improves` trong `kg_edges`, hoặc evidence trích
verified_knowledge của A) vẫn đứng nguyên trong prompt của mọi agent —
GraphRAG đang lan truyền thông tin cũ như thể vẫn đúng.

**Công nghệ.** Truth-maintenance / belief revision kiểu TMS: khi claim A bị
hạ (refuted / replay contradiction / replication fail), BFS trên
`kg_edges` theo hướng phụ thuộc + quét evidence có tham chiếu A → đánh dấu
các claim B phụ thuộc thành `stale` (cột mới `stale_since`), **đẩy priority
vào frontier** (cộng vào score của PUCT/EIG) và alert đúng 1 lần (giống
luật once-only của calibration). Prompt render ghi rõ `[STALE — phụ thuộc
claim X vừa bị hạ]`.

**Điểm chạm.** `agentics/verify/beliefs.py` (thuần graph walk trên SQLite);
gọi từ supervisor sweep sau replay; không thêm node graph nào.

**KPI.** Số claim `stale` được re-verify trong vòng 2 giờ; tỉ lệ stale còn
tồn khi hết phiên (mục tiêu: 0).

**Degrade.** Kg rỗng (local router off) → fallback: quét phụ thuộc bằng
khớp claim-id trong protocol source đã lưu (P23 precondition đã lưu
protocol).

## 3. Literature watchdog — lab biết "thế giới đã đổi"

**Vấn đề còn lại.** Lab chỉ tìm tài liệu khi research prompt yêu cầu; giữa
các đợt nó **mù** với arXiv. Một paper mới đúng cơ chế nó đang đào có thể
làm vô dụng cả kiến trúc giả thuyết đang chạy — và lab chỉ phát hiện khi
owner đọc tiếp cho nó.

**Công nghệ.** Đầu dò định kỳ (mỗi N cycle supervisor, mặc định 12): query
arXiv API + Semantic Scholar + OpenAlex bằng **top-K `kg_entities`** (không
tốn cookie — đúng stack free_tools); lọc bằng PEM gate hiện có (chống
near-dup); cái mới và liên quan → `world_note` kind=`literature_alert` +
journal `literature_alert` + kích hoạt lại novelty-check cho các hypothesis
`proposed` trùng mechanism. Source-of-truth cho "đã thấy" là bảng
citations hiện có.

**Điểm chạm.** `agentics/research/watchdog.py`; hook vào supervisor sweep
(cùng chỗ replay sweep, không thêm node).

**KPI.** Số alert/tuần; tỉ lệ alert dẫn tới hypothesis bị sửa/hủy (alert
vô dụng liên tục = ngưỡng quá lỏng).

**Degrade.** Mất mạng → sweep no-op có journal, không crash.

## 4. Learned success prior — frontier học từ chính ledger của nó

**Vấn đề còn lại.** P của PUCT hiện là **rank prior** (thứ hạng trong
discovery buffer) — không học được rằng, ví dụ, các claim nhóm mechanism
`X` trên metric `Y` trong lab này gần như luôn `refuted`.

**Công nghệ.** Meta-model cực nhỏ (logistic regression SGD thuần Python,
~100 dòng, không thêm dependency) train trên settle ledger: features từ
signature quintuple (mechanism category, angle, metric type, depth, độ dài
protocol, has_dataset…), label = verdict thật. Retrain mỗi 20 settle khi
có ≥50 dòng; output `P(supported)` lưu vào cột `learned_prior`, nhân vào
P của PUCT và vào EIG cost. Mỗi ngày in AUC trên 20% holdout — nếu AUC
< 0.55 thì prior bị tắt tự động (tránh tự lừa).

**Điểm chạm.** `agentics/evolution/learned_prior.py`; đọc/ghi qua
SharedMemory; không đụng công thức PUCT (chỉ thay nguồn P).

**KPI.** AUC tuần; frontier_pick `learned_prior` cao nhưng verdict thấp →
cảnh báo tự động.

**Degrade.** <50 dòng hoặc AUC kém → rank prior như hôm nay.

## 5. Paper assembler — owner cầm được tài liệu nghiên cứu

**Vấn đề còn lại.** Mọi kết quả nằm rải rác trong dashboard/CLI (evidence
card, panel, Elo, replication). Owner muốn "sản phẩm nghiên cứu" thì phải
tự lắp.

**Công nghệ.** `agentics paper --window 7d`: template renderer (Jinja-free,
f-string) đọc **verbatim từ SQL** — verified_knowledge, evidence (kèm n
runs + CI từ #1), Elo + matches, replication ledger, kg summary, citations
đầy đủ từ papers DB — ra Markdown (tùy chọn LaTeX). Phần văn xuôi
(abstract, transition) chạy trên **local router** với luật cứng: mọi con số
phải trích nguyên văn từ context SQL, không được sinh số mới (validate
bằng regex so khớp trước khi ghi).

**Điểm chạm.** `agentics/report/` + subcommand CLI + nút dashboard
"Xuất báo cáo".

**KPI.** Owner dùng báo cáo làm đầu vào họp/tiếp tục research; số claim
trong báo cáo == số claim trong DB (không lệch).

**Degrade.** Local router off → bản skeleton thuần bảng số liệu, không văn
xuôi — vẫn hữu ích.

---

## Không chọn (và vì sao)

* **Agent Workflow Memory** — trùng skill library (wave 2) về bản chất
  (recurring pattern tái sử dụng).
* **Sleep-time consolidation** — P21 retention + P23 replay + reflection
  tree đã phủ cùng vòng đời.
* **Shapley/leave-one-out credit** — A2X contract-net awards + dossier đã
  ghi nhận đóng góp theo cơ chế riêng.
* **Song song hóa orchestrator** — đụng core, đúng điều owner muốn tránh.

## Thứ tự triển khai đề xuất

1 → 2 → 3 → 4 → 5. #1 và #2 là nền trung thực cho mọi thứ phía trên;
#3 nạp nguyên liệu ngoài; #4 chỉ đáng train khi ledger đủ dày (cần #1 chạy
một thời gian); #5 thu hoạch cuối để báo cáo có CI và stale-marker của
#1/#2 ngay từ đầu.
