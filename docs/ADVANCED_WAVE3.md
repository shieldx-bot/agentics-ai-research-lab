# Wave 3 — đối chiếu & hòa hợp hai đề xuất TOP 5

Owner yêu cầu (2026-11-14): *"nên ứng dụng thêm các công nghệ tiên tiến
khác vào để cải thiện, thay vì phải cải tiến phần core"*. Hai phiên làm
việc song song đã trả lời bằng hai bộ TOP 5:

* `docs/ADVANCED_STACK_WAVE3.md` (đề xuất của người đi trước — **giữ nguyên
  làm tài liệu gốc**): best-of-N, Z3 formal track, hermetic replay,
  federation, auto-ablation.
* Đề xuất ban đầu của phiên này: statistical experiment design, truth
  maintenance, literature watchdog, learned prior, paper assembler.

Tài liệu này **không thay thế** tài liệu gốc — nó đối chiếu 10 ứng viên,
chỉ rõ chỗ giao nhau, và đề xuất một TOP 5 hợp nhất (có truy nguyên từng
ý) để owner chọn. Nguyên tắc chung giữ nguyên: không đụng core (P1–P24,
graph), tín hiệu chọn lọc chỉ là exit-code / dữ liệu SQL thật, LLM local
trước — cookie sau, mọi consumer degrade khi flag off.

---

## 1. Đối chiếu: chỗ giao nhau giữa hai bộ

| Ý | Đề xuất của ai | Giao nhau với | Kết luận đối chiếu |
|---|---|---|---|
| Two-arm randomized + effect size | phiên này | **auto-ablation** (người đi trước, #5) | TRÙNG bản chất — ablation chính là arm baseline vs treatment. Gộp: giữ tên `auto-ablation` của tài liệu gốc, nhưng **phải xây nền thống kê trước** (mục 3 bên dưới) — delta không có CI là delta nhiễu |
| N seed + mean/std/CI + sequential sampling | phiên này | hermetic replay (#3 của tài liệu gốc) | BỔ TÚC, không trùng: replay fingerprints định danh **môi trường** của mỗi lần chạy; seed-sweep định lượng **phương sai** của kết quả. Hai lớp cùng làm verdict trung thực |
| Hermetic replay | người đi trước | — | độc lập, ưu tiên cao (nhỏ effort, bảo vệ ledger) |
| Best-of-N + machine gate | người đi trước | — | độc lập, nâng chất lượng mọi đường code, free trên local edge |
| Z3 formal track | người đi trước | — | độc lập, payoff nhận thức cao nhất nhưng chi phí thiết kế cao nhất |
| Federation bundles | người đi trước | — | độc lập, cần máy thứ hai ghép cặp |
| Truth maintenance (stale propagation) | phiên này | — | độc lập — **không có gì tương đương trong tài liệu gốc** |
| Literature watchdog | phiên này | — | độc lập — lab hiện chỉ "thấy" arXiv khi research prompt yêu cầu |
| Learned success prior | phiên này | — | độc lập — PUCT P hiện là rank prior, chưa học từ ledger |
| Paper assembler | phiên này | — | độc lập — bề mặt owner, lắp báo cáo từ SQL |

Không có ý nào của hai bộ mâu thuẫn nhau về thiết kế; duy nhất một cặp
trùng bản chất (đã gộp ở trên).

## 2. TOP 5 hợp nhất — đề xuất cuối

Xếp theo đòn bẩy / effort / mức độ bảo vệ những gì lab đã xây:

| # | Công nghệ | Truy nguyên | Chi phí LLM | Effort |
|---|-----------|-------------|-------------|--------|
| 1 | Hermetic replay — environment fingerprint cho replication | tài liệu gốc #3 | 0 | S |
| 2 | Truth maintenance — claim sập → phụ thuộc thành `stale` | phiên này | 0 | S–M |
| 3 | Statistics substrate — seed sweep + CI cho mọi verdict | phiên này (phần bổ túc) | 0 | M |
| 4 | Best-of-N với machine-gate chọn người thắng | tài liệu gốc #1 | local free / cookie có budget | M |
| 5 | Literature watchdog — đầu dò tài liệu mới theo kg_entities | phiên này | 0 (API free) | S–M |

**Vì sao auto-ablation (#5 của tài liệu gốc) không nằm trong top 5 hợp
nhân:** nó phụ thuộc #3 statistics substrate (delta cần CI) và tiêu protocol
cost gấp đôi cho claim supported — đưa lên **bench ngay sau #3** là đúng
thứ tự phụ thuộc, không phải từ chối.

**Bench (lượt sau, theo thứ tự kích hoạt):** auto-ablation (gốc #5, sau
statistics substrate) → learned success prior (cần ledger dày, AUC guard
<0.55 thì tự tắt) → federation (gốc #4, sau hermetic replay) → paper
assembler (thu hoạch cuối, báo cáo sẵn có CI + stale marker) → Z3 pilot
(gốc #2, mỏng nhất: `kind=invariant` trên cận số học).

## 3. Thiết kế hợp nhất cho ba ý của phiên này (tóm tắt thao tác)

* **Truth maintenance** — `agentics/verify/beliefs.py`: khi replay/replication
  hạ một claim, BFS theo hướng phụ thuộc trên `kg_edges` (`uses`/`improves`)
  + quét evidence tham chiếu → cột `stale_since`, priority cộng vào
  frontier, alert once-only, prompt render ghi `[STALE — phụ thuộc claim X]`.
  Degrade: kg rỗng → khớp claim-id trong protocol source đã lưu (P23).
* **Statistics substrate** — `agentics/verify/stats.py` (thuần Python, không
  thêm dependency): chạy N seed (mặc định 5, protocol nhận tham số seed),
  mean/std/CI 95% bootstrap, quy tắc tăng mẫu tuần tự khi |d| lớn nhưng CI
  chắn 0; ledger thêm `n_runs, mean, std, ci_lo, ci_hi`; settle note thêm
  `[n=5 mean=… ±…]`. Degrade: protocol không nhận seed → 1 lần chạy như
  hiện tại, note `unseeded`. Auto-ablation (bench) ăn trực tiếp các cột này.
* **Literature watchdog** — `agentics/research/watchdog.py`: mỗi N cycle
  supervisor (mặc định 12) query arXiv/S2/OpenAlex bằng top-K `kg_entities`,
  lọc PEM gate, cái mới liên quan → `world_note kind=literature_alert` +
  journal + re-check novelty cho hypothesis `proposed` trùng mechanism.
  Degrade: mất mạng → sweep no-op có journal.

## 4. Thứ tự triển khai hợp nhất (thay thế lời khuyên riêng của từng tài liệu)

1 → 2 → 3 → 4 → 5 theo bảng mục 2; chèn auto-ablation ngay sau #3 khi
owner duyệt. Mỗi bước: một module mới + test offline + journal event + KPI
query vào `docs/ADVANCED_STACK.md` (mục wave 3) — đúng khuôn mẫu wave 1/2.
