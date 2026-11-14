# Wave 3 — đối chiếu & hòa hợp ba đề xuất TOP 5

Owner yêu cầu (2026-11-14): *"nên ứng dụng thêm các công nghệ tiên tiến
khác vào để cải thiện, thay vì phải cải tiến phần core"*. Ba phiên làm
việc song song đã trả lời bằng ba bộ TOP 5:

* `docs/ADVANCED_STACK_WAVE3.md` (bộ A — **giữ nguyên làm tài liệu gốc**):
  best-of-N, Z3 formal track, hermetic replay, federation, auto-ablation.
* Đề xuất ban đầu của phiên này (bộ B): statistical experiment design,
  truth maintenance, literature watchdog, learned prior, paper assembler.
* `docs/ADVANCED_TECH_ROADMAP_WAVE3.md` (bộ C — **giữ nguyên**): semantic
  cache, hybrid retrieval + cross-encoder rerank, MCP client, RLVR-lite
  risk gate, fulltext ingestion + citation grounding.

Tài liệu này **không thay thế** hai tài liệu gốc — nó đối chiếu 15 ứng
viên, chỉ rõ chỗ giao nhau, và đề xuất một TOP 5 hợp nhất (có truy nguyên
từng ý) để owner chọn. Nguyên tắc chung giữ nguyên: không đụng core
(P1–P24, graph/orchestrator), tín hiệu chọn lọc chỉ là exit-code / dữ liệu
SQL thật, LLM local trước — cookie sau, mọi consumer degrade khi flag off.

---

## 1. Đối chiếu: chỗ giao nhau giữa ba bộ

| Ý | Bộ | Giao nhau với | Kết luận đối chiếu |
|---|---|---|---|
| Two-arm + effect size | B | **auto-ablation** (A#5) | TRÙNG bản chất — ablation chính là arm baseline vs treatment. Gộp: giữ tên `auto-ablation` của A, nhưng phải xây **nền thống kê** trước (mục 3) — delta không có CI là delta nhiễu |
| N seed + mean/std/CI | B | hermetic replay (A#3) | BỔ TÚC: fingerprint định danh **môi trường** mỗi lần chạy; seed-sweep định lượng **phương sai** kết quả. Hai lớp cùng làm verdict trung thực |
| Learned success prior | B | **RLVR-lite risk gate** (C#4) | HỌ HÀNG — cùng học logistic nhỏ từ ledger, khác consumer: prior cho frontier PUCT (B) vs gate rủi ro cho MR/panel (C). GỘP SUBSTRATE: một `evolution/logreg.py` nhỏ, hai đầu ra |
| Literature watchdog | B | fulltext grounding (C#5) | BỔ TÚC theo pha: watchdog **phát hiện** paper mới (API free, nhẹ); fulltext **đọc sâu** paper đã trích (nặng, 3 pha). Thứ tự: watchdog trước |
| Semantic cache | C | — | độc lập — infra tiết kiệm cookie, KHÔNG cache purpose phán quyết (đúng blacklist triết lý `local_router.allowed_for()`) |
| Hybrid rerank | C | — | độc lập — nâng retrieval cho mọi consumer hiện có |
| MCP client | C | — | độc lập — mở khoá tool qua khai báo config |
| Hermetic replay | A | — | độc lập, ưu tiên cao (nhỏ effort, bảo vệ ledger) |
| Best-of-N machine gate | A | — | độc lập, nâng chất lượng mọi đường code, free trên local edge |
| Z3 formal track | A | — | độc lập, payoff nhận thức cao nhất, chi phí thiết kế cao nhất |
| Federation bundles | A | — | độc lập, cần máy thứ hai ghép cặp |
| Paper assembler | B | — | độc lập — bề mặt owner, lắp báo cáo từ SQL |

Không có cặp nào mâu thuẫn thiết kế; đúng một cặp trùng bản chất (B↔A#5)
và một cặp họ hàng cần gộp substrate (B↔C#4).

## 2. TOP 5 hợp nhất — đề xuất cuối

Xếp theo (mức bảo vệ × đảo effort): ưu tiên những ý nhỏ effort mà bảo vệ
hoặc giải phóng phần lớn những gì lab đã xây.

| # | Công nghệ | Truy nguyên | Chi phí LLM | Effort |
|---|-----------|-------------|-------------|--------|
| 1 | Hermetic replay — env fingerprint cho replication | A#3 | 0 | S |
| 2 | Truth maintenance — claim sập → phụ thuộc `stale` | B | 0 | S–M |
| 3 | Semantic cache có ranh giới trung thực | C#1 | 0 (trả từ cache) | S |
| 4 | Statistics substrate — seed sweep + CI cho verdict | B (phần bổ túc cho A#5) | 0 | M |
| 5 | Best-of-N với machine-gate chọn người thắng | A#1 | local free / cookie có budget | M |

**Vì sao những ý còn lại vào bench:** auto-ablation (A#5) phụ thuộc #4
và tiêu protocol cost gấp đôi — bench ngay sau #4; hybrid rerank (C#2)
nâng chất lượng nhưng mọi consumer vẫn chạy được hôm nay; literature
watchdog (B) là năng lực mới trong khi research loop đã truy vấn paper
on-demand; learned prior + RLVR-lite gate (B + C#4 gộp) cần ledger dày;
fulltext grounding (C#5) lớn effort 3 pha; MCP (C#3) mở hệ sinh thái nhưng
chưa có tool cụ thể khan hiếm; federation (A#4) cần hermetic replay trước;
paper assembler (B) là thu hoạch cuối; Z3 (A#2) pilot mỏng làm sau cùng.

**Bench theo thứ tự kích hoạt:** hybrid rerank (song song #3) → literature
watchdog → auto-ablation (sau #4) → shared logreg substrate (learned prior
+ risk gate, sau khi ledger ≥50 settle) → fulltext grounding (3 pha) →
MCP client → federation → paper assembler → Z3 pilot.

## 3. Thiết kế hợp nhất cho các ý của phiên này (tóm tắt thao tác)

* **Truth maintenance** — `agentics/verify/beliefs.py`: khi replay/replication
  hạ một claim, BFS theo hướng phụ thuộc trên `kg_edges` (`uses`/`improves`)
  + quét evidence tham chiếu → cột `stale_since`, priority cộng vào
  frontier, alert once-only, prompt render ghi `[STALE — phụ thuộc claim X]`.
  Degrade: kg rỗng → khớp claim-id trong protocol source đã lưu (P23).
* **Statistics substrate** — `agentics/verify/stats.py` (thuần Python, không
  thêm dependency): chạy N seed (mặc định 5, protocol nhận tham số seed),
  mean/std/CI 95% bootstrap, tăng mẫu tuần tự khi |d| lớn nhưng CI chắn 0;
  ledger thêm `n_runs, mean, std, ci_lo, ci_hi`; settle note thêm
  `[n=5 mean=… ±…]`. Degrade: protocol không nhận seed → 1 lần như hiện tại,
  note `unseeded`. Auto-ablation (bench) ăn trực tiếp các cột này.
* **Literature watchdog** — `agentics/research/watchdog.py`: mỗi N cycle
  supervisor (mặc định 12) query arXiv/S2/OpenAlex bằng top-K
  `kg_entities`, lọc PEM gate, cái mới liên quan → `world_note
  kind=literature_alert` + journal + re-check novelty cho hypothesis
  `proposed` trùng mechanism. Degrade: mất mạng → sweep no-op có journal.
* **Shared logreg substrate (B#4 + C#4 gộp)** — `agentics/evolution/
  logreg.py`: logistic SGD thuần Python ~100 dòng; đầu ra 1: `learned_prior`
  cho PUCT/EIG; đầu ra 2: gate rủi ro cho MR/panel. Retrain mỗi 20 settle
  khi ≥50 dòng; AUC < 0.55 trên holdout → tự tắt (tránh tự lừa).
* **Paper assembler** — `agentics/report/` + `agentics paper --window 7d`:
  template renderer đọc verbatim từ SQL (verified_knowledge, evidence kèm
  CI, Elo, replication, citations); văn xuôi trên local router, validate
  regex mọi con số. Degrade: local off → skeleton thuần bảng số liệu.

## 4. Thứ tự triển khai hợp nhất (thay lời khuyên riêng của từng tài liệu)

1 → 2 → 3 → 4 → 5 theo bảng mục 2, hybrid rerank chạy song song #3;
chèn auto-ablation ngay sau #4 khi owner duyệt. Mỗi bước: một module mới +
test offline + journal event + KPI query bổ sung vào `docs/ADVANCED_STACK.md`
(mục wave 3) — đúng khuôn mẫu wave 1/2.
