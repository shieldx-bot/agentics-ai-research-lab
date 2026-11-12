# EVOLUTION — áp dụng GEA vào Agentics Lab

Nguồn: Weng et al., *Group-Evolving Agents: Open-Ended Self-Improvement via Experience Sharing*
(arXiv:2602.04837). Ý chính của paper: đơn vị tiến hóa là **nhóm** agent chia sẻ *traces*
(patch đã áp, log thực thi, kết quả/failure) thay vì các nhánh cô lập.

## 1. Ánh xạ paper → code

| Paper | Lab | Code |
|---|---|---|
| Evolutionary traces (Alg. 2: patches, logs, outcome) | Bảng `traces`: làm gì, fail ở đâu, kết quả cuối (merged/rejected/failed/yield) | `memory/shared_memory.py`, `node_reflect` |
| Shared pool đưa vào Reflection của *mọi* agent | `trace_context()` chèn failure + cách làm đã merge của agent khác vào prompt `plan` | `evolution/traces.py`, `node_plan` |
| Performance–Novelty (Alg. 1) | Cài đặt đúng công thức: `score = α·√nov`, nov = khoảng cách cosine trung bình tới M láng giềng | `evolution/scoring.py` (lớp 1) |
| Chọn parent theo score | **Chuyển thể**: xếp hạng issue cho từng agent theo lịch sử merge theo loại issue | `scoring.rank_issues`, `node_claim` |
| Tools chết trong nhánh cô lập (Fig. 4) | `tool_registry`: tool chỉ hiện sau khi PR **merge**; plan phải xem registry | `node_plan`, `node_open_pr`, `MergeQueue` |
| Đánh giá nhiều tầng (sanity → mini → full) | Tầng 1: `py_compile` + `pytest -x` trước khi push. Tầng 2: test đầy đủ trên `main + PR` ngay trước merge | `_tier1`, `RepoWorkspace.verify_merge` |
| Agent tự sửa framework của chính nó | Issue `self-evolution` định kỳ, **chỉ** được sửa vùng allowlist | `node_scan`, `EVOLUTION_ALLOWED_PATHS` |

## 2. Điểm KHÁC paper (đọc trước khi tin vào kết quả)

1. **Không có benchmark cố định.** Paper có `z_i ∈ {0,1}^D` trên các task thăm dò và `α_i` đo
   trên benchmark. Lab dùng *loại issue* làm chiều (D = 8) và *PR được merge* làm "success".
   Tín hiệu thưa và nhiễu (reviewer cũng là LLM). `rank_issues` vì vậy là heuristic cân bằng
   khám phá/khai thác, **không** phải tái hiện thí nghiệm của paper.
2. **`α` được co về trung bình của agent** (Laplace) để loại issue chưa từng làm không bị phạt;
   novelty có sàn 0.25 để chuyên môn của agent không bị chấm 0. Hai tham số này do lab tự đặt.
3. **Chưa có vòng tiến hóa tự động sửa prompt.** Paper đo fitness của bản con trên benchmark rồi
   mới giữ lại. Lab chưa có fitness đó, nên issue `self-evolution` chỉ đi qua đường PR thường
   (claim → tier-1 → cross-review → tier-2 → merge) với allowlist cứng. Prompt mới **chưa**
   được chứng minh là tốt hơn — review phải dựa vào trace được trích trong issue.
4. **Quy mô khác hẳn.** Paper: K=2, ~30 vòng, Claude/GPT-o1 trên SWE-bench. Lab: LLM qua
   Perplexity. Đừng kỳ vọng con số 71.0% vs 56.7% chuyển sang đây.

## 3. Guardrail cho self-evolution (được code ép, không chỉ nhắc trong prompt)

- PR thuộc loại `evolution` chỉ được đụng: `agentics/agents/roles.py`, `agentics/evolution/`,
  `tests/`, `docs/`. Chạm file khác → agent tự **yield**, hoàn tác.
- Không bao giờ nằm trong allowlist: `policies/`, `stop.py`, `github_ops/`, `llm/`, `.github/`
  (công tắc dừng và cơ chế merge không được để agent tự sửa — xem Impact Statement của paper).
- Issue chỉ được mở khi đã có failure trace; mỗi `evolution_every_n_cycles` chu kỳ của
  Supervisor (0 = tắt), và không mở trùng.

## 4. Đo "sự tiến hóa" của lab

```bash
python -m agentics.cli metrics            # toàn thời gian
python -m agentics.cli metrics --days 7   # tuần qua
```

Cần vài chục kết quả đã phán xét (merged/rejected/failed) trước khi `merge rate` hay điểm
xếp hạng đáng tin. Đây là tỉ lệ merge theo thời gian — **không** kiểm soát độ khó của issue,
nên chỉ dùng để thấy xu hướng, không để kết luận nhân quả.

## 5. Bước tiếp theo nếu muốn tiến hóa prompt cho thật

Cần một **probe set** cố định (vd. 10–20 issue mẫu có checker xác định: plan phải parse được
JSON, file nằm đúng vùng ownership, test của repo phải xanh). Khi đó mỗi bản prompt có
`z_i` và `α_i` thật, và `select_parent_group()` (đã có sẵn, generic) dùng được nguyên xi như
Alg. 1. Chi phí: mỗi bản prompt × mỗi probe = 1 lượt gọi LLM — tính theo quota cookie trước.
