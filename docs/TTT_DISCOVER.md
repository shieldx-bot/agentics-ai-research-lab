# TTT_DISCOVER — Đưa khung "Test-Time Training to Discover" vào lab

Nghiên cứu tích hợp khung RL-at-test-time của bài báo **"Learning to Discover at
Test Time"** (Yuksekgonul, Koceja et al. — TTT-Discover, mã nguồn
`discover-main.zip`) vào vòng lặp nghiên cứu của lab, nhằm tối ưu kết quả đầu ra
(khám phá có reward cao hơn trong cùng ngân sách deep-dive).

Tài liệu này là **thiết kế** (design), chưa phải code: mỗi mục ghi rõ điểm chạm
code (file + hàm) để triển khai theo phase.

---

## 1. TTT-Discover nói gì (tóm tắt kỹ thuật từ mã nguồn)

Bài báo định nghĩa mỗi bài toán khoa học là một MDP một bước:

| Khái niệm | Ý nghĩa |
|---|---|
| State `s` | một ứng viên lời giải (code kernel, step function, thuật toán...) |
| Action `a` | output của policy LLM (thinking + code) |
| Transition `T` | parse/thực thi code từ `a` ra state mới |
| Reward `R(s)` | liên tục (1/runtime, 1/upper-bound...), **= 0 nếu fail validity check** |
| Discovery | sự kiện tìm được `s` với `R(s) > r_sota` (best đã biết) |

Điểm khác biệt then chốt với RL chuẩn: mục tiêu là **một lời giải xuất sắc nhất**,
không phải policy có average reward cao. Hai cơ chế khai thác điều đó:

1. **Entropic objective** `Jβ` — ưu tiên phần thưởng MAX thay vì trung bình:
   `w(a) = e^(β·R(s,a)) / E[e^(β·R)]`, advantage `= w − 1`; β được giải thích
   **adaptive** bằng binary search sao cho `KL(q‖uniform) = log 2` (q là phân phối
   softmax trên reward). Mã: `ttt_discover/rl/train.py::compute_advantages`
   (nhánh `entropic_adaptive_beta`, dòng 103–155). Kèm quy tắc
   `remove_constant_reward_groups`: nhóm có toàn reward bằng nhau → bỏ qua
   (không có thông tin).
2. **PUCT state reuse** — chọn state khởi đầu từ buffer kinh nghiệm:
   `score(i) = Q(i) + c·scale·P(i)·√(1+T)/(1+n(i))` với:
   - `Q(i)` = **MAX** reward among con cháu của state `i` (không phải mean!),
     hoặc chính `R(i)` nếu chưa từng được expand;
   - `P(i)` = prior theo rank reward trong buffer;
   - `n(i)` = số lần `i` hoặc hậu duệ đã được expand; `T` = tổng số expansions;
   - **lineage blocking**: trong một batch không chọn hai state cùng huyết thống.
   Mã: `ttt_discover/tinker_utils/sampler.py::PUCTSampler` (dòng 74–262).

Phần "learning" thật (`train: θ ← θ + η∇θJβ`, LoRA rank-32 trên gpt-oss-120b qua
Tinker API, 50 steps × 512 rollouts, ~$500/bài) là phần **cập nhật trọng số**.

---

## 2. Khả thi hoá: cái gì port được, cái gì không

**Sự thật phải nói thẳng:** lab chỉ gọi LLM qua Perplexity trên Hyperbrowser
(cookie account, web product đóng) → **không có trọng số, không có logits, không
có API fine-tune**. Component `train` của TTT-Discover là bất khả thi về nguyên
tắc, không phải thiếu tài nguyên. Mọi lời hứa "RL cập nhật trọng số" trong lab
sẽ là giả.

Nhưng bài báo tự chứng minh phần search-side đã mang giá trị lớn: baseline
**Best-of-25600** (cùng ngân sách, không học) đã vượt luôn construction của
AlphaEvolve trên Erdős; TTT-Discover thắng Best-of-N nhờ **PUCT reuse + entropic
selection**, không chỉ nhờ LoRA. Cả hai cơ chế đó hoạt động trên **buffer kinh
nghiệm**, không cần đụng trọng số → port được nguyên vẹn vào lab.

Bảng ánh xạ khái niệm:

| TTT-Discover | Lab agentics | Ghi chú |
|---|---|---|
| Problem description `d` | `research_topic` + issue + world model | đã có |
| State `s` (ứng viên) | hàng hypothesis trong registry (claim + protocol) | đã có |
| Action `a` | LLM call (ideation / protocol generation) | đã có |
| Transition `T(a)` | chạy protocol → evidence (`verify/runner.py`) | đã có |
| **Reward `R(s)`** | **CHƯA CÓ** — chỉ có status nhị phân supported/refuted | **C1 mới** |
| **`r_sota` tracking** | **CHƯA CÓ** | **C4 mới** |
| Buffer `H` | hypotheses registry + `experiment_ledger` | đã có |
| Reuse heuristic | deep-dive backlog: BFS (depth,id) + bandit **+5 tuyến tính** | **C2 thay bằng PUCT** |
| `train` (cập nhật θ) | **BẤT KHẢ THI** (Perplexity đóng) | C5: thay bằng context-level |
| State-action reuse `c_i` | world model render + rejected_render + insight harvest | đã có (tier "evolutionary" của paper) |

Kết luận: lab hiện đang ở đúng tier **evolutionary search** của bài báo (AlphaEvolve
tier — có state-action reuse qua context). Nấc thang tiếp theo mà kiến trúc cho
phép là **PUCT + entropic selection trên buffer** — chính là khoảng cách giữa
Best-of-N và TTT-Discover trong Figure 1 của paper.

---

## 3. Thiết kế tích hợp (4 component + 1 giới hạn trung thực)

### C1 — Reward liên tục `R(s)` (mới, file `agentics/verify/reward.py`)

Định nghĩa reward của một settle, dùng ngay dữ liệu `criterion` (metric/op/target)
+ `measured` mà `_run_protocol_deep_dive` đã có (nodes.py ~1644):

```
R(s) = 0                                   nếu status != "supported"
     = 0                                   nếu unanchored (self_authored_ratio = 1)
     = margin chuẩn hoá, clip [0, 1]       nếu anchored-supported
       margin = (measured − target)/scale  khi op = ">"  (scale = |target| hoặc
                                                    span lịch sử của metric)
       margin = (target − measured)/scale  khi op = "<"
```

Điểm then chốt về **chính trị đo lường** (khớp قانون settle vừa kích hoạt ở
commit `6d832ec`): trong paper, `R(s) = 0` khi fail validity check. Trong lab,
**settle anchor law chính là validity check**: một verdict mà agent tự chấm toàn
bộ (ratio = 1.0) là verdict fail validity → reward 0, không bao giờ reinforce
search. Nhờ vậy reward không thể bị agent bơm lên bằng cách tự phong supported —
đúng ràng buộc "ai cầm bút thang điểm" đã thống nhất.

### C2 — PUCT frontier thay BFS + bandit tuyến tính (`agentics/memory/shared_memory.py`)

Hiện trạng (`discovery_frontier`, shared_memory.py:1587): sắp xếp BFS
`(depth, id)` — không nhìn reward; bandit `_branch_feedback` (nodes.py:1196) cộng
**+5 tuyến tính** vào novelty của discovery root mỗi khi một branch supported
→ đúng cơ chế "giàu càng giàu" mà vòng review khắc kẽ đã kết án (discovery tự pin
và hút cạn frontier).

Thay bằng PUCTSampler của paper, tính trực tiếp từ các bảng hiện có (không cần
cột mới):

- `n(s)` = số settle đã xảy ra trong **subtree** của `s` (đếm từ
  `experiment_ledger` join con cháu) — số lần expand;
- `Q(s)` = **MAX** reward (C1) among con đã settle trong subtree — theo đúng
  "we care about the best outcome starting from a state, not the average";
  chưa có con → `Q(s) = novelty(s)/100` (proxy R(s) trước khi expand);
- `P(s)` = prior theo rank `Q` trong buffer frontier (`(N − rank)/Σ`, sampler.py:183);
- `T` = tổng settle trong toàn bộ các subtree đang xét;
- `score(s) = Q(s) + c·scale·P(s)·√(1+T)/(1+n(s))`, `c = settings.puct_c` (mặc định 1.0),
  `scale` = max(Q) − min(Q) trong buffer (sampler.py:177);
- **lineage blocking** trong batch lấy top-k: không chọn hai hypothesis cùng một
  parent (đỡ hơn paper: chỉ block cùng family root);

Bonus khai thác `√(1+T)/(1+n)` là **thuốc giải chính** cho rich-get-richer:
mỗi lần deep-dive một branch, n tăng → bonus giảm → sibling chưa ghé thăm trồi
lên, kể cả novelty thấp hơn. Depth cap giữ nguyên như lớp bảo vệ cuối.

Điểm chạm: `discovery_frontier()` trả thêm trường `puct_score/Q/n` cho
`node_research` block "frontier_priority" (nodes.py:1442–1465) dùng để sắp xếp
lại `fresh` thay vì chỉ boost thứ tự BFS.

### C3 — Entropic credit assignment thay +5 tuyến tính (`_branch_feedback`)

Khi một discovery subtree tích luỹ k kết quả reward `r_1..r_k` (C1), mức reinforce
novelty của discovery root là:

```
q_i = softmax_β(r_i)            (β adaptive: binary search sao cho
                                 KL(q‖uniform) ≈ log 2 — port từ
                                 train.py:103–155, thuần math, không torch)
boost_i = bonus_max · k · q_i / Σ_j q_j    (để tổng boost giữ ngân sách cũ)
```

- Con có reward max nhận boost gần như trọn `bonus_max`; con reward thấp nhận
  ~0 — ưu tiên MAX đúng Jβ, thay vì +5 đều cho mọi supported.
- **`remove_constant_reward_groups`**: toàn bộ `r_i` bằng nhau → boost = 0 cho
  cả nhóm (kết quả lặp lại không có thông tin — đúng chẩn đoán "hàng trăm
  dòng replay trùng" của vòng review).
- `refuted`/`unanchored` → `r = 0` → tự động weight thấp nhất, không cần nhánh
  đặc biệt nữa.

Điểm chạm: `_branch_feedback` (nodes.py:1196) đọc reward của subtree qua
`experiment_ledger` rồi gọi `discovery_reinforce(disc_id, boost_i)` — chữ ký hàm
giữ nguyên, chỉ đổi cách tính bonus.

### C4 — `r_sota` tracking + dashboard discovery objective

Paper: *"A discovery is an event where R(s) > r_sota"*. Lab cần memory của
best-so-far để phát hiện đúng sự kiện đó:

- `shared_memory.best_reward()` = MAX reward among anchored-supported settles
  (một query lên `experiment_ledger WHERE anchors != ''`);
- lúc settle: nếu `R(s) > r_sota` cũ → journal `new_state_of_the_art`
  {hyp, reward, margin} — sự kiện discovery theo đúng định nghĩa formal của
  paper, tách bạch khỏi novelty-grade chủ quan (giữ nguyên discovery gate
  double-judge + citation);
- dashboard: đường "best reward so far" theo thời gian + phân phối reward mỗi
  sweep (analogue của Figure 1: reward distribution tại step 0/24/49) — KPI
  "đầu ra tối ưu hơn" đo được, thay vì chỉ đếm số discovery.

### C5 — Nấc "training" thay thế trung thực: entropic context reuse

Không có gradient → "cải thiện policy" duy nhất khả thi là **cải thiện context**.
Áp entropic weight (C3) lên việc chọn kinh nghiệm đưa vào prompt ideation: các
attempt có `w_i` cao (gần reward max) được trích vào seed context (world render /
few-shot "đường đi của discovery tốt nhất") với xác suất tỉ lệ `q_i`; attempt
weight ~0 bị loại khỏi context. Đây chính là `Jβ` thỏa mãn ở tầng prompt-policy:
mỗi sweep, phân phối prompt nghiêng dần về hành vi sinh ra reward max — analogue
hợp pháp duy nhất của "policy tốt dần" trong ràng buộc Perplexity-only.

---

## 4. Chính sách an toàn (bài học từ vòng review khắc kẽ)

1. **Reward chỉ chảy từ anchored settle** — unanchored = validity fail = 0. Không
   có đường nào tự chấm để nuôi search.
2. **Discovery gate giữ nguyên** (double-judge + citation, nodes.py:1289): PUCT
   chỉ sắp xếp thứ tự đào, không hạ bar nhận discovery.
3. **Không ReinforceHacking**: reward tính từ `measured` do protocol chạy ra, mà
   protocol chịu replay gate (P21/P24) + pinned-hash data → con đường bơm
   `measured` để nhận reward bị replay bắt.
4. **Chống độc chiếm frontier**: depth cap (branch_max_depth) + lineage blocking
   + bonus khai thác decay — ba lớp, giữ đa dạng góc đào (mechanism / generalize /
   boundary / amplify).

## 5. Kế hoạch triển khai theo phase

| Phase | Nội dung | Test bắt buộc (FakeLLM/FakeWS pattern) |
|---|---|---|
| 1 | C1 reward + C4 r_sota + journal `new_state_of_the_art` | unanchored/refuted → 0; margin chuẩn hoá đúng cho op `>`/`<`; clip; sota event chỉ khi vượt |
| 2 | C2 PUCT frontier + lineage blocking | n tăng làm bonus decay; Q = max con; batch không chọn cùng parent; buffer rỗng rơi về BFS cũ |
| 3 | C3 entropic credit + constant-group zero | adaptive β hội tụ (KL ≤ log 2); reward đều nhau → boost 0; reward max → boost gần bonus_max |
| 4 | C5 entropic context reuse + dashboard reward curve | chọn context theo q_i; dashboard đọc được best curve |

Config mới (`agentics/config.py`): `puct_c: float = 1.0`,
`entropic_bonus_max: float = 8.0` (thay ngưỡng +5 cũ),
`reward_scale: str = "target"` (hoặc `"history"`).

**Non-goals (ghi rõ để không ai hứa sai):** không LoRA/Tinker, không cập nhật
trọng số, không multi-timestep MDP (state reuse qua registry đã phủ), không đụng
settle anchor law hay discovery gate — hai cơ chế đó là validity check cho toàn
bổ sung này.

## 6. Kỳ vọng đầu ra

Theo Figure 1 của paper, khoảng cách TTT-Discover vs Best-of-N (cùng ngân sách)
là nơi PUCT + entropic selection tạo ra khác biệt: reward-max xuất hiện sớm hơn
và cao hơn trên cùng số deep-dive. Với lab, KPI kiểm chứng sau phase 2–3:
số deep-dive cần để đạt discovery anchored đầu tiên giảm ít nhất 30% so với
log trước đó, và tỷ lệ settle trong subtree discovery tăng trong khi số góc
branch khác nhau được thăm **không giảm** (bonus decay phải bù cho việc Q-max
ưu tiên branch ngon) — đo trực tiếp từ journal `frontier_priority` /
`discovery_confirmed` + `experiment_ledger`.
