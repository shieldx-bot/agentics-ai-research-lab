# LOCAL MODE — lab chạy hoàn toàn trên một máy, không GitHub

> Trạng thái: nhánh `update_interactive_local`. Nhánh này hội tụ **một** dựng
> chế độ local duy nhất (cd8b536) — các proposal song song cũ đã được rà soát
> và chỉ phần nào bổ trợ thật sự mới được port vào (xem §7).

## 1. Nó thay thế GitHub bằng gì?

| Khái niệm GitHub | Thay thế local |
| --- | --- |
| remote repository | bare repo tại `data/lab.git` (`LocalRepo`) |
| issues | bảng `tasks` trong shared memory (SQLite) |
| claim labels (P1) | bảng `claims` — nguồn sự thật duy nhất (`LocalClaimPolicy`) |
| pull requests | bảng `reviews` ("MR") |
| PR review / approval | verdict do Reviewer ghi vào `reviews` (P5: không tự duyệt MR của mình) |
| merge + push lên main | hàng đợi merge tuần tự (`LocalMergeQueue`, P4) merge `--no-ff` vào bare repo |

Không dòng code nào trong vòng nghiên cứu thực hiện call mạng (trừ traffic
LLM/provider). Package `github_ops/` giữ nguyên trên đĩa như mã ngủ đông —
vẫn có unit test, không còn được import bởi vòng lặp.

## 2. Hai câu hỏi của owner, hai cơ chế trả lời

**Q1 — "Agent làm việc local mà có sửa nhầm code agent không?"** → **P10
protected paths** (`agentics/policies/guard.py`), chặn 4 lớp:

1. `RepoWorkspace.write_file` — plan không thể tạo/sửa file protected
2. diff chạm protected path → bị đánh dấu, không mở MR được
3. **pre-receive hook** trong `data/lab.git` (lớp 2.5, thêm ở §7-B) — kể cả
   `git push` thẳng vào origin, bỏ qua mọi tầng app, vẫn bị git từ chối
4. `LocalMergeQueue.drain` + `node_review` — MR chạm protected path không
   bao giờ được duyệt/merge

Ngoại lệ **chỉ** owner cấp: `python -m agentics.cli allow-protected <task_id>`
(theo task) hoặc `ALLOW_PROTECTED_PATHS=1` (theo phiên chạy). `accounts/` và
`configs/secrets.yaml` không bao giờ được miễn, kể cả khi có exemption.

**Q2 — "Owner làm gì biết các agent đang làm gì, vì sao?"** → agent công bố
**intent** (goal / why / steps / expected files) vào heartbeat tại mỗi bước
(claim, plan, review, open_mr…), kèm bảng `journal` ghi mọi quyết định. Ba
cách xem, cùng một nguồn sự thật:

- Web: `python -m agentics.cli dashboard` → `http://127.0.0.1:8080`
  (agent card có mục đích hiện tại, task board, MR pipeline, merge queue,
  memory stats, activity feed — làm mới 2.5s, hoạt động offline)
- Terminal: `python -m agentics.cli journal -f` (stream trực tiếp, thêm ở §7-C),
  hoặc `agentics journal -a a02 -n 5` để xem một agent
- `agentics status` — trạng thái nhanh (node + intent + task đang giữ)

## 3. Backlog tự nuôi (Supervisor)

`TaskSource.supervisor_top_up()` sinh task khi backlog cạn, theo thứ tự ưu
tiên:

1. **Điều tra failure** — trace thất bại mới nhất chưa có task nào nhận →
   task `bugfix` "Investigate failure …" (vòng GEA ăn chính lịch sử sự cố
   của đàn; failure đã được điều tra không bao giờ mở lại)
2. **Survey angle xoay vòng** — góc nghiên cứu ít dùng nhất được mở tiếp
   (State-of-the-art → Benchmark → Empirical comparison → Failure-mode →
   Reproducibility → Tooling), nên backlog cạn mãi không lặp lại cùng một
   survey

Mỗi lần cạn backlog chỉ sinh tối đa 1 task; owner luôn có thể chèn việc riêng
bằng `python -m agentics.cli task "tên việc" --category code`.

## 4. Chạy

```bash
cp .env.example .env            # cấu hình cookie/LLM; KHÔNG cần GITHUB_TOKEN
python -m agentics.cli run      # vòng lặp vô tận, 100% local
python -m agentics.cli dashboard  # tab 2: mắt owner
python -m agentics.cli journal -f # tab 3: stream sự kiện terminal
```

Kiểm tra sức khỏe offline không cần LLM:

```bash
pytest tests/test_local_review_merge.py -q   # E2E thật: claim→implement→review→merge bằng git thật
pytest tests/test_local_ops.py -q            # đơn vị: repo, task, claim, merge queue, hook
```

## 5. Merge queue hoạt động thế nào (P4)

Integrator rút hàng đợi tuần tự: MR có verdict `approved` mới được rút; mỗi
MR chạy tier-2 verify (full test trên main+MR) trước khi merge `--no-ff` lên
main của bare repo; conflict hoặc verify fail → MR bị đánh dấu, task trả về;
`AUTO_MERGE=0` → queue chỉ ghi nhận MR sẵn sàng, owner tự merge tay.

## 6. Biến môi trường chính (local mode)

| Biến | Ý nghĩa |
| --- | --- |
| `AUTO_MERGE` | mặc định `0` — queue chỉ ghi nhận MR sẵn sàng, owner tự merge tay; `1` = merge tự động sau tier-2 |
| `PROTECTED_PATHS` | mở rộng danh sách path protected (phân tách bằng dấu phẩy) |
| `ALLOW_PROTECTED_PATHS` | `1` = owner cho phép cả run chạm code lab (kể cả qua git push) |
| `DASHBOARD_PORT` | port dashboard (mặc định 8080) |
| `GH_SCAN_CACHE_TTL`, `GH_MIN_API_INTERVAL`, `HB_INPUT_TIMEOUT` | reliability tuning hợp về từ main (dormant GH ops / Hyperbrowser fail-fast) |

## 7. Nhật ký tích hợp (decision records)

Mục này ghi rõ các quyết định khi hội tụ các nhánh local-mode song song về
**một nhánh duy nhất** — người đẩy sau phải đối chiếu code của người đi trước
và chỉ port phần bổ trợ, không dựng lại kiến trúc.

**A. Hợp `main` vào nhánh (408e7a5 — production reliability).** Nhánh local
được cắt từ 4a39204, thiếu các fix reliability mà main đã có: LLM failure
giờ degrade về `state["error"]` → reflect (nhả claim, ghi trace + lesson)
thay vì giết LangGraph task; Hyperbrowser fail-fast nhận challenge/login-wall;
GH pacing/retry + scan cache (dormant nhưng vẫn test). Resolution xung đột
orchestrator: giữ phân loại `is_auth_failure` của nhánh (chính xác hơn
heuristic cũ mà main đã xóa) — hai chính sách không mâu thuẫn vì node graph
không còn raise; chỉ lỗi auth thật bubble lên mới cooldown cookie.

**B. Pre-receive hook cho `data/lab.git` (lớp P10 thứ 4).** Ba gate P10 của
cd8b536 chỉ thấy traffic đi qua code agentics; `git push` thẳng vào bare repo
bypass tất cả. Hook được `LocalRepo.ensure()` cài/sửa tự sinh từ danh sách
`protected_paths` hiện tại, chỉ ghi đè hook mang sentinel
`# agentics pre-receive (managed)` — hook của owner không bao giờ bị đụng.
`ALLOW_PROTECTED_PATHS=1` của shell push được tôn trọng (receive-pack kế thừa
env khi push local), giống ngữ nghĩa override ở tầng app.

**C. `agentics journal -f`.** Dashboard web của cd8b536 đã đủ cho owner; lệnh
này là bản terminal cùng nguồn dữ liệu (bảng `journal`) cho ai làm việc bằng
bàn phím — không tạo lớp lưu trữ sự kiện mới, không trùng lặp.

**D. Sửa rotation `supervisor_top_up`.** Bản cd8b536 luôn mở
`SURVEY_ANGLES[0]` nên backlog cạn là lặp lại cùng một survey mãi mãi (dedup
chỉ xét task OPEN). Đã thay bằng: (1) task điều tra failure mới nhất chưa có
task nhận, (2) survey angle ít dùng nhất (xoay vòng thật sự, tính trên toàn
bộ bảng tasks).

**E. Những gì cố ý KHÔNG port.** Kiến trúc emulator GitHub-API của proposal
cũ (duck-typing PyGithub) bị bỏ — bản cd8b536 viết lại node trực tiếp lên
bảng SQLite sạch hơn, không còn lớp giả lập cần giữ đồng bộ. EventBus/
events.jsonl riêng cũng bỏ — bảng `journal` trong SQLite đã là nguồn sự thật
chung cho cả dashboard lẫn `journal -f`; thêm một kênh jsonl nữa chỉ tạo hai
nguồn thật. Smoke script riêng cũng bỏ — `tests/test_local_review_merge.py`
đã là E2E thật với git thật, chạy được như một lệnh kiểm tra.
