# POLICIES — chống xung đột giữa các agents

Các policy dưới đây là **bắt buộc** và được cài đặt trong `agentics/policies/`.
Mục tiêu: nhiều agent (và nhiều contributor người thật) làm việc đồng thời trên
cùng một repo mà không giẫm chân nhau.

## P1 — Issue claim locking
- Issue cần làm phải được **claim** trước khi làm: agent comment `/claim` và gắn
  label `claimed/<agent-id>`.
- Registry 2 lớp: GitHub label + bảng `claims` trong shared memory. Kiểm tra cả hai.
- Mỗi claim có TTL (mặc định 45 phút × số chu kỳ đã vượt). Hết hạn hoặc agent mất
  heartbeat > 2 chu kỳ → claim tự giải phóng, issue về hàng chờ.
- Agent không được claim issue đã có claim của người khác. Tranh chấp → Supervisor
  phân xử (ưu tiên agent còn heartbeat gần nhất).

## P2 — Nhánh và commit
- Nhánh riêng: `agent/<agent-id>/issue-<số>-<slug>`. Cấm push lên nhánh của agent khác
  hoặc thẳng `main` (chỉ Integrator được sửa `main`, qua merge).
- Commit message format: `agent(<agent-id>): <mô tả> (#<issue>)` — truy vết được ai làm gì.

## P3 — Ownership matrix (vùng file)
| Role | Vùng sở hữu |
|---|---|
| Coder | `agentics/`, `tests/`, `scripts/` |
| Researcher | `research_notes/` |
| Writer | `docs/`, `README*` |
| Integrator | `main` (merge), `.github/` |
| Supervisor | issue metadata, `configs/lab.yaml` |

- Trước khi push, agent diff các file thay đổi; nếu > 30% file rơi vào vùng sở hữu
  của role khác đang **active** (heartbeat sống) → agent **yield**: hoàn tác phần đó,
  tạo issue mới giao cho role phù hợp, hoặc đợi chu kỳ sau.
- Contributor người dùng quy ước tương tự: mở issue và claim như agent.

## P4 — Merge queue tuần tự
- Chỉ **Integrator** merge, và merge **một PR mỗi lần** (hàng đợi FIFO theo thời gian
  PR được đánh label `ready-to-merge`).
- Trước mỗi merge: `git fetch` + rebase nhánh PR lên `main` mới nhất; conflict →
  trả lại cho agent tạo PR tự rebase (Integrator comment `/rebase-request`).
- Merge method mặc định: **rebase-merge** (lịch sử phẳng, dễ truy vết).
- **Auto-merge mặc định tắt**. Chỉ owner bật `AUTO_MERGE=true` sau khi đã xác minh
  branch protection, CI bắt buộc, secret scanning và cross-review. PR không được
  tự approve; Integrator vẫn phải kiểm tra trạng thái CI và quyền merge.

## P5 — Cross-review bắt buộc
- PR phải được **1 agent khác** (khác tác giả) review `APPROVE` mới vào hàng đợi merge.
- Reviewer kiểm tra: đúng issue, ownership matrix, chất lượng, citation chuẩn
  (với phần nghiên cứu), không chèn secret.

## P6 — Heartbeat & ốm chết
- Mỗi agent ghi heartbeat vào shared memory đầu mỗi chu kỳ.
- Supervisor quét: agent mất heartbeat > 2 chu kỳ → coi như chết: giải phóng claim,
  đánh dấu workspace cần dọn, ghi decision vào memory.

## P7 — Giao tiếp qua memory, không qua "lời nói"
- Mọi thông tin cần chia sẻ (kết quả research, quyết định, cảnh báo) phải vào
  **shared memory** (bảng `messages`/`findings`/`decisions`) — không agent được dựa
  vào state cục bộ của agent khác.
- Vector store là "đại thư viện" chung: trước khi nghiên cứu chủ đề mới, agent
  phải query xem đã có finding chưa (tránh trùng lặp).

## P8 — Vòng lặp vô tận & dừng
- Agents chạy liên tục; **chỉ owner** quyết định dừng: file `STOP` ở thư mục gốc.
- Khi dừng: agent hoàn tất node hiện tại, giải phóng claim, ghi trạng thái, thoát.
  Không có agent nào bị cắt giữa chừng khi đang push.

## P9 — Traces & tool registry (GEA)
- Mỗi chu kỳ làm việc của Coder/Writer ghi 1 **trace** (đã thử gì, fail ở đâu, kết quả) vào
  shared memory; reviewer/integrator cập nhật kết quả (approved/rejected/merged/failed).
- Trước khi lập plan, agent đọc trace của agent khác và registry tool; tool mới chỉ vào
  registry khi PR được **merge**.

## P10 — Kiểm tra hai tầng
- Tầng 1 (tác giả, trước push): `py_compile` + `pytest -x`. Tầng 2 (Integrator, trước merge):
  test đầy đủ trên `main + PR`; fail → không merge, ghi failure trace + comment vào PR.

## P11 — Self-evolution có rào chắn
- PR `evolution` chỉ được sửa allowlist. Chi tiết và giới hạn: [EVOLUTION.md](EVOLUTION.md).
