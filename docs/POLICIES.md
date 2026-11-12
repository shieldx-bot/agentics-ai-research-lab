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
- Mỗi agent có một clone/workspace riêng dưới `data/workspaces/<agent-id>`; không dùng chung working tree.
- Đường dẫn đọc/ghi phải nằm trong workspace của agent; đường dẫn tuyệt đối, traversal và symlink thoát workspace bị từ chối.
- Lệnh kiểm thử chỉ chạy bằng Python interpreter của môi trường, với `pytest`, `py_compile` hoặc script `.py` nằm trong workspace. Lệnh inline `python -c` và executable tùy ý bị từ chối.
- Process kiểm thử nhận môi trường tối thiểu, không kế thừa GitHub token, Perplexity cookie hay API key; có timeout và dọn process group khi hết hạn.
- **Giới hạn bảo mật:** runner này là host-process guardrail, không phải OS/container sandbox. Không chạy mã không tin cậy trên host; triển khai production phải dùng rootless container/VM sandbox với resource/network policy riêng.

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

## P12 — Nhịp lịch sự với GitHub API
- Mọi agent dùng chung **cache scan issue 45s** (một lần GET list cho cả đàn, không mỗi agent
  một lần) — chống bão 403/secondary rate limit đã gặp trong vận hành thật.
- Agent khởi động **so le** (mỗi agent cách nhau 7s, tối đa 60s) để các chu kỳ đầu không bung
  đồng loạt scan + claim.
- `GitHubClient` cấu hình rõ: `timeout=30s`, `GithubRetry(secondary_rate_wait=60,
  max_rate_limit_wait=900)` — 403 có `Retry-After` được chờ và thử lại trong suốt, quota chính
  cạn thì fail nhanh thay vì đóng băng chu kỳ.
- Claim issue: **label là khoá chính tắc**. Comment claim fail (network) không được rollback
  claim — rollback thất bại sẽ để lại label mồ côi khoá issue vĩnh viễn (đã gặp ở issue #19).
  Nhánh release/sweep của agent sở hữu sẽ dọn label sau.

---

## P10 — Protected paths: code lab là của owner (local-only mode)

**Vấn đề:** khi agents làm việc trực tiếp trên local, nguy cơ lớn nhất là một
plan "lạc đường" tự sửa code của chính hệ thống agent (`agentics/`), cấu hình
runtime, hoặc tệ hơn là chạm vào cookie/credentials.

**Chính sách:** các đường sau bị **chặn cứng** với mọi agent:

```
agentics/   configs/   accounts/   scripts/   .github/
```

Thực thi ở **3 tầng độc lập** (`agentics/policies/guard.py`) + **1 tầng git**:

1. **Ghi file** — `RepoWorkspace.write_file` từ chối ngay khi plan yêu cầu
   tạo/sửa một file protected (ValueError → cycle yield).
2. **Diff trước push** — `node_implement` quét toàn bộ changed files; vi phạm
   → yield, không commit, không push.
   2.5. **Pre-receive hook trong `data/lab.git`** — các tầng trên chỉ thấy
   traffic đi qua code agentics; một `git push` thẳng vào bare repo bỏ qua tất
   cả. Hook tự sinh từ danh sách `protected_paths` hiện hành
   (`LocalRepo.install_pre_receive`) và git từ chối mọi push đưa protected
   path vào origin ngay ở tầng git, kể cả nhánh/MR chưa từng tồn tại. Chỉ
   hook mang sentinel
   `# agentics pre-receive (managed)` mới được agentics cập nhật — hook của
   owner không bao giờ bị ghi đè.
3. **Cửa review + cửa merge** — Reviewer hạ mọi verdict "approve" thành
   "request_changes" nếu diff chạm protected paths; `LocalMergeQueue.drain`
   từ chối merge MR vi phạm kể cả khi đã được duyệt.

**Exception duy nhất thuộc về owner:**

- Theo task: `python -m agentics.cli allow-protected <task_id>` (thu hồi:
  `--revoke`). Cờ nằm trên row của task — agent không tự cấp được.
- Theo phiên: biến môi trường `ALLOW_PROTECTED_PATHS=1`.

`accounts/` và `configs/secrets.yaml` (credentials) **không bao giờ** được
miễn, kể cả với exception.
