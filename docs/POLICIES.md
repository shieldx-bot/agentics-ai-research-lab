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
- Ở chế độ local với `AUTO_MERGE=false`: hàng đợi ghi nhận MR "ready" rồi dừng —
  **owner là người bấm merge** qua `agentics mr list / approve / merge` (lệnh merge
  vẫn chạy đủ P10 guard + tier-2 verification + mọi side-effect như Integrator).

## P5 — Cross-review bắt buộc
- PR phải được **1 agent khác** (khác tác giả) review `APPROVE` mới vào hàng đợi merge.
- Reviewer kiểm tra: đúng issue, ownership matrix, chất lượng, citation chuẩn
  (với phần nghiên cứu), không chèn secret.
- MR bị `request_changes` không bị bỏ rơi: ở chu kỳ kế tiếp chính **tác giả**
  rework ngay MR đó — tiếp tục từ head đã push của nhánh (không reset về main,
  không mở MR thứ hai cho cùng task), push phần sửa rồi mở lại MR `open` cho
  reviewer duyệt lại. Task giữ trạng thái `in_review` trong lúc rework.

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

## P15 — Kiểm tra hai tầng
- Tầng 1 (tác giả, trước push): `py_compile` + `pytest -x`. Tầng 2 (Integrator, trước merge):
  test đầy đủ trên `main + PR`; fail → không merge, ghi failure trace + comment vào PR.

---

## P10 — Protected paths: code lab là của owner (local-only mode)

**Vấn đề:** khi agents làm việc trực tiếp trên local, nguy cơ lớn nhất là một
plan "lạc đường" tự sửa code của chính hệ thống agent (`agentics/`), cấu hình
runtime, hoặc tệ hơn là chạm vào cookie/credentials.

**Chính sách:** các đường sau bị **chặn cứng** với mọi agent:

```
agentics/   configs/   accounts/   scripts/   .github/
```

Thực thi ở **4 tầng độc lập** (`agentics/policies/guard.py`):

1. **Ghi file** — `RepoWorkspace.write_file` từ chối ngay khi plan yêu cầu
   tạo/sửa một file protected (ValueError → cycle yield).
2. **Diff trước push** — `node_implement` quét toàn bộ changed files; vi phạm
   → yield, không commit, không push.
3. **Cửa review — re-scan nhánh thật (tầng 4)** — Reviewer tự quét lại branch
   đã push trong workspace của chính mình (`branch_files_vs_main`) thay vì tin
   danh sách file do workspace của tác giả ghi lại lúc mở MR; vi phạm →
   "request_changes" ngay, không tốn một lần gọi LLM nào. Prompt review cũng
   dùng đúng danh sách file thật của branch.
4. **Cửa merge — re-scan nhánh thật (tầng 4)** — `LocalMergeQueue.drain` nhận
   callback `scan_branch`: verdict của guard dựa trên diff thật của branch
   trong workspace của Integrator; một branch "đầu độc" kèm danh sách file
   sạch vẫn bị từ chối merge (mỗi workspace là một trust domain riêng).

**Exception duy nhất thuộc về owner:**

- Theo task: `python -m agentics.cli allow-protected <task_id>` (thu hồi:
  `--revoke`). Cờ nằm trên row của task — agent không tự cấp được.
- Theo phiên: biến môi trường `ALLOW_PROTECTED_PATHS=1`.

`accounts/` và `configs/secrets.yaml` (credentials) **không bao giờ** được
miễn, kể cả với exception.

**Tầng 4a — DETECT trên cây runtime** (`agentics/sandbox/integrity.py`): 3 tầng
trên ngăn *changeset* của agent chạm vào engine, nhưng chưa có gì bảo vệ **cây
runtime mà chính lab đang chạy từ đó** — một merge hỏng, một process lọt qua
guard, hay owner sửa tay đều có thể thay đổi engine giữa chừng mà không ai hay.
`IntegrityGuard` hash toàn bộ `agentics/`, `configs/`, `scripts/` và các file
dependency lúc khởi động (`data/integrity/baseline.json`); Supervisor kiểm tra
lại mỗi chu kỳ thứ 5; bất kỳ drift nào trở thành `integrity_alert` trên journal
+ alert trong memory — là cảnh báo, không phải dừng vòng lặp: owner quyết định.
Owner kiểm tra bất kỳ lúc nào bằng `python -m agentics.cli verify` (chấp nhận
thay đổi đã review: `--refresh`).

**Tầng git — pre-receive hook** *(bổ trợ, không thay thế các tầng trên)*
(`data/lab.git/hooks/pre-receive`, sinh tự động bởi `LocalRepo.ensure()`):
mọi push đi qua hook này, kể cả push nằm ngoài graph (quy trình lỗi, thao tác
tay). Commit có committer email `agentics+*` bị từ chối nếu chạm vào các đường
**không bao giờ được miễn**:

```
accounts/   configs/secrets.yaml   data/   logs/   STOP   .env
```

Danh sách cố ý hẹp hơn protected paths của guard trong-process: exemption theo
task của owner (`allow-protected`) chỉ có tác dụng ở các tầng trong process,
nên hook chỉ khóa cứng phần chính sách tuyên bố "không bao giờ miễn", cộng
thêm runtime state của lab. Owner (mọi identity khác) không bị hạn chế. Hook
cũng từ chối xóa `main` với identity agent.

---

## P10 bổ sung — whitelist ring "cấm tuyệt đối" + lớp 4b tự phục hồi workspace
*(đóng góp từ người đến sau trên chính nhánh này — bổ trợ tầng 4a, không thay thế)*

**Vòng ngoài (whitelist):** ngoài danh sách protected, guard còn hỗ trợ
`writable_roots` — khi được cấu hình (mặc định trong `configs/lab.yaml` là hợp
các vùng ownership), MỌI file ngoài các gốc đó đều bị từ chối kể cả khi không
phải machinery (file lạ ở gốc repo, thư mục dữ liệu bẩn...). Owner exemption
theo task vẫn nâng được vòng này; `accounts/` và `configs/secrets.yaml` thì
không bao giờ.

**Tầng 4b — DETECT + tự phục hồi trên workspace của agent**
(`agentics/policies/integrity.py`): tầng 4a bảo vệ cây runtime, nhưng agent
*không chạy code trong cây runtime* — chúng chạy trong clone riêng dưới
`data/workspaces/`, nơi script thí nghiệm được phép chạy Python qua
`run_command` và có thể tính toán đường dẫn lúc runtime rồi ghi đè code lab
trong clone của mình. `IntegrityWatcher` re-hash các cây protected trong MỖI
workspace mỗi `INTEGRITY_INTERVAL_S` giây (mặc định 20, 0 = tắt); drift
(sửa / xóa / file mới trong vùng cấm) được **hoàn tác từ git ngay lập tức**
và ghi journal `integrity_drift` — dashboard báo đỏ. Hai watcher chia mặt
phẳng: 4a = cây runtime (chỉ phát hiện, owner quyết định), 4b = workspace
(phát hiện + tự lành, vì clone là tài sản tái tạo được từ main).

`main` của bare repo không cần watcher nào: nó chỉ tiến qua merge queue đã qua
đủ 3 tầng guard.

---

## P13 — File locks: hai agent không đè tay nhau

**Vấn đề:** mỗi agent một workspace riêng, nên hai agent cùng khai báo sửa một
file chỉ phát hiện nhau ở bước merge (conflict) hoặc tệ hơn — làm trùng việc.

**Chính sách:** ở bước **plan**, danh sách file dự kiến được **khóa toàn cục**
(SQLite, TTL = `FILE_LOCK_TTL_MINUTES`, mặc định 30 phút). Agent khác có plan
chạm vào file đang bị giữ sẽ bị **yield ngay trước khi đốt chu kỳ implement**,
kèm journal `lock_conflict` hiển thị trên dashboard. Khóa acquire kiểu
all-or-nothing (không bao giờ giữ nửa plan). Agent chết → khóa tự hết hạn;
supervisor quét dọn lock hết hạn mỗi chu kỳ scan; agent nhả khóa khi kết thúc
chu kỳ (reflect).

---

## P14 — Intent binding: diff phải nằm trong phạm vi đã tuyên bố

**Vấn đề:** dashboard hiển thị intent (goal/steps/expected_files) do agent
công bố ở `report()` — nhưng nếu diff thật đi ra ngoài phạm vi tuyên bố thì
intent chỉ là trưng bày.

**Chính sách:** `node_implement` đối chiếu `changed_files` với
`expected_files` đã khóa ở plan. File nào nằm ngoài phạm vi tuyên bố → yield
toàn bộ chu kỳ, journal `scope_violation` (kèm cả danh sách khai báo lẫn
file vượt phạm vi) — owner thấy ngay sự lệch giữa **lời khai** và **việc làm**.
Plan rỗng file (task thu hồi tài liệu vào memory) không bị kiểm tra này.

Đầy đủ kiến trúc Local Mode và đáp án 2 giả thuyết nghiên cứu:
[LOCAL_MODE.md](LOCAL_MODE.md).

---

## P16 — Kiểm chứng thực nghiệm: verdict đến từ code thật, không từ lời nói LLM

**Vấn đề:** mọi kết luận trong lab (tìm kiếm, đánh giá ý tưởng, verify) đều đi
qua LLM — một agent có thể "khẳng định" giả thuyết đúng mà chưa chạy gì cả.
Owner đặt 3 giả thuyết: (1) AI có thể dùng cmd/code/test như researcher thực
thụ không; (2) AI có thể tự kiểm chứng giả thuyết của chính mình không;
(3) kết quả kiểm chứng có chia sẻ cho mọi AI đều biết không.

**Chính sách:** ý tưởng chỉ được gọi là đã kiểm chứng khi có **bằng chứng chạy
thật** trong bảng `evidence` của memory dùng chung:

1. **Preregistration** — `node_plan` khai báo `verify_cmds` trong JSON kế hoạch
   (định nghĩa trước khi viết code: "exit 0 phải nghĩa là giả thuyết đúng").
   Danh sách lệnh được công bố trong intent cho cả swarm thấy.
2. **Chạy thật** — `node_implement` thực thi từng lệnh qua
   `agentics/verify_ops.py` (`run_experiment`, nằm trên
   `RepoWorkspace.run_command`: chỉ python đang chạy + module pytest/py_compile
   + script trong workspace, môi trường scrubbed, giới hạn thời gian).
3. **Verdict từ exit code, không từ model** — `0 → confirmed`,
   `124/126 → inconclusive` (timeout / bị guard chặn), còn lại `→ refuted`.
   `refuted` chặn push (agent phải sửa code hoặc re-plan);
   `inconclusive` chỉ journal cảnh báo — reviewer quyết, pipeline không
   kẹt vĩnh viễn vì một lệnh không chạy được.
4. **Kiến thức chung** — mỗi lần chạy ghi 1 dòng vào bảng `evidence`
   (task, agent, lệnh, exit, verdict, output ≤1500 ký tự) và broadcast tới
   mọi agent (messages + journal). Reviewer nhận danh sách bằng chứng trong
   prompt và trích dẫn histogram vào verdict. Owner xem toàn bộ qua card
   "Kiểm chứng thực nghiệm" trên dashboard hoặc `agentics evidence
   [task_id] [--search]`; agent khác tra cứu lại bằng
   `memory.evidence_search(...)` thay vì hỏi lại LLM.
5. **Re-verdict** — Reviewer/owner có thể hạ/tăng verdict một dòng bằng chứng
   (`set_evidence_verdict`, kèm `note` + `verdict_by`) — bằng chứng cũng bị
   review, không ai tự tôn vinh kết quả của mình.

Test: `tests/test_verify_ops.py`, `tests/test_evidence_memory.py`,
`tests/test_nodes_integration.py` (phần P16), `tests/test_dashboard_api.py`.
