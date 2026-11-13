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
(`data/lab.git/hooks/pre-receive`, sinh tự động bởi `LocalRepo.ensure()` và
`LocalRepo.ensure_guard()` — được cài lại mỗi lần khởi động lab, tự chữa lành
khi bị xóa/sửa): mọi push đi qua hook này, kể cả push nằm ngoài graph (quy
trình lỗi, thao tác tay). Quy tắc, đúng theo `origin_hook_script`:

- **Owner là người quyết**: push mà commit cuối không có committer email
  `agentics+*` đi qua không hạn chế (owner có thể đẩy bất cứ gì, kể cả code
  lab).
- **Agent không được đụng `main`** — work chỉ đi qua luồng MR; xóa `main` với
  identity agent cũng bị từ chối.
- **Commit của agent bị kiểm theo từng commit** (`git diff-tree` trên
  `base..new`, chỉ xét commit có committer `agentics+*`) — một nhánh được
  rebase trên các commit mới của owner không thể lẻn thay đổi qua lỗ hở.
  Danh sách bị chặn là **đúng danh sách protected paths hiện hành** (mặc định
  `agentics/ configs/ accounts/ scripts/ .github/` + credentials
  `accounts/ configs/secrets.yaml` không bao giờ miễn), và hook được sinh từ
  cùng nguồn với các tầng trong-process (`PROTECTED_PATHS` / lab.yaml) — mở
  rộng danh sách một nơi là mở rộng ở mọi tầng, không lệch pha.
- Merge hợp lệ của Integrator đi qua với env của Integrator.

**Lưu ý thiết kế:** exemption theo task của owner (`allow-protected`) có hiệu
lực ở các tầng trong-process; ở tầng git, commit agent chạm protected path
vẫn bị chặn — phần work đó đến origin qua identity owner hoặc luồng merge
của Integrator. Đây là lựa chọn cố ý: tầng git chỉ tin identity, không đọc
DB exemption.

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

---

## P17 — Reviewer tái tạo bằng chứng: tự khai không phải là lời cuối

**Vấn đề:** P16 cho reviewer ĐỌC bảng evidence của tác giả, nhưng toàn bộ
bằng chứng vẫn là *tự khai* — một protocol buggy (hoặc một trạng thái
workspace thiếu may mắn) có thể ghi "confirmed" và MR đi qua trên lời của
chính tác giả.

**Chính sách:** khi review, reviewer **chạy lại bằng chứng của tác giả
trong workspace của chính mình** (một trust domain riêng), trên branch MR
thật:

1. **Exit-code re-run** — tối đa 3 lệnh `run_command` của tác giả được phục
   hồi từ evidence feed (`cmds_from_evidence`, chỉ nhận dòng
   `method="run_command"`) và chạy lại qua `run_checks_on_branch` (branch
   tạm `review/tmp`, luôn trả lại main sạch).
2. **Tái tạo protocol** — exit code không thể phán một claim định lượng
   (protocol luôn exit 0, criterion mới là verdict). Với các dòng protocol
   `confirmed` của tác giả, reviewer chạy lại TOÀN BỘ adjudication
   2-lần-chạy với criterion đóng băng trong ledger
   (`RepoWorkspace.run_protocols_on_branch`): script vẫn in
   AGENTICS_RESULT nhưng metric không đạt → REFUTED — trường hợp
   exit-code re-run không bao giờ bắt được.
3. **Evidence overrides model opinion** — re-run bị refuted (bất kể đường
   nào) hạ verdict approve → request_changes, **re-verdict dòng gốc** của
   tác giả (`set_evidence_verdict` + `verdict_by`), hạ hypothesis trong
   ledger xuống `refuted` (verified_knowledge ngưng coi claim chết là
   kiến thức chung), journal `reviewer_rerun_refuted`.
4. **Không block oan** — re-run inconclusive (timeout, guard, sandbox lệch
   trạng thái) chỉ ghi chú vào comment MR, không bao giờ bị coi là refuted.

Mọi re-run được ghi thành evidence `method="reviewer_rerun"` và digest
re-run nằm trong comment MR — việc cross-check tự nó cũng là kiến thức
chung (H3), owner thấy ngay MR nào đã được tái tạo kiểm chứng.

Test: `tests/test_reviewer_rerun.py`, `tests/test_p17_protocol_rerun.py`
(kể cả tái tạo thật trên RepoWorkspace: protocol chỉ tồn tại trên branch
MR, main không có — sau re-run main phải sạch như cũ).
Chi tiết: [VERIFICATION.md](VERIFICATION.md) mục 6.

---

## P20 — LLM budget & circuit breaker: agent chạy loạn là sự kiện có biên

**Vấn đề:** P1-P17 ngăn agent giẫm chân nhau về **code**, nhưng không thứ gì
giới hạn việc đốt **nguồn lực LLM dùng chung** (pool cookie/quota là hữu hạn).
Một agent kẹt vòng hallucination hoặc đập vào provider chết có thể rút cạn
pool của cả đàn vô hạn.

**Chính sách** (`agentics/policies/budget.py`, mọi lời gọi LLM đi qua 2 cửa):

1. **Ledger dùng chung** — mỗi lời gọi LLM ghi 1 dòng vào bảng `llm_usage`
   (agent, purpose, ok = provider trả lời thật hay lỗi, duration). Ledger tự
   trim như journal — chỉ cần các dòng gần đây.
2. **Cap theo pool, cửa sổ trượt 1 giờ** — cap là **toàn đàn** vì tài nguyên
   là toàn đàn (`LLM_MAX_CALLS_PER_HOUR`, mặc định 120, `0` = tắt). Hết cap,
   mọi agent bị chặn đến khi cửa sổ trượt qua — không ai "nhờn" quota của ai.
3. **Circuit breaker từng agent** — N lần lỗi provider liên tiếp
   (`LLM_FAIL_STREAK_LIMIT`, mặc định 5) mở cooldown
   (`LLM_BREAKER_COOLDOWN_S`, mặc định 900s): agent dừng gọi, journal
   `budget_breaker_open` (mỗi lần mở ghi đúng 1 lần) + decision. Một câu
   trả lời thật mở lại breaker tức thì.
4. **Chặn trước, ghi sau** — lời gọi bị chặn **không bao giờ đến provider**;
   mọi lời gọi (kể cả bị chặn) đều vào ledger. Lời gọi nghiên cứu bị chặn
   vẫn rơi vào **free-tools fallback** (không tốn một đồng budget nào) —
   owner vẫn nhận nghiên cứu khi pool cạn.
5. **Không phá đường lành** — context không có budget (test, code cũ) đi
   qua như cũ; policy không bao giờ raise.

Owner nhìn tổng thể: card "budget" trên dashboard (`/api/state`) và
`llm_usage` trong shared memory.

Test: `tests/test_budget_policy.py` (+10).

---

## P21 — Audit trail bất biến: hành động quan trọng sống lâu hơn journal

**Vấn đề:** journal vận hành trim còn 5000 dòng; evidence trim để giữ kích
thước. Sau một sự cố (hoặc một tháng chạy), hành động từng xảy ra — merge
gì, ai miễn protected path cho task nào, ai viết lại verdict bằng chứng nào —
không còn dấu vết để đối chiếu.

**Chính sách:** bảng `audit_log` là **append-only, không bao giờ trim, không
bao giờ sửa tại chỗ**. Chỉ những hành động "đáng kể" được ghi, và mỗi hook
bọc try/except — audit không bao giờ cản hành động nó ghi lại:

| Sự kiện | Điểm cắm |
|---|---|
| `merge` | LocalMergeQueue + GitHub MergeQueue (author, auto_merge) |
| `mr_reject` | 2 queue trên (pr/MR, lý do: guard, tier-2, conflict) |
| `evidence_reverdict` | `set_evidence_verdict` (ai phán lại bằng chứng nào) |
| `task_exempt_set/cleared` | `set_task_exempt` (miễn/chấm dứt protected path) |
| `workspace_pruned` | P22 retention (đã dọn workspace của ai) |

Owner tra cứu: `agentics audit [--event merge] [--limit 50]` — CLI không bao
giờ bỏ sót; dashboard `/api/state` mang `audit.total` + 8 dòng mới nhất.

Test: `tests/test_audit_trail.py` (+5, gồm merge thật qua queue local).

---

## P22 — Retention: agent chết để lại workspace sạch, không để lại nợ ổ đĩa

**Vấn đề:** lab chạy nhiều tháng; mỗi agent chết để lại `data/workspaces/<id>`
(một clone đầy đủ) trên đĩa mãi mãi. P6 đánh dấu chết, P13 nhả khóa — không
có gì dọn clone.

**Chính sách** (`agentics/policies/retention.py`, chạy trong node_scan sau
các sweep claims/locks/panel): xóa workspace **CHỈ khi** mọi điều kiện an toàn
thỏa đồng thời:

1. Agent **không còn trong heartbeat set** active (P6 đã tuyên bố chết);
2. Agent **không giữ claim / file lock** nào còn hiệu lực (P1/P13);
3. **Git tree sạch** (`git status --porcelain` rỗng) — mọi commit đã nằm trên
   bare repo dùng chung, clone chỉ là tài sản tái tạo được, không phải bản
   duy nhất;
4. **Idle** quá `WORKSPACE_RETENTION_MINUTES` (mặc định 240, `0` = tắt hẳn).

Tree **bẩn** bị bỏ qua và journal `workspace_retention_skip` — clone cũ là
chuyện khó chịu, kết quả chưa push bị mất là thất thoát thật. Mỗi lần dọn
được journal + **audit** (P21) + alert broadcast — owner thấy ngay đã dọn gì
và vì sao.

Test: `tests/test_retention.py` (+7: prune, active/holder/dirty/fresh được
giữ, kill-switch, audit).

## P19 — Peer panel: đánh giá lẫn nhau bằng bỏ phiếu, một lời phản đối là đủ

P5 chỉ cấm tự duyệt; P17 bắt reviewer tái tạo bằng chứng. Nhưng một reviewer
duy nhất vẫn là một ý kiến. P19 (cơ chế "đánh giá lẫn nhau" của owner) biến
mỗi MR thành một phiên bỏ phiếu của panel đồng nghiệp — kết quả chung quan
trọng hơn ý kiến riêng:

1. **Bỏ phiếu panel** — mọi MR có bảng `mr_panels`; reviewer hợp lệ (agent
   đang sống, không phải tác giả — P5 chặn cứng trong engine) vote
   `accept/reject` kèm lý do; một phiếu/vòng/reviewer, phiếu đầu giữ nguyên
   (audit trail). Panel đủ `PEER_PANEL_SIZE` phiếu accept (mặc định 2, đặt
   lớn = cả swarm) VÀ không có reject nào → MR `approved`.
2. **Một phiếu reject là quyết định** — bất kỳ agent nào không chấp nhận,
   MR lập tức vào vòng **bảo vệ (defense)**: các objection được ghi lại và
   thông báo cho toàn swarm; MR KHÔNG được merge trong lúc đó.
3. **Bảo vệ quan điểm một lần, kiểm tra cơ khí** — tác giả phải phản bác
   TỪNG objection (key theo id người phản đối) trong hạn
   `PEER_DEFENSE_TTL_MINUTES` (mặc định 30 phút), có thể kèm
   `extra_checks` chạy thật trong workspace của chính mình (evidence
   `method="defense_check"`). Thiếu một phản đối → coi như không bảo vệ
   được → **reject chốt (final)**. Thuả thua (`concede`) → reject chốt.
   Quá hạn không bảo vệ → reject chốt (sweep của Supervisor).
4. **Sau bảo vệ: người phản đối giữ quyền phủ quyết** — vòng 2 chỉ thuộc
   các reviewer từng reject. Vẫn còn một reject → MR **rejected final**:
   verdict `failed`, tools gỡ, task trả về hàng đợi `open` cho lượt thử
   mới (ai cũng có thể nhận), reason thành trace để swarm học (GEA).
   Objector mất heartbeat bị coi là rút phản đối — swarm không đợi người
   chết, nhưng phiếu reject đã bỏ thì không bao giờ bị xoá.
5. **Chống lách** — MR đang defense không thể `reopen_review` âm thầm;
   bỏ phiếu sau khi MR đã approved là vô hiệu (MR thuộc merge queue với
   các gate riêng); panel state là nguồn sự thật chung, dashboard hiển thị
   state + phiếu + đồng hồ bảo vệ cho owner.

Engine cơ khí (không LLM): `agentics/peer_review.py` — node graph:
`node_review` (vote), `node_defend` (bảo vệ, router ưu tiên cao nhất cho
mọi role), sweep trong `node_scan`.

**Đường dây trực tiếp của owner trên terminal** — `agentics panel`:
bảng mọi panel (state, vòng, đồng hồ bảo vệ còn lại, dòng kết luận);
`agentics panel --mr N` xem đầy đủ phiếu/objection/bản bảo vệ của một MR
(kể cả panel đã đóng); `agentics panel --stats` trả lời câu hỏi gốc của
cơ chế — nó có đang cải thiện kết quả chung không: tỉ lệ duyệt, vòng bảo
vệ cứu được bao nhiêu %, lý do reject phân loại theo reason prefix cố
định của engine (incomplete/unmoved/conceded/deadline), người phản đối
nhiều nhất. Hai query chỉ đọc: `SharedMemory.panels_overview()` và
`SharedMemory.panel_stats()` — engine và luật bỏ phiếu không đổi.

Test: `tests/test_peer_review.py` (18 kịch bản luật), 
`tests/test_peer_review_memory.py`, `tests/test_cli_panel.py` (lệnh owner),
`tests/test_nodes_integration.py`
(vote mở defense), `tests/test_dashboard_api.py` (owner nhìn thấy panel).

## P24bis — Lặp lại độc lập (independent replication): claim 'supported' phải sống sót qua tay một agent KHÁC

Một verdict 'supported' từng chỉ dựa trên MỘT agent chạy protocol: gate
dedup (P16bis) khiến mọi agent khác CHỈ tái sử dụng kết quả đó, nên một
protocol lỗi (hoặc bị gian lận) đầu độc `verified_knowledge` vĩnh viễn.
Khoa học sửa điều này bằng replication — và replication phải ĐỘC LẬP:
tác giả không thể tự xác nhận claim của chính mình (tinh thần P5, áp cho
nghiên cứu).

Bốn mảnh ghép trên CÙNG sổ cái (không tạo đường chấm điểm thứ hai):

1. **Sổ cái chính tắc (P24)** — `hypotheses.replication` +
   `hypotheses_needing_replication(agent)`: hàng đợi oldest-first, loại
   tác giả, ngân sách thử lại 2 lần cho run flaky;
   `record_replication(hyp_id, outcome)` ghi kết quả một lần/claim.
2. **Provenance (P24bis)** — bảng `replications`:
   AI lặp lại, đo được bao nhiêu, mỗi run thế nào, mất bao lâu — mỗi dòng
   gắn với (evidence, replicator) duy nhất; một peer thứ hai được phép ghi
   dòng của mình (đúng nghĩa ngân sách thử lại).
3. **Engine** — `agentics/verify/replication.py`: peer dựng lại protocol
   trong workspace CỦA MÌNH (không LLM), chạy đủ gate 2-run
   (`verify_metrics.run_protocol`), máy chấm theo tiêu chí đóng băng.
   Refuted → mở lại verdict evidence gốc + hạ claim xuống `refuted`
   (mẫu P17); inconclusive → chỉ ghi chú, KHÔNG BAO GIỜ bác bỏ sai.
4. **Chu kỳ sống** — `node_replicate` (route REVIEWER, ưu tiên sau
   defense P19, trước review thường; knob `REPLICATION_LIMIT`, 0 = tắt) +
   đường piggyback của researcher trong `node_verify`. Hai đường cùng ăn
   một selector → đàn hội tụ về MỘT kết quả lặp lại/claim.

Tiêu dùng (H3): `verified_knowledge()` gắn cờ `replicated` — prompt của
mọi agent phân biệt `SUPPORTED` (tác giả xác nhận) với `SUPPORTED+`
(independently replicated — loại mạnh nhất); dashboard có panel riêng
(đếm + provenance); journal phát `replication_confirmed/_refuted/
_inconclusive/_blocked`.

Test: `tests/test_replication_memory.py`, `tests/test_replication_engine.py`,
`tests/test_node_replicate.py`, `tests/test_replication_surfaces.py`
(sổ cái chính tắc của P24: `tests/test_replication_ledger.py` — không đổi).
