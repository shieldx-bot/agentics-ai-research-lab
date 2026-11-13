# PRODUCTION — triển khai lab ở môi trường thật

Bản hoà hợp của 22 policy (P1-P22, xem [POLICIES.md](POLICIES.md)). Trang này
là checklist cho owner: bật gì, đặt giá trị nào, kiểm gì trước khi để đàn
agent chạy không giám sát dài ngày.

## 1. Biến môi trường vận hành (tất cả đã đọc trong `Settings`)

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `LLM_MAX_CALLS_PER_HOUR` | `120` | P20: cap pool LLM 1 giờ (0 = tắt — chỉ dùng khi debug) |
| `LLM_FAIL_STREAK_LIMIT` | `5` | P20: lỗi provider liên tiếp mở breaker |
| `LLM_BREAKER_COOLDOWN_S` | `900` | P20: thời gian breaker mở |
| `WORKSPACE_RETENTION_MINUTES` | `240` | P22: workspace chết được dọn sau thời gian idle này (0 = tắt) |
| `AUTO_MERGE` | `false` | P4: chỉ bật khi branch protection + CI + secret scanning đã bật ở origin |
| `CLAIM_TTL_MINUTES` | `45` | P1: TTL claim |
| `FILE_LOCK_TTL_MINUTES` | `30` | P13: TTL khóa file |
| `INTEGRITY_INTERVAL_S` | `20` | P10 tầng 4b: chu kỳ watcher (0 = tắt) |
| `PEER_PANEL_SIZE` | `2` | P18: số reviewer đồng thuận trước khi MR qua |
| `PEER_DEFENSE_TTL_MINUTES` | `30` | P18: hạn tác giả bảo vệ MR bị phản đối |

Khuyến nghị production: giữ mặc định; hạ `LLM_MAX_CALLS_PER_HOUR` nếu pool
cookie nhỏ; bật retention (mặc định đã bật).

## 2. Bảo mật — những gì KHÔNG được đổi

- `accounts/`, `configs/secrets.yaml`, credentials: **không bao giờ miễn**
  với mọi exemption (P10); runner là host guardrail — mã không tin cậy chạy
  trong **rootless container** (P2, phần "Giới hạn bảo mật").
- `agentics/ configs/ accounts/ scripts/ .github/` vẫn là protected paths
  (4 tầng guard + pre-receive hook + watcher 4a/4b).
- Lỗi subprocess/audit: hook audit (P21) không bao giờ cản merge — hành động
  được ghi hoặc không, không bao giờ được hủy bởi lỗi audit.

## 3. Quan sát (owner có gì trong tay)

- **Dashboard** `/api/state`: agents, tasks, MR/panel (P18), evidence (P16),
  hypotheses (P16bis), audit (`P21`), budget dial (`P20`), alerts.
- **CLI**: `agentics watch` (terminal), `agentics evidence`, `agentics
  hypotheses`, `agentics audit`, `agentics verify` (integrity), `agentics
  mr list/approve/merge`, `agentics metrics`.
- **Journal events mới cần biết**: `budget_blocked`, `budget_breaker_open`,
  `workspace_pruned`, `workspace_retention_skip` — tất cả cũng vào audit
  trail khi là hành động dọn dẹp.

## 4. Chạy dài ngày — những gì tự chăm sóc

1. **Pool LLM**: đốt quá 1 giờ bị chặn tự nhiên, nghiên cứu rơi free-tools;
   breaker giữ agent lỗi khỏi đập provider chết (P20).
2. **Đĩa**: workspace của agent chết, không giữ gì, tree sạch → dọn tự động
   sau idle window; tree bẩn được giữ và báo rõ lý do (P22).
3. **Bộ nhớ SQLite**: journal/evidence/llm_usage tự trim; audit_log tăng
   chậm (chỉ sự kiện đáng kể) — sao lưu `data/*.db` định kỳ nếu cần đối
   chiếu lâu dài.
4. **Lý lịch MR**: merge/reject/exempt/reverdict đều có dòng audit bất biến
   — đối chiếu sau sự cố không phụ thuộc journal đã trim (P21).

## 5. Checklist trước khi go-live

- [ ] `python -m pytest -q` — full suite xanh trên máy đích.
- [ ] `python -m agentics.cli verify` — integrity baseline khớp.
- [ ] `configs/lab.yaml` có `protected_paths` + `writable_roots` đúng vùng.
- [ ] `AUTO_MERGE=false` cho đến khi branch protection + CI + secret
      scanning bật ở origin.
- [ ] Chiếm slot tài nguyên: `LLM_MAX_CALLS_PER_HOUR` theo dung lượng pool
      cookie thật (người dùng thử đã nói quota hữu hạn).
- [ ] Chạy `scripts/local_mode_demo.py` trong sandbox riêng để thấy trọn
      vòng claim → verify → panel → merge trước khi chạy thật.
- [ ] Đặt lịch sao lưu `data/` (SQLite + bare repo) — audit trail là thứ
      KHÔNG thể tạo lại từ anywhere.
