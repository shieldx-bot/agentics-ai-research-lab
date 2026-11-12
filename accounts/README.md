# accounts/ — cookie của các agent thread

Mỗi file JSON trong thư mục này = **một luồng agent** (= một tài khoản Perplexity).

## Đặt file

- Tên file: `account-01.json`, `account-02.json`, ... (thứ tự = thứ tự gán vai trong
  `configs/lab.yaml → role_map`).
- Nội dung: export cookie từ trình duyệt bằng extension **Cookie-Editor → Export**
  (mảng các object `{name, value, domain, ...}`) — đúng định dạng như file mẫu.
- File được `.gitignore` toàn bộ (`accounts/*.json`) — **không bao giờ commit**.

## Chuẩn hoá

```bash
python scripts/setup_accounts.py
```

Script sẽ: tìm mọi file cookie trong thư mục, lọc chỉ giữ cookie domain
`perplexity.ai`, đổi tên thành `account-01.json` ... và báo cáo cookie nào còn hạn.

## Kiểm tra một account còn "sống"

```bash
python -m agentics.cli test-llm --account 1
```
