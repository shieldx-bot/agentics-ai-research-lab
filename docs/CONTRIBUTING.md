# CONTRIBUTING

Dự án này được thiết kế để **nhiều contributor (người và AI) phát triển song song
mà ít xung đột**. Hãy tuân theo các quy ước sau — chúng giống hệt các policy mà
agents phải tuân thủ (xem `docs/POLICIES.md`).

## Quy ước nhánh
- Nhánh tính năng: `contrib/<tên-bạn>/<mô-tả-ngắn>`
- Nhánh agent: `agent/<agent-id>/issue-<n>-<slug>` (đừng bao giờ dùng tiền tố `agent/`
  nếu bạn là người — để tránh lẫn với nhánh bot).

## Quy trình
1. Mở issue mô tả việc (hoặc chọn issue trống, chưa có label `claimed/*`).
2. Comment `/claim` + tự gắn label `claimed/<tên-bạn>`.
3. Làm việc trên nhánh riêng, commit format: `contrib(<tên-bạn>): <mô tả> (#<issue>)`.
4. Mở PR. PR phải được ít nhất 1 người/agent khác approve.
5. Maintainer/Integrator merge (rebase-merge).

## Vùng module — tránh đụng độ
| Vùng | Người chịu trách nhiệm chính |
|---|---|
| `agentics/llm/` | team LLM adapters |
| `agentics/memory/` | team memory |
| `agentics/github_ops/` | team GitHub automation |
| `agentics/graph/` + `agents/` | team LangGraph |
| `docs/` | ai cũng được, PR review là bắt buộc |

Nếu thay đổi của bạn đụng vùng của team khác: trao đổi qua issue trước, không
tự ý refactor code của vùng khác trong cùng PR.

## Môi trường dev
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pytest            # unit tests (không cần token/cookie)
ruff check .
```

## Bảo mật
Không bao giờ commit token/cookie. Nếu vô tình lộ: thu hồi ngay và báo maintainer.
