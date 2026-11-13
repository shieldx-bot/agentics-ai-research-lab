# FREE TOOLS — nguồn nghiên cứu miễn phí, không cần API key

> Bộ tool nghiên cứu **keyless** (không tài khoản, không token) dùng chính API
> của nhà xuất bản. Metadata (tác giả, tiêu đề, năm, tóm tắt, số trích dẫn) là
> **dữ liệu thật được fetch về** — đúng tinh thần P16: kết luận phải dựa trên
> bằng chứng kiểm chứng được, không phải lời của LLM.

Module: `agentics/research/free_tools.py` · Wire vào graph: `agentics/graph/nodes.py`
(`node_research`) · Cấu hình: `FREE_TOOLS_*` trong `.env` / `configs/lab.yaml`.

## 1. Các nguồn

| Nguồn | Endpoint | Dữ liệu trả về | Lưu ý lịch sự (rate limit) |
|---|---|---|---|
| **arXiv** | `export.arxiv.org/api/query` | title, tác giả, năm, abstract, category, DOI | 1 call / 3 s (process-wide lock), gom id vào 1 `id_list` |
| **OpenAlex** | `api.openalex.org/works` | works học thuật, DOI, venue, citation count | polite-pool qua `mailto`; 429 → retry 1 lần theo `Retry-After` |
| **Crossref** | `api.crossref.org/works` + `doi.org` (content negotiation) | metadata DOI, BibTeX thật | `mailto` polite pool |
| **HuggingFace** | `huggingface.co/api/datasets` | dataset công khai (downloads, likes) | plain GET |
| **Wikipedia** | `en.wikipedia.org/api/rest_v1` | định nghĩa thuật ngữ (grounding, KHÔNG phải nguồn claim) | plain GET |
| **DuckDuckGo** | `html.duckduckgo.com/html/` | web search tổng quát | best-effort, markup không chính thức — có thể hỏng bất cứ lúc nào |

## 2. Hai điểm tích hợp trong chu kỳ nghiên cứu

### 2.1 Fallback khi LLM chết (`node_research`)

`node_research` trước đây phụ thuộc 100% cookie LLM (Perplexity/Qwen): cookie
chết → chu kỳ chết. Giờ khi LLM fail/trả về rỗng, node gọi
`free_tools.fallback_research()` — finding được dựng từ metadata thật của
arXiv + OpenAlex (+ Crossref khi cần) và đi vào **cùng một đường** lưu shared
memory / vector store / broadcast như finding từ LLM:

```
LLM ok?  ──yes──> search_papers (Perplexity academic) ──> grounded BibTeX ─┐
    │no                                                                   ├─> _finish_research
    └──> free_tools.fallback_research (arXiv/OpenAlex/Crossref) ──────────┘
```

- `ctx.meta["research_source"]` = `"llm"` | `"free-tools"` — dashboard/CLI thấy
  được nguồn gốc của mỗi finding.
- Free tools bị tắt (`FREE_TOOLS_ENABLED=0`) hoặc mọi nguồn rỗng → giữ nguyên
  hành vi cũ: error state, chu kỳ dừng sạch (được test neo trong
  `tests/test_free_tools.py` + `tests/test_prod_fixes.py`).

### 2.2 BibTeX từ metadata thật (`enriched_bibtex`)

Trước đây BibTeX được tạo bằng cách scrape thẻ `<title>` của URL — thường sai
thiếu (không tác giả, không năm). Giờ:

- URL arXiv → **1 call** `id_list` gom mọi arXiv id → BibTeX với tác giả thật,
  năm thật, `eprint` + `archivePrefix`.
- URL DOI → Crossref (song song) → BibTeX với tạp chí/tiên đề thật, `doi`.
- URL khác → giữ hành vi legacy (key deterministic + scrape title, chạy trong
  thread để không block event loop). Metadata không biết thì **bỏ trống, không
  bịa** — đúng nguyên tắc của `research/citations.py` gốc.

## 3. Dùng trực tiếp từ CLI (không cần cookie LLM)

```bash
python -m agentics.cli papers "machine learning network function virtualization"
python -m agentics.cli papers "VNF placement" --source arxiv      # chỉ arXiv
python -m agentics.cli papers "VNF placement" --source openalex   # chỉ OpenAlex
python -m agentics.cli papers "VNF placement" --source crossref   # chỉ Crossref
python -m agentics.cli papers --doi 10.1109/netsoft48620.2020.9165348   # BibTeX thật
python -m agentics.cli papers "intrusion detection" --datasets    # dataset HF
python -m agentics.cli papers "network function virtualization" --define  # định nghĩa
python -m agentics.cli papers "NFV security" --web                # DuckDuckGo
```

Output scholarly mặc định kèm BibTeX thật cho toàn bộ kết quả — copy thẳng
vào report.

## 4. Cấu hình

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `FREE_TOOLS_ENABLED` | `1` | Tắt/bật toàn bộ (khi tắt: `node_research` giữ hành vi degrade-to-error cũ) |
| `FREE_TOOLS_TIMEOUT` | `15` | Timeout mỗi HTTP call (giây) |
| `FREE_TOOLS_MAILTO` | (rỗng) | Email cho polite-pool của OpenAlex/Crossref (không bắt buộc, nhưng vào tier rate-limit dễ chịu hơn) |

`configs/lab.yaml` cũng nhận 3 key tương tự (yaml thắng env).

## 5. Khuyến nghị vận hành (đã kiểm chứng live 2026-11-13)

- **IP datacenter dùng chung dễ cạn ngân sách anonymous của OpenAlex** (HTTP
  429 "Insufficient budget"). Đã có retry 1 lần theo `Retry-After` (giới hạn
  10 s) + TTL cache 10 phút; khi OpenAlex vẫn 429, Crossref tự lấp chỗ trống —
  live test: `fallback_research` trả về 8 citation thật + 8 BibTeX dù OpenAlex
  bị rate-limit xuyên suốt.
- Đặt `FREE_TOOLS_MAILTO` khi chạy production để vào polite pool.
- arXiv yêu cầu tối đa 1 call / 3 s — module đã có lock process-wide, và mọi
  arXiv id của một lần enrichment được gom thành 1 call duy nhất.
- Mỗi nguồn fail → trả về rỗng/None, **không bao giờ phá chu kỳ** (mọi exception
  được bắt ở `_request` và ở `_free_tools_fallback`).
- Wikipedia chỉ dùng để hiểu thuật ngữ; claim định lượng vẫn phải đi qua
  P16 (chạy code + đo số) — tool này không thay đổi quy tắc đó.

## 6. Tests

`tests/test_free_tools.py` — 30 test offline (không mạng): parser Atom,
abstract inverted-index của OpenAlex, field Crossref + strip JATS, batching
`id_list` 1 call, dedup phiên bản cùng bài báo, fallback legacy, retry 429/503,
dedup + fill của `search_all`, shape của `fallback_research`, HF/Wikipedia/DDG,
config knobs và wiring `node_research` (fallback khi LLM chết, BibTeX grounded
khi LLM sống, guarantee degrade-to-error của người đi trước khi tool tắt).
