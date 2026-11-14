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
| **Semantic Scholar** | `api.semanticscholar.org/graph/v1/paper/search` | bài báo + citationCount + DOI/arXiv id (`externalIds`) | pool anonymous riêng — budget độc lập với OpenAlex |
| **Europe PMC** | `ebi.ac.uk/europepmc/webservices/rest/search` | bài báo + abstract đầy đủ + citedByCount, preprint | GET thường, budget dồi dào |
| **GitHub** | `api.github.com/search/repositories` | repo thật: stars, language, license, last push | 1 call / 6.5 s (lock), 10 req/min anonymous |
| **PyPI** | `pypi.org/pypi/<name>/json` | package tồn tại? version mới nhất thật, requires_python | GET thường |
| **HuggingFace** | `huggingface.co/api/{datasets,models}` | dataset + model công khai (downloads, likes, task) | plain GET |
| **Wikipedia** | `en.wikipedia.org/api/rest_v1` | định nghĩa thuật ngữ (grounding, KHÔNG phải nguồn claim) | plain GET |
| **DuckDuckGo** | `html.duckduckgo.com/html/` | web search tổng quát | best-effort, markup không chính thức — có thể hỏng bất cứ lúc nào |

### 1.1 Redundancy học thuật: ba budget độc lập, một pool cạn vẫn còn finding

`search_all` chạy **arXiv + OpenAlex + Semantic Scholar song song** (ba budget
rate-limit độc lập), rồi chỉ lấp chỗ trống từ **Crossref** và **Europe PMC**
khi vẫn còn thiếu. Như vậy một pool bị cạn (OpenAlex 429 trên IP datacenter —
hiện tượng đã gặp thật, §5) không thể làm rỗng một finding.

### 1.2 Verification toolkit: từ literature → protocol chạy được

`verification_toolkit(query)` gom 3 nguồn "thực vật cho kiểm chứng" thành 1
call có cache:

- **GitHub repos** — ý tưởng này đã có code chưa? Repo tham chiếu để clone
  vào workspace sandbox chạy thử (phục vụ trực tiếp giả thuyết H1/H2).
- **PyPI pins** — package có tồn tại không + version mới nhất thật để pin vào
  `verify_cmds` (tái lập P16: không pin version bịa).
- **HF datasets/models** — dữ liệu và pretrained weights thật để feed protocol.

`fallback_research` (LLM chết) giờ ghép toolkit vào finding: answer chứa phần
"Reference implementations & data" — node plan/verify nhận được đường dẫn code
và dataset thật, không chỉ danh mục sách. Toolkit không tự tạo finding:
**không có paper thì vẫn trả error state** như hợp đồng cũ của node_research.

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
python -m agentics.cli papers "VNF placement" --source arxiv           # chỉ arXiv
python -m agentics.cli papers "VNF placement" --source openalex        # chỉ OpenAlex
python -m agentics.cli papers "VNF placement" --source crossref        # chỉ Crossref
python -m agentics.cli papers "VNF placement" --source semanticscholar # budget riêng, OpenAlex 429 vẫn sống
python -m agentics.cli papers "VNF placement" --source europepmc       # abstract đầy đủ + preprint
python -m agentics.cli papers --doi 10.1109/netsoft48620.2020.9165348  # BibTeX thật
python -m agentics.cli papers "intrusion detection" --datasets         # dataset HF
python -m agentics.cli papers "bert" --models                          # pretrained weights HF
python -m agentics.cli papers "nfv scheduler" --github                 # repo tham chiếu để clone
python -m agentics.cli papers --pypi scikit-learn                      # version thật để pin verify_cmds
python -m agentics.cli papers "network function virtualization" --define  # định nghĩa
python -m agentics.cli papers "NFV security" --web                     # DuckDuckGo
```

Output scholarly mặc định kèm BibTeX thật cho toàn bộ kết quả — copy thẳng
vào report. `--github`/`--pypi`/`--models` là 3 chân của verification toolkit
để owner tự tay kiểm tra trước khi nhét vào protocol.

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
  10 s) + TTL cache 10 phút. Từ batch này OpenAlex không còn là điểm hỏng
  duy nhất: **Semantic Scholar chạy budget riêng song song**, Crossref và
  Europe PMC lấp chỗ trống theo tầng — live test cho thấy khi OpenAlex bị
  rate-limit xuyên suốt, fallback vẫn trả citation + BibTeX thật.
- **Semantic Scholar anonymous dùng pool chia sẻ toàn cầu** → 429 thỉnh
  thoảng là chuyện bình thường; retry 1 lần + graceful `[]` đã xử lý, các
  nguồn còn lại bù đắp.
- **GitHub search anonymous ~10 call/phút** → module pace 1 call / 6.5 s
  process-wide (giống arXiv) + TTL cache; chỉ dùng khi cần tìm repo, không
  nằm trong đường dẫn nóng.
- Đặt `FREE_TOOLS_MAILTO` khi chạy production để vào polite pool của
  OpenAlex/Crossref.
- arXiv yêu cầu tối đa 1 call / 3 s — module đã có lock process-wide, và mọi
  arXiv id của một lần enrichment được gom thành 1 call duy nhất.
- Mỗi nguồn fail → trả về rỗng/None, **không bao giờ phá chu kỳ** (mọi exception
  được bắt ở `_request` và ở `_free_tools_fallback`).
- Wikipedia chỉ dùng để hiểu thuật ngữ; claim định lượng vẫn phải đi qua
  P16 (chạy code + đo số) — tool này không thay đổi quy tắc đó. Toolkit chỉ
  gợi ý repo/dataset/model — **phải kiểm tra license + availability trước khi
  pin vào protocol**, phần answer đã ghi chú rõ.

## 6. Tests

`tests/test_free_tools.py` — 46 test offline (không mạng): parser Atom,
abstract inverted-index của OpenAlex, field Crossref + strip JATS, fields
Semantic Scholar (externalIds DOI > arXiv, blank-title drop) và Europe PMC
(authorString, strip markup, URL article khi không DOI), batching `id_list`
1 call, dedup phiên bản cùng bài báo, fallback legacy, retry 429/503, chuỗi
fill 3 tầng của `search_all` (S2 song song → Crossref → EPMC), dedup + fill,
shape của `fallback_research` (kèm toolkit + không nhét toolkit vào finding
rỗng), GitHub repos (fields, pace, graceful), PyPI (pin thật, miss cached),
HF models, toolkit bundle + `_toolkit_text`, config knobs và wiring
`node_research` (fallback khi LLM chết, BibTeX grounded khi LLM sống,
guarantee degrade-to-error của người đi trước khi tool tắt).
