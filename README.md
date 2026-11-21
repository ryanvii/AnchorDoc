# AnchorDoc

**Zero-index-cost, vectorless document auditing engine for 100–600 page PDFs and high-stakes compliance documents.**

> 零建索引成本的无向量长文档审计引擎：50ms 构建全文结构锚点图，融合印刷目录跟页、两步缺项补看、合并表格还原、逐字引文校验与确定性裁决。

---

## Why AnchorDoc?

| Approach | Indexing Cost (250-page PDF) | Cross-Page Tables & TOCs | Zero-Ungrounded-Pass Audit |
|---|---|---|---|
| Flat Chunk Vector RAG | Medium (embedding DB) | Breaks tables & misses absence | Weak (hallucinates on Top-K misses) |
| Long-Context Direct | Zero | Attention dilution & over-inference | Expensive tokens, lower neutral recall |
| Full-Book Tree RAG (e.g. PageIndex) | High (dozens–hundreds of LLM calls) | Strong navigation | High upfront indexing latency/cost |
| **AnchorDoc** | **Zero LLM calls (~50ms)** | **AnchorMap + Printed-TOC G7 + OOXML Grid** | **Quote Gate + Python Calc + Conservative Abstain** |

### Verified Pilot Benchmarks (Same Model: `deepseek-v4-flash`, `temperature=0`)

- **FinanceBench (20 real SEC 10-K PDFs, 3,499 pages, 64 official questions — Dev-8 + Blind-12)**
  - Page Retrieval: **86.0% Hit@5 / 82.8% AllGold@5** (**92.0% / 88.0% on Blind-12 unseen filings**) vs Chunk-BM25 **21.9% / 18.7%** (~4x) — with **0 indexing LLM calls**.
  - Numerical Statement Subset (`metrics-generated`): **100.0% AllGold@5** & **92.3% End-to-End Accuracy** (ChunkRAG: 23.1%) via verbatim extraction + deterministic Python arithmetic.
- **ContractNLI-Long (150 full legal contracts >= 10k chars — Dev-60 + Blind-90)**
  - Structure-aware SpanCard + Two-Step Gap-Followup achieves **81.3% Accuracy / 81.2% Macro-F1**, outperforming full **Long-Context Direct (77.3% / 76.9%)** while using **26% fewer tokens**.

---

## Architecture

- `anchordoc/docbase`: Apache-2.0 safe 1-based `PageModel` (`pypdfium2` default, optional `pymupdf` accelerator) + stdlib OOXML merged-cell table grid (`w:vMerge` + `w:gridSpan`).
- `anchordoc/index`: Zero-LLM Structural `AnchorMap`, Rule `PageCard`, Printed-TOC (`G7`) detector with printed-to-physical page offset calibration, and `SpanCard` for long prose contracts.
- `anchordoc/retrieve`: Dual BM25-RRF + `C3` query expansion + `B42` two-step gap-followup + monotonic fusion with full layer-by-layer trace.
- `anchordoc/extract`: Line-anchored `[sNNNN]` requirement/contract extractor with automatic verbatim quote-repair loop and candidate coverage conservation.
- `anchordoc/adjudicate`: Fact-decoupled adjudicator with programmatic verbatim `QuoteGate`, safe AST Python arithmetic evaluator, and conservative abstention truth table.
- `anchordoc/report`: Auditable JSON, Markdown, and standalone HTML trace reports.
- `adapters/`: Thin domain plugins (`finance_10k`, `legal_nda`, `procurement_demo`).

---

## Quickstart

```python
from anchordoc import CachedLLM, PdfDocument, AnchorRetriever, FactAdjudicator

llm = CachedLLM(cache_dir="./cache")
doc = PdfDocument.from_pdf("annual_report_10k.pdf")

retriever = AnchorRetriever(doc, llm=llm)
hit = retriever.retrieve("What is FY2022 capital expenditure in USD millions?", top_k=5)

adjudicator = FactAdjudicator(llm=llm)
verdict = adjudicator.answer_numerical(doc, "What is FY2022 capital expenditure in USD millions?", hit.pages)
print(verdict.status, verdict.answer, verdict.cited_pages)
```

---

## Roadmap

See [docs/ROADMAP_MASTER_PLAN.md](docs/ROADMAP_MASTER_PLAN.md) for the full 4-phase roadmap (`v0.1` Core Consolidation -> `v0.5` Full Public Benchmarks -> `v0.9` Developer UX -> `v1.0` Public Release).

## License

Apache-2.0. See [LICENSE](LICENSE).
