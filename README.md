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

### Verified Public Benchmarks (Same Model: `deepseek-v4-flash`, `temperature=0`, Apache-2.0 `pypdfium2`)

- **FinanceBench — Full Official 150 / 150 Suite (84 real SEC PDFs, 12,013 pages)**
  - **Indexing**: **26.7s total (`0.32s/PDF`), `0` LLM calls**.
  - **Page Retrieval & Ablation**: **84.7% Hit@5 / 81.3% AllGold@5** (MRR `0.7719`) vs Chunk-BM25 **35.3% / 32.0%** (MRR `0.2613`). Ablation confirms **Zero-LLM AnchorMap + G7** alone drives **+11.4pp Hit@5** and **+18.0pp AllGold@5**.
  - **End-to-End Verifiable QA**: **78.7% overall accuracy** vs same-model ChunkRAG **33.3%** (**+45.4pp**).
  - **Numerical Financial Statements (`metrics-generated`, 50/50)**: **100.0% AllGold@5** & **90.0% End-to-End Accuracy** (ChunkRAG: 20.0%) via verbatim fact extraction + Python AST arithmetic.
- **ContractNLI-Long (150 full legal contracts >= 10k chars, balanced 50/50/50)**
  - **SpanCard + B42 Two-Step Gap-Followup + QuoteGate** Pareto-dominates full **Long-Context Direct** (`78.1%` vs `78.0%` Macro-F1, `80.3%` vs `78.5%` high-confidence accuracy, lower false-pass `14.0%` vs `16.0%`, and **25% fewer tokens**).
- **MMLongBench-Doc (101 official questions across 7 non-financial multimodal PDFs)**
  - **AnchorDoc-Multimodal** lifts retrieval Hit@5 to **63.1%** (vs Chunk-BM25 `44.6%`) and boosts Text/Table accuracy to **53.1%** (vs `21.9%`) and Chart/Figure accuracy to **12.1%** (vs `0.0%`).

See [benchmarks/LEADERBOARD.md](benchmarks/LEADERBOARD.md) for full ablations and reproduction commands.

---

## CLI & Interactive HTML Evidence Workbench

```bash
# 1. Inspect Zero-LLM Structural Anchor Map & G7 TOC offset (no API key required!)
python3 -m anchordoc.cli outline annual_report_10k.pdf

# 2. Ask a verifiable question & export a two-pane HTML report with embedded original PDF pages
python3 -m anchordoc.cli query annual_report_10k.pdf \
  "What is FY2023 gross profit margin?" --html report.html

# 3. Audit a PDF against a JSON checklist (with optional multimodal --vision)
python3 -m anchordoc.cli audit submission.pdf --checklist rules.json --html audit_report.html
```

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
