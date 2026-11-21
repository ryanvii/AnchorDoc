<div align="center">

# ⚓ AnchorDoc

**Zero-Index-Cost, Vectorless Verifiable Document Auditing Engine for 100–600 Page PDFs**

[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab.svg)](pyproject.toml)
[![FinanceBench](https://img.shields.io/badge/FinanceBench%20150%2F150-78.7%25%20E2E%20%7C%2090%25%20Metrics-16a34a.svg)](benchmarks/LEADERBOARD.md)
[![MMLongBench](https://img.shields.io/badge/MMLongBench--Doc-95.4%25%20Hit%405-2563eb.svg)](benchmarks/LEADERBOARD.md)

**English** | [简体中文](README_ZH.md)

</div>

---

**AnchorDoc** replaces brittle chunk-based Vector RAG and expensive upfront LLM tree-indexing with a **50ms Zero-LLM Structural Anchor Map**, **Printed-TOC offset calibration**, **Selective VisionPageCards** (triggered *only* on low-text/scanned pages), and **Fact-Decoupled Deterministic Adjudication** (verbatim `QuoteGate` + safe Python AST math + conservative human escalation).

![AnchorDoc Architecture](assets/architecture.svg)

---

## Why AnchorDoc?

| Approach | Indexing Cost (250p PDF) | FinanceBench 150/150 E2E* | Multi-Page Recall (AllGold@5) | Scanned / Slide PDFs |
|---|---:|---:|---:|---|
| Flat Chunk BM25 | `0.3s` · `0` LLM calls | `33.3%` | `32.0%` | Blind |
| Dense Vector RAG (`BGE-small` ONNX) | `47.6s` (neural embed) | `39.3%` | `48.7%` | Blind |
| LLM Tree-Index (PageIndex-style 12p win) | `11.4s` · `19.2` calls · `33k` tok | `35.9%` *(39q subset)* | `43.6%` *(39q subset)* | Requires VLM per page |
| **⚓ AnchorDoc `v0.5`** | **`0.32s` · `0` LLM calls** | **`78.7%` (`90.0%` numerical)** | **`81.3%` (`100%` numerical)** | **Selective `VisionPageCard` (`95.4%` Hit@5)** |

*\*All methods benchmarked on the exact same base model (`deepseek-v4-flash`, `temperature=0`) with Apache-2.0 `pypdfium2`.*

![AnchorDoc Public Benchmarks](assets/benchmark_comparison.svg)

---

## Verified Public Benchmarks (401 Official Questions Across 3 Modalities)

- **FinanceBench (Full Official 150 / 150, 84 SEC PDFs, 12,013 pages)**
  - **Indexing**: **26.7s total (`0.32s/PDF`), `0` LLM calls** (149× faster than CPU Dense BGE embedding, 91× faster than upfront LLM tree indexing).
  - **Retrieval & Ablation**: **84.7% Hit@5 / 81.3% AllGold@5** (MRR `0.7719`) vs Dense BGE **58.0% / 48.7%** and Chunk-BM25 **35.3% / 32.0%**. Full ablation proves `Zero-LLM AnchorMap + G7` alone adds **+18.0pp AllGold@5**.
  - **End-to-End QA**: **78.7% overall** (2.0× Dense BGE `39.3%`, 2.4× ChunkRAG `33.3%`). On all 50 hard numerical statement questions (`metrics-generated`), AnchorDoc achieves **100.0% AllGold@5** and **90.0% E2E accuracy** via verbatim extraction + Python AST arithmetic.
- **MMLongBench-Doc (NeurIPS 2024 — 101 questions, 7 non-financial multimodal PDFs incl. 3 pure-image slide decks)**
  - **Selective `VisionPageCard`** skips 100% of text-rich digital pages (`78/170`) and indexes only low-text/scanned pages (`92/170`), jumping answerable **Hit@5 from `63.1%` → `95.4%` (62/65)** vs Chunk-BM25 `44.6%`.
  - **End-to-End**: **52.5% overall** (vs `36.6%` ChunkRAG), **53.1%** Text/Table (vs `21.9%`), **24.2%** Chart/Figure (vs `0.0%`), and **77.8%** Unanswerable Abstain accuracy (nearly halving unanswerable hallucination via strict Absence Gate).
- **ContractNLI-Long (EMNLP — 150 long legal NDAs `>= 10k` chars, balanced 50/50/50)**
  - Strictly Pareto-dominates stuffing the full contract into **Long-Context Direct** (`78.1%` vs `78.0%` Macro-F1, `80.3%` vs `78.5%` high-confidence accuracy, `14.0%` vs `16.0%` false-pass rate) while using **25% fewer prompt tokens**.

See [benchmarks/LEADERBOARD.md](benchmarks/LEADERBOARD.md) for complete tables, ablations, and one-command reproduction scripts.

---

## 60-Second Quickstart

```bash
git clone https://github.com/ryanvii/AnchorDoc.git
cd AnchorDoc
pip install -e .

# 1. Zero-LLM structural inspection (NO API key required — runs in ~0.2s!)
anchordoc outline annual_report_10k.pdf

# 2. Verifiable numerical/compliance QA + interactive two-pane HTML report (with embedded original PDF pages!)
export ANCHORDOC_LLM_BASE=https://api.openai.com/v1
export ANCHORDOC_LLM_MODEL=gpt-4o-mini
export ANCHORDOC_LLM_KEY=sk-...

anchordoc query annual_report_10k.pdf \
  "What is FY2023 gross profit margin?" \
  --html report.html

# 3. Audit a PDF against a JSON checklist (add --vision for scanned/chart PDFs)
anchordoc audit submission.pdf \
  --checklist examples/sample_checklist.json \
  --html audit_workbench.html
```

### Python SDK (5 Lines)

```python
from anchordoc import AnchorRetriever, CachedLLM, FactAdjudicator, PdfDocument, export_html

doc = PdfDocument.from_pdf("annual_report_10k.pdf")          # Apache-2.0 pypdfium2
llm = CachedLLM(cache_dir="./cache")                         # SHA256 cache + OFFLINE replay
hit = AnchorRetriever(doc, llm=llm).retrieve("FY2023 gross margin?", top_k=5)
verdict = FactAdjudicator(llm=llm).answer_numerical(doc, "FY2023 gross margin?", hit.pages)
export_html([verdict], "report.html", doc=doc)               # Standalone two-pane HTML workbench
```

---

## Core Modules & Domain Adapters

- `anchordoc/docbase`: Thread-safe Apache-2.0 1-based `PageModel` (`pypdfium2`) + stdlib OOXML merged-cell table grid (`w:vMerge` / `w:gridSpan` inheritance, zero false-blanks).
- `anchordoc/index`: `Zero-LLM AnchorMap`, Printed-TOC (`G7`) offset calibrator, `SelectiveVisionIndexer` (`VisionPageCard`), and `SpanCardIndex`.
- `anchordoc/retrieve`: Dual BM25-RRF + `C3` query expansion + `B42` two-step gap-followup + monotonic fusion with full layer-by-layer `RetrievalTrace`.
- `anchordoc/extract`: Line-anchored `[sNNNNN]` requirement extraction with automatic verbatim quote-repair loop.
- `anchordoc/adjudicate`: Fact-decoupled adjudicator with programmatic `QuoteGate`, safe Python AST arithmetic (`safe_calc`), unit-safe comparison (prevents `48 hours` vs `48 days` bugs), strict visual Absence Gate, and conservative truth table.
- `anchordoc/report`: JSON, Markdown, and zero-server interactive two-pane HTML trace workbench with embedded original PDF page previews.
- `adapters/`: Thin domain plugins — `finance_10k`, `legal_nda`, `multimodal_doc`, `procurement_demo`.

---

## Citation

If you use AnchorDoc or its reproducible benchmarks in your work, please cite via [CITATION.cff](CITATION.cff):

```bibtex
@software{wu2026anchordoc,
  author  = {Wu, HuiYang},
  title   = {AnchorDoc: Zero-Index-Cost, Vectorless Verifiable Document Auditing Engine},
  year    = {2026},
  version = {0.5.0},
  url     = {https://github.com/ryanvii/AnchorDoc},
  license = {Apache-2.0}
}
```

## License

Apache-2.0. See [LICENSE](LICENSE). Core dependencies are strictly Apache-2.0 / MIT / BSD compatible; copyleft accelerators (e.g., PyMuPDF) are optional extras only.
