# AnchorDoc Official Public Benchmark Leaderboard

All evaluations below use the **exact same base model (`deepseek-v4-flash`, `temperature=0`)**, **Apache-2.0 `pypdfium2`** PDF parsing, deterministic SHA256 disk caching, and **zero domain-specific regex hardcoding** inside `anchordoc/`.

---

## 1. FinanceBench — Full Official Open-Source Suite (150 / 150 Questions, 84 SEC PDFs, 12,013 Pages)

- **Corpus**: All 84 real SEC 10-K / 10-Q / 8-K filings (**12,013 total physical pages**).
- **Indexing Cost**: **26.7 seconds total (`0.32s / PDF`), `0` LLM calls**.
- **Reproduce**: `python3 benchmarks/run_financebench_full.py --limit 150 --backend pypdfium2`

### 1.1 Page-Level Retrieval & Full 150-Question Ablation Study

| Configuration | Indexing LLM Calls | Hit@1 | Hit@3 | Hit@5 | **AllGold@5** | MRR@10 | Avg Retrieval Calls / Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| `1. Chunk-BM25` (flat 1000-char chunks) | 0 | 18.7% | 29.3% | 35.3% | 32.0% | 0.2613 | 0 / 0 |
| `2. RawPage-BM25` (full-page BM25) | 0 | 17.3% | 26.7% | 36.7% | 33.3% | 0.2533 | 0 / 0 |
| `3. PageCard-RRF` (zero-LLM Dual BM25-RRF) | 0 | 14.7% | 32.0% | 40.0% | 32.7% | 0.2492 | 0 / 0 |
| `4. w/o AnchorMap` (`C3 + B42` only) | 0 | 62.0% | 72.7% | 73.3% | 63.3% | 0.6736 | 3.36 / 5,355 |
| `5. w/o Gap-Followup` (`AnchorMap + G7 + C3`) | 0 | **71.3%** | **83.3%** | **84.7%** | **81.3%** | **0.7730** | **3.99 / 9,133** |
| **`6. AnchorDoc-Full` (`AnchorMap + G7 + C3 + B42`)** | **0** | **71.3%** | **83.3%** | **84.7%** | **81.3%** | **0.7719** | 4.35 / 9,998 |

> **Ablation Takeaway**: Adding **Zero-LLM Structural AnchorMap + G7 Printed-TOC calibration** alone lifts **Hit@5 by +11.4pp (`73.3% -> 84.7%`)** and **AllGold@5 (multi-page evidence completeness) by +18.0pp (`63.3% -> 81.3%`)** across all 150 SEC filings — at **zero indexing LLM cost**.

### 1.2 End-to-End Verifiable QA (150 / 150)

| Method (Same Model: `deepseek-v4-flash`) | Overall Acc | Wrong | Abstain | Answered Precision | Avg Prompt Tokens (Total) |
|---|---:|---:|---:|---:|---:|
| `ChunkRAG-E2E` (top-5 flat chunks) | 33.3% | 8.7% | 58.0% | 79.4% | 1,313 |
| **`AnchorDoc-E2E` (AnchorDoc + Python AST Calc)** | **78.7%** | 18.0% | **3.3%** | **81.4%** | 14,150 |

### 1.3 Breakdown by Official FinanceBench Question Category (50 / 50 / 50)

| Category | Count | AnchorDoc Hit@5 | **AnchorDoc AllGold@5** | ChunkRAG E2E Acc | **AnchorDoc E2E Acc** |
|---|---:|---:|---:|---:|---:|
| **`metrics-generated`** (Hard Numerical Statements) | 50 | **100.0%** | **100.0%** | 20.0% | **90.0%** (0.0% abstain) |
| **`novel-generated`** (Cross-Page Fact Verification) | 50 | 76.0% | 72.0% | 48.0% | **76.0%** (4.0% abstain) |
| **`domain-relevant`** (Qualitative Financial Analysis) | 50 | 78.0% | 72.0% | 32.0% | **70.0%** (6.0% abstain) |
| **Total Official Open-Source Suite** | **150** | **84.7%** | **81.3%** | **33.3%** | **78.7%** |

---

## 2. ContractNLI-Long — Long Legal Contract Compliance Auditing (N = 150, >= 10,000 Chars)

- **Corpus**: Official `ContractNLI-Long` test split (`>= 10,000` chars, up to 42k chars), strictly balanced (50 `entailment` / 50 `contradiction` / 50 `neutral`, `seed=42`).
- **Reproduce**: `python3 benchmarks/run_contractnli_full.py --per-class 50`

| Method (Same Model: `deepseek-v4-flash`) | Accuracy | Macro-F1 | False-Pass Rate | Severe Flip (`C->E`) | High-Conf Coverage / Acc / FER | Avg Prompt Tokens |
|---|---:|---:|---:|---:|---:|---:|
| `ChunkRAG-Direct` (800-char chunks) | 74.7% | 74.9% | 11.0% | 5 | 82.0% / 81.3% / 13.1% | 734 |
| `LongContext-Direct` (Full Contract Stuffed) | 78.0% | 78.0% | 16.0% | 7 | 99.3% / 78.5% / 16.2% | 3,233 |
| **`AnchorDoc` (`SpanCard + B42 + QuoteGate`)** | **78.0%** | **78.1%** | **14.0%** | **6** | **94.7% / 80.3% / 14.7%** | **2,420 (-25%)** |

---

## 3. MMLongBench-Doc (NeurIPS 2024) — Non-Financial Multimodal PDFs (101 Questions, 7 PDFs, 170 Pages)

- **Corpus**: 7 non-financial multimodal PDFs (research reports, slide decks, crisis analyses, surveys — heavy charts, figures, layouts, plus 36 unanswerable questions).
- **Reproduce**: `python3 benchmarks/run_mmlongbench.py`
- **Retrieval Hit@5 (Answerable, n=65)**: `Chunk-BM25`: **44.6%** -> **`AnchorDoc`**: **63.1% (+18.5pp)**.

| Method (Same Multimodal Model: `deepseek-v4-flash`) | Overall Acc | **Text / Table Acc** | **Chart / Figure Acc** | Unanswerable Abstain Acc |
|---|---:|---:|---:|---:|
| `ChunkRAG-TextOnly` | 36.6% | 21.9% | 0.0% | 83.3%* |
| `AnchorDoc-TextOnly` | 32.7% | 37.5% | 3.0% | 55.6% |
| **`AnchorDoc-Multimodal` (Rendered Page PNG Vision)** | **41.6%** | **53.1%** | **12.1%** | 58.3% |

*\*`ChunkRAG-TextOnly` achieves high abstain accuracy mainly because it fails to locate evidence on most questions (0.0% on charts, 21.9% on tables) and defaults to abstaining.*

> **Honest Diagnostic Finding (Next Milestone Target)**: On image-heavy slide decks where chart pages contain minimal extractable PDF text, Zero-LLM text indexing alone cannot surface chart-only pages (`12.1%` Chart/Fig acc). Adding **Selective `VisionPageCard` indexing** (invoking vision summarization **only** on low-text/image-heavy pages) + a strict two-step absence gate for unanswerable questions is our top `v0.4` priority.

---

## 4. Next Planned Additions

- [ ] **Selective `VisionPageCard` indexing** for image-only/low-text PDF pages (targeting `MMLongBench-Doc` Chart/Fig jump).
- [ ] **Dense Vector RAG (`BGE-M3` + Reranker)** & **PageIndex Cost/Latency** direct side-by-side table on FinanceBench 150.
- [ ] **3-Run Variance** report (`mean +/- std`) for camera-ready reproducibility.
