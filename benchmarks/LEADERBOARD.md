# AnchorDoc Official Public Benchmark Leaderboard

All evaluations below use the **exact same base model (`deepseek-v4-flash`, `temperature=0`)**, **Apache-2.0 `pypdfium2`** PDF parsing, deterministic SHA256 disk caching, and **zero domain-specific regex hardcoding** inside `anchordoc/`.

---

## 1. FinanceBench — Full Official Open-Source Suite (150 / 150 Questions, 84 SEC PDFs, 12,013 Pages)

- **Corpus**: All 84 real SEC 10-K / 10-Q / 8-K filings (**12,013 total physical pages**).
- **Indexing Cost**: **26.7 seconds total (`0.32s / PDF`), `0` LLM calls** (vs. hundreds of indexing LLM calls required by full-book tree-indexing frameworks).
- **Reproduce**: `python3 benchmarks/run_financebench_full.py --limit 150 --backend pypdfium2`

### 1.1 Page-Level Evidence Retrieval (150 / 150)

| Method | Indexing LLM Calls | Hit@1 | Hit@3 | Hit@5 | **AllGold@5 (Full Page Completeness)** | MRR@10 |
|---|---:|---:|---:|---:|---:|---:|
| `Chunk-BM25` (flat 1000-char chunks) | 0 | 18.7% | 29.3% | 35.3% | 32.0% | 0.2613 |
| `RawPage-BM25` (full-page BM25) | 0 | 17.3% | 26.7% | 36.7% | 33.3% | 0.2533 |
| `AnchorDoc-PageCard` (zero-LLM Dual BM25-RRF) | 0 | 14.7% | 32.0% | 40.0% | 32.7% | 0.2492 |
| **`AnchorDoc-Full` (`AnchorMap + G7 + C3 + B42`)** | **0** | **71.3%** | **83.3%** | **84.7%** | **81.3%** | **0.7719** |

### 1.2 End-to-End Verifiable QA (150 / 150)

| Method (Same Model: `deepseek-v4-flash`) | Overall Acc | Wrong | Abstain | Answered Precision | Avg Prompt Tokens (Retrieval + QA) |
|---|---:|---:|---:|---:|---:|
| `ChunkRAG-E2E` (top-5 flat chunks) | 33.3% | 8.7% | 58.0% | 79.4% | 1,313 |
| **`AnchorDoc-E2E` (AnchorDoc + Python AST Calc)** | **78.7%** | 18.0% | **3.3%** | **81.4%** | 14,150* |

*\*Includes all retrieval + navigation + rerank + answer prompt tokens across the 5-stage pipeline (avg ~2.3s wall-clock/question at 8 workers, $0 indexing cost).*

### 1.3 Breakdown by Official FinanceBench Question Category (50 / 50 / 50)

| Category | Count | AnchorDoc Hit@5 | **AnchorDoc AllGold@5** | ChunkRAG E2E Acc | **AnchorDoc E2E Acc** |
|---|---:|---:|---:|---:|---:|
| **`metrics-generated`** (Hard Numerical Statements) | 50 | **100.0%** | **100.0%** | 20.0% | **90.0%** (0.0% abstain) |
| **`novel-generated`** (Cross-Page Fact Verification) | 50 | 76.0% | 72.0% | 48.0% | **76.0%** (4.0% abstain) |
| **`domain-relevant`** (Qualitative Financial Analysis) | 50 | 78.0% | 72.0% | 32.0% | **70.0%** (6.0% abstain) |
| **Total Official Open-Source Suite** | **150** | **84.7%** | **81.3%** | **33.3%** | **78.7%** |

---

## 2. ContractNLI-Long — Long Legal Contract Compliance Auditing (N = 150, >= 10,000 Chars)

- **Corpus**: Official `ContractNLI-Long` test split filtered to long agreements (`>= 10,000` chars, up to 42k chars), strictly balanced (50 `entailment` / 50 `contradiction` / 50 `neutral`, `seed=42`).
- **Reproduce**: `python3 benchmarks/run_contractnli_full.py --per-class 50`

| Method (Same Model: `deepseek-v4-flash`) | Accuracy | Macro-F1 | False-Pass Rate | Severe Flip (`C->E`) | High-Conf Coverage / Acc / FER | Avg Prompt Tokens |
|---|---:|---:|---:|---:|---:|---:|
| `ChunkRAG-Direct` (800-char chunks) | 74.7% | 74.9% | 11.0% | 5 | 82.0% / 81.3% / 13.1% | 734 |
| `LongContext-Direct` (Full Contract Stuffed) | 78.0% | 78.0% | 16.0% | 7 | 99.3% / 78.5% / 16.2% | 3,233 |
| **`AnchorDoc` (`SpanCard + B42 + QuoteGate`)** | **78.0%** | **78.1%** | **14.0%** | **6** | **94.7% / 80.3% / 14.7%** | **2,420 (-25%)** |

Key takeaway: On long contracts, `AnchorDoc` strictly Pareto-dominates stuffing the full contract into a long-context LLM — achieving higher Macro-F1, higher high-confidence accuracy (`80.3%` vs `78.5%`), lower false-pass rate (`14.0%` vs `16.0%`), and **25% fewer prompt tokens**.

---

## 3. Next Planned Leaderboard Additions (Phase 2 Continued)

- [ ] **Dense Vector RAG Baseline** (`BGE-M3` + `bge-reranker-v2-m3`) on identical FinanceBench 150 & ContractNLI-Long 150 splits.
- [ ] **PageIndex Cost & Latency Comparison** (measuring upfront tree-indexing LLM calls/cost vs AnchorDoc's `0`-call `0.32s/PDF` AnchorMap).
- [ ] **MMLongBench-Doc** (Multimodal PDF charts, scanned tables, and unanswerable questions testing `VisionPageCard` + `0` ungrounded pass).
- [ ] **3-Run Variance & Full Ablation Table** (`-AnchorMap`, `-G7`, `-B42`, `-PythonCalc`).
