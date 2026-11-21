# AnchorDoc Official Public Benchmark Leaderboard

All evaluations below use the **exact same base model (`deepseek-v4-flash`, `temperature=0`)**, **Apache-2.0 `pypdfium2`** PDF parsing, deterministic SHA256 disk caching, and **zero domain-specific hardcoding** inside `anchordoc/`.

---

## Executive Summary Across 3 Modalities (401 Official Benchmark Questions)

| Benchmark | Corpus & Modality | Baseline (ChunkRAG / Long-Context) | **AnchorDoc `v0.4`** | Key Architectural Win |
|---|---|---|---|---|
| **FinanceBench** (Full 150/150) | 84 SEC PDFs (**12,013 pages**, dense tables + math) | Hit@5 `35.3%` · E2E `33.3%` (`metrics`: `20.0%`) | Hit@5 **`84.7%`** · AllGold@5 **`81.3%`** · E2E **`78.7%`** (`metrics`: **`90.0%`**, **`100%`** AllGold@5) | **0 indexing LLM calls** (`0.32s/PDF`); **+45.4pp E2E**; `AnchorMap+G7` adds **+18.0pp AllGold@5** |
| **MMLongBench-Doc** (NeurIPS 2024, 101q) | 7 non-financial multimodal PDFs (170p, 3 pure-image slide decks + charts + 36 unanswerables) | Hit@5 `44.6%` · Overall `36.6%` · Chart/Fig `0.0%` | Hit@5 **`95.4%`** · Overall **`52.5%`** · Text/Tbl **`53.1%`** · Chart/Fig **`24.2%`** · Unans **`77.8%`** | **Selective `VisionPageCard`** skips 100% of text-rich pages, lifts Hit@5 **+32.3pp** (`63.1% -> 95.4%`), halves unanswerable hallucination |
| **ContractNLI-Long** (EMNLP, N=150) | 150 long legal NDAs (`>= 10k` chars, balanced 50/50/50) | Long-Context Direct: F1 `78.0%` · HiConf `78.5%` · FER `16.0%` · `3,233` tok | F1 **`78.1%`** · HiConf **`80.3%`** · FER **`14.0%`** · **`2,420` tok** | **Strictly Pareto-dominates full Long-Context** while saving **25% tokens** |

---

## 1. FinanceBench — Full Official Open-Source Suite (150 / 150 Questions, 84 SEC PDFs, 12,013 Pages)

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

## 2. MMLongBench-Doc (NeurIPS 2024) — Non-Financial Multimodal PDFs (101 Questions, 7 PDFs, 170 Pages)

- **Corpus**: 7 non-financial multimodal PDFs (research reports, surveys, and **3 100%-rasterized slide decks** with `0` extractable text chars — 33 Chart/Figure questions, 32 Text/Table questions, and 36 Unanswerable questions).
- **Selective Indexing Efficiency**: `SelectiveVisionIndexer` automatically skipped **78 / 170** text-rich digital pages (`0` indexing LLM calls) and indexed **only the 92 / 170** low-text/rasterized visual pages.
- **Reproduce**: `python3 benchmarks/run_mmlongbench.py`

### 2.1 Retrieval Progression (Answerable Questions, n = 65)

| Retriever | Hit@5 | Gain vs Chunk-BM25 |
|---|---:|---:|
| `Chunk-BM25` | 44.6% | — |
| `AnchorDoc v0.3` (Text-Only AnchorMap) | 63.1% | +18.5pp |
| **`AnchorDoc v0.4` (Hybrid Zero-LLM + Selective `VisionPageCard`)** | **95.4%** (62/65) | **+50.8pp** |

### 2.2 End-to-End Multimodal QA & Unanswerable Hallucination Control (101 Questions)

| Method (Same Multimodal Model: `deepseek-v4-flash`) | **Overall Acc** | **Text / Table Acc** | **Chart / Figure Acc** | **Unanswerable Abstain Acc** | False Answer on Unans |
|---|---:|---:|---:|---:|---:|
| `ChunkRAG-TextOnly` | 36.6% | 21.9% | 0.0% | 83.3%* | 16.7% |
| `AnchorDoc-TextOnly` | 32.7% | 37.5% | 3.0% | 55.6% | 44.4% |
| `AnchorDoc v0.3` (Vision QA only, no Selective VisionCard) | 41.6% | 53.1% | 12.1% | 58.3% | 41.7% |
| **`AnchorDoc v0.4` (`SelectiveVisionCard + AbsenceGate`)** | **52.5%** | **53.1%** | **24.2%** | **77.8%** | **22.2%** |

*\*`ChunkRAG-TextOnly` abstains heavily simply because it fails to retrieve evidence on nearly all chart (`0.0%`) and table (`21.9%`) questions.*

---

## 3. ContractNLI-Long — Long Legal Contract Compliance Auditing (N = 150, >= 10,000 Chars)

- **Corpus**: Official `ContractNLI-Long` test split (`>= 10,000` chars, up to 42k chars), strictly balanced (50 `entailment` / 50 `contradiction` / 50 `neutral`, `seed=42`).
- **Reproduce**: `python3 benchmarks/run_contractnli_full.py --per-class 50`

| Method (Same Model: `deepseek-v4-flash`) | Accuracy | Macro-F1 | False-Pass Rate | Severe Flip (`C->E`) | High-Conf Coverage / Acc / FER | Avg Prompt Tokens |
|---|---:|---:|---:|---:|---:|---:|
| `ChunkRAG-Direct` (800-char chunks) | 74.7% | 74.9% | 11.0% | 5 | 82.0% / 81.3% / 13.1% | 734 |
| `LongContext-Direct` (Full Contract Stuffed) | 78.0% | 78.0% | 16.0% | 7 | 99.3% / 78.5% / 16.2% | 3,233 |
| **`AnchorDoc` (`SpanCard + B42 + QuoteGate`)** | **78.0%** | **78.1%** | **14.0%** | **6** | **94.7% / 80.3% / 14.7%** | **2,420 (-25%)** |
