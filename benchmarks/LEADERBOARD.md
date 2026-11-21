# AnchorDoc Official Public Benchmark Leaderboard

All evaluations below use the **exact same base model (`deepseek-v4-flash`, `temperature=0`)**, **Apache-2.0 `pypdfium2`** PDF parsing, deterministic SHA256 disk caching, and **zero domain-specific hardcoding** inside `anchordoc/`.

---

## Executive Summary Across 3 Modalities (401 Official Benchmark Questions)

| Benchmark | Corpus & Modality | Strongest Baselines | AnchorDoc `v0.5` | **AnchorDoc `v0.6`** | Key `v0.6` Gain |
|---|---|---|---|---|---|
| **FinanceBench** (Full 150/150) | 84 SEC PDFs (**12,013 pages**, dense tables + math) | Chunk-BM25: `33.3%` E2E<br/>Dense BGE RAG: `39.3%` E2E (`47.6s/PDF` embed) | Hit@5 `84.7%` · AllGold@5 `81.3%`<br/>E2E `78.7%` (`metrics` `90.0%`)<br/>`14,150` tok/q | Hit@5 **`93.3%`** · AllGold@5 **`89.3%`**<br/>E2E **`85.3%`** (`metrics` **`96.0%`**)<br/>**`10,780` tok/q** | **+8.6pp Hit@5 · +6.6pp E2E · -23.8% tokens** (0 indexing LLM calls, `0.33s/PDF`, **2.17× Dense BGE E2E**) |
| **MMLongBench-Doc** (NeurIPS 2024, 101q) | 7 non-financial multimodal PDFs (170p, 3 pure-image slide decks + charts + 36 unanswerables) | ChunkRAG: `44.6%` Hit@5 · `36.6%` E2E · `0.0%` Chart/Fig | ReadHit `87.7%` · Overall `52.5%`<br/>Chart/Fig `24.2%` · Text/Tbl `53.1%`<br/>Unans `77.8%` · `9,603` tok/q | ReadHit **`93.8%`** · Overall **`67.3%`**<br/>Chart/Fig **`48.5%`** · Text/Tbl **`68.8%`**<br/>Unans **`83.3%`** · **`8,669` tok/q** | **+14.8pp Overall · 2.0× Chart/Fig (+24.3pp) · -5.5pp Hallucination · -9.7% tokens** |
| **ContractNLI-Long** (EMNLP, N=150) | 150 long legal NDAs (`>= 10k` chars, balanced 50/50/50) | Long-Context Direct: F1 `78.0%` · HiConf `78.5%` · FER `16.0%` · `3,233` tok | Acc/F1 `78.0% / 78.1%`<br/>FER `14.0%` · HiConf `80.3%`<br/>`2,420` tok/q | Acc/F1 **`81.3% / 81.3%`**<br/>FER **`10.0%`** · HiConf **`83.3%`**<br/>`3,173` tok/q | **+3.3pp Acc/F1 · -4.0pp False-Pass Rate** (-6.0pp lower FER than stuffing full contract) |

---

## 1. FinanceBench — Full 150 / 150 Head-to-Head vs Dense Vector RAG & Hybrid RAG (84 SEC PDFs, 12,013 Pages)

- **Reproduce**:
  - `python3 benchmarks/run_financebench_full.py --limit 150 --backend pypdfium2`
  - `python3 benchmarks/run_competitive_comparison.py`

### 1.1 Retrieval & End-to-End Comparison (ALL 150 / 150 Official Questions)

| Method (Same Base Model: `deepseek-v4-flash`) | Indexing Time (84 PDFs) | Indexing LLM Calls | Hit@1 | Hit@3 | Hit@5 | **AllGold@5** | MRR@10 | **E2E Acc** | Avg Prompt Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Chunk-BM25` (flat 1000ch) | 26.7s | 0 | 18.7% | 29.3% | 35.3% | 32.0% | 0.2613 | 33.3% | 1,313 |
| `Dense-Chunk-BGE` (`BAAI/bge-small-en-v1.5` ONNX) | 4,000.2s (`47.6s/PDF`) | 0 (neural embed) | 31.3% | — | 58.0% | 48.7% | 0.4306 | 39.3% | ~1,400 |
| `Dense-Page-BGE` (`BAAI/bge-small-en-v1.5` ONNX) | 4,000.2s | 0 (neural embed) | 26.0% | — | 56.0% | 48.7% | 0.3946 | — | — |
| `Hybrid-BGE+BM25` (RRF) | 4,000.2s | 0 (neural embed) | 28.0% | — | 54.7% | 44.0% | 0.4022 | — | — |
| `AnchorDoc v0.5` (`AnchorMap + G7 + C3 + B42`) | 26.7s (`0.32s/PDF`) | 0 | 71.3% | 83.3% | 84.7% | 81.3% | 0.7719 | 78.7% | 14,150 |
| **`AnchorDoc v0.6` (`Query-Aware Locator + Authoritative Calc`)** | **27.6s (`0.33s/PDF`)** | **0** | **71.3%** | **89.3%** | **93.3%** | **89.3%** | **0.8008** | **85.3%** | **10,780 (-23.8%)** |

> **Cost-Amortization Transparency**: AnchorDoc eliminates upfront indexing cost (`0` indexing LLM calls, `0.33s/PDF`), shifting retrieval reasoning to query time (`~10.8k` total prompt tokens and `4.15` LLM calls per question in `v0.6`, down from `14.15k` tokens and `5.35` calls in `v0.5`). Compared to upfront LLM tree indexing (~`33k` indexing tokens/PDF), AnchorDoc is especially cost-effective for ad-hoc, compliance, and due-diligence workloads where each long document receives 1–5 high-stakes queries and requires verifiable page-level provenance.

### 1.2 Upfront Indexing Cost vs PageIndex-Style LLM Tree Indexing (8 SEC 10-Ks, 1,814 Pages, 39 Questions)

| Method (Same Model: `deepseek-v4-flash`) | Indexing Sec / PDF | **Indexing Calls / PDF** | **Indexing Tokens / PDF** | Hit@5 | **AllGold@5** | **E2E Acc** |
|---|---:|---:|---:|---:|---:|---:|
| `LLM-Tree-Index` (PageIndex-style 12-page windows)* | 11.39s | 19.2 | 32,972 | 46.2% | 43.6% | 35.9% |
| **`AnchorDoc v0.5` (`Zero-LLM AnchorMap`)** | **0.125s (91× faster)** | **0** | **0** | **74.4%** | **71.8%** | **79.5%** |

*\*`LLM-Tree-Index` is a faithful open-source implementation of PageIndex-style upfront LLM hierarchical tree indexing (12-page windows) + single-shot tree reasoning search on the exact same `deepseek-v4-flash` model, measuring apples-to-apples indexing overhead vs Zero-LLM AnchorMap (not VectifyAI's commercial GPT-4o cloud service).*

### 1.3 Full 150-Question Ablation Study

| Configuration | Hit@1 | Hit@3 | Hit@5 | **AllGold@5** | MRR@10 | Avg Calls / Prompt Tokens |
|---|---:|---:|---:|---:|---:|---:|
| `1. Chunk-BM25` | 18.7% | 29.3% | 35.3% | 32.0% | 0.2613 | 0 / 0 (retrieval) |
| `2. PageCard-RRF` (zero-LLM) | 16.7% | 33.3% | 42.7% | 35.3% | 0.2685 | 0 / 0 (retrieval) |
| `3. v0.5 w/o AnchorMap` (`C3 + B42`) | 62.0% | 72.7% | 73.3% | 63.3% | 0.6736 | 3.36 / 5,355 (retrieval) |
| `4. v0.5 Full` (`AnchorMap + G7 + Triage/Rerank + B42`) | 71.3% | 83.3% | 84.7% | 81.3% | 0.7719 | 5.35 / 14,150 (total) |
| **`5. v0.6 Full` (`Query-Aware Locator + Soft TOC + Cond. B42`)** | **71.3%** | **89.3%** | **93.3%** | **89.3%** | **0.8008** | **4.15 / 10,780 (total, -23.8%)** |

### 1.4 Breakdown by Official FinanceBench Category (50 / 50 / 50)

| Category | Count | `v0.5` Hit@5 → **`v0.6` Hit@5** | `v0.5` AllGold@5 → **`v0.6` AllGold@5** | ChunkRAG E2E | `v0.5` E2E → **`v0.6` E2E** |
|---|---:|---:|---:|---:|---:|
| **`metrics-generated`** (Hard Numerical Statements) | 50 | 100.0% → **100.0%** | 100.0% → **100.0%** | 20.0% | 90.0% → **96.0%** (48/50) |
| **`novel-generated`** (Cross-Page Fact Verification) | 50 | 76.0% → **90.0%** (+14pp) | 72.0% → **86.0%** (+14pp) | 48.0% | 76.0% → **82.0%** (+6pp) |
| **`domain-relevant`** (Qualitative Financial Analysis) | 50 | 78.0% → **90.0%** (+12pp) | 72.0% → **82.0%** (+10pp) | 32.0% | 70.0% → **78.0%** (+8pp) |

---

## 2. MMLongBench-Doc (NeurIPS 2024) — Non-Financial Multimodal PDFs (101 Questions, 7 PDFs, 170 Pages)

- **Selective Indexing Efficiency**: `SelectiveVisionIndexer` automatically skipped **78 / 170** text-rich digital pages (`0` indexing LLM calls) and indexed **only the 92 / 170** low-text/rasterized visual pages.
- **Read-Pages Retrieval (Answerable, n = 65)**: Reporting hit rate on the exact pages rendered into the multimodal reader — `v0.5` rendered top-4 (`87.7%` Hit@4, `75.4%` AllGold@4); `v0.6` renders all top-5 with printed-page label resolution (`93.8%` Hit@5, **`86.2%` AllGold@5**, **+10.8pp**).

| Method (Same Multimodal Model: `deepseek-v4-flash`) | **Overall Acc** | **Answerable (n=65)** | **Text / Table (n=32)** | **Chart / Figure (n=33)** | **Unans Abstain (n=36)** | Halluc on Unans (↓) | Avg Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ChunkRAG-TextOnly` | 36.6% | 10.8% | 21.9% | 0.0% | 83.3%* | 16.7% | 801 |
| `AnchorDoc-TextOnly` | 32.7% | 20.0% | 37.5% | 3.0% | 55.6% | 44.4% | 7,301 |
| `AnchorDoc v0.3` (Vision QA only, no Selective VisionCard) | 41.6% | 29.2% | 46.9% | 12.1% | 58.3% | 41.7% | — |
| `AnchorDoc v0.5` (`SelectiveVisionCard + AbsenceGate`) | 52.5% | 38.5% | 53.1% | 24.2% | 77.8% | 22.2% | 9,603 |
| **`AnchorDoc v0.6` (`Tri-State Gate + 2×2 High-Res Zoom + Reasoning-First`)** | **67.3%** | **58.5%** | **68.8%** | **48.5%** | **83.3%** | **16.7%** | **8,669 (-9.7%)** |

*\*`ChunkRAG-TextOnly` abstains trivially on most questions (only 10.8% answerable accuracy).*

---

## 3. ContractNLI-Long — Long Legal Contract Compliance Auditing (N = 150, >= 10,000 Chars)

| Method (Same Model: `deepseek-v4-flash`) | Accuracy | Macro-F1 | False-Pass Rate (↓) | Severe Flip (`C->E`) | High-Conf Cov / Acc / FER | Avg Prompt Tokens |
|---|---:|---:|---:|---:|---:|---:|
| `ChunkRAG-Direct` (800-char chunks) | 74.7% | 74.9% | 11.0% | 5 | 82.0% / 81.3% / 13.1% | 734 |
| `LongContext-Direct` (Full Contract Stuffed) | 78.0% | 78.0% | 16.0% | 7 | 99.3% / 78.5% / 16.2% | 3,233 |
| `AnchorDoc v0.5` (`SpanCard + B42 + QuoteGate`) | 78.0% | 78.1% | 14.0% | 6 | 94.7% / 80.3% / 14.7% | 2,420 |
| **`AnchorDoc v0.6` (`SpanCard Fix + Two-Stage Logic NLI`)** | **81.3%** | **81.3%** | **10.0%** | **7** | **80.0% / 83.3% / 13.3%** | **3,173** |

> **Honest Grounding & Auditability Note**: Every `v0.6` runner persists full `per_item` JSON traces (`results/*_v06.json`), and `AuditVerdict.meta` records structured falsifiability signals (`operand_grounded`, `grounding` level `text_verified | vision_card_supported | vlm_attested`, `composition`, and `nli_path`).
