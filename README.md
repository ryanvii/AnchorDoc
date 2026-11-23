<div align="center">

# ⚓ AnchorDoc

**Verifiable Long-PDF Reading & Independent Claim Verification Engine for AI Agents**

[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab.svg)](pyproject.toml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24453581.svg)](https://doi.org/10.5281/zenodo.24453581)
[![Adversarial Verify](https://img.shields.io/badge/Adversarial%20Verify%2028q-100%25%20Catch%20%7C%200%25%20False--Alarm-059669.svg)](#independent-claim-verification-verify-any-ai-answer-v091)
[![FinanceBench](https://img.shields.io/badge/FinanceBench%20150%2F150-85.3%25%20E2E%20%7C%2096%25%20Metrics-16a34a.svg)](benchmarks/LEADERBOARD.md)
[![MMLongBench](https://img.shields.io/badge/MMLongBench--Doc-68.3%25%20E2E%20%7C%2054.5%25%20Charts-2563eb.svg)](benchmarks/LEADERBOARD.md)
[![ContractNLI](https://img.shields.io/badge/ContractNLI--Long-81.3%25%20F1%20%7C%2010%25%20FER-7c3aed.svg)](benchmarks/LEADERBOARD.md)
[![MCP](https://img.shields.io/badge/MCP%20Server-Zero--LLM%204%20Tools-0f172a.svg)](#mcp-server-for-ai-agents-zero-api-key)

**English** | [简体中文](README_ZH.md)

</div>

---

**AnchorDoc** replaces brittle chunk-based Vector RAG and expensive upfront LLM tree-indexing with a **0.3s Zero-LLM Structural Anchor Map**, **Query-Aware Single-Shot Evidence Locator** (preserving full pages + continuation tables + printed-page labels), **Selective VisionPageCards + 2×2 Adaptive High-Res Zoom**, and **Reasoning-First Deterministic Adjudication** (authoritative Python AST math, operand provenance, tri-state visual evidence gate, and two-stage logic NLI truth tables).

![AnchorDoc Demo Preview](assets/demo_preview.gif)

> **Full 1080p HD Launch Demos**: [English 1080p MP4](https://github.com/ryanvii/AnchorDoc/releases/download/v0.9.1/anchordoc_launch_en.mp4) · [中文 1080p MP4](https://github.com/ryanvii/AnchorDoc/releases/download/v0.9.1/anchordoc_launch_zh.mp4)

![AnchorDoc Architecture](assets/architecture.svg)

---

## Why AnchorDoc?

| Approach | Indexing Cost (250p PDF) | FinanceBench 150/150 E2E* | Multi-Page Recall (AllGold@5) | MMLongBench-Doc (101q) |
|---|---:|---:|---:|---|
| Flat Chunk BM25 | `0.3s` · `0` LLM calls | `33.3%` | `32.0%` | `36.6%` (`0.0%` Chart/Fig) |
| Dense Vector RAG (`BGE-small` ONNX) | `47.6s` (neural embed) | `39.3%` | `48.7%` | Blind to pure-image slides |
| LLM Tree-Index (PageIndex-style 12p win) | `11.4s` · `19.2` calls · `33k` tok | `35.9%` *(39q subset)* | `43.6%` *(39q subset)* | Requires VLM per page |
| **⚓ AnchorDoc (`v0.9.0`)** | **`0.33s` · `0` LLM calls** *(warm `3.9ms`)* | **`85.3%`** *(`96.0%` metrics · `84.7%` auto)* | **`89.3%`** *(`100%` metrics)* | **`68.3%`** *(`54.5%` Chart/Fig)* |

*\*Unified base model (`deepseek-v4-flash`, `temperature=0`, Apache-2.0 `pypdfium2`). Retrieval/adjudication benchmarks reflect frozen milestone runs preserved backward-compatibly in `v0.9.0`, while `v0.9.0` hardens sign/period/unit/pp-vs-% verification and adds zero-API-key independent claim verification; complete ablations and cross-model (`DeepSeek-V4.1-Flash`) tables are in [benchmarks/LEADERBOARD.md](benchmarks/LEADERBOARD.md).*

---

## Independent Claim Verification: Verify ANY AI Answer (`v0.9.1`, Zero API Key)

Already using **RAG, PageIndex, or long-context LLMs**? Pass any PDF plus an external claim (quotes, operands bound to row/period/unit, and symbolic or numeric formula) to AnchorDoc **before signing off**. Every report states explicitly **what was verified** (`checks: {quotes, operands, computation, claim_alignment}` + `verified_scope`) with a 4-state verdict (`VERIFIED` | `PARTIALLY_VERIFIED` | `FAILED` | `UNVERIFIED`):

- **`QUOTE_SIGN_FLIP` / `SIGN_FLIP`**: `-12%` vs `12%`, accounting parentheses `$(145)` vs `$145` (even when `+12%` exists elsewhere on the same page)
- **`PERIOD_MISMATCH`**: value exists on the row/page, but came from `FY2023` while claiming `FY2024`
- **`UNIT_SCALE_MISMATCH`**: `in millions` vs `billions` confusion (while passing valid `$1,200M == $1.2B` conversions)
- **`PP_VS_PCT_CONFUSION`**: margin difference `40.0% - 36.0% = 4.0 pp` mislabeled as `4.0%`
- **`FORMULA_OPERAND_UNBOUND`** (anti-laundering): supports direct symbolic formulas (`rev - cogs`) and rejects any numeric literal/symbol in `formula` not bound to a verified cell operand (blocks valid `[1200, 720]` operands masking `1200 - 700 = 500`)
- **`CLAIM_UNSUPPORTED_BY_EVIDENCE`** (scoped honesty): attaching a true-but-irrelevant verbatim quote to an unverified numeric claim returns `PARTIALLY_VERIFIED` (`quotes=verified`, `claim_alignment=unverified`), **never** `VERIFIED`

### Run the Clean-Clone Adversarial Mini-Bench in 1 Second (0 Downloads, 0 API Keys)

```bash
python3 benchmarks/run_adversarial_verify.py
```

| Verifier (28 Cases: 17 Adversarial Traps + 11 Valid Claims) | Trap Catch Rate (↑) | False-Pass Rate (↓) | False-Alarm Rate (↓) | Exact Error Diagnosis |
|---|---:|---:|---:|---:|
| Naive Alphanumeric Citation & Loose Number Check | `11.8%` (2/17) | `88.2%` (15/17) | `9.1%` (1/11) | — |
| **⚓ AnchorDoc `v0.9.1` (`verify_claim` / `doc_verify`)** | **`100.0%` (17/17)** | **`0.0%` (0/17)** | **`0.0%` (0/11)** | **`17 / 17`** |

```bash
# Verify any external AI claim with cell-bound symbolic formula (exit 0 = VERIFIED, exit 2 = FAILED / PARTIAL)
anchordoc verify-claim report.pdf \
  --claim "FY2024 Gross profit is 480 million." \
  --operands-json '[{"name":"rev","value":1200,"unit":"million","row_label":"Total Revenue","period":"FY2024","page":2},{"name":"cogs","value":720,"unit":"million","row_label":"Cost of Goods Sold","period":"FY2024","page":2}]' \
  --formula "rev - cogs" --claimed-value 480 --claimed-unit "million"
```

![AnchorDoc Public Benchmarks](assets/benchmark_comparison.svg)

---

## Verified Public Benchmarks (401 Official Questions Across 3 Modalities)

- **FinanceBench (Full Official 150 / 150, 84 SEC PDFs, 12,013 pages)**
  - **Indexing**: **27.6s total (`0.33s/PDF`), `0` LLM calls** (145× faster than CPU Dense BGE embedding, 91× faster than upfront LLM tree indexing).
  - **Retrieval (`v0.6` Query-Aware Locator)**: **93.3% Hit@5 / 89.3% AllGold@5** (`+8.6pp` / `+8.0pp` over `v0.5`, MRR `0.8008`) vs Dense BGE **58.0% / 48.7%** and Chunk-BM25 **35.3% / 32.0%**, while cutting query prompt tokens by **23.8%** (`14,150 → 10,780`) and LLM calls by **22.4%** (`5.35 → 4.15`).
  - **End-to-End QA**: **85.3% overall** (`+6.6pp` over `v0.5`, paired McNemar $p=0.0129$; **2.17×** Dense BGE `39.3%`). On all 50 hard numerical statement questions (`metrics-generated`), AnchorDoc achieves **100.0% AllGold@5** and **96.0% E2E accuracy (48/50)** via reasoning-first extraction + authoritative Python AST arithmetic (`safe_calc`, `100%` operand provenance). Replicated on official **`DeepSeek-V4.1-Flash`**: **92.0% Hit@5 / 82.7% E2E** (`92.0%` metrics, `101/101` operands grounded).
- **MMLongBench-Doc (NeurIPS 2024 — 101 questions, 7 non-financial multimodal PDFs incl. 3 pure-image slide decks)**
  - **Selective `VisionPageCard`** skips 100% of text-rich digital pages (`78/170`) and indexes only low-text/scanned pages (`92/170`).
  - **End-to-End (`v0.7` `full_pool` + Tri-State Gate + 2×2 High-Res Zoom)**: **68.3% overall** (`+15.8pp` over `v0.5`, vs `36.6%` ChunkRAG), **54.5%** Chart/Figure (**2.25×** `v0.5`'s `24.2%`), **65.6%** Text/Table, and **83.3%** Unanswerable Abstain accuracy (hallucination cut to `16.7%`).
  - **Unseen Blind Generalization (`v0.6.0`, 10 unseen PDFs, 492 pages, 91q incl. 52 Chart/Figure)**: **91.0% Hit@5**, **50.0% Chart/Figure** (vs `1.9%` ChunkRAG), **50.5% overall** (`+19.7pp` over ChunkRAG).
- **ContractNLI-Long (EMNLP — 150 long legal NDAs `>= 10k` chars, balanced 50/50/50)**
  - **Two-Stage Logic NLI + Adaptive `ReadAll` Router**: **81.3% Accuracy / 81.3% Macro-F1** (`10.0%` FER on `test-150`); on the strictly unseen held-out `dev-150` split (34 unseen contracts), `v0.7` `route=auto` maintains statistical parity in accuracy (`79.3%` vs `78.0%`, Macro-F1 `79.5%` vs `77.9%`) while cutting LLM calls by **41%** (`~3.0 → 1.77 calls/q`).
- **Persistent SQLite WAL `BundleStore` & Prefix-Stable `nav` Caching (`v0.8.0`)**
  - Reloads all **84 SEC PDFs (12,013 pages)** in **`0.330s`** vs `22.13s` cold parse (**`67.1×` faster**, `3.9 ms/PDF`), and persists expensive `VisionPageCard` summaries for `0`-LLM / `0`-render reloads.
  - **T7 3-Run Live Variance & Prefix KV Cache** (30 stratified FinanceBench questions across 5 multi-question SEC 10-Ks on official `DeepSeek-V4.1-Flash`, `temp=0`): **`85.55% ± 3.85%`** E2E Acc, **`100.0% ± 0.0%`** Quote Verified (90/90), **`54.7% ± 10.7%`** real server-side KV prefix cache hit rate (rising to **`65.1%`** warm via prefix-stable `anchordoc_nav`), and **`83.33% ± 0.00%`** deterministic SHA-cached offline replay.

> **Version Reproducibility Note**: `v0.6`, `v0.7`, and `v0.7.1` denote the frozen milestone runs for each benchmark suite above; **`v0.8.0`** preserves the exact retrieval/adjudication core backward-compatibly while adding `BundleStore`, prefix-stable `nav` KV caching, and the 4-tool MCP server (verified on `v0.8.0` live T7 replay at `85.55% ± 3.85%` E2E and `100.0%` quote verification).

See [benchmarks/LEADERBOARD.md](benchmarks/LEADERBOARD.md) for complete tables, ablations, cost-amortization notes, and one-command reproduction scripts.

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

## MCP Server for AI Agents (Zero API Key)

Give **Claude Code, Cursor, Windsurf, Antigravity, or Devin** progressive, token-bounded, verifiable access to 500-page PDFs with **zero external LLM calls** (the host agent *is* the LLM — AnchorDoc acts as its zero-LLM index, token guardrail, and deterministic verifier):

### Same-Agent Benchmark: Native Full-PDF vs AnchorDoc Zero-LLM MCP (`DeepSeek-V4.1-Flash`, `temp=0`, 15 SEC 10-K/10-Q filings, avg 144.7p)

| Same Agent (`DeepSeek-V4.1-Flash`) | E2E Acc | Metrics / Novel | Quote Verified | Avg Prompt Tokens | Newly Billed Uncached Tokens |
|---|---:|---:|---:|---:|---:|
| Native Full-PDF Stuffing *(450k-char cap)* | `60.0%` | `80% / 60%` | `53.3%` | `90,824` | `72,827` |
| **⚓ Same Agent + AnchorDoc MCP** *(`0` server LLM calls)* | **`86.7%` (+26.7pp)** | **`100% / 100%`** | **`100.0%` (15/15)** | **`28,188` (-69.0%)** | **`5,430` (-92.5%, 13.4× cheaper)** |

*(Even on the 8 completely untruncated PDFs where 100% of text fit in Native Full-PDF, AnchorDoc MCP still won **`87.5%` vs `50.0%`** by preventing long-context attention decay.)*

### 1-Minute MCP Setup (Claude Code / Cursor / Windsurf / Devin)

```bash
git clone https://github.com/ryanvii/AnchorDoc.git
cd AnchorDoc
pip install -e '.[mcp]'
```

Add to your `mcp.json` (Cursor `.cursor/mcp.json`, Claude Desktop/Code `claude_desktop_config.json`, or Windsurf/Devin MCP config):

```json
{
  "mcpServers": {
    "anchordoc": {
      "command": "anchordoc",
      "args": ["mcp", "--root", "/path/to/your/pdfs"]
    }
  }
}
```

Exposes 4 orthogonal, read-only tools with strict `--root` allowlisting and hard token caps:
- `doc_outline` (L0 map, ≤1.5k tok): structure, printed page range, and `route_hint`.
- `doc_locate` (L1 zero-LLM locator, ≤1.2k tok): lexical BM25-RRF + page-ref + continuation snippets.
- `doc_read` (L2 anchored reader, ≤5 pages): stable `p12:L07` anchors + session duplicate-read ledger + native MCP `ImageContent`.
- `doc_verify`: deterministic quote verification, page localization, AST math (`safe_calc`), and operand provenance.

### Progressive Context Engineering Ecosystem (Documents + Videos)

AnchorDoc shares the exact same **zero-vector, zero-server-LLM, token-budgeted progressive disclosure protocol** with our companion video context MCP server:

| Modality | Repository | Progressive Protocol (`L0 → L1 → L2 → Verify`) | Dependencies |
|---|---|---|---|
| **Long PDFs & Filings** | **⚓ [AnchorDoc](https://github.com/ryanvii/AnchorDoc)** *(this repo)* | `doc_outline` → `doc_locate` → `doc_read` → `doc_verify` | Pure Python + Apache-2.0 `pypdfium2` |
| **Long Videos & Demos** | **🎬 [vidctx-mcp](https://github.com/ryanvii/vidctx-mcp)** | `peek_video_map` → `search_video_moments` → `inspect_frame_range` | `yt-dlp` / `ffmpeg` / `RapidOCR` |

---

## Core Modules & Domain Adapters

- `anchordoc/docbase`: Thread-safe Apache-2.0 1-based `PageModel` (`pypdfium2`) + `PageLabels` + 2×2 high-res tile rendering + stdlib OOXML merged-cell table grid (`w:vMerge` / `w:gridSpan` inheritance, zero false-blanks).
- `anchordoc/store`: Per-document stdlib SQLite WAL `BundleStore` (layered `doc_sha256` / `extractor_fp` / `vision_fp` invalidation, **67× faster** warm reload, zero-render `VisionPageCard` reuse).
- `anchordoc/context.py` & `anchordoc/mcp_server.py`: Zero-LLM progressive `DocContext` API and 4-tool stdio MCP server.
- `anchordoc/index`: `Zero-LLM AnchorMap`, Printed-TOC (`G7`) offset calibrator & soft demotion, printed-page/slide reference resolver, `SelectiveVisionIndexer` (`VisionPageCard`), and `SpanCardIndex`.
- `anchordoc/retrieve`: Cost-derived 3-tier adaptive router (`read_all` | `full_pool` | `navigate`) + Query-Aware Single-Shot Locator (`_pack_snippets`) + continuation-table expansion + conditional decorrelated `B42` gap-followup + full `RetrievalTrace`.
- `anchordoc/extract`: Line-anchored `[sNNNNN]` requirement extraction with automatic verbatim quote-repair loop.
- `anchordoc/adjudicate`: Reasoning-first verifiable adjudicator with deterministic `AnswerContract`, authoritative Python AST arithmetic (`safe_calc`), Grounding Scope Guard, tri-state visual evidence gate (`stated | derivable | absent`), and deterministic truth tables (`decide_compliance`, `decide_nli`).
- `anchordoc/report`: JSON, Markdown, and zero-server interactive two-pane HTML trace workbench with embedded original PDF page previews.
- `adapters/`: Thin domain plugins — `finance_10k`, `legal_nda`, `multimodal_doc`, `procurement_demo`.

---

## Citation

If you use AnchorDoc or its reproducible benchmarks in your work, please cite via [CITATION.cff](CITATION.cff):

```bibtex
@software{wu2026anchordoc,
  author  = {Wu, HuiYang},
  title   = {AnchorDoc: Zero-Index-Cost Long-Document Context Engineering with Falsifiable Evidence Chains},
  year    = {2026},
  version = {0.8.0},
  doi     = {10.5281/zenodo.24453581},
  url     = {https://github.com/ryanvii/AnchorDoc},
  license = {Apache-2.0}
}
```
