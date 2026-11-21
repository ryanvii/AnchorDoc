"""BundleStore Cold-vs-Warm Benchmark & Multi-Query Cost Amortization Report (v0.7.0).

Measures real wall-clock cold vs warm reload times for `BundleStore` (SQLite WAL) across local
PDFs (zero LLM cost when replaying existing caches), and computes the transparent Q-query cost
amortization curve comparing:
  - AnchorDoc v0.6 (cold navigate per query)
  - AnchorDoc v0.7 (adaptive router + BundleStore)
  - LLM-Tree-Index (PageIndex-style upfront tree indexing)
Explicitly marks Q=20 on FinanceBench as projected (max real Q/doc in FinanceBench is 7).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anchordoc.docbase import PdfDocument
from anchordoc.index import DocumentIndex
from anchordoc.store import BundleStore, load_document


def bench_bundle_store(pdf_paths: list[Path], label: str) -> dict[str, Any]:
    """Measure real cold PDF parse + SQLite persist vs warm BundleStore reload."""
    if not pdf_paths:
        return {"label": label, "n_pdfs": 0}
    tmp = Path(tempfile.mkdtemp(prefix="anchordoc_amort_"))
    try:
        store = BundleStore(tmp / "bundles")

        # 1. Cold parse (pypdfium2 + save to BundleStore)
        t0 = time.perf_counter()
        docs_cold = [load_document(p, backend="pypdfium2", store=store) for p in pdf_paths]
        cold_sec = time.perf_counter() - t0
        total_pages = sum(d.num_pages for d in docs_cold)

        # 2. Warm reload (from BundleStore, 0 PDF parses) + in-memory DocumentIndex build
        t1 = time.perf_counter()
        docs_warm = [load_document(p, backend="pypdfium2", store=store) for p in pdf_paths]
        warm_load_sec = time.perf_counter() - t1

        t2 = time.perf_counter()
        _ = [DocumentIndex(d) for d in docs_warm]
        warm_idx_sec = time.perf_counter() - t2

        info = store.info()
        return {
            "label": label,
            "n_pdfs": len(pdf_paths),
            "total_pages": total_pages,
            "cold_parse_and_save_sec": round(cold_sec, 3),
            "warm_bundle_reload_sec": round(warm_load_sec, 3),
            "warm_in_memory_index_sec": round(warm_idx_sec, 3),
            "reload_speedup_x": round(cold_sec / max(warm_load_sec, 1e-6), 1),
            "bundle_total_kb": round(info["total_bytes"] / 1024.0, 1),
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def build_q_curve() -> dict[str, Any]:
    """Build honest multi-query cumulative token amortization curves from measured v0.6/v0.7 data."""
    # Measured FinanceBench components (deepseek-v4-flash):
    # - LLM-Tree-Index (8 SEC 10-Ks, 1814p): 32,972 indexing tok/PDF + 5,818 query tok/q
    # - AnchorDoc v0.6 (150q): 0 indexing tok/PDF + 10,780 query tok/q (of which ~3,300 nav prefix is static)
    # - MMLongBench (7 PDFs, 170p, 101q): 46,304 prompt + 18,443 comp VisionCard indexing (92 pages)
    fb_v06_per_q = 10_780
    fb_tree_idx = 32_972
    fb_tree_per_q = 5_818
    nav_static_prefix = 3_300

    fb_rows = []
    for q in (1, 3, 5, 7, 10, 20):
        v06_tot = q * fb_v06_per_q
        tree_tot = fb_tree_idx + q * fb_tree_per_q
        # With provider prefix caching (r=0.1 billing rate on repeated static nav prefix for queries 2..Q)
        v07_cached_eq = fb_v06_per_q + max(0, q - 1) * ((fb_v06_per_q - nav_static_prefix) + 0.1 * nav_static_prefix)
        fb_rows.append({
            "Q": q,
            "projected": q > 7,
            "anchordoc_v06_tok": v06_tot,
            "anchordoc_prefix_cached_r01_eq_tok": int(round(v07_cached_eq)),
            "llm_tree_no_cache_tok": tree_tot,
        })

    # Scanned 300-page multimodal PDF: cold VisionCard indexing vs warm BundleStore reload
    # Measured MMLongBench avg: 503 prompt + 200 comp = 703 tok/visual page (~211k tok for 300 scanned pages)
    mm_cold_300p_tok = 300 * 703
    return {
        "financebench_10k_q_curve": fb_rows,
        "crossover_q_no_prefix_cache": round(fb_tree_idx / (fb_v06_per_q - fb_tree_per_q), 2),
        "scanned_300p_vision_cold_tok": mm_cold_300p_tok,
        "scanned_300p_vision_warm_bundle_tok": 0,
        "note": "Q>7 on FinanceBench is projected from measured per-component averages (max real Q/doc in FinanceBench is 7).",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure BundleStore speedup and Q-curve amortization.")
    ap.add_argument("--out", default=str(ROOT / "results/amortization_v07.json"))
    args = ap.parse_args()

    fb_pdfs = sorted((ROOT / "data/financebench_pdfs").glob("*.pdf"))
    mm_pdfs = sorted((ROOT / "data/mmlongbench/pdfs").glob("*.pdf"))

    out = {
        "bundle_benchmarks": [
            bench_bundle_store(fb_pdfs, "FinanceBench (84 SEC PDFs)"),
            bench_bundle_store(mm_pdfs, "MMLongBench (7 multimodal PDFs)"),
        ],
        "q_curve": build_q_curve(),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== BundleStore Real Cold-vs-Warm Reload Benchmark ===")
    for b in out["bundle_benchmarks"]:
        if b["n_pdfs"]:
            print(
                f"  {b['label']:32s} | {b['total_pages']:5d}p | "
                f"Cold={b['cold_parse_and_save_sec']:6.3f}s -> WarmReload={b['warm_bundle_reload_sec']:5.3f}s "
                f"({b['reload_speedup_x']:4.1f}x faster) | Store={b['bundle_total_kb']:.0f} KB"
            )
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
