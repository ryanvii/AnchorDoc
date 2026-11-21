"""60-second Quickstart showing AnchorDoc's Zero-LLM AnchorMap and verifiable audit flow.

Run without any API key (Zero-LLM structural inspection):
    python3 examples/quickstart_audit.py path/to/report.pdf

Run with an OpenAI-compatible API key (full verifiable query + HTML report):
    ANCHORDOC_LLM_KEY=sk-... python3 examples/quickstart_audit.py path/to/report.pdf --query "What is total revenue?"
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from anchordoc import (
    AnchorRetriever,
    CachedLLM,
    DocumentIndex,
    FactAdjudicator,
    PdfDocument,
    export_html,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", help="Path to any PDF document.")
    ap.add_argument("--query", default="", help="Optional question (requires ANCHORDOC_LLM_KEY / OPENAI_API_KEY).")
    ap.add_argument("--html", default="anchordoc_quickstart_report.html", help="HTML report path when --query is set.")
    args = ap.parse_args()

    t0 = time.time()
    doc = PdfDocument.from_pdf(args.pdf)
    idx = DocumentIndex(doc)
    toc = idx.find_printed_toc_pages()
    outline = idx.build_structural_outline(max_anchors=30)
    print(f"[Zero-LLM] Loaded {doc.num_pages} pages & built AnchorMap in {time.time() - t0:.2f}s (0 LLM calls)")
    print(f"[Zero-LLM] Printed TOC pages: {toc or 'none'}")
    print("\n--- Top Structural Anchors ---")
    print(outline)

    if not args.query:
        return 0

    if not (os.environ.get("ANCHORDOC_LLM_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("ANCHORDOC_LLM_KEY_FILE")):
        print("\nSet ANCHORDOC_LLM_KEY or OPENAI_API_KEY to run --query.", file=sys.stderr)
        return 1

    llm = CachedLLM(cache_dir="./cache")
    retriever = AnchorRetriever(doc, llm=llm)
    retriever.index = idx
    trace = retriever.retrieve(args.query, top_k=5)
    verdict = FactAdjudicator(llm=llm).answer_numerical(doc, args.query, trace.pages)
    print(f"\n[{verdict.status.upper()}] {verdict.answer}")
    print(f"Cited pages: {verdict.cited_pages} | Quote verified: {verdict.quote_verified}")
    out = export_html([verdict], args.html, title=f"AnchorDoc — {Path(args.pdf).name}", doc=doc)
    print(f"Interactive HTML workbench saved to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
