"""Command-line interface for AnchorDoc (`anchordoc outline | query | audit`)."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .adjudicate import FactAdjudicator
from .docbase import PdfDocument
from .index import DocumentIndex
from .report import export_html, export_json, export_markdown
from .retrieve import AnchorRetriever
from .runtime import CachedLLM
from .types import AuditVerdict


def cmd_outline(args: argparse.Namespace) -> int:
    """Zero-LLM structural inspection: prints AnchorMap & G7 TOC offset in milliseconds."""
    doc = PdfDocument.from_pdf(args.pdf, backend=args.backend)
    idx = DocumentIndex(doc)
    toc_pgs = idx.find_printed_toc_pages()
    offset = idx.calibrate_printed_offset(doc.page_text(toc_pgs[0])) if toc_pgs else 0
    outline = idx.build_structural_outline(max_anchors=args.max_anchors)

    print(f"Document : {doc.path.name if doc.path else args.pdf}")
    print(f"Pages    : {doc.num_pages} (backend={doc.backend})")
    print(f"G7 TOC   : pages={toc_pgs or 'none'} | calibrated printed->physical offset={offset:+d}")
    print(f"Anchors  : {outline.count(chr(10)) + 1 if outline else 0} (0 LLM calls)")
    print("-" * 72)
    print(outline)
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    """Run verifiable single-question retrieval + QA on a PDF."""
    doc = PdfDocument.from_pdf(args.pdf, backend=args.backend)
    llm = CachedLLM(cache_dir=args.cache_dir, model=args.model or None)
    retriever = AnchorRetriever(doc, llm=llm)
    trace = retriever.retrieve(
        args.question,
        top_k=args.top_k,
        use_anchor_map=True,
        use_gap_followup=not args.no_followup,
    )
    adj = FactAdjudicator(llm=llm)
    if args.vision:
        verdict = adj.answer_multimodal(doc, args.question, trace.pages, item_id="q1")
    else:
        verdict = adj.answer_numerical(doc, args.question, trace.pages, item_id="q1")
    verdict.retrieval_trace = asdict(trace)

    print(f"Status      : {verdict.status.upper()} ({verdict.rule_trace})")
    print(f"Cited Pages : {verdict.cited_pages}")
    print(f"Quote OK    : {verdict.quote_verified}")
    print(f"Answer      : {verdict.answer}")
    if verdict.verbatim_quote:
        print(f"Quote       : \"{verdict.verbatim_quote}\"")
    if verdict.reason:
        print(f"Reason      : {verdict.reason}")

    if args.json:
        export_json([verdict], args.json)
        print(f"Saved JSON  : {args.json}")
    if args.html:
        export_html([verdict], args.html, title=f"AnchorDoc Query — {Path(args.pdf).name}", doc=doc)
        print(f"Saved HTML  : {args.html}")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    """Audit a PDF against a JSON checklist and export JSON/Markdown/HTML reports."""
    doc = PdfDocument.from_pdf(args.pdf, backend=args.backend)
    checklist = json.loads(Path(args.checklist).read_text(encoding="utf-8"))
    if isinstance(checklist, dict) and "items" in checklist:
        checklist = checklist["items"]
    if not isinstance(checklist, list):
        raise ValueError("Checklist JSON must be a list of items or {\"items\": [...]}.")

    llm = CachedLLM(cache_dir=args.cache_dir, model=args.model or None)
    retriever = AnchorRetriever(doc, llm=llm)
    adj = FactAdjudicator(llm=llm)
    verdicts: list[AuditVerdict] = []

    for i, item in enumerate(checklist, start=1):
        item_id = str(item.get("id") or f"item_{i:02d}")
        title = str(item.get("title") or item_id)
        req = str(item.get("requirement") or title)
        mandatory = bool(item.get("mandatory", True))
        trace = retriever.retrieve(f"{title} {req}", top_k=args.top_k)
        v = adj.audit_compliance_item(doc, item_id, title, req, trace.pages, mandatory=mandatory)
        v.retrieval_trace = asdict(trace)
        verdicts.append(v)
        print(f"[{v.status.upper():12s}] {v.item_id} | {v.title} | pages={v.cited_pages} | qv={v.quote_verified}")

    if args.json:
        export_json(verdicts, args.json)
        print(f"Saved JSON  : {args.json}")
    if args.markdown:
        Path(args.markdown).write_text(export_markdown(verdicts, title=f"AnchorDoc Audit — {Path(args.pdf).name}"), encoding="utf-8")
        print(f"Saved MD    : {args.markdown}")
    if args.html:
        export_html(verdicts, args.html, title=f"AnchorDoc Audit — {Path(args.pdf).name}", doc=doc)
        print(f"Saved HTML  : {args.html}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="anchordoc", description="Zero-index-cost verifiable document auditing CLI.")
    sub = ap.add_subparsers(dest="command", required=True)

    p_out = sub.add_parser("outline", help="Print Zero-LLM Structural Anchor Map and G7 TOC offset (no API key needed).")
    p_out.add_argument("pdf", help="Path to PDF file.")
    p_out.add_argument("--backend", default="pypdfium2", choices=["pypdfium2", "pymupdf"])
    p_out.add_argument("--max-anchors", type=int, default=95)
    p_out.set_defaults(func=cmd_outline)

    p_q = sub.add_parser("query", help="Ask a verifiable question over a long PDF.")
    p_q.add_argument("pdf", help="Path to PDF file.")
    p_q.add_argument("question", help="Question to verify.")
    p_q.add_argument("--vision", action="store_true", help="Render retrieved pages as PNGs for multimodal VLM verification.")
    p_q.add_argument("--top-k", type=int, default=5)
    p_q.add_argument("--no-followup", action="store_true")
    p_q.add_argument("--backend", default="pypdfium2", choices=["pypdfium2", "pymupdf"])
    p_q.add_argument("--model", default="")
    p_q.add_argument("--cache-dir", default="./cache")
    p_q.add_argument("--json", default="", help="Optional JSON report output path.")
    p_q.add_argument("--html", default="", help="Optional interactive single-file HTML report output path.")
    p_q.set_defaults(func=cmd_query)

    p_a = sub.add_parser("audit", help="Audit a PDF against a JSON checklist.")
    p_a.add_argument("pdf", help="Path to PDF file.")
    p_a.add_argument("--checklist", required=True, help="Path to checklist JSON file.")
    p_a.add_argument("--top-k", type=int, default=5)
    p_a.add_argument("--backend", default="pypdfium2", choices=["pypdfium2", "pymupdf"])
    p_a.add_argument("--model", default="")
    p_a.add_argument("--cache-dir", default="./cache")
    p_a.add_argument("--json", default="")
    p_a.add_argument("--markdown", default="")
    p_a.add_argument("--html", default="")
    p_a.set_defaults(func=cmd_audit)

    return ap


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
