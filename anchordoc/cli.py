"""Command-line interface for AnchorDoc (`anchordoc outline | query | audit | mcp | bundle`)."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .adjudicate import FactAdjudicator
from .docbase import PdfDocument
from .index import DocumentIndex
from .mcp_server import serve_stdio
from .report import export_html, export_json, export_markdown
from .retrieve import AnchorRetriever
from .runtime import CachedLLM
from .store import BundleStore, load_document
from .types import AuditVerdict


def _load_doc(pdf: str, backend: str, store_dir: str) -> PdfDocument:
    store = BundleStore(store_dir) if store_dir else None
    return load_document(pdf, backend=backend, store=store)


def cmd_outline(args: argparse.Namespace) -> int:
    """Zero-LLM structural inspection: prints AnchorMap & G7 TOC offset in milliseconds."""
    doc = _load_doc(args.pdf, backend=args.backend, store_dir=args.store)
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
    doc = _load_doc(args.pdf, backend=args.backend, store_dir=args.store)
    llm = CachedLLM(cache_dir=args.cache_dir, model=args.model or None)
    retriever = AnchorRetriever(doc, llm=llm)
    task = "qa_multimodal" if args.vision else "qa_text"
    trace = retriever.retrieve(
        args.question,
        top_k=args.top_k,
        use_anchor_map=True,
        use_gap_followup=not args.no_followup,
        route=args.route,
        task=task,
    )
    adj = FactAdjudicator(llm=llm)
    if args.vision:
        verdict = adj.answer_multimodal(doc, args.question, trace.pages, item_id="q1", route=trace.route)  # type: ignore[arg-type]
    else:
        verdict = adj.answer_numerical(doc, args.question, trace.pages, item_id="q1", route=trace.route)  # type: ignore[arg-type]
    verdict.retrieval_trace = asdict(trace)

    print(f"Route       : {trace.route} ({trace.route_reason})")
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
    doc = _load_doc(args.pdf, backend=args.backend, store_dir=args.store)
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
        trace = retriever.retrieve(f"{title} {req}", top_k=args.top_k, route=args.route)
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


def cmd_mcp(args: argparse.Namespace) -> int:
    """Start the zero-LLM 4-tool AnchorDoc MCP stdio server."""
    roots = args.root or [str(Path.cwd())]
    return serve_stdio(roots=roots, store_dir=args.store or None, max_file_mb=args.max_file_mb)


def cmd_bundle(args: argparse.Namespace) -> int:
    """Inspect or garbage-collect persistent SQLite WAL document bundles."""
    store = BundleStore(args.store or None)
    if args.bundle_cmd == "info":
        print(json.dumps(store.info(), ensure_ascii=False, indent=2))
        return 0
    if args.bundle_cmd == "gc":
        print(json.dumps(store.gc(max_age_days=args.max_age_days), ensure_ascii=False, indent=2))
        return 0
    return 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="anchordoc", description="Zero-index-cost verifiable document auditing CLI.")
    sub = ap.add_subparsers(dest="command", required=True)

    p_out = sub.add_parser("outline", help="Print Zero-LLM Structural Anchor Map and G7 TOC offset (no API key needed).")
    p_out.add_argument("pdf", help="Path to PDF file.")
    p_out.add_argument("--backend", default="pypdfium2", choices=["pypdfium2", "pymupdf"])
    p_out.add_argument("--max-anchors", type=int, default=95)
    p_out.add_argument("--store", default="", help="Optional BundleStore directory for persistent text caching.")
    p_out.set_defaults(func=cmd_outline)

    p_q = sub.add_parser("query", help="Ask a verifiable question over a long PDF.")
    p_q.add_argument("pdf", help="Path to PDF file.")
    p_q.add_argument("question", help="Question to verify.")
    p_q.add_argument("--vision", action="store_true", help="Render retrieved pages as PNGs for multimodal VLM verification.")
    p_q.add_argument("--top-k", type=int, default=5)
    p_q.add_argument("--no-followup", action="store_true")
    p_q.add_argument("--route", default="auto", choices=["auto", "read_all", "full_pool", "navigate"])
    p_q.add_argument("--store", default="", help="Optional BundleStore directory.")
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
    p_a.add_argument("--route", default="auto", choices=["auto", "read_all", "full_pool", "navigate"])
    p_a.add_argument("--store", default="", help="Optional BundleStore directory.")
    p_a.add_argument("--backend", default="pypdfium2", choices=["pypdfium2", "pymupdf"])
    p_a.add_argument("--model", default="")
    p_a.add_argument("--cache-dir", default="./cache")
    p_a.add_argument("--json", default="")
    p_a.add_argument("--markdown", default="")
    p_a.add_argument("--html", default="")
    p_a.set_defaults(func=cmd_audit)

    p_m = sub.add_parser("mcp", help="Run the zero-LLM 4-tool AnchorDoc MCP server over stdio.")
    p_m.add_argument("--root", action="append", default=[], help="Allowed root directory (may be specified multiple times).")
    p_m.add_argument("--store", default="", help="Optional BundleStore directory.")
    p_m.add_argument("--max-file-mb", type=float, default=100.0)
    p_m.set_defaults(func=cmd_mcp)

    p_b = sub.add_parser("bundle", help="Inspect or garbage-collect persistent SQLite WAL bundles.")
    b_sub = p_b.add_subparsers(dest="bundle_cmd", required=True)
    p_bi = b_sub.add_parser("info", help="List stored document bundles and sizes.")
    p_bi.add_argument("--store", default="")
    p_bi.set_defaults(func=cmd_bundle)
    p_bg = b_sub.add_parser("gc", help="Delete bundles older than --max-age-days.")
    p_bg.add_argument("--store", default="")
    p_bg.add_argument("--max-age-days", type=float, default=30.0)
    p_bg.set_defaults(func=cmd_bundle)

    return ap


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
