"""Zero-LLM 4-tool Progressive MCP Server (`anchordoc mcp`) for AI coding agents.

Exposes four orthogonal read-only tools with strict path allowlisting and zero internal LLM calls:
- `doc_outline`: L0 structural anchor map + route hint (<= 1,500 est_tokens)
- `doc_locate` : L1 zero-LLM query-aware snippet locator (<= 1,200 est_tokens)
- `doc_read`   : L2 anchored page reader `p12:L07` with hard caps, read ledger & native ImageContent
- `doc_verify` : deterministic quote verifier, page localizer, AST math & operand provenance

Logs go strictly to stderr; stdout is reserved exclusively for MCP JSON-RPC.
Install optional dependency with `pip install 'anchordoc[mcp]'`.
"""
from __future__ import annotations

import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Sequence

from .context import DocContext
from .store.bundle import BundleStore

MCP_INSTRUCTIONS = (
    "AnchorDoc progressive PDF protocol (0 external LLM calls): "
    "1) Call doc_outline once per document to inspect structure, page count, and route_hint. "
    "2) Call doc_locate with specific query terms before reading pages (unless route_hint='read_all'). "
    "3) Call doc_read on at most 5 pages per call; cite exact p.N (printed label) and pN:Lxx anchors. "
    "4) Call doc_verify on verbatim quotes and arithmetic formulas before stating your final answer."
)

_READONLY_ANNOTATIONS = {
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
}

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "doc_outline",
        "description": "Build a token-bounded L0 structural outline of a PDF (<=1,500 tokens, zero LLM calls). Returns doc_id, page count, printed-to-physical page calibration, route_hint ('read_all' or 'navigate'), and zoomable section/statement headings. Call this first once per PDF.",
        "annotations": dict(_READONLY_ANNOTATIONS),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or relative path to a PDF file inside allowed --root directories (required on first call if doc_id is omitted)."},
                "doc_id": {"type": "string", "description": "Opaque document handle returned by an earlier doc_outline call to avoid re-specifying path."},
                "page_range": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2, "description": "Optional [start_page, end_page] 1-based physical page bounds to zoom into a specific section."},
                "depth": {"type": "integer", "minimum": 1, "maximum": 3, "default": 1, "description": "Heading granularity depth (1 = top-level sections & primary statements, 2–3 = finer subsections)."},
                "max_items": {"type": "integer", "minimum": 1, "maximum": 120, "default": 40, "description": "Maximum outline items returned to keep token usage bounded."},
            },
        },
    },
    {
        "name": "doc_locate",
        "description": "Locate the most relevant pages and snippets in a PDF without calling any LLM (<=1,200 tokens). Combines lexical BM25-RRF, printed page references, statement-intent boosting, and table continuations. Call after doc_outline before reading pages.",
        "annotations": dict(_READONLY_ANNOTATIONS),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to a PDF inside allowed --root directories (optional if doc_id is supplied)."},
                "doc_id": {"type": "string", "description": "Previously opened doc_id returned by doc_outline."},
                "query": {"type": "string", "description": "Specific search query containing key entities, metrics, table names, or clause terms."},
                "k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5, "description": "Number of top candidate pages to return (1–10)."},
                "page_range": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2, "description": "Optional [start_page, end_page] 1-based physical page range to restrict search."},
                "snippet_chars": {"type": "integer", "minimum": 120, "maximum": 600, "default": 320, "description": "Maximum character length of each returned preview snippet."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "doc_read",
        "description": "Read up to 5 specific PDF pages per call (<=6,000 tokens) with stable line anchors (e.g. p12:L07) and a session duplicate-read ledger. Optionally renders page PNGs or 2x2 zoom tiles as native MCP ImageContent for charts or scanned pages.",
        "annotations": dict(_READONLY_ANNOTATIONS),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to a PDF inside allowed --root directories (optional if doc_id is supplied)."},
                "doc_id": {"type": "string", "description": "Previously opened doc_id returned by doc_outline."},
                "pages": {"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 5, "description": "1-based physical page numbers to read (1 to 5 pages per call)."},
                "line_range": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2, "description": "Optional [start_line, end_line] 1-based line bounds within each requested page."},
                "max_chars_per_page": {"type": "integer", "minimum": 200, "maximum": 8000, "default": 4000, "description": "Character cap per page to protect context window budget."},
                "image": {"type": "string", "enum": ["false", "page", "tiles"], "default": "false", "description": "Visual rendering mode: 'false' (text only), 'page' (full-page PNG), or 'tiles' (2x2 high-res zoom crops)."},
                "force": {"type": "boolean", "default": False, "description": "Set true to re-return pages already recorded in the session read ledger."},
            },
            "required": ["pages"],
        },
    },
    {
        "name": "doc_verify",
        "description": "Deterministically verify any claim against a source PDF with zero LLM calls. Checks sign-preserving verbatim quotes, binds numeric operands to table rows/periods/units, detects sign flips, period/unit/pp-vs-% swaps, and recomputes symbolic or numeric formulas via Python AST. Always call before finalizing high-stakes answers.",
        "annotations": dict(_READONLY_ANNOTATIONS),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to a PDF inside allowed --root directories (optional if doc_id is supplied)."},
                "doc_id": {"type": "string", "description": "Previously opened doc_id returned by doc_outline."},
                "claim_text": {"type": "string", "default": "", "description": "Optional natural-language claim to audit for metric/period/pp-vs-% alignment."},
                "quotes": {
                    "type": "array",
                    "items": {"oneOf": [{"type": "string"}, {"type": "object"}]},
                    "description": "Verbatim quote strings or {text, page?} objects to verify against exact PDF text.",
                },
                "formula": {"type": "string", "default": "", "description": "Symbolic (e.g. 'rev - cogs') or numeric arithmetic formula recomputed deterministically via Python AST."},
                "claimed_value": {"type": ["number", "string"], "description": "Optional claimed numeric result to check against AST recomputation."},
                "claimed_unit": {"type": "string", "default": "", "description": "Optional unit of claimed_value (e.g., million, billion, %, pp)."},
                "operands": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Symbolic variable name used in formula (e.g. 'rev')."},
                            "value": {"type": ["number", "string"], "description": "Numeric operand value extracted from the PDF."},
                            "unit": {"type": "string", "description": "Unit/scale of the operand (e.g. 'million', 'thousand', '%')."},
                            "row_label": {"type": "string", "description": "Table row or line-item label where the operand appears."},
                            "period": {"type": "string", "description": "Reporting period or column header (e.g. 'FY2024')."},
                            "page": {"type": "integer", "description": "1-based physical page number containing this operand."},
                        },
                        "required": ["value"],
                    },
                    "description": "Optional structured operands bound to row_label, period, unit, and page for anti-laundering verification.",
                },
                "cited_pages": {"type": "array", "items": {"type": "integer"}, "description": "Optional list of 1-based physical page numbers cited as evidence."},
                "strict_scale": {"type": "boolean", "default": False, "description": "If true, disallow implicit 1000x/1e6x unit scaling during numeric matching."},
            },
        },
    },
]


def split_mcp_payload(result: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Extract base64 PNGs out of a `doc_read` result so images are sent as MCP ImageContent blocks,
    NEVER stuffed as hundreds of thousands of base64 characters inside JSON TextContent.
    """
    images: list[str] = []
    if not isinstance(result, dict) or not isinstance(result.get("pages"), list):
        return result, images
    clean_pages: list[dict[str, Any]] = []
    for block in result["pages"]:
        if isinstance(block, dict) and "images_b64" in block:
            b_copy = {k: v for k, v in block.items() if k != "images_b64"}
            imgs = [str(x) for x in (block.get("images_b64") or []) if x]
            images.extend(imgs)
            b_copy["images_returned"] = len(imgs)
            clean_pages.append(b_copy)
        else:
            clean_pages.append(block)
    clean_result = dict(result)
    clean_result["pages"] = clean_pages
    return clean_result, images


class AnchorDocMCPApp:
    """Protocol-agnostic MCP application core (100% testable without the optional `mcp` package)."""

    def __init__(
        self,
        roots: Sequence[str | Path],
        store: BundleStore | None = None,
        max_file_mb: float = 100.0,
        backend: str = "pypdfium2",
        max_open_docs: int = 16,
    ) -> None:
        resolved = [Path(r).expanduser().resolve() for r in roots if str(r).strip()]
        if not resolved:
            resolved = [Path.cwd().resolve()]
        self.roots = resolved
        self.store = store
        self.max_file_bytes = int(max_file_mb * 1024 * 1024)
        self.backend = backend
        self.max_open_docs = max(1, int(max_open_docs))
        # LRU cache: doc_id -> (DocContext, path, (size, mtime_ns))
        self._contexts: OrderedDict[str, tuple[DocContext, Path, tuple[int, int]]] = OrderedDict()
        self._id_by_path: dict[Path, str] = {}

    def list_tools(self) -> list[dict[str, Any]]:
        return list(TOOL_DEFINITIONS)

    def _resolve_allowed_pdf(self, raw_path: str) -> Path:
        p = Path(raw_path).expanduser().resolve()
        if not any(p == r or r in p.parents for r in self.roots):
            raise PermissionError(f"Access denied: {raw_path!r} is outside allowed --root directories.")
        if p.suffix.lower() != ".pdf":
            raise ValueError(f"Only .pdf files are supported (got {p.name!r}).")
        if not p.is_file():
            raise FileNotFoundError(f"PDF not found: {p}")
        if p.stat().st_size > self.max_file_bytes:
            raise ValueError(f"PDF exceeds size limit ({self.max_file_bytes // (1024 * 1024)} MB).")
        return p

    @staticmethod
    def _file_sig(p: Path) -> tuple[int, int]:
        st = p.stat()
        return (st.st_size, st.st_mtime_ns)

    def _evict_if_needed(self) -> None:
        while len(self._contexts) > self.max_open_docs:
            old_id, (_, old_path, _) = self._contexts.popitem(last=False)
            if self._id_by_path.get(old_path) == old_id:
                self._id_by_path.pop(old_path, None)

    def _get_context(self, args: dict[str, Any]) -> DocContext:
        doc_id = str(args.get("doc_id") or "").strip()
        raw_path = str(args.get("path") or "").strip()

        if doc_id and doc_id in self._contexts:
            ctx, p, sig = self._contexts[doc_id]
            if p.is_file() and self._file_sig(p) == sig:
                self._contexts.move_to_end(doc_id)
                return ctx
            # File changed on disk: reload fresh
            self._contexts.pop(doc_id, None)
            raw_path = str(p)

        if not raw_path:
            raise ValueError("Either a valid `path` or a previously opened `doc_id` is required.")
        p = self._resolve_allowed_pdf(raw_path)
        sig = self._file_sig(p)
        existing_id = self._id_by_path.get(p)
        if existing_id and existing_id in self._contexts:
            ctx, _, old_sig = self._contexts[existing_id]
            if old_sig == sig:
                self._contexts.move_to_end(existing_id)
                return ctx
            self._contexts.pop(existing_id, None)

        ctx = DocContext.open(p, store=self.store, backend=self.backend)
        self._contexts[ctx.doc_id] = (ctx, p, sig)
        self._id_by_path[p] = ctx.doc_id
        self._evict_if_needed()
        return ctx

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        args = arguments or {}
        if name == "doc_outline":
            ctx = self._get_context(args)
            return ctx.outline(
                page_range=args.get("page_range"),
                depth=int(args.get("depth", 1)),
                max_items=int(args.get("max_items", 40)),
            )
        if name == "doc_locate":
            ctx = self._get_context(args)
            return ctx.locate(
                query=str(args.get("query") or ""),
                k=int(args.get("k", 5)),
                page_range=args.get("page_range"),
                snippet_chars=int(args.get("snippet_chars", 320)),
            )
        if name == "doc_read":
            ctx = self._get_context(args)
            img_arg = args.get("image", False)
            if isinstance(img_arg, str) and img_arg.lower() in {"false", "none", ""}:
                img_arg = False
            return ctx.read(
                pages=args.get("pages") or [],
                line_range=args.get("line_range"),
                max_chars_per_page=int(args.get("max_chars_per_page", 4000)),
                image=img_arg,
                force=bool(args.get("force", False)),
            )
        if name == "doc_verify":
            ctx = self._get_context(args)
            return ctx.verify(
                quotes=args.get("quotes") or (),
                formula=str(args.get("formula") or ""),
                cited_pages=args.get("cited_pages") or (),
                claim_text=str(args.get("claim_text") or ""),
                claimed_value=args.get("claimed_value"),
                claimed_unit=str(args.get("claimed_unit") or ""),
                operands=args.get("operands") or (),
                strict_scale=bool(args.get("strict_scale", False)),
            )
        raise ValueError(f"Unknown AnchorDoc MCP tool: {name!r}")


def _build_mcp_server(app: AnchorDocMCPApp):
    """Instantiate the official MCP `Server` with graceful compatibility across SDK versions."""
    import anyio
    from mcp import types as mcp_types
    from mcp.server import Server

    try:
        server = Server("anchordoc", instructions=MCP_INSTRUCTIONS)
    except TypeError:
        server = Server("anchordoc")

    @server.list_tools()
    async def _list_tools() -> list[mcp_types.Tool]:
        out: list[mcp_types.Tool] = []
        for t in app.list_tools():
            kwargs: dict[str, Any] = {
                "name": t["name"],
                "description": t["description"],
                "inputSchema": t["inputSchema"],
            }
            if hasattr(mcp_types, "ToolAnnotations"):
                kwargs["annotations"] = mcp_types.ToolAnnotations(**t["annotations"])
            try:
                out.append(mcp_types.Tool(**kwargs))
            except TypeError:
                kwargs.pop("annotations", None)
                out.append(mcp_types.Tool(**kwargs))
        return out

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any] | None):
        raw_result = await anyio.to_thread.run_sync(app.call_tool, name, arguments or {})
        clean_result, images_b64 = split_mcp_payload(raw_result)
        blocks: list[Any] = [
            mcp_types.TextContent(type="text", text=json.dumps(clean_result, ensure_ascii=False))
        ]
        for b64 in images_b64:
            blocks.append(mcp_types.ImageContent(type="image", data=b64, mimeType="image/png"))
        return blocks

    return server


def serve_stdio(
    roots: Sequence[str | Path],
    store_dir: str | Path | None = None,
    max_file_mb: float = 100.0,
) -> int:
    """Run the AnchorDoc MCP server over stdio (requires optional `pip install 'anchordoc[mcp]'`)."""
    try:
        import asyncio
        from mcp.server.stdio import stdio_server
    except ImportError:
        print(
            "AnchorDoc MCP server requires the optional `mcp` SDK. Install with: pip install 'anchordoc[mcp]'",
            file=sys.stderr,
        )
        return 1

    store = BundleStore(store_dir) if store_dir else BundleStore()
    app = AnchorDocMCPApp(roots=roots, store=store, max_file_mb=max_file_mb)
    server = _build_mcp_server(app)

    async def _run() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    print(f"[anchordoc-mcp] serving roots={[str(r) for r in app.roots]} store={store.root}", file=sys.stderr)
    asyncio.run(_run())
    return 0
