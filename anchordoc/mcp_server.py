"""Zero-LLM 4-tool Progressive MCP Server (`anchordoc mcp`) for AI coding agents.

Exposes four orthogonal tools with strict path allowlisting and zero internal LLM calls:
- `doc_outline`: L0 structural anchor map + route hint (<= 1,500 est_tokens)
- `doc_locate` : L1 zero-LLM query-aware snippet locator (<= 1,200 est_tokens)
- `doc_read`   : L2 anchored page reader `p12:L07` with hard caps & duplicate-read ledger
- `doc_verify` : deterministic quote verifier, page localizer, AST math & operand provenance

Logs go strictly to stderr; stdout is reserved exclusively for MCP JSON-RPC.
Install optional dependency with `pip install 'anchordoc[mcp]'`.
"""
from __future__ import annotations

import json
import sys
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

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "doc_outline",
        "description": "L0 structural map of a PDF (<=1.5k tok): page count, printed page range, route_hint, and zoomable headings.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to a PDF inside allowed --root directories (or omit if doc_id given)."},
                "doc_id": {"type": "string", "description": "Previously opened doc_id returned by doc_outline."},
                "page_range": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
                "depth": {"type": "integer", "minimum": 1, "maximum": 3, "default": 1},
                "max_items": {"type": "integer", "minimum": 1, "maximum": 120, "default": 40},
            },
        },
    },
    {
        "name": "doc_locate",
        "description": "L1 zero-LLM query-aware snippet locator (<=1.2k tok) combining lexical BM25-RRF, printed page refs, and table continuations.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "doc_id": {"type": "string"},
                "query": {"type": "string"},
                "k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
                "page_range": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
                "snippet_chars": {"type": "integer", "minimum": 120, "maximum": 600, "default": 320},
            },
            "required": ["query"],
        },
    },
    {
        "name": "doc_read",
        "description": "L2 anchored page reader (<=5 pages, <=6k tok) returning stable p12:L07 line anchors with a session duplicate-read ledger.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "doc_id": {"type": "string"},
                "pages": {"type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 5},
                "line_range": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
                "max_chars_per_page": {"type": "integer", "minimum": 200, "maximum": 8000, "default": 4000},
                "image": {"type": "string", "enum": ["false", "page", "tiles"], "default": "false"},
                "force": {"type": "boolean", "default": False},
            },
            "required": ["pages"],
        },
    },
    {
        "name": "doc_verify",
        "description": "Deterministic CiteOrDie verifier: checks verbatim quotes, localizes pages, evaluates safe AST math, and verifies operand provenance.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "doc_id": {"type": "string"},
                "quotes": {"type": "array", "items": {"type": "object"}},
                "formula": {"type": "string", "default": ""},
                "cited_pages": {"type": "array", "items": {"type": "integer"}},
            },
        },
    },
]


class AnchorDocMCPApp:
    """Protocol-agnostic MCP application core (100% testable without the optional `mcp` package)."""

    def __init__(
        self,
        roots: Sequence[str | Path],
        store: BundleStore | None = None,
        max_file_mb: float = 100.0,
        backend: str = "pypdfium2",
    ) -> None:
        resolved = [Path(r).expanduser().resolve() for r in roots if str(r).strip()]
        if not resolved:
            resolved = [Path.cwd().resolve()]
        self.roots = resolved
        self.store = store
        self.max_file_bytes = int(max_file_mb * 1024 * 1024)
        self.backend = backend
        self._contexts_by_id: dict[str, DocContext] = {}
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

    def _get_context(self, args: dict[str, Any]) -> DocContext:
        doc_id = str(args.get("doc_id") or "").strip()
        raw_path = str(args.get("path") or "").strip()
        if doc_id and doc_id in self._contexts_by_id:
            return self._contexts_by_id[doc_id]
        if not raw_path:
            raise ValueError("Either a valid `path` or a previously opened `doc_id` is required.")
        p = self._resolve_allowed_pdf(raw_path)
        existing_id = self._id_by_path.get(p)
        if existing_id and existing_id in self._contexts_by_id:
            return self._contexts_by_id[existing_id]
        ctx = DocContext.open(p, store=self.store, backend=self.backend)
        self._contexts_by_id[ctx.doc_id] = ctx
        self._id_by_path[p] = ctx.doc_id
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
            )
        raise ValueError(f"Unknown AnchorDoc MCP tool: {name!r}")


def serve_stdio(
    roots: Sequence[str | Path],
    store_dir: str | Path | None = None,
    max_file_mb: float = 100.0,
) -> int:
    """Run the AnchorDoc MCP server over stdio (requires optional `pip install 'anchordoc[mcp]'`)."""
    try:
        import asyncio
        from mcp import types as mcp_types
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
    except ImportError:
        print(
            "AnchorDoc MCP server requires the optional `mcp` SDK. Install with: pip install 'anchordoc[mcp]'",
            file=sys.stderr,
        )
        return 1

    store = BundleStore(store_dir) if store_dir else BundleStore()
    app = AnchorDocMCPApp(roots=roots, store=store, max_file_mb=max_file_mb)
    server = Server("anchordoc", instructions=MCP_INSTRUCTIONS)

    @server.list_tools()
    async def _list_tools() -> list[mcp_types.Tool]:
        return [
            mcp_types.Tool(
                name=t["name"],
                description=t["description"],
                inputSchema=t["inputSchema"],
            )
            for t in app.list_tools()
        ]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any] | None) -> list[mcp_types.TextContent]:
        result = app.call_tool(name, arguments or {})
        return [mcp_types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]

    async def _run() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    print(f"[anchordoc-mcp] serving roots={[str(r) for r in app.roots]} store={store.root}", file=sys.stderr)
    asyncio.run(_run())
    return 0
