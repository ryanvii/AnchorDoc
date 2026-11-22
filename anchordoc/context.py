"""Zero-LLM progressive document context interface (`DocContext`) for AI agents and MCP.

Implements 3-layer progressive disclosure + deterministic verification with strict token caps:
- L0 `outline(...)` : Zoomable structural map (<= 1,500 est_tokens) + `route_hint`.
- L1 `locate(...)`  : Zero-LLM query-aware snippet locator (<= 1,200 est_tokens).
- L2 `read(...)`    : Anchored lines `p12:L07` (<= 5 pages, <= 6,000 est_tokens, read ledger).
-    `verify(...)`  : Deterministic quote verification, page localization, AST math & operand provenance.

Requires ZERO LLM API keys and ZERO MCP SDK dependency.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

from .adjudicate.answer_contract import operands_grounded
from .adjudicate.calc import safe_calc
from .docbase.pdf_loader import PdfDocument
from .index.anchor_map import DocumentIndex
from .retrieve.pipeline import AnchorRetriever
from .retrieve.router import plan_route, profile_document
from .runtime.text import estimate_tokens, localize_quote_pages, verify_quote
from .store.bundle import BundleStore, load_document

MAX_OUTLINE_TOKENS = 1_500
MAX_LOCATE_TOKENS = 1_200
MAX_READ_PAGES = 5
MAX_READ_TOKENS = 6_000
MAX_READ_IMAGES = 2


def _page_lines(text: str) -> list[str]:
    return [ln.rstrip() for ln in (text or "").splitlines() if ln.strip()]


def _clamp_range(page_range: Sequence[int] | None, n_pages: int) -> tuple[int, int]:
    if not page_range or len(page_range) != 2:
        return 1, n_pages
    a = max(1, min(n_pages, int(page_range[0])))
    b = max(a, min(n_pages, int(page_range[1])))
    return a, b


class DocContext:
    """Stateful, zero-LLM progressive reader over a single PDF with a session read ledger."""

    def __init__(
        self,
        doc: PdfDocument,
        doc_id: str = "",
        vision_summaries: dict[int, str] | None = None,
    ) -> None:
        self.doc = doc
        self.doc_id = doc_id or doc.sha256[:12] or hashlib.sha256(
            (str(doc.path or "memory") + f":{doc.num_pages}").encode("utf-8")
        ).hexdigest()[:12]
        self.retriever = AnchorRetriever(doc, llm=None, vision_summaries=vision_summaries)
        self.index: DocumentIndex = self.retriever.index
        self.toc_pages = self.index.find_printed_toc_pages()
        toc_joined = "\n\n".join(self.doc.page_text(p) for p in self.toc_pages)[:4500] if self.toc_pages else ""
        self.g7_offset = self.index.calibrate_printed_offset(toc_joined) if toc_joined else 0
        self.printed_map = self.index.printed_page_map()
        self._read_ledger: set[tuple[int, tuple[int, int] | None]] = set()

    @classmethod
    def open(
        cls,
        path: str | Path,
        store: BundleStore | None = None,
        backend: str = "pypdfium2",
        vision_summaries: dict[int, str] | None = None,
    ) -> "DocContext":
        doc = load_document(path, backend=backend, store=store)
        return cls(doc=doc, doc_id=(doc.sha256[:12] if doc.sha256 else ""), vision_summaries=vision_summaries)

    @classmethod
    def from_document(
        cls,
        doc: PdfDocument,
        doc_id: str = "",
        vision_summaries: dict[int, str] | None = None,
    ) -> "DocContext":
        return cls(doc=doc, doc_id=doc_id, vision_summaries=vision_summaries)

    def _page_tier(self, p: int) -> int:
        c = self.index.cards[p - 1]
        if p in self.toc_pages:
            return 0
        if c.section_path and (p == 1 or self.index.cards[p - 2].section_path != c.section_path):
            return 1
        if c.has_table or c.table_cues or c.vision_summary:
            return 2
        return 3

    def _page_tags(self, p: int) -> list[str]:
        c = self.index.cards[p - 1]
        tags: list[str] = []
        if p in self.toc_pages:
            tags.append("TOC")
        if c.has_table:
            tags.append("TABLE")
        if c.vision_summary:
            tags.append("VISION")
        elif len((self.doc.page_text(p) or "").strip()) < 180:
            tags.append("LOW_TEXT")
        return tags

    def _doc_kind(self, n_visual: int) -> str:
        if self.doc.num_pages == 0 or n_visual == 0:
            return "text"
        if n_visual >= max(1, int(0.75 * self.doc.num_pages)):
            return "scanned"
        return "multimodal"

    # ------------------------------------------------------------------
    # L0: Zoomable Structural Outline
    # ------------------------------------------------------------------
    def outline(
        self,
        page_range: Sequence[int] | None = None,
        depth: int = 1,
        max_items: int = 40,
    ) -> dict[str, Any]:
        """Return a token-bounded structural map (<= 1,500 est_tokens) + routing hint."""
        depth = max(1, min(3, int(depth)))
        max_items = max(1, min(120, int(max_items)))
        lo, hi = _clamp_range(page_range, self.doc.num_pages)

        prof = profile_document(self.doc, vision_summaries=self.index.vision_summaries)
        kind = self._doc_kind(prof.n_visual_pages)
        task = "qa_multimodal" if kind != "text" else "qa_text"
        plan = plan_route(prof, task=task)

        # Tier cutoff per depth: depth=1 -> tiers 0..1 (+fallback to 2 if sparse); depth=2 -> 0..2; depth=3 -> all
        max_tier = 1 if depth == 1 else (2 if depth == 2 else 3)
        raw_items: list[dict[str, Any]] = []
        for p in range(lo, hi + 1):
            tier = self._page_tier(p)
            if tier > max_tier:
                continue
            c = self.index.cards[p - 1]
            title = (c.section_path or c.heading or f"Page {p}")[:110]
            raw_items.append({
                "p": p,
                "label": self.printed_map.get(p, str(p)),
                "tier": tier,
                "title": title,
                "tags": self._page_tags(p),
            })

        if not raw_items and self.doc.num_pages > 0:
            # Uniformly sample anchor pages when no headings match tier cutoff
            step = max(1, (hi - lo + 1) // min(max_items, max(1, hi - lo + 1)))
            for p in range(lo, hi + 1, step):
                c = self.index.cards[p - 1]
                raw_items.append({
                    "p": p,
                    "label": self.printed_map.get(p, str(p)),
                    "tier": self._page_tier(p),
                    "title": (c.heading or f"Page {p}")[:110],
                    "tags": self._page_tags(p),
                })

        items: list[dict[str, Any]] = []
        used_tok = 90  # header overhead
        truncated = False
        for it in raw_items:
            if len(items) >= max_items:
                truncated = True
                break
            cost = estimate_tokens(f"{it['p']} {it['label']} {it['title']} {' '.join(it['tags'])}") + 10
            if used_tok + cost > MAX_OUTLINE_TOKENS and items:
                truncated = True
                break
            items.append(it)
            used_tok += cost

        next_cursor = None
        if truncated and items and items[-1]["p"] < hi:
            next_cursor = {"page_range": [items[-1]["p"] + 1, hi], "depth": depth}

        printed_vals = [self.printed_map[p] for p in range(lo, hi + 1) if p in self.printed_map]
        printed_range = f"{printed_vals[0]}..{printed_vals[-1]}" if printed_vals else f"{lo}..{hi}"
        return {
            "doc_id": self.doc_id,
            "path": str(self.doc.path.name) if self.doc.path else "memory",
            "n_pages": self.doc.num_pages,
            "page_range": [lo, hi],
            "printed_range": printed_range,
            "kind": kind,
            "est_full_tokens": prof.est_text_tokens,
            "route_hint": plan.route,
            "toc_pages": list(self.toc_pages),
            "g7_offset": self.g7_offset,
            "items": items,
            "est_tokens": used_tok,
            "truncated": truncated,
            "next": next_cursor,
        }

    # ------------------------------------------------------------------
    # L1: Zero-LLM Query-Aware Snippet Locator
    # ------------------------------------------------------------------
    def locate(
        self,
        query: str,
        k: int = 5,
        page_range: Sequence[int] | None = None,
        snippet_chars: int = 320,
    ) -> dict[str, Any]:
        """Locate top-k candidate pages with short query-relevant snippets (<= 1,200 est_tokens)."""
        k = max(1, min(10, int(k)))
        snippet_chars = max(120, min(600, int(snippet_chars)))
        lo, hi = _clamp_range(page_range, self.doc.num_pages)
        allowed = set(range(lo, hi + 1))

        page_refs = [p for p in self.index.resolve_page_refs(query) if p in allowed]
        if page_range is not None:
            lex = self.index.demote_toc(self.index.rank_pages_subset(query, sorted(allowed), top_k=max(k * 2, 10)))
        else:
            lex = [p for p in self.retriever._lexical(query, max(k * 2, 10)) if p in allowed]

        reasons: dict[int, list[str]] = {}
        ordered: list[int] = []
        for p in page_refs[:2]:
            reasons.setdefault(p, []).append("page_ref")
            if p not in ordered:
                ordered.append(p)
        for p in lex:
            reasons.setdefault(p, []).append("lexical")
            if p not in ordered:
                ordered.append(p)
        for p in list(ordered[:3]):
            for nb in self.index.continuation_neighbors(p):
                if nb in allowed:
                    reasons.setdefault(nb, []).append("continuation")
                    if nb not in ordered:
                        ordered.append(nb)

        weights = self.retriever._term_weights(query)
        hits: list[dict[str, Any]] = []
        used_tok = 50
        truncated = False
        for p in ordered[:k]:
            c = self.index.cards[p - 1]
            raw_snip = self.retriever._page_snippet(p, weights, budget_chars=snippet_chars, win_chars=min(280, snippet_chars))
            # Strip redundant header line since we return structured fields
            body = "\n".join(raw_snip.splitlines()[1:]).strip()[:snippet_chars]
            cost = estimate_tokens(body) + estimate_tokens(c.section_path or "") + 24
            if used_tok + cost > MAX_LOCATE_TOKENS and hits:
                truncated = True
                break
            hits.append({
                "p": p,
                "label": self.printed_map.get(p, str(p)),
                "section": c.section_path or c.heading,
                "snippet": body,
                "why": reasons.get(p, ["lexical"]),
            })
            used_tok += cost

        if len(ordered) > len(hits) and len(hits) < k:
            truncated = True
        return {
            "doc_id": self.doc_id,
            "query": query,
            "hits": hits,
            "est_tokens": used_tok,
            "truncated": truncated,
            "next": {"k": min(10, k + 5), "page_range": [lo, hi]} if truncated else None,
        }

    # ------------------------------------------------------------------
    # L2: Anchored Page Reader with Hard Caps & Read Ledger
    # ------------------------------------------------------------------
    def read(
        self,
        pages: Sequence[int],
        line_range: Sequence[int] | None = None,
        max_chars_per_page: int = 4000,
        image: bool | str = False,
        force: bool = False,
    ) -> dict[str, Any]:
        """Read up to 5 pages with stable ``p12:L07`` anchors and a duplicate-read ledger."""
        req_pages: list[int] = []
        for p in pages:
            pi = int(p)
            if pi not in req_pages:
                req_pages.append(pi)
        if not req_pages:
            raise ValueError("pages must be a non-empty list of 1-based page numbers.")
        if len(req_pages) > MAX_READ_PAGES:
            raise ValueError(f"At most {MAX_READ_PAGES} pages may be read per call (got {len(req_pages)}).")
        for p in req_pages:
            if p < 1 or p > self.doc.num_pages:
                raise ValueError(f"Page {p} out of range 1..{self.doc.num_pages}.")

        lr_key: tuple[int, int] | None = None
        if line_range is not None:
            if len(line_range) != 2:
                raise ValueError("line_range must be [start_line, end_line] (1-based inclusive).")
            lr_key = (max(1, int(line_range[0])), max(1, int(line_range[1])))

        max_chars = max(200, min(8000, int(max_chars_per_page)))
        img_mode = "page" if image is True else (str(image).strip().lower() if image else "")

        out_pages: list[dict[str, Any]] = []
        used_tok = 40
        images_returned = 0
        truncated = False
        next_cursor: dict[str, Any] | None = None

        budget_hit = False
        for idx_p, p in enumerate(req_pages):
            if budget_hit:
                break
            ledger_key = (p, lr_key)
            if not force and ledger_key in self._read_ledger:
                out_pages.append({
                    "p": p,
                    "label": self.printed_map.get(p, str(p)),
                    "already_returned": True,
                    "hint": "Page already returned in this session; pass force=true to re-read.",
                })
                used_tok += 16
                continue

            lines = _page_lines(self.doc.page_text(p))
            start_ln = lr_key[0] if lr_key else 1
            end_ln = min(len(lines), lr_key[1]) if lr_key else len(lines)
            anchored: list[str] = []
            char_used = 0
            page_trunc = False
            last_ln = start_ln - 1

            for ln_no in range(start_ln, end_ln + 1):
                entry = f"p{p}:L{ln_no:02d} {lines[ln_no - 1]}"
                est = estimate_tokens(entry) + 2
                if used_tok + est > MAX_READ_TOKENS and (anchored or out_pages):
                    page_trunc = True
                    truncated = True
                    budget_hit = True
                    if next_cursor is None:
                        next_cursor = {"pages": [p] + req_pages[idx_p + 1 :], "line_range": [ln_no, len(lines)]}
                    break
                if char_used + len(entry) > max_chars and anchored:
                    page_trunc = True
                    truncated = True
                    if next_cursor is None:
                        next_cursor = {"pages": [p], "line_range": [ln_no, len(lines)]}
                    break
                anchored.append(entry)
                char_used += len(entry) + 1
                used_tok += est
                last_ln = ln_no

            if budget_hit and not anchored:
                break

            block: dict[str, Any] = {
                "p": p,
                "label": self.printed_map.get(p, str(p)),
                "total_lines": len(lines),
                "line_range": [start_ln, last_ln] if anchored else [start_ln, start_ln - 1],
                "text": "\n".join(anchored),
                "truncated": page_trunc,
            }
            if p in self.index.vision_summaries:
                block["vision_card"] = self.index.vision_summaries[p]

            if img_mode in {"page", "tiles"} and images_returned < MAX_READ_IMAGES and self.doc.path and self.doc.path.is_file():
                try:
                    if img_mode == "tiles":
                        tiles = self.doc.render_page_tiles_b64(p, grid=2, overlap=0.1, scale=2.0)
                        allow = max(0, MAX_READ_IMAGES - images_returned)
                        block["images_b64"] = tiles[:allow]
                        images_returned += len(block["images_b64"])
                    else:
                        block["images_b64"] = [self.doc.render_page_png_b64(p, scale=1.25)]
                        images_returned += 1
                except Exception:
                    pass

            if not page_trunc:
                self._read_ledger.add(ledger_key)
            out_pages.append(block)

        return {
            "doc_id": self.doc_id,
            "pages": out_pages,
            "est_tokens": used_tok,
            "truncated": truncated,
            "next": next_cursor,
        }

    # ------------------------------------------------------------------
    # Deterministic Verification Gate (Quote + Math + Operand Provenance)
    # ------------------------------------------------------------------
    def verify(
        self,
        quotes: Sequence[str | dict[str, Any]] = (),
        formula: str = "",
        cited_pages: Sequence[int] = (),
    ) -> dict[str, Any]:
        """Verify verbatim quotes, localize exact pages, evaluate AST formula & operand provenance."""
        quote_results: list[dict[str, Any]] = []
        localized_pages: list[int] = []

        for q in quotes or ():
            if isinstance(q, dict):
                qtext = str(q.get("text") or q.get("quote") or "").strip()
                hint_p = q.get("page")
            else:
                qtext = str(q or "").strip()
                hint_p = None
            if not qtext:
                continue
            found_pages = localize_quote_pages(qtext, self.doc.pages)
            if hint_p is not None:
                try:
                    hp = int(hint_p)
                    ok_hint, ov_hint = verify_quote(qtext, self.doc.page_text(hp))
                except Exception:
                    hp, ok_hint, ov_hint = None, False, 0.0
            else:
                hp, ok_hint, ov_hint = None, bool(found_pages), (1.0 if found_pages else 0.0)
            ok = bool(found_pages) or ok_hint
            # Ignore generic boilerplate quotes matching >3 pages when building grounding scope
            if 0 < len(found_pages) <= 3:
                for fp in found_pages:
                    if fp not in localized_pages:
                        localized_pages.append(fp)
            quote_results.append({
                "text": qtext[:200],
                "verified": ok,
                "hint_page": hp,
                "matched_pages": found_pages,
                "overlap": 1.0 if ok else ov_hint,
            })

        valid_cited = [int(p) for p in (cited_pages or ()) if 1 <= int(p) <= self.doc.num_pages]
        union_pages = list(localized_pages)
        for cp in valid_cited:
            if cp not in union_pages:
                union_pages.append(cp)
        if localized_pages and valid_cited:
            scope_pages, scope_kind = union_pages, "quote_and_cited"
        elif localized_pages:
            scope_pages, scope_kind = localized_pages, "quote_localized"
        elif valid_cited:
            scope_pages, scope_kind = valid_cited, "cited_pages"
        else:
            scope_pages, scope_kind = [], "none"

        calc_val = safe_calc(formula) if formula else None
        if formula:
            if scope_pages:
                scope_text = "\n".join(self.doc.page_text(p) for p in scope_pages)
                op_ok, ungrounded = operands_grounded(formula, scope_text)
            else:
                op_ok, ungrounded = False, [formula]
        else:
            op_ok, ungrounded = True, []

        return {
            "doc_id": self.doc_id,
            "quotes": quote_results,
            "all_quotes_verified": all(r["verified"] for r in quote_results) if quote_results else False,
            "formula": formula,
            "calc_value": calc_val,
            "operands_grounded": op_ok,
            "ungrounded_operands": ungrounded,
            "grounding_scope": scope_kind,
            "grounding_pages": scope_pages,
            "citations": [f"p.{p} (printed {self.printed_map.get(p, str(p))})" for p in scope_pages],
        }
