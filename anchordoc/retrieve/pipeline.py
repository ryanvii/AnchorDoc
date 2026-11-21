"""Unified Zero-Index-Cost Retrieval Pipelines with Layer-by-Layer Audit Trace.

Combines:
- Zero-LLM Structural AnchorMap
- Printed-TOC (G7) navigation with printed-to-physical offset calibration
- Query expansion (C3) + Dual BM25-RRF
- Two-Step Gap-Followup (B42)
- Monotonic RRF fusion (never drops verified strong evidence pages)
"""
from __future__ import annotations

from ..docbase.pdf_loader import PdfDocument
from ..index.anchor_map import DocumentIndex
from ..index.spancard import SpanCardIndex
from ..runtime.llm import LLMClient
from ..runtime.text import rrf
from ..types import RetrievalTrace, SpanCard


class AnchorRetriever:
    """Page-level retriever for 100-600 page PDFs with full RetrievalTrace."""

    def __init__(
        self,
        doc: PdfDocument,
        llm: LLMClient | None = None,
        vision_summaries: dict[int, str] | None = None,
    ) -> None:
        self.doc = doc
        self.index = DocumentIndex(doc, vision_summaries=vision_summaries)
        self.llm = llm

    def _pack_pages(self, pages: list[int], char_limit: int = 1600) -> str:
        blocks = []
        for p in pages:
            c = self.index.cards[p - 1]
            blocks.append(
                f"[Page {p}] Section: {c.section_path or 'N/A'} | Heading: {c.heading}\n"
                f"{self.doc.page_text(p)[:char_limit]}"
            )
        return "\n\n---\n\n".join(blocks)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_anchor_map: bool = True,
        use_gap_followup: bool = True,
    ) -> RetrievalTrace:
        """Execute full C3 + AnchorMap + G7 + B42 monotonic pipeline."""
        trace = RetrievalTrace(query=query)
        toc_pgs = self.index.find_printed_toc_pages()
        trace.g7_toc_pages = toc_pgs

        # Zero-LLM fallback mode when no LLM is supplied
        if self.llm is None:
            rrf_pages = self.index.rank_pagecard_rrf(query, top_k=top_k, exclude_toc=True)
            trace.pages = rrf_pages
            trace.layers["pagecard_rrf"] = list(rrf_pages)
            return trace

        # Layer 1 (C3): Query Expansion + Dual BM25-RRF
        exp, _ = self.llm.chat_json(
            "anchordoc_qexp",
            "You are a precision document retrieval planner. Return JSON only.",
            (
                f"Query: {query}\n"
                "Generate concise keywords targeting exact section titles, table names, line items, "
                "dates/years, or compliance obligation terms.\n"
                'Return JSON: {"keywords": ["..."], "section_hint": "..."}'
            ),
            max_tokens=180,
        )
        kws = exp.get("keywords") if isinstance(exp.get("keywords"), list) else []
        hint = str(exp.get("section_hint") or "")
        trace.expanded_keywords = [str(x) for x in kws]
        q_exp = f"{query} {' '.join(trace.expanded_keywords)} {hint}"

        cand_rrf = self.index.rank_pagecard_rrf(q_exp, top_k=10, exclude_toc=True)
        trace.layers["c3_rrf"] = list(cand_rrf)

        # Layer 2 (C3 Triage): Inspect top-5 candidate pages
        top5 = cand_rrf[:5]
        tri, _ = self.llm.chat_json(
            "anchordoc_triage",
            "You are a strict evidence auditor. Return JSON only.",
            (
                f"Query: {query}\n\nCandidate Pages:\n{self._pack_pages(top5, 1600)}\n\n"
                "Select pages directly containing the required evidence. If evidence is missing or "
                "incomplete, set complete=false and provide targeted missing_terms.\n"
                'Return JSON: {"selected_pages": [int], "complete": true/false, "missing_terms": ["..."]}'
            ),
            max_tokens=220,
        )
        sel = [
            int(p) for p in (tri.get("selected_pages") or [])
            if isinstance(p, (int, float)) and 1 <= int(p) <= self.doc.num_pages
        ]
        complete = bool(tri.get("complete"))

        # Layer 3 (B42): Two-Step Gap-Followup if incomplete or empty
        if use_gap_followup and (not complete or not sel) and (tri.get("missing_terms") or len(cand_rrf) > 5):
            trace.followup_triggered = True
            miss = tri.get("missing_terms") if isinstance(tri.get("missing_terms"), list) else []
            q2 = f"{' '.join(str(x) for x in miss)} {hint} {query}"
            c2 = self.index.rank_pagecard_rrf(q2, top_k=12, exclude_toc=True)
            adj = [p + dp for p in sel[:2] for dp in (-1, 1) if 1 <= p + dp <= self.doc.num_pages]
            extra_pool: list[int] = []
            for p in adj + c2 + cand_rrf[5:]:
                if p not in top5 and p not in extra_pool:
                    extra_pool.append(p)
                if len(extra_pool) >= 5:
                    break
            if extra_pool:
                tri2, _ = self.llm.chat_json(
                    "anchordoc_followup",
                    "You are a strict evidence auditor. Return JSON only.",
                    (
                        f"Query: {query}\nMissing terms from round 1: {miss}\n\n"
                        f"Additional Candidate Pages:\n{self._pack_pages(extra_pool, 1600)}\n\n"
                        "Which of these pages contain the exact missing evidence?\n"
                        'Return JSON: {"selected_pages": [int]}'
                    ),
                    max_tokens=160,
                )
                for p in (tri2.get("selected_pages") or []):
                    if isinstance(p, (int, float)) and 1 <= int(p) <= self.doc.num_pages and int(p) not in sel:
                        sel.append(int(p))
        trace.selected_c3 = list(sel)
        trace.layers["b42_selected"] = list(sel)

        # Layer 4 (AnchorMap + G7 Printed-TOC): Single-shot global structural navigation
        direct_pages: list[int] = []
        range_ranked: list[int] = []
        if use_anchor_map and self.doc.num_pages > 8:
            toc_text = "\n\n".join(self.doc.page_text(p) for p in toc_pgs)[:4500] if toc_pgs else ""
            offset = self.index.calibrate_printed_offset(toc_text) if toc_text else 0
            trace.g7_offset = offset
            outline = self.index.build_structural_outline(max_anchors=95)
            nav, _ = self.llm.chat_json(
                "anchordoc_nav",
                "You are a precision document structure navigator. Return JSON only.",
                (
                    f"Query: {query}\n\n"
                    f"Printed TOC (printed page + {offset} approx physical PDF page):\n{toc_text or '(none)'}\n\n"
                    f"Zero-LLM Structural Anchor Map (physical pages p.N):\n{outline}\n\n"
                    "Return up to 5 specific physical PDF pages (direct_pages) and up to 2 short physical ranges "
                    "(ranges: [start, end], max 15 pages each) most likely to hold the primary evidence.\n"
                    'Return JSON: {"direct_pages": [int], "ranges": [[int, int]], "rationale": ""}'
                ),
                max_tokens=240,
            )
            direct_pages = [
                int(p) for p in (nav.get("direct_pages") or [])
                if isinstance(p, (int, float)) and 1 <= int(p) <= self.doc.num_pages and int(p) not in toc_pgs
            ]
            trace.anchor_direct = list(direct_pages)
            trace.layers["anchor_direct"] = list(direct_pages)

            range_pages: list[int] = []
            for rg in (nav.get("ranges") or [])[:2]:
                if isinstance(rg, list) and len(rg) == 2 and all(isinstance(x, (int, float)) for x in rg):
                    a = max(1, int(rg[0]))
                    b = min(self.doc.num_pages, max(a, int(rg[1])), a + 20)
                    for p in range(a, b + 1):
                        if p not in toc_pgs and p not in range_pages:
                            range_pages.append(p)
            if range_pages:
                range_ranked = self.index.rank_pages_subset(q_exp, range_pages, top_k=6)
                trace.layers["anchor_range"] = list(range_ranked)

        # Layer 5: Monotonic Joint Verification Rerank + RRF fusion (never lose strong hits)
        pool: list[int] = []
        for p in direct_pages[:5] + sel[:4] + range_ranked[:4] + cand_rrf[:4]:
            if p not in pool and p not in toc_pgs:
                pool.append(p)
            if len(pool) >= 10:
                break

        ver_ranked: list[int] = []
        if pool:
            ver, _ = self.llm.chat_json(
                "anchordoc_rerank",
                "You are a strict evidence ranker. Return JSON only.",
                (
                    f"Query: {query}\n\nCandidate Pages:\n{self._pack_pages(pool, 1700)}\n\n"
                    "Rank up to 5 pages containing the exact primary evidence, strongest FIRST.\n"
                    'Return JSON: {"ranked_pages": [int]}'
                ),
                max_tokens=160,
            )
            ver_ranked = [
                int(p) for p in (ver.get("ranked_pages") or [])
                if isinstance(p, (int, float)) and 1 <= int(p) <= self.doc.num_pages and int(p) not in toc_pgs
            ]
            trace.layers["verified_rerank"] = list(ver_ranked)

        fused_tail = rrf(ver_ranked, direct_pages, sel, range_ranked, cand_rrf, k=30)
        final_pages: list[int] = []
        for p in ver_ranked[:3] + fused_tail:
            if p not in final_pages and p not in toc_pgs:
                final_pages.append(p)
            if len(final_pages) >= top_k:
                break

        trace.pages = final_pages
        return trace


class SpanRetriever:
    """SpanCard + Two-Step Gap-Followup retriever for long prose/legal contracts."""

    def __init__(self, text: str, llm: LLMClient | None = None) -> None:
        self.index = SpanCardIndex(text)
        self.llm = llm

    @staticmethod
    def format_spans(spans: list[SpanCard]) -> str:
        return "\n\n".join(f"[{sp.tag} | {sp.section}]\n{sp.text}" for sp in spans)

    def retrieve(self, requirement: str, top_k: int = 7, enable_followup: bool = True) -> tuple[list[SpanCard], bool]:
        if self.llm is None:
            return self.index.retrieve(requirement, top_k=top_k, with_neighbors=True), False

        q_out, _ = self.llm.chat_json(
            "anchordoc_span_qexp",
            "You are a legal/compliance clause retrieval assistant. Return JSON only.",
            (
                f"Requirement: {requirement}\n"
                "Provide 6-10 exact keywords, synonyms, or exception cues to locate relevant clauses.\n"
                'Return JSON: {"keywords": ["..."]}'
            ),
            max_tokens=140,
        )
        kws = q_out.get("keywords") if isinstance(q_out.get("keywords"), list) else []
        spans = self.index.retrieve(f"{requirement} {' '.join(str(x) for x in kws)}", top_k=top_k, with_neighbors=True)
        if not enable_followup:
            return spans, False

        chk, _ = self.llm.chat_json(
            "anchordoc_span_gap",
            "You are a strict evidence completeness checker. Return JSON only.",
            (
                f"Requirement: {requirement}\n\nRetrieved Clauses:\n{self.format_spans(spans)}\n\n"
                "Do these clauses explicitly address the requirement (affirming OR contradicting it)? "
                "If NOT, provide 4-6 alternate phrases to search the rest of the document.\n"
                'Return JSON: {"direct_clause_present": true/false, "missing_phrases": ["..."]}'
            ),
            max_tokens=140,
        )
        if bool(chk.get("direct_clause_present")):
            return spans, False

        miss = chk.get("missing_phrases") if isinstance(chk.get("missing_phrases"), list) else []
        extra = self.index.retrieve(f"{' '.join(str(x) for x in miss)} {requirement}", top_k=6, with_neighbors=False)
        seen = {s.sid for s in spans}
        merged = list(spans)
        for sp in extra:
            if sp.sid not in seen:
                merged.append(sp)
                seen.add(sp.sid)
        merged.sort(key=lambda s: s.sid)
        return merged[:14], True
