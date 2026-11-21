"""Unified Zero-Index-Cost Retrieval Pipelines with Layer-by-Layer Audit Trace.

Combines:
- Zero-LLM Structural AnchorMap
- Printed-TOC (G7) navigation with printed-to-physical offset calibration
- Query expansion (C3) + Dual BM25-RRF
- v0.6 Query-aware Locator: ONE LLM call over query-relevant page snippets (replaces the
  former triage + rerank pair that only saw each page's first 1600 characters)
- Deterministic page-reference resolver ("page 5", "slide 3", "last page") and table
  continuation expansion (p-1 / p+1) feeding the locator pool
- Conditional Two-Step Gap-Followup (B42) through a decorrelated channel
- Monotonic RRF fusion (never drops verified strong evidence pages); TOC pages are softly
  demoted instead of hard-excluded
"""
from __future__ import annotations

import math
import re
from collections import Counter

from typing import Literal

from ..docbase.pdf_loader import PdfDocument
from ..index.anchor_map import DocumentIndex
from ..index.spancard import SpanCardIndex
from ..runtime.llm import LLMClient
from ..runtime.text import rrf, tokenize
from ..types import RetrievalTrace, SpanCard
from .router import Route, RoutePlan, Task, plan_prose_route, plan_route, profile_document

_NUM_TOKEN_RE = re.compile(r"[$¥€(]?-?\d[\d,.]*%?\)?")
_YEARISH_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_UNIT_CUE_RE = re.compile(
    r"\b(?:in\s+(?:millions|thousands|billions)|except\s+(?:per[\s-]share|share|where\s+noted)|"
    r"\(unaudited\)|(?:fiscal\s+)?years?\s+ended|as\s+of|three\s+months\s+ended|six\s+months\s+ended|"
    r"nine\s+months\s+ended|amounts?\s+in)\b|单位[:：]|人民币",
    re.I,
)

LOCATE_POOL_MAX = 15


def _num_count(line: str) -> int:
    return len(_NUM_TOKEN_RE.findall(line))


def _split_long_lines(text: str, max_len: int = 220) -> list[str]:
    """Split page text into lines, breaking very long extracted lines at whitespace."""
    out: list[str] = []
    for raw in (text or "").splitlines():
        ln = raw.strip()
        while len(ln) > max_len:
            cut = ln.rfind(" ", 0, max_len)
            cut = cut if cut > max_len // 2 else max_len
            out.append(ln[:cut].strip())
            ln = ln[cut:].strip()
        if ln:
            out.append(ln)
    return out


def _as_bool(v: object) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"true", "yes", "1"}


def _dedup(seq, limit: int | None = None) -> list[int]:
    out: list[int] = []
    for p in seq:
        if p not in out:
            out.append(p)
            if limit is not None and len(out) >= limit:
                break
    return out


class AnchorRetriever:
    """Page-level retriever for 10-600 page PDFs with full RetrievalTrace."""

    def __init__(
        self,
        doc: PdfDocument,
        llm: LLMClient | None = None,
        vision_summaries: dict[int, str] | None = None,
    ) -> None:
        self.doc = doc
        self.index = DocumentIndex(doc, vision_summaries=vision_summaries)
        self.llm = llm

    # ------------------------------------------------------------------------------------
    # Query-aware snippet packing
    # ------------------------------------------------------------------------------------
    def _term_weights(self, query: str, extra: str = "", extra_weight: float = 0.6) -> dict[str, float]:
        """IDF-weighted query terms (document-level IDF: rare terms dominate window choice)."""
        idf = self.index.bm_raw.idf
        w: dict[str, float] = {}
        for t in tokenize(extra):
            w[t] = max(w.get(t, 0.0), extra_weight * idf.get(t, 0.0))
        for t in tokenize(query):
            w[t] = max(w.get(t, 0.0), idf.get(t, 0.0))
        return {t: v for t, v in w.items() if v > 0}

    def _page_snippet(self, page: int, weights: dict[str, float], budget_chars: int = 900, win_chars: int = 400) -> str:
        """Header + table header lines + top-2 query-relevant ~400-char windows (+ VisionCard)."""
        c = self.index.cards[page - 1]
        printed = self.index.printed_page_map().get(page, "")
        tags = []
        if printed and printed != str(page):
            tags.append(f"printed p.{printed}")
        if page in self.index.toc_pages:
            tags.append("TOC")
        if c.has_table:
            tags.append("TABLE")
        tag_s = f" ({', '.join(tags)})" if tags else ""
        header = f"[Page {page}]{tag_s} Section: {c.section_path or 'N/A'} | Heading: {c.heading}"

        parts: list[str] = []
        vsum = (self.index.vision_summaries.get(page, "") or "").strip()
        if vsum:
            parts.append(f"Visual: {vsum[:240]}")

        lines = _split_long_lines(self.doc.page_text(page))
        if not lines:
            body = "\n".join(parts) if parts else "(no extractable text)"
            return f"{header}\n{body}"[: budget_chars + len(header)]

        # Table header / unit lines (column years, "in millions", "years ended ...").
        hdr_idx: list[int] = []
        for i, ln in enumerate(lines):
            if len(ln) > 100:  # column headers / unit lines are short; skip prose sentences
                continue
            years = len(_YEARISH_RE.findall(ln))
            if _UNIT_CUE_RE.search(ln) or (years >= 2 and _num_count(ln) <= years + 1):
                hdr_idx.append(i)
            if len(hdr_idx) >= 2:
                break

        # Score sliding line-windows by IDF-weighted term overlap, boosted by numeric density.
        windows: list[tuple[float, int, int]] = []
        low_lines = [ln.lower() for ln in lines]
        tok_lines = [Counter(tokenize(ln)) for ln in low_lines]
        for i in range(len(lines)):
            j, size = i, 0
            while j < len(lines) and size < win_chars:
                size += len(lines[j]) + 1
                j += 1
            tf: Counter[str] = Counter()
            for k in range(i, j):
                tf.update(tok_lines[k])
            score = sum(wt * (1.0 + math.log(tf[t])) for t, wt in weights.items() if tf.get(t))
            if score > 0:
                dense = sum(1 for k in range(i, j) if _num_count(lines[k]) >= 2)
                score *= 1.0 + 0.12 * min(dense, 5)
            windows.append((score, i, j))
            if j >= len(lines):
                break

        chosen: list[tuple[int, int]] = []
        for score, i, j in sorted(windows, key=lambda x: (-x[0], x[1])):
            if score <= 0 or len(chosen) >= 2:
                break
            if all(j <= a or i >= b for a, b in chosen):
                chosen.append((i, j))
        if not chosen:  # no lexical hit: show the page head so the locator can still judge it
            chosen = [(windows[0][1], windows[0][2])]
        chosen.sort()

        covered = {k for a, b in chosen for k in range(a, b)}
        hdr_lines = [lines[i] for i in hdr_idx if i not in covered]
        if hdr_lines:
            parts.append("Header: " + " | ".join(hdr_lines))
        remaining = max(200, budget_chars - sum(len(x) + 1 for x in parts))
        per = max(150, remaining // len(chosen))

        def line_score(k: int) -> float:
            return sum(wt for t, wt in weights.items() if tok_lines[k].get(t))

        for a, b in chosen:
            # Trim an over-long window from whichever end carries less query signal, so the
            # matching lines survive (never blindly cut the tail).
            while b - a > 1 and sum(len(lines[k]) + 1 for k in range(a, b)) > per:
                if line_score(a) < line_score(b - 1) or (line_score(a) == line_score(b - 1) and a < b - 1 and
                                                         line_score(a) == 0):
                    a += 1
                else:
                    b -= 1
            pre = "… " if a > 0 else ""
            post = " …" if b < len(lines) else ""
            parts.append((pre + "\n".join(lines[a:b]) + post)[: per + 4])
        return f"{header}\n" + "\n".join(parts)

    def _pack_snippets(
        self, pages: list[int], query_terms: dict[str, float] | str, budget_chars: int = 900
    ) -> str:
        """Pack query-relevant page snippets (~``budget_chars`` per page) for a locator prompt.

        ``query_terms`` may be a raw query string or a pre-computed {term: weight} map.
        """
        weights = self._term_weights(query_terms) if isinstance(query_terms, str) else dict(query_terms)
        blocks = [self._page_snippet(p, weights, budget_chars) for p in pages if 1 <= p <= self.doc.num_pages]
        return "\n\n---\n\n".join(blocks)

    def _pack_pages(self, pages: list[int], char_limit: int = 1600) -> str:
        """Legacy head-truncation packer (kept for backward compatibility; unused by retrieve)."""
        blocks = []
        for p in pages:
            c = self.index.cards[p - 1]
            raw = (self.doc.page_text(p) or "").strip()
            vsum = self.index.vision_summaries.get(p, "")
            body = raw[:char_limit]
            if vsum:
                body = f"{vsum}\n{body}".strip()[:char_limit]
            blocks.append(
                f"[Page {p}] Section: {c.section_path or 'N/A'} | Heading: {c.heading}\n{body}"
            )
        return "\n\n---\n\n".join(blocks)

    # ------------------------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------------------------
    def _valid_pages(self, raw: object, allowed: set[int] | None = None) -> list[int]:
        out: list[int] = []
        for p in raw if isinstance(raw, list) else []:
            if isinstance(p, bool) or not isinstance(p, (int, float, str)):
                continue
            try:
                v = int(float(p))
            except (TypeError, ValueError):
                continue
            if 1 <= v <= self.doc.num_pages and (allowed is None or v in allowed) and v not in out:
                out.append(v)
        return out

    def _lexical(self, query: str, k: int) -> list[int]:
        """Dual BM25-RRF with soft TOC demotion (TOC pages kept, ranked after non-TOC pages)."""
        return self.index.rank_pagecard_rrf(
            query, top_k=k + len(self.index.toc_pages), exclude_toc=False, demote_toc=True
        )[:k]

    def _finalize(self, order: list[int], top_k: int, protected: set[int], llm_chosen: set[int]) -> list[int]:
        """Soft TOC demotion on the final list.

        Non-TOC pages (and protected pages: nav direct / explicit page refs) keep their order.
        A TOC page explicitly chosen by the locator may take the LAST slot; other TOC pages only
        fill remaining room. Nothing is ever hard-excluded.
        """
        toc = self.index.toc_pages
        normal = [p for p in order if p not in toc or p in protected]
        toc_chosen = [p for p in order if p in toc and p not in protected and p in llm_chosen]
        toc_other = [p for p in order if p in toc and p not in protected and p not in llm_chosen]
        if toc_chosen and len(normal) >= top_k:
            return normal[: top_k - 1] + toc_chosen[:1]
        return _dedup(normal + toc_chosen + toc_other, top_k)

    # ------------------------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_anchor_map: bool = True,
        use_gap_followup: bool = True,
        *,
        route: Route | Literal["auto"] = "auto",
        task: Task = "qa_text",
    ) -> RetrievalTrace:
        """Adaptive 3-tier retrieval: read_all (0 calls) | full_pool (skip nav) | navigate (v0.6)."""
        trace = RetrievalTrace(query=query)
        toc_pgs = self.index.find_printed_toc_pages()
        trace.g7_toc_pages = toc_pgs

        page_refs = self.index.resolve_page_refs(query)
        if page_refs:
            trace.layers["page_refs"] = list(page_refs)

        profile = profile_document(self.doc, vision_summaries=self.index.vision_summaries)
        plan = plan_route(profile, task=task, force=route)
        trace.route = plan.route
        trace.route_reason = plan.reason
        trace.est_tokens = plan.est_tokens

        # Tier 1 (read_all): 0 retrieval LLM calls — pass every physical page to the reader.
        # For multimodal read_all, order by zero-LLM lexical RRF so the top-ranked page (not the
        # cover page) gets the 2x2 zoom tiles; for text read_all, preserve physical page order so
        # multi-page tables remain contiguous.
        if plan.route == "read_all":
            all_pages = list(range(1, self.doc.num_pages + 1))
            if task == "qa_multimodal":
                rrf_pages = self._lexical(query, len(all_pages))
                ordered = _dedup(page_refs + rrf_pages + all_pages)
                trace.layers["read_all_rrf"] = list(ordered)
                trace.pages = ordered
            else:
                trace.layers["read_all_pages"] = list(all_pages)
                trace.pages = all_pages
            return trace

        # Zero-LLM fallback mode when no LLM is supplied
        if self.llm is None:
            rrf_pages = self._lexical(query, max(top_k, 10))
            trace.layers["pagecard_rrf"] = list(rrf_pages[:top_k])
            order = _dedup(page_refs[:2] + rrf_pages)
            trace.pages = self._finalize(order, top_k, protected=set(page_refs[:2]), llm_chosen=set())
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
        exp = exp if isinstance(exp, dict) else {}
        kws = exp.get("keywords") if isinstance(exp.get("keywords"), list) else []
        hint = str(exp.get("section_hint") or "")
        trace.expanded_keywords = [str(x) for x in kws]
        q_exp = f"{query} {' '.join(trace.expanded_keywords)} {hint}"

        cand_rrf = self._lexical(q_exp, 10)
        trace.layers["c3_rrf"] = list(cand_rrf)

        # Tier 2 (full_pool): skip global nav when doc <= FULLPOOL_MAX_PAGES; feed ALL pages into
        # one query-aware locate call under LOCATE_CHAR_BUDGET. B42 naturally no-ops (all pages in pool).
        direct_pages: list[int] = []
        range_ranked: list[int] = []
        range_pages: list[int] = []
        snippet_chars = 900
        if plan.route == "full_pool":
            pool = list(range(1, self.doc.num_pages + 1))
            snippet_chars = plan.snippet_chars
            trace.layers["locate_pool"] = list(pool)
        else:
            # Tier 3 (navigate): byte-for-byte identical to v0.6.0
            # Layer 2 (AnchorMap + G7 Printed-TOC): single-shot global structural navigation
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
                nav = nav if isinstance(nav, dict) else {}
                # nav direct pages are never excluded (not even TOC pages)
                direct_pages = self._valid_pages(nav.get("direct_pages"))[:5]
                trace.anchor_direct = list(direct_pages)
                trace.layers["anchor_direct"] = list(direct_pages)

                for rg in (nav.get("ranges") or [])[:2] if isinstance(nav.get("ranges"), list) else []:
                    if isinstance(rg, list) and len(rg) == 2 and all(
                        isinstance(x, (int, float)) and not isinstance(x, bool) for x in rg
                    ):
                        a = max(1, int(rg[0]))
                        b = min(self.doc.num_pages, max(a, int(rg[1])), a + 20)
                        for p in range(a, b + 1):
                            if p not in range_pages:
                                range_pages.append(p)
                if range_pages:
                    range_ranked = self.index.demote_toc(self.index.rank_pages_subset(q_exp, range_pages, top_k=6))
                    trace.layers["anchor_range"] = list(range_ranked)

            # Layer 3: Query-aware Locator pool (lexical ∪ nav ∪ page refs ∪ table continuations)
            refs_in_pool = page_refs[:4]
            seed = rrf(refs_in_pool, direct_pages, cand_rrf[:8], range_ranked, k=30)
            cont: list[int] = []
            for p in seed[:3]:
                for q in self.index.continuation_neighbors(p):
                    if q not in seed[:3] and q not in cont:
                        cont.append(q)
            if cont:
                trace.layers["continuation"] = list(cont)
            # Explicit page refs and nav direct pages are guaranteed a slot; the lexical tail is
            # what gets cut when the pool exceeds LOCATE_POOL_MAX.
            must = _dedup(refs_in_pool + direct_pages)
            pool = _dedup(seed[:3] + cont + seed[3:], LOCATE_POOL_MAX)
            for p in must:
                if p not in pool:
                    pool = pool[: LOCATE_POOL_MAX - 1] + [p] if len(pool) >= LOCATE_POOL_MAX else pool + [p]
            trace.layers["locate_pool"] = list(pool)

        weights = self._term_weights(query, f"{' '.join(trace.expanded_keywords)} {hint}")
        ranked: list[int] = []
        sufficient = False
        missing: list[str] = []
        if pool:
            loc, _ = self.llm.chat_json(
                "anchordoc_locate",
                "You are a strict evidence locator. Return JSON only.",
                (
                    f"Query: {query}\n\n"
                    "Candidate pages (each block shows only the most query-relevant excerpts of that page; "
                    "'printed p.N' is the page number printed on the page, 'TOC' marks a table-of-contents page):\n"
                    f"{self._pack_snippets(pool, weights, snippet_chars)}\n\n"
                    f"Rank up to {max(top_k, 5)} pages that contain the primary evidence needed to answer, "
                    "strongest FIRST. Include every page needed when evidence spans pages (e.g. a table and its "
                    "continuation). Set sufficient=true only if the ranked pages together contain ALL evidence "
                    "needed; otherwise list the specific missing_terms (entities, line items, periods) still to find.\n"
                    'Return JSON: {"ranked_pages": [int], "sufficient": true/false, "missing_terms": ["..."]}'
                ),
                max_tokens=220,
            )
            loc = loc if isinstance(loc, dict) else {}
            ranked = self._valid_pages(loc.get("ranked_pages"), allowed=set(pool))
            sufficient = _as_bool(loc.get("sufficient")) and bool(ranked)
            missing = [str(x) for x in loc.get("missing_terms") or [] if str(x).strip()] if isinstance(
                loc.get("missing_terms"), list
            ) else []
        trace.layers["locate_ranked"] = list(ranked)
        trace.layers["locate_sufficient"] = [int(sufficient)]

        # Layer 4 (B42): conditional gap-followup through a decorrelated channel
        followup_sel: list[int] = []
        nav_lex_disjoint = not (set(direct_pages) & set(cand_rrf[:10]))
        if use_gap_followup and pool and not sufficient and (nav_lex_disjoint or missing):
            q2 = f"{' '.join(missing)} {hint} {query}"
            c2 = self._lexical(q2, 12)
            chunk_hits = self.index.rank_chunk_bm25_pages(q2, top_k=8)  # finer-grained channel
            adj = [p + dp for p in ranked[:2] for dp in (-1, 1) if 1 <= p + dp <= self.doc.num_pages]
            interleaved = [x for pair in zip(chunk_hits, c2) for x in pair] + chunk_hits + c2
            extra_pool = [
                p for p in _dedup(adj + interleaved + range_pages + cand_rrf[8:]) if p not in pool
            ][:6]
            if extra_pool:
                trace.followup_triggered = True
                trace.layers["b42_pool"] = list(extra_pool)
                w2 = self._term_weights(query, " ".join(missing), extra_weight=1.2)
                tri2, _ = self.llm.chat_json(
                    "anchordoc_followup",
                    "You are a strict evidence auditor. Return JSON only.",
                    (
                        f"Query: {query}\nMissing evidence from round 1: {missing or '(unspecified)'}\n\n"
                        f"Additional Candidate Pages (query-relevant excerpts):\n{self._pack_snippets(extra_pool, w2, 900)}\n\n"
                        "Which of these pages contain the missing evidence? Strongest first.\n"
                        'Return JSON: {"selected_pages": [int]}'
                    ),
                    max_tokens=160,
                )
                tri2 = tri2 if isinstance(tri2, dict) else {}
                followup_sel = [p for p in self._valid_pages(tri2.get("selected_pages"), set(extra_pool)) if p not in ranked]
        trace.layers["b42_selected"] = list(followup_sel)

        # Follow-up hits fill the gap the locator reported, so they must reach the top-k:
        # interleave them right after the locator's strongest pages instead of appending at the tail.
        chosen = _dedup(ranked[:3] + followup_sel[:2] + ranked[3:] + followup_sel[2:])
        trace.selected_c3 = list(chosen)

        # Layer 5: locate-ranked first, then monotonic RRF tail (never lose strong hits)
        fused_tail = rrf(chosen, direct_pages, range_ranked, cand_rrf, page_refs, k=30)
        order = _dedup(chosen + fused_tail)
        trace.pages = self._finalize(
            order, top_k, protected=set(direct_pages) | set(page_refs), llm_chosen=set(chosen)
        )
        return trace


class SpanRetriever:
    """SpanCard + Two-Step Gap-Followup retriever for long prose/legal contracts."""

    def __init__(self, text: str, llm: LLMClient | None = None) -> None:
        self.index = SpanCardIndex(text)
        self.llm = llm
        self.last_route: RoutePlan = plan_prose_route(self.index.full_text)

    @staticmethod
    def format_spans(spans: list[SpanCard]) -> str:
        return "\n\n".join(f"[{sp.tag} | {sp.section}]\n{sp.text}" for sp in spans)

    def retrieve(
        self,
        requirement: str,
        top_k: int = 7,
        enable_followup: bool = True,
        *,
        route: Route | Literal["auto"] = "auto",
    ) -> tuple[list[SpanCard], bool]:
        plan = plan_prose_route(self.index.full_text, force=route)
        self.last_route = plan
        if plan.route == "read_all":
            return list(self.index.spans), False

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
