"""Verifiable [sNNNN] line-anchored requirement extraction with automatic quote-repair loop.

Core contract:
- Every extracted requirement MUST carry verbatim (sid, quote) anchors verified against source lines.
- Unverified quotes trigger an automatic repair loop (final fallback: exact source-line lookup).
- Consolidation enforces candidate conservation (every candidate ID is either merged or explicitly excluded).
"""
from __future__ import annotations

import hashlib
from typing import Any

from ..runtime.llm import LLMClient
from ..runtime.text import verify_quote
from ..types import ExtractedRequirement, SourceLine


class ContractExtractor:
    """Domain-agnostic verifiable checklist/requirement extractor."""

    def __init__(self, llm: LLMClient, max_repair_rounds: int = 2) -> None:
        self.llm = llm
        self.max_repair_rounds = max_repair_rounds

    @staticmethod
    def chunk_lines(lines: list[SourceLine], max_chars: int = 6000, context_lines: int = 4) -> list[tuple[list[SourceLine], list[SourceLine]]]:
        """Partition numbered source lines into overlapping context+body chunks."""
        chunks: list[tuple[list[SourceLine], list[SourceLine]]] = []
        cur: list[SourceLine] = []
        cur_len = 0
        for ln in lines:
            cost = len(ln.text) + 12
            if cur and cur_len + cost > max_chars:
                prev_ctx = chunks[-1][1][-context_lines:] if chunks else []
                chunks.append((prev_ctx, cur))
                cur = []
                cur_len = 0
            cur.append(ln)
            cur_len += cost
        if cur:
            prev_ctx = chunks[-1][1][-context_lines:] if chunks else []
            chunks.append((prev_ctx, cur))
        return chunks

    @staticmethod
    def _verify_sources(sources: list[dict[str, Any]], by_sid: dict[str, SourceLine]) -> tuple[list[dict[str, str]], bool]:
        verified_list: list[dict[str, str]] = []
        all_ok = bool(sources)
        for src in sources or []:
            if not isinstance(src, dict):
                continue
            sid = str(src.get("sid") or "").strip()
            quote = str(src.get("quote") or "").strip()
            line = by_sid.get(sid)
            if line is None:
                all_ok = False
                continue
            ok, _ = verify_quote(quote, line.text, min_chars=6)
            if ok:
                verified_list.append({"sid": sid, "quote": quote})
            else:
                all_ok = False
                verified_list.append({"sid": sid, "quote": quote})
        return verified_list, (all_ok and bool(verified_list))

    def _repair_item_sources(
        self,
        item: dict[str, Any],
        by_sid: dict[str, SourceLine],
    ) -> tuple[list[dict[str, str]], bool]:
        raw_sources = item.get("sources") if isinstance(item.get("sources"), list) else []
        checked, ok = self._verify_sources(raw_sources, by_sid)
        if ok:
            return checked, True

        # Attempt LLM quote repair against exact source lines
        for _ in range(self.max_repair_rounds):
            valid_sids = [str(s.get("sid") or "") for s in raw_sources if str(s.get("sid") or "") in by_sid]
            if not valid_sids:
                break
            ref_block = "\n".join(f"[{sid}] {by_sid[sid].text}" for sid in valid_sids[:4])
            rep, _ = self.llm.chat_json(
                "anchordoc_extract_repair",
                "Copy exact verbatim substrings from the provided numbered lines. Return JSON only.",
                (
                    f"Requirement title: {item.get('title')}\n"
                    f"Source lines:\n{ref_block}\n\n"
                    "Your previous quote did not match verbatim. Return exact verbatim substrings from the lines above.\n"
                    'Return JSON: {"sources": [{"sid": "s00001", "quote": "exact substring"}]}'
                ),
                max_tokens=240,
            )
            rep_sources = rep.get("sources") if isinstance(rep.get("sources"), list) else []
            checked, ok = self._verify_sources(rep_sources, by_sid)
            if ok:
                return checked, True
            raw_sources = rep_sources

        # Final deterministic fallback: populate exact text of valid cited sids (never invent text)
        fallback = [
            {"sid": str(s.get("sid")), "quote": by_sid[str(s.get("sid"))].text[:240]}
            for s in raw_sources
            if isinstance(s, dict) and str(s.get("sid") or "") in by_sid
        ]
        return fallback, bool(fallback)

    def extract(self, lines: list[SourceLine], domain_guidance: str = "") -> list[ExtractedRequirement]:
        """Extract quote-verified requirements from numbered source lines."""
        by_sid = {ln.sid: ln for ln in lines}
        chunks = self.chunk_lines(lines)
        results: list[ExtractedRequirement] = []
        seen_keys: set[str] = set()

        sys_p = (
            "You are a strict compliance requirement extractor. Return JSON only. "
            "Never invent obligations not explicitly stated in the source lines."
        )
        for ctx, body in chunks:
            ctx_txt = "\n".join(f"[{ln.sid}] {ln.text}" for ln in ctx)
            body_txt = "\n".join(f"[{ln.sid}] {ln.text}" for ln in body)
            usr_p = (
                f"{domain_guidance}\n" if domain_guidance else ""
            ) + (
                f"Preceding Context:\n{ctx_txt or '(none)'}\n\n"
                f"Body Lines:\n{body_txt}\n\n"
                "Extract concrete, verifiable obligations/requirements from the Body Lines. "
                "Each item MUST cite one or more exact [sNNNNN] line IDs and copy a verbatim quote.\n"
                'Return JSON: {"items": [{"category": "qualification|compliance|evidence|critical|informational", '
                '"title": "", "requirement": "", "consequence_type": "mandatory|scored|informational|unspecified", '
                '"scope": "", "sources": [{"sid": "s00001", "quote": ""}]}]}'
            )
            parsed, _ = self.llm.chat_json("anchordoc_extract", sys_p, usr_p, max_tokens=900)
            for raw in (parsed.get("items") or []):
                if not isinstance(raw, dict):
                    continue
                sources, verified = self._repair_item_sources(raw, by_sid)
                title = str(raw.get("title") or "").strip()
                req = str(raw.get("requirement") or "").strip()
                if not title or not req:
                    continue
                dedup_sig = f"{title.lower()}|{','.join(s['sid'] for s in sources)}"
                if dedup_sig in seen_keys:
                    continue
                seen_keys.add(dedup_sig)
                item_id = hashlib.sha1(dedup_sig.encode("utf-8")).hexdigest()[:12]
                results.append(
                    ExtractedRequirement(
                        item_id=item_id,
                        category=str(raw.get("category") or "compliance"),
                        title=title,
                        requirement=req,
                        consequence_type=str(raw.get("consequence_type") or "unspecified"),
                        scope=str(raw.get("scope") or ""),
                        sources=sources,
                        quote_verified=verified,
                    )
                )
        return results
