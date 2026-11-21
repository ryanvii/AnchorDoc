"""Structure-aware SpanCard index for long prose/contract documents."""
from __future__ import annotations

import re

from ..runtime.text import BM25Okapi, rrf, tokenize
from ..types import SpanCard

_SEC_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*\.?\s+[A-Z][^\n]{2,70}|[A-Z][A-Z\s,&/-]{3,60}|ARTICLE\s+[IVX\d]+|SECTION\s+\d+|第[一二三四五六七八九十百\d]+[章节条])"
)


class SpanCardIndex:
    """Numbered [sNNNN] clause cards with inherited section headings + Dual BM25-RRF."""

    def __init__(self, text: str) -> None:
        self.full_text = text or ""
        self.spans: list[SpanCard] = self._build_spans(self.full_text)
        self.bm_raw = BM25Okapi([tokenize(s.text) for s in self.spans])
        self.bm_card = BM25Okapi([tokenize(s.card_text()) for s in self.spans])

    @staticmethod
    def _build_spans(text: str) -> list[SpanCard]:
        raw_lines = [ln.strip() for ln in re.split(r"\n+", text) if ln.strip()]
        spans: list[SpanCard] = []
        cur_sec = "Preamble"
        buf = ""
        for ln in raw_lines:
            if len(ln) <= 85 and _SEC_RE.match(ln) and not ln.endswith("."):
                if buf:
                    spans.append(SpanCard(len(spans) + 1, cur_sec, buf))
                    buf = ""
                cur_sec = ln
                continue
            if len(ln) < 120 and buf and len(buf) < 420:
                buf = f"{buf} {ln}"
            else:
                if buf:
                    spans.append(SpanCard(len(spans) + 1, cur_sec, buf))
                buf = ln
        if buf:
            spans.append(SpanCard(len(spans) + 1, cur_sec, buf))
        if not spans:
            spans.append(SpanCard(1, "Document", text))
        return spans

    def retrieve(
        self,
        query: str,
        top_k: int = 7,
        with_neighbors: bool = True,
        neighbor_char_budget: int = 2400,
    ) -> list[SpanCard]:
        """Dual BM25-RRF clause retrieval.

        v0.6 fix: ALL ``top_k`` fused hits are always kept first; neighbours of the top-3 hits are
        only added afterwards within ``neighbor_char_budget`` (v0.5 filled the slots with neighbours
        first and silently dropped fused ranks 4..k).
        """
        qt = tokenize(query)
        s_raw = self.bm_raw.get_scores(qt)
        s_card = self.bm_card.get_scores(qt)
        r_raw = sorted(range(len(self.spans)), key=lambda i: -s_raw[i])[:20]
        r_card = sorted(range(len(self.spans)), key=lambda i: -s_card[i])[:20]
        fused = rrf(r_raw, r_card, k=60)[:top_k]
        if with_neighbors:
            selected = list(fused)
            budget = neighbor_char_budget
            for idx in fused[:3]:
                for nb in (idx - 1, idx + 1):
                    if 0 <= nb < len(self.spans) and nb not in selected:
                        cost = len(self.spans[nb].text)
                        if cost <= budget:
                            selected.append(nb)
                            budget -= cost
            fused = selected
        return [self.spans[i] for i in sorted(fused)]
