"""Zero-LLM Structural AnchorMap, Rule PageCards, and Printed-TOC (G7) calibration."""
from __future__ import annotations

import re
from typing import Iterable

from ..docbase.pdf_loader import PdfDocument
from ..runtime.text import BM25Okapi, rrf, tokenize
from ..types import PageCard

_TOC_DOTS = re.compile(r"\.{4,}\s*\d+")
_HEADING_RE = re.compile(
    r"^(?:ITEM\s+\d+[A-Z]?\.?|PART\s+[IVX]+|NOTE\s+\d+|SECTION\s+\d+|CHAPTER\s+\d+|"
    r"第[一二三四五六七八九十百\d]+[章节条款]|\d+(?:\.\d+){0,3}\.?|[A-Z][A-Z\s,&/-]{5,60})$"
)
_TABLE_CUE_RE = re.compile(
    r"(?:consolidated\s+(?:statements?|balance\s+sheets?)|cash\s+flows?|operations|income|"
    r"shareholders'?\s+equity|segment|revenue|operating\s+income|capital\s+expenditures?|"
    r"long-term\s+debt|summary\s+of\s+financial\s+performance|schedule\s+\w+|table\s+\d+|"
    r"偏离表|报价表|一览表|明细表|索引表|财务报表|审计报告)",
    re.I,
)
_STMT_RE = re.compile(
    r"(?:consolidated\s+(?:statements?\s+of\s+(?:income|operations|earnings|comprehensive|cash\s+flows?|stockholders|shareholders|equity)|balance\s+sheets?)"
    r"|summary\s+of\s+financial\s+performance|selected\s+financial\s+data|table\s+\d+\s*:|"
    r"资格审查|符合性审查|开标一览表|偏离表)",
    re.I,
)
_NOTE_RE = re.compile(
    r"^(?:NOTE\s+\d+[\s.:\-—]+.{3,75}|ITEM\s+\d+[A-Z]?[\s.:\-—]+.{2,75}|PART\s+[IVX]+|"
    r"CHAPTER\s+\d+|SECTION\s+\d+|第[一二三四五六七八九十百\d]+[章节])",
    re.I,
)
_YEAR_RE = re.compile(r"\b20(?:1[5-9]|2[0-9])\b")


def is_toc_like(text: str, page_num: int) -> bool:
    """Heuristic TOC-page detector to prevent index pages from crowding evidence retrieval."""
    if page_num > 12:
        return False
    low = (text or "").lower()
    if ("table of contents" in low or "目 录" in text or "目录" in text[:120]) and len(_TOC_DOTS.findall(text)) >= 3:
        return True
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return False
    page_end = sum(1 for ln in lines if re.search(r"\s+\d{1,3}$", ln) and len(ln) < 110)
    if len(lines) >= 10 and page_end / len(lines) >= 0.45:
        return True
    return len(_TOC_DOTS.findall(text)) >= 6


class DocumentIndex:
    """Zero-LLM structural index over a 1-based PdfDocument."""

    def __init__(self, doc: PdfDocument, vision_summaries: dict[int, str] | None = None) -> None:
        self.doc = doc
        self.n = doc.num_pages
        self.vision_summaries = vision_summaries or {}
        self.cards: list[PageCard] = self._build_cards()
        self.toc_pages: set[int] = set(self.find_printed_toc_pages()) | {c.page for c in self.cards if c.is_toc}

        # Dual BM25 indices (raw physical page + structured PageCard incl. Selective VisionCard)
        self.bm_raw = BM25Okapi(
            [tokenize((p or "") + " " + self.vision_summaries.get(i + 1, "")) for i, p in enumerate(doc.pages)]
        )
        self.bm_card = BM25Okapi([tokenize(c.card_text()) for c in self.cards])

        # Flat chunk index (for baseline comparison and fine-grained excerpting)
        self.chunks: list[tuple[int, str]] = []
        for pno, text in enumerate(doc.pages, start=1):
            t = (text or "").strip()
            if not t:
                continue
            for start in range(0, len(t), 850):
                ch = t[start : start + 1000].strip()
                if len(ch) >= 40:
                    self.chunks.append((pno, ch))
        self.bm_chunk = BM25Okapi([tokenize(c[1]) for c in self.chunks])

    def _build_cards(self) -> list[PageCard]:
        sec_by_page: dict[int, str] = {}
        if self.doc.toc_bookmarks:
            stack: list[tuple[int, str]] = []
            for lvl, title, pg in self.doc.toc_bookmarks:
                if pg < 1 or pg > self.n:
                    continue
                while stack and stack[-1][0] >= lvl:
                    stack.pop()
                stack.append((lvl, title))
                sec_by_page[pg] = " > ".join(t for _, t in stack[-3:])

        cards: list[PageCard] = []
        cur_sec = ""
        for pno, text in enumerate(self.doc.pages, start=1):
            if pno in sec_by_page:
                cur_sec = sec_by_page[pno]
            lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
            heading = ""
            for ln in lines[:8]:
                if 3 <= len(ln) <= 95 and (_HEADING_RE.match(ln) or _TABLE_CUE_RE.search(ln)):
                    heading = ln
                    if not sec_by_page and _NOTE_RE.match(ln):
                        cur_sec = ln
                    break
            if not heading and lines:
                heading = lines[0][:90]
            num_lines = sum(1 for ln in lines if len(re.findall(r"[$¥€]?\d[\d,.]*", ln)) >= 3)
            cues = tuple(sorted({m.group(0).lower() for m in _TABLE_CUE_RE.finditer((text or "")[:2500])})[:8])
            years = tuple(sorted(set(_YEAR_RE.findall((text or "")[:2000])))[:6])
            cards.append(
                PageCard(
                    page=pno,
                    section_path=cur_sec,
                    heading=heading,
                    is_toc=is_toc_like(text, pno),
                    has_table=num_lines >= 4,
                    table_cues=cues,
                    years=years,
                    text_head=" ".join(lines[:10])[:450],
                    vision_summary=self.vision_summaries.get(pno, ""),
                )
            )
        return cards

    def find_printed_toc_pages(self) -> list[int]:
        """Detect printed Table-of-Contents / Index pages in the opening pages (G7 gate)."""
        hits: list[int] = []
        for i in range(min(8, self.n)):
            txt = self.doc.pages[i] or ""
            low = txt.lower()
            item_cnt = len(
                re.findall(r"(?:\b(?:item\s+\d+[a-z]?|part\s+[ivx]+|note\s+\d+|section\s+\d+)\b|第[一二三四五六七八九十\d]+[章节])", low)
            )
            num_lines = len(re.findall(r"(?m)^\s*\d{1,3}\s*$", txt)) + len(re.findall(r"\.{3,}\s*\d{1,3}", txt))
            if (item_cnt >= 4 and num_lines >= 4) or (("table of contents" in low or "目录" in txt[:120]) and item_cnt >= 3):
                hits.append(i + 1)
        return hits[:2]

    def calibrate_printed_offset(self, toc_text: str) -> int:
        """Calibrate printed_page -> physical_pdf_page offset (distilled from G7)."""
        anchors: list[tuple[str, int]] = []
        lines = [ln.strip() for ln in (toc_text or "").splitlines() if ln.strip()]
        for idx, ln in enumerate(lines):
            m = re.match(r"^(ITEM\s+(?:1A?|7|8)\b[^\n]{0,45}|CHAPTER\s+\d+[^\n]{0,45}|第[一二三四五六七八九十\d]+章[^\n]{0,30})$", ln, re.I)
            if not m:
                continue
            pnum = None
            m_same = re.search(r"\b(\d{1,3})\s*$", ln)
            if m_same:
                pnum = int(m_same.group(1))
            else:
                for off in (1, 2):
                    if idx + off < len(lines) and re.fullmatch(r"\d{1,3}", lines[idx + off]):
                        pnum = int(lines[idx + off])
                        break
            if pnum and 1 <= pnum < self.n:
                key_m = re.match(r"(?:ITEM\s+(?:1A?|7|8)|CHAPTER\s+\d+|第[一二三四五六七八九十\d]+章)", ln, re.I)
                if key_m:
                    anchors.append((key_m.group(0).lower(), pnum))

        offsets: list[int] = []
        for key, pnum in anchors[:5]:
            for delta in range(-2, 10):
                phys = pnum + delta
                if 1 <= phys <= self.n and phys > 3:
                    head = (self.doc.pages[phys - 1] or "")[:600].lower()
                    if key in head and "table of contents" not in head[:180]:
                        offsets.append(delta)
                        break
        if not offsets:
            return 0
        offsets.sort()
        return offsets[len(offsets) // 2]

    def build_structural_outline(self, max_anchors: int = 95) -> str:
        """Build Zero-LLM Structural Anchor Map (~50ms, 0 LLM calls) for global navigation."""
        anchors: list[tuple[int, int, str]] = []
        toc_set = set(self.find_printed_toc_pages())
        subhead_re = re.compile(r"^[A-Z][A-Za-z0-9\s,&/'-]{4,68}$")

        for p in range(1, self.n + 1):
            if p in toc_set:
                continue
            txt = self.doc.pages[p - 1] or ""
            vsum = self.vision_summaries.get(p, "")
            lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
            clean = [ln for ln in lines[:12] if ln.lower() != "table of contents" and not re.fullmatch(r"\d{1,3}", ln)]
            if not clean and vsum:
                anchors.append((1, p, f"p.{p}: {vsum[:150]}"))
                continue
            if not clean:
                continue
            head_blob = " | ".join(clean[:4])[:140]
            if vsum:
                head_blob = f"{head_blob} | {vsum[:90]}"[:155]
            c = self.cards[p - 1]
            tag = " [TABLE]" if c.has_table else ""

            if _STMT_RE.search(" ".join(clean[:6])):
                anchors.append((0, p, f"p.{p}: {head_blob}{tag}"))
            elif any(_NOTE_RE.match(ln) for ln in clean[:5]):
                anchors.append((1, p, f"p.{p}: {head_blob}{tag}"))
            elif c.has_table and len(c.table_cues) >= 2:
                anchors.append((2, p, f"p.{p}: {head_blob}{tag}"))
            elif p <= 18 or (len(clean[0]) <= 68 and subhead_re.match(clean[0]) and len(clean) >= 4):
                anchors.append((3, p, f"p.{p}: {head_blob}{tag}"))

        kept = [a for a in anchors if a[0] <= 2]
        rem = [a for a in anchors if a[0] > 2]
        budget = max(0, max_anchors - len(kept))
        if budget > 0 and rem:
            if len(rem) <= budget:
                kept.extend(rem)
            else:
                step = len(rem) / budget
                kept.extend(rem[int(i * step)] for i in range(budget))
        kept.sort(key=lambda x: x[1])
        return "\n".join(x[2] for x in kept[:max_anchors])

    def rank_pagecard_rrf(self, query: str, top_k: int = 10, exclude_toc: bool = True) -> list[int]:
        """Dual BM25 (raw page + PageCard) fused via RRF(k=60)."""
        qt = tokenize(query)
        s_raw = self.bm_raw.get_scores(qt)
        s_card = self.bm_card.get_scores(qt)
        r_raw = [i + 1 for i in sorted(range(self.n), key=lambda i: -s_raw[i]) if not exclude_toc or (i + 1) not in self.toc_pages][:60]
        r_card = [i + 1 for i in sorted(range(self.n), key=lambda i: -s_card[i]) if not exclude_toc or (i + 1) not in self.toc_pages][:60]
        return rrf(r_raw, r_card, k=60)[:top_k]

    def rank_pages_subset(self, query: str, pages: Iterable[int], top_k: int = 6) -> list[int]:
        """Score a specific page subset using combined raw + PageCard BM25."""
        qt = tokenize(query)
        s_raw = self.bm_raw.get_scores(qt)
        s_card = self.bm_card.get_scores(qt)
        uniq = [p for p in dict.fromkeys(pages) if 1 <= p <= self.n]
        return sorted(uniq, key=lambda p: -(s_raw[p - 1] + s_card[p - 1]))[:top_k]

    def rank_chunk_bm25_pages(self, query: str, top_k: int = 5) -> list[int]:
        """Baseline flat-chunk BM25 mapped back to 1-based physical pages."""
        scores = self.bm_chunk.get_scores(tokenize(query))
        order = sorted(range(len(self.chunks)), key=lambda i: -scores[i])
        out: list[int] = []
        for idx in order:
            pg = self.chunks[idx][0]
            if pg not in out:
                out.append(pg)
                if len(out) >= top_k:
                    break
        return out
