"""Zero-LLM Structural AnchorMap, Rule PageCards, and Printed-TOC (G7) calibration."""
from __future__ import annotations

import re
from typing import Iterable

from ..docbase.pdf_loader import PdfDocument
from ..runtime.text import BM25Okapi, rrf, tokenize
from ..types import PageCard

_HEADING_RE = re.compile(
    r"^(?:ITEM\s+\d+[A-Z]?\.?|PART\s+[IVX]+|NOTE\s+\d+|SECTION\s+\d+|CHAPTER\s+\d+|"
    r"第[一二三四五六七八九十百\d]+[章节条款]|\d+(?:\.\d+){0,3}\.?|[A-Z][A-Z\s,&/-]{5,60})$"
)
_TABLE_CUE_RE = re.compile(
    r"(?:consolidated\s+(?:statements?|balance\s+sheets?)|cash\s+flows?|operations|income|"
    r"shareholders['’]?\s+equity|segment|revenue|operating\s+income|capital\s+expenditures?|"
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

# --- Printed-TOC signals -------------------------------------------------------------------
# A real TOC is a list of entries that END in a page number, and those page numbers grow
# monotonically down the page. Keyword counts ("Section 12(b)", "Item 405") are NOT a TOC
# signal: a 10-K cover page is full of them, which is exactly how v0.5 misclassified p1.
_LONE_NUM_RE = re.compile(r"^\d{1,3}$")
_NAV_LINK_LINES = frozenset({"table of contents", "contents", "back to contents", "return to table of contents"})
_DOT_LEADER_RE = re.compile(r"(?:(?:\.\s?){3,}|…{2,}|·{3,})\s*\d{1,3}\s*$")
_TRAIL_NUM_RE = re.compile(r"(?:\s|\.{2,}|…+|·+)(\d{1,3})\s*$")

# --- Printed page number in header/footer ----------------------------------------------------
_FOOTER_NUM_RES = (
    re.compile(r"^(?:page|p\.|pg\.?|slide)?\s*[-–—]?\s*(\d{1,3})\s*[-–—]?(?:\s*(?:of|/)\s*\d{1,4})?$", re.I),
    re.compile(r"^.{2,80}?\s*[|•·]\s*(\d{1,3})$"),
    re.compile(r"^(\d{1,3})\s*[|•·]\s*.{2,80}$"),
    re.compile(r"^第\s*(\d{1,3})\s*页(?:\s*[/共]\s*\d{1,4}\s*页?)?$"),
)

# --- Page-reference parsing in user queries ---------------------------------------------------
_NUM_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}
_ORDINAL_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
    "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13,
    "fourteenth": 14, "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
    "nineteenth": 19, "twentieth": 20,
}
_PG = r"(?:pages?|slides?)"
_REF_NUM_RE = re.compile(
    r"(?:\b(?:pages?|slides?|pg\.?)|(?<![A-Za-z])pp?\.)\s*(?:no\.?\s*|number\s*|#\s*)?(\d{1,4})"
    r"(?:\s*(?:-|–|to|and|&|,|through)\s*(\d{1,4}))?",
    re.I,
)
_REF_WORD_RE = re.compile(r"\b" + _PG + r"\s+(" + "|".join(_NUM_WORDS) + r")\b", re.I)
_REF_ORD_NUM_RE = re.compile(r"\b(\d{1,3})(?:st|nd|rd|th)\s+" + _PG + r"\b", re.I)
_REF_ORD_WORD_RE = re.compile(r"\b(" + "|".join(_ORDINAL_WORDS) + r")\s+" + _PG + r"\b", re.I)
_REF_LAST_RE = re.compile(r"\b(?:last|final|closing|back)\s+" + _PG + r"\b", re.I)
_REF_PENULT_RE = re.compile(r"\b(?:penultimate|second[\s-]+(?:to|from)[\s-]+(?:the[\s-]+)?last)\s+" + _PG + r"\b", re.I)
_REF_COVER_RE = re.compile(r"\b(?:cover|front|title)\s+" + _PG + r"\b", re.I)
_REF_CJK_RE = re.compile(r"第\s*(\d{1,4})\s*[页頁]")

_NUM_TOKEN_RE = re.compile(r"[$¥€(]?-?\d[\d,.]*%?\)?")
_CONTINUED_RE = re.compile(r"\(?\b(?:continued|cont'd|cont\.)\)?|续表|（续）|\(续\)", re.I)


def _is_numeric_dense(line: str, min_nums: int = 2) -> bool:
    return len(_NUM_TOKEN_RE.findall(line)) >= min_nums


def _toc_signals(text: str) -> tuple[int, int, list[int]]:
    """Return (dot_leader_lines, entry_lines, trailing_page_refs) for printed-TOC detection.

    An entry "has a page ref" when it ends with a 1-3 digit number (after whitespace or dot
    leaders), or when the NEXT line is a lone number (common in PDF text extraction where the
    page column is emitted as its own line).
    """
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    dots = sum(1 for ln in lines if _DOT_LEADER_RE.search(ln))
    entries = 0
    refs: list[int] = []
    for i, ln in enumerate(lines):
        if _LONE_NUM_RE.fullmatch(ln):
            continue
        entries += 1
        # Dot leaders inflate line length; measure the line with leader runs collapsed.
        if len(re.sub(r"(?:\.\s?){3,}|…+|·{3,}", " ", ln)) > 120 or _is_table_row(ln):
            continue  # prose paragraphs and numeric table rows are not TOC entries
        m = _TRAIL_NUM_RE.search(ln)
        if m:
            refs.append(int(m.group(1)))
        elif i + 1 < len(lines) and _LONE_NUM_RE.fullmatch(lines[i + 1]):
            refs.append(int(lines[i + 1]))
    return dots, entries, refs


_LEAD_NUMBERING_RE = re.compile(
    r"^(?:(?:item|note|part|section|chapter|article|schedule|exhibit|appendix)\s+)?"
    r"\d+(?:\.\d+)*[A-Z]?[.):]?\s+",
    re.I,
)
_STANDALONE_NUM_RE = re.compile(r"^[$¥€(]?-?\d[\d,.]*%?\)?$")


def _is_table_row(line: str) -> bool:
    """True when a line carries >= 2 standalone numbers besides its leading section numbering.

    TOC entries are "<numbering> <title> <page>"; table rows are "<label> <value> <value>".
    """
    toks = _LEAD_NUMBERING_RE.sub("", line.strip()).split()
    return sum(1 for t in toks if _STANDALONE_NUM_RE.match(t)) >= 2


def _mostly_non_decreasing(nums: list[int], min_ratio: float = 0.7) -> bool:
    if len(nums) < 2:
        return False
    ok = sum(1 for a, b in zip(nums, nums[1:]) if b >= a)
    return ok / (len(nums) - 1) >= min_ratio


def looks_like_printed_toc(text: str, max_page: int | None = None) -> bool:
    """Generic printed-TOC detector.

    Requires (a) >= 3 dot-leader lines, OR (b) >= 45% of entry lines ending in a page number,
    OR (c) a separately extracted page-number column (a block of >= 6 lone numbers covering
    >= 45% of the entries). In every case the page numbers must be mostly non-decreasing (a TOC
    lists sections in document order; a financial table's numbers do not grow monotonically)
    and, when ``max_page`` is known, must not point past the end of the document.
    """
    dots, entries, refs = _toc_signals(text)

    def plausible(nums: list[int]) -> bool:
        if not _mostly_non_decreasing(nums):
            return False
        if max_page is None:
            return True
        return sum(1 for v in nums if v <= max_page + 5) >= 0.9 * len(nums)

    if dots >= 3 and plausible(refs):
        return True
    if entries >= 5 and len(refs) >= 4 and len(refs) / entries >= 0.45 and plausible(refs):
        return True
    lone = [int(ln.strip()) for ln in (text or "").splitlines() if _LONE_NUM_RE.fullmatch(ln.strip())]
    return len(lone) >= 6 and entries >= 6 and len(lone) / entries >= 0.45 and plausible(lone)


def is_toc_like(text: str, page_num: int) -> bool:
    """Heuristic TOC-page detector (only the opening pages can be a printed TOC)."""
    if page_num > 12:
        return False
    return looks_like_printed_toc(text or "")


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
            # Many filings repeat a "Table of Contents" back-link (and a page number) on every page;
            # those must not become the page heading.
            lines = [ln for ln in lines if ln.lower() not in _NAV_LINK_LINES and not _LONE_NUM_RE.fullmatch(ln)] or lines
            heading = ""
            for ln in lines[:8]:
                if 3 <= len(ln) <= 95 and (_NOTE_RE.match(ln) or _HEADING_RE.match(ln) or _TABLE_CUE_RE.search(ln)):
                    heading = ln
                    if _NOTE_RE.match(ln):
                        # Update current section when no explicit bookmark exists on this page,
                        # so filings with incomplete PDF bookmarks (e.g. missing Item 8) stay accurate.
                        if not sec_by_page or (pno not in sec_by_page and re.match(r"^(?:ITEM\s+\d+[A-Z]?|PART\s+[IVX]+)\b", ln, re.I)):
                            cur_sec = ln
                    break
            if not heading:
                # Scan slightly deeper for mid-page Note / Item headers (e.g. Note 6 starting after a debt table)
                for ln in lines[8:18]:
                    if 5 <= len(ln) <= 90 and _NOTE_RE.match(ln):
                        heading = ln
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
        """Detect printed Table-of-Contents pages in the opening pages (G7 gate).

        v0.6: keyword counts alone ("Section 12(b)", "Item 405" ...) no longer qualify a page;
        a TOC must show dot leaders or a high ratio of entries ending in monotonic page numbers.
        A "Contents" title lowers the bar slightly but still requires page-number evidence.
        """
        cached = getattr(self, "_printed_toc_cache", None)
        if cached is not None:
            return list(cached)
        hits: list[int] = []
        for i in range(min(8, self.n)):
            txt = self.doc.pages[i] or ""
            head = txt[:200].lower()
            titled = ("contents" in head) or ("目录" in txt[:200]) or ("目 录" in txt[:200])
            if looks_like_printed_toc(txt, max_page=self.n):
                hits.append(i + 1)
                continue
            if titled:
                _, entries, refs = _toc_signals(txt)
                if len(refs) >= 3 and len(refs) / max(1, entries) >= 0.3 and _mostly_non_decreasing(refs):
                    hits.append(i + 1)
        self._printed_toc_cache = hits[:2]
        return list(self._printed_toc_cache)

    # ------------------------------------------------------------------------------------
    # Printed page labels & explicit page references ("page 5", "slide 3", "last page")
    # ------------------------------------------------------------------------------------
    def _detect_footer_number(self, page: int) -> int | None:
        lines = [ln.strip() for ln in (self.doc.pages[page - 1] or "").splitlines() if ln.strip()]
        if not lines:
            return None
        # Footers are more common than headers, so check the last lines first.
        cands = lines[-3:][::-1] + lines[:3]
        for ln in cands:
            if len(ln) > 90:
                continue
            for rx in _FOOTER_NUM_RES:
                m = rx.match(ln)
                if m:
                    v = int(m.group(1))
                    if 1 <= v <= 999:
                        return v
        return None

    def printed_page_map(self) -> dict[int, str]:
        """Physical page -> printed page label.

        Priority: (1) the PDF /PageLabels tree (authoritative when present); (2) lone page
        numbers in the first/last 3 lines, accepted only when their offset to the physical
        page is consistent across several pages (one stray number in a table is not a label).
        Pages between the first and last confirmed detection inherit the dominant offset.
        """
        cached = getattr(self, "_printed_map_cache", None)
        if cached is not None:
            return cached
        out: dict[int, str] = {}
        labels = getattr(self.doc, "page_labels", None) or []
        if len(labels) == self.n:
            out = {i + 1: lb for i, lb in enumerate(labels) if lb}
        else:
            det: dict[int, int] = {}
            for p in range(1, self.n + 1):
                v = self._detect_footer_number(p)
                if v is not None:
                    det[p] = v
            if det:
                offs: dict[int, int] = {}
                for p, v in det.items():
                    offs[v - p] = offs.get(v - p, 0) + 1
                need = max(2, int(0.2 * len(det)))
                valid = {o for o, c in offs.items() if c >= need}
                for p, v in det.items():
                    if (v - p) in valid:
                        out[p] = str(v)
                if valid:
                    dom = max(valid, key=lambda o: offs[o])
                    confirmed = [p for p in out if int(out[p]) - p == dom]
                    if len(confirmed) >= 3:
                        for p in range(min(confirmed), max(confirmed) + 1):
                            if p not in out and p + dom >= 1:
                                out[p] = str(p + dom)
        self._printed_map_cache = out
        return out

    def resolve_page_refs(self, query: str, max_pages: int = 8) -> list[int]:
        """Map explicit page references in a query to candidate PHYSICAL pages.

        Handles "page N", "pages N-M", "slide N", "p. N", "pp. N", "page two", "2nd page",
        "second slide", "last page", "penultimate page", "cover page", "第N页". For every
        number N both the physical page N and every page whose printed label is N are returned,
        because users quote whatever number they see printed on the page.
        """
        q = query or ""
        if not q.strip() or self.n == 0:
            return []
        nums: list[int] = []
        specials: list[int] = []
        for m in _REF_NUM_RE.finditer(q):
            a = int(m.group(1))
            b = int(m.group(2)) if m.group(2) else None
            if b is not None and a < b <= a + 5:
                nums.extend(range(a, b + 1))
            else:
                nums.append(a)
                if b is not None:
                    nums.append(b)
        for m in _REF_WORD_RE.finditer(q):
            nums.append(_NUM_WORDS[m.group(1).lower()])
        for m in _REF_ORD_NUM_RE.finditer(q):
            nums.append(int(m.group(1)))
        for m in _REF_ORD_WORD_RE.finditer(q):
            nums.append(_ORDINAL_WORDS[m.group(1).lower()])
        for m in _REF_CJK_RE.finditer(q):
            nums.append(int(m.group(1)))
        if _REF_PENULT_RE.search(q):
            specials.append(max(1, self.n - 1))
        elif _REF_LAST_RE.search(q):
            specials.append(self.n)
            # A near-empty back cover usually means the user means the last content page.
            if self.n > 1 and len((self.doc.pages[-1] or "").strip()) < 20 and not self.vision_summaries.get(self.n):
                specials.append(self.n - 1)
        if _REF_COVER_RE.search(q):
            specials.append(1)

        printed = self.printed_page_map()
        by_label: dict[str, list[int]] = {}
        for p, lb in printed.items():
            by_label.setdefault(lb.strip().lower(), []).append(p)

        uniq = list(dict.fromkeys(nums))
        cands = list(specials)
        cands += [n_ for n_ in uniq if 1 <= n_ <= self.n]  # physical interpretation first
        for n_ in uniq:  # then printed-label interpretation
            cands += sorted(by_label.get(str(n_), []))
        out: list[int] = []
        for p in cands:
            if 1 <= p <= self.n and p not in out:
                out.append(p)
        return out[:max_pages]

    # ------------------------------------------------------------------------------------
    # Table continuation across page breaks
    # ------------------------------------------------------------------------------------
    @staticmethod
    def _is_heading_line(ln: str) -> bool:
        if not (3 <= len(ln) <= 95) or _is_numeric_dense(ln) or re.fullmatch(r"[\d.,\s$%()\-–]+", ln):
            return False
        return bool(_HEADING_RE.match(ln) or _STMT_RE.search(ln) or _NOTE_RE.match(ln) or _TABLE_CUE_RE.search(ln))

    def continuation_neighbors(self, page: int) -> list[int]:
        """Adjacent pages a table on ``page`` probably continues from / into.

        * p-1 when the page carries a table but opens without a heading-like line (or says
          "continued"): the table's title and column header are likely on the previous page.
        * p+1 when the last body lines (footer page numbers ignored) are numeric table rows:
          the table probably runs past the page bottom.
        """
        if not (1 <= page <= self.n):
            return []
        c = self.cards[page - 1]
        lines = [ln.strip() for ln in (self.doc.pages[page - 1] or "").splitlines() if ln.strip()]
        # PageCard.has_table needs 3+ numbers per row; two-period statements only have 2.
        if not (c.has_table or sum(1 for ln in lines if _is_numeric_dense(ln)) >= 4):
            return []
        out: list[int] = []
        head = lines[:6]
        continued = any(_CONTINUED_RE.search(ln) for ln in head)
        if page > 1 and (continued or not any(self._is_heading_line(ln) for ln in head)):
            out.append(page - 1)
        body = [ln for ln in lines if not any(rx.match(ln) for rx in _FOOTER_NUM_RES)]
        tail = body[-3:]
        if page < self.n and len(tail) == 3 and sum(1 for ln in tail if _is_numeric_dense(ln)) >= 2:
            out.append(page + 1)
        return out

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

    def rank_pagecard_rrf(
        self, query: str, top_k: int = 10, exclude_toc: bool = True, demote_toc: bool = False
    ) -> list[int]:
        """Dual BM25 (raw page + PageCard) fused via RRF(k=60).

        ``exclude_toc=True`` (legacy) drops TOC pages entirely. With ``exclude_toc=False`` and
        ``demote_toc=True`` TOC pages are kept but stably pushed behind every non-TOC page.
        """
        qt = tokenize(query)
        s_raw = self.bm_raw.get_scores(qt)
        s_card = self.bm_card.get_scores(qt)
        r_raw = [i + 1 for i in sorted(range(self.n), key=lambda i: -s_raw[i]) if not exclude_toc or (i + 1) not in self.toc_pages][:60]
        r_card = [i + 1 for i in sorted(range(self.n), key=lambda i: -s_card[i]) if not exclude_toc or (i + 1) not in self.toc_pages][:60]
        fused = rrf(r_raw, r_card, k=60)
        if demote_toc and not exclude_toc:
            fused = self.demote_toc(fused)
        return fused[:top_k]

    def demote_toc(self, pages: list[int], protected: Iterable[int] = ()) -> list[int]:
        """Stable partition: non-TOC (or protected) pages first, TOC pages after them."""
        prot = set(protected)
        head = [p for p in pages if p not in self.toc_pages or p in prot]
        tail = [p for p in pages if p in self.toc_pages and p not in prot]
        return head + tail

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
