"""v0.6 Query-aware Locator tests: snippets, TOC detection/demotion, continuation, page refs, tiles."""
from __future__ import annotations

import base64
import io
from typing import Any

from anchordoc.docbase import PdfDocument
from anchordoc.index import DocumentIndex
from anchordoc.retrieve import AnchorRetriever


class FakeLLM:
    """Scripted offline LLM: maps purpose -> JSON reply and records every prompt."""

    model = "fake-locator"

    def __init__(self, replies: dict[str, dict[str, Any]] | None = None) -> None:
        self.replies = replies or {}
        self.calls: list[tuple[str, str]] = []

    def snapshot_stats(self) -> dict[str, Any]:
        return {"calls": len(self.calls)}

    def chat_json(self, purpose: str, system: str, user: str, max_tokens: int = 700,
                  images_b64: list[str] | None = None) -> tuple[dict[str, Any], str]:
        self.calls.append((purpose, user))
        return dict(self.replies.get(purpose, {})), "{}"

    def purposes(self) -> list[str]:
        return [p for p, _ in self.calls]

    def prompt(self, purpose: str) -> str:
        return next(u for p, u in self.calls if p == purpose)


FILLER = "General discussion of the business environment and market conditions. " * 8

COVER_10K = "\n".join([
    "UNITED STATES SECURITIES AND EXCHANGE COMMISSION",
    "Washington, D.C. 20549",
    "FORM 10-K",
    "ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE SECURITIES EXCHANGE ACT OF 1934",
    "For the fiscal year ended December 31, 2023",
    "ACME Industrial Holdings Inc.",
    "Securities registered pursuant to Section 12(b) of the Act:",
    "Title of each class Trading Symbol Name of each exchange on which registered",
    "Common Stock, par value $0.01 per share ACME New York Stock Exchange",
    "1.750% Notes due 2030 ACME30 New York Stock Exchange",
    "Securities registered pursuant to Section 12(g) of the Act: None",
    "Indicate by check mark if the registrant is a well-known seasoned issuer.",
    "1",
    "Indicate by check mark whether the registrant has filed all reports required by Section 13 or 15(d).",
    "2",
    "Indicate by check mark whether disclosure of delinquent filers pursuant to Item 405 is not contained.",
    "3",
    "Indicate by check mark whether the registrant is a shell company as defined in Rule 12b-2.",
    "4",
    "Aggregate market value of voting stock held by non-affiliates was approximately $12.3 billion.",
    "Documents incorporated by reference: portions of the proxy statement.",
])

TOC_PAGE = "\n".join([
    "TABLE OF CONTENTS",
    "PART I",
    "Item 1. Business .......... 3",
    "Item 1A. Risk Factors .......... 4",
    "Item 7. Management's Discussion and Analysis .......... 6",
    "Item 8. Financial Statements .......... 8",
    "Note 5. Debt .......... 10",
])


def _ten_k_pages() -> list[str]:
    """12-page synthetic filing (generic structure, no real company data)."""
    pages = [
        COVER_10K,
        TOC_PAGE,
        "ITEM 1. BUSINESS\n" + FILLER,
        "ITEM 1A. RISK FACTORS\n" + FILLER,
        "Overview of operations\n" + FILLER,
        "ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS\n" + FILLER,
        "Liquidity discussion\n" + FILLER,
        "\n".join([
            "CONSOLIDATED BALANCE SHEETS",
            "(in millions) 2023 2022",
            "Cash and equivalents 1,200 1,050",
            "Receivables 800 760",
            "Inventories 640 610",
            "Property, plant and equipment 2,300 2,250",
            "Goodwill 900 900",
            "Total assets 5,840 5,570",
        ]),
        "\n".join([
            "Accounts payable 520 500",
            "Accrued liabilities 310 290",
            "Long-term debt 1,900 1,950",
            "Other liabilities 220 210",
            "Total liabilities 2,950 2,950",
            "Total equity 2,890 2,620",
        ]),
        "NOTE 5. DEBT\n" + FILLER,
        "Commitments and contingencies\n" + FILLER,
        "Signatures\n" + FILLER,
    ]
    return pages


# ---------------------------------------------------------------------------------------------
# 1. Snippet windowing sees evidence far beyond the first 1600 characters
# ---------------------------------------------------------------------------------------------
def test_pack_snippets_finds_evidence_beyond_head_truncation():
    long_page = (
        "Management Report\n" + ("Narrative paragraph about strategy and outlook.\n" * 80)
        + "Registered notes outstanding: 1.750% Notes due 2030 principal 4,321 million\n"
        + "Closing remarks."
    )
    assert long_page.index("1.750% Notes") > 3000
    doc = PdfDocument.from_pages(["Cover", long_page, "Appendix"])
    r = AnchorRetriever(doc)
    snip = r._pack_snippets([2], "principal of the notes due 2030")
    assert "[Page 2]" in snip
    assert "4,321" in snip
    assert len(snip) < 1300  # ~900-char body budget + header
    # Legacy head truncation could never see it
    assert "4,321" not in r._pack_pages([2], 1600)


def test_pack_snippets_keeps_table_header_and_vision_summary():
    page = "\n".join(["Segment results", "(in millions) 2023 2022"] + [f"Line item {i} {i*10} {i*9}" for i in range(60)]
                     + ["Line item target 9,999 8,888"])
    doc = PdfDocument.from_pages([page, "x"])
    r = AnchorRetriever(doc, vision_summaries={1: "Bar chart of segment revenue"})
    snip = r._pack_snippets([1], "line item target")
    assert "9,999" in snip
    assert "(in millions) 2023 2022" in snip  # column header carried even though window is far below
    assert "Bar chart of segment revenue" in snip


# ---------------------------------------------------------------------------------------------
# 2. TOC detection: 10-K cover page is NOT a TOC; real TOC still detected
# ---------------------------------------------------------------------------------------------
def test_cover_page_not_toc_and_real_toc_detected():
    idx = DocumentIndex(PdfDocument.from_pages(_ten_k_pages()))
    assert idx.find_printed_toc_pages() == [2]
    assert 1 not in idx.toc_pages
    assert 2 in idx.toc_pages


def test_financial_table_page_not_toc():
    page = "\n".join(f"Row {i} value {v}" for i, v in enumerate([45, 12, 300, 7, 88, 19, 230, 5, 61, 140]))
    idx = DocumentIndex(PdfDocument.from_pages([page, "x", "y"]))
    assert idx.find_printed_toc_pages() == []


# ---------------------------------------------------------------------------------------------
# 3. Soft TOC demotion
# ---------------------------------------------------------------------------------------------
def test_soft_toc_demotion_zero_llm():
    doc = PdfDocument.from_pages(_ten_k_pages())
    r = AnchorRetriever(doc)
    tr = r.retrieve("Item 8 Financial Statements Note 5 Debt", top_k=12)
    assert tr.g7_toc_pages == [2]
    assert 2 in tr.pages  # not hard-excluded any more ...
    non_toc = [p for p in tr.pages if p != 2]
    assert tr.pages.index(2) == len(non_toc)  # ... but pushed behind every non-TOC page
    tr5 = r.retrieve("Item 8 Financial Statements Note 5 Debt", top_k=5)
    assert 2 not in tr5.pages  # lexical-only TOC hit does not steal a top-5 slot


def test_nav_direct_toc_page_and_cover_never_excluded():
    doc = PdfDocument.from_pages(_ten_k_pages())
    llm = FakeLLM({
        "anchordoc_qexp": {"keywords": ["securities registered", "Section 12(b)"], "section_hint": "cover"},
        "anchordoc_nav": {"direct_pages": [1, 2], "ranges": []},
        "anchordoc_locate": {"ranked_pages": [1], "sufficient": True, "missing_terms": []},
    })
    tr = AnchorRetriever(doc, llm=llm).retrieve("Which notes are registered under Section 12(b)?", top_k=5)
    assert tr.pages[0] == 1
    assert 2 in tr.pages  # nav direct page is protected even though it is a TOC page
    assert llm.purposes() == ["anchordoc_qexp", "anchordoc_nav", "anchordoc_locate"]


# ---------------------------------------------------------------------------------------------
# 4. Single locate call, conditional B42, graceful degradation
# ---------------------------------------------------------------------------------------------
def test_single_locate_call_and_ranked_first():
    doc = PdfDocument.from_pages(_ten_k_pages())
    llm = FakeLLM({
        "anchordoc_qexp": {"keywords": ["total assets", "balance sheet"], "section_hint": "balance sheets"},
        "anchordoc_nav": {"direct_pages": [8], "ranges": [[8, 9]]},
        "anchordoc_locate": {"ranked_pages": [8, 9], "sufficient": True, "missing_terms": []},
    })
    tr = AnchorRetriever(doc, llm=llm).retrieve("What were total assets in 2023?", top_k=5)
    assert tr.pages[:2] == [8, 9]
    assert "anchordoc_triage" not in llm.purposes() and "anchordoc_rerank" not in llm.purposes()
    assert llm.purposes().count("anchordoc_locate") == 1
    assert not tr.followup_triggered
    assert len(tr.layers["locate_pool"]) <= 15
    assert "[Page 8]" in llm.prompt("anchordoc_locate")


def test_followup_only_when_insufficient_and_gap():
    pages = _ten_k_pages()
    base = {
        "anchordoc_qexp": {"keywords": ["total assets"], "section_hint": ""},
        "anchordoc_nav": {"direct_pages": [8], "ranges": []},
        "anchordoc_followup": {"selected_pages": [10]},
    }
    # insufficient + missing terms -> followup fires
    llm = FakeLLM({**base, "anchordoc_locate": {"ranked_pages": [8], "sufficient": False, "missing_terms": ["debt maturity"]}})
    tr = AnchorRetriever(PdfDocument.from_pages(pages), llm=llm).retrieve("total assets and debt maturity", top_k=5)
    assert tr.followup_triggered
    assert llm.purposes()[-1] == "anchordoc_followup"
    # insufficient, no missing terms, but nav overlaps lexical -> no followup
    llm2 = FakeLLM({**base, "anchordoc_locate": {"ranked_pages": [8], "sufficient": False, "missing_terms": []}})
    tr2 = AnchorRetriever(PdfDocument.from_pages(pages), llm=llm2).retrieve("total assets", top_k=5)
    assert 8 in tr2.layers["c3_rrf"]
    assert not tr2.followup_triggered
    # use_gap_followup=False always disables it
    llm3 = FakeLLM({**base, "anchordoc_locate": {"ranked_pages": [], "sufficient": False, "missing_terms": ["x"]}})
    tr3 = AnchorRetriever(PdfDocument.from_pages(pages), llm=llm3).retrieve("total assets", use_gap_followup=False)
    assert not tr3.followup_triggered and "anchordoc_followup" not in llm3.purposes()


def test_locate_empty_reply_degrades_to_nav_and_rrf():
    doc = PdfDocument.from_pages(_ten_k_pages())
    llm = FakeLLM({"anchordoc_nav": {"direct_pages": [8], "ranges": []}})  # locate -> {}
    tr = AnchorRetriever(doc, llm=llm).retrieve("total assets balance sheet", top_k=5, use_gap_followup=False)
    assert len(tr.pages) == 5
    assert 8 in tr.pages


# ---------------------------------------------------------------------------------------------
# 5. Table continuation expansion
# ---------------------------------------------------------------------------------------------
def test_continuation_neighbors_and_pool_injection():
    doc = PdfDocument.from_pages(_ten_k_pages())
    idx = DocumentIndex(doc)
    # p9 is a headless table page -> its title/header lives on p8
    assert 8 in idx.continuation_neighbors(9)
    # p8 table runs to page bottom -> continues on p9
    assert 9 in idx.continuation_neighbors(8)
    # prose page -> nothing
    assert idx.continuation_neighbors(3) == []

    llm = FakeLLM({
        "anchordoc_qexp": {"keywords": ["long-term debt"], "section_hint": ""},
        "anchordoc_nav": {"direct_pages": [9], "ranges": []},
        "anchordoc_locate": {"ranked_pages": [9], "sufficient": True},
    })
    tr = AnchorRetriever(doc, llm=llm).retrieve("long-term debt accrued liabilities", top_k=5)
    assert 8 in tr.layers.get("continuation", [])
    assert 8 in tr.layers["locate_pool"]
    assert tr.pages[0] == 9  # continuation enters the pool without forcing a top rank


# ---------------------------------------------------------------------------------------------
# 6. Page-reference resolution (physical AND printed labels)
# ---------------------------------------------------------------------------------------------
def test_resolve_page_refs_physical_printed_and_words():
    # 12 physical pages; front matter = 2 pages, then footers print 1..10 (offset -2)
    pages = ["Cover", "Contents"] + [f"Body text of page {i}\nmore text\n{i - 2}" for i in range(3, 13)]
    idx = DocumentIndex(PdfDocument.from_pages(pages))
    assert idx.printed_page_map()[7] == "5"
    refs = idx.resolve_page_refs("What does the chart on page 5 show?")
    assert refs[:2] == [5, 7]  # physical 5 and printed-5 (= physical 7)
    assert idx.resolve_page_refs("Summarize slide two")[0] == 2
    assert idx.resolve_page_refs("What is on the 3rd page?")[0] == 3
    assert idx.resolve_page_refs("What is on the last page?") == [12]
    assert idx.resolve_page_refs("In pp. 4-5, list the items")[:2] == [4, 5]
    assert idx.resolve_page_refs("registered under Section 12(b) of the Act") == []


def test_resolve_page_refs_uses_pdf_page_labels():
    labels = ["i", "ii"] + [str(i) for i in range(1, 9)]
    doc = PdfDocument.from_pages([f"page body {i}" for i in range(10)], page_labels=labels)
    idx = DocumentIndex(doc)
    assert idx.resolve_page_refs("see p. 3") == [3, 5]


def test_page_refs_injected_into_pool_and_final_zero_llm():
    pages = [f"Generic slide content number {i}" for i in range(1, 21)]
    tr = AnchorRetriever(PdfDocument.from_pages(pages)).retrieve("What is the title of slide 17?", top_k=5)
    assert tr.layers["page_refs"][0] == 17
    assert 17 in tr.pages


# ---------------------------------------------------------------------------------------------
# 7. Tile rendering
# ---------------------------------------------------------------------------------------------
def test_render_page_tiles_count_and_overlap(tmp_path):
    import pypdfium2 as pdfium
    from PIL import Image

    pdf_path = tmp_path / "blank.pdf"
    pdf = pdfium.PdfDocument.new()
    pdf.new_page(400, 300)
    pdf.save(str(pdf_path))
    pdf.close()

    doc = PdfDocument.from_pdf(pdf_path)
    tiles = doc.render_page_tiles_b64(1, grid=2, overlap=0.1, scale=1.0)
    assert len(tiles) == 4
    w, h = Image.open(io.BytesIO(base64.b64decode(tiles[0]))).size
    assert 200 < w < 260 and 150 < h < 200  # half page + 10% overlap
    assert doc.page_labels == []  # no /PageLabels tree -> empty list
