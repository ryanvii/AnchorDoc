"""Unit tests for Zero-LLM Structural AnchorMap and G7 printed-TOC page offset calibration."""
from anchordoc.docbase import PdfDocument
from anchordoc.index import DocumentIndex


def test_anchor_map_and_g7_offset_calibration():
    pages = [
        "ACME CORP 2025 FORM 10-K\nCover Page",
        (
            "TABLE OF CONTENTS\n"
            "PART I\n"
            "ITEM 1. BUSINESS\n1\n"
            "ITEM 1A. RISK FACTORS\n2\n"
            "ITEM 7. MD&A\n3\n"
            "ITEM 8. FINANCIAL STATEMENTS\n4\n"
        ),
        "ITEM 1. BUSINESS\nACME sells industrial robotics and cloud telemetry.",
        "ITEM 1A. RISK FACTORS\nSupply chain and customer concentration risks.",
        "ITEM 7. MD&A\nSummary of Financial Performance\nRevenue 2025 1200 2024 1000\nMargin 40% 38%\nCapex 150 120\nDebt 300 320",
        "ITEM 8. FINANCIAL STATEMENTS\nConsolidated Statements of Operations\nRevenues $1,200 $1,000 $900\nCost of goods $720 $620 $560\nGross profit $480 $380 $340\nOperating income $240 $180 $150",
    ]
    doc = PdfDocument.from_pages(pages)
    idx = DocumentIndex(doc)

    toc_pgs = idx.find_printed_toc_pages()
    assert toc_pgs == [2]

    # Printed page 1 ("ITEM 1. BUSINESS") is physical page 3 -> offset = +2!
    offset = idx.calibrate_printed_offset(doc.page_text(2))
    assert offset == 2

    outline = idx.build_structural_outline()
    assert "p.6:" in outline and "Consolidated Statements of Operations" in outline
    assert "[TABLE]" in outline
    # Zero-LLM RRF should exclude TOC page 2 and rank Income Statement page 6 first
    ranked = idx.rank_pagecard_rrf("Consolidated Statements of Operations Revenues Gross profit", top_k=3)
    assert 2 not in ranked
    assert ranked[0] == 6
