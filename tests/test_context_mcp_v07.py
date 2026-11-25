"""Unit tests for v0.7.0 Zero-LLM DocContext API and 4-tool MCP server."""
from __future__ import annotations

from pathlib import Path

import pytest

from anchordoc.context import MAX_LOCATE_TOKENS, MAX_OUTLINE_TOKENS, DocContext
from anchordoc.docbase import PdfDocument
from anchordoc.mcp_server import AnchorDocMCPApp
from anchordoc.store import BundleStore, TextLayer, extractor_fingerprint, file_sha256


def _synthetic_500p_doc() -> PdfDocument:
    pages: list[str] = []
    for p in range(1, 501):
        sec = f"ITEM {(p - 1) // 25 + 1}. SECTION {(p - 1) // 25 + 1}" if (p - 1) % 25 == 0 else ""
        if p == 237:
            body = (
                "Consolidated Operating Metrics\n"
                "Fiscal Year 2024 Gross Margin was 64.5% on Revenue of 4,200 million and Cost of 1,491 million.\n"
                "Operating cash flow reached 980 million."
            )
        else:
            body = f"Page {p} routine corporate disclosure and operational commentary for segment {p % 7}."
        pages.append((sec + "\n" if sec else "") + body)
    return PdfDocument.from_pages(pages, page_labels=[str(i) for i in range(1, 501)])


def test_500_page_context_hard_caps_and_zero_llm():
    doc = _synthetic_500p_doc()
    ctx = DocContext.from_document(doc)

    # L0 outline hard cap <= 1,500 est_tokens
    out = ctx.outline(depth=1, max_items=40)
    assert out["n_pages"] == 500
    assert out["route_hint"] == "navigate"
    assert out["est_tokens"] <= MAX_OUTLINE_TOKENS
    assert len(out["items"]) > 0

    # L1 locate hard cap <= 1,200 est_tokens and finds p.237 with 0 LLM calls
    loc = ctx.locate("Gross Margin Revenue 4,200 million Cost 1,491", k=5)
    assert loc["est_tokens"] <= MAX_LOCATE_TOKENS
    assert loc["hits"][0]["p"] == 237
    assert "lexical" in loc["hits"][0]["why"]

    # L2 read rejects > 5 pages
    with pytest.raises(ValueError, match="At most 5 pages"):
        ctx.read([1, 2, 3, 4, 5, 6])

    # Stable line anchors + duplicate read ledger
    r1 = ctx.read([237])
    assert "p237:L01 " in r1["pages"][0]["text"]
    r2 = ctx.read([237])
    assert r2["pages"][0].get("already_returned") is True
    r3 = ctx.read([237], force=True)
    assert "p237:L01 " in r3["pages"][0]["text"]

    # Second DocContext over same doc has identical line anchors
    ctx2 = DocContext.from_document(doc)
    assert ctx2.read([237])["pages"][0]["text"] == r1["pages"][0]["text"]


def test_doc_verify_detects_forged_quote_and_computes_grounded_math():
    ctx = DocContext.from_document(_synthetic_500p_doc())
    res = ctx.verify(
        quotes=[
            {"text": "Revenue of 4,200 million and Cost of 1,491 million", "page": 237},
            {"text": "This sentence was completely invented by an LLM"},
        ],
        formula="(4200 - 1491) / 4200 * 100",
    )
    assert res["quotes"][0]["verified"] is True
    assert res["quotes"][0]["matched_pages"] == [237]
    assert res["quotes"][1]["verified"] is False
    assert res["all_quotes_verified"] is False
    assert res["grounding_scope"] == "quote_localized"
    assert res["grounding_pages"] == [237]
    assert abs(res["calc_value"] - 64.5) < 0.01
    assert res["operands_grounded"] is True


def test_mcp_app_four_tools_and_root_allowlist(tmp_path):
    allowed_dir = tmp_path / "allowed"
    forbidden_dir = tmp_path / "forbidden"
    allowed_dir.mkdir()
    forbidden_dir.mkdir()

    import pypdfium2 as pdfium

    def _mk_pdf(p: Path) -> None:
        pdf = pdfium.PdfDocument.new()
        pdf.new_page(200, 200)
        with open(p, "wb") as f:
            pdf.save(f)
        pdf.close()

    ok_pdf = allowed_dir / "report.pdf"
    bad_pdf = forbidden_dir / "secret.pdf"
    bad_txt = allowed_dir / "notes.txt"
    _mk_pdf(ok_pdf)
    _mk_pdf(bad_pdf)
    bad_txt.write_text("not a pdf", encoding="utf-8")

    # Seed frozen text in BundleStore so read/verify work on synthetic blank PDF
    store = BundleStore(tmp_path / "store")
    sha = file_sha256(ok_pdf)
    store.save_text(sha, extractor_fingerprint("pypdfium2"), TextLayer(pages=["Report Title\nNet Profit 250 million."]))

    app = AnchorDocMCPApp(roots=[allowed_dir], store=store)
    tools = app.list_tools()
    assert [t["name"] for t in tools] == ["doc_outline", "doc_locate", "doc_read", "doc_verify"]

    # Reject outside --root and non-PDF
    with pytest.raises(PermissionError):
        app.call_tool("doc_outline", {"path": str(bad_pdf)})
    with pytest.raises(ValueError):
        app.call_tool("doc_outline", {"path": str(bad_txt)})

    # End-to-end 4-tool workflow using doc_id
    out = app.call_tool("doc_outline", {"path": str(ok_pdf)})
    doc_id = out["doc_id"]
    assert out["route_hint"] == "read_all"
    rd = app.call_tool("doc_read", {"doc_id": doc_id, "pages": [1]})
    assert "p1:L02 Net Profit 250 million." in rd["pages"][0]["text"]
    vf = app.call_tool("doc_verify", {"doc_id": doc_id, "quotes": [{"text": "Net Profit 250 million"}], "formula": "250 * 2"})
    assert vf["all_quotes_verified"] is True
    assert vf["calc_value"] == 500.0
    assert vf["operands_grounded"] is True


def test_outline_exposes_primary_statements_and_locate_boosts_cash_flows():
    """Verify general 10-K structural indexing and zero-LLM locate improvements:
    1. Light inflection folding matches `repurchases`, `repurchased`, and `repurchase`.
    2. Primary financial statements (e.g. Cash Flows) appear at tier=1 in depth=1 outline
       even when preceded by an incomplete bookmark (e.g. Item 7 without an Item 8 bookmark).
    3. Statement-intent boosting surfaces the Cash Flows statement in Top-5 when the query
       asks about cash spent/paid, even when verbose quarterly/narrative pages repeat keywords.
    """
    from anchordoc.runtime.text import tokenize

    assert "repurchase" in tokenize("Repurchases of common stock")
    assert "repurchase" in tokenize("shares were repurchased")

    pages = [
        "Item 7. Management's Discussion and Analysis\nQuarterly overview of stock repurchase program.",
        "Item 5. Market for Registrant's Common Equity\nQ4 repurchase of common stock: repurchased 10 million shares in final three months.",
        "Item 8. Financial Statements and Supplementary Data\nIndex to Consolidated Financial Statements\nConsolidated Statements of Cash Flows 5",
        "CONSOLIDATED STATEMENTS OF SHAREHOLDERS’ EQUITY\nRepurchase of common stock (405,549) (73,056)\nRetained earnings 70,400",
        "CONSOLIDATED STATEMENTS OF CASH FLOWS\n(In millions)\nOperating activities: Net income 59,531\nFinancing activities:\nRepurchases of common stock (72,738)",
        "Note 6 - Shareholders' Equity\nDuring 2018 the company repurchased 405.5 million shares for $73.1 billion.",
        "PART IV\nExhibits and schedules regarding common stock repurchase authorizations and shares.",
        "PART IV\nAdditional repurchase program details and shares issued under equity plans.",
    ]
    doc = PdfDocument(path=None, pages=pages, toc_bookmarks=[(1, "Item 7. Management's Discussion and Analysis", 1)])
    ctx = DocContext.from_document(doc, doc_id="filing_test")

    out = ctx.outline(depth=1, max_items=20)
    by_page = {it["p"]: it for it in out["items"]}
    # Page 3 Item 8 refreshes section_path; Pages 4 & 5 surface as tier=1 with real statement titles!
    assert 4 in by_page and "SHAREHOLDERS’ EQUITY" in by_page[4]["title"]
    assert 5 in by_page and "CASH FLOWS" in by_page[5]["title"]
    assert "Item 8" in by_page[5]["title"]

    for q in (
        "In FY2018, how much cash did the company actually spend to repurchase common stock, and how many shares were repurchased?",
        "How much cash did the company spend to buy back stock in 2018?",
    ):
        loc = ctx.locate(q, k=5)
        top_pages = [h["p"] for h in loc["hits"]]
        assert 5 in top_pages[:3], f"Expected Cash Flows page 5 in top-3 for {q!r}, got {top_pages}"
        assert 4 in top_pages[:3], f"Expected Equity page 4 in top-3 for {q!r}, got {top_pages}"


def test_pagecard_captures_multiple_subheadings_and_surfaces_in_outline_and_locate():
    """Verify full-page subheading capture for multi-section pages (lower-page Notes & subsections)."""
    lower_note_page = "\n".join(
        [
            "Note 6 - Debt and Credit Facilities",
            "Maturity schedule of senior notes:",
            "2025 1,200 1,150 4.5%",
            "2026 2,400 2,350 4.8%",
            "2027 3,100 3,050 5.0%",
            "2028 1,800 1,750 5.1%",
            *[f"Debt covenant disclosure paragraph line {i} detailing interest coverage ratios." for i in range(1, 18)],
            "Note 7 - Restructuring and Impairment Charges",
            "During FY2024 the company booked $180 million of one-time restructuring charges inside COGS.",
        ]
    )
    cn_multi_sub_page = "\n".join(
        [
            "第三章 商务与技术要求",
            "3.1 交货期限",
            "合同签订后 30 日历天内完成全部设备供货并到场安装。",
            "3.2 质保期限",
            "本项目整体免费质保期不少于 3 年。",
            "3.3 售后服务响应",
            "接到故障通知后 2 小时内响应，24 小时内到达现场。",
        ]
    )
    pages = [
        "Overview\nGeneral summary mentioning restructuring briefly in passing.",
        lower_note_page,
        cn_multi_sub_page,
    ]
    doc = PdfDocument.from_pages(pages, page_labels=["1", "2", "3"])
    ctx = DocContext.from_document(doc, doc_id="multi_subhead_test")

    c2 = ctx.index.cards[1]
    assert c2.heading == "Note 6 - Debt and Credit Facilities"
    assert "Note 7 - Restructuring and Impairment Charges" in c2.subheadings
    # Numeric table rows must NOT be treated as subheadings
    assert not any(sh.startswith("2025") for sh in c2.subheadings)
    assert "Note 7 - Restructuring and Impairment Charges" in c2.card_text()

    c3 = ctx.index.cards[2]
    assert "3.1 交货期限" in c3.subheadings
    assert "3.2 质保期限" in c3.subheadings
    assert "3.3 售后服务响应" in c3.subheadings

    # Global structural outline & L0 doc_outline(depth=2) both surface lower-page Note 7 & subsections
    struct_outline = ctx.index.build_structural_outline()
    assert "Note 7 - Restructuring and Impairment Charges" in struct_outline

    out = ctx.outline(depth=2, max_items=20)
    assert out["est_tokens"] <= MAX_OUTLINE_TOKENS
    by_p = {it["p"]: it for it in out["items"]}
    assert "Note 7 - Restructuring and Impairment Charges" in by_p[2]["subheadings"]
    assert "3.3 售后服务响应" in by_p[3]["subheadings"]

    loc = ctx.locate("Note 7 Restructuring and Impairment Charges", k=3)
    assert loc["hits"][0]["p"] == 2


