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
