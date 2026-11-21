"""End-to-end offline pipeline tests covering all 3 adapters, extraction repair, and HTML/MD reporting."""
from __future__ import annotations

from typing import Any

from adapters import Finance10KAdapter, LegalNDAAdapter, ProcurementAuditAdapter
from anchordoc.docbase import PdfDocument
from anchordoc.report import export_html, export_json, export_markdown


class MockLLM:
    """Deterministic offline test double exercising every prompt stage including quote repair."""

    model = "mock-flash-v1"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def snapshot_stats(self) -> dict[str, Any]:
        return {"calls": len(self.calls)}

    def chat_json(
        self,
        purpose: str,
        system: str,
        user: str,
        max_tokens: int = 700,
        images_b64: list[str] | None = None,
    ) -> tuple[dict[str, Any], str]:
        self.calls.append(purpose)
        if purpose == "anchordoc_qexp":
            return {"keywords": ["Consolidated", "Operations", "Revenues", "ISO 9001"], "section_hint": "Financial"}, "{}"
        if purpose == "anchordoc_triage":
            return {"selected_pages": [2], "complete": False, "missing_terms": ["Gross profit", "ISO"]}, "{}"
        if purpose == "anchordoc_followup":
            return {"selected_pages": [3]}, "{}"
        if purpose == "anchordoc_nav":
            return {"direct_pages": [3], "ranges": [[2, 3]], "rationale": "statement"}, "{}"
        if purpose == "anchordoc_rerank":
            return {"ranked_pages": [3, 2]}, "{}"
        if purpose == "anchordoc_num_qa":
            return {
                "can_answer": True,
                "verbatim_quote": "Revenues $1,200 Cost of sales $720",
                "formula": "(1200 - 720) / 1200 * 100",
                "unit": "%",
                "answer": "40.0%",
                "cited_pages": [3],
                "reason": "Computed directly from Consolidated Statements of Operations.",
            }, "{}"
        if purpose == "anchordoc_locate":
            return {"ranked_pages": [3, 2], "sufficient": True, "missing_terms": []}, "{}"
        if purpose == "anchordoc_num_qa_v6":
            return {
                "evidence": [{"page": 3, "quote": "Revenues $1,200 Cost of sales $720"}],
                "operands": [{"name": "Revenues", "value": "1,200", "page": 3}, {"name": "Cost of sales", "value": "720", "page": 3}],
                "formula": "(1200 - 720) / 1200 * 100",
                "reasoning": "Gross margin = (Revenues - Cost of sales) / Revenues.",
                "can_answer": True,
                "unit": "%",
                "final_answer": "40.0%",
                "cited_pages": [3],
            }, "{}"
        if purpose == "anchordoc_prose_nli_v6":
            return {
                "topic_clauses": ["s0001"],
                "verbatim_quote": "Receiving Party shall not disclose Confidential Information to any third party without prior written consent.",
                "analysis": "Explicit prohibition clause found.",
                "hyp_quantifier": "plain",
                "stance": "supports",
                "permitted_in_some_case": False,
                "counterexample": False,
            }, "{}"
        if purpose == "anchordoc_span_qexp":
            return {"keywords": ["third parties", "disclose", "written consent"]}, "{}"
        if purpose == "anchordoc_span_gap":
            return {"direct_clause_present": True, "missing_phrases": []}, "{}"
        if purpose == "anchordoc_prose_audit":
            return {
                "label": "entailment",
                "confidence": "high",
                "verbatim_quote": "Receiving Party shall not disclose Confidential Information to any third party without prior written consent.",
                "reason": "Explicit prohibition clause found.",
            }, "{}"
        if purpose == "anchordoc_extract":
            # Intentionally return a paraphrased quote on first pass to trigger quote-repair loop!
            return {
                "items": [
                    {
                        "category": "qualification",
                        "title": "ISO 9001 Certificate",
                        "requirement": "Vendor must attach a valid ISO 9001 quality certificate.",
                        "consequence_type": "mandatory",
                        "scope": "all vendors",
                        "sources": [{"sid": "s00002", "quote": "paraphrased non-verbatim text"}],
                    }
                ]
            }, "{}"
        if purpose == "anchordoc_extract_repair":
            return {
                "sources": [{"sid": "s00002", "quote": "Bidder must attach a valid ISO 9001 certificate."}]
            }, "{}"
        if purpose == "anchordoc_compliance_facts":
            return {
                "material_required": True,
                "offer_kind": "material",
                "covered": True,
                "against": False,
                "form_issue": False,
                "verbatim_quote": "Attached: ISO 9001:2015 Certificate No. QA-99881 valid through 2028.",
                "cited_pages": [3],
                "reason": "Valid ISO 9001 certificate attached on page 3.",
            }, "{}"
        return {}, "{}"


def test_all_three_adapters_and_exporters(tmp_path):
    llm = MockLLM()

    # 1. Finance10KAdapter end-to-end with Python AST arithmetic verification
    fin_doc = PdfDocument.from_pages(
        [
            "ACME 10-K Cover",
            "Item 1 Business Overview",
            "Consolidated Statements of Operations\nRevenues $1,200 Cost of sales $720\nGross profit $480",
        ]
    )
    v_fin = Finance10KAdapter(fin_doc, llm).query("What is FY2025 gross margin?")
    assert v_fin.status == "answered"
    assert v_fin.calc_value == 40.0
    assert v_fin.quote_verified is True
    assert v_fin.cited_pages == [3]
    # v0.6 locate normal path (single query-aware locator call instead of triage + rerank)
    assert "anchordoc_locate" in llm.calls and "anchordoc_triage" not in llm.calls
    assert v_fin.retrieval_trace["layers"]["locate_ranked"] == [3, 2]
    assert v_fin.retrieval_trace["pages"][:2] == [3, 2]
    assert v_fin.meta["composition"] in {"CALC_CONSISTENT", "CALC_REPLACED"}

    # 2. LegalNDAAdapter end-to-end
    nda = (
        "1. CONFIDENTIALITY\n"
        "Receiving Party shall not disclose Confidential Information to any third party without prior written consent.\n"
        "2. TERM\nThis Agreement lasts three years."
    )
    v_nda = LegalNDAAdapter(nda, llm).check_clause("Receiving Party shall not disclose to third parties.")
    assert v_nda.status == "pass"
    assert v_nda.quote_verified is True

    # 3. ProcurementAuditAdapter: extraction with automatic quote-repair + compliance audit
    spec_doc = PdfDocument.from_pages(
        ["Section 1 Qualification\nBidder must attach a valid ISO 9001 certificate."]
    )
    sub_doc = PdfDocument.from_pages(
        [
            "Submission Cover",
            "Commercial Response",
            "Quality Certifications\nAttached: ISO 9001:2015 Certificate No. QA-99881 valid through 2028.",
        ]
    )
    proc = ProcurementAuditAdapter(llm)
    checklist = proc.extract_checklist(spec_doc)
    assert len(checklist) == 1
    assert checklist[0].quote_verified is True
    assert "anchordoc_extract_repair" in llm.calls  # Quote-repair loop was exercised!

    verdicts = proc.audit_submission(sub_doc, checklist)
    assert len(verdicts) == 1
    assert verdicts[0].status == "pass"
    assert verdicts[0].quote_verified is True

    # 4. Exporters (JSON, Markdown, standalone HTML)
    all_verdicts = [v_fin, v_nda, *verdicts]
    jp = export_json(all_verdicts, tmp_path / "report.json")
    hp = export_html(all_verdicts, tmp_path / "report.html")
    md = export_markdown(all_verdicts)
    assert jp.is_file() and hp.is_file()
    assert "Python Calc" in hp.read_text(encoding="utf-8")
    assert "ISO 9001 Certificate" in md
