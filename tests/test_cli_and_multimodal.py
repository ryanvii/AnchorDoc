"""Unit tests for multimodal VLM adapter, CLI argument parsing, and two-pane HTML exporter."""
from __future__ import annotations

from typing import Any

from adapters import MultimodalDocAdapter
from anchordoc.cli import build_parser
from anchordoc.docbase import PdfDocument
from anchordoc.report import export_html


class MockMultimodalLLM:
    model = "mock-vlm-v1"

    def snapshot_stats(self) -> dict[str, Any]:
        return {"calls": 1}

    def chat_json(
        self,
        purpose: str,
        system: str,
        user: str,
        max_tokens: int = 700,
        images_b64: list[str] | None = None,
    ) -> tuple[dict[str, Any], str]:
        if purpose == "anchordoc_qexp":
            return {"keywords": ["Chart", "Sales", "2025"], "section_hint": "Results"}, "{}"
        if purpose == "anchordoc_triage":
            return {"selected_pages": [2], "complete": True, "missing_terms": []}, "{}"
        if purpose == "anchordoc_rerank":
            return {"ranked_pages": [2]}, "{}"
        if purpose == "anchordoc_locate":
            return {"ranked_pages": [2], "sufficient": True, "missing_terms": []}, "{}"
        if purpose == "anchordoc_mm_qa_v6":
            if "missing_topic" in user:
                return {"observations": [], "evidence_status": "absent", "entity_time_match": True,
                        "final_answer": "Not answerable", "cited_pages": []}, "{}"
            return {
                "observations": [{"page": 2, "detail": "2025 North America 42%"}],
                "evidence_status": "stated",
                "entity_time_match": True,
                "derivation": "",
                "formula": "42 / 100 * 100",
                "reasoning": "Read directly from bar chart on page 2.",
                "final_answer": "42%",
                "cited_pages": [2],
            }, "{}"
        if purpose in {"anchordoc_mm_qa", "anchordoc_mm_qa_v2"}:
            if "missing_topic" in user:
                return {
                    "exact_subject_present": False,
                    "exact_fact_visibly_supported": False,
                    "can_answer": False,
                    "answer": "Not answerable",
                    "cited_pages": [],
                    "reason": "Not found",
                }, "{}"
            return {
                "exact_subject_present": True,
                "exact_fact_visibly_supported": True,
                "can_answer": True,
                "answer": "42%",
                "formula": "42 / 100 * 100",
                "cited_pages": [2],
                "supporting_detail": "2025 North America 42%",
                "reason": "Read directly from bar chart on page 2.",
            }, "{}"
        return {}, "{}"


def test_selective_vision_indexer_skips_text_rich_pages(tmp_path):
    from anchordoc.index import SelectiveVisionIndexer

    fake_pdf = tmp_path / "dummy.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4")
    doc = PdfDocument.from_pages(
        [
            "A" * 500,  # text-rich digital page -> skip!
            "Short",    # low-text scanned page -> select!
            "",         # pure image page -> select!
        ],
        path=fake_pdf,
    )
    indexer = SelectiveVisionIndexer(llm=MockMultimodalLLM(), min_text_chars=180)
    assert indexer.select_visual_pages(doc) == [2, 3]


def test_multimodal_adapter_answered_and_unanswerable(tmp_path):
    doc = PdfDocument.from_pages(
        [
            "Report Cover",
            "Figure 1: Regional Growth Chart\n2025 North America 42%",
        ]
    )
    adapter = MultimodalDocAdapter(doc, MockMultimodalLLM())
    v_ok = adapter.query("What percentage is shown in Figure 1 for 2025?")
    assert v_ok.status == "answered"
    assert v_ok.calc_value == 42.0
    assert v_ok.cited_pages == [2]
    assert v_ok.retrieval_trace["layers"]["locate_ranked"] == [2]
    assert v_ok.meta["evidence_status"] == "stated"

    v_unans = adapter.query("What is missing_topic revenue?")
    assert v_unans.status == "abstain"
    assert v_unans.answer == "Not answerable"

    hp = export_html([v_ok, v_unans], tmp_path / "mm_report.html", title="MM Demo")
    html_text = hp.read_text(encoding="utf-8")
    assert "showPage(" in html_text
    assert "Original Page Evidence Preview" in html_text
    assert "42%" in html_text


def test_cli_parser_subcommands():
    parser = build_parser()
    a1 = parser.parse_args(["outline", "sample.pdf"])
    assert a1.command == "outline" and a1.pdf == "sample.pdf"
    a2 = parser.parse_args(["query", "sample.pdf", "What is CapEx?", "--vision", "--html", "out.html"])
    assert a2.command == "query" and a2.vision is True and a2.html == "out.html"
    a3 = parser.parse_args(["audit", "sample.pdf", "--checklist", "rules.json", "--html", "out.html"])
    assert a3.command == "audit" and a3.checklist == "rules.json"
