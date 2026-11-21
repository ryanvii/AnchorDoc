"""Unit tests for v0.7.0 ReadAll Grounding Scope Guard & multimodal read_all zoom selection."""
from __future__ import annotations

from anchordoc.adjudicate import FactAdjudicator
from anchordoc.docbase import PdfDocument
from anchordoc.retrieve import AnchorRetriever, SpanRetriever
from anchordoc.runtime import localize_quote_pages


class _StubLLM:
    model = "fake-grounding-model"

    def __init__(self, reply: dict) -> None:
        self.reply = reply

    def chat_json(self, purpose: str, system: str, user: str, max_tokens: int = 700, images_b64=None):
        return dict(self.reply), "{}"

    def snapshot_stats(self):
        return {"calls": 1}


def test_readall_wrong_cited_page_marks_operands_ungrounded():
    doc = PdfDocument.from_pages([
        "Page 1 cover sheet with no financials.",
        "Page 2 revenue table: Total Revenue 800, Cost 500.",
        "Page 3 notes and legal disclaimers.",
    ])
    llm = _StubLLM({
        "evidence": [{"page": 1, "quote": "fabricated quote not on page"}],
        "operands": [{"name": "rev", "value": "800", "page": 1}],
        "formula": "800 - 500",
        "reasoning": "Gross profit is 300.",
        "can_answer": True,
        "unit": "",
        "final_answer": "300",
        "cited_pages": [1],  # wrong page! figures are on page 2
    })
    v = FactAdjudicator(llm).answer_numerical(doc, "What is gross profit?", [1, 2, 3], route="read_all")
    assert v.meta["grounding_scope"] == "cited_pages"
    assert v.meta["operand_grounded"] is False


def test_readall_quote_localizes_page_when_cited_pages_empty():
    doc = PdfDocument.from_pages([
        "Page 1 intro text.",
        "Page 2 Consolidated Statement: Net Sales 1,200 and Operating Income 300.",
        "Page 3 signature page.",
    ])
    assert localize_quote_pages("Net Sales 1,200 and Operating Income 300", doc.pages) == [2]
    llm = _StubLLM({
        "evidence": [{"page": 0, "quote": "Net Sales 1,200 and Operating Income 300"}],
        "operands": [{"name": "margin", "value": "300/1200", "page": 0}],
        "formula": "300 / 1200 * 100",
        "reasoning": "Operating margin is 25%.",
        "can_answer": True,
        "unit": "%",
        "final_answer": "25%",
        "cited_pages": [],  # empty citation -> recovered deterministically from verified quote!
    })
    v = FactAdjudicator(llm).answer_numerical(doc, "What is operating margin?", [1, 2, 3], route="read_all")
    assert v.quote_verified is True
    assert v.meta["grounding_scope"] == "quote_localized"
    assert v.cited_pages == [2]
    assert v.meta["operand_grounded"] is True


def test_readall_uncited_and_unquoted_flags_none_scope():
    doc = PdfDocument.from_pages([
        "Page 1 data: 400 and 100.",
        "Page 2 other data: 999.",
    ])
    llm = _StubLLM({
        "evidence": [],
        "operands": [],
        "formula": "400 / 100",
        "reasoning": "Ratio is 4.",
        "can_answer": True,
        "unit": "ratio",
        "final_answer": "4",
        "cited_pages": [],
    })
    v = FactAdjudicator(llm).answer_numerical(doc, "What is the ratio?", [1, 2], route="read_all")
    assert v.meta["grounding_scope"] == "none"
    assert v.meta["operand_grounded"] is False
    assert "READALL_UNCITED" in v.rule_trace


def test_multimodal_readall_picks_lexically_top_page_for_zoom_not_cover():
    doc = PdfDocument.from_pages([
        "Quarterly Deck Cover Page",
        "Table of Contents",
        "Bar Chart: 2025 European Sales grew to 75 units across 5 regions",
        "Thank You Slide",
    ])
    tr = AnchorRetriever(doc, llm=None).retrieve("How many European Sales units in Bar Chart?", task="qa_multimodal")
    assert tr.route == "read_all"
    assert tr.pages[0] == 3
    zoom = FactAdjudicator._pick_zoom_page(doc, "How many European Sales units in Bar Chart?", tr.pages, {})
    assert zoom == 3 and zoom != 1


def test_prose_readall_passes_all_spans_and_keeps_decide_nli():
    nda = (
        "1. Confidential Information\n"
        "Receiving Party shall keep all Confidential Information strictly confidential.\n\n"
        "2. Return of Materials\n"
        "Receiving Party shall not retain any copies of Confidential Information after termination.\n"
    )
    sr = SpanRetriever(nda, llm=None)
    spans, followup = sr.retrieve("Receiving Party may retain some copies after termination.", route="auto")
    assert sr.last_route.route == "read_all"
    assert len(spans) == len(sr.index.spans)
    assert not followup

    llm = _StubLLM({
        "topic_clauses": ["s0002"],
        "verbatim_quote": "Receiving Party shall not retain any copies of Confidential Information after termination.",
        "analysis": "Clause 2 explicitly prohibits retaining any copies.",
        "hyp_quantifier": "existential",
        "stance": "opposes",
        "permitted_in_some_case": False,
        "counterexample": True,
    })
    v = FactAdjudicator(llm).audit_prose_clause(nda, "Receiving Party may retain some copies.", spans)
    assert v.answer == "contradiction"
    assert v.status == "fail"
    assert v.quote_verified is True
