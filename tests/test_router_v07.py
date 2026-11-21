"""Unit tests for v0.7.0 three-tier adaptive router (`read_all` | `full_pool` | `navigate`)."""
from __future__ import annotations

from anchordoc.docbase import PdfDocument
from anchordoc.retrieve import (
    AnchorRetriever,
    DocProfile,
    SpanRetriever,
    plan_prose_route,
    plan_route,
    profile_document,
)
from anchordoc.runtime import estimate_tokens


class _RecordingLLM:
    model = "fake-router-model"

    def __init__(self, responses: dict | None = None) -> None:
        self.responses = responses or {}
        self.calls: list[tuple[str, str]] = []

    def chat_json(self, purpose: str, system: str, user: str, max_tokens: int = 700, images_b64=None):
        self.calls.append((purpose, user))
        return dict(self.responses.get(purpose, {})), "{}"

    def snapshot_stats(self):
        return {"calls": len(self.calls)}


def test_prose_route_english_and_cjk_boundaries():
    short_en = "Confidential information shall not be disclosed. " * 250  # ~12k chars (~3k tok)
    long_en = "Confidential information shall not be disclosed. " * 900   # ~43k chars (~10.7k tok)
    assert plan_prose_route(short_en).route == "read_all"
    assert plan_prose_route(long_en).route == "navigate"

    # 5,000+ CJK chars must estimate >= 5,000 tokens -> must NOT take read_all (<= 4,000)
    cjk_text = "本协议保密条款约定双方不得向第三方泄露技术资料。" * 220  # 5,280 CJK chars
    assert estimate_tokens(cjk_text) >= 5000
    assert plan_prose_route(cjk_text).route == "navigate"


def test_multimodal_and_text_route_thresholds():
    # Multimodal: <=5 -> read_all, 6..40 -> full_pool, >=41 -> navigate
    assert plan_route(DocProfile(5, 1000, 5), task="qa_multimodal").route == "read_all"
    p6 = plan_route(DocProfile(6, 1000, 6), task="qa_multimodal")
    assert p6.route == "full_pool" and p6.snippet_chars == 900
    p20 = plan_route(DocProfile(20, 4000, 20), task="qa_multimodal")
    assert p20.route == "full_pool" and p20.snippet_chars == 675
    p40 = plan_route(DocProfile(40, 8000, 40), task="qa_multimodal")
    assert p40.route == "full_pool" and p40.snippet_chars == 337
    assert plan_route(DocProfile(41, 8000, 41), task="qa_multimodal").route == "navigate"

    # Text QA: visual pages forbid text read_all even when est_text_tokens is tiny
    assert plan_route(DocProfile(4, 2000, 0), task="qa_text").route == "read_all"
    assert plan_route(DocProfile(4, 2000, 1), task="qa_text").route == "full_pool"
    assert plan_route(DocProfile(100, 60000, 0), task="qa_text").route == "navigate"


def test_full_pool_skips_nav_and_uses_computed_budget():
    pages = [f"Page {i} content with revenue {100 + i} million and margin {10 + i}% " + ("detail " * 260) for i in range(1, 21)]
    doc = PdfDocument.from_pages(pages)
    assert profile_document(doc).n_visual_pages == 0
    llm = _RecordingLLM({
        "anchordoc_qexp": {"keywords": ["revenue", "margin"], "section_hint": ""},
        "anchordoc_locate": {"ranked_pages": [7, 8], "sufficient": True, "missing_terms": []},
    })
    tr = AnchorRetriever(doc, llm=llm).retrieve("What is revenue on page 7?", top_k=5, route="auto", task="qa_text")
    assert tr.route == "full_pool"
    assert [p for p, _ in llm.calls] == ["anchordoc_qexp", "anchordoc_locate"]
    assert tr.layers["locate_pool"] == list(range(1, 21))
    assert tr.pages[:2] == [7, 8]


def test_read_all_makes_zero_retrieval_llm_calls_for_text_mm_and_prose():
    # Text read_all (4 text-rich pages, <= 6k tokens)
    pages = [f"Section {i}\n" + ("Financial summary clause " * 20) for i in range(1, 5)]
    llm = _RecordingLLM()
    tr = AnchorRetriever(PdfDocument.from_pages(pages), llm=llm).retrieve("Summary?", task="qa_text")
    assert tr.route == "read_all"
    assert tr.pages == [1, 2, 3, 4]
    assert llm.calls == []

    # Multimodal read_all (4 pages <= 5) orders by lexical RRF, not cover page first
    mm_pages = [
        "Annual Report Cover Slide",
        "Agenda Overview",
        "Chart 1: 2025 North America Sales Growth 42%",
        "Appendix References",
    ]
    tr_mm = AnchorRetriever(PdfDocument.from_pages(mm_pages), llm=llm).retrieve(
        "North America Sales Growth 42%", task="qa_multimodal"
    )
    assert tr_mm.route == "read_all"
    assert tr_mm.pages[0] == 3
    assert llm.calls == []

    # Prose read_all (short NDA <= 4k tokens) returns all spans with 0 LLM calls
    nda = "\n\n".join(f"{i}. Clause {i}\nReceiving Party shall protect confidential data item {i}." for i in range(1, 10))
    sr = SpanRetriever(nda, llm=llm)
    spans, followup = sr.retrieve("Receiving Party shall protect confidential data.")
    assert sr.last_route.route == "read_all"
    assert len(spans) == len(sr.index.spans)
    assert not followup
    assert llm.calls == []


def test_long_doc_100p_navigate_matches_v06_golden_prompts():
    pages = [f"ITEM {i} Section\nRevenue was {1000 + i} million in FY2024.\n" + ("text " * 120) for i in range(1, 101)]
    doc = PdfDocument.from_pages(pages)
    llm_auto = _RecordingLLM({
        "anchordoc_qexp": {"keywords": ["revenue"], "section_hint": "ITEM 50"},
        "anchordoc_nav": {"direct_pages": [50], "ranges": [[48, 52]]},
        "anchordoc_locate": {"ranked_pages": [50], "sufficient": True, "missing_terms": []},
    })
    llm_nav = _RecordingLLM(llm_auto.responses)
    tr_auto = AnchorRetriever(doc, llm=llm_auto).retrieve("FY2024 revenue", route="auto", task="qa_text")
    tr_nav = AnchorRetriever(doc, llm=llm_nav).retrieve("FY2024 revenue", route="navigate", task="qa_text")
    assert tr_auto.route == "navigate"
    assert llm_auto.calls == llm_nav.calls
    assert tr_auto.pages == tr_nav.pages
