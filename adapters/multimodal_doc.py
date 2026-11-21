"""Thin adapter for multimodal long PDF documents (charts, figures, scans, unanswerable checks)."""
from __future__ import annotations

from dataclasses import asdict

from anchordoc.adjudicate import FactAdjudicator
from anchordoc.docbase import PdfDocument
from anchordoc.retrieve import AnchorRetriever
from anchordoc.runtime import LLMClient
from anchordoc.types import AuditVerdict


class MultimodalDocAdapter:
    """Retrieve candidate pages via AnchorDoc and verify charts/figures directly on rendered PNGs."""

    def __init__(self, doc: PdfDocument, llm: LLMClient) -> None:
        self.doc = doc
        self.retriever = AnchorRetriever(doc, llm=llm)
        self.adjudicator = FactAdjudicator(llm=llm)

    def query(
        self,
        question: str,
        item_id: str = "mm_1",
        top_k: int = 5,
        use_vision: bool = True,
    ) -> AuditVerdict:
        trace = self.retriever.retrieve(question, top_k=top_k, use_anchor_map=True, use_gap_followup=True)
        if use_vision:
            verdict = self.adjudicator.answer_multimodal(self.doc, question, trace.pages, item_id=item_id)
        else:
            verdict = self.adjudicator.answer_numerical(self.doc, question, trace.pages, item_id=item_id)
        verdict.retrieval_trace = asdict(trace)
        return verdict
