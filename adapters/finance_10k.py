"""Thin adapter for SEC 10-K / financial filing auditing (FinanceBench)."""
from __future__ import annotations

from dataclasses import asdict

from anchordoc.adjudicate import FactAdjudicator
from anchordoc.docbase import PdfDocument
from anchordoc.retrieve import AnchorRetriever
from anchordoc.runtime import LLMClient
from anchordoc.types import AuditVerdict


class Finance10KAdapter:
    """End-to-end verifiable financial filing auditor over AnchorDoc core."""

    def __init__(self, doc: PdfDocument, llm: LLMClient) -> None:
        self.doc = doc
        self.retriever = AnchorRetriever(doc, llm=llm)
        self.adjudicator = FactAdjudicator(llm=llm)

    def query(self, question: str, item_id: str = "fin_1", top_k: int = 5) -> AuditVerdict:
        trace = self.retriever.retrieve(question, top_k=top_k, use_anchor_map=True, use_gap_followup=True)
        verdict = self.adjudicator.answer_numerical(self.doc, question, trace.pages, item_id=item_id)
        verdict.retrieval_trace = asdict(trace)
        return verdict
