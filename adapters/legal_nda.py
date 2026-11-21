"""Thin adapter for long legal contract compliance auditing (ContractNLI-Long)."""
from __future__ import annotations

from anchordoc.adjudicate import FactAdjudicator
from anchordoc.retrieve import SpanRetriever
from anchordoc.runtime import LLMClient
from anchordoc.types import AuditVerdict


class LegalNDAAdapter:
    """Verify compliance clauses across long prose legal contracts."""

    def __init__(self, contract_text: str, llm: LLMClient) -> None:
        self.text = contract_text
        self.retriever = SpanRetriever(contract_text, llm=llm)
        self.adjudicator = FactAdjudicator(llm=llm)

    def check_clause(
        self,
        hypothesis: str,
        item_id: str = "nda_1",
        *,
        route: str = "auto",
    ) -> AuditVerdict:
        spans, followup = self.retriever.retrieve(
            hypothesis, top_k=7, enable_followup=True, route=route  # type: ignore[arg-type]
        )
        verdict = self.adjudicator.audit_prose_clause(self.text, hypothesis, spans, item_id=item_id)
        verdict.retrieval_trace = {
            "spans": [sp.tag for sp in spans],
            "followup_triggered": followup,
            "route": self.retriever.last_route.route,
            "route_reason": self.retriever.last_route.reason,
            "est_tokens": self.retriever.last_route.est_tokens,
        }
        return verdict
