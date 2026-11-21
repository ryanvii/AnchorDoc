"""Thin, 100% generic procurement compliance adapter (zero proprietary business terms)."""
from __future__ import annotations

from dataclasses import asdict

from anchordoc.adjudicate import FactAdjudicator
from anchordoc.docbase import PdfDocument
from anchordoc.extract import ContractExtractor
from anchordoc.retrieve import AnchorRetriever
from anchordoc.runtime import LLMClient
from anchordoc.types import AuditVerdict, ExtractedRequirement

_GENERIC_PROCUREMENT_GUIDANCE = (
    "Extract only verifiable qualification, mandatory compliance, and required independent proof items. "
    "Exclude ordinary technical parameter tables and general administrative boilerplate."
)


class ProcurementAuditAdapter:
    """Two-stage procurement audit: (1) Extract quote-anchored checklist, (2) Audit submission PDF."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        self.extractor = ContractExtractor(llm=llm)
        self.adjudicator = FactAdjudicator(llm=llm)

    def extract_checklist(self, spec_doc: PdfDocument) -> list[ExtractedRequirement]:
        lines = spec_doc.to_source_lines()
        return self.extractor.extract(lines, domain_guidance=_GENERIC_PROCUREMENT_GUIDANCE)

    def audit_submission(
        self,
        submission_doc: PdfDocument,
        checklist: list[ExtractedRequirement],
        top_k: int = 5,
    ) -> list[AuditVerdict]:
        retriever = AnchorRetriever(submission_doc, llm=self.llm)
        verdicts: list[AuditVerdict] = []
        for item in checklist:
            query = f"{item.title} {item.requirement}"
            trace = retriever.retrieve(query, top_k=top_k, use_anchor_map=True, use_gap_followup=True)
            verdict = self.adjudicator.audit_compliance_item(
                submission_doc,
                item_id=item.item_id,
                title=item.title,
                requirement=item.requirement,
                candidate_pages=trace.pages,
                mandatory=(item.consequence_type == "mandatory"),
            )
            verdict.retrieval_trace = asdict(trace)
            verdicts.append(verdict)
        return verdicts
