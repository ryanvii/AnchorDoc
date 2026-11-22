"""Thin adapter for multimodal long PDF documents (Selective VisionPageCard + VLM verification)."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from anchordoc.adjudicate import FactAdjudicator
from anchordoc.docbase import PdfDocument
from anchordoc.index import SelectiveVisionIndexer
from anchordoc.retrieve import AnchorRetriever
from anchordoc.runtime import LLMClient
from anchordoc.types import AuditVerdict


class MultimodalDocAdapter:
    """Hybrid Zero-LLM + Selective VisionPageCard retriever with rendered-page VLM verification.

    - Text-rich digital pages incur 0 indexing LLM calls.
    - Low-text / rasterized / chart-only pages (< min_text_chars) automatically receive a cached
      VisionPageCard so 100% scanned slide decks & certificates become fully searchable.
    """

    def __init__(
        self,
        doc: PdfDocument,
        llm: LLMClient,
        auto_vision_index: bool = True,
        min_text_chars: int = 180,
        store: Any = None,
    ) -> None:
        self.doc = doc
        self.llm = llm
        self.vision_index_stats: dict[str, Any] = {"total_pages": doc.num_pages, "vision_indexed_pages": 0}
        vision_summaries: dict[int, str] = {}
        if auto_vision_index:
            indexer = SelectiveVisionIndexer(llm=llm, min_text_chars=min_text_chars)
            vision_summaries, self.vision_index_stats = indexer.index_document(doc, store=store)
        self.retriever = AnchorRetriever(doc, llm=llm, vision_summaries=vision_summaries)
        self.adjudicator = FactAdjudicator(llm=llm)

    def query(
        self,
        question: str,
        item_id: str = "mm_1",
        top_k: int = 5,
        use_vision: bool = True,
        *,
        route: str = "auto",
    ) -> AuditVerdict:
        task = "qa_multimodal" if use_vision else "qa_text"
        trace = self.retriever.retrieve(
            question, top_k=top_k, use_anchor_map=True, use_gap_followup=True, route=route, task=task  # type: ignore[arg-type]
        )
        if use_vision:
            loc_suf_raw = trace.layers.get("locate_sufficient")
            loc_suf = bool(loc_suf_raw[0]) if isinstance(loc_suf_raw, list) and loc_suf_raw else None
            verdict = self.adjudicator.answer_multimodal(
                self.doc,
                question,
                trace.pages,
                item_id=item_id,
                vision_summaries=self.retriever.index.vision_summaries,
                route=trace.route,  # type: ignore[arg-type]
                locate_sufficient=loc_suf,
                b42_selected=trace.layers.get("b42_selected"),
            )
        else:
            verdict = self.adjudicator.answer_numerical(
                self.doc, question, trace.pages, item_id=item_id, route=trace.route  # type: ignore[arg-type]
            )
        verdict.retrieval_trace = asdict(trace)
        return verdict
