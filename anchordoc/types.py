"""Core domain-agnostic data structures for AnchorDoc."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SourceLine:
    """A numbered line or intact table row [s00001] used for quote-anchored extraction."""
    sid: str
    page: int
    section: str
    text: str
    kind: str = "paragraph"  # paragraph | table_row


@dataclass(frozen=True)
class PageCard:
    """Lightweight Zero-LLM structural metadata card for a 1-based physical page."""
    page: int
    section_path: str
    heading: str
    is_toc: bool
    has_table: bool
    table_cues: tuple[str, ...] = ()
    years: tuple[str, ...] = ()
    text_head: str = ""
    vision_summary: str = ""

    def card_text(self) -> str:
        parts = [
            (self.section_path + " ") * 2 if self.section_path else "",
            (self.heading + " ") * 2 if self.heading else "",
            (" ".join(self.table_cues) + " ") * 2 if self.table_cues else "",
            " ".join(self.years),
            "structured_table numeric_table" if self.has_table else "",
            self.vision_summary,
            self.text_head,
        ]
        return " ".join(p.strip() for p in parts if p and p.strip())


@dataclass(frozen=True)
class SpanCard:
    """Structure-aware numbered clause/paragraph card for long prose/contract documents."""
    sid: int
    section: str
    text: str

    @property
    def tag(self) -> str:
        return f"s{self.sid:04d}"

    def card_text(self) -> str:
        return f"{self.section} {self.section} {self.text}" if self.section else self.text


@dataclass
class RetrievalTrace:
    """Layer-by-layer audit trail explaining why each page/span was selected."""
    query: str
    pages: list[int] = field(default_factory=list)
    selected_c3: list[int] = field(default_factory=list)
    anchor_direct: list[int] = field(default_factory=list)
    g7_toc_pages: list[int] = field(default_factory=list)
    g7_offset: int = 0
    followup_triggered: bool = False
    expanded_keywords: list[str] = field(default_factory=list)
    layers: dict[str, list[int]] = field(default_factory=dict)
    route: str = "navigate"
    route_reason: str = ""
    est_tokens: int = 0


@dataclass
class ExtractedRequirement:
    """A quote-anchored, verifiable requirement extracted from a specification/tender document."""
    item_id: str
    category: str
    title: str
    requirement: str
    consequence_type: str = "unspecified"  # mandatory | scored | informational | unspecified
    scope: str = ""
    sources: list[dict[str, str]] = field(default_factory=list)  # [{"sid": "s0001", "quote": "..."}]
    quote_verified: bool = False
    member_cids: list[str] = field(default_factory=list)


@dataclass
class AuditVerdict:
    """Final auditable verdict for one compliance item or verifiable query."""
    item_id: str
    title: str
    status: str  # pass | fail | needs_review | answered | abstain
    answer: str = ""
    reason: str = ""
    cited_pages: list[int] = field(default_factory=list)
    verbatim_quote: str = ""
    quote_verified: bool = False
    rule_trace: str = ""
    formula: str = ""
    calc_value: float | None = None
    facts: dict[str, Any] = field(default_factory=dict)
    retrieval_trace: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)  # structured audit fields (v0.6: provenance, contract, grounding)
