"""AnchorDoc: Zero-index-cost, vectorless verifiable document auditing engine."""
from .adjudicate import FactAdjudicator, compare_duration_or_count, decide_compliance, safe_calc
from .context import DocContext
from .docbase import PdfDocument, extract_docx_tables, resolve_table_rows
from .extract import ContractExtractor
from .index import DocumentIndex, SelectiveVisionIndexer, SpanCardIndex
from .report import export_html, export_json, export_markdown
from .retrieve import AnchorRetriever, SpanRetriever, plan_prose_route, plan_route, profile_document
from .runtime import BM25Okapi, CachedLLM, estimate_tokens, localize_quote_pages, rrf, tokenize, verify_quote
from .store import BundleStore, load_document
from .types import AuditVerdict, ExtractedRequirement, PageCard, RetrievalTrace, SourceLine, SpanCard

__version__ = "0.7.0"

__all__ = [
    "__version__",
    "CachedLLM",
    "PdfDocument",
    "BundleStore",
    "load_document",
    "DocContext",
    "DocumentIndex",
    "SelectiveVisionIndexer",
    "SpanCardIndex",
    "AnchorRetriever",
    "SpanRetriever",
    "profile_document",
    "plan_route",
    "plan_prose_route",
    "ContractExtractor",
    "FactAdjudicator",
    "decide_compliance",
    "safe_calc",
    "compare_duration_or_count",
    "extract_docx_tables",
    "resolve_table_rows",
    "BM25Okapi",
    "estimate_tokens",
    "localize_quote_pages",
    "rrf",
    "tokenize",
    "verify_quote",
    "export_json",
    "export_markdown",
    "export_html",
    "AuditVerdict",
    "ExtractedRequirement",
    "PageCard",
    "RetrievalTrace",
    "SourceLine",
    "SpanCard",
]
