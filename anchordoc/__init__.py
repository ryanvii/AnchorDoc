"""AnchorDoc: Zero-index-cost, vectorless verifiable document auditing engine."""
from .adjudicate import FactAdjudicator, compare_duration_or_count, decide_compliance, safe_calc
from .docbase import PdfDocument, extract_docx_tables, resolve_table_rows
from .extract import ContractExtractor
from .index import DocumentIndex, SelectiveVisionIndexer, SpanCardIndex
from .report import export_html, export_json, export_markdown
from .retrieve import AnchorRetriever, SpanRetriever
from .runtime import BM25Okapi, CachedLLM, rrf, tokenize, verify_quote
from .types import AuditVerdict, ExtractedRequirement, PageCard, RetrievalTrace, SourceLine, SpanCard

__version__ = "0.5.0"

__all__ = [
    "__version__",
    "CachedLLM",
    "PdfDocument",
    "DocumentIndex",
    "SelectiveVisionIndexer",
    "SpanCardIndex",
    "AnchorRetriever",
    "SpanRetriever",
    "ContractExtractor",
    "FactAdjudicator",
    "decide_compliance",
    "safe_calc",
    "compare_duration_or_count",
    "extract_docx_tables",
    "resolve_table_rows",
    "BM25Okapi",
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
