"""Selective VisionPageCard indexer: invokes VLM ONLY on low-text / image-only / scanned pages.

Design principle:
- Digital text-rich pages (>= min_text_chars) incur ZERO indexing LLM calls.
- Low-text / rasterized / scanned / chart-only pages (< min_text_chars) are rendered via
  Apache-2.0 pypdfium2 and summarized into structured VisionPageCards (cached via CachedLLM
  and optionally persisted in BundleStore so reloads incur 0 renders and 0 LLM calls).
- Summaries feed seamlessly into DocumentIndex (Dual BM25-RRF + Structural AnchorMap) and
  AnchorRetriever locator, making 100% scanned PDFs and slide decks fully searchable.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..docbase.pdf_loader import PdfDocument
from ..runtime.llm import LLMClient

VISION_SCHEMA_VERSION = "v1"
VISION_MAX_TOKENS = 360
VISION_CARD_SYSTEM = (
    "You are a precision document indexer reading a rasterized/scanned/chart PDF page. "
    "Extract searchable structured metadata including exact titles, chart/table labels, "
    "numbers, percentages, dates, and entities. Return JSON only."
)
VISION_CARD_USER_TEMPLATE = (
    "Analyze physical Page {page_num} of this document.\n"
    "Extract:\n"
    "1. page_type: 'chart' | 'figure' | 'table' | 'slide' | 'scan_doc' | 'cover' | 'toc'\n"
    "2. title: exact headline or chart/table title on the page\n"
    "3. summary: concise 2-3 sentence summary of key facts, trends, or clauses shown\n"
    "4. key_data: list of exact numbers, percentages, axis labels, dates, or named entities shown (up to 15)\n"
    'Return JSON: {"page_type":"slide","title":"","summary":"","key_data":["..."]}'
)


class SelectiveVisionIndexer:
    """Builds structured visual summaries strictly for low-text / scanned PDF pages."""

    def __init__(
        self,
        llm: LLMClient,
        min_text_chars: int = 180,
        render_scale: float = 1.15,
        max_workers: int = 6,
    ) -> None:
        self.llm = llm
        self.min_text_chars = min_text_chars
        self.render_scale = render_scale
        self.max_workers = max_workers

    def select_visual_pages(self, doc: PdfDocument) -> list[int]:
        """Return 1-based physical pages needing visual indexing (low extractable text)."""
        if doc.path is None or not doc.path.is_file():
            return []
        return [
            pno
            for pno, txt in enumerate(doc.pages, start=1)
            if len((txt or "").strip()) < self.min_text_chars
        ]

    def _summarize_one(self, doc: PdfDocument, page_num: int) -> tuple[int, str, dict[str, Any]]:
        b64 = doc.render_page_png_b64(page_num, scale=self.render_scale)
        usr_p = VISION_CARD_USER_TEMPLATE.replace("{page_num}", str(page_num))
        parsed, _ = self.llm.chat_json(
            "anchordoc_vision_card",
            VISION_CARD_SYSTEM,
            usr_p,
            max_tokens=VISION_MAX_TOKENS,
            images_b64=[b64],
        )
        ptype = str(parsed.get("page_type") or "visual").upper()
        title = str(parsed.get("title") or "").strip()
        summary = str(parsed.get("summary") or "").strip()
        kdata = parsed.get("key_data") if isinstance(parsed.get("key_data"), list) else []
        data_str = "; ".join(str(x) for x in kdata[:15] if str(x).strip())
        card_str = f"[VISION:{ptype}] {title} | {summary}" + (f" | Data: {data_str}" if data_str else "")
        return page_num, card_str.strip(), parsed

    def index_document(
        self,
        doc: PdfDocument,
        store: Any = None,
    ) -> tuple[dict[int, str], dict[str, Any]]:
        """Build vision summaries only for low-text pages. Returns (summaries_by_page, stats).

        When ``store`` (a ``BundleStore``) is provided, pages already persisted under the active
        ``vision_fingerprint`` are loaded directly with 0 rendering and 0 LLM calls.
        """
        targets = self.select_visual_pages(doc)
        if not targets:
            return {}, {
                "total_pages": doc.num_pages,
                "vision_indexed_pages": 0,
                "from_store": 0,
                "newly_indexed": 0,
            }

        summaries: dict[int, str] = {}
        doc_sha = doc.sha256
        v_fp = ""
        if store is not None:
            from ..store.bundle import file_sha256, vision_fingerprint

            if not doc_sha and doc.path is not None and doc.path.is_file():
                doc_sha = file_sha256(doc.path)
                doc.sha256 = doc_sha
            if doc_sha:
                v_fp = vision_fingerprint(
                    model=str(getattr(self.llm, "model", "unknown")),
                    prompt_template=VISION_CARD_SYSTEM + "\n" + VISION_CARD_USER_TEMPLATE,
                    render_scale=self.render_scale,
                    max_tokens=VISION_MAX_TOKENS,
                    schema_version=VISION_SCHEMA_VERSION,
                )
                cached = store.get_vision_cards(doc_sha, v_fp)
                for p in targets:
                    if p in cached:
                        summaries[p] = cached[p]

        from_store = len(summaries)
        missing = [p for p in targets if p not in summaries]
        newly_indexed = 0

        if missing:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(missing))) as ex:
                for pno, card_str, parsed in ex.map(lambda p: self._summarize_one(doc, p), missing):
                    if card_str:
                        summaries[pno] = card_str
                        newly_indexed += 1
                        if store is not None and doc_sha and v_fp:
                            store.put_vision_card(doc_sha, v_fp, pno, card_str, raw=parsed)

        return summaries, {
            "total_pages": doc.num_pages,
            "vision_indexed_pages": len(summaries),
            "from_store": from_store,
            "newly_indexed": newly_indexed,
        }
