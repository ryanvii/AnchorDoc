"""Apache-2.0 safe 1-based PageModel PDF loader (pypdfium2 primary, optional PyMuPDF)."""
from __future__ import annotations

import base64
import io
import threading
from dataclasses import dataclass, field
from pathlib import Path

from ..types import SourceLine

# PDFium C++ library is not thread-safe across concurrent document loads
_PDFIUM_LOCK = threading.Lock()


@dataclass
class PdfDocument:
    """Unified 1-based physical page model for long PDF documents."""
    path: Path | None
    pages: list[str]
    toc_bookmarks: list[tuple[int, str, int]] = field(default_factory=list)
    backend: str = "pypdfium2"
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    @property
    def num_pages(self) -> int:
        return len(self.pages)

    def page_text(self, page_num: int) -> str:
        """Return text of 1-based physical page."""
        if page_num < 1 or page_num > len(self.pages):
            return ""
        return self.pages[page_num - 1]

    @classmethod
    def from_pages(cls, pages: list[str], path: str | Path | None = None) -> "PdfDocument":
        """Construct a document directly from in-memory 1-based page texts (useful for tests/DOCX)."""
        return cls(path=Path(path) if path else None, pages=list(pages), backend="memory")

    @classmethod
    def from_pdf(cls, pdf_path: str | Path, backend: str = "pypdfium2") -> "PdfDocument":
        """Load 1-based page texts from PDF using Apache-2.0 pypdfium2 (default) or optional pymupdf."""
        p = Path(pdf_path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"PDF not found: {p}")

        if backend == "pymupdf":
            import pymupdf

            doc = pymupdf.open(p)
            pages = [doc[i].get_text("text") or "" for i in range(len(doc))]
            toc = [(int(lvl), str(title).strip(), int(pg)) for lvl, title, pg in (doc.get_toc() or [])]
            doc.close()
            return cls(path=p, pages=pages, toc_bookmarks=toc, backend="pymupdf")

        import pypdfium2 as pdfium

        with _PDFIUM_LOCK:
            pdf = pdfium.PdfDocument(str(p))
            pages: list[str] = []
            try:
                for i in range(len(pdf)):
                    try:
                        page = pdf[i]
                        tp = page.get_textpage()
                        pages.append(tp.get_text_bounded() or "")
                        tp.close()
                        page.close()
                    except Exception:
                        pages.append("")
            finally:
                pdf.close()
        return cls(path=p, pages=pages, toc_bookmarks=[], backend="pypdfium2")

    def render_page_png_b64(self, page_num: int, scale: float = 1.5) -> str:
        """Render a 1-based physical PDF page to base64 PNG for vision verification."""
        if self.path is None or not self.path.is_file():
            raise RuntimeError("Cannot render page image without an underlying PDF file.")
        if page_num < 1 or page_num > self.num_pages:
            raise ValueError(f"Page {page_num} out of range 1..{self.num_pages}")

        with self._lock:
            if self.backend == "pymupdf":
                import pymupdf

                doc = pymupdf.open(self.path)
                raw = doc[page_num - 1].get_pixmap(matrix=pymupdf.Matrix(scale, scale)).tobytes("png")
                doc.close()
                return base64.b64encode(raw).decode("ascii")

            import pypdfium2 as pdfium

            pdf = pdfium.PdfDocument(str(self.path))
            try:
                page = pdf[page_num - 1]
                pil_img = page.render(scale=scale).to_pil()
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                page.close()
                return base64.b64encode(buf.getvalue()).decode("ascii")
            finally:
                pdf.close()

    def to_source_lines(self) -> list[SourceLine]:
        """Convert 1-based pages into globally numbered [s00001] lines for quote-anchored extraction."""
        out: list[SourceLine] = []
        sid = 0
        for pno, text in enumerate(self.pages, start=1):
            for raw_ln in (text or "").splitlines():
                ln = raw_ln.strip()
                if not ln:
                    continue
                sid += 1
                out.append(SourceLine(sid=f"s{sid:05d}", page=pno, section="", text=ln))
        return out
