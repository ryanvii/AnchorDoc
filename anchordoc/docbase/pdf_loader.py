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
    # Printed page labels from the PDF /PageLabels tree (index i -> label of physical page i+1).
    # Empty list when the PDF defines no labels or the backend does not expose them.
    page_labels: list[str] = field(default_factory=list)
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
    def from_pages(
        cls,
        pages: list[str],
        path: str | Path | None = None,
        page_labels: list[str] | None = None,
    ) -> "PdfDocument":
        """Construct a document directly from in-memory 1-based page texts (useful for tests/DOCX)."""
        return cls(
            path=Path(path) if path else None,
            pages=list(pages),
            backend="memory",
            page_labels=_clean_labels(list(page_labels or []), len(pages)),
        )

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
            labels: list[str] = []
            try:
                labels = [str(doc[i].get_label() or "") for i in range(len(doc))]
            except Exception:
                labels = []
            doc.close()
            return cls(path=p, pages=pages, toc_bookmarks=toc, backend="pymupdf",
                       page_labels=_clean_labels(labels, len(pages)))

        import pypdfium2 as pdfium

        with _PDFIUM_LOCK:
            pdf = pdfium.PdfDocument(str(p))
            pages: list[str] = []
            labels: list[str] = []
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
                # /PageLabels is optional; older pypdfium2 builds may lack get_page_label.
                getter = getattr(pdf, "get_page_label", None)
                if callable(getter):
                    for i in range(len(pdf)):
                        try:
                            labels.append(str(getter(i) or ""))
                        except Exception:
                            labels.append("")
            finally:
                pdf.close()
        return cls(path=p, pages=pages, toc_bookmarks=[], backend="pypdfium2",
                   page_labels=_clean_labels(labels, len(pages)))

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

    def render_page_tiles_b64(
        self, page_num: int, grid: int = 2, overlap: float = 0.1, scale: float = 2.0
    ) -> list[str]:
        """Render a 1-based page at high scale and cut it into grid x grid overlapping PNG tiles.

        Tiles are returned row-major (top-left, top-right, bottom-left, bottom-right for grid=2).
        Each tile is extended by ``overlap`` x tile-size on every inner edge so that legends,
        axis labels or table cells that straddle a cut line stay readable in at least one tile.
        """
        if self.path is None or not self.path.is_file():
            raise RuntimeError("Cannot render page tiles without an underlying PDF file.")
        if page_num < 1 or page_num > self.num_pages:
            raise ValueError(f"Page {page_num} out of range 1..{self.num_pages}")
        grid = max(1, int(grid))
        overlap = min(max(float(overlap), 0.0), 0.45)

        with self._lock:
            pil_img = self._render_pil_unlocked(page_num, scale)
        return _tile_image_b64(pil_img, grid, overlap)

    def _render_pil_unlocked(self, page_num: int, scale: float):
        """Render to a PIL image. Caller must hold ``self._lock``."""
        if self.backend == "pymupdf":
            import pymupdf
            from PIL import Image

            doc = pymupdf.open(self.path)
            try:
                raw = doc[page_num - 1].get_pixmap(matrix=pymupdf.Matrix(scale, scale)).tobytes("png")
            finally:
                doc.close()
            img = Image.open(io.BytesIO(raw))
            img.load()
            return img

        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(self.path))
        try:
            page = pdf[page_num - 1]
            img = page.render(scale=scale).to_pil()
            page.close()
            return img
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


def _clean_labels(labels: list[str], n_pages: int) -> list[str]:
    """Keep page labels only if they are complete and carry information.

    PDFs without a /PageLabels tree yield all-empty labels; we normalise that to ``[]`` so
    downstream code can simply test ``if doc.page_labels``.
    """
    if len(labels) != n_pages or not any(lb.strip() for lb in labels):
        return []
    return [lb.strip() for lb in labels]


def _tile_image_b64(pil_img, grid: int, overlap: float) -> list[str]:
    """Crop a PIL image into grid x grid overlapping tiles and return PNG base64 strings."""
    w, h = pil_img.size
    tw, th = w / grid, h / grid
    ox, oy = tw * overlap, th * overlap
    out: list[str] = []
    for r in range(grid):
        for c in range(grid):
            x0 = max(0, int(c * tw - ox))
            y0 = max(0, int(r * th - oy))
            x1 = min(w, int((c + 1) * tw + ox))
            y1 = min(h, int((r + 1) * th + oy))
            buf = io.BytesIO()
            pil_img.crop((x0, y0, x1, y1)).save(buf, format="PNG")
            out.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    return out
