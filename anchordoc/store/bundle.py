"""Per-document SQLite WAL persistent store (`BundleStore`) for AnchorDoc.

Persists ONLY expensive or drift-sensitive document assets:
1. Frozen text layer + /PageLabels + PDF bookmarks (guarantees stable line anchors across
   library upgrades and avoids re-parsing 300-page PDFs on repeated sessions).
2. Selective VisionPageCards (eliminates expensive VLM calls & page rendering on reloads).

Zero external DB dependencies (pure Python stdlib `sqlite3` in WAL mode). Fast, deterministic
assets (PageCard, BM25, AnchorMap, G7 TOC offset) are rebuilt in milliseconds in memory.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..docbase.pdf_loader import PdfDocument, _clean_labels

BUNDLE_SCHEMA_VERSION = "1"
TEXT_SCHEMA_VERSION = "v1"

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    k TEXT PRIMARY KEY,
    v TEXT
);
CREATE TABLE IF NOT EXISTS text_layer (
    extractor_fp TEXT NOT NULL,
    page INT NOT NULL,
    text TEXT NOT NULL,
    label TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (extractor_fp, page)
);
CREATE TABLE IF NOT EXISTS bookmarks (
    extractor_fp TEXT NOT NULL,
    ord INT NOT NULL,
    level INT NOT NULL,
    title TEXT NOT NULL,
    page INT NOT NULL,
    PRIMARY KEY (extractor_fp, ord)
);
CREATE TABLE IF NOT EXISTS vision_cards (
    vision_fp TEXT NOT NULL,
    page INT NOT NULL,
    card TEXT NOT NULL,
    raw_json TEXT NOT NULL DEFAULT '{}',
    prompt_tokens INT NOT NULL DEFAULT 0,
    completion_tokens INT NOT NULL DEFAULT 0,
    PRIMARY KEY (vision_fp, page)
);
"""


@dataclass(frozen=True)
class TextLayer:
    """Persisted 1-based text pages, page labels, and PDF outline bookmarks."""

    pages: list[str]
    page_labels: list[str] = field(default_factory=list)
    toc_bookmarks: list[tuple[int, str, int]] = field(default_factory=list)


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    """Stream-hash a file's exact bytes into a lowercase SHA256 hex digest."""
    h = hashlib.sha256()
    with open(Path(path), "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _backend_version(backend: str) -> str:
    try:
        if backend == "pymupdf":
            import pymupdf

            return str(getattr(pymupdf, "__version__", "unknown"))
        import pypdfium2

        return str(getattr(pypdfium2, "__version__", getattr(pypdfium2, "V_LIBPDFIUM", "unknown")))
    except Exception:
        return "unknown"


def extractor_fingerprint(backend: str) -> str:
    """Fingerprint of PDF text extractor + library version + text schema."""
    payload = f"{backend}|{_backend_version(backend)}|{TEXT_SCHEMA_VERSION}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def vision_fingerprint(
    model: str,
    prompt_template: str,
    render_scale: float,
    max_tokens: int,
    schema_version: str = "v1",
) -> str:
    """Fingerprint of VisionPageCard generation settings (deliberately excludes min_text_chars)."""
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt_template,
            "render_scale": round(float(render_scale), 4),
            "max_tokens": int(max_tokens),
            "schema": schema_version,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


_CORRUPT_MARKERS = (
    "file is not a database",
    "database disk image is malformed",
    "malformed",
    "corrupt",
    "not a database",
)


def _is_corruption_error(exc: Exception) -> bool:
    """True only for genuine SQLite file corruption (never transient OperationalError like locks/EMFILE/NFS)."""
    msg = str(exc).lower()
    if any(m in msg for m in _CORRUPT_MARKERS):
        return True
    # Plain DatabaseError (not OperationalError subclass) indicates structural corruption
    return isinstance(exc, sqlite3.DatabaseError) and not isinstance(exc, sqlite3.OperationalError)


class BundleStore:
    """Thread-safe per-document SQLite WAL store with layered fingerprint invalidation."""

    def __init__(self, root: str | Path | None = None) -> None:
        env_root = os.environ.get("ANCHORDOC_STORE", "").strip()
        base = Path(root) if root is not None else (Path(env_root) if env_root else Path.home() / ".cache/anchordoc/bundles")
        self.root = base.expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self._locks_guard = threading.Lock()
        self._locks: dict[str, threading.Lock] = {}

    def _lock_for(self, doc_sha: str) -> threading.Lock:
        with self._locks_guard:
            lk = self._locks.get(doc_sha)
            if lk is None:
                lk = threading.Lock()
                self._locks[doc_sha] = lk
            return lk

    def bundle_path(self, doc_sha: str) -> Path:
        prefix = (doc_sha or "00")[:2]
        return self.root / prefix / f"{doc_sha}.sqlite"

    def _init_conn(self, db_path: Path) -> sqlite3.Connection:
        conn = sqlite3.connect(str(db_path), timeout=15.0)
        try:
            # WAL may fail on read-only mounts or NFS; fall back gracefully without quarantining.
            try:
                conn.execute("PRAGMA journal_mode=WAL;")
            except sqlite3.OperationalError:
                pass
            conn.execute("PRAGMA busy_timeout=10000;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.executescript(_SCHEMA_SQL)
            return conn
        except Exception:
            conn.close()
            raise

    def _open_conn(self, db_path: Path) -> sqlite3.Connection:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            return self._init_conn(db_path)
        except sqlite3.DatabaseError as exc:
            if not _is_corruption_error(exc):
                raise
            self._quarantine_corrupt(db_path)
            return self._init_conn(db_path)

    @staticmethod
    def _quarantine_corrupt(db_path: Path) -> None:
        """Rename a verified-corrupt main SQLite file to `.corrupt` (never deletes `-wal` blindly)."""
        corrupt = db_path.with_suffix(".sqlite.corrupt")
        try:
            if corrupt.exists():
                corrupt.unlink()
            if db_path.exists():
                db_path.rename(corrupt)
        except OSError:
            pass

    def load_text(self, doc_sha: str, extractor_fp: str) -> TextLayer | None:
        """Load persisted pages/labels/bookmarks for (doc_sha, extractor_fp), or None on miss."""
        db_path = self.bundle_path(doc_sha)
        if not db_path.is_file():
            return None
        with self._lock_for(doc_sha):
            try:
                conn = self._open_conn(db_path)
                try:
                    rows = conn.execute(
                        "SELECT page, text, label FROM text_layer WHERE extractor_fp = ? ORDER BY page ASC",
                        (extractor_fp,),
                    ).fetchall()
                    if not rows:
                        return None
                    pages = [str(r[1]) for r in rows]
                    labels = _clean_labels([str(r[2] or "") for r in rows], len(pages))
                    bm_rows = conn.execute(
                        "SELECT level, title, page FROM bookmarks WHERE extractor_fp = ? ORDER BY ord ASC",
                        (extractor_fp,),
                    ).fetchall()
                    bookmarks = [(int(r[0]), str(r[1]), int(r[2])) for r in bm_rows]
                    return TextLayer(pages=pages, page_labels=labels, toc_bookmarks=bookmarks)
                finally:
                    conn.close()
            except sqlite3.DatabaseError as exc:
                if _is_corruption_error(exc):
                    self._quarantine_corrupt(db_path)
                    return None
                raise

    def save_text(self, doc_sha: str, extractor_fp: str, layer: TextLayer, meta: dict[str, Any] | None = None) -> None:
        """Atomically persist extracted text layer, labels, bookmarks, and metadata."""
        db_path = self.bundle_path(doc_sha)
        labels = layer.page_labels if len(layer.page_labels) == len(layer.pages) else [""] * len(layer.pages)
        meta_items = {
            "doc_sha256": doc_sha,
            "n_pages": str(len(layer.pages)),
            "bundle_schema": BUNDLE_SCHEMA_VERSION,
            "updated_at": str(int(time.time())),
        }
        for k, v in (meta or {}).items():
            meta_items[str(k)] = str(v)

        with self._lock_for(doc_sha):
            conn = self._open_conn(db_path)
            try:
                with conn:
                    conn.executemany(
                        "INSERT OR REPLACE INTO meta(k, v) VALUES(?, ?)",
                        list(meta_items.items()),
                    )
                    conn.executemany(
                        "INSERT OR REPLACE INTO text_layer(extractor_fp, page, text, label) VALUES(?, ?, ?, ?)",
                        [(extractor_fp, i, txt, labels[i - 1]) for i, txt in enumerate(layer.pages, start=1)],
                    )
                    conn.execute("DELETE FROM bookmarks WHERE extractor_fp = ?", (extractor_fp,))
                    if layer.toc_bookmarks:
                        conn.executemany(
                            "INSERT OR REPLACE INTO bookmarks(extractor_fp, ord, level, title, page) VALUES(?, ?, ?, ?, ?)",
                            [(extractor_fp, idx, int(lv), str(tt), int(pg)) for idx, (lv, tt, pg) in enumerate(layer.toc_bookmarks)],
                        )
            finally:
                conn.close()

    def get_vision_cards(self, doc_sha: str, vision_fp: str) -> dict[int, str]:
        """Return all persisted VisionPageCard summaries for (doc_sha, vision_fp)."""
        db_path = self.bundle_path(doc_sha)
        if not db_path.is_file():
            return {}
        with self._lock_for(doc_sha):
            try:
                conn = self._open_conn(db_path)
                try:
                    rows = conn.execute(
                        "SELECT page, card FROM vision_cards WHERE vision_fp = ? ORDER BY page ASC",
                        (vision_fp,),
                    ).fetchall()
                    return {int(r[0]): str(r[1]) for r in rows if str(r[1] or "").strip()}
                finally:
                    conn.close()
            except sqlite3.DatabaseError as exc:
                if _is_corruption_error(exc):
                    self._quarantine_corrupt(db_path)
                    return {}
                raise

    def put_vision_card(
        self,
        doc_sha: str,
        vision_fp: str,
        page: int,
        card: str,
        raw: dict[str, Any] | None = None,
        usage: dict[str, Any] | None = None,
    ) -> None:
        """Persist a single VisionPageCard atomically (safe under concurrent threads)."""
        db_path = self.bundle_path(doc_sha)
        u = usage or {}
        pt = int(u.get("prompt_tokens") or 0)
        ct = int(u.get("completion_tokens") or 0)
        raw_s = json.dumps(raw or {}, ensure_ascii=False)
        with self._lock_for(doc_sha):
            conn = self._open_conn(db_path)
            try:
                with conn:
                    conn.execute(
                        "INSERT OR REPLACE INTO meta(k, v) VALUES(?, ?)",
                        ("doc_sha256", doc_sha),
                    )
                    conn.execute(
                        "INSERT OR REPLACE INTO vision_cards(vision_fp, page, card, raw_json, prompt_tokens, completion_tokens) "
                        "VALUES(?, ?, ?, ?, ?, ?)",
                        (vision_fp, int(page), card, raw_s, pt, ct),
                    )
            finally:
                conn.close()

    def info(self) -> dict[str, Any]:
        """Return summary statistics of all bundles currently stored under ``self.root``."""
        files = sorted(self.root.glob("*/*.sqlite"))
        total_bytes = sum(p.stat().st_size for p in files if p.is_file())
        bundles: list[dict[str, Any]] = []
        for p in files:
            try:
                conn = sqlite3.connect(str(p), timeout=5.0)
                try:
                    meta = dict(conn.execute("SELECT k, v FROM meta").fetchall())
                    n_text = conn.execute("SELECT COUNT(DISTINCT page) FROM text_layer").fetchone()[0]
                    n_vis = conn.execute("SELECT COUNT(*) FROM vision_cards").fetchone()[0]
                finally:
                    conn.close()
                bundles.append({
                    "sha": p.stem,
                    "bytes": p.stat().st_size,
                    "path_hint": meta.get("path_hint", ""),
                    "n_pages": int(meta.get("n_pages") or n_text or 0),
                    "vision_cards": int(n_vis),
                })
            except Exception:
                bundles.append({"sha": p.stem, "bytes": p.stat().st_size, "corrupt": True})
        return {
            "root": str(self.root),
            "bundle_count": len(files),
            "total_bytes": total_bytes,
            "bundles": bundles,
        }

    def gc(self, max_age_days: float = 30.0) -> dict[str, Any]:
        """Remove bundle files (and corrupt backups) whose last-modified time is older than ``max_age_days``."""
        cutoff = time.time() - max_age_days * 86400.0
        removed: list[str] = []
        for pattern in ("*/*.sqlite", "*/*.sqlite.corrupt"):
            for p in list(self.root.glob(pattern)):
                sha_key = p.name.split(".")[0]
                with self._lock_for(sha_key):
                    try:
                        if p.exists() and p.stat().st_mtime < cutoff:
                            p.unlink()
                            removed.append(p.name)
                    except OSError:
                        continue
        return {"removed_count": len(removed), "removed": removed}


def load_document(
    path: str | Path,
    backend: str = "pypdfium2",
    store: BundleStore | None = None,
) -> PdfDocument:
    """Load a PDF document with optional persistent BundleStore acceleration."""
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"PDF not found: {p}")
    doc_sha = file_sha256(p)
    if store is None:
        doc = PdfDocument.from_pdf(p, backend=backend)
        doc.sha256 = doc_sha
        return doc

    ext_fp = extractor_fingerprint(backend)
    cached = store.load_text(doc_sha, ext_fp)
    if cached is not None:
        return PdfDocument(
            path=p,
            pages=cached.pages,
            toc_bookmarks=cached.toc_bookmarks,
            backend=backend,
            page_labels=cached.page_labels,
            sha256=doc_sha,
        )

    doc = PdfDocument.from_pdf(p, backend=backend)
    doc.sha256 = doc_sha
    st = p.stat()
    store.save_text(
        doc_sha,
        ext_fp,
        TextLayer(pages=doc.pages, page_labels=doc.page_labels, toc_bookmarks=doc.toc_bookmarks),
        meta={"size": st.st_size, "mtime_ns": st.st_mtime_ns, "path_hint": p.name, "backend": backend},
    )
    return doc
