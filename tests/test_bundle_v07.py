"""Unit tests for v0.7.0 persistent SQLite WAL BundleStore and VisionPageCard hot reload."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from anchordoc.docbase import PdfDocument
from anchordoc.index import SelectiveVisionIndexer
from anchordoc.store import (
    BundleStore,
    TextLayer,
    extractor_fingerprint,
    file_sha256,
    load_document,
    vision_fingerprint,
)


def _write_minimal_pdf(path: Path, n_pages: int = 2) -> None:
    """Create a valid minimal PDF via pypdfium2 so real PDF loading works offline."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument.new()
    for _ in range(n_pages):
        pdf.new_page(200, 200)
    with open(path, "wb") as f:
        pdf.save(f)
    pdf.close()


def test_bundle_roundtrip_and_layered_invalidation(tmp_path):
    store = BundleStore(tmp_path / "bundles")
    sha1 = "a" * 64
    sha2 = "b" * 64
    ext1 = "ext_fp_1"
    ext2 = "ext_fp_2"
    vis1 = vision_fingerprint("m1", "tpl1", 1.15, 360)
    vis2 = vision_fingerprint("m2", "tpl1", 1.15, 360)

    layer = TextLayer(
        pages=["Page one text", "Page two text"],
        page_labels=["i", "1"],
        toc_bookmarks=[(1, "Intro", 1)],
    )
    store.save_text(sha1, ext1, layer, meta={"path_hint": "sample.pdf"})
    store.put_vision_card(sha1, vis1, 2, "[VISION:CHART] Sales | Up 20%", raw={"title": "Sales"}, usage={"prompt_tokens": 500})

    # Exact hit
    loaded = store.load_text(sha1, ext1)
    assert loaded is not None
    assert loaded.pages == ["Page one text", "Page two text"]
    assert loaded.page_labels == ["i", "1"]
    assert loaded.toc_bookmarks == [(1, "Intro", 1)]
    assert store.get_vision_cards(sha1, vis1) == {2: "[VISION:CHART] Sales | Up 20%"}

    # Different doc sha -> miss
    assert store.load_text(sha2, ext1) is None
    # Different extractor_fp -> text miss, vision still hits!
    assert store.load_text(sha1, ext2) is None
    assert store.get_vision_cards(sha1, vis1) == {2: "[VISION:CHART] Sales | Up 20%"}
    # Different vision_fp -> vision miss, text still hits!
    assert store.get_vision_cards(sha1, vis2) == {}
    assert store.load_text(sha1, ext1) is not None


def test_bundle_concurrent_writes_zero_loss(tmp_path):
    store = BundleStore(tmp_path / "bundles")
    sha = "c" * 64
    vfp = "vis_concurrent"

    def _write(p: int) -> None:
        store.put_vision_card(sha, vfp, p, f"[VISION:SLIDE] Slide {p}", raw={"p": p})

    with ThreadPoolExecutor(max_workers=6) as ex:
        list(ex.map(_write, range(1, 31)))

    cards = store.get_vision_cards(sha, vfp)
    assert len(cards) == 30
    assert cards[1] == "[VISION:SLIDE] Slide 1"
    assert cards[30] == "[VISION:SLIDE] Slide 30"


def test_bundle_corrupt_db_auto_quarantines_and_rebuilds(tmp_path):
    store = BundleStore(tmp_path / "bundles")
    sha = "d" * 64
    db_path = store.bundle_path(sha)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_bytes(b"NOT A VALID SQLITE FILE CORRUPTED BYTES")

    assert store.load_text(sha, "ext1") is None
    assert db_path.with_suffix(".sqlite.corrupt").exists()

    # Rebuilding afterwards succeeds cleanly
    store.save_text(sha, "ext1", TextLayer(pages=["Recovered page"]))
    got = store.load_text(sha, "ext1")
    assert got is not None and got.pages == ["Recovered page"]


def test_vision_indexer_second_run_zero_llm_and_zero_renders(tmp_path, monkeypatch):
    pdf_path = tmp_path / "slides.pdf"
    _write_minimal_pdf(pdf_path, n_pages=3)
    store = BundleStore(tmp_path / "bundles")

    class _CountingVLM:
        model = "test-vlm"

        def __init__(self) -> None:
            self.calls = 0

        def chat_json(self, purpose, system, user, max_tokens=360, images_b64=None):
            self.calls += 1
            return {"page_type": "slide", "title": f"Slide {self.calls}", "summary": "Chart data", "key_data": ["42%"]}, "{}"

        def snapshot_stats(self):
            return {"calls": self.calls}

    doc1 = load_document(pdf_path, store=store)
    assert doc1.sha256 == file_sha256(pdf_path)
    llm1 = _CountingVLM()
    idx1 = SelectiveVisionIndexer(llm1, min_text_chars=180)
    cards1, st1 = idx1.index_document(doc1, store=store)
    assert st1["newly_indexed"] == 3 and st1["from_store"] == 0 and llm1.calls == 3
    assert len(cards1) == 3

    # Second run: forbid BOTH rendering and LLM calls!
    doc2 = load_document(pdf_path, store=store)

    def _boom(*args, **kwargs):
        raise AssertionError("render_page_png_b64 must NOT be called when VisionCards hit BundleStore!")

    monkeypatch.setattr(doc2, "render_page_png_b64", _boom)

    class _BoomLLM:
        model = "test-vlm"

        def chat_json(self, *args, **kwargs):
            raise AssertionError("LLM must NOT be called when VisionCards hit BundleStore!")

        def snapshot_stats(self):
            return {"calls": 0}

    cards2, st2 = SelectiveVisionIndexer(_BoomLLM(), min_text_chars=180).index_document(doc2, store=store)
    assert cards2 == cards1
    assert st2["from_store"] == 3 and st2["newly_indexed"] == 0
    assert store.info()["bundle_count"] == 1
