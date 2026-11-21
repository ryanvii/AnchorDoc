"""Gateway multi-image capacity probe (one lightweight request per image count).

Checks that the configured OpenAI-compatible VLM endpoint accepts N images in a single request
(e.g. 5 full pages + 4 zoom tiles = 9) and reports prompt tokens per call, so the per-image token
budget (and therefore the effective resolution) can be measured instead of assumed.

Usage:
  ANCHORDOC_LLM_BASE=... ANCHORDOC_LLM_MODEL=... ANCHORDOC_LLM_KEY_FILE=... \
    python3 tools/probe_multi_image.py path/to/any.pdf [--counts 1,5,9]
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anchordoc.docbase import PdfDocument  # noqa: E402
from anchordoc.runtime import CachedLLM  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--counts", default="1,5,9")
    ap.add_argument("--page", type=int, default=1)
    args = ap.parse_args()

    doc = PdfDocument.from_pdf(args.pdf)
    page = max(1, min(args.page, doc.num_pages))
    full = doc.render_page_png_b64(page, scale=1.25)
    tiler = getattr(doc, "render_page_tiles_b64", None)
    tiles = list(tiler(page, grid=2, overlap=0.1, scale=2.0)) if callable(tiler) else []
    pool = [full] + tiles
    with tempfile.TemporaryDirectory() as td:  # fresh cache: always hits the gateway
        llm = CachedLLM(cache_dir=td)
        for n in [int(x) for x in args.counts.split(",") if x.strip()]:
            imgs = [pool[i % len(pool)] for i in range(n)]
            llm.reset_stats()
            try:
                parsed, _ = llm.chat_json(
                    f"probe_{n}",
                    "Return JSON only.",
                    f'You received {n} images. Return JSON: {{"images_seen": <int>, "title_of_first": ""}}',
                    max_tokens=60,
                    images_b64=imgs,
                )
                st = llm.snapshot_stats()
                print(f"n_images={n:2d} OK  prompt_tokens={st['prompt_tokens']:6d} "
                      f"(~{st['prompt_tokens'] / max(1, n):.0f}/image incl. text) reply={parsed}")
            except Exception as exc:  # report and continue with the next size
                print(f"n_images={n:2d} FAILED: {str(exc)[:200]}")


if __name__ == "__main__":
    main()
