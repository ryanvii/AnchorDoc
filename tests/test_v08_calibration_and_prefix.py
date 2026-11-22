"""Unit tests for v0.8.0: localized unanswerable calibration & prefix-stable nav prompt."""

from __future__ import annotations

import unittest
from typing import Any

from anchordoc.adjudicate.engine import FactAdjudicator
from anchordoc.docbase import PdfDocument
from anchordoc.index import DocumentIndex
from anchordoc.retrieve.pipeline import AnchorRetriever


class _StubVLM:
    model = "stub-vlm"

    def __init__(self, payload: dict[str, Any]):
        self.payload = payload

    def snapshot_stats(self) -> dict[str, Any]:
        return {"calls": 1}

    def chat_json(
        self,
        purpose: str,
        system: str,
        user: str,
        max_tokens: int = 700,
        images_b64: list[str] | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], str]:
        return dict(self.payload), "{}"


class _CaptureNavLLM:
    model = "stub-nav"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def snapshot_stats(self) -> dict[str, Any]:
        return {"calls": len(self.calls)}

    def chat_json(
        self,
        purpose: str,
        system: str,
        user: str,
        max_tokens: int = 700,
        images_b64: list[str] | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], str]:
        self.calls.append((purpose, user))
        if purpose == "anchordoc_nav":
            return {"direct_pages": [2], "ranges": [[1, 3]], "rationale": "ok"}, "{}"
        if purpose == "anchordoc_locate":
            return {"selected_pages": [2], "sufficient": True, "missing": ""}, "{}"
        if purpose == "anchordoc_qexp":
            return {"keywords": ["revenue"], "synonyms": ["sales"]}, "{}"
        return {}, "{}"


class _StubPdfDoc(PdfDocument):
    def render_page_png_b64(self, page_num: int, dpi: int = 140) -> str:
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="


class TestV08CalibrationAndPrefix(unittest.TestCase):
    def _make_doc(self) -> PdfDocument:
        return _StubPdfDoc(
            path=None,
            pages=[
                "Page 1: Introduction to AlphaCorp annual overview.",
                "Page 2: Table of operating expenses for 2022.",
                "Page 3: Appendix mentioning Project Zephyr in 2024.",
            ],
        )

    def test_dual_channel_insufficient_is_opt_in_strict_abstention(self) -> None:
        doc = self._make_doc()
        llm = _StubVLM({
            "answer": "42%",
            "evidence_status": "derivable",
            "entity_time_match": True,
            "evidence_pages": [2],
            "observations": [{"page": 2, "detail": "guessed value"}],
            "reasoning": "rough inference",
        })
        adj = FactAdjudicator(llm)  # type: ignore[arg-type]
        # Default mode preserves visual chart derivations while recording diagnostic metadata
        v_default = adj.answer_multimodal(
            doc,
            "What was AlphaCorp's 2022 operating margin?",
            [2],
            locate_sufficient=False,
            b42_selected=[],
        )
        self.assertEqual(v_default.answer, "42%")
        self.assertTrue(v_default.meta["dual_channel_insufficient"])
        # Opt-in strict_abstention enforces the gate for high-precision deployments
        v_strict = adj.answer_multimodal(
            doc,
            "What was AlphaCorp's 2022 operating margin?",
            [2],
            locate_sufficient=False,
            b42_selected=[],
            strict_abstention=True,
        )
        self.assertEqual(v_strict.answer, "Not answerable")
        self.assertTrue(v_strict.rule_trace.startswith("MM_DUAL_CHANNEL_INSUFFICIENT"))

    def test_dual_channel_insufficient_does_not_block_stated(self) -> None:
        doc = self._make_doc()
        llm = _StubVLM({
            "answer": "$120M",
            "evidence_status": "stated",
            "entity_time_match": True,
            "evidence_pages": [2],
            "observations": [{"page": 2, "detail": "operating expenses $120M in 2022"}],
            "reasoning": "explicitly shown",
        })
        adj = FactAdjudicator(llm)  # type: ignore[arg-type]
        v = adj.answer_multimodal(
            doc,
            "What were AlphaCorp's 2022 operating expenses?",
            [2],
            locate_sufficient=False,
            b42_selected=[],
            strict_abstention=True,
        )
        self.assertEqual(v.answer, "$120M")
        self.assertEqual(v.status, "answered")

    def test_local_unattested_gates_only_in_strict_mode(self) -> None:
        doc = self._make_doc()
        llm = _StubVLM({
            "answer": "15%",
            "evidence_status": "derivable",
            "entity_time_match": True,
            "evidence_pages": [2],
            "observations": [{"page": 2, "detail": "some table"}],
            "reasoning": "inferred",
        })
        adj = FactAdjudicator(llm)  # type: ignore[arg-type]
        # '2024' exists on Page 3 (global haystack passes), but NOT on rendered Page 2.
        v_default = adj.answer_multimodal(
            doc,
            "What was the 2024 growth rate of Project Zephyr?",
            [2],
            locate_sufficient=True,
            b42_selected=[2],
        )
        self.assertEqual(v_default.answer, "15%")
        self.assertIn("2024", v_default.meta["local_unattested_terms"])
        v_strict = adj.answer_multimodal(
            doc,
            "What was the 2024 growth rate of Project Zephyr?",
            [2],
            locate_sufficient=True,
            b42_selected=[2],
            strict_abstention=True,
        )
        self.assertEqual(v_strict.answer, "Not answerable")
        self.assertTrue(v_strict.rule_trace.startswith("MM_LOCAL_UNATTESTED"))

    def test_nav_prompt_is_prefix_stable_across_queries(self) -> None:
        pages = [f"Section {i}\nDetailed content for page {i} about financial metrics." for i in range(1, 35)]
        doc = PdfDocument.from_pages(pages)
        llm = _CaptureNavLLM()
        ret = AnchorRetriever(doc, llm)  # type: ignore[arg-type]

        ret.retrieve("What was total revenue in FY2022?", top_k=3, route="navigate")
        ret.retrieve("What was operating income in FY2023?", top_k=3, route="navigate")

        nav_prompts = [user for purpose, user in llm.calls if purpose == "anchordoc_nav"]
        self.assertEqual(len(nav_prompts), 2)
        p1, p2 = nav_prompts
        self.assertLess(p1.index("Printed TOC"), p1.index("Query:"))
        self.assertLess(p1.index("Zero-LLM Structural Anchor Map"), p1.index("Query:"))
        # Exact shared prefix up to "Query:" must be byte-identical!
        prefix1 = p1[: p1.index("Query:")]
        prefix2 = p2[: p2.index("Query:")]
        self.assertEqual(prefix1, prefix2)
        self.assertGreater(len(prefix1), 200)


if __name__ == "__main__":
    unittest.main()
