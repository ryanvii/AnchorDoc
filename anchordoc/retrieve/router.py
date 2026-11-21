"""Cost-derived three-tier adaptive router (`read_all` | `full_pool` | `navigate`).

Pure zero-LLM routing decisions derived solely from token/image cost arithmetic
(never tuned on benchmark accuracy):
- `read_all`: skip retrieval LLM hops when the entire document fits comfortably inside
  one reader call (<= 4,000 tokens for prose NLI; <= 6,000 tokens & 0 visual pages for
  text QA; <= 5 pages for multimodal QA).
- `full_pool`: skip global `nav` when `n_pages <= 40`; feed all pages into the single
  query-aware `locate` call under a fixed character budget (`LOCATE_CHAR_BUDGET`).
- `navigate`: standard AnchorMap + G7 + Query-Aware Locator + conditional B42 pipeline
  (byte-for-byte identical to v0.6 for long documents).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..docbase.pdf_loader import PdfDocument
from ..runtime.text import estimate_tokens

Route = Literal["read_all", "full_pool", "navigate"]
Task = Literal["qa_text", "qa_multimodal", "prose_nli"]

# Pre-registered cost-derived thresholds (v0.7.0)
PROSE_READALL_MAX_TOKENS = 4_000
TEXT_READALL_MAX_TOKENS = 6_000
MM_READALL_MAX_PAGES = 5
FULLPOOL_MAX_PAGES = 40
LOCATE_CHAR_BUDGET = 13_500
FULLPOOL_MIN_SNIPPET = 320
DEFAULT_SNIPPET_CHARS = 900


@dataclass(frozen=True)
class DocProfile:
    """Lightweight Zero-LLM structural profile of a document for routing."""

    n_pages: int
    est_text_tokens: int
    n_visual_pages: int


@dataclass(frozen=True)
class RoutePlan:
    """Deterministic routing plan produced by ``plan_route`` / ``plan_prose_route``."""

    route: Route
    reason: str
    est_tokens: int
    snippet_chars: int = DEFAULT_SNIPPET_CHARS


def profile_document(
    doc: PdfDocument,
    vision_summaries: dict[int, str] | None = None,
    min_text_chars: int = 180,
) -> DocProfile:
    """Compute a document's page count, CJK-aware text token estimate, and visual-page count."""
    vsums = vision_summaries or {}
    est_text = sum(estimate_tokens(p or "") for p in doc.pages)
    n_visual = 0
    for pno, txt in enumerate(doc.pages, start=1):
        if len((txt or "").strip()) < min_text_chars or pno in vsums:
            n_visual += 1
    return DocProfile(n_pages=doc.num_pages, est_text_tokens=est_text, n_visual_pages=n_visual)


def _full_pool_snippet_chars(n_pages: int) -> int:
    if n_pages <= 0:
        return DEFAULT_SNIPPET_CHARS
    return max(FULLPOOL_MIN_SNIPPET, min(DEFAULT_SNIPPET_CHARS, LOCATE_CHAR_BUDGET // n_pages))


def plan_route(
    profile: DocProfile,
    task: Task,
    force: Route | Literal["auto"] = "auto",
) -> RoutePlan:
    """Choose between `read_all`, `full_pool`, and `navigate` deterministically."""
    if force != "auto":
        sc = _full_pool_snippet_chars(profile.n_pages) if force == "full_pool" else DEFAULT_SNIPPET_CHARS
        return RoutePlan(route=force, reason=f"forced:{force}", est_tokens=profile.est_text_tokens, snippet_chars=sc)

    if task == "prose_nli":
        if profile.est_text_tokens <= PROSE_READALL_MAX_TOKENS:
            return RoutePlan(
                route="read_all",
                reason=f"prose_est_tok={profile.est_text_tokens}<={PROSE_READALL_MAX_TOKENS}",
                est_tokens=profile.est_text_tokens,
            )
        return RoutePlan(
            route="navigate",
            reason=f"prose_est_tok={profile.est_text_tokens}>{PROSE_READALL_MAX_TOKENS}",
            est_tokens=profile.est_text_tokens,
        )

    if task == "qa_multimodal":
        if profile.n_pages <= MM_READALL_MAX_PAGES:
            return RoutePlan(
                route="read_all",
                reason=f"mm_pages={profile.n_pages}<={MM_READALL_MAX_PAGES}",
                est_tokens=profile.est_text_tokens,
            )
        if profile.n_pages <= FULLPOOL_MAX_PAGES:
            return RoutePlan(
                route="full_pool",
                reason=f"mm_pages={profile.n_pages}<={FULLPOOL_MAX_PAGES}",
                est_tokens=profile.est_text_tokens,
                snippet_chars=_full_pool_snippet_chars(profile.n_pages),
            )
        return RoutePlan(
            route="navigate",
            reason=f"mm_pages={profile.n_pages}>{FULLPOOL_MAX_PAGES}",
            est_tokens=profile.est_text_tokens,
        )

    # task == "qa_text"
    if profile.n_visual_pages == 0 and profile.est_text_tokens <= TEXT_READALL_MAX_TOKENS:
        return RoutePlan(
            route="read_all",
            reason=f"text_only_est_tok={profile.est_text_tokens}<={TEXT_READALL_MAX_TOKENS}",
            est_tokens=profile.est_text_tokens,
        )
    if profile.n_pages <= FULLPOOL_MAX_PAGES:
        return RoutePlan(
            route="full_pool",
            reason=f"text_pages={profile.n_pages}<={FULLPOOL_MAX_PAGES}",
            est_tokens=profile.est_text_tokens,
            snippet_chars=_full_pool_snippet_chars(profile.n_pages),
        )
    return RoutePlan(
        route="navigate",
        reason=f"text_pages={profile.n_pages}>{FULLPOOL_MAX_PAGES}",
        est_tokens=profile.est_text_tokens,
    )


def plan_prose_route(full_text: str, force: Route | Literal["auto"] = "auto") -> RoutePlan:
    """Convenience router for single-string prose/contract documents."""
    est = estimate_tokens(full_text)
    prof = DocProfile(n_pages=1, est_text_tokens=est, n_visual_pages=0)
    return plan_route(prof, task="prose_nli", force=force)
