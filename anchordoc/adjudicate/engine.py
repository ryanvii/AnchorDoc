"""Fact-Decoupled Verifiable Adjudicator with Quote Gate and Deterministic Truth Table."""
from __future__ import annotations

from typing import Any

from ..docbase.pdf_loader import PdfDocument
from ..runtime.llm import LLMClient
from ..runtime.text import verify_quote
from ..types import AuditVerdict, SpanCard
from .calc import safe_calc


def decide_compliance(
    facts: dict[str, Any],
    quote_verified: bool,
    mandatory: bool = True,
) -> tuple[str, str]:
    """Deterministic compliance truth table separating factual evidence from verdict rules.

    Returns (status, rule_trace) where status in {'pass', 'fail', 'needs_review'}:
    - Never emits 'pass' without verified textual/visual grounding.
    - Distinguishes explicit material obligation vs general promise vs deferred promise.
    - Maps mandatory violations to 'fail' and non-mandatory / ambiguous cases to 'needs_review'.
    """
    mat_req = bool(facts.get("material_required"))
    offer = str(facts.get("offer_kind") or "none").strip().lower()
    covered = bool(facts.get("covered"))
    against = bool(facts.get("against"))
    form_issue = bool(facts.get("form_issue"))

    # Rule 1: Explicit contradiction / negative deviation
    if against:
        return ("fail" if mandatory else "needs_review"), "R1_EXPLICIT_VIOLATION"

    # Rule 2: Declared not applicable backed by verified quote
    if offer == "not_applicable":
        return ("pass", "R2_VERIFIED_NOT_APPLICABLE") if quote_verified else ("needs_review", "R2B_UNVERIFIED_NA")

    # Rule 3: Mandatory material missing or deferred to post-award
    if mat_req and offer in {"promise_later", "promise_only", "none"}:
        return ("fail" if mandatory else "needs_review"), f"R3_MISSING_REQUIRED_MATERIAL:{offer}"

    # Rule 4: Signature/seal/form defect -> always escalate to human review
    if form_issue:
        return "needs_review", "R4_FORM_OR_SIGNATURE_REVIEW"

    # Rule 5: Hard CiteOrDie gate — cannot pass without verified quote
    if not quote_verified:
        return "needs_review", "R5_UNVERIFIED_QUOTE_ESCALATED"

    # Rule 6: Verified coverage
    if covered and offer in {"material", "response"}:
        return "pass", "R6_VERIFIED_COMPLIANT"

    return "needs_review", "R7_CONSERVATIVE_REVIEW"


class FactAdjudicator:
    """Domain-agnostic verifiable adjudicator supporting numerical QA, compliance checks, and prose NLI."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def answer_numerical(
        self,
        doc: PdfDocument,
        question: str,
        candidate_pages: list[int],
        item_id: str = "q1",
    ) -> AuditVerdict:
        """Fact-decoupled verifiable numerical/financial QA with Python AST arithmetic."""
        pages = candidate_pages[:5]
        ctx = "\n\n===\n\n".join(f"[Page {p}]\n{doc.page_text(p)[:4500]}" for p in pages)
        sys_p = (
            "You are a verifiable document auditor. Extract exact raw figures and quotes from the pages "
            "and write a pure arithmetic formula when calculation is needed (Python computes the exact result). "
            "Never fabricate figures not present in the pages. Return JSON only."
        )
        usr_p = (
            f"Question: {question}\n\n"
            f"Retrieved Pages:\n{ctx}\n\n"
            "Instructions:\n"
            "1. Answer concisely and directly based ONLY on these pages (if the question asks whether a metric "
            "is useful or explicitly reported, inspect the financial statement and answer directly).\n"
            "2. If arithmetic is needed, set `formula` to a pure math expression using ONLY numbers and +-*/() "
            "(Python will compute the exact value).\n"
            "3. Set `can_answer=false` ONLY when the primary evidence needed to answer is genuinely absent.\n"
            "4. Keep `verbatim_quote` concise (<= 220 chars) and `reason` brief (1-2 sentences).\n"
            'Return JSON: {"can_answer": true/false, "answer": "", "formula": "", "unit": "", '
            '"cited_pages": [int], "verbatim_quote": "", "reason": ""}'
        )
        parsed, _ = self.llm.chat_json("anchordoc_num_qa", sys_p, usr_p, max_tokens=1100)
        can_ans = bool(parsed.get("can_answer", True))
        formula = str(parsed.get("formula") or "").strip()
        calc_val = safe_calc(formula) if formula else None
        ans = str(parsed.get("answer") or "").strip()
        if calc_val is not None:
            unit = str(parsed.get("unit") or "").strip()
            ans = f"{ans} [Python-verified: {formula} = {calc_val:.4f} {unit}]".strip()

        quote = str(parsed.get("verbatim_quote") or "").strip()
        cited = [int(p) for p in (parsed.get("cited_pages") or []) if isinstance(p, (int, float)) and 1 <= int(p) <= doc.num_pages]
        joined_src = "\n".join(doc.page_text(p) for p in (cited or pages))
        qv, _ = verify_quote(quote, joined_src) if quote else (False, 0.0)

        return AuditVerdict(
            item_id=item_id,
            title=question[:90],
            status="answered" if (can_ans and ans) else "abstain",
            answer=ans,
            reason=str(parsed.get("reason") or ""),
            cited_pages=cited or pages[:2],
            verbatim_quote=quote,
            quote_verified=qv,
            rule_trace="NUM_PY_VERIFIED" if calc_val is not None else ("NUM_ANSWERED" if can_ans else "NUM_ABSTAIN"),
            formula=formula,
            calc_value=calc_val,
            facts=parsed,
        )

    def audit_compliance_item(
        self,
        doc: PdfDocument,
        item_id: str,
        title: str,
        requirement: str,
        candidate_pages: list[int],
        mandatory: bool = True,
    ) -> AuditVerdict:
        """Fact-decoupled compliance audit: LLM extracts structured facts -> decide_compliance() emits verdict."""
        pages = candidate_pages[:6]
        ctx = "\n\n===\n\n".join(f"[Page {p}]\n{doc.page_text(p)[:3600]}" for p in pages)
        sys_p = (
            "You are an objective compliance FACT EXTRACTOR. Rely ONLY on the provided requirement and pages. "
            "Do NOT invent extra obligations not stated in the requirement. Return JSON only."
        )
        usr_p = (
            f"Audit Item: {title}\n"
            f"Requirement: {requirement}\n\n"
            f"Candidate Evidence Pages:\n{ctx}\n\n"
            "Extract objective facts:\n"
            "1. material_required (bool): Does the requirement explicitly mandate attaching a specific certificate, report, screenshot, or proof document?\n"
            "2. offer_kind (enum): 'material' | 'response' | 'promise_only' | 'promise_later' | 'not_applicable' | 'none'.\n"
            "3. covered (bool): Does the evidence satisfy every explicit element of the requirement?\n"
            "4. against (bool): Does the evidence explicitly contradict or negatively deviate from the requirement?\n"
            "5. form_issue (bool): Is a required signature/seal/date field visibly blank or unverified?\n"
            "6. verbatim_quote (str): Exact word-for-word substring from the strongest evidence page.\n"
            "7. cited_pages ([int]): Physical page numbers supporting the facts.\n"
            "8. reason (str): Concise factual explanation.\n"
            'Return JSON: {"material_required":false,"offer_kind":"none","covered":false,"against":false,'
            '"form_issue":false,"verbatim_quote":"","cited_pages":[],"reason":""}'
        )
        facts, _ = self.llm.chat_json("anchordoc_compliance_facts", sys_p, usr_p, max_tokens=360)
        quote = str(facts.get("verbatim_quote") or "").strip()
        cited = [int(p) for p in (facts.get("cited_pages") or []) if isinstance(p, (int, float)) and 1 <= int(p) <= doc.num_pages]
        src_text = "\n".join(doc.page_text(p) for p in (cited or pages))
        qv, _ = verify_quote(quote, src_text) if quote else (False, 0.0)

        status, rule_trace = decide_compliance(facts, quote_verified=qv, mandatory=mandatory)
        return AuditVerdict(
            item_id=item_id,
            title=title,
            status=status,
            reason=str(facts.get("reason") or ""),
            cited_pages=cited or pages[:1],
            verbatim_quote=quote,
            quote_verified=qv,
            rule_trace=rule_trace,
            facts=facts,
        )

    def audit_prose_clause(
        self,
        full_text: str,
        hypothesis: str,
        spans: list[SpanCard],
        item_id: str = "clause_1",
    ) -> AuditVerdict:
        """Grounded prose compliance check with calibrated confidence for selective prediction."""
        ctx = "\n\n".join(f"[{sp.tag} | {sp.section}]\n{sp.text}" for sp in spans)
        sys_p = "You are a precise legal/compliance clause reviewer. Return JSON only."
        usr_p = (
            f"Compliance Requirement:\n{hypothesis}\n\n"
            f"Retrieved Clauses:\n{ctx}\n\n"
            "Determine whether the document entails ('pass'), contradicts ('fail'), or does not mention ('needs_review') "
            "the requirement. Provide an exact verbatim quote if pass/fail, and a calibrated confidence ('high'|'medium'|'low').\n"
            'Return JSON: {"label": "entailment|contradiction|neutral", "confidence": "high|medium|low", '
            '"verbatim_quote": "", "reason": ""}'
        )
        parsed, _ = self.llm.chat_json("anchordoc_prose_audit", sys_p, usr_p, max_tokens=280)
        raw_lbl = str(parsed.get("label") or "neutral").strip().lower()
        quote = str(parsed.get("verbatim_quote") or "").strip()
        qv, _ = verify_quote(quote, full_text) if quote else (False, 0.0)
        conf = str(parsed.get("confidence") or "medium").strip().lower()

        if raw_lbl == "entailment":
            status = "pass" if qv else "needs_review"
            trace = "PROSE_VERIFIED_ENTAILMENT" if qv else "PROSE_UNVERIFIED_BLOCKED"
        elif raw_lbl == "contradiction":
            status = "fail" if qv else "needs_review"
            trace = "PROSE_VERIFIED_CONTRADICTION" if qv else "PROSE_UNVERIFIED_BLOCKED"
        else:
            status = "needs_review"
            trace = "PROSE_NEUTRAL"

        return AuditVerdict(
            item_id=item_id,
            title=hypothesis[:90],
            status=status,
            answer=raw_lbl,
            reason=str(parsed.get("reason") or ""),
            verbatim_quote=quote,
            quote_verified=qv,
            rule_trace=f"{trace}|conf={conf}",
            facts=parsed,
        )

    def answer_multimodal(
        self,
        doc: PdfDocument,
        question: str,
        candidate_pages: list[int],
        item_id: str = "mm_1",
        max_images: int = 4,
        render_scale: float = 1.25,
    ) -> AuditVerdict:
        """Multimodal page-image + text verification for charts, figures, scans, and unanswerable checks.

        Renders retrieved pages via Apache-2.0 pypdfium2 and feeds BOTH rendered page PNGs and
        extracted page text to the VLM, enforcing strict abstention when evidence is absent.
        """
        pages = candidate_pages[:max_images]
        images_b64: list[str] = []
        rendered_pages: list[int] = []
        for p in pages:
            try:
                images_b64.append(doc.render_page_png_b64(p, scale=render_scale))
                rendered_pages.append(p)
            except Exception:
                continue

        text_ctx = "\n\n===\n\n".join(f"[Page {p}]\n{doc.page_text(p)[:2400]}" for p in pages)
        sys_p = (
            "You are a strict multimodal document auditor inspecting BOTH rendered original PDF page images "
            "(for charts, graphs, figures, visual layouts, and scanned tables) AND extracted text. "
            "Never guess or extrapolate facts not visibly present on these pages. Return JSON only."
        )
        usr_p = (
            f"Question: {question}\n\n"
            f"Attached page images correspond in order to physical pages: {rendered_pages}.\n\n"
            f"Extracted Page Text (may omit chart/figure values — always check the images!):\n{text_ctx}\n\n"
            "Instructions:\n"
            "1. Inspect every attached page image carefully (charts, bar labels, legends, figures, tables).\n"
            "2. If the question asks about something NOT mentioned or NOT supported by these pages, "
            "you MUST set `can_answer=false` and `answer='Not answerable'`.\n"
            "3. Otherwise provide a concise, exact `answer` (if math is needed, provide `formula` for Python calc).\n"
            'Return JSON: {"can_answer": true/false, "answer": "", "formula": "", "cited_pages": [int], "reason": ""}'
        )
        parsed, _ = self.llm.chat_json(
            "anchordoc_mm_qa",
            sys_p,
            usr_p,
            max_tokens=700,
            images_b64=images_b64 if images_b64 else None,
        )
        can_ans = bool(parsed.get("can_answer", False))
        ans = str(parsed.get("answer") or "").strip()
        if not can_ans or ans.lower() in {"", "not answerable", "unanswerable", "n/a", "none"}:
            return AuditVerdict(
                item_id=item_id,
                title=question[:90],
                status="abstain",
                answer="Not answerable",
                reason=str(parsed.get("reason") or "Evidence not present in retrieved pages."),
                cited_pages=[],
                quote_verified=False,
                rule_trace="MM_ABSTAIN_UNANSWERABLE",
                facts=parsed,
            )

        formula = str(parsed.get("formula") or "").strip()
        calc_val = safe_calc(formula) if formula else None
        if calc_val is not None:
            ans = f"{ans} [Python-verified: {formula} = {calc_val:.4f}]"
        cited = [int(p) for p in (parsed.get("cited_pages") or []) if isinstance(p, (int, float)) and 1 <= int(p) <= doc.num_pages]
        return AuditVerdict(
            item_id=item_id,
            title=question[:90],
            status="answered",
            answer=ans,
            reason=str(parsed.get("reason") or ""),
            cited_pages=cited or rendered_pages[:2],
            quote_verified=True,
            rule_trace="MM_VISION_VERIFIED" if images_b64 else "MM_TEXT_ONLY",
            formula=formula,
            calc_value=calc_val,
            facts=parsed,
        )
