"""Fact-Decoupled Verifiable Adjudicator with Quote Gate and Deterministic Truth Tables.

v0.6 design principles (all domain-agnostic):
- Reasoning-first JSON schemas: evidence -> operands/observations -> formula -> reasoning -> final answer.
- Python arithmetic is authoritative; LLM-written numbers never override ``safe_calc`` results.
- LLMs extract facts; deterministic truth tables (``decide_compliance`` / ``decide_nli``) decide verdicts.
- Grounding is reported honestly in ``AuditVerdict.meta`` (never hard-coded as verified).
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import Any

from ..docbase.pdf_loader import PdfDocument
from ..retrieve.router import Route
from ..runtime.llm import LLMClient
from ..runtime.text import localize_quote_pages, verify_quote
from ..types import AuditVerdict, SpanCard
from .answer_contract import (
    AnswerContract,
    check_contract,
    compose_numeric_answer,
    infer_answer_contract,
    operands_grounded,
    unattested_terms,
)
from .calc import safe_calc

_NA_ANSWERS = frozenset({"", "not answerable", "unanswerable", "n/a", "none", "not available", "cannot be determined"})


def _is_na_answer(text: str) -> bool:
    """True when an answer is empty or a standard abstention phrase (ignoring trailing punctuation)."""
    norm = re.sub(r"[.\s!?:;\"']+$", "", (text or "").strip().lower())
    return norm in _NA_ANSWERS
_VISUAL_CUE_RE = re.compile(
    r"\b(?:how\s+many|count|colou?rs?|charts?|graphs?|figures?|legends?|images?|pictures?|photos?|logos?|diagrams?|"
    r"maps?|icons?|axis|bars?|pie|slides?|screenshots?|people|persons)\b",
    re.I,
)


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



def _as_bool(val: Any) -> bool:
    """Robust bool parsing for LLM JSON (handles "false"/"no"/"0" strings)."""
    if isinstance(val, str):
        return val.strip().lower() in {"true", "yes", "1", "y"}
    return bool(val)


def decide_nli(facts: dict[str, Any], quote_verified: bool = False) -> tuple[str, str, str]:
    """Deterministic NLI truth table over LLM-extracted facts (dual-fact agreement + calibrated confidence).

    Returns (label, rule_path, confidence) with label in {entailment, contradiction, neutral}.
    Pure logic, no domain vocabulary:
    - no governing clause and stance silent                         -> neutral (high confidence)
    - existential hypothesis: require BOTH permitted_in_some_case AND stance=supports for entailment;
      conflicting facts (permitted=True while stance!=supports) abstain to neutral (low confidence)
    - universal hypothesis: require BOTH counterexample AND stance=opposes for counterexample path;
      conflicting facts (counterexample=True while stance!=opposes) abstain to neutral (low confidence)
    - otherwise follow the extracted stance (supports / opposes / silent)
    Confidence is derived from rule-path reliability + quote verification, not self-reported.
    """
    topic = facts.get("topic_clauses")
    has_topic = bool(topic) if isinstance(topic, list) else bool(str(topic or "").strip())
    stance = str(facts.get("stance") or "silent").strip().lower()
    if stance not in {"supports", "opposes", "silent"}:
        stance = "supports" if "support" in stance or "entail" in stance else ("opposes" if "oppos" in stance or "contra" in stance else "silent")
    quant = str(facts.get("hyp_quantifier") or "plain").strip().lower()
    permitted = _as_bool(facts.get("permitted_in_some_case"))
    counter = _as_bool(facts.get("counterexample"))

    if not has_topic and stance == "silent":
        return "neutral", "NLI_TOPIC_ABSENT", "high"
    if quant == "existential":
        if permitted and stance == "supports":
            label, path, agreed = "entailment", "NLI_EXISTENTIAL_WITNESS", True
        elif permitted and stance != "supports":
            return "neutral", "NLI_EXISTENTIAL_CONFLICT", "low"
        elif stance == "opposes":
            label, path, agreed = "contradiction", "NLI_EXISTENTIAL_EXCLUDED", True
        elif stance == "supports":
            label, path, agreed = "entailment", "NLI_STANCE_SUPPORTS", True
        else:
            label, path, agreed = "neutral", "NLI_TOPIC_SILENT", False
    elif quant == "universal":
        if counter and stance == "opposes":
            label, path, agreed = "contradiction", "NLI_UNIVERSAL_COUNTEREXAMPLE", True
        elif counter and stance != "opposes":
            return "neutral", "NLI_UNIVERSAL_CONFLICT", "low"
        elif stance == "opposes":
            label, path, agreed = "contradiction", "NLI_STANCE_OPPOSES", True
        elif stance == "supports":
            label, path, agreed = "entailment", "NLI_STANCE_SUPPORTS", True
        else:
            label, path, agreed = "neutral", "NLI_TOPIC_SILENT", False
    else:
        if stance == "supports":
            label, path, agreed = "entailment", "NLI_STANCE_SUPPORTS", True
        elif stance == "opposes":
            label, path, agreed = "contradiction", "NLI_STANCE_OPPOSES", True
        else:
            label, path, agreed = "neutral", "NLI_TOPIC_SILENT", False

    if label == "neutral":
        conf = "low" if has_topic else "medium"
    elif agreed and quote_verified and has_topic:
        conf = "high"
    elif quote_verified or has_topic:
        conf = "medium"
    else:
        conf = "low"
    return label, path, conf


class FactAdjudicator:
    """Domain-agnostic verifiable adjudicator supporting numerical QA, compliance checks, and prose NLI."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm


    # ------------------------------------------------------------------
    # Shared helpers (v0.6)
    # ------------------------------------------------------------------
    @staticmethod
    def _int_pages(vals: Any, num_pages: int) -> list[int]:
        out: list[int] = []
        for p in vals or []:
            if isinstance(p, (int, float)) and not isinstance(p, bool) and 1 <= int(p) <= num_pages and int(p) not in out:
                out.append(int(p))
        return out

    @staticmethod
    def _grounding_scope(
        doc: PdfDocument,
        cited: list[int],
        verified_quotes: list[str],
        route: Route,
        fallback: list[int],
    ) -> tuple[list[int], str]:
        """Determine the page scope used for operand provenance (Grounding Scope Guard).

        In `read_all` mode, NEVER fall back to the entire document when the model omits
        citations — otherwise every number anywhere in the document would pass as grounded.
        Instead, deterministically localize pages from verbatim-verified quotes first, then
        cited pages, and otherwise return an empty scope (``"none"``).
        """
        if route == "read_all":
            localized: list[int] = []
            for qt in verified_quotes:
                hits = localize_quote_pages(qt, doc.pages)
                # Ignore non-discriminative boilerplate quotes (e.g. "in millions") hitting >3 pages
                if 0 < len(hits) <= 3:
                    for p in hits:
                        if p not in localized:
                            localized.append(p)
            union_pages = list(localized)
            for p in cited:
                if p not in union_pages:
                    union_pages.append(p)
            if localized and cited:
                return union_pages, "quote_and_cited"
            if localized:
                return localized, "quote_localized"
            if cited:
                return list(cited), "cited_pages"
            return [], "none"
        if cited:
            return list(cited), "cited_pages"
        return list(fallback), "retrieved_fallback"

    def _repair_answer(
        self,
        purpose: str,
        question: str,
        contract: AnswerContract,
        issues: list[str],
        previous: dict[str, Any],
        context: str,
    ) -> dict[str, Any]:
        """At most ONE cheap repair call when the answer violates the deterministic contract."""
        sys_p = "You are a verifiable document auditor fixing the FORM of an answer. Return JSON only."
        usr_p = (
            f"Question: {question}\n"
            f"Expected answer: {contract.describe()}\n"
            f"Detected problems: {', '.join(issues)}\n\n"
            f"Previous extraction (JSON):\n{json.dumps(previous, ensure_ascii=False)[:2500]}\n\n"
            f"Source excerpt:\n{context[:3000]}\n\n"
            "Fix ONLY the detected problems using figures that appear above (e.g., convert a percentage into a count "
            "using the stated base size n, or convert the amount into the requested scale). If the needed figure is not "
            "available, keep the previous answer.\n"
            'Return JSON: {"formula": "", "unit": "", "final_answer": ""}'
        )
        fixed, _ = self.llm.chat_json(purpose, sys_p, usr_p, max_tokens=400)
        return fixed if isinstance(fixed, dict) else {}

    # ------------------------------------------------------------------
    # Numerical / financial QA (v0.6: reasoning-first + deterministic composer)
    # ------------------------------------------------------------------
    def answer_numerical(
        self,
        doc: PdfDocument,
        question: str,
        candidate_pages: list[int],
        item_id: str = "q1",
        allow_repair: bool = True,
        *,
        route: Route = "navigate",
        page_char_cap: int = 4500,
    ) -> AuditVerdict:
        """Reasoning-first verifiable QA: evidence -> operands -> formula -> reasoning -> final answer.

        - Python (``safe_calc``) is authoritative for arithmetic: for numeric answers the computed value
          REPLACES the LLM-written number; stance answers keep their Yes/No and get the verified value.
        - Answer contract (type / scale / rounding) is inferred from the question; one repair call max.
        - Operand provenance and quote verification are recorded as structured audit fields in ``meta``.
        - In ``read_all`` mode, all candidate pages are read and ``_grounding_scope`` prevents
          whole-document fallback when verifying operands.
        """
        pages = list(candidate_pages) if route == "read_all" else candidate_pages[:5]
        eff_cap = max(page_char_cap, 24000 // max(1, len(pages))) if route == "read_all" else page_char_cap
        ctx = "\n\n===\n\n".join(f"[Page {p}]\n{doc.page_text(p)[:eff_cap]}" for p in pages)
        contract = infer_answer_contract(question)
        sys_p = (
            "You are a verifiable document auditor. Work strictly in the JSON field order given: collect evidence, "
            "list operands, write a formula, reason, and ONLY THEN state the final answer (never decide the answer "
            "before the evidence and computation). Python computes formulas exactly. Never fabricate figures. "
            "Return JSON only."
        )
        usr_p = (
            f"Question: {question}\n\n"
            f"Retrieved Pages:\n{ctx}\n\n"
            f"Expected answer form: {contract.describe()}.\n\n"
            "Instructions:\n"
            "1. evidence: up to 4 items copied VERBATIM from the pages (<= 200 chars each) with their page number.\n"
            "2. operands: every raw figure you use, exactly as printed, with its page (watch table scale such as "
            "'in millions', the reporting period, and the exact entity/segment asked).\n"
            "3. formula: a pure arithmetic expression using ONLY numbers and + - * / ( ) ** (Python computes it); "
            "empty if no calculation is needed.\n"
            "4. reasoning: <= 4 short sentences. If the question asks whether a metric is useful/relevant or whether "
            "something is healthy/improving, first compute or locate the metric, then judge it.\n"
            "5. can_answer=false ONLY when the primary evidence needed is genuinely absent from these pages.\n"
            "6. final_answer: concise and consistent with the reasoning and formula; yes/no stance FIRST when applicable.\n"
            'Return JSON: {"evidence": [{"page": 0, "quote": ""}], "operands": [{"name": "", "value": "", "page": 0}], '
            '"formula": "", "reasoning": "", "can_answer": true, "unit": "", "final_answer": "", "cited_pages": [0]}'
        )
        parsed, _ = self.llm.chat_json("anchordoc_num_qa_v6", sys_p, usr_p, max_tokens=1400)
        parsed = parsed if isinstance(parsed, dict) else {}
        can_ans = _as_bool(parsed.get("can_answer", True))
        formula = str(parsed.get("formula") or "").strip()
        unit = str(parsed.get("unit") or "").strip()
        ans = str(parsed.get("final_answer") or parsed.get("answer") or "").strip()
        cited = self._int_pages(parsed.get("cited_pages"), doc.num_pages)

        # Verbatim quote verification (checked first so verified quotes can localize pages in read_all)
        ev = parsed.get("evidence") if isinstance(parsed.get("evidence"), list) else []
        quotes = [str(e.get("quote") or "").strip() for e in ev if isinstance(e, dict) and str(e.get("quote") or "").strip()]
        if not quotes and parsed.get("verbatim_quote"):
            quotes = [str(parsed.get("verbatim_quote")).strip()]
        quote_search_src = "\n".join(doc.page_text(p) for p in (pages if route == "read_all" else (cited or pages)))
        verified_quotes: list[str] = []
        for qt in quotes:
            ok, _ = verify_quote(qt, quote_search_src)
            if ok:
                verified_quotes.append(qt)
        quote = verified_quotes[0] if verified_quotes else (quotes[0] if quotes else "")
        qv = bool(verified_quotes)

        src_pages, grounding_scope = self._grounding_scope(doc, cited, verified_quotes, route, fallback=pages)
        joined_src = "\n".join(doc.page_text(p) for p in src_pages)

        # Answer contract check -> at most one cheap repair call
        issues = check_contract(contract, ans, unit) if (can_ans and ans) else []
        repaired = False
        if issues and allow_repair:
            repair_pages = src_pages[:2] if src_pages else pages[:2]
            fixed = self._repair_answer(
                "anchordoc_num_repair", question, contract, issues,
                {k: parsed.get(k) for k in ("evidence", "operands", "formula", "reasoning", "unit", "final_answer")},
                "\n\n".join(f"[Page {p}]\n{doc.page_text(p)[:1500]}" for p in repair_pages),
            )
            new_ans = str(fixed.get("final_answer") or "").strip()
            if new_ans:
                ans, repaired = new_ans, True
                formula = str(fixed.get("formula") or "").strip() or formula
                unit = str(fixed.get("unit") or "").strip() or unit
                issues = check_contract(contract, ans, unit)

        calc_val = safe_calc(formula) if formula else None
        if grounding_scope == "none":
            grounded, ungrounded = (False, [formula] if formula else [])
        else:
            grounded, ungrounded = operands_grounded(formula, joined_src) if formula else (True, [])
        ans, comp_tag = compose_numeric_answer(
            ans, calc_val, formula, contract, unit, operand_grounded=grounded
        )

        if calc_val is not None:
            rule = f"NUM_PY_VERIFIED|{comp_tag}"
        else:
            rule = "NUM_ANSWERED" if can_ans else "NUM_ABSTAIN"
        if formula:
            rule += "|OPERANDS_GROUNDED" if grounded else "|OPERANDS_UNGROUNDED"
        if grounding_scope == "none":
            rule += "|READALL_UNCITED"
        return AuditVerdict(
            item_id=item_id,
            title=question[:90],
            status="answered" if (can_ans and ans) else "abstain",
            answer=ans,
            reason=str(parsed.get("reasoning") or parsed.get("reason") or ""),
            cited_pages=src_pages if route == "read_all" else (cited or pages[:2]),
            verbatim_quote=quote[:240],
            quote_verified=qv,
            rule_trace=rule,
            formula=formula,
            calc_value=calc_val,
            facts=parsed,
            meta={
                "route": route,
                "grounding_scope": grounding_scope,
                "answer_contract": asdict(contract),
                "contract_issues": issues,
                "repaired": repaired,
                "composition": comp_tag,
                "operand_grounded": grounded,
                "ungrounded_operands": ungrounded,
            },
        )

    def audit_compliance_item(
        self,
        doc: PdfDocument,
        item_id: str,
        title: str,
        requirement: str,
        candidate_pages: list[int],
        mandatory: bool = True,
        *,
        route: Route = "navigate",
    ) -> AuditVerdict:
        """Fact-decoupled compliance audit: LLM extracts structured facts -> decide_compliance() emits verdict."""
        pages = list(candidate_pages) if route == "read_all" else candidate_pages[:6]
        eff_cap = max(3600, 24000 // max(1, len(pages))) if route == "read_all" else 3600
        ctx = "\n\n===\n\n".join(f"[Page {p}]\n{doc.page_text(p)[:eff_cap]}" for p in pages)
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


    # ------------------------------------------------------------------
    # Prose NLI (v0.6: fact extraction -> deterministic truth table)
    # ------------------------------------------------------------------
    def audit_prose_clause(
        self,
        full_text: str,
        hypothesis: str,
        spans: list[SpanCard],
        item_id: str = "clause_1",
    ) -> AuditVerdict:
        """Two-stage grounded NLI: the LLM extracts topic/modality/quantifier facts; ``decide_nli`` labels.

        Backward compatible: ``answer`` is the label (entailment|contradiction|neutral), ``status`` maps to
        pass / fail / needs_review, and ``facts['confidence']`` carries a rule-derived confidence tier.
        """
        ctx = "\n\n".join(f"[{sp.tag} | {sp.section}]\n{sp.text}" for sp in spans)
        sys_p = (
            "You are a precise contract analyst doing natural language inference between a contract and a hypothesis. "
            "You only extract facts; a deterministic rule engine assigns the final label. Return JSON only."
        )
        usr_p = (
            f"Hypothesis:\n{hypothesis}\n\n"
            f"Retrieved Clauses:\n{ctx}\n\n"
            "Fill the fields in order:\n"
            "1. topic_clauses: tags (e.g. \"s0012\") of clauses governing the SAME subject matter as the hypothesis "
            "(same party, action and object), or [] if none.\n"
            "2. verbatim_quote: exact word-for-word substring of the most relevant clause ('' if none).\n"
            "3. analysis: <= 3 sentences comparing clause and hypothesis; pay attention to modality (shall / may / "
            "shall not), quantifiers (some / all / only / any) and conditions.\n"
            "4. hyp_quantifier: 'existential' if the hypothesis only claims something is permitted/possible/true in at "
            "least SOME case (e.g. 'may ... some ...', 'in some circumstances'); 'universal' if it claims something holds "
            "for ALL cases or ONLY in a restricted way ('all', 'only', 'any', 'never', 'shall not ... any'); otherwise 'plain'.\n"
            "5. stance: 'supports' if the contract states or clearly implies the hypothesis; 'opposes' if the contract "
            "states or clearly implies its negation (e.g. prohibits what the hypothesis permits, or permits/requires what "
            "it forbids); 'silent' if the governing clauses do not settle the specific point.\n"
            "6. permitted_in_some_case (bool): the contract allows or requires what the hypothesis describes in at least "
            "one case, even if subject to conditions, consent, request or legal requirements.\n"
            "7. counterexample (bool): the contract explicitly provides a case that falsifies the hypothesis's universal "
            "claim (covers what the hypothesis excludes, or allows an exception the hypothesis rules out).\n"
            'Return JSON: {"topic_clauses": [], "verbatim_quote": "", "analysis": "", "hyp_quantifier": "plain", '
            '"stance": "silent", "permitted_in_some_case": false, "counterexample": false}'
        )
        parsed, _ = self.llm.chat_json("anchordoc_prose_nli_v6", sys_p, usr_p, max_tokens=450)
        parsed = parsed if isinstance(parsed, dict) else {}
        quote = str(parsed.get("verbatim_quote") or "").strip()
        qv, _ = verify_quote(quote, full_text) if quote else (False, 0.0)
        label, path, conf = decide_nli(parsed, quote_verified=qv)

        if label == "entailment":
            status = "pass" if qv else "needs_review"
        elif label == "contradiction":
            status = "fail" if qv else "needs_review"
        else:
            status = "needs_review"
        if label != "neutral" and not qv:
            path = f"{path}|PROSE_UNVERIFIED_BLOCKED"
        facts = dict(parsed)
        facts["label"] = label
        facts["confidence"] = conf
        return AuditVerdict(
            item_id=item_id,
            title=hypothesis[:90],
            status=status,
            answer=label,
            reason=str(parsed.get("analysis") or parsed.get("reason") or ""),
            verbatim_quote=quote,
            quote_verified=qv,
            rule_trace=f"{path}|conf={conf}",
            facts=facts,
            meta={"nli_path": path, "confidence_source": "rule_path"},
        )

    # ------------------------------------------------------------------
    # Multimodal QA (v0.6: tri-state evidence gate + visual fidelity)
    # ------------------------------------------------------------------
    @staticmethod
    def _pick_zoom_page(
        doc: PdfDocument,
        question: str,
        rendered_pages: list[int],
        vision_summaries: dict[int, str],
    ) -> int | None:
        """Choose the top-ranked page to zoom: visual question cue, or a low-text chart/figure page."""
        if not rendered_pages:
            return None
        top = rendered_pages[0]
        if _VISUAL_CUE_RE.search(question or ""):
            return top
        vs = (vision_summaries or {}).get(top, "")
        low_text = len((doc.page_text(top) or "").strip()) < 180
        if low_text and re.search(r"\[VISION:(?:CHART|FIGURE|TABLE|SLIDE)\]", vs):
            return top
        return None

    def answer_multimodal(
        self,
        doc: PdfDocument,
        question: str,
        candidate_pages: list[int],
        item_id: str = "mm_1",
        max_images: int = 5,
        render_scale: float = 1.25,
        vision_summaries: dict[int, str] | None = None,
        zoom: bool = True,
        max_total_images: int = 9,
        allow_repair: bool = True,
        *,
        route: Route = "navigate",
        locate_sufficient: bool | None = None,
        b42_selected: list[int] | None = None,
        strict_abstention: bool = False,
    ) -> AuditVerdict:
        """Multimodal page-image + text QA with a calibrated tri-state evidence gate.

        - Renders ALL top-k pages (default 5) plus 2x2 zoom tiles of the top visual page.
        - evidence_status in {stated, derivable, absent}: abstain ONLY when absent, when the pages show a
          different entity/time period, or when a hard proper noun / year of the question has zero
          attestation in the whole document (text + VisionPageCards).
        - Grounding is reported honestly: text_verified | vision_card_supported | vlm_attested.
        """
        vsums = vision_summaries or {}
        pages = candidate_pages[:max_images]
        images_b64: list[str] = []
        rendered_pages: list[int] = []
        for p in pages:
            try:
                images_b64.append(doc.render_page_png_b64(p, scale=render_scale))
                rendered_pages.append(p)
            except Exception:
                continue

        zoom_page = self._pick_zoom_page(doc, question, rendered_pages, vsums) if zoom else None
        n_tiles = 0
        tiler = getattr(doc, "render_page_tiles_b64", None)
        if zoom_page is not None and callable(tiler) and len(images_b64) < max_total_images:
            try:
                tiles = list(tiler(zoom_page, grid=2, overlap=0.1, scale=2.0))[: max_total_images - len(images_b64)]
                images_b64.extend(tiles)
                n_tiles = len(tiles)
            except Exception:
                n_tiles = 0

        contract = infer_answer_contract(question)
        text_blocks = []
        for p in pages:
            vs = vsums.get(p, "")
            body = (doc.page_text(p) or "")[:2400]
            text_blocks.append(f"[Page {p}]" + (f"\n(VisionCard) {vs[:600]}" if vs else "") + f"\n{body}")
        text_ctx = "\n\n===\n\n".join(text_blocks)
        legend = f"Images 1-{len(rendered_pages)} are full renders of physical pages {rendered_pages} (in this order)."
        if n_tiles:
            legend += (
                f" Images {len(rendered_pages) + 1}-{len(rendered_pages) + n_tiles} are high-resolution 2x2 zoom tiles "
                f"(top-left, top-right, bottom-left, bottom-right) of page {zoom_page} for reading small labels."
            )
        sys_p = (
            "You are a meticulous multimodal document auditor inspecting rendered PDF page images (charts, legends, "
            "figures, tables, photos) AND extracted text. Work strictly in the JSON field order: observe, classify the "
            "evidence, derive, reason, and ONLY THEN answer. Never invent values that are not visible. Return JSON only."
        )
        usr_p = (
            f"Question: {question}\n\n{legend}\n\n"
            f"Extracted Page Text (may be empty for scanned/slide pages - inspect the images carefully):\n{text_ctx}\n\n"
            f"Expected answer form: {contract.describe()}.\n\n"
            "Fill the fields in order:\n"
            "1. observations: up to 8 exact visible labels / values / sentences relevant to the question, each with its page.\n"
            "2. evidence_status: 'stated' if the answer is directly shown; 'derivable' if it can be computed, counted, "
            "compared or combined from what is shown (e.g. a share from two counts, a difference of two values, a count of "
            "visible or listed items across pages, a count from a percentage and a stated base size); 'absent' if the "
            "specific subject or metric asked is not on these pages.\n"
            "3. entity_time_match (bool): false if the pages only show a DIFFERENT entity, group, time period or metric "
            "than the one asked.\n"
            "4. derivation: the formula or the enumerated items (with pages) when derivable; '' otherwise.\n"
            "5. formula: pure arithmetic (numbers and + - * / ( ) **) when a calculation is needed, else ''.\n"
            "6. reasoning: <= 4 short sentences.\n"
            "7. final_answer: in the expected form; 'Not answerable' when evidence_status is 'absent' or entity_time_match is false.\n"
            'Return JSON: {"observations": [{"page": 0, "detail": ""}], "evidence_status": "absent", '
            '"entity_time_match": true, "derivation": "", "formula": "", "reasoning": "", "final_answer": "", "cited_pages": [0]}'
        )
        parsed, _ = self.llm.chat_json(
            "anchordoc_mm_qa_v6",
            sys_p,
            usr_p,
            max_tokens=1400,
            images_b64=images_b64 if images_b64 else None,
        )
        parsed = parsed if isinstance(parsed, dict) else {}
        status_ev = str(parsed.get("evidence_status") or "absent").strip().lower()
        if status_ev not in {"stated", "derivable", "absent"}:
            status_ev = "absent"
        etm = parsed.get("entity_time_match", True)
        etm = not (etm is False or str(etm).strip().lower() == "false")
        ans = str(parsed.get("final_answer") or parsed.get("answer") or "").strip()
        obs = parsed.get("observations") if isinstance(parsed.get("observations"), list) else []
        obs_details = [str(o.get("detail") or "").strip() for o in obs if isinstance(o, dict) and str(o.get("detail") or "").strip()]

        # Conservative deterministic attestation over the WHOLE document AND locally on rendered pages
        # (never include unverified VLM obs_details so hallucinated terms cannot self-attest).
        haystack = list(doc.pages) + list(vsums.values())
        missing_terms = unattested_terms(question, haystack)
        local_haystack = [doc.page_text(p) for p in pages] + [vsums.get(p, "") for p in pages]
        local_missing = unattested_terms(question, local_haystack) if any(x.strip() for x in local_haystack) else []
        dual_channel_insuff = (
            locate_sufficient is False
            and not (b42_selected or [])
            and status_ev != "stated"
        )
        meta: dict[str, Any] = {
            "evidence_status": status_ev,
            "entity_time_match": etm,
            "unattested_terms": missing_terms,
            "local_unattested_terms": local_missing,
            "locate_sufficient": locate_sufficient,
            "dual_channel_insufficient": dual_channel_insuff,
            "rendered_pages": rendered_pages,
            "zoom_page": zoom_page,
            "zoom_tiles": n_tiles,
            "answer_contract": asdict(contract),
        }
        gate = None
        if status_ev == "absent":
            gate = "MM_EVIDENCE_ABSENT"
        elif not etm:
            gate = "MM_ENTITY_TIME_MISMATCH"
        elif missing_terms:
            gate = "MM_UNATTESTED_TERMS"
        elif strict_abstention and dual_channel_insuff:
            # Opt-in only: text-only locators cannot read pixels, so unconditional gating
            # suppresses valid chart/figure derivations (-18.2pp on MMLongBench Chart/Fig).
            gate = "MM_DUAL_CHANNEL_INSUFFICIENT"
        elif strict_abstention and local_missing and status_ev != "stated":
            gate = "MM_LOCAL_UNATTESTED"
        elif _is_na_answer(ans):
            gate = "MM_MODEL_ABSTAINED"
        if gate:
            return AuditVerdict(
                item_id=item_id,
                title=question[:90],
                status="abstain",
                answer="Not answerable",
                reason=str(parsed.get("reasoning") or "Calibrated evidence gate: required evidence not present."),
                cited_pages=[],
                quote_verified=False,
                rule_trace=f"{gate}(status={status_ev},etm={etm})",
                facts=parsed,
                meta=meta,
            )

        cited = self._int_pages(parsed.get("cited_pages"), doc.num_pages)
        verified_obs = [d for d in obs_details if verify_quote(d, "\n".join(doc.page_text(p) for p in pages))[0]]
        src_pages, grounding_scope = self._grounding_scope(
            doc, cited, verified_obs, route, fallback=rendered_pages[:2]
        )
        formula = str(parsed.get("formula") or "").strip()
        issues = check_contract(contract, ans, "")
        repaired = False
        if issues and allow_repair:
            repair_pages = src_pages[:2] if src_pages else rendered_pages[:2]
            fixed = self._repair_answer(
                "anchordoc_mm_repair", question, contract, issues,
                {k: parsed.get(k) for k in ("observations", "derivation", "formula", "reasoning", "final_answer")},
                "\n\n".join(f"[Page {p}]\n{vsums.get(p, '')}\n{(doc.page_text(p) or '')[:1200]}" for p in repair_pages),
            )
            new_ans = str(fixed.get("final_answer") or "").strip()
            if new_ans and not _is_na_answer(new_ans):
                ans, repaired = new_ans, True
                formula = str(fixed.get("formula") or "").strip() or formula
                issues = check_contract(contract, ans, "")
        calc_val = safe_calc(formula) if formula else None

        # Honest grounding level
        grounding, quote = "vlm_attested", (obs_details[0] if obs_details else str(parsed.get("derivation") or ""))
        page_text_src = "\n".join(doc.page_text(p) for p in src_pages)
        card_src = " ".join(vsums.get(p, "") for p in src_pages)
        for d in obs_details:
            ok, overlap = verify_quote(d, page_text_src)
            if ok:
                grounding, quote = "text_verified", d
                break
            if card_src:
                ok_c, ov_c = verify_quote(d, card_src)
                if (ok_c or ov_c >= 0.6) and grounding == "vlm_attested":
                    grounding, quote = "vision_card_supported", d
        if grounding_scope == "none":
            grounded, ungrounded = (False, [formula] if formula else [])
        else:
            grounded, ungrounded = (
                operands_grounded(formula, page_text_src + " " + card_src + " " + " ".join(obs_details))
                if formula
                else (True, [])
            )
        ans, comp_tag = compose_numeric_answer(
            ans, calc_val, formula, contract, "", operand_grounded=grounded
        )
        meta.update({
            "route": route,
            "grounding_scope": grounding_scope,
            "grounding": grounding,
            "contract_issues": issues,
            "repaired": repaired,
            "composition": comp_tag,
            "operand_grounded": grounded,
            "ungrounded_operands": ungrounded,
        })
        base_rule = "MM_VISION_" + status_ev.upper() if images_b64 else "MM_TEXT_" + status_ev.upper()
        rule_trace = f"{base_rule}|{grounding.upper()}|{comp_tag}"
        if grounding_scope == "none":
            rule_trace += "|READALL_UNCITED"
        return AuditVerdict(
            item_id=item_id,
            title=question[:90],
            status="answered",
            answer=ans,
            reason=str(parsed.get("reasoning") or ""),
            cited_pages=src_pages,
            verbatim_quote=quote[:240],
            quote_verified=grounding == "text_verified",
            rule_trace=rule_trace,
            formula=formula,
            calc_value=calc_val,
            facts=parsed,
            meta=meta,
        )
