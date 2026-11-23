"""Zero-dependency BM25, Reciprocal Rank Fusion, and programmatic quote verification."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, Sequence

_CJK_RE = re.compile(r"[\u3000-\u303f\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\uff00-\uffef]")


def estimate_tokens(text: str) -> int:
    """CJK-aware fast token estimator (0 dependencies): ceil(n_cjk * 1.0 + n_other / 4.0)."""
    s = text or ""
    if not s:
        return 0
    n_cjk = len(_CJK_RE.findall(s))
    n_other = len(s) - n_cjk
    return int(math.ceil(n_cjk * 1.0 + n_other / 4.0))

_STOPWORDS = frozenset(
    "the a an and or of to in for on with by at from as is are be was were been "
    "that this these those it its shall may will should would could can not "
    "no any all such other into under upon within between without per via".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:['._-][a-z0-9]+)*|[\u4e00-\u9fff]")


def _light_inflection_stems(tok: str) -> list[str]:
    """Conservative zero-dependency English inflection folding for BM25 recall.

    Emits normalized base forms alongside the original token so ``repurchases``,
    ``repurchased``, and ``repurchase`` match without losing exact-token specificity.
    """
    if len(tok) < 5 or not tok.isascii() or not tok.isalpha():
        return []
    stems: list[str] = []
    if tok.endswith("ies") and len(tok) >= 6:
        stems.append(tok[:-3] + "y")
    elif (tok.endswith(("ches", "shes", "xes", "zes")) or tok.endswith("sses")) and len(tok) >= 6:
        stems.append(tok[:-2])
    elif tok.endswith("s") and not tok.endswith(("ss", "us", "is")):
        stems.append(tok[:-1])
    elif tok.endswith("ed") and len(tok) >= 6:
        # Silent-e verbs (`repurchased` -> `repurchase`, `acquired` -> `acquire`)
        # plus consonant-ending verbs (`reported` -> `report`, `expanded` -> `expand`).
        stems.append(tok[:-1])
        if tok[-3] not in "aeioul":
            stems.append(tok[:-2])
    elif tok.endswith("ing") and len(tok) >= 7:
        base = tok[:-3]
        stems.append(base)
        stems.append(base + "e")
    return [s for s in stems if len(s) >= 3 and s != tok and s not in _STOPWORDS]


def tokenize(text: str, drop_stop: bool = True) -> list[str]:
    """Bilingual (English word + light inflection stem + CJK character) tokenizer."""
    raw = _TOKEN_RE.findall((text or "").lower())
    out: list[str] = []
    for t in raw:
        if drop_stop and ((len(t) <= 1 and not ("\u4e00" <= t <= "\u9fff")) or t in _STOPWORDS):
            continue
        out.append(t)
        for st in _light_inflection_stems(t):
            out.append(st)
    return out


class BM25Okapi:
    """Pure-Python Robertson-Walker BM25Okapi (zero external dependencies)."""

    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.N = len(corpus_tokens)
        self.doc_len = [len(d) for d in corpus_tokens]
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 1.0
        self.tfs: list[Counter[str]] = []
        df: Counter[str] = Counter()
        for doc in corpus_tokens:
            c = Counter(doc)
            self.tfs.append(c)
            for term in c:
                df[term] += 1
        self.idf: dict[str, float] = {
            term: math.log((self.N - freq + 0.5) / (freq + 0.5) + 1.0)
            for term, freq in df.items()
        }

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores = [0.0] * self.N
        if not query_tokens or not self.N:
            return scores
        for term in query_tokens:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for idx, tf_map in enumerate(self.tfs):
                tf = tf_map.get(term, 0)
                if not tf:
                    continue
                dl = self.doc_len[idx]
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl)
                scores[idx] += idf * (tf * (self.k1 + 1.0)) / denom
        return scores


def rrf(*ranked_lists: Iterable[int], k: int = 60) -> list[int]:
    """Scale-invariant Reciprocal Rank Fusion."""
    scores: dict[int, float] = {}
    for lst in ranked_lists:
        for rank, item in enumerate(lst):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    return [item for item, _ in sorted(scores.items(), key=lambda x: -x[1])]


def _norm_ws_punct(text: str) -> str:
    s = (text or "").lower()
    s = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# Financial / numerical semantic normalization: preserve sign (- / + / accounting parentheses)
# and percent/decimal structure so `-12%` NEVER verifies against `12%` and `$(500)` NEVER
# verifies against `$500`.
_UNICODE_MINUS_RE = re.compile(r"[\u2212\u2013\u2014\ufe63\uff0d]")
_ACCT_NEG_RE = re.compile(
    r"(?:(?P<curr>[$¥€£])\s*\(\s*(?P<n1>\d[\d,]*(?:\.\d+)?)\s*(?P<pct1>%?)\s*\)"
    r"|\(\s*(?P<curr2>[$¥€£])?\s*(?P<n2>\d[\d,]*(?:\.\d+)?)\s*(?P<pct2>%)\s*\)"
    r"|\(\s*(?P<curr3>[$¥€£])?\s*(?P<n3>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+)\s*\)"
    r"|(?<=\s)\(\s*(?P<n4>\d{2,})\s*\))"
)
_SIGNED_NUM_TOKEN_RE = re.compile(
    r"(?<![a-z0-9.])(?P<sign>[+-])?\s*(?P<curr>[$¥€£])?\s*(?P<num>\d[\d,]*(?:\.\d+)?)(?P<pct>\s*%)?"
)


def _normalize_financial_text(text: str) -> str:
    """Normalize unicode minus and accounting parenthetical negatives while keeping sign & %."""
    s = _UNICODE_MINUS_RE.sub("-", text or "")

    def _acct_repl(m: re.Match[str]) -> str:
        n4 = m.group("n4")
        if n4 is not None and 1900 <= int(n4) <= 2099:
            return m.group(0)
        num = (
            m.group("n1") or m.group("n2") or m.group("n3") or n4 or ""
        ).replace(",", "")
        pct = m.group("pct1") or m.group("pct2") or ""
        return f" -{num}{pct} "

    s = _ACCT_NEG_RE.sub(_acct_repl, s)
    # Remove thousands commas inside numbers so `1,200` matches `1200`
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    return s


def extract_signed_numbers(text: str) -> list[tuple[float, bool, str]]:
    """Extract (signed_value, is_percent, raw_normalized) preserving financial signs."""
    s = _normalize_financial_text(text)
    out: list[tuple[float, bool, str]] = []
    for m in _SIGNED_NUM_TOKEN_RE.finditer(s):
        raw_num = m.group("num")
        sign = m.group("sign") or ""
        # Ignore hyphenated words/dates like `FY-2024` or `2024-01-01` where `-` is a separator
        if sign == "-":
            pos = m.start("sign")
            if pos > 0 and s[pos - 1].isalnum():
                sign = ""
        pct = bool((m.group("pct") or "").strip())
        try:
            val = float(raw_num)
        except ValueError:
            continue
        if sign == "-":
            val = -val
        norm_str = f"{'-' if val < 0 else ''}{raw_num}{'%' if pct else ''}"
        out.append((val, pct, norm_str))
    return out


def _norm_semantic_quote(text: str) -> str:
    """Whitespace/punctuation normalization that PRESERVES financial signs (-/+), decimals and %."""
    s = _normalize_financial_text(text).lower()
    # Mark explicit negative sign glued to a number (excluding hyphenated tokens like `non-gaap` or `2024-2025`)
    def _tag_num(m: re.Match[str]) -> str:
        sign = m.group("sign") or ""
        pos = m.start()
        if sign == "-" and pos > 0 and s[pos - 1].isalnum():
            sign = ""
        num = m.group("num")
        pct = " pct " if (m.group("pct") or "").strip() else ""
        sign_tok = " neg " if sign == "-" else ""
        return f"{sign_tok}{num.replace('.', ' dot ')}{pct}"

    s = _SIGNED_NUM_TOKEN_RE.sub(_tag_num, s)
    s = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _sign_compatible_in_window(quote: str, source_text: str) -> bool:
    """Verify quote preserves exact local sign, decimal, and percent semantics in source_text.

    Uses local semantic-preserving substring match so even if `+12%` appears elsewhere on the
    same page, `Operating margin was 12%` STILL fails against `Operating margin was -12%`.
    """
    if not extract_signed_numbers(quote):
        return True
    return _norm_semantic_quote(quote) in _norm_semantic_quote(source_text)


def verify_quote(quote: str, source_text: str, min_chars: int = 8) -> tuple[bool, float]:
    """Programmatic CiteOrDie verbatim quote verification gate.

    Preserves financial sign semantics (`-12%`, `$(500)`, `(12.5%)`) and percent markers so
    sign-flipped quotes (e.g., `12%` vs `-12%`) are strictly rejected.

    Returns (is_verified, overlap_ratio):
    - Exact normalized substring with intact sign/percent semantics -> (True, 1.0)
    - Ordered ellipsis-split verbatim segments with intact signs -> (True, 0.95)
    - Sign-flipped / paraphrased / fabricated quote -> (False, overlap)
    """
    nq = _norm_ws_punct(quote)
    ns = _norm_ws_punct(source_text)
    if len(nq) < min_chars or not ns:
        return False, 0.0
    if nq in ns:
        if _sign_compatible_in_window(quote, source_text):
            return True, 1.0
        return False, 0.0
    parts = [p.strip() for p in re.split(r"\.\.\.|…", quote or "") if len(_norm_ws_punct(p)) >= min_chars]
    if len(parts) >= 2:
        pos = 0
        ok = True
        for part in parts:
            np = _norm_ws_punct(part)
            idx = ns.find(np, pos)
            if idx == -1 or not _sign_compatible_in_window(part, source_text):
                ok = False
                break
            pos = idx + len(np)
        if ok:
            return True, 0.95
    q_toks = set(tokenize(nq, drop_stop=False))
    s_toks = set(tokenize(ns, drop_stop=False))
    overlap = (len(q_toks & s_toks) / len(q_toks)) if q_toks else 0.0
    return False, round(overlap, 3)


def localize_quote_pages(quote: str, pages: Sequence[str], min_chars: int = 8) -> list[int]:
    """Return 1-based physical page numbers whose text verbatim-verifies ``quote``."""
    if not quote or not pages:
        return []
    hits: list[int] = []
    for pno, txt in enumerate(pages, start=1):
        ok, _ = verify_quote(quote, txt, min_chars=min_chars)
        if ok:
            hits.append(pno)
    return hits
