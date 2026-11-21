"""Zero-dependency BM25, Reciprocal Rank Fusion, and programmatic quote verification."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

_STOPWORDS = frozenset(
    "the a an and or of to in for on with by at from as is are be was were been "
    "that this these those it its shall may will should would could can not "
    "no any all such other into under upon within between without per via".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:['._-][a-z0-9]+)*|[\u4e00-\u9fff]")


def tokenize(text: str, drop_stop: bool = True) -> list[str]:
    """Bilingual (English word + CJK character) tokenizer."""
    toks = _TOKEN_RE.findall((text or "").lower())
    if drop_stop:
        toks = [t for t in toks if (len(t) > 1 or "\u4e00" <= t <= "\u9fff") and t not in _STOPWORDS]
    return toks


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


def verify_quote(quote: str, source_text: str, min_chars: int = 8) -> tuple[bool, float]:
    """Programmatic CiteOrDie verbatim quote verification gate.

    Returns (is_verified, overlap_ratio):
    - Exact whitespace/punctuation-normalized substring -> (True, 1.0)
    - Ordered ellipsis-split verbatim segments -> (True, 0.95)
    - Paraphrased / fabricated quote -> (False, token_overlap)
    """
    nq = _norm_ws_punct(quote)
    ns = _norm_ws_punct(source_text)
    if len(nq) < min_chars or not ns:
        return False, 0.0
    if nq in ns:
        return True, 1.0
    parts = [p.strip() for p in re.split(r"\.\.\.|…", quote or "") if len(_norm_ws_punct(p)) >= min_chars]
    if len(parts) >= 2:
        pos = 0
        ok = True
        for part in parts:
            np = _norm_ws_punct(part)
            idx = ns.find(np, pos)
            if idx == -1:
                ok = False
                break
            pos = idx + len(np)
        if ok:
            return True, 0.95
    q_toks = set(tokenize(nq, drop_stop=False))
    s_toks = set(tokenize(ns, drop_stop=False))
    overlap = (len(q_toks & s_toks) / len(q_toks)) if q_toks else 0.0
    return False, round(overlap, 3)
