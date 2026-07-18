"""
Lightweight, dependency-free BM25 keyword scoring for hybrid retrieval.

The dense (vector/cosine) retriever captures semantic similarity but can miss exact
lexical matches — an acronym or an exact tool name that the embedding blurs. BM25 is
the classic sparse counterpart: it rewards documents that literally contain the query
terms, weighted by term rarity (idf) and length-normalised term frequency. Fusing the
two (see retrieval_service) improves accuracy over either alone.

Kept deterministic and self-contained (Golden Rule: algorithms over examples) — no
external index, no network, computed over the resume's own chunks at query time.
"""
import math
import re
from collections import Counter
from typing import List

# A token starts with an alphanumeric and may carry tech punctuation (c++, ci/cd, node.js).
_TOKEN = re.compile(r"[a-z0-9][a-z0-9+#./-]*")


def tokenize(text: str) -> List[str]:
    return _TOKEN.findall((text or "").lower())


def bm25_scores(query: str, docs: List[str], k1: float = 1.5, b: float = 0.75) -> List[float]:
    """
    BM25 score of ``query`` against each document in ``docs`` (the corpus). Returns a
    list of scores aligned to ``docs`` (0.0 when a doc shares no query term). The corpus
    itself supplies the idf statistics, so scores are self-normalised per resume.
    """
    doc_tokens = [tokenize(d) for d in docs]
    n = len(doc_tokens)
    if n == 0:
        return []
    avgdl = sum(len(t) for t in doc_tokens) / n
    if avgdl == 0:
        return [0.0] * n

    df: Counter = Counter()
    for toks in doc_tokens:
        for term in set(toks):
            df[term] += 1

    q_terms = [t for t in tokenize(query) if df.get(t)]
    if not q_terms:
        return [0.0] * n

    idf = {t: math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) for t in set(q_terms)}

    scores: List[float] = []
    for toks in doc_tokens:
        tf = Counter(toks)
        dl = len(toks)
        s = 0.0
        for term in q_terms:
            f = tf.get(term, 0)
            if not f:
                continue
            denom = f + k1 * (1 - b + b * dl / avgdl)
            s += idf[term] * (f * (k1 + 1)) / denom
        scores.append(s)
    return scores
