"""Descriptive geometry metrics with explicit ties and uninformative truth."""
from __future__ import annotations
import numpy as np
from scipy.stats import rankdata


def cosine_matrix(values):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Expected finite two-dimensional features")
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    unit = np.divide(x, norm, out=np.zeros_like(x), where=norm > 0)
    return np.clip(unit @ unit.T, -1, 1)


def rank_correlation(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.ndim != 1 or a.shape != b.shape or len(a) < 2:
        raise ValueError("Paired vectors required")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite pair values")
    ar, br = rankdata(a), rankdata(b)
    ar, br = ar-ar.mean(), br-br.mean()
    denom = np.linalg.norm(ar)*np.linalg.norm(br)
    return float(np.clip(np.dot(ar, br)/denom, -1, 1)) if denom > 0 else np.nan


def fractional_topk(scores, k):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not np.isfinite(scores).all() or not 0 < k < len(scores):
        raise ValueError("Finite candidate scores and k < candidate count required")
    threshold = np.partition(scores, len(scores)-k)[len(scores)-k]
    above, equal = scores > threshold, scores == threshold
    result = above.astype(np.float64)
    result[equal] = (k-int(above.sum()))/int(equal.sum())
    return result


def expected_dcg(predicted_scores, relevance, k):
    scores, rel = np.asarray(predicted_scores, dtype=np.float64), np.asarray(relevance, dtype=np.float64)
    if scores.ndim != 1 or scores.shape != rel.shape or not np.isfinite(scores).all() or not np.isfinite(rel).all():
        raise ValueError("Finite aligned candidate arrays required")
    if not 0 < k <= len(scores) or (rel < 0).any():
        raise ValueError("Invalid cutoff or relevance")
    order = np.argsort(-scores, kind="stable")
    descending, ordered_rel = scores[order], rel[order]
    discounts = 1/np.log2(np.arange(2, k+2))
    result, first = 0.0, 0
    while first < k:
        end = int(np.searchsorted(-descending, -descending[first], side="right"))
        result += float(ordered_rel[first:end].mean()*discounts[first:min(end,k)].sum())
        first = end
    return result


def prepare_truth(similarity, k=10, minimum_group_n=12):
    values = np.asarray(similarity, dtype=np.float64)
    n = len(values)
    if values.shape != (n,n) or not np.isfinite(values).all():
        raise ValueError("Finite square truth similarities required")
    relevance = np.zeros((n,n), dtype=np.float64)
    idcg = np.full(n, np.nan)
    random_ndcg = np.full(n, np.nan)
    status = np.full(n, "INSUFFICIENT_CANDIDATES", dtype="U32")
    unique = np.zeros(n, dtype=int)
    if n < minimum_group_n or n-1 <= k:
        return relevance, idcg, random_ndcg, status, unique
    discounts = 1/np.log2(np.arange(2,k+2))
    for i in range(n):
        others = np.arange(n) != i
        scores = values[i,others]
        unique[i] = len(np.unique(scores))
        rel = fractional_topk(scores,k)
        relevance[i,others] = rel
        if unique[i] == 1:
            status[i] = "UNINFORMATIVE_TRUTH"
            continue
        idcg[i] = np.dot(np.sort(rel)[::-1][:k],discounts)
        random_ndcg[i] = rel.mean()*discounts.sum()/idcg[i]
        status[i] = "VALID"
    return relevance, idcg, random_ndcg, status, unique


def score_neighbors(similarity, relevance, idcg, k=10):
    n = len(idcg)
    similarity = np.asarray(similarity, dtype=np.float64)
    if similarity.shape != (n,n) or relevance.shape != (n,n) or not np.isfinite(similarity).all():
        raise ValueError("Aligned finite square prediction/relevance matrices required")
    scores = np.full(n,np.nan)
    for i in np.flatnonzero(np.isfinite(idcg)):
        others = np.arange(n) != i
        scores[i] = expected_dcg(similarity[i,others],relevance[i,others],k)/idcg[i]
    return scores
