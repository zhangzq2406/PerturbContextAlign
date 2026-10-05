from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.stats import rankdata
from sklearn.metrics.pairwise import cosine_similarity


def jaccard_binary(X):
    X = sparse.csr_matrix(X).astype(bool).astype(np.int8)
    inter = (X @ X.T).toarray().astype(float)
    n = np.asarray(X.sum(axis=1)).ravel().astype(float)
    union = n[:, None] + n[None, :] - inter
    out = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    np.fill_diagonal(out, 1.0)
    return out


def entity_similarity(X, metric: str):
    if metric == "cosine":
        return cosine_similarity(X)
    if metric in {"jaccard", "tanimoto"}:
        return jaccard_binary(X)
    raise ValueError(metric)


def best_match_set_similarity(left: list[int], right: list[int], entity_S: np.ndarray) -> float:
    block = entity_S[np.ix_(left, right)]
    return float(0.5 * (np.nanmax(block, axis=1).mean() + np.nanmax(block, axis=0).mean()))


def condition_similarity(component_rows: list[list[int]], entity_S: np.ndarray) -> np.ndarray:
    n = len(component_rows)
    if component_rows and all(len(x) == 1 for x in component_rows):
        idx = [x[0] for x in component_rows]
        return np.asarray(entity_S)[np.ix_(idx, idx)].astype(np.float32, copy=False)
    out = np.eye(n, dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            value = best_match_set_similarity(component_rows[i], component_rows[j], entity_S)
            out[i, j] = out[j, i] = value
    return out


def align_matrices(A, a_ids, B, b_ids):
    ai = {str(x): i for i, x in enumerate(a_ids)}
    bi = {str(x): i for i, x in enumerate(b_ids)}
    common = [str(x) for x in b_ids if str(x) in ai]
    ia, ib = [ai[x] for x in common], [bi[x] for x in common]
    return np.asarray(A)[np.ix_(ia, ia)], np.asarray(B)[np.ix_(ib, ib)], np.asarray(common, object)


def align_three(A, a_ids, B, b_ids, C, c_ids):
    apos = {str(x): i for i, x in enumerate(a_ids)}
    bpos = {str(x): i for i, x in enumerate(b_ids)}
    cpos = {str(x): i for i, x in enumerate(c_ids)}
    common = [str(x) for x in c_ids if str(x) in apos and str(x) in bpos]
    ia, ib, ic = [apos[x] for x in common], [bpos[x] for x in common], [cpos[x] for x in common]
    return np.asarray(A)[np.ix_(ia, ia)], np.asarray(B)[np.ix_(ib, ib)], np.asarray(C)[np.ix_(ic, ic)], np.asarray(common, object)


def rank_normalize_similarity(S):
    S = np.asarray(S, float)
    n = len(S)
    idx = np.triu_indices(n, 1)
    values = S[idx]
    ranks = rankdata(values, method="average")
    ranks = (ranks - 1) / max(1, len(ranks) - 1)
    out = np.eye(n, dtype=float)
    out[idx] = ranks
    out[(idx[1], idx[0])] = ranks
    return out


def hybrid_similarity(language_S, native_S, language_weight=0.5, method="rank"):
    if method == "rank":
        language_S, native_S = rank_normalize_similarity(language_S), rank_normalize_similarity(native_S)
    return language_weight * np.asarray(language_S) + (1.0 - language_weight) * np.asarray(native_S)
