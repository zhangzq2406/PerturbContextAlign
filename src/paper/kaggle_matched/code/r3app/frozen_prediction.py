"""Auditable, per-unit response scores for a fixed evaluation cohort.

All prediction/truth inputs are real, finite matrices with shape
``(n_conditions, n_genes)`` and must already have identical row and gene IDs
in identical order. Numerical shape checks cannot establish semantic identity.
No data are filtered, selected, fitted, aggregated across studies, or modified.

For gene-wise scores the caller MUST select one fixed context and exposure
and one row per perturbation before calling; do not pool dose repeats as
independent perturbations. Undefined correlations remain NaN (including a
constant prediction), while MAE/RMSE remain available. Non-finite model output
is rejected so that the run can be marked failed instead of silently dropping
its failed predictions. These scores do not provide inferential p-values.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def _real_array(value, name: str) -> np.ndarray:
    """Return a float64 view/copy without changing any caller-owned values."""
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must contain real, not complex, values")
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a numeric array") from exc
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _prediction_pair(prediction, truth) -> tuple[np.ndarray, np.ndarray]:
    prediction = _real_array(prediction, "prediction")
    truth = _real_array(truth, "truth")
    if prediction.ndim != 2 or truth.ndim != 2:
        raise ValueError("prediction and truth must be two-dimensional")
    if prediction.shape != truth.shape:
        raise ValueError("prediction and truth must have identical shapes")
    if prediction.shape[0] == 0 or prediction.shape[1] == 0:
        raise ValueError("prediction and truth must contain at least one row and gene")
    return prediction, truth


def _spearman_rows(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Pearson correlation of average ranks; never replaces undefined with 0."""
    n_rows, n_values = prediction.shape
    result = np.full(n_rows, np.nan, dtype=np.float64)
    if n_values < 2:
        return result
    pred_rank = rankdata(prediction, method="average", axis=1)
    true_rank = rankdata(truth, method="average", axis=1)
    pred_rank -= pred_rank.mean(axis=1, keepdims=True)
    true_rank -= true_rank.mean(axis=1, keepdims=True)
    denominator = np.sqrt(
        np.sum(pred_rank * pred_rank, axis=1)
        * np.sum(true_rank * true_rank, axis=1)
    )
    np.divide(
        np.sum(pred_rank * true_rank, axis=1), denominator,
        out=result, where=denominator > 0,
    )
    # Rounding can produce 1 + epsilon, which is not a valid correlation.
    return np.clip(result, -1.0, 1.0)


def condition_metrics(prediction, truth) -> dict[str, np.ndarray]:
    """Return one MAE, RMSE and across-gene Spearman for every condition.

    Keys: ``mae``, ``rmse``, ``spearman``, ``n_genes``, ``spearman_valid``
    and ``spearman_n``. ``n_genes`` is the unchanged denominator;
    ``spearman_n`` is n_genes when correlation is defined and 0 otherwise.
    This distinction records undefined scores, not excluded genes. No minimum
    biological evaluation size is imposed here beyond mathematical validity;
    the benchmark's preregistered eligibility rule belongs in the caller.
    """
    prediction, truth = _prediction_pair(prediction, truth)
    with np.errstate(over="ignore", invalid="ignore"):
        error = prediction - truth
    if not np.isfinite(error).all():
        raise ValueError("prediction minus truth produced non-finite values")
    absolute_error = np.abs(error)
    scale = np.max(absolute_error, axis=1)
    scaled_error = np.divide(
        absolute_error, scale[:, None], out=np.zeros_like(error),
        where=scale[:, None] > 0,
    )
    # Scaling avoids overflow when squaring or summing large finite errors.
    mae = scaled_error.mean(axis=1) * scale
    rmse = np.sqrt(np.mean(scaled_error * scaled_error, axis=1)) * scale
    spearman = _spearman_rows(prediction, truth)
    valid = np.isfinite(spearman)
    n_genes = np.full(prediction.shape[0], prediction.shape[1], dtype=np.int64)
    return {
        "mae": mae,
        "rmse": rmse,
        "spearman": spearman,
        "n_genes": n_genes,
        "spearman_valid": valid,
        "spearman_n": np.where(valid, n_genes, 0),
    }


def gene_metrics(prediction, truth) -> dict[str, np.ndarray]:
    """Return one across-perturbation Spearman for every gene.

    Rows must be unique perturbations at one fixed context and exposure;
    duplicate rows, context/exposure membership and gene eligibility must be
    checked against metadata by the caller. There is no silent deduplication.
    Keys: ``spearman``, ``n_conditions``, ``spearman_valid``, ``spearman_n``.
    The unchanged n_conditions and valid correlation n are both returned.
    """
    prediction, truth = _prediction_pair(prediction, truth)
    spearman = _spearman_rows(prediction.T, truth.T)
    valid = np.isfinite(spearman)
    n_conditions = np.full(prediction.shape[1], prediction.shape[0], dtype=np.int64)
    return {
        "spearman": spearman,
        "n_conditions": n_conditions,
        "spearman_valid": valid,
        "spearman_n": np.where(valid, n_conditions, 0),
    }


def paired_order_accuracy_by_gene(prediction, truth) -> dict[str, np.ndarray]:
    """Score the order of unordered perturbation pairs separately per gene.

    Only pairs with unequal true responses contribute. Correct predicted order
    scores 1, reversed order 0, and an exact predicted tie 0.5. Ties are exact
    equalities of the supplied finite float64 numbers, not a learned tolerance.
    A gene without a truth-distinguishable pair receives NaN and n_pairs=0.
    A constant predictor with varying truth scores 0.5, NOT a correlation of 0.

    Returns arrays ``accuracy`` and ``n_pairs`` of length n_genes. Pair counts
    are denominators, NOT independent biological sample sizes. The caller's
    fixed-context/exposure/unique-perturbation contract is the same as for
    gene_metrics. Runtime is O(n_genes * n_conditions**2); pair-index storage
    is O(n_conditions**2), without an n_conditions**2 * n_genes allocation.
    """
    prediction, truth = _prediction_pair(prediction, truth)
    n_conditions, n_genes = prediction.shape
    first, second = np.triu_indices(n_conditions, k=1)
    accuracy = np.full(n_genes, np.nan, dtype=np.float64)
    n_pairs = np.zeros(n_genes, dtype=np.int64)
    for gene in range(n_genes):
        true_first, true_second = truth[first, gene], truth[second, gene]
        eligible = true_first != true_second
        count = np.count_nonzero(eligible)
        n_pairs[gene] = count
        if not count:
            continue
        # Direct comparisons avoid overflow in pairwise differences.
        pred_first = prediction[first[eligible], gene]
        pred_second = prediction[second[eligible], gene]
        correct = (
            (pred_first > pred_second)
            == (true_first[eligible] > true_second[eligible])
        )
        tied = pred_first == pred_second
        accuracy[gene] = (
            np.count_nonzero(correct & ~tied) + 0.5 * np.count_nonzero(tied)
        ) / count
    return {"accuracy": accuracy, "n_pairs": n_pairs}


def paired_metric_difference(metric, baseline) -> np.ndarray:
    """Return metric minus baseline elementwise on an explicitly shared cohort.

    Inputs must be identically shaped, non-empty, finite arrays. There is no
    broadcasting, averaging, reordering or model-specific missing-value drop.
    For error metrics a negative difference is improvement; for utility scores
    a positive difference is improvement. Undefined correlations need explicit
    externally reported eligibility/missingness handling before calling this
    helper, and cannot be silently converted to 0. Shape equality alone does
    not establish matching identities; the caller must enforce shared IDs.
    """
    metric = _real_array(metric, "metric")
    baseline = _real_array(baseline, "baseline")
    if metric.shape != baseline.shape:
        raise ValueError("metric and baseline must have identical shapes")
    if metric.ndim == 0 or metric.size == 0:
        raise ValueError("metric and baseline must be non-empty arrays")
    with np.errstate(over="ignore", invalid="ignore"):
        difference = metric - baseline
    if not np.isfinite(difference).all():
        raise ValueError("paired metric difference produced non-finite values")
    return difference


def shared_train_mean_invariance(
    prediction, truth, train_mean, *, rtol: float = 1e-10, atol: float = 1e-12
) -> dict:
    """Audit invariants after subtracting one training gene-mean from both sides.

    train_mean is an externally supplied, finite vector of length n_genes; the
    caller must prove it was fitted on training data only. The helper never
    estimates a mean from evaluation truth. It returns raw and residual scores
    plus elementwise checks for MAE, RMSE, gene Spearman and pair ordering.
    Across-gene condition Spearman is deliberately NOT declared invariant.

    Mathematically shared shifts cancel in errors and preserve within-gene
    order. Extreme floating-point shifts can erase ranks; checks expose such
    numerical failures and do not automatically repair or suppress them.
    Equal undefined scores count as invariant, not as valid correlations.
    """
    prediction, truth = _prediction_pair(prediction, truth)
    train_mean = _real_array(train_mean, "train_mean")
    if train_mean.ndim != 1 or train_mean.shape[0] != prediction.shape[1]:
        raise ValueError("train_mean must be a vector with one value per gene")
    for name, tolerance in (("rtol", rtol), ("atol", atol)):
        if np.ndim(tolerance) != 0 or np.iscomplexobj(tolerance):
            raise ValueError(f"{name} must be a finite non-negative scalar")
        try:
            number = float(tolerance)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a finite non-negative scalar") from exc
        if not np.isfinite(number) or number < 0:
            raise ValueError(f"{name} must be a finite non-negative scalar")
    rtol, atol = float(rtol), float(atol)
    with np.errstate(over="ignore", invalid="ignore"):
        residual_prediction = prediction - train_mean[None, :]
        residual_truth = truth - train_mean[None, :]
    residual_prediction, residual_truth = _prediction_pair(
        residual_prediction, residual_truth
    )
    raw = {
        "condition": condition_metrics(prediction, truth),
        "gene": gene_metrics(prediction, truth),
        "order": paired_order_accuracy_by_gene(prediction, truth),
    }
    residual = {
        "condition": condition_metrics(residual_prediction, residual_truth),
        "gene": gene_metrics(residual_prediction, residual_truth),
        "order": paired_order_accuracy_by_gene(residual_prediction, residual_truth),
    }
    checks = {
        name: np.isclose(raw[group][key], residual[group][key],
                         rtol=rtol, atol=atol, equal_nan=True)
        for name, group, key in (
            ("mae_invariant", "condition", "mae"),
            ("rmse_invariant", "condition", "rmse"),
            ("gene_spearman_invariant", "gene", "spearman"),
            ("gene_order_accuracy_invariant", "order", "accuracy"),
        )
    }
    checks["gene_order_n_pairs_invariant"] = (
        raw["order"]["n_pairs"] == residual["order"]["n_pairs"]
    )
    return {"raw": raw, "residual": residual, "checks": checks}
