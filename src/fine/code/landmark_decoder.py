"""Train-only landmark features and a fixed-alpha NumPy ridge decoder.

Rows are conditions and columns are an already fixed set of training landmarks.
The caller must guarantee landmark identity/order through its manifest, as
numeric matrices alone cannot verify column identity. This module does not
select landmarks, split data, tune alpha, or inspect test outcomes.
"""

from __future__ import annotations

import numpy as np


def _matrix(value, name: str, *, allow_empty_rows: bool = True) -> np.ndarray:
    """Validate a real, finite 2-D matrix without modifying the caller's input."""
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must contain real values, not complex values")
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a numeric matrix") from exc
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    if array.shape[1] == 0:
        raise ValueError(f"{name} must have at least one column")
    if not allow_empty_rows and array.shape[0] == 0:
        raise ValueError(f"{name} must have at least one training row")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def cosine_similarity(samples, landmarks) -> np.ndarray:
    """Return pairwise cosine similarities, with zero-vector similarity set to 0.

    Inputs have shapes (n_conditions, d) and (n_landmarks, d). Each nonzero
    vector is normalized independently; no sample/batch statistics are fitted.
    Scaling before the norm avoids overflow for very large finite components.
    """
    samples = _matrix(samples, "samples")
    landmarks = _matrix(landmarks, "landmarks", allow_empty_rows=False)
    if samples.shape[1] != landmarks.shape[1]:
        raise ValueError("samples and landmarks must have the same feature dimension")

    def unit_rows(array: np.ndarray) -> np.ndarray:
        maximum = np.max(np.abs(array), axis=1, keepdims=True)
        scaled = np.divide(array, maximum, out=np.zeros_like(array), where=maximum > 0)
        norm = np.linalg.norm(scaled, axis=1, keepdims=True)
        return np.divide(scaled, norm, out=np.zeros_like(scaled), where=norm > 0)

    return np.clip(unit_rows(samples) @ unit_rows(landmarks).T, -1.0, 1.0)


def build_landmark_features(
    perturbation_kernel, context_kernel=None, *, mode: str = "perturbation_only"
) -> np.ndarray:
    """Build perturbation-only features or a product-kernel interaction.

    For ``interaction``, feature (i, j) is
    K_perturbation(condition_i, landmark_j) *
    K_context(condition_i, landmark_j). Both matrices must describe identical
    condition rows and landmark columns in identical order; the caller's
    manifest must establish those identities. This function checks shape and
    finite values, not semantic order. Kernels are never fitted here.
    """
    if mode not in {"perturbation_only", "interaction"}:
        raise ValueError("mode must be 'perturbation_only' or 'interaction'")
    perturbation = _matrix(perturbation_kernel, "perturbation_kernel")
    if mode == "perturbation_only":
        if context_kernel is not None:
            raise ValueError("perturbation_only mode does not accept a context_kernel")
        return perturbation.copy()
    if context_kernel is None:
        raise ValueError("interaction mode requires a context_kernel")
    context = _matrix(context_kernel, "context_kernel")
    if perturbation.shape != context.shape:
        raise ValueError("perturbation and context kernels must have identical shapes")
    with np.errstate(over="ignore", invalid="ignore"):
        features = perturbation * context
    if not np.isfinite(features).all():
        raise ValueError("kernel interaction produced non-finite values")
    return features


class LandmarkRidge:
    """Ridge regression on fixed landmark features, fitted on training rows only.

    ``alpha`` must be finite and strictly positive. The fitted objective is
    ||Z @ coef - (Y - training_target_mean)||_F**2 + alpha * ||coef||_F**2,
    where Z uses training column means and population standard deviations
    (ddof=0). Columns with std < 1e-8 receive scale 1, preventing amplification
    of near-constant kernel noise. This fixed numerical threshold follows the
    legacy scaling rule; it is not selected from prediction results. The target
    mean acts as an unpenalized intercept. A one-dimensional target preserves
    1-D predictions.

    A successful fit is immutable through the API: fitting the same instance
    again raises RuntimeError. Use a separate instance for each training fold.
    Only the caller can establish that fit rows/landmarks are training-only.
    """

    def __init__(self, alpha: float):
        if np.iscomplexobj(alpha) or np.ndim(alpha) != 0:
            raise ValueError("alpha must be a finite, strictly positive scalar")
        try:
            alpha = float(alpha)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("alpha must be a finite, strictly positive scalar") from exc
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be a finite, strictly positive scalar")
        self._alpha = alpha
        self._fitted = False

    @property
    def alpha(self) -> float:
        """The externally specified penalty; no tuning is performed."""
        return self._alpha

    def fit(self, train_features, train_y) -> "LandmarkRidge":
        """Fit using (n_train, n_landmarks) features and matched training targets."""
        if self._fitted:
            raise RuntimeError("This instance is already fitted; create a new instance")
        features = _matrix(train_features, "train_features", allow_empty_rows=False)
        if np.iscomplexobj(train_y):
            raise ValueError("train_y must contain real values, not complex values")
        try:
            targets = np.asarray(train_y, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("train_y must be a numeric array") from exc
        target_was_1d = targets.ndim == 1
        if target_was_1d:
            targets = targets[:, None]
        targets = _matrix(targets, "train_y", allow_empty_rows=False)
        if features.shape[0] != targets.shape[0]:
            raise ValueError("train_features and train_y must have the same row count")

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            feature_mean = features.mean(axis=0)
            feature_scale = features.std(axis=0, ddof=0)
            feature_scale = np.where(feature_scale < 1e-8, 1.0, feature_scale)
            target_mean = targets.mean(axis=0)
            standardized = (features - feature_mean) / feature_scale
            centered_targets = targets - target_mean
            gram = standardized.T @ standardized
            gram.flat[:: gram.shape[0] + 1] += self.alpha
            rhs = standardized.T @ centered_targets
        if not all(np.isfinite(a).all() for a in (
            feature_mean, feature_scale, target_mean, standardized,
            centered_targets, gram, rhs,
        )):
            raise ValueError("training transformation produced non-finite values")
        try:
            coefficient = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError as exc:
            raise ValueError("ridge system could not be solved") from exc
        if not np.isfinite(coefficient).all():
            raise ValueError("ridge solution produced non-finite values")

        # Commit state only after the complete fit has succeeded. These arrays
        # are new allocations, so later caller mutations cannot alter the fit.
        for array in (feature_mean, feature_scale, target_mean, coefficient):
            array.setflags(write=False)
        self.feature_mean_ = feature_mean
        self.feature_scale_ = feature_scale
        self.target_mean_ = target_mean
        self.coef_ = coefficient
        self.n_features_in_ = features.shape[1]
        self.n_outputs_ = targets.shape[1]
        self._target_was_1d = target_was_1d
        self._fitted = True
        return self

    def predict(self, features) -> np.ndarray:
        """Predict using the saved training transform, without fitting test data."""
        if not self._fitted:
            raise RuntimeError("fit must be called before predict")
        features = _matrix(features, "features")
        if features.shape[1] != self.n_features_in_:
            raise ValueError("prediction feature count differs from the fitted landmarks")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            standardized = (features - self.feature_mean_) / self.feature_scale_
            prediction = standardized @ self.coef_ + self.target_mean_
        if not np.isfinite(prediction).all():
            raise ValueError("prediction produced non-finite values")
        return prediction[:, 0] if self._target_was_1d else prediction
