#!/usr/bin/env python3
"""Frozen, descriptive three-fold sci-Plex LOCO technical loop.

No target response is passed to feature fitting or fit_predict_fold. Input
NPZ stores all responses in one compressed array (not an OS-isolated scorer),
but evaluation slices are accessed only after prediction artifacts are saved.
The restricted cohort is rescored from the SAME predictions, never refitted.
This is a two-encoder plus Morgan technical loop, not the six-model benchmark.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import sys
import time

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
from landmark_decoder import LandmarkRidge, build_landmark_features, cosine_similarity
from prediction_metrics import condition_metrics, gene_metrics, paired_order_accuracy_by_gene

ROOT = Path(__file__).resolve().parents[1]
BASE_METHODS = ["zero", "source_mean", "source_median", "same_drug_dose_mean"]
PERTURBATION_METHODS = ["identity_exposure", "tfidf_entity_exposure", "bge_entity_exposure", "sapbert_entity_exposure", "morgan_entity_exposure"]
DIRECT_METHODS = ["tfidf_complete_metadata", "bge_complete_metadata", "sapbert_complete_metadata"]
METHODS = BASE_METHODS + [key + suffix for key in PERTURBATION_METHODS for suffix in ["", "__control_state"]] + DIRECT_METHODS


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def read_tsv(path):
    return pd.read_csv(path, sep="\t", keep_default_na=False)


def boolean_mask(series):
    values = series.astype(str).str.lower()
    require(values.isin(["true", "false"]).all(), "eligibility must contain explicit True/False")
    return values.eq("true").to_numpy()


def validate_balanced_cohort(metadata, mask, cell_lines):
    selected = metadata.loc[mask]
    require(selected.atomic_id.is_unique, "duplicate atomic ID")
    require(set(selected.cell_line) == set(cell_lines), "wrong cell lines")
    require(not selected.duplicated(["source_entity_key", "dose_value", "cell_line"]).any(), "duplicate drug-dose-context")
    coverage = selected.groupby(["source_entity_key", "dose_value"]).cell_line.agg(set)
    require(coverage.map(lambda value: value == set(cell_lines)).all(), "cohort not shared drug-dose grid across three contexts")
    require(selected.time.nunique() == 1 and selected.dose_unit.nunique() == 1, "mixed time/exposure unit")


def choose_landmarks(metadata, train_rows, count, seed):
    """One training atom per source drug, then fill by fixed metadata-only hash."""
    train_rows = np.asarray(train_rows, dtype=int)
    require(len(np.unique(train_rows)) == len(train_rows), "duplicate train row")
    training = metadata.iloc[train_rows].copy()
    require(training.atomic_id.is_unique, "duplicate source atomic ID")
    training["_row"] = train_rows
    training["_hash"] = training.atomic_id.map(lambda uid: hashlib.sha256(f"{seed}|{uid}".encode()).hexdigest())
    ordered = training.sort_values(["_hash", "atomic_id"], kind="stable")
    first = ordered.drop_duplicates("source_entity_key", keep="first")
    require(len(first) <= count <= len(training), "landmark count cannot retain every training drug")
    remainder = ordered[~ordered._row.isin(first._row)]
    selected = pd.concat([first, remainder.head(count - len(first))])
    require(len(selected) == count and selected._row.is_unique, "landmark allocation failed")
    return selected._row.to_numpy(dtype=int)


def identity_exposure_features(metadata, train_rows):
    """Training-vocabulary two-hot features; unknown values fail this seen-drug task."""
    drugs = sorted(metadata.iloc[train_rows].source_entity_key.unique())
    doses = sorted(metadata.iloc[train_rows].dose_value.unique())
    drug_map, dose_map = {value: i for i, value in enumerate(drugs)}, {value: i for i, value in enumerate(doses)}
    require(metadata.source_entity_key.isin(drugs).all() and metadata.dose_value.isin(doses).all(), "unseen drug/dose in seen-drug task")
    matrix = np.zeros((len(metadata), len(drugs) + len(doses)), dtype=np.float64)
    matrix[np.arange(len(metadata)), metadata.source_entity_key.map(drug_map).to_numpy()] = 1
    matrix[np.arange(len(metadata)), len(drugs) + metadata.dose_value.map(dose_map).to_numpy()] = 1
    return matrix, {"source_drugs": drugs, "source_doses": [float(value) for value in doses], "vocabulary_fit_rows": metadata.iloc[train_rows].atomic_id.tolist()}


def fit_tfidf_kernel(texts, train_rows, landmark_rows):
    """Only source unique exact texts fit vocabulary/IDF; no dimensional reducer."""
    texts = np.asarray(texts, dtype=str)
    source_texts = sorted(set(texts[np.asarray(train_rows, dtype=int)]))
    vectorizer = TfidfVectorizer(lowercase=True, analyzer="word", ngram_range=(1, 2),
                               min_df=1, max_features=20000, sublinear_tf=True)
    vectorizer.fit(source_texts)
    transformed = normalize(vectorizer.transform(texts), norm="l2", copy=True)
    kernel = (transformed @ transformed[np.asarray(landmark_rows, dtype=int)].T).toarray()
    vocabulary = vectorizer.get_feature_names_out()
    audit = {"n_unique_source_texts": len(source_texts), "n_features": len(vocabulary),
             "source_text_sha256": [hashlib.sha256(value.encode()).hexdigest() for value in source_texts],
             "vocabulary_sha256": hashlib.sha256("\n".join(vocabulary).encode()).hexdigest(),
             "idf_sha256": hashlib.sha256(np.asarray(vectorizer.idf_, dtype="<f8").tobytes()).hexdigest(),
             "zero_vector_rows": int(np.count_nonzero(np.asarray(transformed.getnnz(axis=1)) == 0)),
             "fit_scope": "source unique exact texts only", "svd_used": False}
    return kernel, audit


def same_drug_dose_prediction(metadata, train_rows, test_rows, train_y):
    """Average precisely the two source contexts at each matched drug-dose."""
    source = metadata.iloc[train_rows]
    lookup = {}
    for position, row in enumerate(source.itertuples()):
        lookup.setdefault((row.source_entity_key, row.dose_value), []).append(position)
    result = []
    for row in metadata.iloc[test_rows].itertuples():
        positions = lookup.get((row.source_entity_key, row.dose_value), [])
        require(len(positions) == 2 and source.iloc[positions].cell_line.nunique() == 2, "same-drug baseline requires exactly two source contexts")
        result.append(np.asarray(train_y, dtype=np.float64)[positions].mean(axis=0))
    return np.asarray(result)


def tanimoto_similarity(samples, references):
    """Binary Tanimoto with int64 accumulation; never overflow signed int8.

    Empty-set/empty-set similarity is 1 by convention; actual Morgan inputs
    separately reject all-zero fingerprints so missing structure is not encoded
    as an empty vector.
    """
    samples, references = np.asarray(samples), np.asarray(references)
    require(samples.ndim == references.ndim == 2 and samples.shape[1] == references.shape[1], "fingerprint dimensions differ")
    require(np.isin(samples, [0, 1]).all() and np.isin(references, [0, 1]).all(), "fingerprints must be binary")
    first, second = samples.astype(np.int64), references.astype(np.int64)
    intersection = first @ second.T
    union = first.sum(axis=1)[:, None] + second.sum(axis=1)[None, :] - intersection
    result = np.ones(intersection.shape, dtype=np.float64)
    np.divide(intersection, union, out=result, where=union != 0)
    require(np.isfinite(result).all() and ((result >= 0) & (result <= 1)).all(), "invalid Tanimoto kernel")
    return result


def morgan_exposure_kernel(fingerprints, doses, landmark_rows):
    """Predeclared 0.5*Tanimoto + 0.5*same-dose; no dose hard filtering.

    This additive kernel cannot by itself model drug-by-dose interaction. It is
    an interpretable native-prior baseline, not a claim that chemical structure
    can only support additive dose response.
    """
    fingerprints, doses = np.asarray(fingerprints), np.asarray(doses)
    require(fingerprints.shape[0] == len(doses), "Morgan/dose row mismatch")
    unique, inverse = np.unique(fingerprints, axis=0, return_inverse=True)
    structure = tanimoto_similarity(unique, unique)[inverse[:, None], inverse[np.asarray(landmark_rows)][None, :]]
    same_dose = doses[:, None] == doses[np.asarray(landmark_rows)][None, :]
    return 0.5 * (structure + same_dose.astype(np.float64))


def numerical_rank_audit(matrix):
    """Explicit NumPy-default-equivalent float64 SVD tolerance, not tuned."""
    matrix = np.asarray(matrix, dtype=np.float64)
    values = np.linalg.svd(matrix, compute_uv=False)
    tolerance = float(values.max() * max(matrix.shape) * np.finfo(np.float64).eps) if len(values) else 0.0
    return {"rank": int(np.count_nonzero(values > tolerance)), "tolerance": tolerance,
            "tolerance_rule": "max_singular_value * max(n_rows,n_columns) * float64_epsilon",
            "max_singular_value": float(values.max()) if len(values) else 0.0}


def kernel_diagnostics(features, train_rows, test_rows, fitted_model):
    train, test = features[train_rows], features[test_rows]
    centered_scaled = (train - fitted_model.feature_mean_) / fitted_model.feature_scale_
    return {"train_exact_unique_columns": int(np.unique(train, axis=1).shape[1]),
            "train_exact_unique_rows": int(np.unique(train, axis=0).shape[0]),
            "test_exact_unique_rows": int(np.unique(test, axis=0).shape[0]),
            "train_all_zero_rows": int(np.all(train == 0, axis=1).sum()),
            "test_all_zero_rows": int(np.all(test == 0, axis=1).sum()),
            "raw_train_rank": numerical_rank_audit(train),
            "standardized_train_rank": numerical_rank_audit(centered_scaled),
            "warning": "Equal 256-column budget does not imply equal effective degrees of freedom."}


def fit_predict_fold(metadata, train_rows, test_rows, train_y, text_by_view,
                     embedding_by_model_view, control_state, config):
    """Only train_y enters this function. No held-out effects, scoring or tuning."""
    train_rows, test_rows = np.asarray(train_rows, dtype=int), np.asarray(test_rows, dtype=int)
    require(not set(train_rows) & set(test_rows), "train/test overlap")
    require(len(set(metadata.iloc[train_rows].cell_line)) == 2 and metadata.iloc[test_rows].cell_line.nunique() == 1, "not two-source one-target LOCO")
    require(not set(metadata.iloc[train_rows].cell_line) & set(metadata.iloc[test_rows].cell_line), "target context in training")
    require(np.asarray(train_y).ndim == 2 and np.asarray(train_y).shape[0] == len(train_rows) and np.isfinite(train_y).all(), "invalid training responses")
    landmarks = choose_landmarks(metadata, train_rows, config["n_landmarks"], config["seed"])
    require(set(landmarks) <= set(train_rows), "target landmark leakage")
    n_test, n_genes = len(test_rows), np.asarray(train_y).shape[1]
    predictions = {
        "zero": np.zeros((n_test, n_genes)),
        "source_mean": np.broadcast_to(np.mean(train_y, axis=0, dtype=np.float64), (n_test, n_genes)).copy(),
        "source_median": np.broadcast_to(np.median(np.asarray(train_y, dtype=np.float64), axis=0), (n_test, n_genes)).copy(),
        "same_drug_dose_mean": same_drug_dose_prediction(metadata, train_rows, test_rows, train_y),
    }
    identity, identity_audit = identity_exposure_features(metadata, train_rows)
    kernels = {"identity_exposure": cosine_similarity(identity, identity[landmarks])}
    morgan = embedding_by_model_view[("morgan", "entity")]
    kernels["morgan_entity_exposure"] = morgan_exposure_kernel(morgan, metadata.dose_value.to_numpy(), landmarks)
    tfidf_audits = {}
    for view in ["entity_exposure", "complete_metadata"]:
        kernels["tfidf_" + view], tfidf_audits[view] = fit_tfidf_kernel(text_by_view[view], train_rows, landmarks)
        for display, model in [("bge", "bge_m3"), ("sapbert", "sapbert")]:
            embedding = embedding_by_model_view[(model, view)]
            require(len(embedding) == len(metadata), "embedding/metadata row mismatch")
            kernels[display + "_" + view] = cosine_similarity(embedding, embedding[landmarks])
    require(np.asarray(control_state).shape[0] == len(metadata), "state/metadata row mismatch")
    state_kernel = cosine_similarity(control_state, control_state[landmarks])
    source_contexts = sorted(metadata.iloc[train_rows].cell_line.unique())
    source_context_state = np.asarray([control_state[train_rows[np.flatnonzero(metadata.iloc[train_rows].cell_line.to_numpy() == name)[0]]]
                                       for name in source_contexts], dtype=np.float64)
    state_rank = numerical_rank_audit(source_context_state - source_context_state.mean(axis=0))
    require(state_rank["rank"] <= 1, "two source-context state means centered rank exceeds one")
    state_rank.update({"n_source_contexts": len(source_contexts), "source_contexts": source_contexts,
                       "state_basis_fitted": False, "centered_rank_check_is_diagnostic_not_transform": True})
    model_audits, fitted = {}, {}
    for method in METHODS[len(BASE_METHODS):]:
        has_state = method.endswith("__control_state")
        key = method.removesuffix("__control_state")
        features = build_landmark_features(kernels[key], state_kernel if has_state else None,
                                           mode="interaction" if has_state else "perturbation_only")
        model = LandmarkRidge(config["alpha"]).fit(features[train_rows], train_y)
        predictions[method] = model.predict(features[test_rows])
        fitted[method] = {"feature_mean": model.feature_mean_, "feature_scale": model.feature_scale_,
                          "target_mean": model.target_mean_, "coef": model.coef_}
        model_audits[method] = {"n_landmarks": len(landmarks), "alpha": config["alpha"],
                               "n_parameters": int(model.coef_.size + model.target_mean_.size),
                               "mode": "product_kernel" if has_state else "direct_landmark_kernel",
                               "target_controls_available": has_state,
                               "train_feature_mean_sha256": hashlib.sha256(model.feature_mean_.tobytes()).hexdigest(),
                               "train_feature_scale_sha256": hashlib.sha256(model.feature_scale_.tobytes()).hexdigest(),
                               "kernel_diagnostics": kernel_diagnostics(features, train_rows, test_rows, model)}
    stacked = np.stack([predictions[key] for key in METHODS]).astype(np.float32)
    require(np.isfinite(stacked).all(), "nonfinite prediction; do not silently omit method")
    return stacked, {"landmark_rows": landmarks.tolist(), "landmark_atomic_id": metadata.iloc[landmarks].atomic_id.tolist(),
                     "source_atomic_id": metadata.iloc[train_rows].atomic_id.tolist(),
                     "target_atomic_id": metadata.iloc[test_rows].atomic_id.tolist(),
                     "landmark_unique_drugs": int(metadata.iloc[landmarks].source_entity_key.nunique()),
                     "landmark_dose_distribution": {str(key): int(value) for key, value in metadata.iloc[landmarks].dose_value.value_counts().items()},
                     "landmark_source_context_distribution": {str(key): int(value) for key, value in metadata.iloc[landmarks].cell_line.value_counts().items()},
                     "source_context_state_centered_rank": state_rank,
                     "morgan_prior": {"kernel": "0.5*Tanimoto + 0.5*same_dose", "tanimoto_accumulation": "int64",
                                      "chemical_structures_are_additional_information": True,
                                      "scope": "additive structure/dose prior, no isolated drug-by-dose interaction",
                                      "dose_hard_filter": False, "weights_fitted": False},
                     "baseline_kernel_diagnostics": "Not applicable: four reference methods do not fit a landmark kernel.",
                     "identity": identity_audit, "tfidf": tfidf_audits, "models": model_audits,
                     "test_effects_passed_to_fit": False, "hyperparameter_tuning": False}, fitted


def track(method):
    if method in BASE_METHODS:
        return "shared_reference"
    if method.startswith("morgan_"):
        return "structure_prior_target_controls_available" if method.endswith("__control_state") else "structure_prior_metadata_only"
    return "target_controls_available" if method.endswith("__control_state") else "metadata_only"


def score_fold(predictions, truth, metadata, gene_panel, cohort, min_gene_n):
    """Fixed predictions and a single held-out context; no model selection."""
    require(predictions.shape == (len(METHODS), len(metadata), len(gene_panel)), "prediction axes mismatch")
    require(truth.shape == (len(metadata), len(gene_panel)), "truth axes mismatch")
    require(metadata.cell_line.nunique() == 1 and metadata.time.nunique() == 1, "gene scoring must remain in fixed context and time")
    baseline_index = METHODS.index("same_drug_dose_mean")
    baseline = condition_metrics(predictions[baseline_index], truth)
    condition_rows, gene_rows, summary_rows = [], [], []
    heldout = metadata.cell_line.iloc[0]
    dose_rows, baseline_gene_by_dose = {}, {}
    for dose in sorted(metadata.dose_value.unique()):
        rows = np.flatnonzero(metadata.dose_value.to_numpy() == dose)
        require(metadata.iloc[rows].source_entity_key.is_unique, "duplicate drug within fixed-dose gene scoring")
        dose_rows[dose] = rows
        if len(rows) >= min_gene_n:
            baseline_gene_by_dose[dose] = (gene_metrics(predictions[baseline_index, rows], truth[rows]),
                                           paired_order_accuracy_by_gene(predictions[baseline_index, rows], truth[rows]))
    for method_index, method in enumerate(METHODS):
        scores = condition_metrics(predictions[method_index], truth)
        conditions = metadata[["atomic_id", "source_entity_key", "cell_line", "dose_value", "time"]].reset_index(drop=True).copy()
        conditions["method"], conditions["cohort"], conditions["track"] = method, cohort, track(method)
        for name, value in scores.items():
            conditions[name] = value
        conditions["mae_improvement_vs_same_drug"] = baseline["mae"] - scores["mae"]
        conditions["rmse_improvement_vs_same_drug"] = baseline["rmse"] - scores["rmse"]
        conditions["spearman_improvement_vs_same_drug"] = scores["spearman"] - baseline["spearman"]
        condition_rows.append(conditions)
        dose_summaries = []
        for dose, rows in dose_rows.items():
            enough = len(rows) >= min_gene_n
            if enough:
                baseline_genes, baseline_order = baseline_gene_by_dose[dose]
                if method_index == baseline_index:
                    genes, order = baseline_genes, baseline_order
                else:
                    genes = gene_metrics(predictions[method_index, rows], truth[rows])
                    order = paired_order_accuracy_by_gene(predictions[method_index, rows], truth[rows])
            else:
                genes = {"spearman": np.full(len(gene_panel), np.nan), "spearman_valid": np.zeros(len(gene_panel), bool),
                         "spearman_n": np.zeros(len(gene_panel), int), "n_conditions": np.full(len(gene_panel), len(rows))}
                order = {"accuracy": np.full(len(gene_panel), np.nan), "n_pairs": np.zeros(len(gene_panel), int)}
                baseline_genes, baseline_order = genes, order
            per_gene = gene_panel[["source_feature_row", "original_ensembl_id", "gene_symbol"]].reset_index(drop=True).copy()
            per_gene["method"], per_gene["cohort"], per_gene["track"] = method, cohort, track(method)
            per_gene["heldout_cell_line"], per_gene["dose_value"], per_gene["eligible_min_n"] = heldout, dose, enough
            for name, value in genes.items():
                per_gene[name] = value
            per_gene["pair_order_accuracy"], per_gene["n_pairs"] = order["accuracy"], order["n_pairs"]
            per_gene["spearman_improvement_vs_same_drug"] = genes["spearman"] - baseline_genes["spearman"]
            per_gene["order_improvement_vs_same_drug"] = order["accuracy"] - baseline_order["accuracy"]
            gene_rows.append(per_gene)
            dose_summaries.append({"n_genes": len(gene_panel), "n_valid": int(genes["spearman_valid"].sum()),
                                   "mean_rho": finite_mean(genes["spearman"]), "mean_order": finite_mean(order["accuracy"])})
        summary_rows.append({"heldout_cell_line": heldout, "method": method, "cohort": cohort, "track": track(method),
                             "n_query_atoms": len(metadata), "n_unique_drugs": metadata.source_entity_key.nunique(),
                             "n_genes": len(gene_panel), "mae_mean": float(scores["mae"].mean()),
                             "rmse_mean": float(scores["rmse"].mean()), "condition_spearman_mean": finite_mean(scores["spearman"]),
                             "condition_spearman_valid_n": int(scores["spearman_valid"].sum()),
                             "mae_improvement_vs_same_drug": float(conditions.mae_improvement_vs_same_drug.mean()),
                             "condition_spearman_paired_valid_n": int(np.isfinite(conditions.spearman_improvement_vs_same_drug).sum()),
                             "condition_spearman_improvement_vs_same_drug": finite_mean(conditions.spearman_improvement_vs_same_drug),
                             "gene_spearman_equal_dose_mean": finite_mean([row["mean_rho"] for row in dose_summaries]),
                             "gene_order_equal_dose_mean": finite_mean([row["mean_order"] for row in dose_summaries]),
                             "gene_score_valid_n": sum(row["n_valid"] for row in dose_summaries),
                             "gene_score_total_n": sum(row["n_genes"] for row in dose_summaries),
                             "fit_cohort": "primary", "restricted_evaluation_not_refit": cohort != "primary"})
    return pd.concat(condition_rows, ignore_index=True), pd.concat(gene_rows, ignore_index=True), pd.DataFrame(summary_rows)


def finite_mean(values):
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    return float(values[valid].mean()) if valid.any() else np.nan


def save_npz(path, **values):
    with Path(path).open("xb") as handle:
        np.savez_compressed(handle, **values)


def load_representations(metadata, clean_root, embedding_root):
    encoding_audit = json.loads((embedding_root / "audit.json").read_text())
    require(encoding_audit["status"] == "PASS", "embedding audit not PASS")
    encoded_source_hashes = {row["path"]: row["sha256"] for row in encoding_audit["sources"]}
    for name in ["config.json", "row_to_text_registry.tsv", "unique_texts.tsv"]:
        path = clean_root / name
        require(str(path) in encoded_source_hashes and sha256(path) == encoded_source_hashes[str(path)], "text source changed since encoding: " + name)
    registry = read_tsv(clean_root / "row_to_text_registry.tsv")
    registry = registry[registry.variant == "source_name"]
    text_table = read_tsv(clean_root / "unique_texts.tsv").set_index("text_id")
    text_by_view, text_ids = {}, {}
    for view in ["entity_exposure", "complete_metadata"]:
        mapping = registry[registry.view == view]
        require(mapping.atomic_id.is_unique, "ambiguous atom/view mapping")
        mapping = mapping.set_index("atomic_id").loc[metadata.atomic_id]
        require((mapping.prompt_sha256.to_numpy() == text_table.loc[mapping.text_id].prompt_sha256.to_numpy()).all(), "registry/text hash mismatch")
        text_ids[view] = mapping.text_id.to_numpy(dtype=str)
        text_by_view[view] = text_table.loc[mapping.text_id].prompt_text.to_numpy(dtype=str)
    embeddings = {}
    for model in ["bge_m3", "sapbert"]:
        path = embedding_root / f"{model}__source_name.npz"
        output = next(item for item in encoding_audit["outputs"] if item["model_key"] == model)
        require(sha256(path) == output["sha256"], "embedding hash changed")
        with np.load(path, allow_pickle=False) as archive:
            ids, x, digests = archive["text_id"], archive["X"], archive["prompt_sha256"]
        require(len(set(ids)) == len(ids) and np.isfinite(x).all(), "invalid embedding archive")
        lookup = pd.Index(ids)
        for view in text_ids:
            rows = lookup.get_indexer(text_ids[view])
            require((rows >= 0).all(), "clean embedding absent")
            require(np.array_equal(digests[rows], text_table.loc[text_ids[view]].prompt_sha256.to_numpy(dtype=str)), "embedding exact-text mismatch")
            embeddings[(model, view)] = x[rows]
    return text_by_view, embeddings


def load_morgan(metadata, morgan_root):
    audit = json.loads((morgan_root / "audit.json").read_text())
    require(audit["status"] == "PASS", "Morgan audit not PASS")
    path = morgan_root / "arrays.npz"
    require(sha256(path) == audit["output_sha256"]["arrays.npz"], "Morgan fingerprint hash mismatch")
    with np.load(path, allow_pickle=False) as archive:
        entity_ids, fingerprints = archive["source_entity_key"], archive["fingerprints"]
    require(len(entity_ids) == 188 and len(set(entity_ids)) == 188, "Morgan requires 188 unique source entities")
    require(set(entity_ids) == set(metadata.source_entity_key), "Morgan/source entity coverage mismatch")
    require(fingerprints.dtype == np.uint8 and fingerprints.shape == (188, 2048), "Morgan fingerprint format mismatch")
    require(np.isin(fingerprints, [0, 1]).all() and np.all(fingerprints.sum(axis=1) > 0), "Morgan missing/nonbinary fingerprint")
    rows = pd.Index(entity_ids).get_indexer(metadata.source_entity_key)
    require((rows >= 0).all(), "Morgan unresolved entity")
    return fingerprints[rows]


def run(config_path):
    start = time.perf_counter()
    config = json.loads(Path(config_path).read_text())
    require(config["alpha"] == 10 and config["n_landmarks"] == 256 and config["seed"] == 20260914, "frozen prediction parameters changed")
    require(config["gene_min_n"] == 20 and config["expected_genes_per_fold"] == 3000, "frozen scoring contract changed")
    output = Path(config["output_root"])
    output.mkdir(parents=True, exist_ok=False)
    effects_root, clean_root, embedding_root, morgan_root = [Path(config[name]) for name in ["effects_root", "clean_views_root", "embeddings_root", "morgan_root"]]
    input_paths = [Path(config_path), effects_root / "audit.json", effects_root / "atomic_index.tsv", effects_root / "arrays.npz", effects_root / "fold_gene_panels.tsv",
                   clean_root / "config.json", clean_root / "row_to_text_registry.tsv", clean_root / "unique_texts.tsv", embedding_root / "audit.json",
                   embedding_root / "bge_m3__source_name.npz", embedding_root / "sapbert__source_name.npz", morgan_root / "audit.json", morgan_root / "arrays.npz"]
    input_paths.extend(ROOT / "code" / name for name in ["run_sciplex_prediction.py", "landmark_decoder.py", "prediction_metrics.py"])
    try:
        effect_audit = json.loads((effects_root / "audit.json").read_text())
        require(effect_audit["status"] == "PASS", "effects not PASS; refuse prediction")
        for name in ["atomic_index.tsv", "arrays.npz", "fold_gene_panels.tsv"]:
            require(name in effect_audit["output_sha256"] and sha256(effects_root / name) == effect_audit["output_sha256"][name], "effect artifact hash mismatch: " + name)
        input_manifest = [{"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size} for path in input_paths]
        write_json(output / "input_manifest.json", input_manifest)
        metadata = read_tsv(effects_root / "atomic_index.tsv")
        require(metadata.atomic_id.is_unique and metadata.atomic_id.tolist() == sorted(metadata.atomic_id), "atomic index must be unique and sorted")
        primary, restricted = boolean_mask(metadata.main_eligible), boolean_mask(metadata.sensitivity_eligible)
        require(len(metadata) == 2256 and primary.sum() == 2250 and restricted.sum() == 2202, "frozen cohort counts changed")
        require((~restricted | primary).all(), "restricted cohort not subset of primary")
        cell_lines = ["A549", "K562", "MCF7"]
        validate_balanced_cohort(metadata, primary, cell_lines)
        validate_balanced_cohort(metadata, restricted, cell_lines)
        panels = read_tsv(effects_root / "fold_gene_panels.tsv")
        with np.load(effects_root / "arrays.npz", allow_pickle=False) as arrays:
            require(np.array_equal(arrays["atomic_id"], metadata.atomic_id.to_numpy(dtype=str)), "response/metadata ID order mismatch")
            require(arrays["cell_line"].tolist() == cell_lines, "control-state order mismatch")
            all_effects, all_state, feature_rows = arrays["effect_log2fc"], arrays["control_state_A"], arrays["source_feature_row"]
        require(all_effects.shape == (len(metadata), len(feature_rows)) and all_state.shape == (3, len(feature_rows)), "effect/state/gene shape mismatch")
        require(np.isfinite(all_effects).all() and np.isfinite(all_state).all(), "nonfinite effects/state")
        text_by_view, embeddings = load_representations(metadata, clean_root, embedding_root)
        embeddings[("morgan", "entity")] = load_morgan(metadata, morgan_root)
        summaries, fold_audits = [], []
        for heldout in cell_lines:
            fold_start = time.perf_counter()
            print("FOLD_START", heldout, flush=True)
            fold_output = output / heldout
            fold_output.mkdir()
            train_rows = np.flatnonzero(primary & metadata.cell_line.ne(heldout).to_numpy())
            test_rows = np.flatnonzero(primary & metadata.cell_line.eq(heldout).to_numpy())
            require(len(train_rows) == 1500 and len(test_rows) == 750, "LOCO fold count mismatch")
            gene_panel = panels[panels.heldout_cell_line == heldout].sort_values("rank").reset_index(drop=True)
            require(len(gene_panel) == config["expected_genes_per_fold"] and gene_panel.source_feature_row.is_unique and gene_panel.union_column.is_unique, "gene panel incomplete/duplicate")
            columns = gene_panel.union_column.to_numpy(dtype=int)
            require(np.array_equal(feature_rows[columns], gene_panel.source_feature_row.to_numpy()), "fold gene axis mismatch")
            state_by_atom = all_state[pd.Index(cell_lines).get_indexer(metadata.cell_line)][:, columns]
            # Only this training slice is passed to prediction/feature fitting.
            predictions, fold_audit, fitted = fit_predict_fold(metadata, train_rows, test_rows,
                all_effects[np.ix_(train_rows, columns)], text_by_view, embeddings, state_by_atom, config)
            uid, gene_ids = metadata.iloc[test_rows].atomic_id.to_numpy(dtype=str), gene_panel.original_ensembl_id.to_numpy(dtype=str)
            prediction_path = fold_output / "frozen_predictions.npz"
            save_npz(prediction_path, predictions=predictions, method=np.asarray(METHODS, dtype=str),
                     atomic_id=uid, source_feature_row=gene_panel.source_feature_row.to_numpy(), original_ensembl_id=gene_ids)
            prediction_hash = sha256(prediction_path)
            params = {method + "__" + key: value for method, data in fitted.items() for key, value in data.items()}
            save_npz(fold_output / "fitted_decoder_parameters.npz", **params)
            gene_panel.to_csv(fold_output / "gene_panel.tsv", sep="\t", index=False)
            metadata.iloc[train_rows].to_csv(fold_output / "source_atoms.tsv", sep="\t", index=False)
            metadata.iloc[fold_audit["landmark_rows"]].to_csv(fold_output / "landmarks.tsv", sep="\t", index=False)
            # Target truth is sliced only after all 17 prediction outputs are frozen.
            truth = all_effects[np.ix_(test_rows, columns)]
            save_npz(fold_output / "evaluation_truth.npz", truth=truth, atomic_id=uid, original_ensembl_id=gene_ids,
                     source_feature_row=gene_panel.source_feature_row.to_numpy(), sensitivity_eligible=restricted[test_rows])
            for cohort, subset in [("primary", np.ones(len(test_rows), dtype=bool)), ("restricted_evaluation", restricted[test_rows])]:
                print("SCORING", heldout, cohort, int(subset.sum()), flush=True)
                condition, gene, summary = score_fold(predictions[:, subset], truth[subset], metadata.iloc[test_rows[subset]].reset_index(drop=True), gene_panel, cohort, config["gene_min_n"])
                condition.to_csv(fold_output / f"{cohort}_condition_metrics.tsv", sep="\t", index=False, na_rep="NA")
                gene.to_csv(fold_output / f"{cohort}_gene_metrics.tsv", sep="\t", index=False, na_rep="NA")
                summaries.append(summary)
            require(sha256(prediction_path) == prediction_hash, "scoring modified frozen predictions")
            fold_audit.update({"status": "PASS", "heldout_cell_line": heldout, "prediction_sha256": prediction_hash,
                               "config_sha256": next(row["sha256"] for row in input_manifest if row["path"] == str(config_path)),
                               "input_manifest_sha256": sha256(output / "input_manifest.json"),
                               "upstream_effect_output_hashes_verified": True, "encoded_text_source_hashes_verified": True,
                               "gene_panel_source": "source controls only, frozen by effects stage", "n_genes": len(columns),
                               "primary_test_n": len(test_rows), "restricted_test_n": int(restricted[test_rows].sum()),
                               "restricted_evaluation_not_refit": True, "seconds": time.perf_counter() - fold_start})
            write_json(fold_output / "audit.json", fold_audit)
            fold_audits.append(fold_audit)
            print("FOLD_PASS", heldout, "seconds", fold_audit["seconds"], flush=True)
        summary = pd.concat(summaries, ignore_index=True)
        summary.to_csv(output / "fold_summary.tsv", sep="\t", index=False, na_rep="NA")
        numerical = ["mae_mean", "rmse_mean", "condition_spearman_mean", "mae_improvement_vs_same_drug", "gene_spearman_equal_dose_mean", "gene_order_equal_dose_mean"]
        macro = summary.groupby(["cohort", "method", "track"], sort=False)[numerical].mean().reset_index()
        valid_folds = summary.groupby(["cohort", "method", "track"], sort=False)[numerical].count().add_suffix("__valid_folds").reset_index()
        macro = macro.merge(valid_folds, on=["cohort", "method", "track"], validate="one_to_one")
        macro["n_folds"] = 3
        macro["n_independent_studies"] = 1
        macro["interpretation"] = "descriptive_equal_context_macro_only_no_p_values"
        macro.to_csv(output / "descriptive_macro_summary.tsv", sep="\t", index=False, na_rep="NA")
        for record in input_manifest:
            require(sha256(record["path"]) == record["sha256"], "input mutated during prediction: " + record["path"])
        audit = {"status": "PASS", "created_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
                 "seconds": time.perf_counter() - start, "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                 "config": config, "method_order": METHODS, "n_independent_studies": 1, "n_context_folds": 3,
                 "package_versions": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]},
                 "status_scope": "two-encoder plus Morgan frozen descriptive technical loop, not completed six-model benchmark",
                 "target_effects_used_in_fit": False, "target_control_track_explicit": True, "input_hashes_unchanged": True,
                 "source_scripts": [{"path": str(ROOT / "code" / name), "sha256": sha256(ROOT / "code" / name)} for name in ["run_sciplex_prediction.py", "landmark_decoder.py", "prediction_metrics.py"]],
                 "folds": [{key: row[key] for key in ["heldout_cell_line", "status", "primary_test_n", "restricted_test_n", "seconds"]} for row in fold_audits]}
        write_json(output / "audit.json", audit)
        print("PREDICTION_LOOP_PASS", audit["seconds"], flush=True)
    except Exception as exc:
        write_json(output / "failure.json", {"status": "FAILED", "error": repr(exc), "partial_results_not_complete": True})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    with threadpool_limits(limits=4):
        run(arguments.config)
