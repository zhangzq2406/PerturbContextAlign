#!/usr/bin/env python3
"""Fixed six-fold within-type donor prediction; no tuning or raw X access.

Only generic decoder/metric helpers are imported. Each fit receives source Y
only. The compressed input contains all effects, so this is logical slicing,
not OS-level isolation. ALL six folds are sealed before query truth is sliced
for scoring. State_A is used directly on the released log-normalized scale.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits
from landmark_decoder import LandmarkRidge, build_landmark_features, cosine_similarity
from prediction_metrics import condition_metrics, gene_metrics, paired_order_accuracy_by_gene

ROOT = Path(__file__).resolve().parents[1]
MODEL_PREFIXES = {"bge_m3": "bge", "sapbert": "sapbert", "qwen3_0_6b": "qwen3",
                  "biomedbert": "biomedbert", "medcpt_article": "medcpt_article", "medcpt_query": "medcpt_query"}
BASE_METHODS = ["zero", "source_mean", "source_median", "same_drug_source_mean"]
KERNEL_METHODS = ["identity_exposure", "tfidf_entity_exposure", "morgan_entity_exposure"] + [p + "_entity_exposure" for p in MODEL_PREFIXES.values()]
METHODS = BASE_METHODS + [key + suffix for key in KERNEL_METHODS for suffix in ("", "__control_state")]
METRIC_COLUMNS = ["mae_mean", "rmse_mean", "condition_spearman_mean", "mae_improvement_vs_same_drug",
                  "rmse_improvement_vs_same_drug", "condition_spearman_improvement_vs_same_drug",
                  "gene_spearman_mean", "gene_pair_order_mean", "gene_spearman_improvement_vs_same_drug",
                  "gene_order_improvement_vs_same_drug"]


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
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def write_tsv(path, frame):
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, sep="\t", index=False, na_rep="NA", lineterminator="\n")


def save_npz(path, **arrays):
    require(all(np.asarray(v).dtype.kind != "O" for v in arrays.values()), "object arrays forbidden")
    with Path(path).open("xb") as handle:
        np.savez_compressed(handle, **arrays)


def decimal_text(value):
    value = Decimal(str(value))
    require(value.is_finite(), "nonfinite exposure")
    return format(value.normalize(), "f")


def track(method):
    if method in BASE_METHODS:
        return "shared_reference"
    stem = "structure_prior_" if method.startswith("morgan_") else ""
    return stem + ("target_controls_available" if method.endswith("__control_state") else "metadata_only")


def validate_fold(metadata, source, query, source_y):
    source, query = np.asarray(source, int), np.asarray(query, int)
    require(metadata.atomic_id.is_unique and len(set(source)) == len(source) and len(set(query)) == len(query), "duplicate fold IDs")
    require(len(source) > 0 and len(query) > 0 and not set(source) & set(query), "empty/overlapping fold")
    require(set(source) | set(query) == set(range(len(metadata))), "fold does not cover fixed-type metadata")
    require(metadata.cell_type.nunique() == 1 and set(metadata.timepoint_hr.map(decimal_text)) == {"24"}, "mixed type/time")
    s, q = metadata.iloc[source], metadata.iloc[query]
    require(s.donor_id.nunique() == 2 and q.donor_id.nunique() == 1, "requires two source donors and one query donor")
    require(not set(s.donor_id) & set(q.donor_id), "query donor in training")
    require(np.asarray(source_y).ndim == 2 and np.asarray(source_y).shape[0] == len(source) and np.isfinite(source_y).all(), "invalid source responses")
    for part in [s, q]:
        require(not part.duplicated(["sm_lincs_id", "dose_key", "time_key", "donor_id"]).any(), "duplicate compound/exposure/donor")
    source_keys = s.groupby(["sm_lincs_id", "dose_key", "time_key"]).donor_id.agg(set)
    require(source_keys.map(lambda donors: donors == set(s.donor_id)).all(), "incomplete two-source common support")
    require(set(source_keys.index) == set(map(tuple, q[["sm_lincs_id", "dose_key", "time_key"]].to_numpy())), "source/query exposure support differs")


def choose_landmarks(metadata, source, maximum, seed):
    subset = metadata.iloc[np.asarray(source, int)].copy()
    subset["local_row"] = np.asarray(source, int)
    subset["hash"] = subset.atomic_id.map(lambda uid: hashlib.sha256(f"{seed}|{uid}".encode()).hexdigest())
    ordered = subset.sort_values(["hash", "atomic_id"], kind="stable")
    first = ordered.drop_duplicates("sm_lincs_id")
    count = min(maximum, len(subset))
    require(len(first) <= count and first.atomic_id.is_unique, "landmark budget cannot cover source compounds")
    remaining = ordered[~ordered.atomic_id.isin(first.atomic_id)]
    chosen = pd.concat([first, remaining.head(count - len(first))]).local_row.to_numpy(int)
    require(len(chosen) == count and len(set(chosen)) == count and set(chosen) <= set(source), "invalid source landmarks")
    return chosen


def same_drug_prediction(metadata, source, query, source_y):
    source = np.asarray(source, int)
    lookup = {}
    for position, row in enumerate(metadata.iloc[source].itertuples()):
        lookup.setdefault((row.sm_lincs_id, row.dose_key, row.time_key), []).append(position)
    prediction, pairs = [], []
    for query_position, local_row in enumerate(query):
        row = metadata.iloc[local_row]
        positions = lookup.get((row.sm_lincs_id, row.dose_key, row.time_key), [])
        require(len(positions) == 2 and metadata.iloc[source[positions]].donor_id.nunique() == 2, "same-drug requires exactly two source donors")
        prediction.append(np.asarray(source_y, np.float64)[positions].mean(axis=0))
        for position in positions:
            src = metadata.iloc[source[position]]
            pairs.append({"query_position": query_position, "query_atomic_id": row.atomic_id,
                          "source_position": position, "source_atomic_id": src.atomic_id,
                          "query_donor": row.donor_id, "source_donor": src.donor_id, "cell_type": row.cell_type,
                          "sm_lincs_id": row.sm_lincs_id, "dose_uM": row.dose_uM, "timepoint_hr": row.timepoint_hr, "weight": 0.5})
    return np.asarray(prediction), pd.DataFrame(pairs)


def unique_cosine_kernel(values, landmarks):
    values = np.asarray(values)
    require(values.ndim == 2 and np.isfinite(values).all(), "invalid kernel input")
    unique, inverse = np.unique(values, axis=0, return_inverse=True)
    # Compute each identical representation once, including duplicated source
    # donors. This prevents BLAS batch boundaries from inventing different rows.
    return cosine_similarity(unique, values[landmarks])[inverse]


def identity_kernel(metadata, source, landmarks):
    drugs = sorted(metadata.iloc[source].sm_lincs_id.unique())
    doses = sorted(metadata.iloc[source].dose_key.unique(), key=Decimal)
    require(metadata.sm_lincs_id.isin(drugs).all() and metadata.dose_key.isin(doses).all(), "unseen identity/exposure")
    drug_map, dose_map = {x: i for i, x in enumerate(drugs)}, {x: i for i, x in enumerate(doses)}
    values = np.zeros((len(metadata), len(drugs) + len(doses)), np.float64)
    values[np.arange(len(metadata)), metadata.sm_lincs_id.map(drug_map).to_numpy()] = 1
    values[np.arange(len(metadata)), len(drugs) + metadata.dose_key.map(dose_map).to_numpy()] = 1
    return unique_cosine_kernel(values, landmarks), {"source_drugs": drugs, "source_numeric_doses_uM": doses,
                                                    "fit_atomic_id": metadata.iloc[source].atomic_id.tolist()}


def tfidf_kernel(texts, source, landmarks):
    texts = np.asarray(texts, str)
    source_texts = sorted(set(texts[np.asarray(source, int)]))
    vectorizer = TfidfVectorizer(lowercase=True, analyzer="word", ngram_range=(1, 2), min_df=1,
                                max_features=20000, sublinear_tf=True)
    vectorizer.fit(source_texts)
    unique, inverse = np.unique(texts, return_inverse=True)
    transformed = normalize(vectorizer.transform(unique), norm="l2", copy=True)
    kernel = (transformed @ transformed[inverse[landmarks]].T).toarray()[inverse]
    return kernel, {"source_texts": source_texts, "source_text_sha256": [hashlib.sha256(t.encode()).hexdigest() for t in source_texts],
                    "feature_names": vectorizer.get_feature_names_out().tolist(), "idf": vectorizer.idf_.tolist(),
                    "fit_scope": "source unique exact prompts", "svd_used": False}


def tanimoto(first, second):
    first, second = np.asarray(first), np.asarray(second)
    require(first.ndim == second.ndim == 2 and first.shape[1] == second.shape[1], "Morgan shape mismatch")
    require(np.isin(first, [0, 1]).all() and np.isin(second, [0, 1]).all(), "Morgan not binary")
    first, second = first.astype(np.int64), second.astype(np.int64)
    intersection = first @ second.T
    union = first.sum(1)[:, None] + second.sum(1)[None, :] - intersection
    return np.divide(intersection, union, out=np.ones(intersection.shape, np.float64), where=union != 0)


def morgan_kernel(fingerprints, doses, landmarks):
    unique, inverse = np.unique(fingerprints, axis=0, return_inverse=True)
    structure = tanimoto(unique, unique)[inverse[:, None], inverse[landmarks][None, :]]
    doses = np.asarray([decimal_text(value) for value in doses], str)
    return 0.5 * (structure + (doses[:, None] == doses[landmarks][None, :]))


def predict_unique(model, query_features):
    unique, inverse = np.unique(np.asarray(query_features, np.float64), axis=0, return_inverse=True)
    prediction = model.predict(unique)[inverse]
    return prediction, unique, inverse


def rank_diagnostic(values):
    values = np.asarray(values, np.float64)
    singular = np.linalg.svd(values, compute_uv=False)
    tolerance = float(singular.max() * max(values.shape) * np.finfo(np.float64).eps)
    return {"rank": int(np.sum(singular > tolerance)), "tolerance": tolerance,
            "rule": "max_singular_value * max(shape) * float64_epsilon"}


def fit_predict_fold(metadata, source, query, source_y, texts, embeddings, fingerprints, state_A, config):
    """Only source_y enters fitting; no query effects or metric-driven choices."""
    source, query = np.asarray(source, int), np.asarray(query, int)
    validate_fold(metadata, source, query, source_y)
    require(np.asarray(state_A).shape == (len(metadata), np.asarray(source_y).shape[1]), "state/gene axis mismatch")
    require(np.asarray(state_A).dtype == np.float64 and np.isfinite(state_A).all(), "state must be unchanged finite float64")
    require(set(embeddings) == set(MODEL_PREFIXES), "six encoder inputs required")
    for donor in metadata.donor_id.unique():
        positions = np.flatnonzero(metadata.donor_id.to_numpy() == donor)
        require(np.array_equal(state_A[positions], np.broadcast_to(state_A[positions[0]], state_A[positions].shape)), "state varies across drugs in donor/type")
    landmarks = choose_landmarks(metadata, source, config["n_landmarks"], config["seed"])
    same, pairs = same_drug_prediction(metadata, source, query, source_y)
    nquery, ngenes = len(query), np.asarray(source_y).shape[1]
    outputs = {"zero": np.zeros((nquery, ngenes)),
               "source_mean": np.broadcast_to(np.mean(source_y, axis=0, dtype=np.float64), (nquery, ngenes)).copy(),
               "source_median": np.broadcast_to(np.median(np.asarray(source_y, np.float64), axis=0), (nquery, ngenes)).copy(),
               "same_drug_source_mean": same}
    kernels = {}
    kernels["identity_exposure"], identity = identity_kernel(metadata, source, landmarks)
    kernels["tfidf_entity_exposure"], tfidf = tfidf_kernel(texts, source, landmarks)
    kernels["morgan_entity_exposure"] = morgan_kernel(fingerprints, metadata.dose_key, landmarks)
    for key, prefix in MODEL_PREFIXES.items():
        require(len(embeddings[key]) == len(metadata), "embedding row mismatch")
        kernels[prefix + "_entity_exposure"] = unique_cosine_kernel(embeddings[key], landmarks)
    state_kernel = unique_cosine_kernel(state_A, landmarks)
    source_donors = sorted(metadata.iloc[source].donor_id.unique())
    source_states = np.asarray([state_A[source[np.flatnonzero(metadata.iloc[source].donor_id.to_numpy() == donor)[0]]] for donor in source_donors])
    centered_rank = rank_diagnostic(source_states - source_states.mean(axis=0))
    require(centered_rank["rank"] <= 1, "two-source-state centered rank exceeds one")
    archive = {"all_atomic_id": metadata.atomic_id.to_numpy(str), "source_rows": source, "query_rows": query, "landmark_rows": landmarks,
               "source_atomic_id": metadata.iloc[source].atomic_id.to_numpy(str), "query_atomic_id": metadata.iloc[query].atomic_id.to_numpy(str),
               "landmark_atomic_id": metadata.iloc[landmarks].atomic_id.to_numpy(str), "state_A_cosine": state_kernel,
               **{"base__" + k: v for k, v in kernels.items()}}
    fitted, model_audits = {}, {}
    for method in METHODS[len(BASE_METHODS):]:
        uses_state = method.endswith("__control_state")
        key = method.removesuffix("__control_state")
        features = build_landmark_features(kernels[key], state_kernel if uses_state else None,
                                           mode="interaction" if uses_state else "perturbation_only")
        decoder = LandmarkRidge(config["alpha"]).fit(features[source], source_y)
        outputs[method], unique, inverse = predict_unique(decoder, features[query])
        fitted.update({method + "__feature_mean": decoder.feature_mean_, method + "__feature_scale": decoder.feature_scale_,
                       method + "__target_mean": decoder.target_mean_, method + "__coef": decoder.coef_})
        archive.update({"features__" + method: features, "predict_unique__" + method: unique, "predict_inverse__" + method: inverse})
        model_audits[method] = {"alpha": config["alpha"], "landmark_count": len(landmarks), "target_A_available": uses_state,
                                "query_unique_input_count": len(unique), "predict_once_then_broadcast": True,
                                "source_exact_unique_rows": int(np.unique(features[source], axis=0).shape[0]),
                                "raw_source_rank": rank_diagnostic(features[source]),
                                "standardized_source_rank": rank_diagnostic((features[source] - decoder.feature_mean_) / decoder.feature_scale_),
                                "n_parameters": int(decoder.coef_.size + decoder.target_mean_.size)}
    predictions = np.stack([outputs[method] for method in METHODS]).astype(np.float32)
    require(np.isfinite(predictions).all(), "nonfinite model output; no silent method omission")
    audit = {"source_atomic_id": metadata.iloc[source].atomic_id.tolist(), "query_atomic_id": metadata.iloc[query].atomic_id.tolist(),
             "landmark_atomic_id": metadata.iloc[landmarks].atomic_id.tolist(), "landmark_rows": landmarks.tolist(),
             "source_donors": source_donors, "query_donor": metadata.iloc[query].donor_id.iloc[0],
             "source_state_centered_rank": centered_rank, "state_transform": "NONE: direct provided-scale state_A cosine",
             "target_effects_passed_to_fit": False, "target_B_passed_to_features": False, "hyperparameter_tuning": False,
             "models": model_audits, "method_order": METHODS, "float32_export_once_after_float64_prediction": True}
    return predictions, fitted, archive, audit, pairs, {"identity": identity, "tfidf": tfidf}


def finite_mean(values):
    values = np.asarray(values, np.float64)
    valid = np.isfinite(values)
    return float(values[valid].mean()) if valid.any() else np.nan


def score_fold(predictions, truth, metadata, panel, task_id, minimum):
    require(predictions.shape == (len(METHODS), len(metadata), len(panel)) and truth.shape == predictions.shape[1:], "scoring axes mismatch")
    require(metadata.cell_type.nunique() == metadata.donor_id.nunique() == metadata.time_key.nunique() == 1, "scoring context/time mixed")
    baseline_index = METHODS.index("same_drug_source_mean")
    baseline_condition = condition_metrics(predictions[baseline_index], truth)
    condition_frames, gene_frames, exposure_summaries, main_summaries = [], [], [], []
    groups = {dose: np.flatnonzero(metadata.dose_key.to_numpy() == dose) for dose in sorted(metadata.dose_key.unique(), key=Decimal)}
    baseline_gene = {}
    for dose, rows in groups.items():
        require(metadata.iloc[rows].sm_lincs_id.is_unique, "duplicate compound within fixed-exposure endpoint")
        if len(rows) >= minimum:
            baseline_gene[dose] = (gene_metrics(predictions[baseline_index, rows], truth[rows]), paired_order_accuracy_by_gene(predictions[baseline_index, rows], truth[rows]))
    for method_i, method in enumerate(METHODS):
        scores = condition_metrics(predictions[method_i], truth)
        conditions = metadata[["atomic_id", "sm_lincs_id", "sm_name", "cell_type", "donor_id", "dose_uM", "dose_key", "timepoint_hr"]].reset_index(drop=True).copy()
        conditions["task_id"], conditions["method"], conditions["track"] = task_id, method, track(method)
        for name, value in scores.items():
            conditions[name] = value
        for name in ["mae", "rmse", "spearman"]:
            valid = np.isfinite(scores[name]) & np.isfinite(baseline_condition[name])
            conditions[name + "_paired_valid"] = valid
            sign = -1 if name in {"mae", "rmse"} else 1
            conditions[name + "_improvement_vs_same_drug"] = np.where(valid, sign * (scores[name] - baseline_condition[name]), np.nan)
        condition_frames.append(conditions)
        genes_by_dose = {}
        for dose, rows in groups.items():
            eligible = len(rows) >= minimum
            if eligible:
                genes, order = gene_metrics(predictions[method_i, rows], truth[rows]), paired_order_accuracy_by_gene(predictions[method_i, rows], truth[rows])
                baseline_genes, baseline_order = baseline_gene[dose]
            else:
                genes = {"spearman": np.full(len(panel), np.nan), "spearman_valid": np.zeros(len(panel), bool),
                         "spearman_n": np.zeros(len(panel), int), "n_conditions": np.full(len(panel), len(rows))}
                order = {"accuracy": np.full(len(panel), np.nan), "n_pairs": np.zeros(len(panel), int)}
                baseline_genes, baseline_order = genes, order
            gene = panel[["rank", "source_feature_row", "source_gene_id", "gene_symbol"]].reset_index(drop=True).copy()
            for name, value in {"task_id": task_id, "cell_type": metadata.cell_type.iloc[0], "donor_id": metadata.donor_id.iloc[0],
                                "method": method, "track": track(method), "dose_key": dose, "timepoint_hr": metadata.timepoint_hr.iloc[0],
                                "eligible_min_n": eligible, "n_unique_compounds": len(rows), "minimum_required_compounds": minimum}.items():
                gene[name] = value
            for name, value in genes.items():
                gene[name] = value
            gene["pair_order_accuracy"], gene["n_pairs"] = order["accuracy"], order["n_pairs"]
            gene["spearman_paired_valid"] = np.isfinite(genes["spearman"]) & np.isfinite(baseline_genes["spearman"])
            gene["order_paired_valid"] = np.isfinite(order["accuracy"]) & np.isfinite(baseline_order["accuracy"])
            gene["spearman_improvement_vs_same_drug"] = genes["spearman"] - baseline_genes["spearman"]
            gene["order_improvement_vs_same_drug"] = order["accuracy"] - baseline_order["accuracy"]
            gene_frames.append(gene)
            genes_by_dose[dose] = gene
            exposure_summaries.append(summarize_scores(conditions.iloc[rows], gene, task_id, method, "fixed_exposure", dose))
        supported = [gene for gene in genes_by_dose.values() if bool(gene.eligible_min_n.iloc[0])]
        require(len(supported) == 1 and set(supported[0].dose_key) == {"1"}, "main gene summary must be supported 1uM only")
        main_summaries.append(summarize_scores(conditions, supported[0], task_id, method, "all_query_conditions", "all_conditions;gene_at_1uM_only"))
    return pd.concat(condition_frames, ignore_index=True), pd.concat(gene_frames, ignore_index=True), pd.DataFrame(main_summaries), pd.DataFrame(exposure_summaries)


def summarize_scores(conditions, genes, task_id, method, scope, dose):
    return {"task_id": task_id, "cell_type": conditions.cell_type.iloc[0], "heldout_donor": conditions.donor_id.iloc[0],
            "method": method, "track": track(method), "scope": scope, "dose_scope": dose, "timepoint_hr": conditions.timepoint_hr.iloc[0],
            "n_query_atoms": len(conditions), "n_unique_compounds": conditions.sm_lincs_id.nunique(), "n_genes": int(conditions.n_genes.iloc[0]),
            "mae_mean": float(conditions.mae.mean()), "rmse_mean": float(conditions.rmse.mean()),
            "condition_spearman_mean": finite_mean(conditions.spearman), "condition_spearman_valid_n": int(conditions.spearman_valid.sum()),
            "mae_improvement_vs_same_drug": float(conditions.mae_improvement_vs_same_drug.mean()),
            "rmse_improvement_vs_same_drug": float(conditions.rmse_improvement_vs_same_drug.mean()),
            "condition_spearman_improvement_vs_same_drug": finite_mean(conditions.spearman_improvement_vs_same_drug),
            "condition_spearman_paired_valid_n": int(conditions.spearman_paired_valid.sum()),
            "gene_spearman_mean": finite_mean(genes.spearman), "gene_pair_order_mean": finite_mean(genes.pair_order_accuracy),
            "gene_spearman_improvement_vs_same_drug": finite_mean(genes.spearman_improvement_vs_same_drug),
            "gene_order_improvement_vs_same_drug": finite_mean(genes.order_improvement_vs_same_drug),
            "gene_endpoint_eligible": bool(genes.eligible_min_n.iloc[0]), "gene_n_compounds": int(genes.n_unique_compounds.iloc[0]),
            "gene_total_n": len(genes), "gene_spearman_valid_n": int(genes.spearman_valid.sum()),
            "gene_order_valid_n": int(np.isfinite(genes.pair_order_accuracy).sum()),
            "gene_spearman_paired_valid_n": int(genes.spearman_paired_valid.sum()), "gene_order_paired_valid_n": int(genes.order_paired_valid.sum()),
            "pair_count_is_not_biological_n": True}


def aggregate_summaries(summary):
    require(not summary.duplicated(["task_id", "method"]).any(), "duplicate main fold summary")
    require(summary.task_id.nunique() == 6 and summary.cell_type.nunique() == 2, "expected six type/donor tasks")
    require(summary.groupby(["cell_type", "method"]).size().eq(3).all(), "incomplete three-donor type summary")
    group = summary.groupby(["cell_type", "method", "track"], sort=False)
    type_summary = group[METRIC_COLUMNS].mean().reset_index()
    counts = group[METRIC_COLUMNS].count().add_suffix("__valid_donor_folds").reset_index()
    type_summary = type_summary.merge(counts, on=["cell_type", "method", "track"], validate="one_to_one")
    type_summary["n_donor_folds"] = 3
    macro_group = type_summary.groupby(["method", "track"], sort=False)
    macro = macro_group[METRIC_COLUMNS].mean().reset_index()
    macro = macro.merge(macro_group[METRIC_COLUMNS].count().add_suffix("__valid_types").reset_index(), on=["method", "track"], validate="one_to_one")
    macro["n_types"], macro["n_unique_source_donors"], macro["n_tasks"], macro["n_studies"] = 2, 3, 6, 1
    macro["aggregation"] = "equal_donor_within_type_then_equal_type;finite_valid_counts_reported"
    macro["interpretation"] = "descriptive_only;shared_donors_and_vehicle_wells;no_p_values_or_CI"
    return type_summary, macro


def state_gain_tables(condition, gene, task_id):
    condition_pairs, gene_pairs, summaries = [], [], []
    for direct in KERNEL_METHODS:
        state = direct + "__control_state"
        first = condition[condition.method == direct].reset_index(drop=True)
        second = condition[condition.method == state].reset_index(drop=True)
        require(first.atomic_id.equals(second.atomic_id), "direct/state condition pairing differs")
        cp = first[["task_id", "atomic_id", "sm_lincs_id", "cell_type", "donor_id", "dose_key", "timepoint_hr"]].copy()
        cp["direct_method"], cp["state_method"] = direct, state
        for metric in ["mae", "rmse", "spearman"]:
            valid = np.isfinite(first[metric].to_numpy()) & np.isfinite(second[metric].to_numpy())
            cp[metric + "_paired_valid"] = valid
            sign = -1 if metric in {"mae", "rmse"} else 1
            cp[metric + "_state_gain"] = np.where(valid, sign * (second[metric].to_numpy() - first[metric].to_numpy()), np.nan)
        condition_pairs.append(cp)
        first = gene[gene.method == direct].reset_index(drop=True)
        second = gene[gene.method == state].reset_index(drop=True)
        require(first[["source_feature_row", "dose_key"]].equals(second[["source_feature_row", "dose_key"]]), "direct/state gene pairing differs")
        gp = first[["task_id", "source_feature_row", "source_gene_id", "gene_symbol", "cell_type", "donor_id", "dose_key", "timepoint_hr", "eligible_min_n", "n_unique_compounds", "n_pairs"]].copy()
        gp["direct_method"], gp["state_method"] = direct, state
        for metric in ["spearman", "pair_order_accuracy"]:
            valid = np.isfinite(first[metric].to_numpy()) & np.isfinite(second[metric].to_numpy())
            gp[metric + "_paired_valid"] = valid
            gp[metric + "_state_gain"] = np.where(valid, second[metric].to_numpy() - first[metric].to_numpy(), np.nan)
        gene_pairs.append(gp)
        supported = gp[gp.dose_key == "1"]
        summaries.append({"task_id": task_id, "cell_type": cp.cell_type.iloc[0], "heldout_donor": cp.donor_id.iloc[0],
                          "direct_method": direct, "state_method": state, "n_query_atoms": len(cp),
                          "mae_state_gain": float(cp.mae_state_gain.mean()), "rmse_state_gain": float(cp.rmse_state_gain.mean()),
                          "condition_spearman_state_gain": finite_mean(cp.spearman_state_gain),
                          "condition_spearman_paired_n": int(cp.spearman_paired_valid.sum()),
                          "gene_spearman_state_gain": finite_mean(supported.spearman_state_gain),
                          "gene_order_state_gain": finite_mean(supported.pair_order_accuracy_state_gain),
                          "gene_spearman_paired_n": int(supported.spearman_paired_valid.sum()),
                          "gene_order_paired_n": int(supported.pair_order_accuracy_paired_valid.sum()),
                          "gene_total_n": len(supported), "gene_dose_uM": "1", "timepoint_hr": "24",
                          "error_gain_direction": "direct_minus_state", "utility_gain_direction": "state_minus_direct"})
    return pd.concat(condition_pairs, ignore_index=True), pd.concat(gene_pairs, ignore_index=True), pd.DataFrame(summaries)


def verify_file(path, expected, size=None):
    path = Path(path)
    require(isinstance(expected, str) and len(expected) == 64 and all(c in "0123456789abcdef" for c in expected), "unbound/pending SHA256")
    require(path.is_file() and (size is None or path.stat().st_size == int(size)), "missing/size-mismatched input: " + str(path))
    require(sha256(path) == expected, "changed input: " + str(path))
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": expected}


def verify_manifest(root, expected):
    root = Path(root)
    records = [verify_file(root / "output_manifest.json", expected)]
    manifest = json.loads((root / "output_manifest.json").read_text())
    for record in manifest["files"]:
        path = root / record["path"]
        require(root.resolve() in path.resolve().parents, "manifest path outside artifact root")
        records.append(verify_file(path, record["sha256"], record.get("size_bytes", record.get("bytes"))))
    return records


def verify_gate(gate):
    record = verify_file(gate["path"], gate["sha256"])
    audit = json.loads(Path(gate["path"]).read_text())
    require(audit.get("status") == gate["required_status"], "independent QA not PASS: " + gate["path"])
    require(bool(gate["bindings"]), "independent QA has no artifact binding")
    for binding in gate["bindings"]:
        actual = audit
        for key in binding["keys"]:
            require((isinstance(actual, dict) and key in actual) or
                    (isinstance(actual, list) and isinstance(key, int) and 0 <= key < len(actual)), "independent QA binding absent")
            actual = actual[key]
        require(actual == binding["expected"], "independent QA binds other artifact")
    return record


def validate_inputs(config, config_path):
    require(config["alpha"] == 10 and config["n_landmarks"] == 256 and config["seed"] == 20260914, "frozen decoder settings changed")
    require(config["gene_min_n"] == 20 and config["expected_genes_per_fold"] == 3000, "frozen endpoint settings changed")
    require(config["method_order"] == METHODS and len(METHODS) == 22, "method registry changed")
    require(config["state_transform"] == "none" and config["hyperparameter_tuning"] is False, "state transform/tuning forbidden")
    sources = [verify_file(config["scientific_contract"], config["scientific_contract_sha256"])]
    for gate in config["independent_gates"].values():
        sources.append(verify_gate(gate))
    require(set(config["independent_gates"]) == {"metadata", "effects", "embeddings", "morgan"}, "independent QA gates incomplete")
    for root_key in ["metadata_root", "effects_root", "clean_views_root", "embeddings_root"]:
        sources.extend(verify_manifest(config[root_key], config[root_key + "_manifest_sha256"]))
    for path, digest in config["immutable_helpers"].items():
        sources.append(verify_file(path, digest))
    morgan_root = Path(config["morgan_root"])
    sources.append(verify_file(morgan_root / "audit.json", config["morgan_audit_sha256"]))
    morgan_audit = json.loads((morgan_root / "audit.json").read_text())
    require(morgan_audit["status"] == "PASS", "Morgan not PASS")
    for name, digest in morgan_audit["output_sha256"].items():
        sources.append(verify_file(morgan_root / name, digest))
    for path in [Path(config_path), Path(__file__), ROOT / "tests/test_kaggle_prediction.py"]:
        sources.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)})
    effects_audit = json.loads((Path(config["effects_root"]) / "audit.json").read_text())
    require(effects_audit["status"] == "PASS" and effects_audit["scientific_contract_sha256"] == config["scientific_contract_sha256"], "effects do not bind contract")
    feature_seal = json.loads((Path(config["effects_root"]) / "feature_selection_sealed.json").read_text())
    require(feature_seal["feature_selection_uses_target_A"] is False and feature_seal["feature_selection_uses_any_B"] is False and feature_seal["feature_selection_uses_any_treated"] is False, "feature permission failure")
    require(feature_seal["fold_gene_panels_sha256"] == sha256(Path(config["effects_root"]) / "fold_gene_panels.tsv"), "feature panel seal mismatch")
    return list({record["path"]: record for record in sources}.values())


def load_representations(metadata, config):
    root = Path(config["clean_views_root"])
    registry, texts = read_tsv(root / "row_to_text_registry.tsv"), read_tsv(root / "unique_texts.tsv")
    require(registry.atomic_id.is_unique and registry.atomic_id.tolist() == metadata.atomic_id.tolist(), "text/atomic axis differs")
    require(set(registry.view) == {"entity_exposure"} and set(registry.variant) == {"source_name"}, "unclean text view")
    require(texts.text_id.is_unique and len(texts) == 137, "wrong text registry count")
    for column in ["sm_lincs_id", "sm_name", "dose_uM", "timepoint_hr"]:
        require(registry[column].equals(metadata[column]), "metadata/text exposure mismatch: " + column)
    for row in texts.itertuples():
        expected = row.sm_name + "; dose: " + decimal_text(Decimal(row.dose_uM) * 1000) + " nM; duration: 24 h."
        require(row.prompt_text == expected and Decimal(row.timepoint_hr) == 24, "text/units/time mismatch")
        digest = hashlib.sha256(expected.encode()).hexdigest()
        require(row.prompt_sha256 == digest and row.text_id == "text:" + digest, "text digest mismatch")
    lookup = texts.set_index("text_id")
    require((registry.prompt_sha256 == registry.text_id.map(lookup.prompt_sha256)).all(), "registry/text hash mismatch")
    prompt_by_atom = lookup.loc[registry.text_id].prompt_text.to_numpy(str)
    embeddings = {}
    audit = json.loads((Path(config["embeddings_root"]) / "audit.json").read_text())
    require(audit["status"] == "PASS" and audit["responses_or_prediction_labels_read"] is False, "embedding not clean PASS")
    for key in MODEL_PREFIXES:
        record = next(r for r in audit["outputs"] if r["model_key"] == key)
        path = Path(config["embeddings_root"]) / (key + "__source_name.npz")
        require(sha256(path) == record["sha256"], "embedding payload changed")
        with np.load(path, allow_pickle=False) as z:
            require(set(z.files) == {"X", "text_id", "prompt_sha256"}, "embedding schema differs")
            require(np.array_equal(z["text_id"], texts.text_id.to_numpy(str)) and np.array_equal(z["prompt_sha256"], texts.prompt_sha256.to_numpy(str)), "embedding text axis mismatch")
            require(z["X"].dtype == np.float32 and z["X"].shape == (137, record["dimension"]) and np.isfinite(z["X"]).all(), "invalid embedding matrix")
            positions = pd.Index(z["text_id"]).get_indexer(registry.text_id)
            require((positions >= 0).all(), "missing atom embedding")
            embeddings[key] = z["X"][positions].copy()
    with np.load(Path(config["morgan_root"]) / "arrays.npz", allow_pickle=False) as z:
        require(z["sm_lincs_id"].tolist() == z["source_entity_key"].tolist(), "Morgan source ID axis differs")
        positions = pd.Index(z["sm_lincs_id"]).get_indexer(metadata.sm_lincs_id)
        require((positions >= 0).all(), "missing Morgan compound")
        fingerprints = z["fingerprints"][positions]
    require(fingerprints.shape == (len(metadata), 2048) and np.isin(fingerprints, [0, 1]).all() and (fingerprints.sum(1) > 0).all(), "invalid Morgan fingerprints")
    return prompt_by_atom, embeddings, fingerprints


def effect_slice(path, atomic_rows, columns):
    # The compressed full array must decompress; only this explicitly selected
    # source or (after sealing) truth slice is returned to the caller.
    with np.load(path, allow_pickle=False) as z:
        array = z["effect_provided_scale"]
        require(array.shape == (816, 3586) and array.dtype == np.float32, "effect shape/dtype differs")
        selected = array[np.ix_(atomic_rows, columns)].copy()
    require(np.isfinite(selected).all(), "nonfinite selected effects")
    return selected


def seal_predictions(output, records, config_hash):
    require(len(records) == 6 and len({r["task_id"] for r in records}) == 6, "all six predictions must freeze before scoring")
    for record in records:
        require(sha256(record["path"]) == record["sha256"], "prediction changed before seal")
    write_json(output / "predictions_sealed.json", {"status": "ALL_SIX_FOLDS_FROZEN_BEFORE_SCORING", "config_sha256": config_hash,
                                                   "created_utc": datetime.now(timezone.utc).isoformat(), "records": records})


def run(config_path, *, execute=False):
    require(execute, "explicit --execute required after independent QA and root review")
    started = time.perf_counter()
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    output = Path(config["output_root"])
    require(not output.exists(), "output exists; refusing overwrite")
    sources = validate_inputs(config, config_path)
    effect_root = Path(config["effects_root"])
    metadata = read_tsv(effect_root / "atomic_index.tsv")
    require(len(metadata) == 816 and metadata.atomic_id.is_unique and metadata.atomic_row.astype(int).tolist() == list(range(816)), "atomic axis mismatch")
    metadata["dose_key"] = metadata.dose_uM.map(decimal_text)
    metadata["time_key"] = metadata.timepoint_hr.map(decimal_text)
    tasks = read_tsv(effect_root / "task_index.tsv").sort_values("task_id").reset_index(drop=True)
    panels = read_tsv(effect_root / "fold_gene_panels.tsv")
    memberships = read_tsv(effect_root / "fold_atomic_membership.tsv")
    approved_pairs = read_tsv(effect_root / "same_drug_source_pairs.tsv")
    require(len(tasks) == 6 and tasks.task_id.is_unique, "task axis mismatch")
    prompts, embeddings, fingerprints = load_representations(metadata, config)
    with np.load(effect_root / "arrays.npz", allow_pickle=False) as z:
        require(np.array_equal(z["atomic_id"], metadata.atomic_id.to_numpy(str)), "effects/atomic IDs differ")
        feature_rows, context_ids, state = z["source_feature_row"], z["context_id"], z["state_A"]
        # Deliberately never access the state_B key.
    require(state.dtype == np.float64 and state.shape == (6, 3586) and np.isfinite(state).all(), "state_A shape/dtype/finite contract")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "input_manifest.json", sources)
    write_json(output / "method_registry.json", {"method_order": METHODS, "tracks": {m: track(m) for m in METHODS}})
    try:
        frozen, task_records = [], []
        for task in tasks.itertuples():
            print("FIT_TASK", task.task_id, task.cell_type, task.heldout_donor, flush=True)
            directory = output / task.task_id
            directory.mkdir()
            all_rows = np.flatnonzero(metadata.cell_type.to_numpy() == task.cell_type)
            local = metadata.iloc[all_rows].reset_index(drop=True)
            source = np.flatnonzero(local.donor_id.to_numpy() != task.heldout_donor)
            query = np.flatnonzero(local.donor_id.to_numpy() == task.heldout_donor)
            require(len(source) == int(task.n_source_atomic_conditions) and len(query) == int(task.n_query_atomic_conditions), "task source/query counts changed")
            member = memberships[memberships.task_id == task.task_id]
            require(set(member.loc[member.role == "source", "atomic_id"]) == set(local.iloc[source].atomic_id) and
                    set(member.loc[member.role == "query", "atomic_id"]) == set(local.iloc[query].atomic_id), "frozen membership mismatch")
            panel = panels[panels.task_id == task.task_id].copy()
            panel["rank"] = panel["rank"].astype(int)
            panel = panel.sort_values("rank").reset_index(drop=True)
            columns = panel.union_column.to_numpy(int)
            require(len(panel) == 3000 and panel.source_feature_row.is_unique and panel["rank"].tolist() == list(range(1, 3001)), "source gene panel incomplete")
            require(np.array_equal(feature_rows[columns], panel.source_feature_row.to_numpy(int)), "gene panel union mapping differs")
            require(set(panel.cell_type) == {task.cell_type} and set(panel.heldout_donor) == {task.heldout_donor}, "panel task metadata differs")
            context_rows = pd.Index(context_ids).get_indexer(local.context_id)
            require((context_rows >= 0).all(), "missing control-A context")
            source_y = effect_slice(effect_root / "arrays.npz", all_rows[source], columns)
            predictions, fitted, kernels, audit, pairs, feature_fit = fit_predict_fold(local, source, query, source_y,
                prompts[all_rows], {k: x[all_rows] for k, x in embeddings.items()}, fingerprints[all_rows], state[np.ix_(context_rows, columns)], config)
            match = approved_pairs[approved_pairs.task_id == task.task_id]
            require(set(zip(pairs.query_atomic_id, pairs.source_atomic_id)) == set(zip(match.query_atomic_id, match.source_atomic_id)), "same-drug pair manifest differs")
            require((match.same_drug_source_weight.astype(float) == 0.5).all(), "same-drug source weight differs")
            ids = local.iloc[query].atomic_id.to_numpy(str)
            gene_axis = {"source_feature_row": panel.source_feature_row.to_numpy(int), "source_gene_id": panel.source_gene_id.to_numpy(str)}
            path = directory / "frozen_predictions.npz"
            save_npz(path, predictions=predictions, method=np.asarray(METHODS, str), atomic_id=ids, **gene_axis)
            save_npz(directory / "source_targets.npz", source_y=source_y, atomic_id=local.iloc[source].atomic_id.to_numpy(str), **gene_axis)
            save_npz(directory / "fitted_decoder_parameters.npz", **fitted)
            save_npz(directory / "landmark_kernels.npz", **kernels)
            write_json(directory / "feature_fit_metadata.json", feature_fit)
            for name, frame in [("source_atoms.tsv", local.iloc[source]), ("query_atoms.tsv", local.iloc[query]),
                                ("landmarks.tsv", local.iloc[kernels["landmark_rows"]]), ("gene_panel.tsv", panel),
                                ("same_drug_source_pairs.tsv", pairs)]:
                write_tsv(directory / name, frame)
            audit.update({"status": "PREDICTIONS_FROZEN_NOT_YET_SCORED", "task_id": task.task_id, "cell_type": task.cell_type,
                          "heldout_donor": task.heldout_donor, "n_source": len(source), "n_query": len(query), "n_genes": len(panel),
                          "prediction_sha256": sha256(path), "config_sha256": sha256(config_path)})
            write_json(directory / "fit_audit.json", audit)
            frozen.append({"task_id": task.task_id, "path": str(path), "sha256": sha256(path)})
            task_records.append((task.task_id, all_rows[query], columns))
        seal_predictions(output, frozen, sha256(config_path))
        seal_hash = sha256(output / "predictions_sealed.json")
        write_json(output / "scoring_started.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
            "stage": "AFTER_ALL_SIX_PREDICTIONS_SEALED_BEFORE_FIRST_QUERY_TRUTH_SLICE",
            "predictions_sealed_sha256": seal_hash, "n_frozen_tasks": len(frozen)})
        summaries, dose_summaries, gain_summaries = [], [], []
        for task_id, query_rows, columns in task_records:
            print("SCORE_TASK", task_id, flush=True)
            require(sha256(output / "predictions_sealed.json") == seal_hash, "global prediction seal changed")
            directory = output / task_id
            prediction_record = next(r for r in frozen if r["task_id"] == task_id)
            require(sha256(prediction_record["path"]) == prediction_record["sha256"], "frozen prediction changed")
            with np.load(prediction_record["path"], allow_pickle=False) as z:
                predictions = z["predictions"]
                ids, gene_rows, gene_ids = z["atomic_id"], z["source_feature_row"], z["source_gene_id"]
            truth = effect_slice(effect_root / "arrays.npz", query_rows, columns)
            save_npz(directory / "evaluation_truth.npz", truth=truth, atomic_id=ids, source_feature_row=gene_rows, source_gene_id=gene_ids)
            panel = read_tsv(directory / "gene_panel.tsv")
            condition, gene, summary, dose_summary = score_fold(predictions, truth, metadata.iloc[query_rows].reset_index(drop=True), panel, task_id, config["gene_min_n"])
            cpair, gpair, gains = state_gain_tables(condition, gene, task_id)
            for name, frame in [("condition_metrics.tsv", condition), ("gene_metrics.tsv", gene), ("state_gain_condition.tsv", cpair), ("state_gain_gene.tsv", gpair)]:
                write_tsv(directory / name, frame)
            summaries.append(summary)
            dose_summaries.append(dose_summary)
            gain_summaries.append(gains)
            require(sha256(prediction_record["path"]) == prediction_record["sha256"], "scoring modified predictions")
            write_json(directory / "audit.json", {"status": "PASS", "task_id": task_id, "prediction_sha256": prediction_record["sha256"],
                "global_prediction_seal_sha256": seal_hash, "all_six_predictions_frozen_before_scoring": True,
                "n_methods": 22, "n_queries": len(query_rows), "n_genes": len(columns), "scoring_is_descriptive": True})
        summary = pd.concat(summaries, ignore_index=True)
        type_summary, macro = aggregate_summaries(summary)
        for name, frame in [("summary.tsv", summary), ("task_dose_summary.tsv", pd.concat(dose_summaries, ignore_index=True)),
                            ("state_gain_summary.tsv", pd.concat(gain_summaries, ignore_index=True)),
                            ("type_summary.tsv", type_summary), ("descriptive_macro_summary.tsv", macro)]:
            write_tsv(output / name, frame)
        for record in sources:
            verify_file(record["path"], record["sha256"], record["bytes"])
        write_json(output / "audit.json", {"status": "PASS", "created_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
            "seconds": time.perf_counter() - started, "config_sha256": sha256(config_path), "script_sha256": sha256(__file__),
            "input_manifest_sha256": sha256(output / "input_manifest.json"), "method_order": METHODS, "task_ids": tasks.task_id.tolist(),
            "n_tasks": 6, "n_studies": 1, "n_unique_source_donors": 3, "n_source_annotated_types": 2,
            "measurement_scale": config["measurement_scale"], "state_transform": "none", "raw_X_read": False,
            "target_effects_passed_to_fit": False, "target_B_state_read": False, "all_predictions_frozen_before_scoring": True,
            "global_prediction_seal_sha256": seal_hash, "independent_prediction_QA": False,
            "package_versions": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]},
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "source_hashes_unchanged": True,
            "scope": "retrospective_within_type_seen_drug_donor_plus_nested_plate_library_shift;descriptive_only"})
        files = [{"path": str(p.relative_to(output)), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(output.rglob("*")) if p.is_file()]
        write_json(output / "output_manifest.json", {"files": files})
        print("KAGGLE_PREDICTION_PASS", time.perf_counter() - started, flush=True)
    except Exception as exc:
        write_json(output / "failure.json", {"status": "FAILED", "error": repr(exc), "partial_outputs_not_complete": True})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/kaggle_prediction_v1.json")
    parser.add_argument("--execute", action="store_true", help="Run only after explicit root approval; all independent gates remain mandatory")
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        run(args.config, execute=args.execute)
