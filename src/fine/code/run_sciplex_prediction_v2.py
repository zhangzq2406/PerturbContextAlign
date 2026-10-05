#!/usr/bin/env python3
"""S3 six-encoder sci-Plex LOCO core in one study, with exact S2 regression gates.

The immutable v1 implementation reruns the original 17 methods. Twelve methods
from four additional frozen encoders are appended under the SAME effect, gene,
cohort, landmark, alpha and scoring contracts. No target responses enter fitting.
This is not a completed cross-study benchmark. Importing this module runs nothing.
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
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_sciplex_prediction as legacy
from landmark_decoder import LandmarkRidge, build_landmark_features, cosine_similarity
from prediction_metrics import condition_metrics, gene_metrics, paired_order_accuracy_by_gene

ROOT = Path(__file__).resolve().parents[1]
require, sha256, write_json, read_tsv = legacy.require, legacy.sha256, legacy.write_json, legacy.read_tsv
boolean_mask, validate_balanced_cohort = legacy.boolean_mask, legacy.validate_balanced_cohort
save_npz, load_morgan, track, finite_mean = legacy.save_npz, legacy.load_morgan, legacy.track, legacy.finite_mean
LEGACY_METHODS = tuple(legacy.METHODS)
MODEL_PREFIXES = {
    "bge_m3": "bge", "sapbert": "sapbert", "qwen3_0_6b": "qwen3",
    "biomedbert": "biomedbert", "medcpt_article": "medcpt_article", "medcpt_query": "medcpt_query",
}
VIEWS = ("entity_exposure", "complete_metadata")


def encoder_specs(config, *, require_paths=False):
    """Validate the frozen six model keys and their method prefixes."""
    specs = config.get("encoders")
    require(isinstance(specs, list) and len(specs) == 6, "encoder manifest must contain exactly six entries")
    require(all(isinstance(item, dict) for item in specs), "encoder entries must be objects")
    keys = [item.get("model_key") for item in specs]
    require(len(set(keys)) == 6 and set(keys) == set(MODEL_PREFIXES), "duplicate/missing/unexpected encoder model key")
    for item in specs:
        require(item.get("method_prefix") == MODEL_PREFIXES[item["model_key"]], "encoder method prefix mismatch")
        if require_paths:
            require(isinstance(item.get("embedding_root"), str) and Path(item["embedding_root"]).is_absolute(),
                    "encoder embedding_root must be an absolute path")
    return specs


def method_order(config):
    """Keep original S2 order, then append three methods per additional encoder."""
    methods = list(LEGACY_METHODS)
    for item in encoder_specs(config):
        if item["model_key"] in {"bge_m3", "sapbert"}:
            continue
        name = item["method_prefix"]
        methods.extend([name + "_entity_exposure", name + "_entity_exposure__control_state",
                        name + "_complete_metadata"])
    require(len(methods) == 29 and len(set(methods)) == 29, "six-encoder method order must be 29 unique entries")
    return methods


METHODS = method_order({"encoders": [{"model_key": key, "method_prefix": value} for key, value in MODEL_PREFIXES.items()]})


def fit_predict_fold(metadata, train_rows, test_rows, train_y, text_by_view,
                     embedding_by_model_view, control_state, config):
    """Fit source responses only; old predictions are never used as model input."""
    specs, methods = encoder_specs(config), method_order(config)
    require(tuple(legacy.METHODS) == LEGACY_METHODS, "immutable v1 METHODS changed")
    original, audit, fitted = legacy.fit_predict_fold(
        metadata, train_rows, test_rows, train_y, text_by_view, embedding_by_model_view, control_state, config)
    train_rows, test_rows = np.asarray(train_rows, dtype=int), np.asarray(test_rows, dtype=int)
    landmarks = np.asarray(audit["landmark_rows"], dtype=int)
    state_kernel = cosine_similarity(control_state, np.asarray(control_state)[landmarks])
    outputs = {name: original[index] for index, name in enumerate(LEGACY_METHODS)}
    for item in specs:
        key, prefix = item["model_key"], item["method_prefix"]
        if key in {"bge_m3", "sapbert"}:
            continue
        for view in VIEWS:
            embedding = np.asarray(embedding_by_model_view[(key, view)])
            require(embedding.ndim == 2 and len(embedding) == len(metadata) and np.isfinite(embedding).all(),
                    "invalid embedding/metadata rows: " + key + "/" + view)
            kernel = cosine_similarity(embedding, embedding[landmarks])
            suffixes = ("", "__control_state") if view == "entity_exposure" else ("",)
            for suffix in suffixes:
                method, has_state = prefix + "_" + view + suffix, bool(suffix)
                features = build_landmark_features(kernel, state_kernel if has_state else None,
                    mode="interaction" if has_state else "perturbation_only")
                model = LandmarkRidge(config["alpha"]).fit(features[train_rows], train_y)
                outputs[method] = model.predict(features[test_rows])
                fitted[method] = {"feature_mean": model.feature_mean_, "feature_scale": model.feature_scale_,
                                 "target_mean": model.target_mean_, "coef": model.coef_}
                audit["models"][method] = {
                    "model_key": key, "n_landmarks": len(landmarks), "alpha": config["alpha"],
                    "n_parameters": int(model.coef_.size + model.target_mean_.size),
                    "mode": "product_kernel" if has_state else "direct_landmark_kernel",
                    "target_controls_available": has_state,
                    "train_feature_mean_sha256": hashlib.sha256(model.feature_mean_.tobytes()).hexdigest(),
                    "train_feature_scale_sha256": hashlib.sha256(model.feature_scale_.tobytes()).hexdigest(),
                    "kernel_diagnostics": legacy.kernel_diagnostics(features, train_rows, test_rows, model)}
    predictions = np.stack([outputs[method] for method in methods]).astype(np.float32)
    require(np.isfinite(predictions).all(), "nonfinite six-encoder predictions")
    require(np.array_equal(predictions[:len(LEGACY_METHODS)].view(np.uint32), original.view(np.uint32)),
            "assembling new methods changed rerun v1 float32 predictions")
    audit.update({"stage": "S3_SIX_ENCODER_SINGLE_STUDY_CORE", "method_order": methods,
                  "encoder_keys": [item["model_key"] for item in specs],
                  "legacy_methods_recomputed": True, "legacy_predictions_used_as_fit_input": False,
                  "scope": "six-encoder retrospective single-study core; not cross-study completion"})
    return predictions, audit, fitted


def load_representations(metadata, clean_root, specs):
    """Load each model from its own PASS audit/hash and exact-text registry."""
    specs = encoder_specs({"encoders": specs}, require_paths=True)
    clean_root = Path(clean_root)
    source_names = ("config.json", "row_to_text_registry.tsv", "unique_texts.tsv")
    source_hashes = {str(clean_root / name): sha256(clean_root / name) for name in source_names}
    registry = read_tsv(clean_root / "row_to_text_registry.tsv")
    registry = registry[registry.variant == "source_name"]
    table = read_tsv(clean_root / "unique_texts.tsv")
    require(table.text_id.is_unique, "duplicate exact text IDs")
    table = table.set_index("text_id")
    text_by_view, ids_by_view = {}, {}
    for view in VIEWS:
        mapping = registry[registry.view == view]
        require(mapping.atomic_id.is_unique, "ambiguous atom/view mapping")
        mapping = mapping.set_index("atomic_id").loc[metadata.atomic_id]
        require(np.array_equal(mapping.prompt_sha256.to_numpy(), table.loc[mapping.text_id].prompt_sha256.to_numpy()),
                "registry/text hash mismatch")
        ids_by_view[view] = mapping.text_id.to_numpy(dtype=str)
        text_by_view[view] = table.loc[mapping.text_id].prompt_text.to_numpy(dtype=str)
    embeddings, records, audits = {}, [], {}
    for item in specs:
        model, root = item["model_key"], Path(item["embedding_root"])
        audit_path = root / "audit.json"
        if root not in audits:
            audits[root] = json.loads(audit_path.read_text())
        audit = audits[root]
        require(audit.get("status") == "PASS", "embedding audit not PASS: " + model)
        require(audit.get("variant") == "source_name", "embedding variant is not source_name: " + model)
        encoded_sources = {row["path"]: row["sha256"] for row in audit["sources"]}
        for path, digest in source_hashes.items():
            require(encoded_sources.get(path) == digest, "text source changed since encoding: " + path)
        matches = [row for row in audit["outputs"] if row["model_key"] == model]
        require(len(matches) == 1, "encoding audit must have one output per model: " + model)
        record = matches[0]
        path = root / (model + "__source_name.npz")
        require(Path(record["path"]).resolve() == path.resolve(), "embedding audit/output path mismatch: " + model)
        require(sha256(path) == record["sha256"], "embedding hash changed: " + model)
        with np.load(path, allow_pickle=False) as archive:
            ids, x, digests = archive["text_id"], archive["X"], archive["prompt_sha256"]
        require(ids.ndim == digests.ndim == 1 and len(set(ids)) == len(ids), "invalid embedding IDs: " + model)
        require(x.ndim == 2 and x.shape[0] == len(ids) == len(digests) and x.dtype == np.float32 and np.isfinite(x).all(),
                "invalid embedding archive: " + model)
        require(x.shape == (record["rows"], record["dimension"]), "embedding audit shape mismatch: " + model)
        require(np.array_equal(ids, np.asarray(["text:" + str(value) for value in digests])),
                "embedding text ID/digest identity mismatch: " + model)
        lookup = pd.Index(ids)
        for view, ids_needed in ids_by_view.items():
            rows = lookup.get_indexer(ids_needed)
            require((rows >= 0).all(), "clean embedding absent: " + model + "/" + view)
            require(np.array_equal(digests[rows], table.loc[ids_needed].prompt_sha256.to_numpy(dtype=str)),
                    "embedding exact-text mismatch: " + model + "/" + view)
            embeddings[(model, view)] = x[rows]
        records.append({"model_key": model, "method_prefix": item["method_prefix"],
                        "embedding_path": str(path), "embedding_sha256": record["sha256"],
                        "audit_path": str(audit_path), "audit_sha256": sha256(audit_path),
                        "revision": record.get("revision"), "rows": record["rows"], "dimension": record["dimension"],
                        "dtype": str(x.dtype), "source_text_hashes_verified": True})
    require(len(records) == 6, "six model provenance records required")
    return text_by_view, embeddings, records


FROZEN_FIELDS = (
    "effects_root", "clean_views_root", "morgan_root", "seed", "alpha", "n_landmarks",
    "gene_min_n", "expected_genes_per_fold", "landmark_policy", "context_state_kernel",
    "identity_exposure_features", "tfidf", "cohorts", "hyperparameter_tuning",
    "scientific_scope", "controls_permission", "morgan_kernel", "morgan_scope",
)


def validate_contract(config):
    specs = encoder_specs(config, require_paths=True)
    require(config.get("method_count") == 29 and len(method_order(config)) == 29, "method_count must be 29")
    require(config.get("legacy_regression_policy") == "float32_bitwise_exact", "legacy comparison cannot relax exactness")
    require(config.get("stage") == "S3_SIX_ENCODER_SINGLE_STUDY_CORE", "wrong stage/scope")
    reference = Path(config["legacy_prediction_root"])
    audit = json.loads((reference / "audit.json").read_text())
    require(audit.get("status") == "PASS" and audit["method_order"] == list(LEGACY_METHODS), "legacy 17-method run not PASS")
    for field in FROZEN_FIELDS:
        require(field in config and config[field] == audit["config"][field], "frozen v1 contract field changed: " + field)
    require(Path(config["output_root"]).resolve() != reference.resolve(), "new run cannot overwrite the legacy output root")
    by_model = {item["model_key"]: item for item in specs}
    for key in ("bge_m3", "sapbert"):
        require(by_model[key]["embedding_root"] == audit["config"]["embeddings_root"], "original encoder source changed")
    input_manifest = json.loads((reference / "input_manifest.json").read_text())
    for row in input_manifest:
        require(sha256(row["path"]) == row["sha256"], "legacy input changed since S2: " + row["path"])
    for row in audit["source_scripts"]:
        require(sha256(row["path"]) == row["sha256"], "immutable v1 code changed: " + row["path"])
    # Preflight PASS and output existence BEFORE creating a new output directory.
    for item in specs:
        root = Path(item["embedding_root"])
        encoding_audit = json.loads((root / "audit.json").read_text())
        require(encoding_audit.get("status") == "PASS", "all six encoding audits must PASS before run: " + item["model_key"])
        require((root / (item["model_key"] + "__source_name.npz")).is_file(), "encoder output absent")
    return audit, input_manifest


def compare_legacy_fold(predictions, methods, atomic_ids, gene_panel, fold_audit, reference_root, heldout):
    """Strict regression gate after new predictions freeze and before scoring."""
    reference = Path(reference_root) / heldout
    audit = json.loads((reference / "audit.json").read_text())
    require(audit.get("status") == "PASS", "legacy fold not PASS: " + heldout)
    path = reference / "frozen_predictions.npz"
    require(sha256(path) == audit["prediction_sha256"], "legacy prediction hash mismatch: " + heldout)
    for key in ("landmark_atomic_id", "source_atomic_id", "target_atomic_id"):
        require(fold_audit[key] == audit[key], "legacy regression split/landmark axis mismatch: " + key)
    with np.load(path, allow_pickle=False) as archive:
        old = archive["predictions"]
        old_methods = archive["method"].tolist()
        require(old_methods == list(LEGACY_METHODS), "legacy method order changed")
        require(np.array_equal(archive["atomic_id"], atomic_ids), "legacy query ID mismatch")
        require(np.array_equal(archive["source_feature_row"], gene_panel.source_feature_row.to_numpy()), "legacy feature row mismatch")
        require(np.array_equal(archive["original_ensembl_id"], gene_panel.original_ensembl_id.to_numpy(dtype=str)), "legacy gene ID mismatch")
    require(old.dtype == predictions.dtype == np.float32, "regression requires original float32 arrays")
    selected = np.stack([predictions[methods.index(name)] for name in old_methods])
    require(selected.shape == old.shape and np.isfinite(selected).all() and np.isfinite(old).all(), "legacy prediction shape/nonfinite mismatch")
    different = selected.view(np.uint32) != old.view(np.uint32)
    comparisons = []
    for index, method in enumerate(old_methods):
        count = int(different[index].sum())
        comparisons.append({"method": method, "n_float32_values": int(old[index].size),
                            "n_bitwise_mismatches": count, "float32_bitwise_exact": count == 0})
    if different.any():
        first = np.argwhere(different)[0].tolist()
        details = {"heldout_cell_line": heldout, "n_mismatches": int(different.sum()), "first_index": first,
                   "first_method": old_methods[first[0]],
                   "new_value": float(selected[tuple(first)]), "legacy_value": float(old[tuple(first)]),
                   "max_absolute_difference": float(np.max(np.abs(selected.astype(np.float64) - old))),
                   "policy": "FAIL; no tolerance substitution or silent acceptance"}
        raise ValueError("legacy float32 exact regression failed: " + json.dumps(details, allow_nan=False))
    return {"status": "PASS", "policy": "float32_bitwise_exact", "legacy_prediction_path": str(path),
            "legacy_prediction_sha256": audit["prediction_sha256"], "n_compared_methods": len(old_methods),
            "split_gene_landmark_axes_identical": True, "methods": comparisons}


def score_fold(predictions, truth, metadata, gene_panel, cohort, min_gene_n, methods=None):
    """Fixed predictions and a single held-out context; no model selection."""
    methods = list(METHODS if methods is None else methods)
    require(len(set(methods)) == len(methods) and all(name in methods for name in LEGACY_METHODS), "invalid scoring method registry")
    require(predictions.shape == (len(methods), len(metadata), len(gene_panel)), "prediction axes mismatch")
    require(truth.shape == (len(metadata), len(gene_panel)), "truth axes mismatch")
    require(metadata.cell_line.nunique() == 1 and metadata.time.nunique() == 1, "gene scoring must remain in fixed context and time")
    baseline_index = methods.index("same_drug_dose_mean")
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
    for method_index, method in enumerate(methods):
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



def run(config_path):
    start = time.perf_counter()
    config = json.loads(Path(config_path).read_text())
    require(config["alpha"] == 10 and config["n_landmarks"] == 256 and config["seed"] == 20260914, "frozen prediction parameters changed")
    require(config["gene_min_n"] == 20 and config["expected_genes_per_fold"] == 3000, "frozen scoring contract changed")
    legacy_audit, legacy_inputs = validate_contract(config)
    methods = method_order(config)
    specs = encoder_specs(config, require_paths=True)
    output = Path(config["output_root"])
    output.mkdir(parents=True, exist_ok=False)
    effects_root, clean_root, morgan_root = [Path(config[name]) for name in ["effects_root", "clean_views_root", "morgan_root"]]
    reference_root = Path(config["legacy_prediction_root"])
    input_paths = [Path(config_path), effects_root / "audit.json", effects_root / "atomic_index.tsv",
                   effects_root / "arrays.npz", effects_root / "fold_gene_panels.tsv",
                   clean_root / "config.json", clean_root / "row_to_text_registry.tsv", clean_root / "unique_texts.tsv",
                   morgan_root / "audit.json", morgan_root / "arrays.npz",
                   reference_root / "audit.json", reference_root / "input_manifest.json"]
    for item in specs:
        root = Path(item["embedding_root"])
        input_paths.extend([root / "audit.json", root / (item["model_key"] + "__source_name.npz")])
    for heldout in ("A549", "K562", "MCF7"):
        input_paths.extend([reference_root / heldout / "audit.json", reference_root / heldout / "frozen_predictions.npz"])
    input_paths.extend(Path(row["path"]) for row in legacy_inputs)
    input_paths.extend(ROOT / "code" / name for name in ["run_sciplex_prediction_v2.py", "run_sciplex_prediction.py", "landmark_decoder.py", "prediction_metrics.py"])
    input_paths = list(dict.fromkeys(input_paths))
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
        text_by_view, embeddings, representation_manifest = load_representations(metadata, clean_root, specs)
        write_json(output / "representation_manifest.json", representation_manifest)
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
            save_npz(prediction_path, predictions=predictions, method=np.asarray(methods, dtype=str),
                     atomic_id=uid, source_feature_row=gene_panel.source_feature_row.to_numpy(), original_ensembl_id=gene_ids)
            prediction_hash = sha256(prediction_path)
            params = {method + "__" + key: value for method, data in fitted.items() for key, value in data.items()}
            save_npz(fold_output / "fitted_decoder_parameters.npz", **params)
            gene_panel.to_csv(fold_output / "gene_panel.tsv", sep="\t", index=False)
            metadata.iloc[train_rows].to_csv(fold_output / "source_atoms.tsv", sep="\t", index=False)
            metadata.iloc[fold_audit["landmark_rows"]].to_csv(fold_output / "landmarks.tsv", sep="\t", index=False)
            regression = compare_legacy_fold(predictions, methods, uid, gene_panel, fold_audit, reference_root, heldout)
            write_json(fold_output / "legacy_17_exact_regression.json", regression)
            # Target truth is sliced only after all 29 outputs freeze and the regression gate passes.
            truth = all_effects[np.ix_(test_rows, columns)]
            save_npz(fold_output / "evaluation_truth.npz", truth=truth, atomic_id=uid, original_ensembl_id=gene_ids,
                     source_feature_row=gene_panel.source_feature_row.to_numpy(), sensitivity_eligible=restricted[test_rows])
            for cohort, subset in [("primary", np.ones(len(test_rows), dtype=bool)), ("restricted_evaluation", restricted[test_rows])]:
                print("SCORING", heldout, cohort, int(subset.sum()), flush=True)
                condition, gene, summary = score_fold(predictions[:, subset], truth[subset], metadata.iloc[test_rows[subset]].reset_index(drop=True), gene_panel, cohort, config["gene_min_n"], methods=methods)
                condition.to_csv(fold_output / f"{cohort}_condition_metrics.tsv", sep="\t", index=False, na_rep="NA")
                gene.to_csv(fold_output / f"{cohort}_gene_metrics.tsv", sep="\t", index=False, na_rep="NA")
                summaries.append(summary)
            require(sha256(prediction_path) == prediction_hash, "scoring modified frozen predictions")
            fold_audit.update({"status": "PASS", "heldout_cell_line": heldout, "prediction_sha256": prediction_hash,
                               "legacy_17_float32_bitwise_exact": regression["status"] == "PASS",
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
                 "config": config, "method_order": methods, "n_independent_studies": 1, "n_context_folds": 3,
                 "package_versions": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]},
                 "stage": "S3_SIX_ENCODER_SINGLE_STUDY_CORE",
                 "status_scope": "six-encoder sci-Plex core in one study; not completed cross-study benchmark",
                 "legacy_17_float32_bitwise_exact_all_folds": all(row["legacy_17_float32_bitwise_exact"] for row in fold_audits),
                 "representation_manifest_sha256": sha256(output / "representation_manifest.json"),
                 "target_effects_used_in_fit": False, "target_control_track_explicit": True, "input_hashes_unchanged": True,
                 "source_scripts": [{"path": str(ROOT / "code" / name), "sha256": sha256(ROOT / "code" / name)} for name in ["run_sciplex_prediction_v2.py", "run_sciplex_prediction.py", "landmark_decoder.py", "prediction_metrics.py"]],
                 "folds": [{key: row[key] for key in ["heldout_cell_line", "status", "primary_test_n", "restricted_test_n", "seconds"]} for row in fold_audits]}
        write_json(output / "audit.json", audit)
        print("S3_SIX_ENCODER_CORE_PASS", audit["seconds"], flush=True)
    except Exception as exc:
        write_json(output / "failure.json", {"status": "FAILED", "error": repr(exc), "partial_results_not_complete": True})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    with threadpool_limits(limits=4):
        run(arguments.config)
