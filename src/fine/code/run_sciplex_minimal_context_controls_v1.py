#!/usr/bin/env python3
"""Fifteen fixed retrospective controls on the original physical-A/B protocol.

The original 29 fits and score tables remain immutable. Only a same-drug
reference is copied into the new scored matrix. No target response enters fit.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
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

import run_sciplex_prediction_v2 as core
import run_information_fidelity_v1 as e01
from landmark_decoder import LandmarkRidge, cosine_similarity
from prediction_metrics import condition_metrics, gene_metrics, paired_order_accuracy_by_gene

ROOT = Path(__file__).resolve().parents[1]
require, sha256, write_json = core.require, core.sha256, core.write_json
track, finite_mean = core.track, core.finite_mean
REFERENCE = "same_drug_dose_mean"
CONTROL = "control_state_only__control_state"
RANDOM = "random_field512_entity_exposure"
STATE = "__control_state"
COHORTS = ("primary", "restricted_evaluation")


def method_order(config):
    methods = [REFERENCE, CONTROL, RANDOM, RANDOM + STATE]
    for entry in core.encoder_specs(config):
        prefix = entry["method_prefix"] + "_entity_exposure"
        methods.extend([prefix + "__source_mapping_swap" + STATE,
                        prefix + "__uniform_additive" + STATE])
    require(len(methods) == len(set(methods)) == 16, "expected 15 controls plus one copied reference")
    return methods


def read_scores(path):
    """Round-trip legacy decimal output back to its exact float64 value."""
    return pd.read_csv(path, sep="\t", float_precision="round_trip", keep_default_na=False, na_values=["NA"])


def save_tsv(frame, path):
    with Path(path).open("x", encoding="utf-8") as handle:
        frame.to_csv(handle, sep="\t", index=False, na_rep="NA")


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def source_state_swap(metadata, train_rows, test_rows, state):
    """Exchange the two source-context labels; preserve the actual target state."""
    source = sorted(metadata.iloc[train_rows].cell_line.unique())
    require(len(source) == 2 and metadata.iloc[test_rows].cell_line.nunique() == 1,
            "mapping swap requires exactly two sources and one target")
    target = metadata.iloc[test_rows].cell_line.iloc[0]
    require(target not in source, "target line in source mapping")
    state = np.asarray(state)
    representatives = {}
    for line in source + [target]:
        rows = np.flatnonzero(metadata.cell_line.eq(line))
        representatives[line] = state[rows[0]]
        require(np.array_equal(state[rows], np.broadcast_to(state[rows[0]], state[rows].shape)),
                "state must be exactly constant within each physical context")
    mapping = {source[0]: source[1], source[1]: source[0], target: target}
    swapped = np.stack([representatives[mapping[line]] for line in metadata.cell_line])
    require(np.array_equal(swapped[test_rows], state[test_rows]), "target state changed during source swap")
    return swapped, mapping


def fit_controls(metadata, train_rows, test_rows, train_y, embeddings, state, random_features, config):
    """All learned state sees only source features and source treatment effects."""
    train_rows, test_rows = np.asarray(train_rows), np.asarray(test_rows)
    require(not set(train_rows) & set(test_rows), "source/test overlap")
    require(train_y.shape[0] == len(train_rows) and np.isfinite(train_y).all(), "invalid source response")
    landmarks = core.legacy.choose_landmarks(metadata, train_rows, config["n_landmarks"], config["seed"])
    require(set(landmarks) <= set(train_rows), "target landmark leakage")
    swapped, mapping = source_state_swap(metadata, train_rows, test_rows, state)
    ks = cosine_similarity(state, state[landmarks])
    swapped_ks = cosine_similarity(swapped, swapped[landmarks])
    source_difference = float(np.max(np.abs(ks[train_rows] - swapped_ks[train_rows])))
    require(source_difference <= 2e-12, "source state kernel changed beyond fixed swap tolerance")
    kernels = {CONTROL: ks}
    kr = cosine_similarity(random_features, random_features[landmarks])
    kernels[RANDOM], kernels[RANDOM + STATE] = kr, kr * ks
    for item in core.encoder_specs(config):
        prefix = item["method_prefix"] + "_entity_exposure"
        x = embeddings[(item["model_key"], "entity_exposure")]
        kp = cosine_similarity(x, x[landmarks])
        kernels[prefix + "__source_mapping_swap" + STATE] = kp * swapped_ks
        kernels[prefix + "__uniform_additive" + STATE] = (kp + ks) / 2
    outputs, parameters, diagnostics = {}, {}, {}
    for method, features in kernels.items():
        model = LandmarkRidge(config["alpha"]).fit(features[train_rows], train_y)
        if method == CONTROL:
            require(np.array_equal(state[test_rows], np.broadcast_to(state[test_rows[0]], state[test_rows].shape)),
                    "control-only target physical state not unique")
            # Compute ONE target row so BLAS block round-off cannot generate drug ranks.
            unique_feature = cosine_similarity(state[test_rows[:1]], state[landmarks])
            outputs[method] = np.broadcast_to(model.predict(unique_feature), (len(test_rows), train_y.shape[1])).copy()
        else:
            outputs[method] = model.predict(features[test_rows])
        parameters[method] = {"feature_mean": model.feature_mean_, "feature_scale": model.feature_scale_,
                              "target_mean": model.target_mean_, "coef": model.coef_}
        diagnostics[method] = dict(n_landmarks=len(landmarks), alpha=config["alpha"],
            n_parameters=int(model.coef_.size + model.target_mean_.size),
            target_controls_available=method.endswith(STATE),
            train_feature_mean_sha256=array_sha(model.feature_mean_),
            train_feature_scale_sha256=array_sha(model.feature_scale_),
            kernel_diagnostics=core.legacy.kernel_diagnostics(features, train_rows, test_rows, model))
    methods = method_order(config)[1:]
    predictions = np.stack([outputs[name] for name in methods]).astype(np.float32)
    require(np.isfinite(predictions).all(), "nonfinite new prediction")
    control = predictions[methods.index(CONTROL)]
    require(np.array_equal(control.view(np.uint32), np.broadcast_to(control[0].view(np.uint32), control.shape)),
            "control-only predictions are not bitwise constant across target conditions")
    audit = dict(landmark_rows=landmarks.tolist(), landmark_atomic_id=metadata.iloc[landmarks].atomic_id.tolist(),
        source_atomic_id=metadata.iloc[train_rows].atomic_id.tolist(), target_atomic_id=metadata.iloc[test_rows].atomic_id.tolist(),
        source_mapping=mapping, nonidentity_mapping_count=1, source_state_kernel_max_abs_difference=source_difference,
        target_state_kernel_max_abs_difference=float(np.max(np.abs(ks[test_rows] - swapped_ks[test_rows]))),
        source_state_kernel_atol=2e-12, source_state_kernel_rtol=0,
        control_only_target_prediction_bitwise_constant=True, target_state_unique_predict_then_broadcast=True,
        models=diagnostics, test_effects_passed_to_fit=False, hyperparameter_tuning=False)
    features = {"state_kernel_correct": ks, "state_kernel_source_swapped": swapped_ks,
                "control_state_by_atom": state, "source_swapped_control_state_by_atom": swapped,
                **{method + "__features": value for method, value in kernels.items()}}
    return predictions, audit, parameters, features


def score_subset(predictions, truth, metadata, gene_panel, cohort, min_gene_n, methods=None):
    """Exact core.score_fold computation with only the method-registry guard generalized."""
    methods = list(methods)
    require(len(set(methods)) == len(methods) and REFERENCE in methods, "invalid subset scoring method registry")
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


def assert_scorer_body_identical():
    # Only docstring, method-list initialization, and the registry guard differ.
    old = ast.parse(inspect.getsource(core.score_fold)).body[0].body[3:]
    new = ast.parse(inspect.getsource(score_subset)).body[0].body[3:]
    require([ast.dump(node) for node in old] == [ast.dump(node) for node in new],
            "subset scorer scientific computation is not identical to immutable core")
    return dict(status="PASS", unchanged_scientific_body_ast_sha256=hashlib.sha256(
        "\n".join(ast.dump(node) for node in old).encode()).hexdigest(),
        changed_statements=["docstring", "method-list initialization", "method-registry guard only"])


def exact_score_regression(original, metadata, panels, config, base, output):
    contract = config["scorer_regression"]
    line, selected = contract["heldout_cell_line"], contract["methods"]
    directory = original / line
    with np.load(directory / "frozen_predictions.npz", allow_pickle=False) as data:
        methods, ids = data["method"].tolist(), data["atomic_id"]
        prediction = data["predictions"][[methods.index(name) for name in selected]]
    with np.load(directory / "evaluation_truth.npz", allow_pickle=False) as data:
        truth, restricted = data["truth"], data["sensitivity_eligible"]
        require(np.array_equal(data["atomic_id"], ids), "scorer regression truth IDs differ")
    rows = pd.Index(metadata.atomic_id).get_indexer(ids)
    require((rows >= 0).all(), "regression query absent")
    panel = panels[panels.heldout_cell_line.eq(line)].sort_values("rank").reset_index(drop=True)
    records = []
    old_summary = read_scores(original / "fold_summary.tsv")
    for cohort in contract["cohorts"]:
        subset = np.ones(len(ids), bool) if cohort == "primary" else restricted
        result = score_subset(prediction[:, subset], truth[subset], metadata.iloc[rows[subset]].reset_index(drop=True),
                              panel, cohort, base["gene_min_n"], methods=selected)
        for kind, actual in zip(["condition", "gene", "summary"], result):
            expected = (old_summary[old_summary.heldout_cell_line.eq(line) & old_summary.cohort.eq(cohort)]
                        if kind == "summary" else read_scores(directory / f"{cohort}_{kind}_metrics.tsv"))
            expected = expected[expected.method.isin(selected)].reset_index(drop=True)
            pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected, check_exact=True, check_dtype=False)
            records.append(dict(cohort=cohort, table=kind, n_rows=len(actual), n_columns=len(actual.columns),
                float64_values_exact=True, na_pattern_exact=True, categorical_integer_values_exact=True,
                old_table_parse="float_precision=round_trip", tolerance="none"))
    audit = dict(status="PASS", heldout_cell_line=line, methods=selected, tables=records,
        n_compared_cells=sum(row["n_rows"] * row["n_columns"] for row in records),
        scientific_body=assert_scorer_body_identical(), relaxed_scientific_thresholds=False)
    write_json(output / "subset_scorer_exact_regression.json", audit)
    return audit


def preflight(config_path, config, base):
    original, effects = ROOT / config["original_prediction_root"], ROOT / config["effects_root"]
    require(original == Path(base["output_root"]) and effects == Path(base["effects_root"]), "original A/B sources changed")
    require(config["n_new_settings"] == 15 and config["n_scored_settings_including_copied_reference"] == 16, "control count changed")
    for key, expected in {"seed": 20260914, "alpha": 10, "n_landmarks": 256, "gene_min_n": 20,
                          "expected_genes_per_fold": 3000}.items():
        require(config[key] == base[key] == expected, "frozen parameter changed: " + key)
    require(config["state_input_physical_role"] == config["feature_selection_physical_role"] == "A"
            and config["effect_reference_physical_role"] == "B", "physical A/B roles changed")
    require(config["random_field"]["typed_fields"] == list(e01.VIEW_FIELDS["entity_exposure"]), "random field access changed")
    require(config["random_field"]["dimensions"] == 512 and config["random_field"]["seed"] == 20260914,
            "random definition changed")
    require(config["uniform_additive"]["perturbation_weight"] == config["uniform_additive"]["state_weight"] == 0.5,
            "fixed additive weights changed")
    require(config["source_mapping_swap"]["count_per_fold"] == 1 and
            config["source_mapping_swap"]["source_kernel_atol"] == 2e-12 and
            config["source_mapping_swap"]["source_kernel_rtol"] == 0, "source swap contract changed")
    require(config["hyperparameter_tuning"] is False and config["scale_threshold"] == 1e-8, "frozen fit policy changed")
    old_audit = json.loads((original / "audit.json").read_text())
    qa_path = ROOT / config["original_independent_qa"]
    require(old_audit["status"] == json.loads(qa_path.read_text())["status"] == "PASS", "original core/QA not PASS")
    require(old_audit["config"] == base and old_audit["method_order"] == core.method_order(base), "old core config/registry changed")
    paths = [config_path, ROOT / config["base_config"], Path(__file__), Path(e01.__file__),
             ROOT / "tests/test_minimal_context_controls.py", qa_path,
             original / "audit.json", original / "input_manifest.json", original / "representation_manifest.json", original / "fold_summary.tsv"]
    for record in json.loads((original / "input_manifest.json").read_text()) + old_audit["source_scripts"]:
        require(sha256(record["path"]) == record["sha256"], "immutable original source changed: " + record["path"])
        paths.append(Path(record["path"]))
    effect_audit = json.loads((effects / "audit.json").read_text())
    require(effect_audit["status"] == "PASS", "original effects not PASS")
    for name in ["atomic_index.tsv", "arrays.npz", "fold_gene_panels.tsv", "feature_selection_sealed.json"]:
        require(sha256(effects / name) == effect_audit["output_sha256"][name], "original effect hash mismatch: " + name)
        paths.append(effects / name)
    for line in ["A549", "K562", "MCF7"]:
        fold = original / line
        audit = json.loads((fold / "audit.json").read_text())
        require(audit["status"] == "PASS" and sha256(fold / "frozen_predictions.npz") == audit["prediction_sha256"],
                "original frozen prediction not PASS/hash mismatch")
        paths.extend(path for path in fold.iterdir() if path.is_file())
    random_root = ROOT / config["random_reference_root"]
    random_audit = json.loads((random_root / "audit.json").read_text())
    require(random_audit["status"] == "PASS", "E01 random source not PASS")
    random_inputs = json.loads((random_root / "input_manifest.json").read_text())
    original_random_code = [row for row in random_inputs if row["path"] == str(Path(e01.__file__).resolve())]
    require(len(original_random_code) == 1 and sha256(e01.__file__) == original_random_code[0]["sha256"],
            "immutable E01 random-field implementation changed")
    random_record = [row for row in random_audit["outputs"] if row["path"] == "random_field_unit_vectors.npz"]
    require(len(random_record) == 1 and sha256(random_root / "random_field_unit_vectors.npz") == random_record[0]["sha256"],
            "E01 frozen random-field vectors changed")
    paths += [random_root / "audit.json", random_root / "input_manifest.json", random_root / "random_field_unit_vectors.npz"]
    return [{"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size} for path in dict.fromkeys(paths)]


def validate_random_features(metadata, config):
    root = ROOT / config["random_reference_root"]
    with np.load(root / "random_field_unit_vectors.npz", allow_pickle=False) as data:
        keys, vectors = list(zip(data["field"], data["value"])), data["vectors"]
    rebuilt = np.stack([e01.typed_random_vector(20260914, field, value, 512) for field, value in keys])
    require(np.array_equal(rebuilt, vectors), "E01 typed random vectors do not reproduce exactly")
    fields = e01.VIEW_FIELDS["entity_exposure"]
    available = set(keys)
    required = {(field, value) for field in fields for value in metadata[field].astype(str)}
    require(required <= available, "typed-field reference coverage missing")
    x = e01.random_field_features(metadata, "entity_exposure", 20260914, 512)
    lookup = {key: vectors[index] for index, key in enumerate(keys)}
    reconstructed = np.zeros_like(x)
    for field in fields:
        reconstructed += np.stack([lookup[(field, value)] for value in metadata[field].astype(str)])
    reconstructed = e01.normalize(reconstructed, norm="l2")
    require(np.array_equal(x, reconstructed), "random feature function differs from frozen unit-vector composition")
    return x, dict(status="PASS", typed_unit_vectors_n=len(keys), used_typed_unit_vectors_n=len(required),
        frozen_unit_vectors_exact=True, composed_feature_matrix_exact=True, typed_fields=list(fields),
        view="entity_exposure", n_dimensions=512, seed=20260914, learned_parameters=False,
        uses_cell_line_field=False, interpretation=config["random_field"]["interpretation"])


def planned_pairs(config):
    pairs = [("control_only_vs_same_drug", CONTROL, REFERENCE),
             ("random_product_vs_direct", RANDOM + STATE, RANDOM)]
    for item in core.encoder_specs(config):
        prefix = item["method_prefix"] + "_entity_exposure"
        pairs += [(item["method_prefix"] + "_source_mapping_swap_vs_correct_product", prefix + "__source_mapping_swap" + STATE, prefix + STATE),
                  (item["method_prefix"] + "_uniform_additive_vs_product", prefix + "__uniform_additive" + STATE, prefix + STATE)]
    return pairs


def paired_query_deltas(new, old, config):
    rows = []
    for contrast, method, reference in planned_pairs(config):
        first = new[new.method.eq(method)].copy()
        second_source = new if reference in set(new.method) else old
        second = second_source[second_source.method.eq(reference)].copy()
        keys = ["atomic_id", "source_entity_key", "cell_line", "dose_value", "time", "cohort"]
        require(first.atomic_id.is_unique and second.atomic_id.is_unique, "paired query duplicate")
        require(set(first.atomic_id) == set(second.atomic_id), "paired query set mismatch")
        matched = first.merge(second[keys + ["mae", "rmse", "spearman"]], on=keys,
                              suffixes=("", "_reference"), validate="one_to_one")
        require(len(matched) == len(first), "paired identity/metadata merge incomplete")
        result = matched[keys].copy()
        result["contrast"], result["method"], result["reference_method"] = contrast, method, reference
        for name in ["mae", "rmse", "spearman"]:
            result[name + "_method"], result[name + "_reference"] = matched[name], matched[name + "_reference"]
            result[name + "_paired_valid"] = np.isfinite(matched[name]) & np.isfinite(matched[name + "_reference"])
            result[name + "_improvement"] = (matched[name + "_reference"] - matched[name] if name != "spearman"
                                                else matched[name] - matched[name + "_reference"])
        result["positive_delta_means"] = "new_method_better"
        rows.append(result)
    return pd.concat(rows, ignore_index=True)


def run(config_path):
    started = time.perf_counter()
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    base = json.loads((ROOT / config["base_config"]).read_text())
    output = ROOT / config["output_root"]
    require(not output.exists(), "refuse existing output including incomplete prior run")
    output.mkdir(parents=True)
    write_json(output / "RUNNING.json", dict(status="INCOMPLETE_UNTIL_AUDIT_PASS", pid=os.getpid(),
        started_utc=datetime.now(timezone.utc).isoformat(), config=config))
    try:
        inputs = preflight(config_path, config, base)
        write_json(output / "input_manifest.json", inputs)
        write_json(output / "scorer_body_contract.json", assert_scorer_body_identical())
        original, effects = ROOT / config["original_prediction_root"], ROOT / config["effects_root"]
        metadata, panels = core.read_tsv(effects / "atomic_index.tsv"), core.read_tsv(effects / "fold_gene_panels.tsv")
        require(metadata.atomic_id.is_unique and metadata.atomic_id.tolist() == sorted(metadata.atomic_id), "invalid atom IDs")
        primary, restricted = core.boolean_mask(metadata.main_eligible), core.boolean_mask(metadata.sensitivity_eligible)
        require(len(metadata) == 2256 and int(primary.sum()) == 2250 and int(restricted.sum()) == 2202,
                "frozen cohort counts changed")
        cell_lines = ["A549", "K562", "MCF7"]
        for mask in [primary, restricted]:
            core.validate_balanced_cohort(metadata, mask, cell_lines)
        require((~restricted | primary).all(), "restricted not subset of primary")
        print("SCORER_EXACT_REGRESSION_START", flush=True)
        regression = exact_score_regression(original, metadata, panels, config, base, output)
        print("SCORER_EXACT_REGRESSION_PASS", regression["n_compared_cells"], flush=True)
        _, embeddings, representations = core.load_representations(metadata, Path(base["clean_views_root"]), base["encoders"])
        require(representations == json.loads((original / "representation_manifest.json").read_text()), "encoder provenance changed")
        write_json(output / "representation_manifest.json", representations)
        random_features, random_audit = validate_random_features(metadata, config)
        write_json(output / "random_field_exact_reference.json", random_audit)
        core.save_npz(output / "random_field_features.npz", X=random_features, atomic_id=metadata.atomic_id.to_numpy(dtype=str))
        with np.load(effects / "arrays.npz", allow_pickle=False) as data:
            require(np.array_equal(data["atomic_id"], metadata.atomic_id.to_numpy(dtype=str)), "effects atom axis mismatch")
            require(data["cell_line"].tolist() == cell_lines, "physical state axis mismatch")
            all_effects, all_state, feature_rows = data["effect_log2fc"], data["control_state_A"], data["source_feature_row"]
        methods = method_order(base)
        summaries, pairs, audits = [], [], []
        for line in cell_lines:
            fold_started = time.perf_counter()
            print("FOLD_START", line, flush=True)
            fold = output / line
            fold.mkdir()
            train = np.flatnonzero(primary & metadata.cell_line.ne(line).to_numpy())
            test = np.flatnonzero(primary & metadata.cell_line.eq(line).to_numpy())
            require(len(train) == 1500 and len(test) == 750 and int(restricted[test].sum()) == 734, "fold size mismatch")
            panel = panels[panels.heldout_cell_line.eq(line)].sort_values("rank").reset_index(drop=True)
            require(len(panel) == 3000 and panel.source_feature_row.is_unique and panel.union_column.is_unique, "invalid source-A gene panel")
            pd.testing.assert_frame_equal(panel, read_scores(original / line / "gene_panel.tsv"), check_exact=True)
            columns = panel.union_column.to_numpy(dtype=int)
            require(np.array_equal(feature_rows[columns], panel.source_feature_row), "feature row axis mismatch")
            state = all_state[pd.Index(cell_lines).get_indexer(metadata.cell_line)][:, columns]
            new, audit, parameters, features = fit_controls(metadata, train, test, all_effects[np.ix_(train, columns)],
                embeddings, state, random_features, base)
            old_audit = json.loads((original / line / "audit.json").read_text())
            for key in ["landmark_rows", "landmark_atomic_id", "source_atomic_id", "target_atomic_id"]:
                require(audit[key] == old_audit[key], "original axis changed: " + key)
            ids, genes = metadata.iloc[test].atomic_id.to_numpy(dtype=str), panel.original_ensembl_id.to_numpy(dtype=str)
            with np.load(original / line / "frozen_predictions.npz", allow_pickle=False) as data:
                require(np.array_equal(data["atomic_id"], ids) and np.array_equal(data["source_feature_row"], panel.source_feature_row)
                    and np.array_equal(data["original_ensembl_id"], genes), "old reference prediction axes differ")
                reference = data["predictions"][data["method"].tolist().index(REFERENCE)].copy()
            predictions = np.concatenate([reference[None], new])
            require(np.array_equal(predictions[0].view(np.uint32), reference.view(np.uint32)), "copied reference float32 bits changed")
            prediction_path = fold / "frozen_predictions.npz"
            core.save_npz(prediction_path, predictions=predictions, method=np.asarray(methods, dtype=str), atomic_id=ids,
                source_feature_row=panel.source_feature_row.to_numpy(), original_ensembl_id=genes)
            prediction_hash = sha256(prediction_path)
            core.save_npz(fold / "fitted_decoder_parameters.npz", **{method + "__" + key: value
                for method, values in parameters.items() for key, value in values.items()})
            core.save_npz(fold / "landmark_features.npz", **features, atomic_id=metadata.atomic_id.to_numpy(dtype=str),
                landmark_atomic_id=metadata.iloc[audit["landmark_rows"]].atomic_id.to_numpy(dtype=str))
            for name, table in [("gene_panel", panel), ("source_atoms", metadata.iloc[train]),
                                ("query_atoms", metadata.iloc[test]), ("landmarks", metadata.iloc[audit["landmark_rows"]])]:
                save_tsv(table, fold / (name + ".tsv"))
            # Predictions have frozen before extraction of heldout treatment truth.
            truth = all_effects[np.ix_(test, columns)]
            with np.load(original / line / "evaluation_truth.npz", allow_pickle=False) as data:
                require(np.array_equal(data["truth"], truth) and np.array_equal(data["atomic_id"], ids)
                    and np.array_equal(data["sensitivity_eligible"], restricted[test]), "original truth/cohort changed")
            core.save_npz(fold / "evaluation_truth.npz", truth=truth, atomic_id=ids, original_ensembl_id=genes,
                source_feature_row=panel.source_feature_row.to_numpy(), sensitivity_eligible=restricted[test])
            for cohort, subset in [("primary", np.ones(len(test), bool)), ("restricted_evaluation", restricted[test])]:
                print("SCORING", line, cohort, flush=True)
                condition, gene, summary = score_subset(predictions[:, subset], truth[subset],
                    metadata.iloc[test[subset]].reset_index(drop=True), panel, cohort, base["gene_min_n"], methods=methods)
                save_tsv(condition, fold / f"{cohort}_condition_metrics.tsv")
                save_tsv(gene, fold / f"{cohort}_gene_metrics.tsv")
                control_gene = gene[gene.method.eq(CONTROL)]
                require(control_gene.spearman.isna().all() and not control_gene.spearman_valid.any(),
                        "constant control-only gene correlations must remain NA")
                require(control_gene.loc[control_gene.n_pairs.gt(0), "pair_order_accuracy"].eq(0.5).all(),
                        "constant control-only ordering must count prediction ties as half")
                old_conditions = read_scores(original / line / f"{cohort}_condition_metrics.tsv")
                paired = paired_query_deltas(condition, old_conditions, base)
                save_tsv(paired, fold / f"{cohort}_paired_query_deltas.tsv")
                summaries.append(summary)
                pairs.append(paired)
            require(sha256(prediction_path) == prediction_hash, "scoring altered frozen prediction")
            audit.update(status="PASS", heldout_cell_line=line, prediction_sha256=prediction_hash,
                method_order=methods, original_reference_path=str(original / line / "frozen_predictions.npz"),
                original_reference_archive_sha256=old_audit["prediction_sha256"],
                copied_reference_array_sha256=array_sha(reference), copied_reference_float32_bitwise_exact=True,
                original_axes_exact=True, original_truth_exact=True, source_query_landmarks_unchanged=True,
                n_new_fits=15, n_scored_settings=16, n_genes=3000, primary_test_n=750, restricted_test_n=734,
                restricted_evaluation_not_refit=True, source_state_physical_role="A", effect_reference_physical_role="B",
                feature_selection_physical_role="A", predictions_frozen_before_target_scoring=True,
                seconds=time.perf_counter() - fold_started)
            write_json(fold / "audit.json", audit)
            audits.append(audit)
            print("FOLD_PASS", line, audit["seconds"], flush=True)
        summary, paired = pd.concat(summaries, ignore_index=True), pd.concat(pairs, ignore_index=True)
        save_tsv(summary, output / "fold_summary.tsv")
        numeric = ["mae_mean", "rmse_mean", "condition_spearman_mean", "mae_improvement_vs_same_drug",
                   "gene_spearman_equal_dose_mean", "gene_order_equal_dose_mean"]
        groups = ["cohort", "method", "track"]
        macro = summary.groupby(groups, sort=False)[numeric].mean().reset_index()
        counts = summary.groupby(groups, sort=False)[numeric].count().add_suffix("__valid_folds").reset_index()
        macro = macro.merge(counts, on=groups, validate="one_to_one")
        macro["n_context_folds"], macro["n_independent_studies"] = 3, 1
        macro["interpretation"] = "descriptive_equal_context_macro_no_p_values"
        save_tsv(macro, output / "descriptive_macro_summary.tsv")
        delta_columns = ["mae_improvement", "rmse_improvement", "spearman_improvement"]
        groups = ["cohort", "cell_line", "contrast", "method", "reference_method"]
        fold_pairs = paired.groupby(groups, sort=False)[delta_columns].mean().reset_index()
        counts = paired.groupby(groups, sort=False)[delta_columns].count().add_suffix("__paired_valid_n").reset_index()
        fold_pairs = fold_pairs.merge(counts, on=groups, validate="one_to_one")
        fold_pairs = fold_pairs.merge(paired.groupby(groups, sort=False).size().rename("n_query_pairs").reset_index(),
                                      on=groups, validate="one_to_one")
        save_tsv(fold_pairs, output / "paired_query_fold_summary.tsv")
        groups.remove("cell_line")
        pair_macro = fold_pairs.groupby(groups, sort=False)[delta_columns].mean().reset_index()
        valid = fold_pairs.groupby(groups, sort=False)[delta_columns].count().add_suffix("__valid_folds").reset_index()
        pair_macro = pair_macro.merge(valid, on=groups, validate="one_to_one")
        pair_macro["n_context_folds"], pair_macro["n_independent_studies"] = 3, 1
        pair_macro["interpretation"] = "per_query_paired_then_equal_context_descriptive_no_p_values"
        save_tsv(pair_macro, output / "paired_query_descriptive_macro_summary.tsv")
        denominator_columns = ["heldout_cell_line", "cohort", "method", "n_query_atoms", "n_unique_drugs", "n_genes",
            "condition_spearman_valid_n", "condition_spearman_paired_valid_n", "gene_score_valid_n", "gene_score_total_n"]
        denominators = summary[denominator_columns].copy()
        denominators["condition_spearman_na_n"] = denominators.n_query_atoms - denominators.condition_spearman_valid_n
        denominators["gene_spearman_na_n"] = denominators.gene_score_total_n - denominators.gene_score_valid_n
        save_tsv(denominators, output / "complete_method_cohort_denominators.tsv")
        require(len(summary) == 96 and len(fold_pairs) == 84, "incomplete full settings/cohort/contrast matrix")
        for record in inputs:
            require(sha256(record["path"]) == record["sha256"], "frozen input mutated during run: " + record["path"])
        hashes = {str(path.relative_to(output)): sha256(path) for path in sorted(output.rglob("*")) if path.is_file()}
        audit = dict(status="PASS", independent_qa_status="PENDING_SEPARATE_CHECKER", config=config,
            stage=config["stage"], method_order=methods, n_new_settings=15, n_scored_settings=16,
            n_source_only_fits=45, n_context_folds=3, n_independent_studies=1, n_method_fold_cohort_rows=len(summary),
            n_query_contrast_rows=len(paired), n_contrast_fold_cohort_rows=len(fold_pairs),
            source_state_physical_role="A", effect_reference_physical_role="B", feature_selection_physical_role="A",
            original_prediction_reference_copied_exact_all_folds=True, original_source_query_gene_landmark_axes_exact=True,
            subset_scorer_exact_regression="PASS", target_effects_used_in_fit=False, hyperparameter_tuning=False,
            original_29_settings_recomputed=False, input_hashes_unchanged=True, output_sha256=hashes,
            control_only_constant_predictions_and_gene_na_all_folds=True,
            seconds=time.perf_counter()-started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            package_versions={name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]},
            folds=[{key: row[key] for key in ["heldout_cell_line", "status", "primary_test_n", "restricted_test_n", "seconds"]} for row in audits])
        write_json(output / "audit.json", audit)
        print("MINIMAL_CONTEXT_CONTROLS_PASS", audit["seconds"], flush=True)
    except Exception as exc:
        write_json(output / "failure.json", dict(status="FAILED", error=repr(exc), partial_results_not_complete=True))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/minimal_context_controls_v1.json")
    arguments = parser.parse_args()
    with threadpool_limits(limits=4):
        run(arguments.config)
