#!/usr/bin/env python3
"""Grouped entity CV with seen-context and joint context/entity holdouts.

Frozen, retrospective single-study evaluation. No target responses are passed
to fitting. Existing seen-entity LOCO predictions are reused, never refitted.
Each fixed-dose gene-ranking comparison stays within ONE fitted entity fold.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import run_sciplex_prediction_v2 as core
from landmark_decoder import LandmarkRidge, cosine_similarity
from prediction_metrics import condition_metrics, gene_metrics, paired_order_accuracy_by_gene

ROOT = Path(__file__).resolve().parents[1]
require, sha256, write_json = core.require, core.sha256, core.write_json
TASKS = ("unseen_entity_seen_context", "unseen_entity_unseen_context")
REFERENCE_TASK = "seen_entity_unseen_context_reused"
LINES = ("A549", "K562", "MCF7")


def tsv(path):
    return pd.read_csv(path, sep="\t", keep_default_na=False, float_precision="round_trip")


def save_tsv(frame, path):
    frame.to_csv(path, sep="\t", index=False, na_rep="NA", float_format="%.17g")


def method_order(config):
    kernels = ["identity_exposure", "tfidf_entity_exposure", "morgan_entity_exposure"]
    kernels += [item["method_prefix"] + "_entity_exposure" for item in core.encoder_specs(config)]
    return ["zero", "source_mean", "source_median"] + [key + suffix for key in kernels
        for suffix in ("", "__control_state")]


def entity_folds(metadata, seed, n_folds, entity_groups=None):
    """Round-robin parent-group hash order; all doses/contexts/aliases grouped."""
    if entity_groups is None:
        entity_groups = {key: key for key in metadata.source_entity_key.unique()}
    require(set(entity_groups) == set(metadata.source_entity_key), "split-group coverage mismatch")
    groups = sorted(set(entity_groups.values()), key=lambda key:
        (hashlib.sha256(f"{seed}|{key}".encode()).hexdigest(), key))
    require(len(groups) >= n_folds >= 2, "invalid entity-fold count")
    group_fold = {key: rank % n_folds for rank, key in enumerate(groups)}
    entities = sorted(entity_groups)
    return pd.DataFrame({"source_entity_key": entities,
        "entity_group": [entity_groups[key] for key in entities],
        "entity_fold": [group_fold[entity_groups[key]] for key in entities],
        "split_hash": [hashlib.sha256(f"{seed}|{entity_groups[key]}".encode()).hexdigest() for key in entities]})


def split_rows(metadata, primary, assignment, target, fold, task):
    require(task in TASKS, "unsupported task")
    require(assignment.source_entity_key.is_unique, "ambiguous entity assignment")
    heldout = set(assignment.loc[assignment.entity_fold.eq(fold), "source_entity_key"])
    require(bool(heldout), "empty entity fold")
    is_heldout = metadata.source_entity_key.isin(heldout).to_numpy()
    is_target = metadata.cell_line.eq(target).to_numpy()
    train_mask = primary & ~is_heldout
    if task == "unseen_entity_unseen_context":
        train_mask &= ~is_target
    train, query = np.flatnonzero(train_mask), np.flatnonzero(primary & is_target & is_heldout)
    require(len(train) and len(query), "empty source/query")
    require(not set(train) & set(query), "source/query row leakage")
    require(not set(metadata.iloc[train].source_entity_key) & heldout, "entity leakage across doses/contexts")
    if "entity_group" in metadata:
        require(not set(metadata.iloc[train].entity_group) & set(metadata.iloc[query].entity_group), "parent alias leakage")
    if task == "unseen_entity_unseen_context":
        require(target not in set(metadata.iloc[train].cell_line), "target context leaked")
    else:
        require(target in set(metadata.iloc[train].cell_line), "seen-context task lacks target context")
    return train, query


def unknown_identity_features(metadata, train):
    """Training drug/dose vocabularies: unseen entity has zero entity block."""
    drugs = sorted(metadata.iloc[train].source_entity_key.unique())
    doses = sorted(metadata.iloc[train].dose_value.unique())
    require(metadata.dose_value.isin(doses).all(), "unsupported unseen dose")
    drug_ids = pd.Index(drugs).get_indexer(metadata.source_entity_key)
    dose_ids = pd.Index(doses).get_indexer(metadata.dose_value)
    x = np.zeros((len(metadata), len(drugs) + len(doses)), dtype=np.float64)
    rows = np.flatnonzero(drug_ids >= 0)
    x[rows, drug_ids[rows]] = 1
    x[np.arange(len(metadata)), len(drugs) + dose_ids] = 1
    return x, {"source_drugs": drugs, "source_doses": [float(v) for v in doses],
        "unknown_entity_rows": int((drug_ids < 0).sum()),
        "unknown_policy": "zero entity block; known dose one-hot retained; no unknown learned category"}


def fit_fold(metadata, train, query, train_y, texts, embeddings, state, config):
    """No held-out truth argument; only source_y can fit parameters."""
    methods = method_order(config)
    require(not set(metadata.iloc[train].source_entity_key) & set(metadata.iloc[query].source_entity_key),
            "unseen-entity fitting received target entity")
    require(np.asarray(train_y).shape[0] == len(train), "source response axis mismatch")
    landmarks = core.legacy.choose_landmarks(metadata, train, config["n_landmarks"], config["seed"])
    require(set(landmarks) <= set(train), "non-source landmark")
    identity, identity_audit = unknown_identity_features(metadata, train)
    kernels = {"identity_exposure": cosine_similarity(identity, identity[landmarks])}
    kernels["tfidf_entity_exposure"], tfidf_audit = core.legacy.fit_tfidf_kernel(
        texts["entity_exposure"], train, landmarks)
    kernels["morgan_entity_exposure"] = core.legacy.morgan_exposure_kernel(
        embeddings[("morgan", "entity")], metadata.dose_value.to_numpy(), landmarks)
    for item in config["encoders"]:
        x = embeddings[(item["model_key"], "entity_exposure")]
        kernels[item["method_prefix"] + "_entity_exposure"] = cosine_similarity(x, x[landmarks])
    ks = cosine_similarity(state, state[landmarks])
    shape = (len(query), np.asarray(train_y).shape[1])
    y = np.asarray(train_y, dtype=np.float64)
    outputs = {"zero": np.zeros(shape),
        "source_mean": np.broadcast_to(y.mean(axis=0), shape).copy(),
        "source_median": np.broadcast_to(np.median(y, axis=0), shape).copy()}
    params, diagnostics = {}, {}
    for method in methods[3:]:
        has_state = method.endswith("__control_state")
        key = method.removesuffix("__control_state")
        x = kernels[key] * ks if has_state else kernels[key]
        model = LandmarkRidge(config["alpha"]).fit(x[train], y)
        # Identical feature rows must give bitwise-identical predictions; this
        # prevents batch-matmul rounding from creating fake within-dose ranks.
        unique, inverse = np.unique(x[query], axis=0, return_inverse=True)
        outputs[method] = model.predict(unique)[inverse]
        params[method] = {"feature_mean": model.feature_mean_, "feature_scale": model.feature_scale_,
            "target_mean": model.target_mean_, "coef": model.coef_}
        diagnostics[method] = {"n_unique_query_features": len(unique),
            "minimum_feature_scale": float(model.feature_scale_.min()),
            "n_landmarks": len(landmarks), "target_control_state_used": has_state}
    predictions = np.stack([outputs[name] for name in methods]).astype(np.float32)
    require(np.isfinite(predictions).all(), "nonfinite predictions")
    audit = {"landmark_rows": landmarks.tolist(), "source_atomic_id": metadata.iloc[train].atomic_id.tolist(),
        "target_atomic_id": metadata.iloc[query].atomic_id.tolist(),
        "landmark_atomic_id": metadata.iloc[landmarks].atomic_id.tolist(), "identity": identity_audit,
        "tfidf": tfidf_audit, "models": diagnostics, "target_responses_passed_to_fit": False,
        "n_source_atoms": len(train), "n_source_entities": int(metadata.iloc[train].source_entity_key.nunique()),
        "source_contexts": sorted(metadata.iloc[train].cell_line.unique()),
        "same_drug_baseline": "UNSUPPORTED: no source responses for held-out entities",
        "ridge_summed_loss_alpha_over_n_source": config["alpha"] / len(train),
        "identical_feature_rows_predicted_once": True}
    return predictions, params, {**kernels, "control_state_kernel": ks}, audit


def score(predictions, truth, metadata, panel, methods, cohort, gene_min_n):
    """Same immutable core metrics; references here use zero, not same-drug."""
    require(predictions.shape == (len(methods), len(metadata), len(panel)), "score axes mismatch")
    require(metadata.cell_line.nunique() == metadata.time.nunique() == 1, "mixed gene-score context/time")
    zero = condition_metrics(np.zeros_like(truth), truth)
    references = {name: condition_metrics(predictions[methods.index(name)], truth)
        for name in ("source_mean", "source_median") if name in methods}
    conditions, genes_all, summaries = [], [], []
    for i, method in enumerate(methods):
        scores = condition_metrics(predictions[i], truth)
        rows = metadata[["atomic_id", "source_entity_key", "cell_line", "dose_value", "time"]].copy()
        rows["method"], rows["cohort"] = method, cohort
        for key, value in scores.items():
            rows[key] = value
        rows["mae_improvement_vs_zero"] = zero["mae"] - scores["mae"]
        rows["rmse_improvement_vs_zero"] = zero["rmse"] - scores["rmse"]
        for name, reference in references.items():
            rows["mae_improvement_vs_" + name] = reference["mae"] - scores["mae"]
            rows["rmse_improvement_vs_" + name] = reference["rmse"] - scores["rmse"]
        conditions.append(rows)
        dose_rho, dose_order, valid_n, eligible_doses = [], [], 0, 0
        for dose in sorted(metadata.dose_value.unique()):
            take = np.flatnonzero(metadata.dose_value.to_numpy() == dose)
            require(metadata.iloc[take].source_entity_key.is_unique, "duplicate drug at fixed exposure")
            n_groups = metadata.iloc[take].entity_group.nunique() if "entity_group" in metadata else len(take)
            enough = len(take) >= gene_min_n and n_groups >= gene_min_n
            if enough:
                gm = gene_metrics(predictions[i, take], truth[take])
                order = paired_order_accuracy_by_gene(predictions[i, take], truth[take])
            else:
                gm = dict(spearman=np.full(len(panel), np.nan), spearman_valid=np.zeros(len(panel), bool),
                    spearman_n=np.zeros(len(panel), int), n_conditions=np.full(len(panel), len(take)))
                order = dict(accuracy=np.full(len(panel), np.nan), n_pairs=np.zeros(len(panel), int))
            g = panel[["source_feature_row", "original_ensembl_id", "gene_symbol"]].copy()
            g["method"], g["cohort"], g["dose_value"], g["eligible_min_n"] = method, cohort, dose, enough
            g["n_parent_groups"] = n_groups
            for key, value in gm.items():
                g[key] = value
            g["pair_order_accuracy"], g["n_pairs"] = order["accuracy"], order["n_pairs"]
            genes_all.append(g)
            dose_rho.append(core.finite_mean(gm["spearman"]))
            dose_order.append(core.finite_mean(order["accuracy"]))
            valid_n += int(gm["spearman_valid"].sum())
            eligible_doses += int(enough)
        summary = dict(method=method, cohort=cohort, n_query_atoms=len(metadata),
            n_unique_drugs=metadata.source_entity_key.nunique(), n_genes=len(panel),
            n_parent_groups=metadata.entity_group.nunique() if "entity_group" in metadata else metadata.source_entity_key.nunique(),
            mae_mean=float(scores["mae"].mean()), rmse_mean=float(scores["rmse"].mean()),
            condition_spearman_mean=core.finite_mean(scores["spearman"]),
            condition_spearman_valid_n=int(scores["spearman_valid"].sum()),
            mae_improvement_vs_zero=float(rows.mae_improvement_vs_zero.mean()),
            gene_spearman_equal_dose_mean=core.finite_mean(dose_rho),
            gene_order_equal_dose_mean=core.finite_mean(dose_order),
            gene_score_valid_n=valid_n, gene_score_total_n=len(dose_rho) * len(panel),
            eligible_dose_groups=eligible_doses, restricted_evaluation_not_refit=cohort != "primary")
        for name in references:
            summary["mae_improvement_vs_" + name] = float(rows["mae_improvement_vs_" + name].mean())
        summaries.append(summary)
    return pd.concat(conditions, ignore_index=True), pd.concat(genes_all, ignore_index=True), pd.DataFrame(summaries)


def structure_gate(metadata, path):
    """Structure-only salt/stereo alias safeguard; original features unchanged."""
    from rdkit import Chem, rdBase
    from rdkit.Chem.MolStandardize import rdMolStandardize
    manifest = tsv(path)
    require(set(manifest.source_entity_key) == set(metadata.source_entity_key), "structure coverage mismatch")
    for column in ("source_entity_key", "canonical_isomeric_smiles", "source_cas_number", "fingerprint_sha256"):
        require(manifest[column].is_unique and manifest[column].astype(str).ne("").all(),
                "entity alias/collision requires review before splitting: " + column)
    chooser, uncharger = rdMolStandardize.LargestFragmentChooser(preferOrganic=True), rdMolStandardize.Uncharger()
    parents = []
    for smiles in manifest.canonical_isomeric_smiles:
        molecule = Chem.MolFromSmiles(smiles)
        require(molecule is not None, "invalid source structure")
        parent = uncharger.uncharge(chooser.choose(molecule))
        parents.append(Chem.MolToSmiles(parent, canonical=True, isomericSmiles=False))
    manifest["parent_connectivity_smiles"] = parents
    manifest["entity_group"] = ["parent:" + hashlib.sha256(value.encode()).hexdigest() for value in parents]
    require(manifest.entity_group.nunique() == 185, "frozen parent-group count changed; review required")
    require(sorted(manifest.groupby("entity_group").size().tolist()) == [1] * 182 + [2] * 3, "alias sizes changed")
    groups = manifest.set_index("source_entity_key").entity_group.to_dict()
    audit = {"n_entities": len(manifest), "n_parent_groups": 185, "rdkit_version": rdBase.rdkitVersion,
        "unique_full_structure_CAS_fingerprint": True,
        "parent_rule": "LargestFragmentChooser(preferOrganic=True), Uncharger, canonical SMILES with isomericSmiles=False",
        "source_Morgan_features_changed": False, "tautomer_canonicalization_used": False,
        "scope": "conservative parent-connectivity grouped entity holdout; NOT scaffold, mechanism, or family OOD"}
    return groups, manifest, audit


def run(config_path):
    started = time.perf_counter()
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    base_path = Path(config["base_config"])
    base = json.loads(base_path.read_text())
    for key, expected in {"seed": 20260914, "alpha": 10, "n_landmarks": 256,
                          "gene_min_n": 20, "expected_genes_per_fold": 3000}.items():
        require(config[key] == base[key] == expected, "frozen core parameter changed: " + key)
    require(config["tasks"] == list(TASKS) and config["n_entity_folds"] == 5, "task contract changed")
    config["encoders"] = base["encoders"]
    methods = method_order(config)
    require(len(methods) == len(set(methods)) == 21 and set(methods) <= set(core.METHODS), "method registry mismatch")
    output, effects, original = map(Path, [config["output_root"], base["effects_root"], base["output_root"]])
    require(not output.exists(), "refuse overwriting an existing run")
    original_audit = json.loads((original / "audit.json").read_text())
    require(original_audit["status"] == "PASS", "original predictions not PASS")
    # Verify lineage rather than merely reading an old PASS label.
    for row in json.loads((original / "input_manifest.json").read_text()):
        require(sha256(row["path"]) == row["sha256"], "original input changed: " + row["path"])
    effect_audit = json.loads((effects / "audit.json").read_text())
    require(effect_audit["status"] == "PASS", "effects not PASS")
    for name in ("atomic_index.tsv", "arrays.npz", "fold_gene_panels.tsv"):
        require(sha256(effects / name) == effect_audit["output_sha256"][name], "effect source mutated")
    metadata, panels = tsv(effects / "atomic_index.tsv"), tsv(effects / "fold_gene_panels.tsv")
    primary, restricted = core.boolean_mask(metadata.main_eligible), core.boolean_mask(metadata.sensitivity_eligible)
    require(len(metadata) == 2256 and primary.sum() == 2250 and restricted.sum() == 2202, "cohort drift")
    core.validate_balanced_cohort(metadata, primary, list(LINES))
    structure_path = Path(base["morgan_root"]) / "entity_fingerprint_manifest.tsv"
    entity_groups, group_manifest, structural_audit = structure_gate(metadata, structure_path)
    metadata["entity_group"] = metadata.source_entity_key.map(entity_groups)
    assignment = entity_folds(metadata, config["seed"], config["n_entity_folds"], entity_groups)
    text, embeddings, representation_records = core.load_representations(metadata, base["clean_views_root"], base["encoders"])
    embeddings[("morgan", "entity")] = core.load_morgan(metadata, Path(base["morgan_root"]))
    source_paths = [config_path, base_path, structure_path, effects / "audit.json", effects / "arrays.npz",
        effects / "atomic_index.tsv", effects / "fold_gene_panels.tsv", original / "audit.json", original / "input_manifest.json"]
    source_paths += [Path(record["embedding_path"]) for record in representation_records]
    source_paths += [ROOT / "code" / name for name in (Path(__file__).name, "run_sciplex_prediction_v2.py",
        "run_sciplex_prediction.py", "landmark_decoder.py", "prediction_metrics.py")]
    source_paths += [original / line / name for line in LINES for name in ("audit.json", "frozen_predictions.npz", "evaluation_truth.npz")]
    input_manifest = [{"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size} for path in source_paths]
    output.mkdir(parents=True, exist_ok=False)
    try:
        write_json(output / "input_manifest.json", input_manifest)
        write_json(output / "representation_manifest.json", representation_records)
        save_tsv(assignment, output / "entity_folds.tsv")
        save_tsv(group_manifest, output / "source_entity_parent_groups.tsv")
        # Freeze the entire split contract BEFORE any effect array is opened.
        splits, split_records = {}, []
        for line in LINES:
            for fold in range(config["n_entity_folds"]):
                for task in TASKS:
                    train, query = split_rows(metadata, primary, assignment, line, fold, task)
                    splits[(line, fold, task)] = train, query
                    split_records.append(dict(target_cell_line=line, entity_fold=fold, task=task,
                        source_atomic_id=metadata.iloc[train].atomic_id.tolist(),
                        query_atomic_id=metadata.iloc[query].atomic_id.tolist(),
                        query_dose_counts={str(k): int(v) for k, v in metadata.iloc[query].dose_value.value_counts().items()},
                        n_restricted_queries=int(restricted[query].sum())))
        write_json(output / "split_contract_sealed.json", dict(config=config, method_order=methods,
            created_utc=datetime.now(timezone.utc).isoformat(), structure_gate=structural_audit,
            split_records=split_records, targets_opened_for_this_stage=False,
            gene_axes="original source-two-line control-A panels in ALL tasks, including seen-context"))
        with np.load(effects / "arrays.npz", allow_pickle=False) as z:
            require(np.array_equal(z["atomic_id"], metadata.atomic_id.to_numpy(dtype=str)), "effect row mismatch")
            require(z["cell_line"].tolist() == list(LINES), "control line axis mismatch")
            all_y, all_state, feature_rows = z["effect_log2fc"], z["control_state_A"], z["source_feature_row"]
        summaries, audits, pair_summaries = [], [], []
        for line in LINES:
            panel = panels[panels.heldout_cell_line.eq(line)].sort_values("rank").reset_index(drop=True)
            cols = panel.union_column.to_numpy(dtype=int)
            require(len(cols) == 3000 and np.array_equal(feature_rows[cols], panel.source_feature_row), "gene axis drift")
            state = all_state[pd.Index(LINES).get_indexer(metadata.cell_line)][:, cols]
            old_fold = json.loads((original / line / "audit.json").read_text())
            require(sha256(original / line / "frozen_predictions.npz") == old_fold["prediction_sha256"], "old predictions changed")
            with np.load(original / line / "frozen_predictions.npz", allow_pickle=False) as z:
                old_methods, old_ids, old_predictions = z["method"].tolist(), z["atomic_id"], z["predictions"]
                require(np.array_equal(z["source_feature_row"], panel.source_feature_row), "old gene axis mismatch")
            for fold in range(config["n_entity_folds"]):
                query = splits[(line, fold, TASKS[0])][1]
                old_rows = pd.Index(old_ids).get_indexer(metadata.iloc[query].atomic_id)
                require((old_rows >= 0).all(), "query absent from old LOCO")
                reused = old_predictions[[old_methods.index(name) for name in methods]][:, old_rows]
                scored_conditions = {}
                for task in (*TASKS, REFERENCE_TASK):
                    tick = time.perf_counter()
                    directory = output / task / line / f"entity_fold_{fold}"
                    directory.mkdir(parents=True)
                    print("START", task, line, fold, flush=True)
                    if task in TASKS:
                        train, queries = splits[(line, fold, task)]
                        require(np.array_equal(query, queries), "task comparison query mismatch")
                        pred, params, kernels, audit = fit_fold(metadata, train, query, all_y[np.ix_(train, cols)],
                            text, embeddings, state, config)
                        core.save_npz(directory / "fitted_decoder_parameters.npz", **{method + "__" + name: value
                            for method, values in params.items() for name, value in values.items()})
                        core.save_npz(directory / "kernels.npz", **kernels,
                            atomic_id=metadata.atomic_id.to_numpy(dtype=str),
                            landmark_atomic_id=metadata.iloc[audit["landmark_rows"]].atomic_id.to_numpy(dtype=str))
                        save_tsv(metadata.iloc[train], directory / "source_atoms.tsv")
                        save_tsv(metadata.iloc[audit["landmark_rows"]], directory / "landmarks.tsv")
                    else:
                        pred = reused
                        audit = dict(reused_exact=True, source_prediction_path=str(original / line / "frozen_predictions.npz"),
                            source_prediction_sha256=old_fold["prediction_sha256"], n_source_atoms=1500,
                            n_source_entities=188, source_contexts=[v for v in LINES if v != line],
                            target_atomic_id=metadata.iloc[query].atomic_id.tolist(), refitted=False)
                    core.save_npz(directory / "frozen_predictions.npz", predictions=pred, method=np.asarray(methods, dtype=str),
                        atomic_id=metadata.iloc[query].atomic_id.to_numpy(dtype=str), source_feature_row=panel.source_feature_row.to_numpy())
                    frozen_hash = sha256(directory / "frozen_predictions.npz")
                    save_tsv(panel, directory / "gene_panel.tsv")
                    save_tsv(metadata.iloc[query], directory / "query_atoms.tsv")
                    # Score only after the predictions have been saved/frozen.
                    truth = all_y[np.ix_(query, cols)]
                    core.save_npz(directory / "evaluation_truth.npz", truth=truth,
                        atomic_id=metadata.iloc[query].atomic_id.to_numpy(dtype=str), source_feature_row=panel.source_feature_row.to_numpy(),
                        sensitivity_eligible=restricted[query])
                    with np.load(original / line / "evaluation_truth.npz", allow_pickle=False) as z:
                        require(np.array_equal(truth.view(np.uint32), z["truth"][old_rows].view(np.uint32)), "truth comparison changed")
                    for cohort, mask in [("primary", np.ones(len(query), bool)), ("restricted_evaluation", restricted[query])]:
                        cond, gene, summary = score(pred[:, mask], truth[mask], metadata.iloc[query[mask]].reset_index(drop=True),
                            panel, methods, cohort, config["gene_min_n"])
                        for frame in (cond, gene, summary):
                            frame["task"], frame["target_cell_line"], frame["entity_fold"] = task, line, fold
                        summary["n_source_atoms"], summary["n_source_entities"] = audit["n_source_atoms"], audit["n_source_entities"]
                        save_tsv(cond, directory / f"{cohort}_condition_metrics.tsv.gz")
                        save_tsv(gene, directory / f"{cohort}_gene_metrics.tsv.gz")
                        scored_conditions[(task, cohort)] = cond
                        summaries.append(summary)
                    require(sha256(directory / "frozen_predictions.npz") == frozen_hash, "scoring mutated predictions")
                    audit.update(status="PASS", task=task, target_cell_line=line, entity_fold=fold,
                        n_query_atoms=len(query), n_restricted_queries=int(restricted[query].sum()),
                        prediction_sha256=frozen_hash, seconds=time.perf_counter() - tick)
                    write_json(directory / "audit.json", audit)
                    audits.append(audit)
                    print("PASS", task, line, fold, round(audit["seconds"], 2), flush=True)
                for first, second in [(TASKS[0], TASKS[1]), (REFERENCE_TASK, TASKS[1])]:
                    for cohort in ("primary", "restricted_evaluation"):
                        one = scored_conditions[(first, cohort)].set_index(["atomic_id", "method"])
                        two = scored_conditions[(second, cohort)].set_index(["atomic_id", "method"])
                        require(one.index.equals(two.index), "paired query/method ordering mismatch")
                        pair = two[["source_entity_key", "cell_line", "dose_value", "time"]].copy()
                        pair["mae_increase_second_minus_first"] = two.mae - one.mae
                        pair["rmse_increase_second_minus_first"] = two.rmse - one.rmse
                        pair["condition_spearman_change_second_minus_first"] = two.spearman - one.spearman
                        pair = pair.reset_index()
                        pair["first_task"], pair["second_task"], pair["cohort"] = first, second, cohort
                        pair["target_cell_line"], pair["entity_fold"] = line, fold
                        pair_summaries.append(pair)
        summary = pd.concat(summaries, ignore_index=True)
        save_tsv(summary, output / "fold_summary.tsv")
        pair = pd.concat(pair_summaries, ignore_index=True)
        save_tsv(pair, output / "paired_query_task_changes.tsv.gz")
        metrics = ["mae_mean", "rmse_mean", "condition_spearman_mean", "mae_improvement_vs_zero",
                   "mae_improvement_vs_source_mean", "mae_improvement_vs_source_median",
                   "gene_spearman_equal_dose_mean", "gene_order_equal_dose_mean"]
        # Five entity folds form ONE CV partition, NOT five independent repeats.
        # Do not pool cross-fold predictions for a rank score.
        keys = ["task", "cohort", "method"]
        line_summary = summary.groupby(keys + ["target_cell_line"], sort=False)[metrics].mean().reset_index()
        line_valid = summary.groupby(keys + ["target_cell_line"], sort=False)[metrics].count().add_suffix("__valid_entity_folds").reset_index()
        line_summary = line_summary.merge(line_valid, on=keys + ["target_cell_line"], validate="one_to_one")
        save_tsv(line_summary, output / "descriptive_cell_line_summary.tsv")
        macro = line_summary.groupby(keys, sort=False)[metrics].mean().reset_index()
        valid = summary.groupby(keys, sort=False)[metrics].count().add_suffix("__valid_splits").reset_index()
        macro = macro.merge(valid, on=keys, validate="one_to_one")
        valid_lines = line_summary.groupby(keys, sort=False)[metrics].count().add_suffix("__valid_cell_lines").reset_index()
        macro = macro.merge(valid_lines, on=keys, validate="one_to_one")
        macro["aggregation"] = "equal entity fold then equal cell line; gene metrics first equal dose within fold"
        macro["n_independent_studies"], macro["n_entity_partitions"], macro["n_cell_lines"] = 1, 1, 3
        save_tsv(macro, output / "descriptive_macro_summary.tsv")
        for row in input_manifest:
            require(sha256(row["path"]) == row["sha256"], "input mutated: " + row["path"])
        write_json(output / "audit.json", dict(status="PASS", created_utc=datetime.now(timezone.utc).isoformat(),
            seconds=time.perf_counter() - started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            n_fitted_splits=30, n_reused_splits=15, n_new_model_fits=540, n_methods=21,
            n_fold_cohort_method_rows=len(summary), n_paired_query_rows=len(pair), method_order=methods,
            input_hashes_unchanged=True, n_independent_studies=1, config=config,
            interpretation="retrospective one-study grouped identity stress tests; not scaffold OOD, not causal context-loss decomposition",
            scientific_scope="source sample sizes and summed-loss alpha/n differ by task/fold; same feature budget is not equal effective complexity; seen-context source responses share line-B controls with queries",
            scoring="fixed-dose ranks within each fitted fold only; constant predictors retain NA correlation",
            split_contract_sha256=sha256(output / "split_contract_sealed.json")))
        print("COMPLETE", output, flush=True)
    except Exception as exc:
        write_json(output / "failure.json", dict(error=repr(exc), type=type(exc).__name__,
            created_utc=datetime.now(timezone.utc).isoformat()))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        run(args.config)
