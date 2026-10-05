#!/usr/bin/env python3
"""Frozen E01 metadata-only fidelity. No expression response is opened.

Grouped-drug ridge readouts and closed-inventory cross-view retrieval are
distinct estimands. Outputs are descriptive, not independent-study inference.
All derived artifacts are written exclusively to a new output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import resource
import sys
import time

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
from landmark_decoder import LandmarkRidge, cosine_similarity
from run_sciplex_prediction import choose_landmarks, read_tsv, require, sha256, write_json

VIEWS = ("entity", "entity_exposure", "entity_context", "complete_metadata")
VIEW_FIELDS = {
    "entity": ("source_entity_key",),
    "entity_exposure": ("source_entity_key", "dose_value", "time", "dose_unit"),
    "entity_context": ("source_entity_key", "cell_line"),
    "complete_metadata": ("source_entity_key", "dose_value", "time", "dose_unit", "cell_line"),
}
MODEL_KEYS = ("bge_m3", "sapbert", "qwen3_0_6b", "biomedbert", "medcpt_article", "medcpt_query")
BASELINES = ("tfidf", "structured_onehot", "random_field512")
FIELDS = ("cell_line", "dose_value")


def save_npz(path, **values):
    with Path(path).open("xb") as handle:
        np.savez_compressed(handle, **values)


def save_tsv(frame, path):
    with Path(path).open("x", encoding="utf-8") as handle:
        frame.to_csv(handle, sep="\t", index=False)


def grouped_folds(metadata, seed, n_folds):
    entities = sorted(metadata.source_entity_key.unique(),
        key=lambda key: (hashlib.sha256(f"{seed}|{key}".encode()).hexdigest(), key))
    lookup = {key: i % n_folds for i, key in enumerate(entities)}
    return metadata.source_entity_key.map(lookup).to_numpy(dtype=int)


def typed_random_vector(seed, field, value, dimension):
    seed_bytes = hashlib.sha256(f"{seed}|{field}|{str(value)}".encode()).digest()[:8]
    vector = np.random.default_rng(int.from_bytes(seed_bytes, "little")).standard_normal(dimension)
    return vector / np.linalg.norm(vector)


def random_field_features(metadata, view, seed, dimension=512):
    result = np.zeros((len(metadata), dimension), dtype=np.float64)
    for field in VIEW_FIELDS[view]:
        values = metadata[field].astype(str).to_numpy()
        draws = {v: typed_random_vector(seed, field, v, dimension) for v in sorted(set(values))}
        result += np.stack([draws[v] for v in values])
    return normalize(result, norm="l2")


def structured_features(fit_metadata, fit_view, metadata, view):
    """Fit categories/feature fields only on source or gallery, ignore unknowns."""
    categories = [(field, str(value)) for field in VIEW_FIELDS[fit_view]
                  for value in sorted(set(fit_metadata[field].astype(str)))]
    matrix = np.zeros((len(metadata), len(categories)), dtype=np.float64)
    supplied = set(VIEW_FIELDS[view])
    for j, (field, value) in enumerate(categories):
        if field in supplied:
            matrix[:, j] = metadata[field].astype(str).eq(value).to_numpy()
    return matrix, categories


def baseline_features(model, fit_metadata, fit_view, fit_text, query_metadata, query_view, query_text,
                      config):
    if model == "tfidf":
        tf = config["tfidf"]
        vectorizer = TfidfVectorizer(ngram_range=tuple(tf["ngram_range"]),
            sublinear_tf=tf["sublinear_tf"], lowercase=tf["lowercase"],
            min_df=tf["min_df"], max_features=tf["max_features"])
        unique_text = sorted(set(fit_text))
        vectorizer.fit(unique_text)
        fit_x = normalize(vectorizer.transform(fit_text), norm="l2")
        query_x = normalize(vectorizer.transform(query_text), norm="l2")
        parameters = {"feature_names": vectorizer.get_feature_names_out(), "idf": vectorizer.idf_}
        audit = {"fitted_unique_text_sha256": [hashlib.sha256(t.encode()).hexdigest() for t in unique_text],
                 "fitted_unique_text_count": len(unique_text), "svd_used": False}
    elif model == "structured_onehot":
        fit_x, categories = structured_features(fit_metadata, fit_view, fit_metadata, fit_view)
        query_x, check = structured_features(fit_metadata, fit_view, query_metadata, query_view)
        require(categories == check, "one-hot column mismatch")
        parameters = {"feature_field": np.asarray([v[0] for v in categories]),
                      "feature_value": np.asarray([v[1] for v in categories])}
        audit = {"unseen_category_policy": "ignore", "fitted_fields": list(VIEW_FIELDS[fit_view])}
    elif model == "random_field512":
        seed, dim = config["random_field"]["seed"], config["random_field"]["dimensions"]
        fit_x = random_field_features(fit_metadata, fit_view, seed, dim)
        query_x = random_field_features(query_metadata, query_view, seed, dim)
        parameters = {"seed": np.asarray(seed), "dimension": np.asarray(dim)}
        audit = {"learned_parameters": False, "reference": "random_field_unit_vectors.npz",
                 "query_supplied_fields": list(VIEW_FIELDS[query_view]), "not_random_guessing": True}
    else:
        raise ValueError("unknown baseline " + model)
    audit.update({"fit_view": fit_view, "query_view": query_view,
                  "fit_atomic_id": fit_metadata.atomic_id.tolist()})
    return fit_x, query_x, parameters, audit


def similarity(x, y):
    if hasattr(x, "tocsr"):
        require(hasattr(y, "tocsr"), "sparse type mismatch")
        result = (normalize(x) @ normalize(y).T).toarray()
        require(np.isfinite(result).all(), "nonfinite sparse cosine")
        return np.clip(result, -1., 1.)
    return cosine_similarity(x, y)


def expected_tie_retrieval(scores, target_index):
    scores, target_index = np.asarray(scores, dtype=float), np.asarray(target_index, dtype=int)
    require(scores.ndim == 2 and np.isfinite(scores).all(), "invalid scores")
    require(target_index.shape == (len(scores),) and ((target_index >= 0) & (target_index < scores.shape[1])).all(),
            "invalid target index")
    relevant = scores[np.arange(len(scores)), target_index]
    above = (scores > relevant[:, None]).sum(axis=1)
    ties = (scores == relevant[:, None]).sum(axis=1)
    require((ties > 0).all(), "relevant candidate lost from ties")
    harmonic = np.r_[0., np.cumsum(1. / np.arange(1, scores.shape[1] + 1))]
    mrr = (harmonic[above + ties] - harmonic[above]) / ties
    hit = np.where(above == 0, 1. / ties, 0.)
    return {"expected_mrr": mrr, "expected_hit1": hit, "n_strictly_above": above, "n_tied": ties}


def collision_diagnostics(keys, labels):
    """Best deterministic class accuracy possible from these exact input groups."""
    keys, labels = np.asarray(keys), np.asarray(labels).astype(str)
    require(len(keys) == len(labels), "collision row mismatch")
    if keys.ndim == 2:
        _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    else:
        _, inverse, counts = np.unique(keys.astype(str), return_inverse=True, return_counts=True)
    best = sum(pd.Series(labels[inverse == group]).value_counts().max() for group in range(len(counts)))
    return {"n_rows": len(labels), "n_unique_inputs": len(counts),
            "n_rows_in_collision_groups": int(counts[counts > 1].sum()),
            "largest_collision_group": int(counts.max()), "deterministic_accuracy_ceiling": float(best / len(labels))}


def dense_if_sparse(value):
    return value.toarray() if hasattr(value, "toarray") else value


def load_inputs(config):
    metadata = read_tsv(config["metadata_path"])
    require(len(metadata) == config["expected_atoms"] and metadata.atomic_id.is_unique, "atom count/ID mismatch")
    require(metadata.source_entity_key.nunique() == config["expected_entities"], "entity count mismatch")
    require(not metadata.duplicated(["source_entity_key", "cell_line", "dose_value"]).any(), "duplicate grid atoms")
    require(metadata.groupby("source_entity_key").size().eq(12).all(), "drug grid incomplete")
    require(metadata.cell_line.nunique() == 3 and metadata.dose_value.nunique() == 4, "field class count")
    require(all(metadata[field].nunique() == 1 for field in config["constant_fields_not_evaluated"]), "constant changed")
    rep = json.loads(Path(config["representation_config"]).read_text())
    clean = Path(rep["clean_views_root"])
    sources = [Path(config["metadata_path"]), Path(config["representation_config"])]
    sources += [clean / name for name in ("config.json", "row_to_text_registry.tsv", "unique_texts.tsv")]
    clean_hashes = {str(path): sha256(path) for path in sources[-3:]}
    registry = read_tsv(clean / "row_to_text_registry.tsv")
    registry = registry[registry.variant == config["variant"]]
    table = read_tsv(clean / "unique_texts.tsv")
    require(table.text_id.is_unique, "duplicate text IDs")
    table = table.set_index("text_id")
    texts, text_ids, embeddings, records = {}, {}, {}, []
    for view in config["views"]:
        mapping = registry[registry.view == view]
        require(mapping.atomic_id.is_unique and len(mapping) == len(metadata), "registry count/cardinality")
        mapping = mapping.set_index("atomic_id").loc[metadata.atomic_id]
        text_ids[view] = mapping.text_id.to_numpy(dtype=str)
        selected = table.loc[text_ids[view]]
        require(np.array_equal(mapping.prompt_sha256, selected.prompt_sha256), "mapping digest mismatch")
        texts[view] = selected.prompt_text.to_numpy(dtype=str)
        require(all(hashlib.sha256(text.encode()).hexdigest() == digest for text, digest in zip(texts[view], selected.prompt_sha256)),
                "actual text digest mismatch")
    require([spec["model_key"] for spec in rep["encoders"]] == list(MODEL_KEYS), "six encoder order mismatch")
    for spec in rep["encoders"]:
        key, root = spec["model_key"], Path(spec["embedding_root"])
        audit_path = root / "audit.json"
        audit = json.loads(audit_path.read_text())
        require(audit["status"] == "PASS" and audit["variant"] == "source_name", "encoder not PASS")
        saved_sources = {row["path"]: row["sha256"] for row in audit["sources"]}
        require(all(saved_sources.get(path) == digest for path, digest in clean_hashes.items()), "encoding source mutation")
        outputs = [row for row in audit["outputs"] if row["model_key"] == key]
        require(len(outputs) == 1, "encoding output ambiguous")
        record = outputs[0]
        path = root / (key + "__source_name.npz")
        require(Path(record["path"]).resolve() == path.resolve() and sha256(path) == record["sha256"], "encoding hash mismatch")
        with np.load(path, allow_pickle=False) as data:
            ids, x, digests = data["text_id"], data["X"], data["prompt_sha256"]
        require(x.shape == (3760, record["dimension"]) and x.dtype == np.float32 and np.isfinite(x).all(), "invalid matrix")
        require(len(set(ids)) == len(ids) and np.array_equal(ids, np.asarray(["text:" + str(d) for d in digests])), "invalid IDs")
        lookup = pd.Index(ids)
        for view in VIEWS:
            rows = lookup.get_indexer(text_ids[view])
            require((rows >= 0).all(), "missing encoding text")
            embeddings[(key, view)] = x[rows]
        records.append({"model_key": key, "path": str(path), "sha256": record["sha256"],
                        "dimension": x.shape[1], "rows": len(x), "audit_path": str(audit_path), "audit_sha256": sha256(audit_path)})
        sources += [path, audit_path]
    return metadata, texts, text_ids, embeddings, records, list(dict.fromkeys(sources))


def evaluate_readouts(config, output, metadata, texts, text_ids, embeddings):
    folds = grouped_folds(metadata, config["seed"], config["folds"])
    fold_table = metadata[["atomic_id", "source_entity_key"]].copy()
    fold_table["fold"] = folds
    save_tsv(fold_table, output / "folds.tsv")
    classes = {field: sorted(metadata[field].astype(str).unique()) for field in FIELDS}
    write_json(output / "classes.json", classes)
    truth = {field: np.asarray([classes[field].index(str(v)) for v in metadata[field]]) for field in FIELDS}
    joint_y = np.concatenate([np.eye(len(classes[f]))[truth[f]] for f in FIELDS], axis=1)
    offsets, start = {}, 0
    for field in FIELDS:
        offsets[field] = slice(start, start + len(classes[field]))
        start += len(classes[field])
    summaries, diagnostics = [], []
    for model in (*MODEL_KEYS, *BASELINES):
        for view in VIEWS:
            destination = output / "readout" / model / view
            destination.mkdir(parents=True)
            oof = np.full((len(metadata), start), np.nan)
            majority = {field: np.full(len(metadata), -1, dtype=int) for field in FIELDS}
            for fold in range(config["folds"]):
                source, test = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
                require(not set(metadata.iloc[source].source_entity_key) & set(metadata.iloc[test].source_entity_key), "drug leakage")
                landmarks = choose_landmarks(metadata, source, config["n_landmarks"], config["seed"])
                fold_dir = destination / f"fold{fold}"
                fold_dir.mkdir()
                for name, rows in (("source_atoms", source), ("test_atoms", test), ("landmarks", landmarks)):
                    save_tsv(metadata.iloc[rows][["atomic_id", "source_entity_key", "cell_line", "dose_value"]], fold_dir / (name + ".tsv"))
                if model in MODEL_KEYS:
                    x = embeddings[(model, view)]
                    audit = {"learned_representation_parameters": False, "model_key": model}
                else:
                    _, x, params, audit = baseline_features(model, metadata.iloc[source], view, texts[view][source],
                                                            metadata, view, texts[view], config)
                    save_npz(fold_dir / "baseline_parameters.npz", **params)
                kernel = similarity(x, x[landmarks])
                probe = LandmarkRidge(config["alpha"]).fit(kernel[source], joint_y[source])
                oof[test] = probe.predict(kernel[test])
                save_npz(fold_dir / "fitted_parameters.npz", feature_mean=probe.feature_mean_,
                         feature_scale=probe.feature_scale_, target_mean=probe.target_mean_, coef=probe.coef_)
                for field in FIELDS:
                    counts = np.bincount(truth[field][source], minlength=len(classes[field]))
                    majority[field][test] = counts.argmax()
                    collision = collision_diagnostics(dense_if_sparse(x[test]), truth[field][test])
                    diagnostics.append({"task": "field_readout", "model": model, "view": view, "field": field,
                                        "fold": fold, "collision_type": "exact_vector", **collision})
                audit.update({"status": "PASS", "fold": fold, "source_only_fit": True,
                              "source_atomic_id": metadata.iloc[source].atomic_id.tolist(),
                              "test_atomic_id": metadata.iloc[test].atomic_id.tolist(),
                              "landmark_atomic_id": metadata.iloc[landmarks].atomic_id.tolist(),
                              "n_landmarks": len(landmarks), "alpha": config["alpha"], "joint_target_fields": list(FIELDS),
                              "train_target_counts": {f: np.bincount(truth[f][source], minlength=len(classes[f])).tolist() for f in FIELDS},
                              "n_source": len(source), "n_test": len(test)})
                write_json(fold_dir / "fit_audit.json", audit)
            require(np.isfinite(oof).all(), "incomplete OOF")
            arrays = {"atomic_id": metadata.atomic_id.to_numpy(dtype=str), "fold": folds}
            table = metadata[["atomic_id", "source_entity_key", "cell_line", "dose_value"]].copy()
            table["fold"] = folds
            for field in FIELDS:
                scores = oof[:, offsets[field]]
                prediction = scores.argmax(axis=1)
                arrays.update({field + "_score": scores, field + "_truth": truth[field],
                               field + "_classes": np.asarray(classes[field]), field + "_training_majority": majority[field]})
                table[field + "_prediction"] = np.asarray(classes[field])[prediction]
                table[field + "_correct"] = prediction == truth[field]
                table[field + "_max_score_ties"] = (scores == scores.max(axis=1, keepdims=True)).sum(axis=1)
                table[field + "_training_majority"] = np.asarray(classes[field])[majority[field]]
                for fold in [-1, *range(config["folds"])]:
                    rows = np.arange(len(metadata)) if fold == -1 else np.flatnonzero(folds == fold)
                    summaries.append({"model": model, "view": view, "field": field, "fold": fold,
                        "aggregation": "pooled_oof" if fold == -1 else "fold", "n_total": len(rows), "n_valid": len(rows),
                        "accuracy": accuracy_score(truth[field][rows], prediction[rows]),
                        "balanced_accuracy": balanced_accuracy_score(truth[field][rows], prediction[rows]),
                        "macro_f1": f1_score(truth[field][rows], prediction[rows], average="macro", zero_division=0),
                        "uniform_expected_accuracy": 1 / len(classes[field]),
                        "training_majority_accuracy": accuracy_score(truth[field][rows], majority[field][rows]),
                        "n_exact_score_ties": int((table[field + "_max_score_ties"].to_numpy()[rows] > 1).sum()),
                        "field_supplied": field in VIEW_FIELDS[view], "status": "PASS"})
                arrays[field + "_confusion"] = confusion_matrix(truth[field], prediction, labels=np.arange(len(classes[field])))
                diagnostics.append({"task": "field_readout", "model": model, "view": view, "field": field,
                                    "fold": -1, "collision_type": "exact_text", **collision_diagnostics(text_ids[view], truth[field])})
            save_npz(destination / "oof_scores.npz", **arrays)
            save_tsv(table, destination / "oof_predictions.tsv")
            print("READOUT_PASS", model, view, flush=True)
    save_tsv(pd.DataFrame(summaries), output / "readout_summary.tsv")
    save_tsv(pd.DataFrame(diagnostics), output / "readout_collision_diagnostics.tsv")
    unsupported = [{"view": view, "field": field, "status": "UNSUPPORTED_CONSTANT_FIELD", "n_classes": 1,
                    "n_total": len(metadata), "n_valid": 0} for view in VIEWS for field in config["constant_fields_not_evaluated"]]
    save_tsv(pd.DataFrame(unsupported), output / "unsupported_fields.tsv")


def retrieval_groups(metadata, texts, config):
    groups = []
    entity_gallery = metadata.sort_values(["source_entity_key", "atomic_id"]).drop_duplicates("source_entity_key").index.to_numpy()
    require(len(entity_gallery) == config["entity_retrieval"]["gallery_n"], "entity gallery size")
    for query_view in config["entity_retrieval"]["query_views"]:
        query = pd.DataFrame({"row": np.arange(len(metadata)), "text": texts[query_view]}).drop_duplicates("text").row.to_numpy()
        lookup = {entity: i for i, entity in enumerate(metadata.iloc[entity_gallery].source_entity_key)}
        target = np.asarray([lookup[entity] for entity in metadata.iloc[query].source_entity_key])
        groups.append(("entity__" + query_view, "entity", query_view, "entity", query, entity_gallery, target, "all"))
    for query_view, gallery_view in config["condition_retrieval"]["directions"]:
        for line in sorted(metadata.cell_line.unique()):
            rows = metadata[metadata.cell_line == line].sort_values(["source_entity_key", "dose_value"]).index.to_numpy()
            require(len(rows) == config["condition_retrieval"]["gallery_n"], "condition gallery size")
            require(not metadata.iloc[rows].duplicated(["source_entity_key", "dose_value"]).any(), "condition targets duplicate")
            name = "condition__" + query_view + "__" + gallery_view + "__" + line
            groups.append((name, "condition", query_view, gallery_view, rows, rows, np.arange(len(rows)), line))
    return groups


def evaluate_retrieval(config, output, metadata, texts, text_ids, embeddings):
    summaries, collisions = [], []
    for model in (*MODEL_KEYS, *BASELINES):
        for name, task, qview, gview, query, gallery, target, line in retrieval_groups(metadata, texts, config):
            destination = output / "retrieval" / model / name
            destination.mkdir(parents=True)
            if model in MODEL_KEYS:
                qx, gx = embeddings[(model, qview)][query], embeddings[(model, gview)][gallery]
                audit = {"learned_representation_parameters": False, "model_key": model}
            else:
                gx, qx, params, audit = baseline_features(model, metadata.iloc[gallery], gview, texts[gview][gallery],
                                                        metadata.iloc[query], qview, texts[qview][query], config)
                save_npz(destination / "baseline_parameters.npz", **params)
            scores = similarity(qx, gx)
            metrics = expected_tie_retrieval(scores, target)
            save_npz(destination / "scores.npz", scores=scores,
                query_atomic_id=metadata.iloc[query].atomic_id.to_numpy(dtype=str), query_text_id=text_ids[qview][query],
                gallery_atomic_id=metadata.iloc[gallery].atomic_id.to_numpy(dtype=str), gallery_text_id=text_ids[gview][gallery],
                target_index=target)
            result = metadata.iloc[query][["atomic_id", "source_entity_key", "cell_line", "dose_value"]].reset_index(drop=True)
            result["query_text_id"] = text_ids[qview][query]
            result["target_gallery_index"] = target
            for key, values in metrics.items():
                result[key] = values
            random_mrr = float(np.sum(1 / np.arange(1, len(gallery) + 1)) / len(gallery))
            result["random_expected_mrr"], result["random_expected_hit1"] = random_mrr, 1 / len(gallery)
            result["n_candidates"], result["valid"], result["status"] = len(gallery), True, "PASS"
            save_tsv(result, destination / "query_metrics.tsv")
            audit.update({"status": "PASS", "task": task, "query_view": qview, "gallery_view": gview,
                "fit_scope": "gallery_only" if model in ("tfidf", "structured_onehot") else "no_task_fit",
                "fixed_cell_line_supplied": line if task == "condition" else None,
                "query_atomic_id": metadata.iloc[query].atomic_id.tolist(), "gallery_atomic_id": metadata.iloc[gallery].atomic_id.tolist(),
                "n_queries": len(query), "n_candidates": len(gallery), "exact_tie_atol": 0,
                "query_deduplication": "exact_text" if task == "entity" else "within_given_cell_line_unique_drug_dose",
                "n_zero_query_vectors": int((np.linalg.norm(dense_if_sparse(qx), axis=1) == 0).sum()),
                "n_zero_gallery_vectors": int((np.linalg.norm(dense_if_sparse(gx), axis=1) == 0).sum())})
            write_json(destination / "fit_audit.json", audit)
            summaries.append({"model": model, "group": name, "task": task, "query_view": qview, "gallery_view": gview,
                "fixed_cell_line": line, "n_total": len(query), "n_valid": len(query), "n_candidates": len(gallery),
                "expected_mrr": float(metrics["expected_mrr"].mean()), "expected_hit1": float(metrics["expected_hit1"].mean()),
                "random_expected_mrr": random_mrr, "random_expected_hit1": 1 / len(gallery),
                "n_relevant_score_ties": int((metrics["n_tied"] > 1).sum()), "status": "PASS"})
            for kind, keys in (("exact_query_text", text_ids[qview][query]), ("exact_query_vector", dense_if_sparse(qx))):
                collisions.append({"model": model, "group": name, "collision_type": kind,
                                   **collision_diagnostics(keys, target)})
        print("RETRIEVAL_PASS", model, flush=True)
    save_tsv(pd.DataFrame(summaries), output / "retrieval_summary.tsv")
    save_tsv(pd.DataFrame(collisions), output / "retrieval_collision_diagnostics.tsv")


def validate_config(config):
    require(tuple(config["views"]) == VIEWS and tuple(config["fields"]) == FIELDS, "frozen views/fields changed")
    require(tuple(config["baselines"]) == BASELINES and config["variant"] == "source_name", "models/variant changed")
    require(config["expected_atoms"] == 2256 and config["expected_entities"] == 188, "cohort changed")
    require(config["folds"] == 5 and config["seed"] == 20260914, "folds/seed changed")
    require(config["n_landmarks"] == 256 and config["alpha"] == 10 and config["scale_threshold"] == 1e-8, "decoder changed")
    require(config["classification_tie_order"] == "sorted_class_lexical_first" and config["retrieval_tie_atol"] == 0, "ties changed")
    require(config["random_field"]["dimensions"] == 512 and config["random_field"]["seed"] == config["seed"], "random definition changed")
    require(config["tfidf"] == {"ngram_range": [1, 2], "sublinear_tf": True, "lowercase": True, "min_df": 1,
                                "max_features": 20000, "svd": False}, "TF-IDF contract changed")


def run(config_path):
    started = time.monotonic()
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    validate_config(config)
    output = Path(config["output_root"])
    require(not output.exists(), "refuse existing output directory including partial runs")
    metadata, texts, text_ids, embeddings, records, sources = load_inputs(config)
    sources += [config_path, Path(__file__).resolve(), Path(__file__).with_name("landmark_decoder.py"),
                Path(__file__).with_name("run_sciplex_prediction.py")]
    manifest = [{"path": str(path), "size": path.stat().st_size, "sha256": sha256(path)} for path in sources]
    output.mkdir(parents=True)
    write_json(output / "RUNNING.json", {"status": "RUNNING", "config": config})
    write_json(output / "input_manifest.json", manifest)
    write_json(output / "representation_manifest.json", records)
    save_tsv(metadata, output / "atomic_index.tsv")
    random_keys = [(field, value) for field in VIEW_FIELDS["complete_metadata"] for value in sorted(set(metadata[field].astype(str)))]
    save_npz(output / "random_field_unit_vectors.npz", field=np.asarray([item[0] for item in random_keys]),
        value=np.asarray([item[1] for item in random_keys]),
        vectors=np.stack([typed_random_vector(config["seed"], field, value, 512) for field, value in random_keys]))
    with threadpool_limits(limits=4):
        evaluate_readouts(config, output, metadata, texts, text_ids, embeddings)
        evaluate_retrieval(config, output, metadata, texts, text_ids, embeddings)
    for record in manifest:
        require(sha256(record["path"]) == record["sha256"], "input mutation detected: " + record["path"])
    outputs = [{"path": str(path.relative_to(output)), "size": path.stat().st_size, "sha256": sha256(path)}
               for path in sorted(output.rglob("*")) if path.is_file()]
    audit = {"status": "PASS", "config": config, "n_atoms": len(metadata), "n_drugs": metadata.source_entity_key.nunique(),
        "n_methods": 9, "n_views": 4, "n_readout_fits": 180, "n_readout_fields": 2, "n_retrieval_groups": 81,
        "expression_opened": False, "hyperparameter_tuning": False, "sources_unchanged": True,
        "elapsed_seconds": time.monotonic() - started, "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "package_versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn")},
        "outputs": outputs, "scope": config["scope"], "independent_qa_status": "PENDING"}
    write_json(output / "audit.json", audit)
    print(json.dumps({key: audit[key] for key in ("status", "elapsed_seconds", "max_rss_kib", "n_readout_fits", "n_retrieval_groups")} ), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    run(parser.parse_args().config)
