#!/usr/bin/env python3
"""Three existing anonymous mappings under the frozen original E01/E02 rules.

The only in-process E01 adapter converts feature-name serialization to Unicode;
it neither changes features nor replaces scientific parameters. E02 reads the
original immutable truth geometries, not expression or raw datasets.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import resource
import shutil
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import run_information_fidelity_v1 as e01
import run_effect_alignment_v1 as e02
from effect_alignment_metrics import cosine_matrix, rank_correlation, score_neighbors

ROOT = Path(__file__).resolve().parents[1]
require, sha, write_json, save_npz = e01.require, e01.sha256, e01.write_json, e01.save_npz
REFERENCES = ("structured_onehot", "random_field512")
METHODS = (*e01.MODEL_KEYS, *e01.BASELINES)


def exact_array(a, b, message):
    a, b = np.asarray(a), np.asarray(b)
    numeric = a.dtype.kind in "fc" and b.dtype.kind in "fc"
    require(np.array_equal(a, b, equal_nan=True) if numeric else np.array_equal(a, b), message)


@contextmanager
def unicode_feature_name_adapter():
    """Explicit, serialization-only adapter; restore even when evaluation fails."""
    original = e01.baseline_features
    def adapter(*args, **kwargs):
        fit, query, parameters, audit = original(*args, **kwargs)
        if "feature_names" in parameters:
            old = parameters["feature_names"]
            require(all(isinstance(value, str) for value in old), "unexpected feature-name objects")
            parameters = dict(parameters)
            parameters["feature_names"] = np.asarray(old, dtype=str)
            exact_array(old, parameters["feature_names"], "serialization changed feature names")
        return fit, query, parameters, audit
    e01.baseline_features = adapter
    try:
        yield
    finally:
        e01.baseline_features = original


def table(path):
    return pd.read_csv(path, sep="\t", keep_default_na=False, na_values=["NA"])


def save_table(frame, path):
    require(not Path(path).exists(), "refuse existing table")
    frame.to_csv(path, sep="\t", index=False, na_rep="NA", compression="gzip" if str(path).endswith(".gz") else None)


def verify_sources(config, config_path):
    require(config["variants"] == [f"anonymous_seed_{s}" for s in [20260914, 20260915, 20260916]], "variant selection changed")
    require(tuple(config["reference_methods"]) == REFERENCES, "reference identity changed")
    for name, expected in config["source_locked_hashes"].items():
        require(sha(ROOT / name) == expected, "locked original changed: " + name)
    c1, c2 = [json.loads(Path(config[key]).read_text()) for key in ("source_e01_config", "source_e02_config")]
    source1, source2 = Path(c1["output_root"]), Path(c2["output_root"])
    require(all(json.loads((path / "audit.json").read_text())["status"] == "PASS" for path in [source1, source2]), "original run not PASS")
    sources = [Path(config_path).resolve(), Path(__file__).resolve(), Path(config["contract"])]
    sources += [ROOT / name for name in config["source_locked_hashes"]]
    original1_audit = json.loads((source1 / "audit.json").read_text())
    for item in original1_audit["outputs"]:
        path = source1 / item["path"]
        require(sha(path) == item["sha256"], "original E01 output changed: " + str(path))
        sources.append(path)
    for path in source2.rglob("*"):
        if path.is_file():
            sources.append(path)
    for item in json.loads((source2 / "representation_manifest.json").read_text()):
        require(sha(item["path"]) == item["sha256"], "original E02 kernel changed")
    anon_root = Path(config["anonymous_embedding_root"])
    anon = json.loads((anon_root / "audit.json").read_text())
    require(anon["status"] == "PASS" and anon["variant"] == "anonymous_shared_text" and anon["variants"] == config["variants"], "anonymous encoding mismatch")
    require(anon["full_text_count"] == 3760 and anon["mapping_summary"]["source_name_text_overlap"] == 0, "anonymous text definition mismatch")
    for item in anon["sources"]:
        require(sha(item["path"]) == item["sha256"], "anonymous encoding source changed")
        sources.append(Path(item["path"]))
    for item in anon["outputs"]:
        require(sha(item["path"]) == item["sha256"], "anonymous vector changed")
        sources.append(Path(item["path"]))
    sources += [anon_root / name for name in ("audit.json", "row_to_text_registry.tsv", "encoded_texts.tsv", "mapping_summary.json")]
    sources = list(dict.fromkeys(sources))
    manifest = [{"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size} for path in sources]
    return c1, c2, anon, manifest


def load_variant(config, variant, c1, anonymous):
    metadata = e01.read_tsv(Path(c1["output_root"]) / "atomic_index.tsv")
    require(len(metadata) == 2256 and metadata.atomic_id.is_unique, "original E01 metadata mismatch")
    root = Path(config["anonymous_embedding_root"])
    registry = e01.read_tsv(root / "row_to_text_registry.tsv")
    registry = registry[registry.variant == variant]
    require(len(registry) == 9024 and not registry.duplicated(["atomic_id", "view", "variant"]).any(), "anonymous registry cardinality")
    require(set(registry.anonymous_seed.astype(str)) == {variant.removeprefix("anonymous_seed_")}, "wrong mapping seed")
    texts = e01.read_tsv(root / "encoded_texts.tsv").set_index("text_id")
    require(texts.index.is_unique and len(texts) == 3760, "shared text registry count")
    ids_by_view, text_by_view = {}, {}
    for view in e01.VIEWS:
        mapping = registry[registry.view == view].set_index("atomic_id").loc[metadata.atomic_id]
        exact_array(mapping.source_entity_key, metadata.source_entity_key, "mapping changed source identity")
        exact_array(mapping.prompt_sha256, texts.loc[mapping.text_id].prompt_sha256, "mapping digest mismatch")
        ids_by_view[view] = mapping.text_id.to_numpy(dtype=str)
        text_by_view[view] = texts.loc[mapping.text_id].prompt_text.to_numpy(dtype=str)
    embeddings, records = {}, []
    for record in anonymous["outputs"]:
        model = record["model_key"]
        with np.load(record["path"], allow_pickle=False) as data:
            ids, x, digests = data["text_id"], data["X"], data["prompt_sha256"]
        require(x.dtype == np.float32 and x.shape == (3760, record["dimension"]), "embedding shape/dtype")
        require(np.isfinite(x).all() and len(set(ids)) == 3760, "embedding invalid")
        lookup = pd.Index(ids)
        for view in e01.VIEWS:
            rows = lookup.get_indexer(ids_by_view[view])
            require((rows >= 0).all(), "missing exact anonymous text")
            exact_array(digests[rows], texts.loc[ids_by_view[view]].prompt_sha256, "anonymous digest mismatch")
            embeddings[(model, view)] = x[rows]
        records.append({key: record[key] for key in ("model_key", "path", "sha256", "revision", "pooling", "rows", "dimension")})
    require({record["model_key"] for record in records} == set(e01.MODEL_KEYS), "six encoders missing")
    return metadata, text_by_view, ids_by_view, embeddings, records


def assert_e01_invariants(output, original):
    for name in ("folds.tsv", "classes.json", "atomic_index.tsv"):
        require((output / name).read_bytes() == (original / name).read_bytes(), "E01 identity changed: " + name)
    checked_scores = checked_retrieval = 0
    for model in METHODS:
        for view in e01.VIEWS:
            directory, source = output / "readout" / model / view, original / "readout" / model / view
            for fold in range(5):
                for filename in ("source_atoms.tsv", "test_atoms.tsv", "landmarks.tsv"):
                    require((directory / f"fold{fold}" / filename).read_bytes() == (source / f"fold{fold}" / filename).read_bytes(), "source fold/landmark change")
            with np.load(directory / "oof_scores.npz", allow_pickle=False) as new, np.load(source / "oof_scores.npz", allow_pickle=False) as old:
                for name in ("atomic_id", "fold", "cell_line_truth", "dose_value_truth", "cell_line_classes", "dose_value_classes",
                             "cell_line_training_majority", "dose_value_training_majority"):
                    exact_array(new[name], old[name], "OOF target/fold identity changed")
                if model in REFERENCES:
                    for name in new.files:
                        exact_array(new[name], old[name], "unchanged-reference OOF mismatch " + model + "/" + name)
                    checked_scores += int(new["cell_line_score"].size + new["dose_value_score"].size)
        for directory in (output / "retrieval" / model).iterdir():
            source = original / "retrieval" / model / directory.name
            with np.load(directory / "scores.npz", allow_pickle=False) as new, np.load(source / "scores.npz", allow_pickle=False) as old:
                for name in ("query_atomic_id", "gallery_atomic_id", "target_index"):
                    exact_array(new[name], old[name], "retrieval atoms/gallery changed")
                if model in REFERENCES:
                    exact_array(new["scores"], old["scores"], "unchanged-reference retrieval scores changed")
                    checked_retrieval += int(new["scores"].size)
    return {"source_folds_landmarks_classes_and_candidates_identical": True,
            "reference_oof_score_values_checked": checked_scores, "reference_retrieval_score_values_checked": checked_retrieval}


def run_e01(output, config, c1, variant, metadata, texts, ids, embeddings, records):
    output.mkdir()
    write_json(output / "config.json", {"original_config": c1, "variant": variant, "serialization_adapter": config["serialization_only_adapter"]})
    e01.save_tsv(metadata, output / "atomic_index.tsv")
    write_json(output / "representation_manifest.json", records)
    original_function = e01.baseline_features
    with unicode_feature_name_adapter():
        e01.evaluate_readouts(c1, output, metadata, texts, ids, embeddings)
        e01.evaluate_retrieval(c1, output, metadata, texts, ids, embeddings)
    require(e01.baseline_features is original_function, "E01 adapter was not restored")
    invariants = assert_e01_invariants(output, Path(c1["output_root"]))
    # References point to the identical immutable field-code definition instead
    # of duplicating its parameters under a different anonymous identity.
    source_random = Path(c1["output_root"]) / "random_field_unit_vectors.npz"
    shutil.copyfile(source_random, output / "random_field_unit_vectors.npz")
    require(sha(source_random) == sha(output / "random_field_unit_vectors.npz"), "random-field archive copy changed")
    write_json(output / "random_field_reference.json", {"path": str(source_random), "sha256": sha(source_random),
                                                        "uses_original_source_entity_key": True})
    write_json(output / "audit.json", {"status": "PASS", "variant": variant, "n_atoms": 2256,
        "n_readout_fits": 180, "n_retrieval_groups": 81, "no_raw_read": True,
        "serialization_only_adapter_restored": True, "independent_qa": "PENDING", **invariants})


def load_truth_groups(source, metadata):
    groups, references = [], []
    for record in json.loads((source / "group_manifest.json").read_text()):
        path = source / "groups" / record["group_id"] / "truth_geometry.npz"
        with np.load(path, allow_pickle=False) as data:
            saved = {key: data[key] for key in data.files}
        rows = saved["primary_rows"]
        exact_array(saved["atomic_id"], metadata.iloc[rows].atomic_id.to_numpy(dtype=str), "frozen truth atom order changed")
        require(len(rows) == record["n_atoms"] and len(saved["source_feature_row"]) == record["n_genes"], "truth dimensions changed")
        require(len(saved["similarity_upper"]) == record["n_pairs"], "truth pair count changed")
        groups.append((record, saved))
        references.append({"group_id": record["group_id"], "path": str(path), "sha256": sha(path),
                           "n_atoms": len(rows), "n_genes": record["n_genes"], "recomputed": False})
    require(len(groups) == 16, "frozen truth groups missing")
    return groups, references


def summarize_e02(summary, output):
    metrics = ["rsa", "ndcg_mean", "random_ndcg_mean", "excess_ndcg_mean"]
    local = summary[summary.group_kind.eq("within_cell_line_dose")]
    byline = local.groupby(["method", "view", "cell_line"], sort=False)[metrics].mean().reset_index()
    counts = local.groupby(["method", "view", "cell_line"], sort=False)[metrics].count().add_suffix("__valid_dose_groups").reset_index()
    byline = byline.merge(counts, on=["method", "view", "cell_line"])
    save_table(byline, output / "equal_dose_by_cell_summary.tsv")
    macro = byline.groupby(["method", "view"], sort=False)[metrics].mean().reset_index()
    counts = byline.groupby(["method", "view"], sort=False)[metrics].count().add_suffix("__valid_cell_groups").reset_index()
    macro = macro.merge(counts, on=["method", "view"])
    macro["n_independent_studies"] = 1
    save_table(macro, output / "descriptive_equal_dose_cell_summary.tsv")


def run_e02(output, c2, variant, full_metadata, texts, embeddings):
    original = Path(c2["output_root"])
    metadata = e01.read_tsv(original / "atomic_index.tsv")
    full_rows = pd.Index(full_metadata.atomic_id).get_indexer(metadata.atomic_id)
    require(len(metadata) == 2250 and (full_rows >= 0).all(), "primary atom mapping failure")
    exact_array(metadata.source_entity_key, full_metadata.iloc[full_rows].source_entity_key, "primary entity identity changed")
    groups, references = load_truth_groups(original, metadata)
    output.mkdir()
    (output / "representations").mkdir()
    e01.save_tsv(metadata, output / "atomic_index.tsv")
    write_json(output / "config.json", {"original_config": c2, "variant": variant})
    write_json(output / "truth_reference_manifest.json", references)
    write_json(output / "group_manifest.json", [record for record, _ in groups])
    summary_rows, query_tables, representation_records = [], [], []
    reference_kernel_values = 0
    for method in METHODS:
        for view in e01.VIEWS:
            if method in e01.MODEL_KEYS:
                features = embeddings[(method, view)][full_rows]
                parameters, kernel = {}, cosine_matrix(features)
            else:
                features, parameters = e02.baseline_features(method, view, metadata, texts[view][full_rows], c2)
                kernel = np.asarray((features @ features.T).toarray()) if hasattr(features, "tocsr") else cosine_matrix(features)
                kernel = np.clip(kernel, -1, 1)
            require(kernel.shape == (2250, 2250) and np.isfinite(kernel).all(), "invalid representation geometry")
            upper = kernel[np.triu_indices(2250, 1)]
            path = output / "representations" / (method + "__" + view + ".npz")
            save_npz(path, atomic_id=metadata.atomic_id.to_numpy(dtype=str), similarity_upper=upper, **parameters)
            if method in REFERENCES:
                with np.load(original / "representations" / path.name, allow_pickle=False) as old:
                    exact_array(upper, old["similarity_upper"], "unchanged-reference E02 kernel changed")
                    reference_kernel_values += len(upper)
            representation_records.append({"method": method, "view": view, "path": str(path), "sha256": sha(path),
                "feature_dimension": features.shape[1], "variant": variant,
                "fit_scope": "frozen_pretrained" if method in e01.MODEL_KEYS else c2["baseline_fit_scope"]})
            for record, saved in groups:
                rows = saved["primary_rows"]
                small = kernel[np.ix_(rows, rows)]
                rsa = rank_correlation(saved["similarity_upper"], small[np.triu_indices(len(rows), 1)])
                ndcg = score_neighbors(small, saved["relevance"], saved["idcg"])
                excess = ndcg - saved["random_ndcg"]
                summary_rows.append(dict(method=method, view=view, **record, rsa=rsa,
                    rsa_status="VALID" if np.isfinite(rsa) else "CONSTANT_PAIR_GEOMETRY",
                    ndcg_mean=e02.finite_mean(ndcg), random_ndcg_mean=e02.finite_mean(saved["random_ndcg"]),
                    excess_ndcg_mean=e02.finite_mean(excess), valid_ndcg_n=int(np.isfinite(ndcg).sum()), n_independent_studies=1))
                query_tables.append(pd.DataFrame(dict(method=method, view=view, group_id=record["group_id"],
                    group_kind=record["group_kind"], atomic_id=metadata.iloc[rows].atomic_id.to_numpy(),
                    source_entity_key=metadata.iloc[rows].source_entity_key.to_numpy(), n_candidates=len(rows)-1,
                    ndcg=ndcg, random_ndcg=saved["random_ndcg"], excess_ndcg=excess,
                    status=saved["status"], n_unique_effect_scores=saved["n_unique_effect_scores"])))
            print("ANONYMOUS_E02_PASS", variant, method, view, flush=True)
    summary, query = pd.DataFrame(summary_rows), pd.concat(query_tables, ignore_index=True)
    require(len(summary) == 576 and len(query) == 243000, "anonymous E02 incomplete matrix")
    save_table(summary, output / "group_summary.tsv")
    save_table(query, output / "query_metrics.tsv.gz")
    write_json(output / "representation_manifest.json", representation_records)
    summarize_e02(summary, output)
    for name, frame, keys in (("group_summary.tsv", summary, ["method", "view", "group_id"]),
                             ("query_metrics.tsv.gz", query, ["method", "view", "group_id", "atomic_id"])):
        source = table(original / name)
        exported = table(output / name)
        for method in REFERENCES:
            a, b = [f[f.method.eq(method)].sort_values(keys).reset_index(drop=True) for f in (exported, source)]
            for column in a:
                # The original text export round-trips floats using pandas;
                # compare the same exported representation, not memory vs CSV.
                exact_array(a[column], b[column], "unchanged-reference E02 exported metric changed: " + column)
    write_json(output / "audit.json", {"status": "PASS", "variant": variant, "n_atoms": 2250,
        "n_group_results": 576, "n_query_rows": 243000, "n_truth_groups_reused": 16,
        "no_raw_or_expression_read": True, "reference_kernel_values_checked": reference_kernel_values,
        "reference_exported_metrics_exact": True, "independent_qa": "PENDING"})


def paired_merge(anonymous, original, keys, metrics, fixed=()):
    require(not anonymous.duplicated(keys).any() and not original.duplicated(keys).any(), "ambiguous pairing")
    selected = list(dict.fromkeys([*keys, *metrics, *fixed]))
    merged = anonymous[selected].merge(original[selected], on=keys, how="outer", validate="one_to_one",
        suffixes=("__anonymous", "__source_name"), indicator=True)
    require(len(merged) == len(anonymous) == len(original) and merged._merge.eq("both").all(), "paired cohort changed")
    merged = merged.drop(columns="_merge")
    for name in fixed:
        exact_array(merged[name + "__anonymous"], merged[name + "__source_name"], "paired fixed field changed: " + name)
    for metric in metrics:
        merged[metric + "__delta"] = merged[metric + "__anonymous"] - merged[metric + "__source_name"]
    return merged


def paired_outputs(seed_root, source1, source2, variant):
    destination = seed_root / "paired"
    destination.mkdir()
    field_tables, retrieval_tables = [], []
    for model in METHODS:
        for view in e01.VIEWS:
            relative = Path("readout") / model / view / "oof_scores.npz"
            with np.load(seed_root / "e01" / relative, allow_pickle=False) as anon, np.load(source1 / relative, allow_pickle=False) as source:
                for field in e01.FIELDS:
                    ac = (anon[field + "_score"].argmax(axis=1) == anon[field + "_truth"]).astype(int)
                    sc = (source[field + "_score"].argmax(axis=1) == source[field + "_truth"]).astype(int)
                    field_tables.append(pd.DataFrame({"model": model, "view": view, "field": field,
                        "atomic_id": anon["atomic_id"], "fold": anon["fold"], "anonymous_correct": ac,
                        "source_name_correct": sc, "delta_correct": ac - sc}))
        for directory in sorted((seed_root / "e01/retrieval" / model).iterdir()):
            anon, old = table(directory / "query_metrics.tsv"), table(source1 / "retrieval" / model / directory.name / "query_metrics.tsv")
            paired = paired_merge(anon, old, ["atomic_id"], ["expected_mrr", "expected_hit1"],
                ["source_entity_key", "target_gallery_index", "n_candidates", "random_expected_mrr", "random_expected_hit1"])
            paired.insert(0, "model", model)
            paired.insert(1, "group", directory.name)
            retrieval_tables.append(paired)
    field_data, retrieval_data = pd.concat(field_tables, ignore_index=True), pd.concat(retrieval_tables, ignore_index=True)
    require(len(field_data) == 162432 and len(retrieval_data) == 72756, "paired E01 expected count")
    save_table(field_data, destination / "e01_field_query_delta.tsv.gz")
    save_table(retrieval_data, destination / "e01_retrieval_query_delta.tsv.gz")
    e01_field_summary = field_data.groupby(["model", "view", "field"], sort=False).agg(
        n_total=("atomic_id", "size"), n_valid=("delta_correct", "count"),
        anonymous_accuracy=("anonymous_correct", "mean"), source_name_accuracy=("source_name_correct", "mean"),
        delta_accuracy=("delta_correct", "mean")).reset_index()
    save_table(e01_field_summary, destination / "e01_field_summary.tsv")
    retrieval_summary = retrieval_data.groupby(["model", "group"], sort=False).agg(
        n_total=("atomic_id", "size"), n_valid=("expected_mrr__delta", "count"),
        delta_expected_mrr=("expected_mrr__delta", "mean"), delta_expected_hit1=("expected_hit1__delta", "mean")).reset_index()
    save_table(retrieval_summary, destination / "e01_retrieval_summary.tsv")
    group = paired_merge(table(seed_root / "e02/group_summary.tsv"), table(source2 / "group_summary.tsv"),
        ["method", "view", "group_id"], ["rsa", "ndcg_mean", "excess_ndcg_mean"],
        ["group_kind", "cell_line", "dose_value", "n_atoms", "n_genes", "n_pairs", "valid_ndcg_n", "random_ndcg_mean"])
    query = paired_merge(table(seed_root / "e02/query_metrics.tsv.gz"), table(source2 / "query_metrics.tsv.gz"),
        ["method", "view", "group_id", "atomic_id"], ["ndcg", "excess_ndcg"],
        ["source_entity_key", "group_kind", "n_candidates", "random_ndcg", "status", "n_unique_effect_scores"])
    save_table(group, destination / "e02_group_delta.tsv")
    save_table(query, destination / "e02_query_delta.tsv.gz")
    macro = paired_merge(table(seed_root / "e02/descriptive_equal_dose_cell_summary.tsv"),
        table(source2 / "descriptive_equal_dose_cell_summary.tsv"), ["method", "view"],
        ["rsa", "ndcg_mean", "excess_ndcg_mean"], ["n_independent_studies"])
    save_table(macro, destination / "e02_equal_dose_cell_delta.tsv")
    # Pair-first query mean must reproduce each group difference, including NA.
    aggregate = query.groupby(["method", "view", "group_id"], sort=False)[["ndcg__delta", "excess_ndcg__delta"]].mean().reset_index()
    joined = group.merge(aggregate, on=["method", "view", "group_id"], validate="one_to_one")
    for left, right in (("ndcg_mean__delta", "ndcg__delta"), ("excess_ndcg_mean__delta", "excess_ndcg__delta")):
        require(np.allclose(joined[left], joined[right], rtol=0, atol=2e-15, equal_nan=True), "paired-before-aggregate mismatch")
    write_json(destination / "audit.json", {"status": "PASS", "variant": variant,
        "e01_field_query_rows": len(field_data), "e01_retrieval_query_rows": len(retrieval_data),
        "e02_group_rows": len(group), "e02_query_rows": len(query), "paired_before_aggregation": True,
        "delta_direction": "anonymous_minus_source_name", "independent_qa": "PENDING"})


def run(config_path):
    started = time.perf_counter()
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    output = Path(config["output_root"])
    require(not output.exists(), "refuse existing or partial anonymous output root")
    c1, c2, anonymous, manifest = verify_sources(config, config_path)
    output.mkdir(parents=True)
    write_json(output / "input_manifest.json", manifest)
    write_json(output / "RUNNING.json", {"status": "RUNNING", "config": config})
    with threadpool_limits(limits=config["cpu_threads"]):
        for variant in config["variants"]:
            seed_root = output / variant
            seed_root.mkdir()
            metadata, texts, ids, embeddings, records = load_variant(config, variant, c1, anonymous)
            run_e01(seed_root / "e01", config, c1, variant, metadata, texts, ids, embeddings, records)
            run_e02(seed_root / "e02", c2, variant, metadata, texts, embeddings)
            paired_outputs(seed_root, Path(c1["output_root"]), Path(c2["output_root"]), variant)
            print("ANONYMOUS_VARIANT_PASS", variant, flush=True)
    summaries = {}
    for filename in ("e01_field_summary.tsv", "e01_retrieval_summary.tsv", "e02_group_delta.tsv", "e02_equal_dose_cell_delta.tsv"):
        collected = []
        for variant in config["variants"]:
            frame = table(output / variant / "paired" / filename)
            frame.insert(0, "variant", variant)
            frame.insert(1, "n_independent_studies", 1) if "n_independent_studies" not in frame else None
            collected.append(frame)
        combined = pd.concat(collected, ignore_index=True)
        save_table(combined, output / ("all_variants__" + filename))
        summaries[filename] = len(combined)
    for item in manifest:
        require(sha(item["path"]) == item["sha256"], "original input changed during run: " + item["path"])
    outputs = [{"path": str(path.relative_to(output)), "sha256": sha(path), "bytes": path.stat().st_size}
               for path in sorted(output.rglob("*")) if path.is_file()]
    write_json(output / "audit.json", {"status": "PASS", "config": config, "n_mapping_seeds": 3,
        "n_independent_studies": 1, "n_e01_fits": 540, "n_e01_retrieval_groups": 243,
        "n_e02_group_results": 1728, "n_e02_query_rows": 729000,
        "original_inputs_unchanged": True, "no_raw_or_expression_read": True, "no_anonymous_prediction_run": True,
        "all_reference_invariance_gates_pass": True, "all_variants_retained": True, "summaries": summaries,
        "scientific_scope": "sensitivity to entity names and lexical form; does not separately identify pretrained biological knowledge or information leakage",
        "seconds": time.perf_counter() - started, "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "outputs": outputs, "independent_qa": "PENDING"})
    print("ANONYMOUS_E01_E02_CORE_PASS", time.perf_counter() - started, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    run(parser.parse_args().config)
