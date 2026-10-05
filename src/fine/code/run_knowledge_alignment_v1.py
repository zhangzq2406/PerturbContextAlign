#!/usr/bin/env python3
"""E08 frozen local knowledge/prior geometry; no fitting, raw reads or encoding."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import resource
import sys
import time

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits
from effect_alignment_metrics import prepare_truth, expected_dcg, rank_correlation

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["bge_m3", "sapbert", "qwen3_0_6b", "biomedbert", "medcpt_article", "medcpt_query"]
COHORTS = ["mechanism_text", "human_target", "morgan_all"]
GROUP_KEYS = ["cohort", "group_id", "cell_line", "dose_value"]
METRICS = ["rsa", "ndcg_mean", "random_ndcg_mean", "excess_ndcg_mean"]
PAIR_METRICS = ["rsa_change", "ndcg_change_paired_mean", "random_ndcg_change_paired_mean", "excess_ndcg_change_paired_mean"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def save_npz(destination, **values):
    with Path(destination).open("xb") as stream:
        np.savez_compressed(stream, **values)


def read_table(path):
    return pd.read_csv(path, sep="\t", keep_default_na=False, na_values=["NA"], float_precision="round_trip")


def table(path, frame):
    with Path(path).open("xb") as stream:
        frame.to_csv(stream, sep="\t", index=False, na_rep="NA",
                     compression={"method": "gzip", "mtime": 0} if str(path).endswith(".gz") else None)


def flags(values):
    strings = pd.Series(values).astype(str).str.lower()
    require(strings.isin(["true", "false"]).all(), "Malformed boolean admission/primary flags")
    return strings.eq("true").to_numpy()


def finite_mean(values):
    x = np.asarray(values, dtype=float)
    return float(x[np.isfinite(x)].mean()) if np.isfinite(x).any() else np.nan


def registry():
    objects, contrasts = [], []
    for cohort in COHORTS:
        for model in MODELS:
            complete = model+"__complete_metadata"
            objects.append(dict(cohort=cohort, object_id=complete, model=model, view="complete_metadata", kind="archived_complete"))
            if cohort == "mechanism_text":
                for view in ["complete_metadata+knowledge", "knowledge_only"]:
                    obj = model+"__"+view
                    objects.append(dict(cohort=cohort, object_id=obj, model=model, view=view, kind="knowledge_cosine"))
                    suffix = "knowledge_only_minus_complete" if view == "knowledge_only" else "augmentation_minus_complete"
                    contrasts.append(dict(cohort=cohort, contrast_id=model+"__"+suffix, left=obj, right=complete, direction="left_minus_right"))
        if cohort != "mechanism_text":
            prior = "native_human_target" if cohort == "human_target" else "native_morgan"
            objects.append(dict(cohort=cohort, object_id=prior, model=prior, view="native_prior", kind="native"))
            for model in MODELS:
                complete, fusion = model+"__complete_metadata", model+"__fusion_"+prior
                objects.append(dict(cohort=cohort, object_id=fusion, model=model, view="fixed_rank_fusion", kind="fusion",
                                    component_a=complete, component_b=prior))
                for suffix, left, right in [("native_minus_complete", prior, complete),
                                             ("fusion_minus_complete", fusion, complete), ("fusion_minus_native", fusion, prior)]:
                    contrasts.append(dict(cohort=cohort, contrast_id=model+"__"+suffix, left=left, right=right, direction="left_minus_right"))
    return objects, contrasts


def cosine(values, unique_rows=False):
    x = np.asarray(values, dtype=np.float64)
    require(x.ndim == 2 and len(x) > 0 and np.isfinite(x).all(), "Finite nonempty matrix required")
    norms = np.linalg.norm(x, axis=1)
    require((norms > 0).all(), "Exact zero direction: fail closed without candidate deletion")
    if unique_rows:
        unique, inverse = np.unique(x, axis=0, return_inverse=True)
        unit = unique/np.linalg.norm(unique, axis=1, keepdims=True)
        small = np.clip(unit @ unit.T, -1, 1)
        return small[np.ix_(inverse, inverse)], norms
    unit = x/norms[:, None]
    return np.clip(unit @ unit.T, -1, 1), norms


def binary_overlap(values):
    x = np.asarray(values)
    require(x.ndim == 2 and len(x) and x.shape[1] and np.isin(x, [0, 1]).all(), "Binary membership matrix required")
    x = x.astype(np.int64)
    counts = x.sum(axis=1)
    require((counts > 0).all(), "Missing/empty prior must not become an all-zero row")
    intersection = x @ x.T
    union = counts[:, None]+counts[None, :]-intersection
    return intersection/union


def target_features(entity_ids, members):
    require(not members.duplicated(["source_entity_key", "target_chembl_id"]).any(), "Duplicate canonical target membership")
    require(members.organism.eq("Homo sapiens").all() and members.tax_id.astype(str).eq("9606").all(), "Nonhuman target record")
    require(members.target_chembl_id.astype(str).str.fullmatch(r"CHEMBL\d+").all(), "Unqualified target identifier")
    admitted = set(entity_ids)
    require(set(members.source_entity_key) == admitted, "Target entities/membership admission mismatch")
    targets = sorted(set(members.target_chembl_id))
    erows, tcols = {e: i for i, e in enumerate(entity_ids)}, {t: i for i, t in enumerate(targets)}
    x = np.zeros((len(entity_ids), len(targets)), dtype=np.uint8)
    for row in members.itertuples(index=False):
        x[erows[row.source_entity_key], tcols[row.target_chembl_id]] = 1
    require((x.sum(axis=1) > 0).all(), "Admitted target entity has no target")
    return x, np.asarray(targets, dtype=str)


def matrix_from_upper(upper, n):
    upper = np.asarray(upper)
    require(upper.dtype == np.float64 and upper.shape == (n*(n-1)//2,) and np.isfinite(upper).all(), "Wrong pair axis/dtype")
    result = np.eye(n, dtype=np.float64)
    i, j = np.triu_indices(n, 1)
    result[i, j] = result[j, i] = upper
    return result


def extract_upper(upper, rows, total_n):
    rows = np.asarray(rows, dtype=np.int64)
    require(rows.ndim == 1 and len(np.unique(rows)) == len(rows), "Duplicate subset rows")
    require((rows >= 0).all() and (rows < total_n).all(), "Subset axis out of range")
    require(np.shape(upper) == (total_n*(total_n-1)//2,), "Full pair axis differs")
    i, j = np.triu_indices(len(rows), 1)
    a, b = np.minimum(rows[i], rows[j]), np.maximum(rows[i], rows[j])
    return np.asarray(upper)[total_n*a-a*(a+1)//2+b-a-1]


def rank_fusion(upper_a, upper_b):
    a, b = np.asarray(upper_a, dtype=np.float64), np.asarray(upper_b, dtype=np.float64)
    require(a.ndim == 1 and a.shape == b.shape and len(a) > 1, "Aligned strict-upper pairs required")
    require(np.isfinite(a).all() and np.isfinite(b).all(), "Nonfinite fusion source")
    # Doubled average ranks are exact integers. Sum BEFORE the one division so
    # mathematically equal sums cannot be split by separate normalization rounding.
    rank2_a = (2*rankdata(a, method="average")).astype(np.int64)
    rank2_b = (2*rankdata(b, method="average")).astype(np.int64)
    fused = (rank2_a+rank2_b-4)/(4*(len(a)-1))
    return fused, rank2_a, rank2_b


def truth_geometry(values, k=10, minimum_n=12):
    similarity, norms = cosine(values)
    rel, ideal, random, status, unique = prepare_truth(similarity, k, minimum_n)
    return dict(similarity=similarity, arithmetic_row_norm=norms, relevance=rel, idcg=ideal,
                random_ndcg=random, status=status, n_unique_effect_scores=unique)


def score_geometry(upper, truth, k=10, minimum_n=12):
    n = len(truth["idcg"])
    similarity = matrix_from_upper(upper, n)
    ix = np.triu_indices(n, 1)
    rsa, rsa_status = np.nan, "VALID"
    if n < minimum_n:
        rsa_status = "INSUFFICIENT_CANDIDATES"
    elif np.ptp(truth["similarity"][ix]) == 0:
        rsa_status = "CONSTANT_TRUTH_PAIR_GEOMETRY"
    elif np.ptp(upper) == 0:
        rsa_status = "CONSTANT_REPRESENTATION_PAIR_GEOMETRY"
    else:
        rsa = rank_correlation(upper, truth["similarity"][ix])
    ndcg, random, excess = [np.full(n, np.nan) for _ in range(3)]
    status = np.asarray(truth["status"], dtype="U80").copy()
    unique = np.zeros(n, dtype=np.int64)
    for row in range(n):
        other = np.arange(n) != row
        scores = similarity[row, other]
        unique[row] = len(np.unique(scores))
        if not np.isfinite(truth["idcg"][row]):
            continue
        random[row] = truth["random_ndcg"][row]
        if unique[row] == 1:
            ndcg[row], excess[row], status[row] = random[row], 0.0, "VALID_ALL_PREDICTION_SCORES_TIED"
        else:
            ndcg[row] = expected_dcg(scores, truth["relevance"][row, other], k)/truth["idcg"][row]
            excess[row], status[row] = ndcg[row]-random[row], "VALID"
    return dict(rsa=rsa, rsa_status=rsa_status, ndcg=ndcg, random_ndcg=random, excess_ndcg=excess,
                status=status, n_unique_representation_scores=unique)


def paired_tables(summary, query, contrasts):
    paired_queries, paired_groups = [], []
    qkeys = GROUP_KEYS+["atomic_id", "source_entity_key", "n_candidates"]
    for spec in contrasts:
        cohort = spec["cohort"]
        qleft = query[query.cohort.eq(cohort) & query.object_id.eq(spec["left"])].drop(columns=["object_id", "model", "view", "kind"])
        qright = query[query.cohort.eq(cohort) & query.object_id.eq(spec["right"])].drop(columns=["object_id", "model", "view", "kind"])
        paired = qleft.merge(qright, on=qkeys, suffixes=("_left", "_right"), validate="one_to_one", sort=False)
        require(len(paired) == len(qleft) == len(qright), "Paired queries differ in coverage")
        paired["contrast_id"], paired["left"], paired["right"] = spec["contrast_id"], spec["left"], spec["right"]
        paired["left_valid"] = np.isfinite(paired.ndcg_left)
        paired["right_valid"] = np.isfinite(paired.ndcg_right)
        paired["paired_valid"] = paired.left_valid & paired.right_valid
        for metric in ["ndcg", "random_ndcg", "excess_ndcg"]:
            paired[metric+"_change"] = (paired[metric+"_left"]-paired[metric+"_right"]).where(paired.paired_valid)
        left = summary[summary.cohort.eq(cohort) & summary.object_id.eq(spec["left"])].drop(columns=["object_id", "model", "view", "kind"])
        right = summary[summary.cohort.eq(cohort) & summary.object_id.eq(spec["right"])].drop(columns=["object_id", "model", "view", "kind"])
        groups = left.merge(right, on=GROUP_KEYS, suffixes=("_left", "_right"), validate="one_to_one", sort=False)
        require(len(groups) == len(left) == len(right), "Paired group registry mismatch")
        groups["contrast_id"], groups["left"], groups["right"] = spec["contrast_id"], spec["left"], spec["right"]
        groups["rsa_paired_valid"] = np.isfinite(groups.rsa_left) & np.isfinite(groups.rsa_right)
        groups["rsa_change"] = (groups.rsa_left-groups.rsa_right).where(groups.rsa_paired_valid)
        records = []
        for keys, frame in paired.groupby(GROUP_KEYS, sort=False):
            selected = frame[frame.paired_valid]
            record = dict(zip(GROUP_KEYS, keys), n_queries=len(frame), left_valid_n=int(frame.left_valid.sum()),
                          right_valid_n=int(frame.right_valid.sum()), paired_valid_n=len(selected))
            for metric in ["ndcg", "random_ndcg", "excess_ndcg"]:
                for side in ["left", "right"]:
                    record[metric+"_"+side+"_paired_mean"] = finite_mean(selected[metric+"_"+side])
                record[metric+"_change_paired_mean"] = finite_mean(selected[metric+"_change"])
            records.append(record)
        paired_queries.append(paired)
        paired_groups.append(groups.merge(pd.DataFrame(records), on=GROUP_KEYS, validate="one_to_one", sort=False))
    return pd.concat(paired_queries, ignore_index=True), pd.concat(paired_groups, ignore_index=True)


def equal_weight_summaries(frame, paired=False):
    keys = ["cohort", "contrast_id", "left", "right"] if paired else ["cohort", "object_id", "model", "view", "kind"]
    metrics = PAIR_METRICS if paired else METRICS
    counts = ["n_queries", "left_valid_n", "right_valid_n", "paired_valid_n"] if paired else ["n_atoms", "valid_ndcg_n", "truth_valid_n"]
    lines = []
    for levels, group in frame.groupby(keys+["cell_line"], sort=False):
        record = dict(zip(keys+["cell_line"], levels), n_dose_groups=len(group))
        for metric in metrics:
            record[metric] = finite_mean(group[metric])
            record[metric+"__valid_dose_groups"] = int(np.isfinite(group[metric]).sum())
        for name in counts:
            record[name] = int(group[name].sum())
        lines.append(record)
    lines = pd.DataFrame(lines)
    macros = []
    for levels, group in lines.groupby(keys, sort=False):
        record = dict(zip(keys, levels), n_cell_lines=len(group), n_independent_studies=1)
        for metric in metrics:
            record[metric] = finite_mean(group[metric])
            record[metric+"__valid_cell_lines"] = int(np.isfinite(group[metric]).sum())
            record[metric+"__valid_dose_groups"] = int(group[metric+"__valid_dose_groups"].sum())
        for name in counts:
            record[name] = int(group[name].sum())
        macros.append(record)
    return lines, pd.DataFrame(macros)


def verify_file(path, expected, hashes):
    path = Path(path).resolve()
    require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected), "Unfrozen SHA256: "+str(path))
    require(sha(path) == expected, "Input hash mismatch: "+str(path))
    hashes[str(path)] = expected
    return path


def verify_manifest(manifest_path, hashes):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    require(isinstance(manifest, dict) and isinstance(manifest.get("files"), list), "Wrong output manifest schema")
    names = [item["path"] for item in manifest["files"]]
    require(len(names) == len(set(names)), "Duplicate manifest files")
    for item in manifest["files"]:
        relative = Path(item["path"])
        require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe manifest path")
        path = verify_file(manifest_path.parent/relative, item["sha256"], hashes)
        require(path.is_relative_to(manifest_path.parent.resolve()), "Escaped manifest root")
        require(path.stat().st_size == item.get("size_bytes", item.get("bytes")), "Manifest byte size differs")
    actual = {str(p.relative_to(manifest_path.parent)) for p in manifest_path.parent.rglob("*") if p.is_file()}
    require(actual == set(names) | {manifest_path.name}, "Unsealed files in input payload")
    return manifest


def audit_binding(gate, manifest_path, hashes):
    path = verify_file(ROOT/gate["path"], gate["sha256"], hashes)
    value = json.loads(path.read_text())
    require(value["status"] == gate["status"], "Independent gate not PASS")
    bound = value
    for key in gate["manifest_sha256_field"].split("."):
        bound = bound[key]
    require(bound == hashes[str(Path(manifest_path).resolve())], "Independent audit not bound to exact payload")


def preflight(config_path, static_only=False):
    """Hashes/metadata only; never call np.load or read new geometry values."""
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    output = ROOT/config["output_root"]
    if output.exists():
        raise FileExistsError(output)
    require(output.resolve() == ROOT/"metrics/knowledge_alignment_v1", "Unapproved output location")
    require(config["models"] == MODELS and config["folds"] == ["A549", "K562", "MCF7"] and config["doses"] == [10, 100, 1000, 10000], "Scope changed")
    require(config["neighbor_k"] == 10 and config["minimum_group_n"] == 12, "Metric cutoff changed")
    objects, contrasts = registry()
    require(config["objects"] == objects and config["contrasts"] == contrasts, "Object/contrast registry changed")
    require(len(objects) == config["expected_objects"] == 44 and len(contrasts) == config["expected_contrasts"] == 48, "Registry counts")
    require(config["embedding_norm_atol"] == 2e-6, "Vector tolerance changed")
    hashes = {str(config_path): sha(config_path), str(Path(__file__).resolve()): sha(__file__)}
    for gate in config["static_gates"]:
        path = verify_file(ROOT/gate["path"], gate["sha256"], hashes)
        if "status" in gate:
            require(json.loads(path.read_text())["status"] == gate["status"], "Static audit not PASS")
    metadata_root, e02, effects = [ROOT/config[key] for key in ["metadata_root", "e02_root", "effects_root"]]
    verify_manifest(metadata_root/"output_manifest.json", hashes)
    effect_audit = json.loads((effects/"audit.json").read_text())
    for name in ["arrays.npz", "atomic_index.tsv", "fold_gene_panels.tsv"]:
        verify_file(effects/name, effect_audit["output_sha256"][name], hashes)
    morgan = ROOT/config["morgan_root"]
    morgan_audit = json.loads((morgan/"audit.json").read_text())
    require(morgan_audit["radius"] == 2 and morgan_audit["fp_size"] == 2048 and morgan_audit["include_chirality"] is True, "Morgan prior changed")
    for name in ["arrays.npz", "entity_fingerprint_manifest.tsv"]:
        verify_file(morgan/name, morgan_audit["output_sha256"][name], hashes)
    original = json.loads((e02/"representation_manifest.json").read_text())
    selected = []
    for model in MODELS:
        matches = [r for r in original if (r["method"], r["view"]) == (model, "complete_metadata")]
        require(len(matches) == 1, "Missing/duplicate authenticated E02 complete geometry")
        record = matches[0]
        path = verify_file(record["path"], record["sha256"], hashes)
        require(path.parent == e02/"representations", "Unexpected E02 source path")
        selected.append(record)
    e02_inputs = {str((ROOT/r["path"]).resolve()): r["sha256"] for r in json.loads((e02/"input_manifest.json").read_text())}
    for name in ["row_to_text_registry.tsv", "unique_texts.tsv"]:
        path = ROOT/config["clean_views_root"]/name
        verify_file(path, e02_inputs[str(path)], hashes)
    legacy = json.loads((ROOT/"predictions/sciplex_loco_six_encoder_v1/representation_manifest.json").read_text())
    require([r["model_key"] for r in legacy] == MODELS, "Legacy encoder model order changed")
    for item in legacy:
        verify_file(item["embedding_path"], item["embedding_sha256"], hashes)
        verify_file(item["audit_path"], item["audit_sha256"], hashes)
    # Validate admitted axes, texts and original-C mapping without opening X/effect arrays.
    load_metadata(config)
    if not static_only:
        require(config["configuration_status"] == "READY_FOR_EXPLICIT_EXECUTION", "Runtime gates not frozen/approved")
        runtime = config["runtime_gates"]
        audit_binding(runtime["metadata_independent"], metadata_root/"output_manifest.json", hashes)
        embedding_root = ROOT/config["embedding_root"]
        manifest_gate = runtime["encoding_output_manifest"]
        require((ROOT/manifest_gate["path"]).resolve() == embedding_root/"output_manifest.json", "Encoding manifest root changed")
        manifest_path = verify_file(ROOT/manifest_gate["path"], manifest_gate["sha256"], hashes)
        verify_manifest(manifest_path, hashes)
        audit_binding(runtime["encoding_independent"], manifest_path, hashes)
        audit = json.loads((embedding_root/"audit.json").read_text())
        require(audit["status"] == "PASS" and audit["stage"] == "full", "Embedding production gate not full PASS")
        require(audit["metadata_output_manifest_sha256"] == hashes[str(metadata_root/"output_manifest.json")], "Encoder metadata differs")
        require(audit["metadata_independent_audit_sha256"] == runtime["metadata_independent"]["sha256"], "Encoder independent metadata gate differs")
        for model in MODELS:
            require(str(embedding_root/(model+"__"+config["variant"]+".npz")) in hashes, "Embedding archive missing from sealed manifest")
    return config, hashes, selected


def load_metadata(config):
    root = ROOT/config["metadata_root"]
    meta_all = read_table(ROOT/config["effects_root"]/"atomic_index.tsv")
    require(len(meta_all) == config["expected_all_effect_atoms"] == 2256 and meta_all.atomic_id.is_unique, "Effect index changed")
    primary = flags(meta_all.main_eligible)
    meta = meta_all[primary].reset_index(drop=True)
    require(len(meta) == config["expected_primary_atoms"] == 2250 and meta.source_entity_key.nunique() == 188, "Primary cohort changed")
    require(meta.dose_unit.eq("nM").all() and meta.time.eq(24).all(), "Exposure semantics differ")
    coverage = read_table(root/"entity_coverage.tsv")
    require(coverage.source_entity_key.is_unique and set(coverage.source_entity_key) == set(meta.source_entity_key), "Coverage entity universe changed")
    admitted = {}
    for spec in config["cohorts"]:
        entities = set(meta.source_entity_key) if spec["flag"] == "all_primary" else set(coverage.loc[flags(coverage[spec["flag"]]), "source_entity_key"])
        require(len(entities) == spec["expected_entities"], "Admitted entity count changed")
        rows = np.flatnonzero(meta.source_entity_key.isin(entities))
        require(len(rows) == spec["expected_atoms"], "Admitted atom count changed")
        admitted[spec["cohort"]] = (entities, rows)
    require(admitted["human_target"][0] <= admitted["mechanism_text"][0], "Target cohort not a mechanism subset")
    rowmap, texts = read_table(root/"rowmap.tsv"), read_table(root/"unique_texts.tsv")
    require(len(texts) == config["expected_unique_knowledge_texts"] == 729 and texts.text_id.is_unique, "Knowledge text universe changed")
    np.testing.assert_array_equal(texts.text_row, np.arange(len(texts)))
    digests = np.array([hashlib.sha256(t.encode()).hexdigest() for t in texts.prompt_text])
    np.testing.assert_array_equal(texts.prompt_sha256, digests)
    np.testing.assert_array_equal(texts.text_id, np.char.add("text:", digests))
    require(len(rowmap) == 1368 and not rowmap.duplicated(["atomic_id", "view"]).any(), "Knowledge rowmap cardinality")
    require(rowmap.variant.eq(config["variant"]).all() and set(rowmap.view) == set(config["knowledge_views"]), "Knowledge views/variant changed")
    old = read_table(ROOT/config["clean_views_root"]/"row_to_text_registry.tsv")
    old = old[old.view.eq("complete_metadata") & old.variant.eq("source_name")].set_index("atomic_id")
    require(old.index.is_unique, "Original complete row map duplicate")
    np.testing.assert_array_equal(old.loc[rowmap.atomic_id].text_id, rowmap.original_complete_text_id)
    lookup = meta.set_index("atomic_id")
    for key in ["source_entity_key", "cell_line", "dose_value"]:
        np.testing.assert_array_equal(rowmap[key], lookup.loc[rowmap.atomic_id, key])
    mapping = {}
    mechanism_ids = meta.iloc[admitted["mechanism_text"][1]].atomic_id.to_numpy(dtype=str)
    for view in config["knowledge_views"]:
        frame = rowmap[rowmap.view.eq(view)].set_index("atomic_id")
        require(set(frame.index) == set(mechanism_ids), "Per-view knowledge atom coverage differs")
        frame = frame.loc[mechanism_ids]
        indexes = pd.Index(texts.text_id).get_indexer(frame.text_id)
        require((indexes >= 0).all(), "Unresolved knowledge text")
        np.testing.assert_array_equal(texts.iloc[indexes].prompt_sha256, frame.prompt_sha256)
        mapping[view] = pd.Series(indexes, index=mechanism_ids)
    return meta_all, primary, meta, coverage, admitted, texts, mapping


def run(config_path, confirmation_sha256):
    started = time.perf_counter()
    require(confirmation_sha256 == sha(config_path), "Explicit current config SHA256 confirmation required")
    config, hashes, original_specs = preflight(config_path)
    output = ROOT/config["output_root"]
    output.mkdir(parents=False, exist_ok=False)
    with (output/"frozen_contract.json").open("xb") as stream:
        stream.write(Path(config_path).read_bytes())
    write_json(output/"input_manifest.json", [dict(path=p, sha256=h, bytes=Path(p).stat().st_size) for p, h in hashes.items()])
    write_json(output/"run_started.json", dict(created_utc=utc(), config_sha256=confirmation_sha256,
        input_manifest_sha256=sha(output/"input_manifest.json"), pid=os.getpid(), python=sys.executable,
        explicit_execution_confirmation=True, scope=config["scope"]))
    meta_all, primary, meta, coverage, admitted, texts, mapping = load_metadata(config)
    table(output/"atomic_index.tsv", meta)
    table(output/"cohort_membership.tsv", pd.concat([meta.iloc[rows].assign(cohort=cohort) for cohort, (_, rows) in admitted.items()], ignore_index=True))
    write_json(output/"object_registry.json", config["objects"])
    write_json(output/"contrast_registry.json", config["contrasts"])
    with np.load(ROOT/config["effects_root"]/"arrays.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["atomic_id"], meta_all.atomic_id.to_numpy(dtype=str))
        response, feature_axis = data["effect_log2fc"][primary].astype(np.float64), data["source_feature_row"]
    require(np.isfinite(response).all() and pd.Index(feature_axis).is_unique, "Bad frozen effect arrays")
    panels = read_table(ROOT/config["effects_root"]/"fold_gene_panels.tsv")
    panel_by_line = {}
    (output/"panels").mkdir()
    for line in config["folds"]:
        panel = panels[panels.heldout_cell_line.eq(line)].sort_values("rank").reset_index(drop=True)
        require(len(panel) == 3000 and panel.source_feature_row.is_unique, "Unchanged 3000-gene panel required")
        np.testing.assert_array_equal(panel["rank"], np.arange(1, 3001))
        columns = pd.Index(feature_axis).get_indexer(panel.source_feature_row)
        require((columns >= 0).all(), "Panel absent from exact effect feature axis")
        panel_by_line[line] = (panel, columns)
        table(output/"panels"/(line+".tsv"), panel)
    original_upper = {}
    for spec in original_specs:
        with np.load(spec["path"], allow_pickle=False) as data:
            np.testing.assert_array_equal(data["atomic_id"], meta.atomic_id.to_numpy(dtype=str))
            upper = data["similarity_upper"]
        require(upper.dtype == np.float64 and upper.shape == (2250*2249//2,) and np.isfinite(upper).all(), "E02 original geometry schema")
        original_upper[spec["method"]] = upper
    embeddings = {}
    for model in MODELS:
        with np.load(ROOT/config["embedding_root"]/(model+"__"+config["variant"]+".npz"), allow_pickle=False) as data:
            require(set(data.files) == {"X", "text_id", "prompt_sha256"}, "Knowledge embedding schema differs")
            np.testing.assert_array_equal(data["text_id"], texts.text_id.to_numpy(dtype=str))
            np.testing.assert_array_equal(data["prompt_sha256"], texts.prompt_sha256.to_numpy(dtype=str))
            x = data["X"]
        dim = 1024 if model in ["bge_m3", "qwen3_0_6b"] else 768
        require(x.dtype == np.float32 and x.shape == (len(texts), dim) and np.isfinite(x).all(), "Knowledge vector dtype/shape/finite check")
        require(np.max(np.abs(np.linalg.norm(x.astype(np.float64), axis=1)-1)) <= config["embedding_norm_atol"], "Knowledge vector unit norm check")
        embeddings[model] = x
    members = read_table(ROOT/config["metadata_root"]/"human_target_membership.tsv")
    target_entities = sorted(admitted["human_target"][0])
    target_x, target_axis = target_features(target_entities, members)
    with np.load(ROOT/config["morgan_root"]/"arrays.npz", allow_pickle=False) as data:
        morgan_entities, morgan_x = data["source_entity_key"], data["fingerprints"]
    require(len(morgan_entities) == len(set(morgan_entities)) == 188 and set(morgan_entities) == set(meta.source_entity_key), "Morgan entity universe differs")
    require(morgan_x.dtype == np.uint8 and morgan_x.shape == (188, 2048), "Frozen Morgan bit schema differs")
    native = {"human_target": (pd.Index(target_entities), binary_overlap(target_x)),
              "morgan_all": (pd.Index(morgan_entities), binary_overlap(morgan_x))}
    (output/"native_priors").mkdir()
    save_npz(output/"native_priors"/"human_target.npz", source_entity_key=np.array(target_entities), target_chembl_id=target_axis, membership=target_x)
    save_npz(output/"native_priors"/"morgan.npz", source_entity_key=morgan_entities, fingerprints=morgan_x)
    (output/"groups").mkdir()
    summaries, queries, group_manifest, representation_manifest = [], [], [], []
    for cohort_spec in config["cohorts"]:
        cohort = cohort_spec["cohort"]
        _, cohort_rows = admitted[cohort]
        for line in config["folds"]:
            panel, columns = panel_by_line[line]
            for dose_index, dose in enumerate(config["doses"]):
                rows = cohort_rows[meta.iloc[cohort_rows].cell_line.eq(line).to_numpy() & meta.iloc[cohort_rows].dose_value.eq(dose).to_numpy()]
                group_meta = meta.iloc[rows]
                n = len(rows)
                require(n == cohort_spec["group_sizes"][dose_index] and group_meta.source_entity_key.is_unique, "Fixed group entity support changed")
                gid = f"{cohort}__{line}__dose_{dose}"
                folder = output/"groups"/gid
                folder.mkdir(); (folder/"representations").mkdir()
                ids = group_meta.atomic_id.to_numpy(dtype=str)
                entities = group_meta.source_entity_key.to_numpy(dtype=str)
                ix = np.triu_indices(n, 1)
                truth = truth_geometry(response[np.ix_(rows, columns)], config["neighbor_k"], config["minimum_group_n"])
                save_npz(folder/"truth_geometry.npz", atomic_id=ids, source_entity_key=entities, primary_rows=rows,
                         source_feature_row=panel.source_feature_row.to_numpy(), original_ensembl_id=panel.original_ensembl_id.to_numpy(dtype=str),
                         similarity_upper=truth["similarity"][ix], **{key: value for key, value in truth.items() if key != "similarity"})
                record = dict(cohort=cohort, group_id=gid, cell_line=line, dose_value=dose, n_atoms=n, n_candidates=n-1, n_pairs=len(ix[0]), n_genes=3000)
                group_manifest.append(dict(**record, truth_valid_n=int(np.isfinite(truth["idcg"]).sum())))
                geometries = {}
                for spec in [s for s in config["objects"] if s["cohort"] == cohort]:
                    kind, model = spec["kind"], spec["model"]
                    extra = {}
                    if kind == "archived_complete":
                        upper = extract_upper(original_upper[model], rows, len(meta))
                    elif kind == "knowledge_cosine":
                        text_rows = mapping[spec["view"]].loc[ids].to_numpy()
                        values = embeddings[model][text_rows]
                        similarity, norms = cosine(values, unique_rows=True)
                        upper = similarity[ix]
                        extra = dict(text_id=texts.iloc[text_rows].text_id.to_numpy(dtype=str), prompt_sha256=texts.iloc[text_rows].prompt_sha256.to_numpy(dtype=str), arithmetic_row_norm=norms)
                    elif kind == "native":
                        lookup, similarity = native[cohort]
                        indexes = lookup.get_indexer(entities)
                        require((indexes >= 0).all(), "Native source missing admitted entity")
                        upper = similarity[np.ix_(indexes, indexes)][ix]
                    else:
                        upper, rank2_a, rank2_b = rank_fusion(geometries[spec["component_a"]], geometries[spec["component_b"]])
                        extra = dict(component_a=np.array(spec["component_a"]), component_b=np.array(spec["component_b"]), rank2_a=rank2_a, rank2_b=rank2_b)
                    geometries[spec["object_id"]] = upper
                    destination = folder/"representations"/(spec["object_id"]+".npz")
                    save_npz(destination, atomic_id=ids, similarity_upper=upper, **extra)
                    representation_manifest.append(dict(**spec, group_id=gid, path=str(destination.relative_to(output)), sha256=sha(destination)))
                    score = score_geometry(upper, truth, config["neighbor_k"], config["minimum_group_n"])
                    labels = dict(object_id=spec["object_id"], model=model, view=spec["view"], kind=kind)
                    summaries.append(dict(**record, **labels, rsa=score["rsa"], rsa_status=score["rsa_status"],
                        rsa_valid_pairs=len(ix[0]) if np.isfinite(score["rsa"]) else 0,
                        ndcg_mean=finite_mean(score["ndcg"]), random_ndcg_mean=finite_mean(score["random_ndcg"]),
                        excess_ndcg_mean=finite_mean(score["excess_ndcg"]), valid_ndcg_n=int(np.isfinite(score["ndcg"]).sum()),
                        truth_valid_n=int(np.isfinite(truth["idcg"]).sum()), n_independent_studies=1))
                    queries.append(pd.DataFrame(dict(**{key: record[key] for key in GROUP_KEYS+["n_candidates"]}, **labels,
                        atomic_id=ids, source_entity_key=entities, ndcg=score["ndcg"], random_ndcg=score["random_ndcg"],
                        excess_ndcg=score["excess_ndcg"], status=score["status"], truth_status=truth["status"],
                        truth_random_ndcg=truth["random_ndcg"], n_unique_effect_scores=truth["n_unique_effect_scores"],
                        n_unique_representation_scores=score["n_unique_representation_scores"])))
                print("KNOWLEDGE_ALIGNMENT_GROUP_PASS", gid, n, flush=True)
    summary, query = pd.DataFrame(summaries), pd.concat(queries, ignore_index=True)
    require(len(summary) == config["expected_group_results"] == 528 and len(query) == config["expected_query_rows"] == 50298, "Incomplete result registry")
    paired_query, paired_group = paired_tables(summary, query, config["contrasts"])
    require(len(paired_group) == config["expected_paired_group_results"] == 576 and len(paired_query) == config["expected_paired_query_rows"] == 60804, "Incomplete paired registry")
    for name, frame in [("group_summary.tsv", summary), ("query_metrics.tsv.gz", query),
                        ("paired_query_changes.tsv.gz", paired_query), ("paired_group_changes.tsv", paired_group)]:
        table(output/name, frame)
    for prefix, frame, paired in [("", summary, False), ("paired_", paired_group, True)]:
        lines, macro = equal_weight_summaries(frame, paired)
        table(output/(prefix+"equal_dose_by_cell_summary.tsv"), lines)
        table(output/(prefix+"descriptive_equal_dose_cell_summary.tsv"), macro)
    write_json(output/"group_manifest.json", group_manifest)
    write_json(output/"representation_manifest.json", representation_manifest)
    for path, expected in hashes.items():
        require(sha(path) == expected, "Input mutated during production: "+path)
    write_json(output/"audit.json", dict(status="PASS", independent_numeric_acceptance="PENDING_SEPARATE_CHECKER", created_utc=utc(),
        config_sha256=confirmation_sha256, script_sha256=sha(__file__), n_objects=44, n_contrasts=48,
        n_cohort_groups=len(group_manifest), n_group_results=len(summary), n_query_rows=len(query),
        n_paired_groups=len(paired_group), n_paired_queries=len(paired_query), n_independent_studies=1,
        all_input_hashes_unchanged=True, no_raw_reads=True, no_fits=True, no_reencoding=True, no_upstream_writes=True,
        original_complete_geometry_reused_bitwise=True, source_only_panels_unchanged=True,
        scope=config["scope"], broader_e08_knowledge_prediction_completed=False,
        software={name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scipy", "threadpoolctl"]},
        python=sys.version, executable=sys.executable, platform=platform.platform(),
        seconds=time.perf_counter()-started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    write_json(output/"output_manifest.json", dict(files=[dict(path=str(p.relative_to(output)), bytes=p.stat().st_size, sha256=sha(p))
        for p in sorted(output.rglob("*")) if p.is_file()]))
    print("KNOWLEDGE_ALIGNMENT_PASS", time.perf_counter()-started, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/knowledge_alignment_v1.json")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--static-preflight-only", action="store_true")
    parser.add_argument("--confirm-contract-sha256")
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        if args.preflight_only or args.static_preflight_only:
            config, hashes, specs = preflight(args.config, static_only=args.static_preflight_only)
            print(json.dumps(dict(status="PASS_STATIC_HASH_ONLY_RUNTIME_GATES_PENDING" if args.static_preflight_only else "PASS_HASH_ONLY_READY",
                n_inputs=len(hashes), n_original_representations=len(specs), config_sha256=sha(args.config), script_sha256=sha(__file__),
                numerical_arrays_loaded=False, production_output_created=False)))
        else:
            run(args.config, args.confirm_contract_sha256)
