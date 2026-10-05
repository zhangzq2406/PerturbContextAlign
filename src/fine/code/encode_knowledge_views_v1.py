#!/usr/bin/env python3
"""Thin offline adapter for independently admitted, immutable knowledge texts.

This module never decides knowledge admissibility or constructs prompts. It
encodes exact root-frozen UTF-8 strings and only reuses authenticated original
six-model clean caches. No expression/effect/prediction inputs are accepted.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
    os.environ[key] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import encode_clean_views as v1
import encode_clean_views_v2 as v2

ROOT = Path(__file__).resolve().parents[1]
VIEWS = ("complete_metadata+knowledge", "knowledge_only")
VARIANT = "verified_chembl37_snapshot"
MODELS = ("bge_m3", "sapbert", "qwen3_0_6b", "biomedbert", "medcpt_article", "medcpt_query")
POLICY = dict(max_length=512, batch_size=16, seed=20260711, device="cuda:0", dtype="float32",
              cpu_threads=4, l2_normalize=True, network_access="disabled", input_expression_or_effect_data=False,
              instruction_prompt=None, smoke_texts_per_view=3, smoke_cache_text_count=3,
              smoke_full_absolute_tolerance=0.0001)
HELPERS = {"encode_clean_views.py": "8bb2df4b1774338f89d74015d2d8b07a9a50344215431179aba30b09d5ecf409",
           "encode_clean_views_v2.py": "b53a24f201f85227b4c91c6553b6742e2fb50d51392ea79da83bdb4b9c35b7b0"}
LEGACY_CONFIGS = {"encoding_v1.json": "72fea4fdad20bd3d15a97eed1cb7857645cdfc3ce0fb6b4fdcbea41e0ea3def9",
                  "encoding_v2_additional_four.json": "88095de266308f158d6fa8120f4a98face38a2c02a82e09f3126794cf1a3dc5c"}
CACHE_ROOTS = (ROOT/"representations/clean_embeddings_v1/full", ROOT/"representations/clean_embeddings_v2_additional_four/full")
PACKAGES = ("torch", "transformers", "sentence-transformers", "numpy", "pandas")
TEXT_COLUMNS = ["text_row", "text_id", "prompt_sha256", "prompt_text", "views", "variants"]
MAP_COLUMNS = ["atomic_id", "source_entity_key", "view", "variant", "text_id", "text_row"]
require, sha256, write_json = v1.require, v1.sha256, v1.write_json


def read_tsv(path, usecols=None):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, usecols=usecols)


def write_tsv(path, frame):
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, sep="\t", index=False, lineterminator="\n")


def verify_file(path, expected, size=None):
    path = Path(path)
    require(path.is_file(), "missing input: "+str(path))
    require(size is None or path.stat().st_size == int(size), "byte count mismatch: "+str(path))
    require(sha256(path) == expected, "checksum mismatch: "+str(path))
    return dict(path=str(path), bytes=path.stat().st_size, sha256=expected)


def validate_config(config):
    require(config.get("configuration_status") == "FROZEN", "template/unfrozen configuration is not runnable")
    require(config.get("views") == list(VIEWS) and config.get("variant") == VARIANT, "view/variant policy changed")
    for key, expected in POLICY.items():
        require(key in config and config[key] == expected, "frozen inference policy changed: "+key)
    for key in ["expected_text_count", "expected_rowmap_count"]:
        require(type(config.get(key)) is int and config[key] > 0, "missing positive frozen count: "+key)
    require(config.get("metadata_independent_status") == "PASS_INDEPENDENT_KNOWLEDGE_INPUTS", "independent gate status changed")


def validate_registry(texts, rowmap, config):
    """Validate, but never rewrite, frozen text strings or admit/exclude entities."""
    require(set(TEXT_COLUMNS) <= set(texts) and set(MAP_COLUMNS) <= set(rowmap), "missing frozen input columns")
    require(len(texts) == config["expected_text_count"] and len(rowmap) == config["expected_rowmap_count"], "frozen input counts differ")
    require(texts.text_id.is_unique and texts.prompt_text.is_unique, "duplicate text ID or exact prompt")
    require(texts.prompt_text.map(lambda s: isinstance(s, str) and bool(s)).all(), "empty/nontext prompt")
    digest = texts.prompt_text.map(lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest())
    require(np.array_equal(digest, texts.prompt_sha256), "prompt bytes/hash mismatch")
    require(np.array_equal(texts.text_id, "text:"+digest), "text ID is not text:SHA256")
    expected_rows = [str(i) for i in range(len(texts))]
    require(texts.text_row.astype(str).tolist() == expected_rows, "text_row must preserve contiguous frozen file order")
    require(set(rowmap.view) == set(VIEWS) and set(rowmap.variant) == {VARIANT}, "unexpected mapping view/variant")
    require(not rowmap.duplicated(["atomic_id", "view", "variant"]).any(), "duplicate atom/view/variant")
    require(rowmap.atomic_id.ne("").all() and rowmap.source_entity_key.ne("").all(), "empty source join key")
    require(rowmap.groupby("atomic_id").view.nunique().eq(2).all(), "incomplete paired two-view atomic mapping")
    require(rowmap.groupby("atomic_id").source_entity_key.nunique().eq(1).all(), "one atom has inconsistent source entities")
    require(set(rowmap.text_id) == set(texts.text_id), "unreferenced or missing text ID")
    lookup = texts.set_index("text_id").text_row.astype(str)
    require(np.array_equal(rowmap.text_id.map(lookup), rowmap.text_row.astype(str)), "rowmap text_row identity mismatch")
    annotations = rowmap.groupby("text_id").agg(views=("view", lambda x: set(x)), variants=("variant", lambda x: set(x)))
    for row in texts.itertuples():
        require(set(row.views.split("|")) == annotations.loc[row.text_id, "views"], "text views annotation mismatch")
        require(set(row.variants.split("|")) == annotations.loc[row.text_id, "variants"], "text variants annotation mismatch")
    return texts.copy(), rowmap.copy()


def verify_independent_inputs(config):
    validate_config(config)
    root = Path(config["metadata_root"]).resolve()
    manifest_path = root/"output_manifest.json"
    sources = [verify_file(manifest_path, config["metadata_output_manifest_sha256"]),
               verify_file(config["metadata_independent_audit"], config["metadata_independent_audit_sha256"])]
    audit = json.loads(Path(config["metadata_independent_audit"]).read_text())
    require(audit.get("status") == "PASS_INDEPENDENT_KNOWLEDGE_INPUTS", "independent knowledge gate is not PASS")
    require(audit.get("producer_output_manifest_sha256") == config["metadata_output_manifest_sha256"], "independent gate is not bound to this producer manifest")
    manifest = json.loads(manifest_path.read_text())
    require(isinstance(manifest.get("files"), list), "metadata manifest requires files list")
    names = []
    for record in manifest["files"]:
        relative = Path(record["path"])
        require(not relative.is_absolute() and ".." not in relative.parts, "unsafe manifest path")
        path = (root/relative).resolve()
        require(path.is_relative_to(root), "manifest symlink leaves frozen metadata root")
        require(path.suffix in {".tsv", ".csv", ".json", ".md", ".txt"}, "non-metadata input disallowed")
        names.append(str(relative))
        sources.append(verify_file(path, record["sha256"], record.get("bytes", record.get("size_bytes"))))
    require(len(names) == len(set(names)), "duplicate metadata manifest path")
    require({"unique_texts.tsv", "rowmap.tsv"} <= set(names), "text/rowmap is not manifest-bound")
    texts, rowmap = validate_registry(read_tsv(root/"unique_texts.tsv", TEXT_COLUMNS), read_tsv(root/"rowmap.tsv", MAP_COLUMNS), config)
    return texts, rowmap, sources


def pooling_semantics(model):
    if model["model_key"] != "bge_m3":
        return v2.verify_pooling_settings(model)
    require(model["backend"] == "sentence_transformers" and model["pooling"] == "bundled", "BGE pooling/backend changed")
    snapshot = Path(model["snapshot_path"])
    pool = json.loads((snapshot/"1_Pooling/config.json").read_text())
    modes = {k: v for k, v in pool.items() if k.startswith("pooling_mode_")}
    require(modes.get("pooling_mode_cls_token") is True and sum(map(bool, modes.values())) == 1, "BGE bundled pooling changed")
    require(json.loads((snapshot/"config_sentence_transformers.json").read_text()).get("default_prompt_name") is None, "BGE default prompt changed")
    return dict(configured_pooling="bundled", effective_pooling="cls", instruction_prompt=None)


def authenticate_six_models(config):
    """Legacy validation only sees its original 3,760 texts, never E08 text counts."""
    sources, models, caches, old_text_reference, versions = [], [], {}, None, None
    require(len(config["legacy_groups"]) == 2, "exactly two original cache groups required")
    for group, allowed_root in zip(config["legacy_groups"], CACHE_ROOTS):
        config_path = Path(group["config_path"])
        require(config_path.resolve().parent == (ROOT/"configs").resolve() and config_path.name in LEGACY_CONFIGS, "unexpected legacy config source")
        require(group["config_sha256"] == LEGACY_CONFIGS[config_path.name], "legacy config lock differs")
        sources.append(verify_file(config_path, group["config_sha256"]))
        helper = Path(group["helper_path"])
        require(helper.resolve().parent == Path(__file__).resolve().parent and group["helper_sha256"] == HELPERS.get(helper.name), "helper lock differs")
        sources.append(verify_file(helper, group["helper_sha256"]))
        legacy = json.loads(config_path.read_text())
        for key in ["max_length", "batch_size", "seed", "device", "dtype", "cpu_threads", "l2_normalize", "network_access", "input_expression_or_effect_data", "smoke_full_absolute_tolerance"]:
            require(legacy[key] == config[key], "legacy inference policy differs: "+key)
        old_texts, _, old_sources, checked = v1.verify_inputs(legacy)
        sources.extend(old_sources)
        for name in ["input_manifest", "model_manifest", "model_files_manifest"]:
            path = Path(legacy["input_root"])/"output_sha256.tsv" if name == "input_manifest" else Path(legacy[name])
            sources.append(verify_file(path, legacy[name+"_sha256"]))
        root = Path(group["cache_root"]).resolve()
        require(root == allowed_root.resolve(), "cache outside original clean-six full roots")
        for name, key in [("audit.json", "audit_sha256"), ("encoding_manifest.tsv", "encoding_manifest_sha256"), ("encoded_texts.tsv", "encoded_texts_sha256")]:
            sources.append(verify_file(root/name, group[key]))
        audit = json.loads((root/"audit.json").read_text())
        require(audit["status"] == "PASS" and audit["stage"] == "full" and audit["variant"] == "source_name", "invalid original cache stage/variant")
        require(audit["config_sha256"] == group["config_sha256"] and audit["script_sha256"] == group["helper_sha256"], "unbound cache software")
        require(audit["expression_or_effect_data_read"] is False, "unclean legacy cache")
        require(audit["max_length"] == 512 and audit["batch_size"] == 16, "legacy cache inference differs")
        stored_texts = read_tsv(root/"encoded_texts.tsv")
        text_cols = ["text_id", "prompt_sha256", "prompt_text"]
        require(stored_texts[text_cols].equals(old_texts[text_cols]), "cache exact text/order differs")
        if old_text_reference is None:
            old_text_reference, versions = old_texts[text_cols].copy(), audit["package_versions"]
        require(old_text_reference.equals(old_texts[text_cols]), "six model cache text axes differ")
        require(versions == audit["package_versions"], "legacy cache package environments differ")
        encoding = read_tsv(root/"encoding_manifest.tsv").set_index("model_key")
        require(encoding.index.is_unique, "duplicate model cache manifest")
        for model in checked:
            key = model["model_key"]
            record = next(r for r in audit["outputs"] if r["model_key"] == key)
            require(model == next(r for r in audit["model_records"] if r["model_key"] == key), "cache revision/model/pooling/checkpoint record differs")
            archive = root/f"{key}__source_name.npz"
            sources.append(verify_file(archive, record["sha256"]))
            require(encoding.loc[key, "sha256"] == record["sha256"] and encoding.loc[key, "revision"] == model["revision"], "cache vector manifest differs")
            with np.load(archive, allow_pickle=False) as saved:
                require(set(saved.files) == {"X", "text_id", "prompt_sha256"}, "legacy archive schema differs")
                require(np.array_equal(saved["text_id"], old_texts.text_id.to_numpy(dtype=str)), "legacy vector IDs differ")
                require(np.array_equal(saved["prompt_sha256"], old_texts.prompt_sha256.to_numpy(dtype=str)), "legacy vector hashes differ")
                v1.validate_vectors(saved["X"], old_texts, model["dimension"])
                caches[key] = dict(X=saved["X"].copy(), texts=old_texts[text_cols].copy(), path=str(archive), sha256=record["sha256"])
            model["pooling_semantics"] = pooling_semantics(model)
            models.append(model)
    require(tuple(m["model_key"] for m in models) == MODELS and set(caches) == set(MODELS), "six-model registry/order differs")
    actual_versions = {name: importlib.metadata.version(name) for name in PACKAGES}
    require(actual_versions == versions, "runtime package versions differ from locked original caches")
    return models, caches, sources, versions


def plan_exact_cache(texts, models, caches):
    records, hit_sets = [], []
    for model in models:
        key = model["model_key"]
        cache = caches[key]
        previous = cache["texts"].set_index("text_id")
        positions = pd.Index(cache["texts"].text_id).get_indexer(texts.text_id)
        hits = set()
        for row, position in zip(texts.itertuples(), positions):
            hit = position >= 0
            if hit:
                prior = previous.loc[row.text_id]
                require(prior.prompt_sha256 == row.prompt_sha256 and prior.prompt_text == row.prompt_text, "cache ID collision or exact text mismatch")
                hits.add(row.text_id)
            records.append(dict(model_key=key, model_id=model["model_id"], revision=model["revision"],
                pooling=model["pooling"], effective_pooling=model["pooling_semantics"]["effective_pooling"],
                text_row=int(row.text_row), text_id=row.text_id, prompt_sha256=row.prompt_sha256,
                cache_hit=hit, needs_new_encoding=not hit, origin="exact_legacy_cache_copy" if hit else "new_offline_encoding",
                cache_path=cache["path"] if hit else "", cache_sha256=cache["sha256"] if hit else "", cache_row=int(position)))
        hit_sets.append(hits)
    require(all(hits == hit_sets[0] for hits in hit_sets), "model-specific cache membership differs")
    return pd.DataFrame(records), hit_sets[0]


def smoke_selection(texts, rowmap, old_texts):
    """Fixed longest-three per view, plus lexical-three original-cache controls."""
    selected_ids = set()
    for view in VIEWS:
        frame = texts[texts.text_id.isin(rowmap.loc[rowmap.view.eq(view), "text_id"])].copy()
        frame["n_characters"] = frame.prompt_text.str.len()
        selected_ids.update(frame.sort_values(["n_characters", "text_id"], ascending=[False, True]).head(3).text_id)
    controls = old_texts.sort_values("text_id").head(3).copy()
    require(len(controls) == 3, "fewer than three legacy smoke controls")
    keep = texts[texts.text_id.isin(selected_ids)][["text_id", "prompt_sha256", "prompt_text"]]
    frame = pd.concat([keep, controls[["text_id", "prompt_sha256", "prompt_text"]]], ignore_index=True)
    require(frame.groupby("text_id").prompt_text.nunique().eq(1).all(), "smoke exact text collision")
    frame = frame.drop_duplicates("text_id").sort_values("text_id").reset_index(drop=True)
    frame["is_knowledge_input"] = frame.text_id.isin(selected_ids)
    frame["is_legacy_reencode_control"] = frame.text_id.isin(controls.text_id)
    return frame


def check_token_lengths(frame, model_key, lengths):
    require(len(lengths) == len(frame) and len(lengths) > 0, "token-length axis mismatch")
    require(all(type(n) is int and 0 < n <= 512 for n in lengths), "token truncation requires explicit new review; >512 or invalid length")
    return pd.DataFrame(dict(model_key=model_key, text_id=frame.text_id, n_tokens=lengths))


def assemble_full(texts, encoded_frame, encoded, cache):
    positions = pd.Index(texts.text_id).get_indexer(encoded_frame.text_id)
    require((positions >= 0).all() and len(set(positions)) == len(positions), "new encoded text axis mismatch")
    old_positions = pd.Index(cache["texts"].text_id).get_indexer(texts.text_id)
    hits = old_positions >= 0
    require(set(positions) == set(np.flatnonzero(~hits)), "new encoding/cache partition differs")
    require(encoded.dtype == np.float32 and encoded.shape == (len(encoded_frame), cache["X"].shape[1]), "new vector schema mismatch")
    x = np.empty((len(texts), cache["X"].shape[1]), dtype=np.float32)
    x[positions] = encoded
    x[hits] = cache["X"][old_positions[hits]]
    require(np.array_equal(x[hits].view(np.uint32), cache["X"][old_positions[hits]].view(np.uint32)), "cache reuse is not bitwise exact")
    return x, int(hits.sum())


def encoder_for_model(model_key):
    require(model_key in MODELS, "model outside frozen six")
    return v1.encode if model_key in {"bge_m3", "sapbert"} else v2.encode


def output_manifest(folder):
    files = [dict(path=str(p.relative_to(folder)), bytes=p.stat().st_size, sha256=sha256(p))
             for p in sorted(folder.rglob("*")) if p.is_file()]
    write_json(folder/"output_manifest.json", dict(files=files))


def verify_stage(folder, config_hash, script_hash):
    manifest = json.loads((folder/"output_manifest.json").read_text())
    require({str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file()} == {r["path"] for r in manifest["files"]} | {"output_manifest.json"}, "stage inventory differs")
    for record in manifest["files"]:
        verify_file(folder/record["path"], record["sha256"], record["bytes"])
    audit = json.loads((folder/"audit.json").read_text())
    require(audit["status"] == "PASS" and audit["config_sha256"] == config_hash and audit["script_sha256"] == script_hash, "prior stage is not PASS with identical config/script")
    return audit


def run(stage, config_path):
    started = time.perf_counter()
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    validate_config(config)
    require(stage in {"preflight", "smoke", "full"}, "unknown stage")
    root, folder = Path(config["output_root"]), Path(config["output_root"])/stage
    require(not folder.exists(), "existing/partial output; refusing overwrite")
    config_hash, script_hash = sha256(config_path), sha256(__file__)
    prior = {}
    for name in ([] if stage == "preflight" else ["preflight"] if stage == "smoke" else ["preflight", "smoke"]):
        prior[name] = verify_stage(root/name, config_hash, script_hash)
    texts, rowmap, sources = verify_independent_inputs(config)
    models, caches, old_sources, versions = authenticate_six_models(config)
    sources.extend(old_sources)
    for path in [config_path, Path(__file__), ROOT/"tests/test_knowledge_encoding_v1.py"]:
        sources.append(verify_file(path, sha256(path)))
    sources = list({s["path"]: s for s in sources}.values())
    lineage, hit_ids = plan_exact_cache(texts, models, caches)
    old_texts = caches[MODELS[0]]["texts"]
    smoke = smoke_selection(texts, rowmap, old_texts)
    initial_stats = v2.checkpoint_stats(models)
    folder.mkdir(parents=True, exist_ok=False)
    log_handle = (folder/"run.log").open("x")
    previous_out, previous_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = v1.Tee(previous_out, log_handle), v1.Tee(previous_err, log_handle)
    audit = dict(stage=stage, created_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(),
        config_sha256=config_hash, script_sha256=script_hash, package_versions=versions,
        model_records=models, sources=sources, metadata_independent_audit_sha256=config["metadata_independent_audit_sha256"],
        metadata_output_manifest_sha256=config["metadata_output_manifest_sha256"], variant=VARIANT, views=list(VIEWS),
        unique_text_count=len(texts), registry_row_count=len(rowmap), atomic_count=int(rowmap.atomic_id.nunique()),
        source_entity_count=int(rowmap.source_entity_key.nunique()), cache_hits_per_model=len(hit_ids), new_texts_per_model=len(texts)-len(hit_ids),
        network_access="offline; local_files_only=True; trust_remote_code=False", expression_or_effect_data_read=False,
        prompt_construction_or_admission_decisions=False, historical_weight_identity_claim=False,
        outputs=[], independent_embedding_validation=False)
    try:
        print("START", stage, "PID", os.getpid(), flush=True)
        write_tsv(folder/"unique_texts.tsv", texts)
        write_tsv(folder/"rowmap.tsv", rowmap)
        write_tsv(folder/"cache_lineage.tsv", lineage)
        write_tsv(folder/"smoke_selection.tsv", smoke)
        if stage == "preflight":
            from transformers import AutoTokenizer
            # Include legacy smoke controls even when none is an E08 cache hit.
            token_frame = pd.concat([texts[["text_id", "prompt_sha256", "prompt_text"]], smoke[["text_id", "prompt_sha256", "prompt_text"]]])
            token_frame = token_frame.drop_duplicates("text_id").sort_values("text_id").reset_index(drop=True)
            rows = []
            for model in models:
                tokenizer = AutoTokenizer.from_pretrained(model["snapshot_path"], local_files_only=True, trust_remote_code=False)
                lengths = [len(ids) for ids in tokenizer(token_frame.prompt_text.tolist(), padding=False, truncation=False, add_special_tokens=True)["input_ids"]]
                rows.append(check_token_lengths(token_frame, model["model_key"], lengths))
            token_table = pd.concat(rows, ignore_index=True)
            token_table["is_knowledge_input"] = token_table.text_id.isin(texts.text_id)
            token_table["is_legacy_reencode_control"] = token_table.text_id.isin(smoke.loc[smoke.is_legacy_reencode_control, "text_id"])
            write_tsv(folder/"token_lengths.tsv", token_table)
            audit["truncated_prompts"] = 0
        else:
            import torch
            require(torch.cuda.is_available(), "CUDA unavailable; no silent CPU fallback")
            torch.set_num_threads(4); torch.manual_seed(20260711); np.random.seed(20260711)
            torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
            audit.update(gpu_name=torch.cuda.get_device_name(), torch_cuda=torch.version.cuda, device="cuda:0",
                         max_length=512, batch_size=16, seed=20260711, tf32_enabled=False, precision="float32; no autocast")
            selected = smoke if stage == "smoke" else texts.loc[~texts.text_id.isin(hit_ids)].reset_index(drop=True)
            write_tsv(folder/"newly_encoded_texts.tsv", selected)
            for model in models:
                key = model["model_key"]
                print("ENCODING", stage, key, len(selected), "texts", flush=True)
                if len(selected):
                    encoder = encoder_for_model(key)
                    encoded, runtime = encoder(selected, model, config)
                    v1.validate_vectors(encoded, selected, model["dimension"])
                else:
                    encoded, runtime = np.empty((0, model["dimension"]), dtype=np.float32), dict(no_new_texts_model_not_loaded=True)
                cache = caches[key]
                if stage == "smoke":
                    control_rows = np.flatnonzero(selected.is_legacy_reencode_control)
                    old_rows = pd.Index(cache["texts"].text_id).get_indexer(selected.iloc[control_rows].text_id)
                    require((old_rows >= 0).all(), "legacy smoke control missing from cache")
                    original = cache["X"][old_rows]
                    difference = float(np.max(np.abs(encoded[control_rows]-original)))
                    bitwise = bool(np.array_equal(encoded[control_rows].view(np.uint32), original.view(np.uint32)))
                    require(difference <= 0.0001, "legacy smoke control exceeds frozen 1e-4 tolerance")
                    frame, x = selected, encoded
                    qa = dict(legacy_control_n=len(control_rows), legacy_control_max_absolute_difference=difference,
                              legacy_control_bitwise_equal=bitwise, fixed_absolute_tolerance=0.0001)
                else:
                    x, copied = assemble_full(texts, selected, encoded, cache)
                    frame = texts
                    smoke_path = root/"smoke"/f"{key}__{VARIANT}.npz"
                    record = next(r for r in prior["smoke"]["outputs"] if r["model_key"] == key)
                    verify_file(smoke_path, record["sha256"])
                    with np.load(smoke_path, allow_pickle=False) as saved:
                        full_rows = pd.Index(texts.text_id).get_indexer(saved["text_id"])
                        keep = full_rows >= 0
                        require(keep.any(), "no knowledge smoke/full overlap")
                        difference = float(np.max(np.abs(x[full_rows[keep]]-saved["X"][keep])))
                        require(difference <= 0.0001, "knowledge smoke/full exceeds frozen 1e-4 tolerance")
                    qa = dict(bitwise_copied_cache_rows=copied, newly_encoded_rows=len(selected), all_cache_copies_bitwise_equal=True,
                              knowledge_smoke_full_n=int(keep.sum()), knowledge_smoke_full_max_absolute_difference=difference,
                              fixed_absolute_tolerance=0.0001)
                qa.update(v1.validate_vectors(x, frame, model["dimension"]))
                path = folder/f"{key}__{VARIANT}.npz"
                v1.save_vectors(path, frame, x)
                audit["outputs"].append(dict(model_key=key, model_id=model["model_id"], revision=model["revision"],
                    pooling=model["pooling"], effective_pooling=model["pooling_semantics"]["effective_pooling"], path=str(path),
                    sha256=sha256(path), bytes=path.stat().st_size, rows=len(frame), dimension=model["dimension"], dtype=str(x.dtype), qa=qa, runtime=runtime))
                print("MODEL_PASS", key, flush=True)
            write_tsv(folder/"encoded_texts.tsv", smoke if stage == "smoke" else texts)
            write_tsv(folder/"encoding_manifest.tsv", pd.DataFrame([{k: v for k, v in r.items() if k not in {"qa", "runtime"}} for r in audit["outputs"]]))
        require(initial_stats == v2.checkpoint_stats(models), "checkpoint stats changed during stage")
        for source in sources:
            verify_file(source["path"], source["sha256"], source["bytes"])
        audit.update(status="PASS", seconds=time.perf_counter()-started, checkpoint_files_rehashed_at_start=True,
                     checkpoint_start_end_stats_unchanged=True, checkpoint_stats=initial_stats, all_source_and_code_hashes_unchanged=True)
        write_json(folder/"audit.json", audit)
        print("STAGE_PASS", stage, audit["seconds"], flush=True)
    except Exception as exc:
        write_json(folder/"failure.json", dict(status="FAILED", error=repr(exc), partial_outputs_must_not_be_used=True))
        raise
    finally:
        sys.stdout, sys.stderr = previous_out, previous_err
        log_handle.close()
    # Seal only after closing the log so its hash cannot change after sealing.
    output_manifest(folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["preflight", "smoke", "full"], required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(args.stage, args.config)
