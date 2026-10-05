#!/usr/bin/env python3
"""Frozen source-name/exposure texts and six offline embeddings; no responses.

Stages: prepare (metadata, cache/checkpoint authentication, token lengths),
smoke (three cached texts re-encoded per model), full (130 new texts per model,
seven exact copied cache rows). Every destination is exclusive/no-overwrite.
Only entity/exposure columns are parsed from the approved atomic/compound TSVs.
The legacy helpers are hash-locked, imported without modification, and their
old 3,760-text validation is used only for the old cache, never the Kaggle data.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

for _key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
    os.environ[_key] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import encode_clean_views as v1
import encode_clean_views_v2 as v2

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/kaggle_encoding_v1.json"
TEMPLATE = "{entity}; dose: {dose} nM; duration: 24 h."
MODEL_KEYS = {"bge_m3", "sapbert", "qwen3_0_6b", "biomedbert", "medcpt_article", "medcpt_query"}
PACKAGES = ["torch", "transformers", "sentence-transformers", "numpy", "pandas"]
require, sha256, write_json = v1.require, v1.sha256, v1.write_json


def read_tsv(path, usecols=None):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, usecols=usecols)


def write_tsv(path, frame):
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, sep="\t", index=False, lineterminator="\n")


def verify_file(path, expected, size=None):
    path = Path(path)
    require(path.is_file(), f"missing input: {path}")
    require(size is None or path.stat().st_size == int(size), f"size mismatch: {path}")
    require(sha256(path) == expected, f"checksum mismatch: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": expected}


def canonical_decimal(value):
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("invalid decimal") from exc
    require(number.is_finite(), "nonfinite decimal")
    return format(number.normalize(), "f")


def dose_nm(value):
    # Never parse dose via binary floating point before the unit conversion.
    text = canonical_decimal(value)
    require(Decimal(text) > 0, "nonpositive active dose")
    return canonical_decimal(Decimal(text) * Decimal("1000"))


def make_prompt(entity, dose_um, duration):
    require(isinstance(entity, str) and entity != "", "missing source name")
    require(not any(char in entity for char in "\n\r\t"), "multiline source name")
    require(canonical_decimal(duration) == "24", "non-24-hour exposure")
    text = TEMPLATE.format(entity=entity, dose=dose_nm(dose_um))
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return text, digest, "text:" + digest


def build_registry(atoms, compounds, config):
    require(len(atoms) == config["expected_atom_count"], "unexpected atom count")
    require(atoms.atomic_id.is_unique and atoms.atomic_row.is_unique, "duplicate atom key")
    require(sorted(map(int, atoms.atomic_row)) == list(range(len(atoms))), "noncontiguous atomic row")
    keys = ["sm_lincs_id", "dose_uM", "timepoint_hr"]
    require(not compounds.duplicated(keys).any(), "ambiguous compound/exposure mapping")
    name_lookup = compounds.set_index(keys).sm_name.to_dict()
    rows = []
    for atom in atoms.to_dict("records"):
        key = tuple(atom[k] for k in keys)
        require(key in name_lookup and atom["sm_name"] == name_lookup[key], "source compound name mismatch")
        text, digest, uid = make_prompt(atom["sm_name"], atom["dose_uM"], atom["timepoint_hr"])
        rows.append({**atom, "source_entity_key": atom["sm_lincs_id"], "display_entity_name": atom["sm_name"],
                     "dose_nM": dose_nm(atom["dose_uM"]), "view": "entity_exposure", "variant": "source_name",
                     "text_id": uid, "prompt_sha256": digest, "prompt_text": text})
    registry = pd.DataFrame(rows)
    registry["atomic_row"] = registry.atomic_row.astype(int)
    registry = registry.sort_values("atomic_row").reset_index(drop=True)
    identity = ["sm_lincs_id", "sm_name", "dose_uM", "dose_nM", "timepoint_hr", "text_id", "prompt_sha256", "prompt_text"]
    texts = registry[identity].drop_duplicates().sort_values("text_id").reset_index(drop=True)
    require(texts.text_id.is_unique and texts.prompt_text.is_unique, "ambiguous distinct compound-to-text mapping")
    require(len(texts) == config["expected_text_count"], "unexpected text count")
    texts.insert(0, "text_row", range(len(texts)))
    texts["n_registry_rows"] = texts.text_id.map(registry.text_id.value_counts())
    texts["views"], texts["variants"] = "entity_exposure", "source_name"
    registry["text_row"] = registry.text_id.map(texts.set_index("text_id").text_row)
    return texts, registry


def verify_metadata(config):
    root = Path(config["metadata_root"])
    sources = [verify_file(root / "output_manifest.json", config["metadata_output_manifest_sha256"]),
               verify_file(config["metadata_independent_audit"], config["metadata_independent_audit_sha256"])]
    independent = json.loads(Path(config["metadata_independent_audit"]).read_text())
    require(independent["status"] == "PASS_INDEPENDENT_METADATA_ROLES_AND_FOLDS", "independent metadata gate not PASS")
    require(independent["producer_output_manifest_sha256"] == config["metadata_output_manifest_sha256"], "independent gate binds other metadata")
    require(independent["frozen_contract_sha256"] == config["frozen_study_contract_sha256"], "independent gate contract mismatch")
    manifest = json.loads((root / "output_manifest.json").read_text())
    require(manifest["contract_sha256"] == config["frozen_study_contract_sha256"], "metadata contract mismatch")
    # Hash all metadata artifacts, but parse no cell measurements, responses or labels.
    for record in manifest["files"]:
        sources.append(verify_file(root / record["path"], record["sha256"], record["size_bytes"]))
    atom_cols = ["atomic_row", "atomic_id", "sm_lincs_id", "sm_name", "dose_uM", "timepoint_hr"]
    compound_cols = ["sm_lincs_id", "sm_name", "dose_uM", "timepoint_hr"]
    texts, registry = build_registry(read_tsv(root / "atomic_index.tsv", atom_cols),
                                     read_tsv(root / "compound_index.tsv", compound_cols), config)
    return texts, registry, sources


def pooling_semantics(model):
    if model["model_key"] != "bge_m3":
        return v2.verify_pooling_settings(model)
    require(model["backend"] == "sentence_transformers" and model["pooling"] == "bundled", "BGE configuration changed")
    snapshot = Path(model["snapshot_path"])
    pool = json.loads((snapshot / "1_Pooling/config.json").read_text())
    modes = {k: v for k, v in pool.items() if k.startswith("pooling_mode_")}
    require(modes.get("pooling_mode_cls_token") is True and sum(map(bool, modes.values())) == 1, "BGE bundled pooling changed")
    sentence = json.loads((snapshot / "config_sentence_transformers.json").read_text())
    require(sentence.get("default_prompt_name") is None, "BGE default prompt changed")
    return {"configured_pooling": "bundled", "effective_pooling": "cls", "instruction_prompt": None}


def verify_legacy(config, texts):
    models, sources, caches, lineage_rows, expected_versions = [], [], {}, [], None
    for group in config["legacy_groups"]:
        sources.append(verify_file(group["config_path"], group["config_sha256"]))
        sources.append(verify_file(group["helper_path"], group["helper_sha256"]))
        old_config = json.loads(Path(group["config_path"]).read_text())
        for key in ["max_length", "batch_size", "seed", "device", "dtype", "cpu_threads", "l2_normalize", "smoke_full_absolute_tolerance"]:
            require(config[key] == old_config[key], "changed legacy inference policy: " + key)
        require(set(group["models"]) == {m["model_key"] for m in old_config["models"]}, "legacy model set differs")
        old_texts, old_registry, old_sources, checked_models = v1.verify_inputs(old_config)
        sources.extend(old_sources)
        for key in ["input_manifest", "model_manifest", "model_files_manifest"]:
            path = Path(old_config["input_root"]) / "output_sha256.tsv" if key == "input_manifest" else Path(old_config[key])
            sources.append(verify_file(path, old_config[key + "_sha256"]))
        cache_root = Path(group["cache_root"])
        for filename, key in [("audit.json", "audit_sha256"), ("encoding_manifest.tsv", "encoding_manifest_sha256"), ("encoded_texts.tsv", "encoded_texts_sha256")]:
            sources.append(verify_file(cache_root / filename, group[key]))
        audit = json.loads((cache_root / "audit.json").read_text())
        require(audit["status"] == "PASS" and audit["stage"] == "full", "cache is not PASS full")
        require(audit["config_sha256"] == group["config_sha256"] and audit["script_sha256"] == group["helper_sha256"], "cache software binding mismatch")
        require(audit["variant"] == "source_name" and audit["expression_or_effect_data_read"] is False, "cache is not clean source-name")
        require(audit["batch_size"] == config["batch_size"] and audit["max_length"] == config["max_length"], "cache inference settings mismatch")
        if expected_versions is None:
            expected_versions = audit["package_versions"]
        require(expected_versions == audit["package_versions"], "cache environments differ")
        encoded_texts = read_tsv(cache_root / "encoded_texts.tsv")
        require(encoded_texts[["text_id", "prompt_sha256", "prompt_text"]].equals(old_texts[["text_id", "prompt_sha256", "prompt_text"]]), "cache exact text/order binding differs")
        allowed = set(old_registry.loc[(old_registry.variant == "source_name") & (old_registry.view == "entity_exposure"), "text_id"])
        old_lookup = old_texts.set_index("text_id")
        hits = set(texts.text_id) & allowed
        require(len(hits) == config["expected_cache_hits_per_model"], "unexpected exact-text cache-hit count")
        for row in texts[texts.text_id.isin(hits)].to_dict("records"):
            prior = old_lookup.loc[row["text_id"]]
            require(row["prompt_sha256"] == prior.prompt_sha256 and row["prompt_text"] == prior.prompt_text, "text hash collision or text mismatch")
        manifest = read_tsv(cache_root / "encoding_manifest.tsv").set_index("model_key")
        for model in checked_models:
            key = model["model_key"]
            record = next(r for r in audit["outputs"] if r["model_key"] == key)
            audit_model = next(r for r in audit["model_records"] if r["model_key"] == key)
            require(model == audit_model, "cache model/checkpoint/revision/pooling record differs")
            archive = cache_root / f"{key}__source_name.npz"
            sources.append(verify_file(archive, record["sha256"]))
            require(manifest.loc[key, "sha256"] == record["sha256"] and manifest.loc[key, "revision"] == model["revision"], "cache manifest mismatch")
            with np.load(archive, allow_pickle=False) as saved:
                require(set(saved.files) == {"X", "text_id", "prompt_sha256"}, "cache NPZ schema mismatch")
                require(np.array_equal(saved["text_id"], old_texts.text_id.to_numpy(dtype=str)), "cache UID order differs")
                require(np.array_equal(saved["prompt_sha256"], old_texts.prompt_sha256.to_numpy(dtype=str)), "cache prompt hashes differ")
                v1.validate_vectors(saved["X"], old_texts, model["dimension"])
                rows = pd.Index(saved["text_id"]).get_indexer(texts.text_id)
                positions = {uid: i for uid, i in zip(texts.text_id, rows) if uid in hits}
                caches[key] = {"vectors": {uid: saved["X"][i].copy() for uid, i in positions.items()},
                               "positions": positions, "path": str(archive), "sha256": record["sha256"]}
            model["pooling_semantics"] = pooling_semantics(model)
            model["legacy_config_path"] = group["config_path"]
            model["helper_path"] = group["helper_path"]
            models.append(model)
            for row in texts.to_dict("records"):
                hit = row["text_id"] in hits
                lineage_rows.append({"model_key": key, "model_id": model["model_id"], "revision": model["revision"],
                                     "backend": model["backend"], "pooling": model["pooling"],
                                     "effective_pooling": model["pooling_semantics"]["effective_pooling"],
                                     "text_row": row["text_row"], "text_id": row["text_id"], "prompt_sha256": row["prompt_sha256"],
                                     "cache_hit": hit, "needs_new_encoding": not hit,
                                     "origin": "exact_legacy_cache_copy" if hit else "new_offline_encoding",
                                     "cache_path": str(archive) if hit else "", "cache_sha256": record["sha256"] if hit else "",
                                     "cache_row": caches[key]["positions"].get(row["text_id"], -1)})
    require({m["model_key"] for m in models} == MODEL_KEYS and len(models) == 6, "six-model set mismatch")
    first_hits = set(next(iter(caches.values()))["vectors"])
    require(all(set(c["vectors"]) == first_hits for c in caches.values()), "model-specific cache membership differs")
    require(len(texts) - len(first_hits) == config["expected_new_texts_per_model"], "unexpected new-text count")
    actual_versions = {name: importlib.metadata.version(name) for name in PACKAGES}
    require(actual_versions == expected_versions, "package versions differ from authenticated legacy caches")
    return models, caches, pd.DataFrame(lineage_rows), sources, actual_versions


def output_manifest(out):
    files = [{"path": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(out.iterdir()) if p.is_file()]
    write_json(out / "output_manifest.json", {"files": files})


def verify_stage(out, config_hash, script_hash):
    manifest = json.loads((out / "output_manifest.json").read_text())
    require({p.name for p in out.iterdir() if p.is_file()} == {r["path"] for r in manifest["files"]} | {"output_manifest.json"}, "stage inventory differs")
    for row in manifest["files"]:
        verify_file(out / row["path"], row["sha256"], row["bytes"])
    audit = json.loads((out / "audit.json").read_text())
    require(audit["status"] == "PASS" and audit["config_sha256"] == config_hash and audit["script_sha256"] == script_hash, "stage not PASS with identical code/config")
    return audit


def new_destination(path):
    Path(path).mkdir(parents=True, exist_ok=False)


def run(stage, config_path):
    start = time.perf_counter()
    config_path = Path(config_path)
    config = json.loads(config_path.read_text())
    require(config["template"] == TEMPLATE and config["instruction_prompt"] is None, "prompt policy differs")
    require(config["network_access"] == "disabled" and config["input_expression_or_effect_data"] is False, "nonclean input policy")
    views, output = Path(config["input_root"]), Path(config["output_root"])
    out = output / ("preflight" if stage == "prepare" else stage)
    # Refuse before reading large inputs, loading models or making partial files.
    require(not out.exists(), f"output exists; refusing overwrite: {out}")
    if stage == "prepare":
        require(not views.exists(), f"output exists; refusing overwrite: {views}")
    config_hash, script_hash = sha256(config_path), sha256(__file__)
    sources = [{"path": str(config_path), "bytes": config_path.stat().st_size, "sha256": config_hash},
               {"path": str(Path(__file__)), "bytes": Path(__file__).stat().st_size, "sha256": script_hash}]
    test_path = ROOT / "tests/test_kaggle_clean_encoding.py"
    sources.append({"path": str(test_path), "bytes": test_path.stat().st_size, "sha256": sha256(test_path)})
    texts, registry, metadata_sources = verify_metadata(config)
    sources.extend(metadata_sources)
    models, caches, lineage, old_sources, versions = verify_legacy(config, texts)
    sources.extend(old_sources)
    sources = list({item["path"]: item for item in sources}.values())
    cache_ids = set(next(iter(caches.values()))["vectors"])
    texts["cache_hit_all_six_models"] = texts.text_id.isin(cache_ids)
    texts["needs_new_encoding"] = ~texts.cache_hit_all_six_models
    texts["origin"] = np.where(texts.cache_hit_all_six_models, "exact_legacy_cache_copy", "new_offline_encoding")
    registry = registry.merge(texts[["text_id", "cache_hit_all_six_models", "needs_new_encoding", "origin"]], on="text_id", how="left", validate="many_to_one")
    initial_stats = v2.checkpoint_stats(models)
    audit = {"stage": stage, "created_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
             "config_sha256": config_hash, "script_sha256": script_hash, "package_versions": versions,
             "model_records": models, "network_access": "offline; local_files_only=True; trust_remote_code=False",
             "expression_or_effect_data_read": False, "responses_or_prediction_labels_read": False,
             "atomic_metadata_parsed_columns": ["atomic_row", "atomic_id", "sm_lincs_id", "sm_name", "dose_uM", "timepoint_hr"],
             "metadata_independent_gate": config["metadata_independent_audit_sha256"],
             "atomic_count": len(registry), "unique_text_count": len(texts), "exact_cache_hit_count_per_model": len(cache_ids),
             "new_text_count_per_model": len(texts) - len(cache_ids), "pooling_semantics": {m["model_key"]: m["pooling_semantics"] for m in models},
             "sources": sources, "outputs": []}
    if stage != "prepare":
        for prior in [views, output / "preflight"] + ([output / "smoke"] if stage == "full" else []):
            verify_stage(prior, config_hash, script_hash)
        require(read_tsv(views / "unique_texts.tsv")["text_id"].tolist() == texts.text_id.tolist(), "prepared text order changed")
    new_destination(out)
    try:
        if stage == "prepare":
            new_destination(views)
            write_tsv(views / "unique_texts.tsv", texts)
            write_tsv(views / "row_to_text_registry.tsv", registry)
            write_tsv(views / "cache_lineage.tsv", lineage)
            write_json(views / "config.json", config)
            from transformers import AutoTokenizer
            token_rows, token_qa = [], []
            for model in models:
                tokenizer = AutoTokenizer.from_pretrained(model["snapshot_path"], local_files_only=True, trust_remote_code=False)
                lengths = [len(ids) for ids in tokenizer(texts.prompt_text.tolist(), padding=False, truncation=False, add_special_tokens=True)["input_ids"]]
                require(max(lengths) <= config["max_length"], "token truncation requires review")
                token_rows.extend({"model_key": model["model_key"], "text_id": uid, "n_tokens": n} for uid, n in zip(texts.text_id, lengths))
                token_qa.append({"model_key": model["model_key"], "min_tokens": min(lengths), "max_tokens": max(lengths), "truncated": 0})
            write_tsv(out / "token_lengths.tsv", pd.DataFrame(token_rows))
            audit["token_qa"] = token_qa
        else:
            import torch
            require(torch.cuda.is_available(), "CUDA unavailable; no silent CPU fallback")
            torch.set_num_threads(config["cpu_threads"])
            torch.manual_seed(config["seed"])
            np.random.seed(config["seed"])
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            audit.update({"gpu_name": torch.cuda.get_device_name(), "torch_cuda": torch.version.cuda, "device": config["device"],
                          "batch_size": config["batch_size"], "max_length": config["max_length"], "seed": config["seed"], "tf32_enabled": False,
                          "precision_policy": "explicit float32 parameters and outputs, no autocast"})
            selected = texts[texts.text_id.isin(sorted(cache_ids)[:config["smoke_cache_text_count"]])] if stage == "smoke" else texts[~texts.text_id.isin(cache_ids)]
            selected = selected.reset_index(drop=True)
            require(len(selected) == (config["smoke_cache_text_count"] if stage == "smoke" else config["expected_new_texts_per_model"]), "encoding batch count differs")
            write_tsv(out / "newly_encoded_texts.tsv", selected)
            for model in models:
                key = model["model_key"]
                print("ENCODING", stage, key, len(selected), "texts", flush=True)
                encoder = v1.encode if key in {"bge_m3", "sapbert"} else v2.encode
                encoded, runtime = encoder(selected, model, config)
                v1.validate_vectors(encoded, selected, model["dimension"])
                if stage == "smoke":
                    prior = np.stack([caches[key]["vectors"][uid] for uid in selected.text_id])
                    difference = float(np.max(np.abs(encoded - prior)))
                    bitwise = bool(np.array_equal(encoded.view(np.uint32), prior.view(np.uint32)))
                    require(difference <= config["smoke_full_absolute_tolerance"], "cached-text re-encoding exceeds frozen legacy tolerance")
                    qa = {"cached_reencode_max_absolute_difference": difference, "cached_reencode_bitwise_equal": bitwise,
                          "fixed_absolute_tolerance": config["smoke_full_absolute_tolerance"], "tolerance_unchanged_from_legacy_config": True,
                          "not_a_bitwise_reencoding_claim": not bitwise}
                    frame, x = selected, encoded
                else:
                    positions = pd.Index(texts.text_id).get_indexer(selected.text_id)
                    x = np.empty((len(texts), model["dimension"]), dtype=np.float32)
                    x[positions] = encoded
                    for i, uid in enumerate(texts.text_id):
                        if uid in cache_ids:
                            x[i] = caches[key]["vectors"][uid]
                    require(all(np.array_equal(x[i].view(np.uint32), caches[key]["vectors"][uid].view(np.uint32))
                                for i, uid in enumerate(texts.text_id) if uid in cache_ids), "cache copy is not bitwise identical")
                    frame = texts
                    qa = {"newly_encoded_rows": len(selected), "bitwise_copied_cache_rows": len(cache_ids), "all_cache_copies_bitwise_equal": True}
                qa.update(v1.validate_vectors(x, frame, model["dimension"]))
                target = out / f"{key}__source_name.npz"
                v1.save_vectors(target, frame, x)
                audit["outputs"].append({"model_key": key, "model_id": model["model_id"], "revision": model["revision"],
                                         "pooling": model["pooling"], "effective_pooling": model["pooling_semantics"]["effective_pooling"],
                                         "path": str(target), "bytes": target.stat().st_size, "sha256": sha256(target),
                                         "rows": len(frame), "dimension": model["dimension"], "dtype": str(x.dtype), "qa": qa, "runtime": runtime})
                print("MODEL_PASS", key, qa, flush=True)
            write_tsv(out / "encoding_manifest.tsv", pd.DataFrame([{k: v for k, v in row.items() if k not in {"qa", "runtime"}} for row in audit["outputs"]]))
            write_tsv(out / "encoded_texts.tsv", selected if stage == "smoke" else texts)
            write_tsv(out / "cache_lineage.tsv", lineage[lineage.text_id.isin(selected.text_id)] if stage == "smoke" else lineage)
        require(initial_stats == v2.checkpoint_stats(models), "checkpoint file stats changed during stage")
        for source in sources:
            verify_file(source["path"], source["sha256"], source["bytes"])
        audit.update({"status": "PASS", "seconds": time.perf_counter() - start, "checkpoint_files_rehashed_at_start": True,
                      "checkpoint_start_end_stat_unchanged": True, "source_and_code_hashes_unchanged": True,
                      "checkpoint_stats": initial_stats, "independent_embedding_validation": False})
        write_json(out / "audit.json", audit)
        output_manifest(out)
        if stage == "prepare":
            write_json(views / "audit.json", audit)
            output_manifest(views)
        print("STAGE_PASS", stage, audit["seconds"], flush=True)
    except Exception as exc:
        write_json(out / "failure.json", {"status": "FAILED", "error": repr(exc), "partial_outputs_must_not_be_used": True})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "smoke", "full"], required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    run(args.stage, args.config)
