#!/usr/bin/env python3
"""Reuse locked encoders once for the shared exact text set of three mappings.

No original-name outputs or helpers are modified. Shared text vectors must be
joined through the original variant-aware registry; they are not shared drug
assignments. Anonymous performance evaluation is intentionally out of scope.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
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

require, sha256, write_json = v1.require, v1.sha256, v1.write_json
EXPECTED_MODELS = ("bge_m3", "sapbert", "qwen3_0_6b", "biomedbert", "medcpt_article", "medcpt_query")


def inherited_config(config):
    legacy = []
    for item in config["legacy_configs"]:
        require(sha256(item["path"]) == item["sha256"], "legacy config changed")
        legacy.append(json.loads(Path(item["path"]).read_text()))
    base, additional = legacy
    common = ("input_root", "model_manifest", "model_files_manifest", "views", "expected_text_count",
              "expected_atom_count", "max_length", "batch_size", "seed", "device", "dtype", "cpu_threads",
              "l2_normalize", "network_access", "input_expression_or_effect_data", "smoke_texts_per_view",
              "smoke_full_absolute_tolerance", "input_manifest_sha256", "model_manifest_sha256", "model_files_manifest_sha256")
    require(all(base[key] == additional[key] for key in common), "original six-model common settings differ")
    require(base["max_length"] == 512 and base["batch_size"] == 16 and base["dtype"] == "float32" and base["seed"] == 20260711,
            "frozen inference settings changed")
    for filename, expected in config["helper_hashes"].items():
        require(sha256(Path(__file__).with_name(filename)) == expected, "immutable helper changed")
    models = base["models"] + additional["models"]
    require(tuple(item["model_key"] for item in models) == EXPECTED_MODELS, "six model keys/order changed")
    result = dict(base)
    result.update({"models": models, "variant": config["variants"][0]})
    return result


def select_shared_registry(texts, registry, config, inference):
    variants = config["variants"]
    require(variants == [f"anonymous_seed_{seed}" for seed in config["anonymous_mapping_seeds"]], "anonymous seed/variant mismatch")
    require(variants == ["anonymous_seed_20260914", "anonymous_seed_20260915", "anonymous_seed_20260916"], "unknown mapping seed")
    reference_ids, frames, mappings, summaries = None, [], [], []
    for variant, seed in zip(variants, config["anonymous_mapping_seeds"]):
        local = {**inference, "variant": variant}
        frame, mapping = v1.select_source_texts(texts, registry, local)
        require(set(mapping.anonymous_seed.astype(str)) == {str(seed)}, "wrong anonymous mapping assignment")
        require(len(mapping) == config["expected_variant_registry_rows"], "variant registry row count")
        require(mapping.atomic_id.nunique() == config["expected_atoms_per_variant"], "variant atom count")
        require(mapping.groupby("source_entity_key").display_entity_name.nunique().eq(1).all(), "one entity maps to several codes")
        entities = mapping[["source_entity_key", "display_entity_name"]].drop_duplicates()
        require(len(entities) == 188 and entities.display_entity_name.is_unique, "anonymous mapping is not a bijection")
        ids = set(frame.text_id)
        if reference_ids is None:
            reference_ids = ids
        require(ids == reference_ids, "anonymous exact text sets differ; cannot silently pool/re-encode")
        frames.append(frame)
        mappings.append(mapping)
        summaries.append({"variant": variant, "anonymous_mapping_seed": seed, "registry_rows": len(mapping),
            "atoms": int(mapping.atomic_id.nunique()), "entities": len(entities), "unique_texts": len(ids),
            "view_unique_texts": {view: int(group.text_id.nunique()) for view, group in mapping.groupby("view")}})
    shared = pd.concat(frames).drop_duplicates("text_id").sort_values("text_id").reset_index(drop=True)
    joined = pd.concat(mappings, ignore_index=True)
    require(len(shared) == config["expected_shared_unique_texts"] and len(joined) == config["expected_union_registry_rows"], "union count changed")
    require(not joined.duplicated(["atomic_id", "view", "variant"]).any(), "registry join key duplicated")
    overlap = len(set(shared.text_id) & set(registry.loc[registry.variant == "source_name", "text_id"]))
    require(overlap == config["source_name_text_overlap"] == 0, "anonymous/original exact text overlap")
    return shared, joined, {"variants": summaries, "shared_text_count": len(shared), "registry_rows": len(joined),
        "anonymous_text_sets_identical": True, "source_name_text_overlap": overlap,
        "mapping_regenerated": False, "text_encode_repetitions_per_model": 1, "join_policy": config["join_policy"]}


def run(stage, config_path):
    started = time.perf_counter()
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    inference = inherited_config(config)
    root, out = Path(config["output_root"]), Path(config["output_root"]) / stage
    require(stage in ("preflight", "smoke", "full"), "invalid stage")
    require(not out.exists(), "refuse existing stage including partial outputs")
    config_hash = sha256(config_path)
    for prior_stage in ([] if stage == "preflight" else ["preflight"] if stage == "smoke" else ["preflight", "smoke"]):
        prior = json.loads((root / prior_stage / "audit.json").read_text())
        require(prior["status"] == "PASS" and prior["config_sha256"] == config_hash, "prior stage not PASS or config changed")
        require(prior["script_sha256"] == sha256(__file__), "script changed after preflight")
    # This immutable helper verifies the complete text manifest and all six
    # model-file inventories/checksums while selecting the first mapping only.
    _, _, sources, models = v1.verify_inputs(inference)
    clean_root = Path(inference["input_root"])
    frame, registry, mapping_summary = select_shared_registry(v1.tsv(clean_root / "unique_texts.tsv"),
        v1.tsv(clean_root / "row_to_text_registry.tsv"), config, inference)
    initial_stats = v2.checkpoint_stats(models)
    code_hashes = {str(path): sha256(path) for path in [config_path, Path(__file__), Path(v1.__file__), Path(v2.__file__)]}
    original_paths = []
    for item in config["legacy_configs"]:
        old_config = json.loads(Path(item["path"]).read_text())
        old_audit = Path(old_config["output_root"]) / "full/audit.json"
        audit = json.loads(old_audit.read_text())
        require(audit["status"] == "PASS" and audit["variant"] == "source_name", "original-name encoding not PASS")
        original_paths.append({"path": str(old_audit), "sha256": sha256(old_audit)})
        for record in audit["outputs"]:
            require(sha256(record["path"]) == record["sha256"], "original-name vector changed")
            original_paths.append({"path": record["path"], "sha256": record["sha256"]})
    out.mkdir(parents=True, exist_ok=False)
    log_root = Path(config["log_root"])
    log_root.mkdir(parents=True, exist_ok=True)
    log = (log_root / f"{stage}.log").open("x")
    oldout, olderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = v1.Tee(oldout, log), v1.Tee(olderr, log)
    try:
        print("START", stage, datetime.now(timezone.utc).isoformat(), "PID", os.getpid(), flush=True)
        registry.to_csv(out / "row_to_text_registry.tsv", sep="\t", index=False)
        write_json(out / "mapping_summary.json", mapping_summary)
        audit = {"stage": stage, "config_sha256": config_hash, "script_sha256": sha256(__file__),
            "variant": config["variant"], "variants": config["variants"], "full_text_count": len(frame),
            "registry_row_count": len(registry), "atomic_count_per_variant": 2256, "sources": sources,
            "model_records": models, "created_utc": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(), "package_versions": {name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "sentence-transformers", "numpy", "pandas")},
            "network_access": "offline; local_files_only=True", "expression_or_effect_data_read": False,
            "mapping_regenerated": False, "mapping_summary": mapping_summary,
            "code_and_config_hashes": code_hashes, "original_name_artifacts": original_paths,
            "scope": config["scope"], "anonymous_evaluation_performed": False}
        outputs = []
        if stage == "preflight":
            from transformers import AutoTokenizer
            tokens, token_qa = [], []
            for model in models:
                tokenizer = AutoTokenizer.from_pretrained(model["snapshot_path"], local_files_only=True, trust_remote_code=False)
                lengths = [len(item) for item in tokenizer(frame.prompt_text.tolist(), padding=False,
                    truncation=False, add_special_tokens=True)["input_ids"]]
                require(max(lengths) <= inference["max_length"], "anonymous text truncation requires review")
                tokens.extend({"model_key": model["model_key"], "text_id": uid, "n_tokens": int(length)}
                    for uid, length in zip(frame.text_id, lengths))
                token_qa.append({"model_key": model["model_key"], "min_tokens": min(lengths),
                    "max_tokens": max(lengths), "median_tokens": float(np.median(lengths)), "truncated": 0})
            pd.DataFrame(tokens).to_csv(out / "token_lengths.tsv", sep="\t", index=False)
            frame.to_csv(out / "selected_texts.tsv", sep="\t", index=False)
            audit["token_qa"] = token_qa
            print("TOKENIZER_PASS", token_qa, flush=True)
        else:
            import torch
            require(torch.cuda.is_available(), "CUDA unavailable; no silent CPU fallback")
            torch.set_num_threads(inference["cpu_threads"])
            torch.manual_seed(inference["seed"])
            np.random.seed(inference["seed"])
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            if stage == "smoke":
                frame = v1.smoke_selection(frame, registry, inference["smoke_texts_per_view"])
                require(len(frame) == 12, "expected 12 fixed smoke texts")
            frame.to_csv(out / "encoded_texts.tsv", sep="\t", index=False)
            audit.update({"device": inference["device"], "gpu_name": torch.cuda.get_device_name(),
                "torch_cuda": torch.version.cuda, "n_encoded_texts": len(frame), "batch_size": 16,
                "max_length": 512, "inference_seed": inference["seed"], "tf32_enabled": False,
                "precision_policy": "unchanged explicit float32 inference; no autocast"})
            for model in models:
                print("ENCODING", model["model_key"], len(frame), flush=True)
                encoder = v1.encode if model["model_key"] in ("bge_m3", "sapbert") else v2.encode
                x, runtime = encoder(frame, model, inference)
                qa = v1.validate_vectors(x, frame, model["dimension"])
                if stage == "full":
                    smoke_path = root / "smoke" / f'{model["model_key"]}__anonymous_shared_text.npz'
                    smoke_audit = json.loads((root / "smoke/audit.json").read_text())
                    smoke_record = next(item for item in smoke_audit["outputs"] if item["model_key"] == model["model_key"])
                    require(sha256(smoke_path) == smoke_record["sha256"], "smoke output changed")
                    with np.load(smoke_path, allow_pickle=False) as smoke:
                        rows = pd.Index(frame.text_id).get_indexer(smoke["text_id"])
                        require((rows >= 0).all(), "smoke text missing from full")
                        difference = float(np.max(np.abs(x[rows] - smoke["X"])))
                        require(difference <= inference["smoke_full_absolute_tolerance"], "smoke/full mismatch")
                        qa["smoke_full_max_absolute_difference"] = difference
                path = out / f'{model["model_key"]}__anonymous_shared_text.npz'
                v1.save_vectors(path, frame, x)
                outputs.append({"model_key": model["model_key"], "path": str(path), "sha256": sha256(path),
                    "revision": model["revision"], "pooling": model["pooling"], "rows": len(frame),
                    "dimension": x.shape[1], "dtype": str(x.dtype), "runtime": runtime, "qa": qa,
                    "shared_variants": config["variants"]})
                print("MODEL_PASS", model["model_key"], runtime, flush=True)
                del x
            pd.DataFrame([{key: value for key, value in record.items() if key not in ("runtime", "qa", "shared_variants")}
                          for record in outputs]).to_csv(out / "encoding_manifest.tsv", sep="\t", index=False)
        require(v2.checkpoint_stats(models) == initial_stats, "checkpoint stat changed during stage")
        for path, digest in code_hashes.items():
            require(sha256(path) == digest, "code/config changed during stage")
        for record in sources + original_paths:
            require(sha256(record["path"]) == record["sha256"], "source/original vector changed during stage")
        for item in config["legacy_configs"]:
            require(sha256(item["path"]) == item["sha256"], "legacy config changed during stage")
        audit.update({"status": "PASS", "seconds": time.perf_counter() - started, "outputs": outputs,
            "checkpoint_files_rehashed_at_start": True, "checkpoint_start_end_stat_unchanged": True,
            "source_code_and_original_name_hashes_unchanged": True})
        write_json(out / "audit.json", audit)
        print("STAGE_PASS", stage, audit["seconds"], flush=True)
    except Exception as exc:
        write_json(out / "failure.json", {"status": "FAILED", "error": repr(exc), "partial_outputs_must_not_be_used": True})
        raise
    finally:
        sys.stdout, sys.stderr = oldout, olderr
        log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "smoke", "full"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(args.stage, args.config)
