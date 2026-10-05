#!/usr/bin/env python3
"""Offline remaining-four-model extension; immutable v1 helpers are imported.

Unlike the deliberately mean-only v1 HF inference branch, this version uses
the exact locked mean/CLS setting. Qwen retains its bundled last-token pooling
without any instruction prompt. No response, effect, or expression is read.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import sys
import time

for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
    os.environ[name] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import encode_clean_views as v1

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/encoding_v2_additional_four.json"
require, sha256, write_json = v1.require, v1.sha256, v1.write_json


def pool_hidden(hidden, attention_mask, pooling):
    """Explicit locked HF pooling; unknown settings are errors, never defaults."""
    require(pooling in {"mean", "cls"}, "unsupported HF pooling")
    require(hidden.ndim == 3 and attention_mask.ndim == 2 and hidden.shape[:2] == attention_mask.shape, "hidden/mask shape mismatch")
    require(bool((attention_mask.sum(dim=1) > 0).all()), "empty attention mask")
    if pooling == "cls":
        require(bool((attention_mask[:, 0] != 0).all()), "CLS token masked out")
        return hidden[:, 0]
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)


def verify_pooling_settings(model_record):
    snapshot = Path(model_record["snapshot_path"])
    if model_record["backend"] == "sentence_transformers":
        require(model_record["model_key"] == "qwen3_0_6b" and model_record["pooling"] == "bundled", "unexpected ST model for remaining-four run")
        pool = json.loads((snapshot / "1_Pooling/config.json").read_text())
        modes = {name: value for name, value in pool.items() if name.startswith("pooling_mode_")}
        require(modes.get("pooling_mode_lasttoken") is True and sum(bool(value) for value in modes.values()) == 1, "Qwen pooling is not exclusively last-token")
        sentence = json.loads((snapshot / "config_sentence_transformers.json").read_text())
        require(sentence.get("default_prompt_name") is None, "unexpected Qwen default instruction")
        return {"configured_pooling": "bundled", "effective_pooling": "last_token", "instruction_prompt": None,
                "include_prompt": pool.get("include_prompt"), "checkpoint_dtype_declaration": json.loads((snapshot / "config.json").read_text()).get("torch_dtype")}
    require(model_record["backend"] == "hf_mean" and model_record["pooling"] in {"mean", "cls"}, "unsupported locked backend/pooling")
    return {"configured_pooling": model_record["pooling"], "effective_pooling": model_record["pooling"], "instruction_prompt": None}


def checkpoint_stats(models):
    records = []
    for model in models:
        for row in model["verified_files"]:
            path = Path(row["file"])
            stat = path.stat()
            records.append({"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                            "ctime_ns": stat.st_ctime_ns, "inode": stat.st_ino, "device": stat.st_dev})
    return records


def encode(frame, model_record, config):
    import torch
    from transformers import AutoModel, AutoTokenizer
    semantics = verify_pooling_settings(model_record)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    snapshot = model_record["snapshot_path"]
    if model_record["backend"] == "sentence_transformers":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(snapshot, device=config["device"], local_files_only=True, trust_remote_code=False)
        model.max_seq_length = config["max_length"]
        model.float().eval()
        require(getattr(model, "default_prompt_name", None) is None, "loaded model has default prompt")
        x = np.asarray(model.encode(frame.prompt_text.tolist(), batch_size=config["batch_size"],
                       show_progress_bar=True, normalize_embeddings=True, convert_to_numpy=True), dtype=np.float32)
    else:
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
        model = AutoModel.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False).float().to(config["device"]).eval()
        chunks = []
        with torch.inference_mode():
            for offset in range(0, len(frame), config["batch_size"]):
                texts = frame.prompt_text.iloc[offset:offset + config["batch_size"]].tolist()
                batch = tokenizer(texts, padding=True, truncation=True, max_length=config["max_length"], return_tensors="pt")
                batch = {key: value.to(config["device"]) for key, value in batch.items()}
                hidden = model(**batch).last_hidden_state
                pooled = pool_hidden(hidden, batch["attention_mask"], model_record["pooling"])
                chunks.append(torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy().astype(np.float32))
                if offset % (config["batch_size"] * 40) == 0:
                    print(model_record["model_key"], min(offset + config["batch_size"], len(frame)), "/", len(frame), flush=True)
        x = np.concatenate(chunks)
    x = np.asarray(x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12), dtype=np.float32)
    torch.cuda.synchronize()
    runtime = {"seconds": time.perf_counter() - started,
               "model_parameter_dtypes": sorted({str(value.dtype) for value in model.parameters()}),
               "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
               "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
               "process_peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
               "pooling_semantics": semantics}
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return x, runtime


def run(stage, config_path):
    start = time.perf_counter()
    config_path = Path(config_path)
    config = json.loads(config_path.read_text())
    require(config["max_length"] == 512 and config["batch_size"] == 16 and config["dtype"] == "float32", "frozen inference settings changed")
    require({row["model_key"] for row in config["models"]} == {"qwen3_0_6b", "biomedbert", "medcpt_article", "medcpt_query"}, "remaining-four model set changed")
    require(sha256(v1.__file__) == config["v1_helper_sha256"], "immutable v1 helper hash changed")
    root = Path(config["output_root"])
    out = root / stage
    out.mkdir(parents=True, exist_ok=False)
    log_root = Path(config["log_root"])
    log_root.mkdir(parents=True, exist_ok=True)
    handle = (log_root / f"{stage}.log").open("x")
    previous_stdout, previous_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = v1.Tee(previous_stdout, handle), v1.Tee(previous_stderr, handle)
    try:
        print("START", stage, datetime.now(timezone.utc).isoformat(), "PID", os.getpid(), flush=True)
        frame, registry, sources, models = v1.verify_inputs(config)
        initial_stats = checkpoint_stats(models)
        code_config_hashes = {str(path): sha256(path) for path in [config_path, Path(__file__), Path(v1.__file__)]}
        config_hash = sha256(config_path)
        for required_stage in ([] if stage == "preflight" else ["preflight"] if stage == "smoke" else ["preflight", "smoke"]):
            audit = json.loads((root / required_stage / "audit.json").read_text())
            require(audit["status"] == "PASS" and audit["config_sha256"] == config_hash, "required stage absent or config changed")
        audit = {"stage": stage, "config_sha256": config_hash, "script_sha256": sha256(__file__),
                 "v1_helper_sha256": sha256(v1.__file__), "created_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
                 "package_versions": {name: importlib.metadata.version(name) for name in ["torch", "transformers", "sentence-transformers", "numpy", "pandas"]},
                 "sources": sources, "model_records": models, "network_access": "offline; local_files_only=True",
                 "expression_or_effect_data_read": False, "variant": config["variant"], "full_text_count": len(frame),
                 "registry_row_count": len(registry), "atomic_count": int(registry.atomic_id.nunique()),
                 "pooling_semantics": {model["model_key"]: verify_pooling_settings(model) for model in models}}
        outputs = []
        if stage == "preflight":
            from transformers import AutoTokenizer
            tokens, token_qa = [], []
            for model in models:
                tokenizer = AutoTokenizer.from_pretrained(model["snapshot_path"], local_files_only=True, trust_remote_code=False)
                lengths = [len(value) for value in tokenizer(frame.prompt_text.tolist(), padding=False, truncation=False, add_special_tokens=True)["input_ids"]]
                require(max(lengths) <= 512, "token truncation requires review")
                tokens.extend({"model_key": model["model_key"], "text_id": uid, "n_tokens": int(length)} for uid, length in zip(frame.text_id, lengths))
                token_qa.append({"model_key": model["model_key"], "min_tokens": min(lengths), "max_tokens": max(lengths), "median_tokens": float(np.median(lengths)), "truncated": 0})
            pd.DataFrame(tokens).to_csv(out / "token_lengths.tsv", sep="\t", index=False)
            frame.to_csv(out / "selected_texts.tsv", sep="\t", index=False)
            audit["token_qa"] = token_qa
            print("TOKENIZER_PASS", token_qa, flush=True)
        else:
            import torch
            require(torch.cuda.is_available(), "CUDA unavailable, no silent CPU fallback")
            torch.set_num_threads(config["cpu_threads"])
            torch.manual_seed(config["seed"])
            np.random.seed(config["seed"])
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            frame = v1.smoke_selection(frame, registry, config["smoke_texts_per_view"]) if stage == "smoke" else frame
            frame.to_csv(out / "encoded_texts.tsv", sep="\t", index=False)
            audit.update({"device": config["device"], "gpu_name": torch.cuda.get_device_name(), "torch_cuda": torch.version.cuda,
                          "batch_size": 16, "max_length": 512, "seed": config["seed"], "tf32_enabled": False,
                          "precision_policy": "explicit float32 parameters and outputs, no autocast", "n_encoded_texts": len(frame)})
            for model in models:
                print("ENCODING", model["model_key"], len(frame), flush=True)
                x, runtime = encode(frame, model, config)
                qa = v1.validate_vectors(x, frame, model["dimension"])
                if stage == "full":
                    smoke_path = root / "smoke" / f'{model["model_key"]}__source_name.npz'
                    smoke_audit = json.loads((root / "smoke/audit.json").read_text())
                    record = next(item for item in smoke_audit["outputs"] if item["model_key"] == model["model_key"])
                    require(sha256(smoke_path) == record["sha256"], "smoke hash changed")
                    with np.load(smoke_path, allow_pickle=False) as smoke:
                        rows = pd.Index(frame.text_id).get_indexer(smoke["text_id"])
                        require((rows >= 0).all(), "smoke UID missing")
                        delta = float(np.max(np.abs(x[rows] - smoke["X"])))
                        require(delta <= config["smoke_full_absolute_tolerance"], "smoke/full vector discrepancy")
                        qa["smoke_full_max_absolute_difference"] = delta
                target = out / f'{model["model_key"]}__source_name.npz'
                v1.save_vectors(target, frame, x)
                outputs.append({"model_key": model["model_key"], "path": str(target), "sha256": sha256(target),
                                "revision": model["revision"], "pooling": model["pooling"], "rows": x.shape[0], "dimension": x.shape[1],
                                "dtype": str(x.dtype), "qa": qa, "runtime": runtime})
                print("MODEL_PASS", model["model_key"], runtime, flush=True)
                del x
            pd.DataFrame([{key: value for key, value in item.items() if key not in {"qa", "runtime"}} for item in outputs]).to_csv(out / "encoding_manifest.tsv", sep="\t", index=False)
        final_stats = checkpoint_stats(models)
        require(initial_stats == final_stats, "checkpoint content-related stat changed during stage")
        for path, initial in code_config_hashes.items():
            require(sha256(path) == initial, "code/config changed during stage")
        for source in sources:
            require(sha256(source["path"]) == source["sha256"], "text source changed during stage")
        audit.update({"status": "PASS", "seconds": time.perf_counter() - start, "outputs": outputs,
                      "checkpoint_files_rehashed_at_start": True, "checkpoint_start_end_stat_unchanged": True,
                      "checkpoint_stats": final_stats, "source_and_code_hashes_unchanged": True})
        write_json(out / "audit.json", audit)
        print("STAGE_PASS", stage, audit["seconds"], flush=True)
    except Exception as exc:
        write_json(out / "failure.json", {"status": "FAILED", "error": repr(exc), "partial_outputs_must_not_be_used": True})
        raise
    finally:
        sys.stdout, sys.stderr = previous_stdout, previous_stderr
        handle.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["preflight", "smoke", "full"], required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    run(args.stage, args.config)
