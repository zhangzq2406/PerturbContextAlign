#!/usr/bin/env python3
"""Offline, hash-locked clean-view inference; no expression/effect access.

Run preflight, smoke, then full. Each stage refuses an existing destination.
The archive keys are text_id, prompt_sha256, X; join text_id using the frozen
row_to_text_registry, retaining the variant key. No atomic UID is fabricated.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import sys
import time
from datetime import datetime, timezone

for _key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
    os.environ[_key] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/encoding_v1.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def tsv(path):
    return pd.read_csv(path, sep="\t", keep_default_na=False)


def select_source_texts(texts, registry, config):
    """Validate exact UTF-8 IDs and frozen many-atoms-to-one-text linkage."""
    require(texts.text_id.is_unique, "duplicate text_id")
    require(texts.prompt_text.is_unique, "duplicate exact text")
    hashes = texts.prompt_text.map(lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest())
    require(hashes.equals(texts.prompt_sha256), "prompt hash mismatch")
    require((texts.text_id == "text:" + hashes).all(), "text_id is not text:SHA256")
    selected_registry = registry[registry.variant == config["variant"]].copy()
    require(set(selected_registry.view) == set(config["views"]), "wrong view set")
    require(not selected_registry.duplicated(["atomic_id", "view", "variant"]).any(), "duplicate atom/view/variant")
    counts = selected_registry.groupby("atomic_id").view.nunique()
    require(len(counts) == config["expected_atom_count"] and (counts == 4).all(), "incomplete atom-view registry")
    selected = texts[texts.text_id.isin(selected_registry.text_id)].sort_values("text_id").reset_index(drop=True)
    require(len(selected) == config["expected_text_count"], "wrong unique source-name text count")
    require(set(selected.text_id) == set(selected_registry.text_id), "registry references missing text")
    lookup = selected.set_index("text_id").prompt_sha256
    require((selected_registry.text_id.map(lookup) == selected_registry.prompt_sha256).all(), "registry hash mismatch")
    require(selected.variants.map(lambda item: config["variant"] in item.split("|")).all(), "wrong variants annotation")
    return selected, selected_registry


def verify_inputs(config):
    root = Path(config["input_root"])
    require(sha256(root / "output_sha256.tsv") == config["input_manifest_sha256"], "changed clean-view manifest")
    source_records = []
    for row in tsv(root / "output_sha256.tsv").to_dict("records"):
        path = root / row["file"]
        require(path.stat().st_size == int(row["bytes"]) and sha256(path) == row["sha256"], f"changed input: {path}")
        source_records.append({"path": str(path), "bytes": int(row["bytes"]), "sha256": row["sha256"]})
    frame, registry = select_source_texts(tsv(root / "unique_texts.tsv"), tsv(root / "row_to_text_registry.tsv"), config)
    for key in ("model_manifest", "model_files_manifest"):
        require(sha256(config[key]) == config[key + "_sha256"], "changed " + key)
    model_manifest = tsv(config["model_manifest"])
    files = tsv(config["model_files_manifest"])
    models = []
    for model in config["models"]:
        row = model_manifest[model_manifest.model_key == model["model_key"]]
        require(len(row) == 1, "ambiguous model manifest")
        row = row.iloc[0]
        require(row.locked_revision == model["revision"] and row.model_id_original == model["model_id"], "model lock mismatch")
        require(row.pooling_config == model["pooling"] and row.backend == model["backend"], "pooling/backend changed")
        snapshot = Path(row.snapshot_path)
        require(snapshot.name == model["revision"] and snapshot.is_dir(), "missing exact snapshot")
        matched_files = files[files.model_key == model["model_key"]]
        require(not matched_files.empty, "missing checkpoint file manifest")
        require(set(matched_files.revision) == {model["revision"]}, "file revision mismatch")
        actual_files = {str(path) for path in snapshot.rglob("*") if path.is_file()}
        require(actual_files == set(matched_files.file), "snapshot file inventory changed")
        checked = []
        for record in matched_files.to_dict("records"):
            path = Path(record["file"])
            require(path.stat().st_size == int(record["bytes"]) and sha256(path) == record["sha256"], f"model checksum mismatch: {path}")
            checked.append(record)
        models.append({**model, "snapshot_path": str(snapshot), "verified_files": checked})
    return frame, registry, source_records, models


def validate_vectors(x, frame, dimension):
    require(x.dtype == np.float32, "wrong output dtype")
    require(x.shape == (len(frame), dimension), "wrong output shape")
    require(np.isfinite(x).all(), "non-finite embedding")
    norms = np.linalg.norm(x, axis=1)
    require(np.allclose(norms, 1.0, atol=2e-6, rtol=0), "non-unit embedding")
    require(frame.text_id.is_unique, "nonunique output UID")
    return {"min_norm": float(norms.min()), "max_norm": float(norms.max()), "all_finite": True, "text_id_unique": True}


def save_vectors(path, frame, x):
    with Path(path).open("xb") as handle:
        np.savez_compressed(handle, X=x, text_id=frame.text_id.to_numpy(dtype=str),
                            prompt_sha256=frame.prompt_sha256.to_numpy(dtype=str))
    with np.load(path, allow_pickle=False) as saved:
        require(set(saved.files) == {"X", "text_id", "prompt_sha256"}, "unexpected NPZ keys")
        require(np.array_equal(saved["text_id"], frame.text_id.to_numpy(dtype=str)), "saved UID order differs")
        require(np.array_equal(saved["X"], x), "saved vectors differ")


def smoke_selection(frame, registry, n_per_view):
    ids = set()
    for view in sorted(registry.view.unique()):
        candidates = frame[frame.text_id.isin(registry.loc[registry.view == view, "text_id"])].copy()
        candidates["length"] = candidates.prompt_text.str.len()
        ids.update(candidates.sort_values(["length", "text_id"], ascending=[False, True]).head(n_per_view).text_id)
    return frame[frame.text_id.isin(ids)].copy().reset_index(drop=True)


def encode(frame, model_record, config):
    import torch
    from transformers import AutoModel, AutoTokenizer
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    snapshot = model_record["snapshot_path"]
    if model_record["backend"] == "sentence_transformers":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(snapshot, device=config["device"], local_files_only=True, trust_remote_code=False)
        model.max_seq_length = config["max_length"]
        model.float().eval()
        require(not getattr(model, "default_prompt_name", None), "unexpected bundled default prompt")
        x = np.asarray(model.encode(frame.prompt_text.tolist(), batch_size=config["batch_size"],
                       normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True), dtype=np.float32)
    else:
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
        model = AutoModel.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False).float().to(config["device"]).eval()
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(frame), config["batch_size"]):
                text = frame.prompt_text.iloc[start:start + config["batch_size"]].tolist()
                batch = tokenizer(text, padding=True, truncation=True, max_length=config["max_length"], return_tensors="pt")
                batch = {key: value.to(config["device"]) for key, value in batch.items()}
                hidden = model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
                chunks.append(torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy().astype(np.float32))
                if start % (config["batch_size"] * 40) == 0:
                    print(model_record["model_key"], min(start + config["batch_size"], len(frame)), "/", len(frame), flush=True)
        x = np.concatenate(chunks)
    # Same final l2_dense step as locked legacy code.
    x = np.asarray(x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12), dtype=np.float32)
    torch.cuda.synchronize()
    runtime = {"seconds": time.perf_counter() - started,
               "model_parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
               "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
               "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
               "process_peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return x, runtime


class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, value):
        for stream in self.streams:
            stream.write(value)
        return len(value)
    def flush(self):
        for stream in self.streams:
            stream.flush()


def run(stage, config_path):
    start = time.perf_counter()
    config = json.loads(Path(config_path).read_text())
    require(config["max_length"] == 512 and config["batch_size"] == 16 and config["dtype"] == "float32", "frozen inference settings changed")
    root = Path(config["output_root"])
    out = root / stage
    out.mkdir(parents=True, exist_ok=False)
    log_root = Path(config["log_root"])
    log_root.mkdir(parents=True, exist_ok=True)
    log_handle = (log_root / f"{stage}.log").open("x")
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = Tee(old_stdout, log_handle), Tee(old_stderr, log_handle)
    try:
        print("START", stage, datetime.now(timezone.utc).isoformat(), "PID", os.getpid(), flush=True)
        frame, registry, sources, models = verify_inputs(config)
        config_hash = sha256(config_path)
        for required_stage in ([] if stage == "preflight" else ["preflight"] if stage == "smoke" else ["preflight", "smoke"]):
            prior = json.loads((root / required_stage / "audit.json").read_text())
            require(prior["status"] == "PASS" and prior["config_sha256"] == config_hash, "required stage absent or changed config")
        base = {"stage": stage, "config_sha256": config_hash, "script_sha256": sha256(__file__),
                "created_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
                "package_versions": {name: importlib.metadata.version(name) for name in ["torch", "transformers", "sentence-transformers", "numpy", "pandas"]},
                "sources": sources, "model_records": models, "network_access": "offline; local_files_only=True",
                "expression_or_effect_data_read": False, "variant": config["variant"],
                "full_text_count": len(frame), "registry_row_count": len(registry), "atomic_count": registry.atomic_id.nunique()}
        outputs = []
        if stage == "preflight":
            from transformers import AutoTokenizer
            token_rows = []
            token_qa = []
            for model in models:
                tokenizer = AutoTokenizer.from_pretrained(model["snapshot_path"], local_files_only=True, trust_remote_code=False)
                lengths = [len(ids) for ids in tokenizer(frame.prompt_text.tolist(), padding=False, truncation=False, add_special_tokens=True)["input_ids"]]
                require(max(lengths) <= config["max_length"], "prompt truncation needs explicit review")
                token_rows.extend({"model_key": model["model_key"], "text_id": uid, "n_tokens": n} for uid, n in zip(frame.text_id, lengths))
                token_qa.append({"model_key": model["model_key"], "min_tokens": min(lengths), "max_tokens": max(lengths), "median_tokens": float(np.median(lengths)), "truncated": 0})
            pd.DataFrame(token_rows).to_csv(out / "token_lengths.tsv", sep="\t", index=False)
            frame.to_csv(out / "selected_texts.tsv", sep="\t", index=False)
            base["token_qa"] = token_qa
            print("TOKENIZER_PASS", token_qa, flush=True)
        else:
            import torch
            require(torch.cuda.is_available(), "CUDA unavailable; no silent CPU substitution")
            torch.set_num_threads(config["cpu_threads"])
            torch.manual_seed(config["seed"])
            np.random.seed(config["seed"])
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            frame = smoke_selection(frame, registry, config["smoke_texts_per_view"]) if stage == "smoke" else frame
            frame.to_csv(out / "encoded_texts.tsv", sep="\t", index=False)
            base.update({"device": config["device"], "gpu_name": torch.cuda.get_device_name(), "torch_cuda": torch.version.cuda,
                         "batch_size": config["batch_size"], "max_length": config["max_length"], "seed": config["seed"], "tf32_enabled": False,
                         "precision_override": "explicit float32 matching legacy inference parameter dtype", "n_encoded_texts": len(frame)})
            for model in models:
                print("ENCODING", model["model_key"], len(frame), "texts", flush=True)
                x, runtime = encode(frame, model, config)
                qa = validate_vectors(x, frame, model["dimension"])
                if stage == "full":
                    smoke_path = root / "smoke" / f'{model["model_key"]}__source_name.npz'
                    smoke_audit = json.loads((root / "smoke/audit.json").read_text())
                    smoke_rec = next(item for item in smoke_audit["outputs"] if item["model_key"] == model["model_key"])
                    require(sha256(smoke_path) == smoke_rec["sha256"], "smoke output hash changed")
                    with np.load(smoke_path, allow_pickle=False) as smoke:
                        rows = pd.Index(frame.text_id).get_indexer(smoke["text_id"])
                        require((rows >= 0).all(), "smoke UID absent in full")
                        difference = float(np.max(np.abs(x[rows] - smoke["X"])))
                        require(difference <= config["smoke_full_absolute_tolerance"], "smoke/full embedding mismatch")
                        qa["smoke_full_max_absolute_difference"] = difference
                target = out / f'{model["model_key"]}__source_name.npz'
                save_vectors(target, frame, x)
                outputs.append({"model_key": model["model_key"], "path": str(target), "sha256": sha256(target),
                                "revision": model["revision"], "rows": x.shape[0], "dimension": x.shape[1], "dtype": str(x.dtype),
                                "qa": qa, "runtime": runtime})
                print("MODEL_PASS", model["model_key"], runtime, flush=True)
                del x
            pd.DataFrame([{key: value for key, value in item.items() if key not in ["runtime", "qa"]} for item in outputs]).to_csv(out / "encoding_manifest.tsv", sep="\t", index=False)
        base.update({"status": "PASS", "seconds": time.perf_counter() - start, "outputs": outputs})
        write_json(out / "audit.json", base)
        print("STAGE_PASS", stage, "seconds", base["seconds"], flush=True)
    except Exception as exc:
        write_json(out / "failure.json", {"status": "FAILED", "error": repr(exc), "stage": stage,
                                          "partial_outputs_must_not_be_used": True})
        raise
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        log_handle.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["preflight", "smoke", "full"], required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    arguments = parser.parse_args()
    run(arguments.stage, arguments.config)
