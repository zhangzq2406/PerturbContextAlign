from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

# Exact R1 offline policy.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch


# ======================================================================
# Frozen R1 encoder contract
# ======================================================================

SEED = 20260711
MAX_LENGTH = 512
ENCODER_BATCH_SIZE = 64
TEXT_CHUNK_SIZE = 2048
AGG_COMPONENT_CHUNK = 4096

MODEL_SPECS = [
    {
        "model_key": "bge_m3",
        "display_name": "BGE-M3",
        "group": "general_purpose",
        "model_id": "BAAI/bge-m3",
        "backend": "sentence_transformers",
        "pooling": "sentence_transformers",
        "trust_remote_code": False,
        "expected_dim": 1024,
    },
    {
        "model_key": "qwen3_0_6b",
        "display_name": "Qwen3-Embedding-0.6B",
        "group": "general_purpose",
        "model_id": "Qwen/Qwen3-Embedding-0.6B",
        "backend": "sentence_transformers",
        "pooling": "sentence_transformers",
        "trust_remote_code": True,
        "expected_dim": 1024,
    },
    {
        "model_key": "sapbert",
        "display_name": "SapBERT",
        "group": "biomedical",
        "model_id": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        "backend": "transformers",
        "pooling": "mean",
        "trust_remote_code": False,
        "expected_dim": 768,
    },
    {
        "model_key": "biomedbert",
        "display_name": "BiomedBERT",
        "group": "biomedical",
        "model_id": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        "backend": "transformers",
        "pooling": "mean",
        "trust_remote_code": False,
        "expected_dim": 768,
    },
    {
        "model_key": "medcpt_query",
        "display_name": "MedCPT Query",
        "group": "biomedical",
        "model_id": "ncbi/MedCPT-Query-Encoder",
        "backend": "transformers",
        "pooling": "cls",
        "trust_remote_code": False,
        "expected_dim": 768,
    },
    {
        "model_key": "medcpt_article",
        "display_name": "MedCPT Article",
        "group": "biomedical",
        "model_id": "ncbi/MedCPT-Article-Encoder",
        "backend": "transformers",
        "pooling": "cls",
        "trust_remote_code": False,
        "expected_dim": 768,
    },
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Encode frozen R2 response-aligned component texts using the exact "
            "six-encoder contract from frozen corrected R1, then aggregate "
            "component embeddings to response-atom embeddings."
        )
    )
    ap.add_argument(
        "--root",
        default=".",
    )
    ap.add_argument(
        "--device",
        default="cuda:0",
        help="PyTorch device after CUDA_VISIBLE_DEVICES remapping.",
    )
    ap.add_argument(
        "--models",
        nargs="*",
        default=[],
        help="Optional subset of model keys. Default: all six.",
    )
    ap.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate inputs and locally load each requested model, but do not encode.",
    )
    return ap.parse_args()


ARGS = parse_args()

ROOT = Path(ARGS.root).expanduser().resolve()

NEW = (
    ROOT
    / "0_ProjectCodeAndLogs"
    / "PerturbContextAlign_R2_corrected_v1_20261002"
)

TEXT_ROOT = NEW / "05_response_aligned_text"

COMPONENT_TSV = (
    TEXT_ROOT
    / "R2_RESPONSE_ALIGNED_COMPONENT_PROMPTS_v1.tsv.gz"
)

TEXT_AUTH_TSV = (
    TEXT_ROOT
    / "R2_RESPONSE_ALIGNED_TEXT_AUTHORITY_v1.tsv"
)

TEXT_CATALOG_TSV = (
    TEXT_ROOT
    / "R2_RESPONSE_ALIGNED_TEXT_CATALOG_v1.tsv.gz"
)

TEXT_MANIFEST_JSON = (
    TEXT_ROOT
    / "R2_RESPONSE_ALIGNED_TEXT_AUTHORITY_MANIFEST_v1.json"
)

ATOM_AUTHORITY_TSV = (
    NEW
    / "02_input_authority"
    / "R2_RESPONSE_ATOM_AUTHORITY_v1.tsv"
)

TEXT_CONTRACT_MD = (
    NEW
    / "01_contract"
    / "R2_RESPONSE_ALIGNED_TEXT_CONTRACT_v1.md"
)

OUT = TEXT_ROOT / "02_embeddings_v1"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_SPEC_TSV = OUT / "R2_TEXT_EMBEDDING_MODEL_SPECS_v1.tsv"
ATOM_INDEX_TSV = OUT / "R2_RESPONSE_ATOM_EMBEDDING_INDEX_v1.tsv"
AUDIT_TSV = OUT / "R2_RESPONSE_ALIGNED_TEXT_EMBEDDING_AUDIT_v1.tsv"
AUDIT_TXT = OUT / "R2_RESPONSE_ALIGNED_TEXT_EMBEDDING_AUDIT_v1.txt"
MANIFEST_JSON = OUT / "R2_RESPONSE_ALIGNED_TEXT_EMBEDDING_MANIFEST_v1.json"


for p in [
    COMPONENT_TSV,
    TEXT_AUTH_TSV,
    TEXT_CATALOG_TSV,
    TEXT_MANIFEST_JSON,
    ATOM_AUTHORITY_TSV,
    TEXT_CONTRACT_MD,
]:
    if not p.is_file():
        raise FileNotFoundError(p)


DEVICE = ARGS.device

if DEVICE.startswith("cuda") and not torch.cuda.is_available():
    raise RuntimeError(
        f"Requested device {DEVICE!r}, but torch.cuda.is_available() is False."
    )


def log(msg: str) -> None:
    print(msg, flush=True)


def step(name: str) -> None:
    log("")
    log("=" * 120)
    log(f"step={name}")
    log("=" * 120)


def sha256_file(path: Path, chunk: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def write_json(obj: Any, path: Path) -> None:
    path.write_text(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def l2_dense(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(~np.isfinite(denom)):
        raise RuntimeError("Non-finite vector norm.")
    if np.any(denom <= 1e-12):
        raise RuntimeError("Zero-norm vector encountered.")
    return x / denom


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def mean_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom


# ======================================================================
# Input authority
# ======================================================================

step("01_input_authority")

text_manifest = json.loads(
    TEXT_MANIFEST_JSON.read_text(encoding="utf-8")
)

if text_manifest.get("status") != "PASS":
    raise RuntimeError(
        "Response-aligned text authority manifest is not PASS."
    )

components = pd.read_csv(
    COMPONENT_TSV,
    sep="\t",
    compression="gzip",
    low_memory=False,
)

text_auth = pd.read_csv(
    TEXT_AUTH_TSV,
    sep="\t",
    low_memory=False,
)

catalog = pd.read_csv(
    TEXT_CATALOG_TSV,
    sep="\t",
    compression="gzip",
    low_memory=False,
)

atoms = pd.read_csv(
    ATOM_AUTHORITY_TSV,
    sep="\t",
    low_memory=False,
)

required_component_cols = {
    "dataset_id",
    "response_atom_id",
    "component_id",
    "response_aligned_prompt",
    "response_aligned_prompt_sha256",
    "text_index",
}
required_catalog_cols = {
    "text_index",
    "text_sha256",
    "prompt_text",
}
required_atom_cols = {
    "dataset_id",
    "response_atom_id",
    "condition_id",
}

missing = required_component_cols - set(components.columns)
if missing:
    raise KeyError(
        f"Component authority missing columns: {sorted(missing)}"
    )

missing = required_catalog_cols - set(catalog.columns)
if missing:
    raise KeyError(
        f"Text catalog missing columns: {sorted(missing)}"
    )

missing = required_atom_cols - set(atoms.columns)
if missing:
    raise KeyError(
        f"Atom authority missing columns: {sorted(missing)}"
    )

components = components.copy()
catalog = catalog.copy()
atoms = atoms.copy()

components["dataset_id"] = components["dataset_id"].astype(str)
components["response_atom_id"] = components["response_atom_id"].astype(str)
components["component_id"] = components["component_id"].astype(str)
components["text_index"] = components["text_index"].astype(np.int64)

catalog["text_index"] = catalog["text_index"].astype(np.int64)
catalog["text_sha256"] = catalog["text_sha256"].astype(str)
catalog["prompt_text"] = catalog["prompt_text"].astype(str)

atoms["dataset_id"] = atoms["dataset_id"].astype(str)
atoms["response_atom_id"] = atoms["response_atom_id"].astype(str)
atoms["condition_id"] = atoms["condition_id"].astype(str)

catalog = catalog.sort_values(
    "text_index"
).reset_index(drop=True)

expected_indices = np.arange(
    len(catalog),
    dtype=np.int64,
)

if not np.array_equal(
    catalog["text_index"].to_numpy(),
    expected_indices,
):
    raise RuntimeError(
        "Text catalog indices are not contiguous 0..N-1."
    )

observed_hashes = catalog["prompt_text"].map(
    sha256_text
)

if not observed_hashes.eq(
    catalog["text_sha256"]
).all():
    n_bad = int(
        (~observed_hashes.eq(catalog["text_sha256"]))
        .sum()
    )
    raise RuntimeError(
        f"Text catalog SHA mismatch rows={n_bad}."
    )

max_component_text_index = int(
    components["text_index"].max()
) if len(components) else -1

if max_component_text_index >= len(catalog):
    raise RuntimeError(
        "Component text_index exceeds catalog size."
    )

component_catalog_text = catalog.set_index(
    "text_index"
)["prompt_text"]

mapped_text = components["text_index"].map(
    component_catalog_text
)

if not mapped_text.eq(
    components["response_aligned_prompt"]
).all():
    raise RuntimeError(
        "Component prompt text does not reproduce from text catalog."
    )

atom_ids = atoms[
    "response_atom_id"
].astype(str).tolist()

if len(atom_ids) != len(set(atom_ids)):
    raise RuntimeError(
        "Duplicate response_atom_id in atom authority."
    )

component_atom_ids = set(
    components["response_atom_id"].astype(str)
)

authority_atom_ids = set(atom_ids)

if component_atom_ids != authority_atom_ids:
    raise RuntimeError(
        "Component/atom namespace mismatch: "
        f"missing={len(authority_atom_ids-component_atom_ids)} "
        f"extra={len(component_atom_ids-authority_atom_ids)}"
    )

if len(text_auth) != len(atoms):
    raise RuntimeError(
        f"Text authority atom count {len(text_auth)} "
        f"!= atom authority count {len(atoms)}"
    )

logical_catalog_payload = "\n".join(
    f"{int(i)}\t{h}"
    for i, h in zip(
        catalog["text_index"],
        catalog["text_sha256"],
    )
)

logical_text_catalog_sha = sha256_text(
    logical_catalog_payload
)

log(
    f"text_n={len(catalog)} "
    f"component_row_n={len(components)} "
    f"response_atom_n={len(atoms)}"
)
log(
    f"text_catalog_logical_sha256="
    f"{logical_text_catalog_sha}"
)


# ======================================================================
# Exact model registry
# ======================================================================

requested = set(ARGS.models)

if requested:
    known = {x["model_key"] for x in MODEL_SPECS}
    unknown = requested - known
    if unknown:
        raise ValueError(
            f"Unknown model keys: {sorted(unknown)}"
        )
    model_specs = [
        x for x in MODEL_SPECS
        if x["model_key"] in requested
    ]
else:
    model_specs = list(MODEL_SPECS)

pd.DataFrame(
    [
        {
            **spec,
            "max_length": MAX_LENGTH,
            "encoder_batch_size": ENCODER_BATCH_SIZE,
            "text_chunk_size": TEXT_CHUNK_SIZE,
            "component_aggregation_chunk": AGG_COMPONENT_CHUNK,
            "normalized": True,
            "source_contract": "frozen_corrected_R1_r1_full_pipeline.py",
        }
        for spec in MODEL_SPECS
    ]
).to_csv(
    MODEL_SPEC_TSV,
    sep="\t",
    index=False,
)


# ======================================================================
# Encoder implementation — exact R1 logic
# ======================================================================

class EncoderRunner:
    def __init__(
        self,
        spec: dict[str, Any],
    ):
        self.spec = spec
        self.model = None
        self.tokenizer = None
        self.dim = int(
            spec["expected_dim"]
        )

        if spec["backend"] == "sentence_transformers":
            from sentence_transformers import SentenceTransformer

            kwargs = {
                "device": DEVICE,
                "trust_remote_code": spec["trust_remote_code"],
            }

            try:
                self.model = SentenceTransformer(
                    spec["model_id"],
                    local_files_only=True,
                    **kwargs,
                )
            except TypeError:
                # Exact R1 compatibility fallback.
                self.model = SentenceTransformer(
                    spec["model_id"],
                    **kwargs,
                )

            if hasattr(
                self.model,
                "max_seq_length",
            ):
                self.model.max_seq_length = MAX_LENGTH

            try:
                observed_dim = int(
                    self.model
                    .get_sentence_embedding_dimension()
                )
            except Exception:
                observed_dim = self.dim

            if observed_dim != self.dim:
                raise RuntimeError(
                    f"{spec['model_key']}: dim "
                    f"{observed_dim} != expected {self.dim}"
                )

        elif spec["backend"] == "transformers":
            from transformers import (
                AutoModel,
                AutoTokenizer,
            )

            self.tokenizer = (
                AutoTokenizer.from_pretrained(
                    spec["model_id"],
                    local_files_only=True,
                    trust_remote_code=spec["trust_remote_code"],
                )
            )

            self.model = (
                AutoModel.from_pretrained(
                    spec["model_id"],
                    local_files_only=True,
                    trust_remote_code=spec["trust_remote_code"],
                )
                .to(DEVICE)
            )

            self.model.eval()

            observed_dim = int(
                getattr(
                    self.model.config,
                    "hidden_size",
                    self.dim,
                )
            )

            if observed_dim != self.dim:
                raise RuntimeError(
                    f"{spec['model_key']}: dim "
                    f"{observed_dim} != expected {self.dim}"
                )

        else:
            raise ValueError(
                spec["backend"]
            )

    def encode(
        self,
        texts: list[str],
    ) -> np.ndarray:
        spec = self.spec

        if spec["backend"] == "sentence_transformers":
            z = self.model.encode(
                texts,
                batch_size=ENCODER_BATCH_SIZE,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

            return np.asarray(
                z,
                dtype=np.float32,
            )

        chunks = []

        for start in range(
            0,
            len(texts),
            ENCODER_BATCH_SIZE,
        ):
            local = texts[
                start : start + ENCODER_BATCH_SIZE
            ]

            batch = self.tokenizer(
                local,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )

            batch = {
                k: v.to(DEVICE)
                for k, v in batch.items()
            }

            with torch.inference_mode():
                model_out = self.model(
                    **batch
                )

                if spec["pooling"] == "mean":
                    z = mean_pool(
                        model_out.last_hidden_state,
                        batch["attention_mask"],
                    )

                elif spec["pooling"] == "cls":
                    z = (
                        model_out
                        .last_hidden_state[:, 0]
                    )

                else:
                    raise ValueError(
                        spec["pooling"]
                    )

            chunks.append(
                z.float().cpu().numpy()
            )

        return l2_dense(
            np.concatenate(
                chunks,
                axis=0,
            )
        )

    def close(self) -> None:
        try:
            del self.model
        except Exception:
            pass

        try:
            del self.tokenizer
        except Exception:
            pass

        self.model = None
        self.tokenizer = None

        cleanup_cuda()


# ======================================================================
# Optional preflight
# ======================================================================

if ARGS.preflight_only:
    step("02_model_preflight")

    for spec in model_specs:
        key = spec["model_key"]
        log(f"[preflight] loading {key}")
        runner = EncoderRunner(spec)
        try:
            sample = runner.encode(
                catalog["prompt_text"]
                .iloc[: min(2, len(catalog))]
                .tolist()
            )

            expected = (
                min(2, len(catalog)),
                int(spec["expected_dim"]),
            )

            if sample.shape != expected:
                raise RuntimeError(
                    f"{key}: preflight shape "
                    f"{sample.shape} != {expected}"
                )

            if not np.isfinite(
                sample
            ).all():
                raise RuntimeError(
                    f"{key}: nonfinite preflight embedding"
                )

            norms = np.linalg.norm(
                sample,
                axis=1,
            )

            if float(
                np.max(
                    np.abs(
                        norms - 1.0
                    )
                )
            ) > 5e-4:
                raise RuntimeError(
                    f"{key}: preflight norm failure"
                )

            log(
                f"[preflight] PASS {key} "
                f"shape={sample.shape}"
            )

        finally:
            runner.close()

    log("R2_TEXT_EMBEDDING_PREFLIGHT=PASS")
    raise SystemExit(0)


# ======================================================================
# Encode unique response-aligned texts
# ======================================================================

all_texts = catalog[
    "prompt_text"
].astype(str).tolist()


def encode_text_cache(
    spec: dict[str, Any],
) -> Path:
    key = spec["model_key"]

    out_dir = OUT / key
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        out_dir
        / "text_embeddings.npy"
    )

    progress_path = (
        out_dir
        / "text_progress.json"
    )

    done_path = (
        out_dir
        / "text_DONE.json"
    )

    expected_shape = (
        len(all_texts),
        int(spec["expected_dim"]),
    )

    if (
        done_path.is_file()
        and path.is_file()
    ):
        done = json.loads(
            done_path.read_text(
                encoding="utf-8"
            )
        )

        try:
            arr = np.load(
                path,
                mmap_mode="r",
            )
            shape_ok = (
                tuple(arr.shape)
                == expected_shape
            )
        except Exception:
            shape_ok = False

        if (
            shape_ok
            and done.get(
                "text_catalog_logical_sha256"
            )
            == logical_text_catalog_sha
            and done.get(
                "model_id"
            )
            == spec["model_id"]
            and done.get(
                "status"
            )
            == "PASS"
        ):
            log(
                f"[resume] {key} "
                f"text embeddings already DONE"
            )
            return path

    next_index = 0
    reuse = False

    if (
        path.is_file()
        and progress_path.is_file()
    ):
        progress = json.loads(
            progress_path.read_text(
                encoding="utf-8"
            )
        )

        try:
            arr = np.load(
                path,
                mmap_mode="r",
            )
            shape_ok = (
                tuple(arr.shape)
                == expected_shape
            )
        except Exception:
            shape_ok = False

        reuse = bool(
            shape_ok
            and progress.get(
                "text_catalog_logical_sha256"
            )
            == logical_text_catalog_sha
            and progress.get(
                "model_id"
            )
            == spec["model_id"]
        )

        if reuse:
            next_index = int(
                progress.get(
                    "next_index",
                    0,
                )
            )

            if not (
                0
                <= next_index
                <= len(all_texts)
            ):
                raise RuntimeError(
                    f"{key}: invalid resume index "
                    f"{next_index}"
                )

            log(
                f"[resume] {key} "
                f"from text index {next_index}"
            )

    if not reuse:
        path.unlink(
            missing_ok=True
        )

        progress_path.unlink(
            missing_ok=True
        )

        arr = (
            np.lib.format.open_memmap(
                path,
                mode="w+",
                dtype="float32",
                shape=expected_shape,
            )
        )

        arr.flush()
        del arr

        next_index = 0

        write_json(
            {
                "model_key": key,
                "model_id": spec["model_id"],
                "text_catalog_logical_sha256":
                    logical_text_catalog_sha,
                "next_index": 0,
                "shape": list(
                    expected_shape
                ),
            },
            progress_path,
        )

    runner = EncoderRunner(spec)

    try:
        arr = np.load(
            path,
            mmap_mode="r+",
        )

        for start in range(
            next_index,
            len(all_texts),
            TEXT_CHUNK_SIZE,
        ):
            stop = min(
                len(all_texts),
                start + TEXT_CHUNK_SIZE,
            )

            t0 = time.time()

            z = runner.encode(
                all_texts[start:stop]
            )

            expected_local = (
                stop - start,
                int(spec["expected_dim"]),
            )

            if z.shape != expected_local:
                raise RuntimeError(
                    f"{key}: encode shape "
                    f"{z.shape} != "
                    f"{expected_local}"
                )

            if not np.isfinite(
                z
            ).all():
                raise RuntimeError(
                    f"{key}: nonfinite "
                    f"text embeddings"
                )

            norms = np.linalg.norm(
                z,
                axis=1,
            )

            max_dev = float(
                np.max(
                    np.abs(
                        norms - 1.0
                    )
                )
            )

            if max_dev > 5e-4:
                raise RuntimeError(
                    f"{key}: text embedding "
                    f"norm failure "
                    f"max_dev={max_dev}"
                )

            arr[start:stop] = z
            arr.flush()

            write_json(
                {
                    "model_key": key,
                    "model_id": spec["model_id"],
                    "text_catalog_logical_sha256":
                        logical_text_catalog_sha,
                    "next_index": stop,
                    "shape": list(
                        expected_shape
                    ),
                },
                progress_path,
            )

            log(
                f"[{key}] texts "
                f"{stop}/{len(all_texts)} "
                f"chunk_seconds="
                f"{time.time() - t0:.1f}"
            )

        del arr

    finally:
        runner.close()

    write_json(
        {
            "status": "PASS",
            "model_key": key,
            "model_id": spec["model_id"],
            "backend": spec["backend"],
            "pooling": spec["pooling"],
            "dimension": int(
                spec["expected_dim"]
            ),
            "normalize": True,
            "max_length": MAX_LENGTH,
            "encoder_batch_size":
                ENCODER_BATCH_SIZE,
            "text_n": len(all_texts),
            "text_catalog_logical_sha256":
                logical_text_catalog_sha,
        },
        done_path,
    )

    return path


# ======================================================================
# Frozen atom index
# ======================================================================

atom_index = atoms[
    [
        c
        for c in [
            "dataset_id",
            "response_atom_id",
            "condition_id",
            "support_status_pre_nuisance",
            "species",
            "cell_type",
            "cell_line",
            "tissue",
            "disease",
        ]
        if c in atoms.columns
    ]
].copy()

atom_index.insert(
    0,
    "row_index",
    np.arange(
        len(atom_index),
        dtype=np.int64,
    ),
)

atom_index.to_csv(
    ATOM_INDEX_TSV,
    sep="\t",
    index=False,
)

atom_to_row = {
    rid: i
    for i, rid in enumerate(
        atom_index[
            "response_atom_id"
        ].astype(str)
    )
}

component_rows = (
    components[
        "response_atom_id"
    ]
    .astype(str)
    .map(atom_to_row)
)

if component_rows.isna().any():
    raise RuntimeError(
        "Component row failed to map "
        "to response-atom index."
    )

component_rows = (
    component_rows
    .to_numpy(
        dtype=np.int64
    )
)

component_text_indices = (
    components[
        "text_index"
    ]
    .to_numpy(
        dtype=np.int64
    )
)


# ======================================================================
# Aggregate component embeddings to response atoms
# ======================================================================

def aggregate_response_atoms(
    text_embedding_path: Path,
    spec: dict[str, Any],
) -> Path:
    key = spec["model_key"]
    dim = int(spec["expected_dim"])

    out_dir = OUT / key
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        out_dir
        / "response_atom_embeddings.npy"
    )

    done_path = (
        out_dir
        / "aggregate_DONE.json"
    )

    expected_shape = (
        len(atom_index),
        dim,
    )

    if (
        done_path.is_file()
        and path.is_file()
    ):
        done = json.loads(
            done_path.read_text(
                encoding="utf-8"
            )
        )

        try:
            arr = np.load(
                path,
                mmap_mode="r",
            )
            shape_ok = (
                tuple(arr.shape)
                == expected_shape
            )
        except Exception:
            shape_ok = False

        if (
            shape_ok
            and done.get(
                "status"
            )
            == "PASS"
            and done.get(
                "text_catalog_logical_sha256"
            )
            == logical_text_catalog_sha
            and done.get(
                "response_atom_authority_sha256"
            )
            == sha256_file(
                ATOM_AUTHORITY_TSV
            )
        ):
            log(
                f"[resume] {key} "
                f"response-atom aggregation "
                f"already DONE"
            )

            return path

    text_emb = np.load(
        text_embedding_path,
        mmap_mode="r",
    )

    if tuple(
        text_emb.shape
    ) != (
        len(catalog),
        dim,
    ):
        raise RuntimeError(
            f"{key}: text embedding "
            f"shape mismatch "
            f"{text_emb.shape}"
        )

    path.unlink(
        missing_ok=True
    )

    out_arr = (
        np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype="float32",
            shape=expected_shape,
        )
    )

    out_arr[:] = 0
    out_arr.flush()

    counts = np.zeros(
        len(atom_index),
        dtype=np.int32,
    )

    for start in range(
        0,
        len(components),
        AGG_COMPONENT_CHUNK,
    ):
        stop = min(
            len(components),
            start + AGG_COMPONENT_CHUNK,
        )

        rows = component_rows[
            start:stop
        ]

        tids = component_text_indices[
            start:stop
        ]

        vecs = np.asarray(
            text_emb[tids],
            dtype=np.float32,
        )

        np.add.at(
            out_arr,
            rows,
            vecs,
        )

        np.add.at(
            counts,
            rows,
            1,
        )

    if np.any(
        counts <= 0
    ):
        bad = np.flatnonzero(
            counts <= 0
        )

        raise RuntimeError(
            f"{key}: {len(bad)} response "
            f"atoms without components."
        )

    for start in range(
        0,
        len(atom_index),
        AGG_COMPONENT_CHUNK,
    ):
        stop = min(
            len(atom_index),
            start + AGG_COMPONENT_CHUNK,
        )

        block = np.asarray(
            out_arr[start:stop],
            dtype=np.float32,
        )

        block = (
            block
            / counts[
                start:stop,
                None,
            ].astype(
                np.float32
            )
        )

        block = l2_dense(
            block
        )

        out_arr[
            start:stop
        ] = block

    out_arr.flush()

    # Final QC before marking DONE.
    norm_min = float("inf")
    norm_max = float("-inf")
    finite_all = True

    for start in range(
        0,
        len(atom_index),
        AGG_COMPONENT_CHUNK,
    ):
        stop = min(
            len(atom_index),
            start + AGG_COMPONENT_CHUNK,
        )

        block = np.asarray(
            out_arr[start:stop],
            dtype=np.float32,
        )

        finite_all = (
            finite_all
            and bool(
                np.isfinite(
                    block
                ).all()
            )
        )

        norms = np.linalg.norm(
            block,
            axis=1,
        )

        norm_min = min(
            norm_min,
            float(
                norms.min()
            ),
        )

        norm_max = max(
            norm_max,
            float(
                norms.max()
            ),
        )

    del out_arr

    if not finite_all:
        raise RuntimeError(
            f"{key}: nonfinite response "
            f"atom embeddings."
        )

    max_norm_dev = max(
        abs(
            norm_min - 1.0
        ),
        abs(
            norm_max - 1.0
        ),
    )

    if max_norm_dev > 5e-4:
        raise RuntimeError(
            f"{key}: response-atom "
            f"norm failure "
            f"min={norm_min} max={norm_max}"
        )

    write_json(
        {
            "status": "PASS",
            "model_key": key,
            "model_id": spec["model_id"],
            "dimension": dim,
            "response_atom_n": len(
                atom_index
            ),
            "component_row_n": len(
                components
            ),
            "pooling":
                "masked_mean_over_component_embeddings_then_l2",
            "norm_min": norm_min,
            "norm_max": norm_max,
            "text_catalog_logical_sha256":
                logical_text_catalog_sha,
            "response_atom_authority_sha256":
                sha256_file(
                    ATOM_AUTHORITY_TSV
                ),
        },
        done_path,
    )

    return path


# ======================================================================
# Run models sequentially
# ======================================================================

step("02_six_language_encoders")

audit_rows = []

for spec in model_specs:
    key = spec["model_key"]

    log(
        f"model_start={key} "
        f"device={DEVICE}"
    )

    t0 = time.time()

    text_path = encode_text_cache(
        spec
    )

    atom_path = (
        aggregate_response_atoms(
            text_path,
            spec,
        )
    )

    text_arr = np.load(
        text_path,
        mmap_mode="r",
    )

    atom_arr = np.load(
        atom_path,
        mmap_mode="r",
    )

    atom_norm_min = float("inf")
    atom_norm_max = float("-inf")
    atom_finite = True

    for start in range(
        0,
        len(atom_arr),
        AGG_COMPONENT_CHUNK,
    ):
        stop = min(
            len(atom_arr),
            start + AGG_COMPONENT_CHUNK,
        )

        block = np.asarray(
            atom_arr[start:stop],
            dtype=np.float32,
        )

        atom_finite = (
            atom_finite
            and bool(
                np.isfinite(
                    block
                ).all()
            )
        )

        norms = np.linalg.norm(
            block,
            axis=1,
        )

        atom_norm_min = min(
            atom_norm_min,
            float(
                norms.min()
            ),
        )

        atom_norm_max = max(
            atom_norm_max,
            float(
                norms.max()
            ),
        )

    audit_rows.append(
        {
            "model_key": key,
            "display_name":
                spec["display_name"],
            "group":
                spec["group"],
            "model_id":
                spec["model_id"],
            "backend":
                spec["backend"],
            "pooling":
                spec["pooling"],
            "dimension":
                int(
                    spec["expected_dim"]
                ),
            "text_n":
                int(
                    text_arr.shape[0]
                ),
            "response_atom_n":
                int(
                    atom_arr.shape[0]
                ),
            "text_shape":
                json.dumps(
                    list(
                        text_arr.shape
                    )
                ),
            "response_atom_shape":
                json.dumps(
                    list(
                        atom_arr.shape
                    )
                ),
            "atom_finite":
                atom_finite,
            "atom_norm_min":
                atom_norm_min,
            "atom_norm_max":
                atom_norm_max,
            "text_done":
                (
                    OUT
                    / key
                    / "text_DONE.json"
                ).is_file(),
            "aggregate_done":
                (
                    OUT
                    / key
                    / "aggregate_DONE.json"
                ).is_file(),
            "elapsed_seconds":
                time.time() - t0,
            "status":
                (
                    "PASS"
                    if (
                        atom_finite
                        and abs(
                            atom_norm_min
                            - 1.0
                        )
                        <= 5e-4
                        and abs(
                            atom_norm_max
                            - 1.0
                        )
                        <= 5e-4
                    )
                    else "FAIL"
                ),
        }
    )

    del text_arr
    del atom_arr

    log(
        f"model_done={key} "
        f"elapsed_seconds="
        f"{time.time() - t0:.1f}"
    )

    cleanup_cuda()


# ======================================================================
# Final audit
# ======================================================================

step("03_final_audit")

audit = pd.DataFrame(
    audit_rows
)

audit.to_csv(
    AUDIT_TSV,
    sep="\t",
    index=False,
)

all_requested_pass = bool(
    len(audit)
    and audit[
        "status"
    ].eq(
        "PASS"
    ).all()
)

expected_requested_n = len(
    model_specs
)

if len(audit) != expected_requested_n:
    all_requested_pass = False

status = (
    "PASS"
    if all_requested_pass
    else "FAIL"
)

lines = [
    "PERTURBCONTEXTALIGN R2 RESPONSE-ALIGNED TEXT EMBEDDING AUDIT v1",
    "=" * 120,
    "",
    "MODE",
    "-" * 120,
    "source_text_authority=FROZEN",
    "encoder_contract=frozen_corrected_R1_exact",
    "expression_read=FALSE",
    "measured_response_values_read=FALSE",
    "offline_model_loading=TRUE",
    f"device={DEVICE}",
    f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '')}",
    "",
    "CONTRACT",
    "-" * 120,
    f"max_length={MAX_LENGTH}",
    f"encoder_batch_size={ENCODER_BATCH_SIZE}",
    f"text_chunk_size={TEXT_CHUNK_SIZE}",
    f"component_aggregation_chunk={AGG_COMPONENT_CHUNK}",
    "component_pooling=masked_mean_over_component_embeddings_then_l2",
    "all_text_embeddings_l2_normalized=TRUE",
    "all_response_atom_embeddings_l2_normalized=TRUE",
    "",
    "INPUT",
    "-" * 120,
    f"unique_response_aligned_text_n={len(catalog)}",
    f"response_aligned_component_row_n={len(components)}",
    f"response_atom_n={len(atoms)}",
    f"text_catalog_logical_sha256={logical_text_catalog_sha}",
    "",
    "MODEL AUDIT",
    "-" * 120,
    audit.to_string(index=False),
    "",
    "ENVIRONMENT",
    "-" * 120,
    f"python={sys.version.split()[0]}",
    f"platform={platform.platform()}",
    f"torch={torch.__version__}",
    (
        f"cuda_device_name="
        f"{torch.cuda.get_device_name(0)}"
        if torch.cuda.is_available()
        else "cuda_device_name=NA"
    ),
    "",
    "STATUS",
    "-" * 120,
    f"R2_RESPONSE_ALIGNED_TEXT_EMBEDDING={status}",
]

if status == "PASS":
    lines += [
        "NEXT=R2_MEASURED_RESPONSE_EFFECT_CONSTRUCTION",
    ]
else:
    lines += [
        "NEXT=REVIEW_FAILED_ENCODER_BEFORE_EFFECT_CONSTRUCTION",
    ]

AUDIT_TXT.write_text(
    "\n".join(
        lines
    )
    + "\n",
    encoding="utf-8",
)

manifest = {
    "version":
        "R2_RESPONSE_ALIGNED_TEXT_EMBEDDING_v1",
    "status": status,
    "source_contract":
        "frozen corrected R1 r1_full_pipeline.py",
    "encoder_parameters": {
        "seed": SEED,
        "max_length": MAX_LENGTH,
        "encoder_batch_size":
            ENCODER_BATCH_SIZE,
        "text_chunk_size":
            TEXT_CHUNK_SIZE,
        "aggregation_chunk":
            AGG_COMPONENT_CHUNK,
        "device": DEVICE,
        "normalized": True,
        "component_pooling":
            "masked_mean_over_component_embeddings_then_l2",
    },
    "models": MODEL_SPECS,
    "requested_models": [
        x["model_key"]
        for x in model_specs
    ],
    "inputs": {
        "component_authority": {
            "path":
                str(
                    COMPONENT_TSV
                ),
            "sha256":
                sha256_file(
                    COMPONENT_TSV
                ),
        },
        "text_authority": {
            "path":
                str(
                    TEXT_AUTH_TSV
                ),
            "sha256":
                sha256_file(
                    TEXT_AUTH_TSV
                ),
        },
        "text_catalog": {
            "path":
                str(
                    TEXT_CATALOG_TSV
                ),
            "sha256":
                sha256_file(
                    TEXT_CATALOG_TSV
                ),
            "logical_sha256":
                logical_text_catalog_sha,
        },
        "response_atom_authority": {
            "path":
                str(
                    ATOM_AUTHORITY_TSV
                ),
            "sha256":
                sha256_file(
                    ATOM_AUTHORITY_TSV
                ),
        },
    },
    "outputs": {
        "root": str(OUT),
        "model_specs":
            str(
                MODEL_SPEC_TSV
            ),
        "atom_index":
            str(
                ATOM_INDEX_TSV
            ),
        "audit_tsv":
            str(
                AUDIT_TSV
            ),
        "audit_txt":
            str(
                AUDIT_TXT
            ),
    },
}

write_json(
    manifest,
    MANIFEST_JSON,
)

log(
    AUDIT_TXT.read_text(
        encoding="utf-8"
    )
)

if status != "PASS":
    raise SystemExit(2)
