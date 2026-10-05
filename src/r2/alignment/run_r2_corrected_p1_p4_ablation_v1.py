from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata


# ======================================================================================
# CLI / constants
# ======================================================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Corrected PerturbContextAlign R2 P1-P4 ablation using the frozen "
            "response-atom namespace and measured-response geometry."
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
        help="Optional subset of model keys. Default: all six frozen encoders.",
    )
    ap.add_argument(
        "--preflight-only",
        action="store_true",
    )
    return ap.parse_args()


ARGS = parse_args()

ROOT = Path(ARGS.root).expanduser().resolve()

NEW = (
    ROOT
    / "0_ProjectCodeAndLogs"
    / "PerturbContextAlign_R2_corrected_v1_20261002"
)

DEVICE = ARGS.device
SEED = 20260711
MAX_LENGTH = 512
ENCODER_BATCH_SIZE = 64
TEXT_CHUNK_SIZE = 2048
AGG_CHUNK = 4096
BOOTSTRAP_REPEATS = 10000

VIEWS = ["P1", "P2", "P3", "P4"]

DELTA_DEFS = [
    ("P1_to_P2_add_biology", "P1", "P2"),
    ("P1_to_P3_add_exposure", "P1", "P3"),
    ("P2_to_P4_add_exposure_after_biology", "P2", "P4"),
    ("P3_to_P4_add_biology_after_exposure", "P3", "P4"),
]

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

DATASET_ORDER = [
    "norman_2019",
    "replogle_k562_essential",
    "replogle_rpe1",
    "tian_activation",
    "tian_inhibition",
    "srivatsan_sciplex3",
    "mcfarland_2020",
    "kaggle_cross_patient",
    "combo_sciplex",
]


# ======================================================================================
# Paths
# ======================================================================================

CLOSURE_ROOT = NEW / "06_alignment_metrics" / "02_paper_closure_v2"

VIEW_COMPONENTS = (
    CLOSURE_ROOT
    / "R2_RESPONSE_ALIGNED_P1P4_COMPONENT_TEXTS_v2.tsv.gz"
)

VIEW_TEXT_CATALOG = (
    CLOSURE_ROOT
    / "R2_RESPONSE_ALIGNED_P1P4_TEXT_CATALOG_v2.tsv.gz"
)

ATOM_AUTHORITY = (
    NEW
    / "02_input_authority"
    / "R2_RESPONSE_ATOM_AUTHORITY_v1.tsv"
)

PRIMARY_EMB_ROOT = (
    NEW
    / "05_response_aligned_text"
    / "02_embeddings_v1"
)

PRIMARY_EMB_INDEX = (
    PRIMARY_EMB_ROOT
    / "R2_RESPONSE_ATOM_EMBEDDING_INDEX_v1.tsv"
)

RESP_ROOT = (
    NEW
    / "04_response_geometry"
    / "02_similarity_v1"
)

RESP_MANIFEST = RESP_ROOT / "R2_RESPONSE_SIMILARITY_MANIFEST_v1.json"

HIST_CONFIG = (
    NEW
    / "01_contract"
    / "R2_HISTORICAL_EFFECTIVE_CONFIG_v1.json"
)

OUT = (
    NEW
    / "06_alignment_metrics"
    / "03_p1_p4_ablation_v1"
)

OUT.mkdir(parents=True, exist_ok=True)

EMB_ROOT = OUT / "embeddings"
EMB_ROOT.mkdir(parents=True, exist_ok=True)

BY_DATASET_TSV = OUT / "R2_P1P4_ALIGNMENT_BY_DATASET_v1.tsv"
DELTA_BY_DATASET_TSV = OUT / "R2_P1P4_DELTA_BY_DATASET_v1.tsv"
DELTA_SUMMARY_TSV = OUT / "R2_P1P4_DELTA_SUMMARY_v1.tsv"
ABSOLUTE_SUMMARY_TSV = OUT / "R2_P1P4_ABSOLUTE_SUMMARY_v1.tsv"
P4_REPRO_TSV = OUT / "R2_P4_EMBEDDING_REPRODUCTION_v1.tsv"
AUDIT_TXT = OUT / "R2_CORRECTED_P1_P4_ABLATION_AUDIT_v1.txt"
MANIFEST_JSON = OUT / "R2_CORRECTED_P1_P4_ABLATION_MANIFEST_v1.json"


# ======================================================================================
# Helpers
# ======================================================================================

def log(msg: str) -> None:
    print(msg, flush=True)


def sha256_file(path: Path, chunk: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def stable_seed(*parts: Any) -> int:
    raw = "|".join(map(str, parts)).encode("utf-8")
    digest = hashlib.sha256(raw).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


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
    a = np.asarray(x, dtype=np.float32)
    denom = np.linalg.norm(a, axis=1, keepdims=True)

    if np.any(~np.isfinite(denom)):
        raise RuntimeError("Non-finite vector norm.")

    if np.any(denom <= 1e-12):
        raise RuntimeError("Zero-norm vector.")

    return a / denom


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


def load_similarity_bundle(path: Path):
    with np.load(path, allow_pickle=True) as z:
        key = "S" if "S" in z.files else "X"
        S = np.asarray(z[key], dtype=np.float32)

        id_key = (
            "response_atom_ids"
            if "response_atom_ids" in z.files
            else "condition_ids"
        )

        ids = np.asarray(z[id_key]).astype(str)

    return S, ids


def upper_vec(S: np.ndarray) -> np.ndarray:
    A = np.asarray(S, dtype=np.float64)
    idx = np.triu_indices(A.shape[0], k=1)
    return A[idx]


def safe_corr(
    x: np.ndarray,
    y: np.ndarray,
    method: str = "spearman",
) -> tuple[float, str]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]

    if len(x) < 3:
        return np.nan, "invalid_too_few_pairs"

    if np.ptp(x) <= 1e-12:
        return np.nan, "invalid_constant_text"

    if np.ptp(y) <= 1e-12:
        return np.nan, "invalid_constant_response"

    if method == "spearman":
        x = rankdata(x)
        y = rankdata(y)
    elif method != "pearson":
        raise ValueError(method)

    x -= x.mean()
    y -= y.mean()

    denom = math.sqrt(
        float(np.dot(x, x))
        * float(np.dot(y, y))
    )

    if denom <= 0 or not np.isfinite(denom):
        return np.nan, "invalid_zero_variance"

    return float(np.dot(x, y) / denom), "ok"


def random_tie_ranking(
    S: np.ndarray,
    i: int,
    rng: np.random.Generator,
) -> np.ndarray:
    row = np.asarray(S[i], dtype=np.float64).copy()
    row[~np.isfinite(row)] = -np.inf
    row[i] = -np.inf

    order = np.lexsort(
        (
            rng.random(len(row)),
            -row,
        )
    )

    return order[order != i]


def retrieval_metrics(
    text_S: np.ndarray,
    response_S: np.ndarray,
    gold_k: int,
    tie_repeats: int,
    seed: int,
) -> dict[str, Any]:
    n = int(text_S.shape[0])

    if text_S.shape != response_S.shape:
        raise ValueError("Matrix shape mismatch.")

    rows = []

    for repeat in range(max(1, int(tie_repeats))):
        response_rng = np.random.default_rng(int(seed) + repeat)
        text_rng = np.random.default_rng(int(seed) + 1000003 + repeat)

        for i in range(n):
            response_rank = random_tie_ranking(
                response_S,
                i,
                response_rng,
            )

            text_rank = random_tie_ranking(
                text_S,
                i,
                text_rng,
            )

            g = min(int(gold_k), n - 1)
            gold = set(response_rank[:g].tolist())

            if not gold:
                continue

            row = {}

            for k in [5, 10, 20]:
                kk = min(k, n - 1)

                row[f"recall_at_{k}"] = (
                    len(
                        gold.intersection(
                            text_rank[:kk]
                        )
                    )
                    / len(gold)
                )

            for k in [10, 20]:
                kk = min(k, n - 1)

                rel = np.asarray(
                    [j in gold for j in text_rank[:kk]],
                    dtype=np.float64,
                )

                discounts = (
                    1.0
                    / np.log2(
                        np.arange(
                            2,
                            kk + 2,
                            dtype=np.float64,
                        )
                    )
                )

                dcg = float(np.dot(rel, discounts))
                ideal = min(len(gold), kk)

                idcg = float(
                    np.sum(
                        1.0
                        / np.log2(
                            np.arange(
                                2,
                                ideal + 2,
                                dtype=np.float64,
                            )
                        )
                    )
                )

                row[f"ndcg_at_{k}"] = (
                    dcg / idcg
                    if idcg > 0
                    else np.nan
                )

            rows.append(row)

    frame = pd.DataFrame(rows)

    if frame.empty:
        return {
            "status": "invalid_no_queries",
            "n_queries": 0,
        }

    out = {
        col: float(frame[col].mean())
        for col in frame.columns
    }

    out.update({
        "status": "ok",
        "n_queries": n,
        "tie_repeats": int(tie_repeats),
    })

    return out


def exact_random_ndcg_expectation(
    n_atoms: int,
    gold_k: int,
    eval_k: int,
) -> float:
    candidates = int(n_atoms) - 1

    if candidates <= 0:
        return np.nan

    g = min(int(gold_k), candidates)
    k = min(int(eval_k), candidates)

    if g <= 0 or k <= 0:
        return np.nan

    discounts = (
        1.0
        / np.log2(
            np.arange(
                2,
                k + 2,
                dtype=np.float64,
            )
        )
    )

    expected_dcg = (
        float(g)
        / float(candidates)
        * float(discounts.sum())
    )

    ideal = min(g, k)

    idcg = float(
        np.sum(
            1.0
            / np.log2(
                np.arange(
                    2,
                    ideal + 2,
                    dtype=np.float64,
                )
            )
        )
    )

    return expected_dcg / idcg


def bootstrap_ci(
    values: np.ndarray,
    seed: int,
    repeats: int = BOOTSTRAP_REPEATS,
) -> tuple[float, float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan, np.nan

    if len(x) == 1:
        return float(x[0]), float(x[0])

    rng = np.random.default_rng(seed)

    idx = rng.integers(
        0,
        len(x),
        size=(int(repeats), len(x)),
    )

    means = x[idx].mean(axis=1)

    return (
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


# ======================================================================================
# Encoder
# ======================================================================================

class EncoderRunner:
    def __init__(self, spec: dict[str, Any]):
        self.spec = spec
        self.model = None
        self.tokenizer = None
        self.dim = int(spec["expected_dim"])

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
                self.model = SentenceTransformer(
                    spec["model_id"],
                    **kwargs,
                )

            if hasattr(self.model, "max_seq_length"):
                self.model.max_seq_length = MAX_LENGTH

        elif spec["backend"] == "transformers":
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                spec["model_id"],
                local_files_only=True,
                trust_remote_code=spec["trust_remote_code"],
            )

            self.model = AutoModel.from_pretrained(
                spec["model_id"],
                local_files_only=True,
                trust_remote_code=spec["trust_remote_code"],
            ).to(DEVICE)

            self.model.eval()

        else:
            raise ValueError(spec["backend"])

    def encode(self, texts: list[str]) -> np.ndarray:
        spec = self.spec

        if spec["backend"] == "sentence_transformers":
            z = self.model.encode(
                texts,
                batch_size=ENCODER_BATCH_SIZE,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

            return np.asarray(z, dtype=np.float32)

        chunks = []

        for start in range(0, len(texts), ENCODER_BATCH_SIZE):
            local = texts[start:start + ENCODER_BATCH_SIZE]

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
                out = self.model(**batch)

                if spec["pooling"] == "mean":
                    z = mean_pool(
                        out.last_hidden_state,
                        batch["attention_mask"],
                    )
                elif spec["pooling"] == "cls":
                    z = out.last_hidden_state[:, 0]
                else:
                    raise ValueError(spec["pooling"])

            chunks.append(
                z.float().cpu().numpy()
            )

        return l2_dense(
            np.concatenate(
                chunks,
                axis=0,
            )
        )

    def close(self):
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


# ======================================================================================
# Upstream
# ======================================================================================

for p in [
    VIEW_COMPONENTS,
    VIEW_TEXT_CATALOG,
    ATOM_AUTHORITY,
    PRIMARY_EMB_INDEX,
    RESP_MANIFEST,
    HIST_CONFIG,
]:
    if not p.is_file():
        raise FileNotFoundError(p)


resp_manifest = json.loads(
    RESP_MANIFEST.read_text(encoding="utf-8")
)

if resp_manifest.get("status") != "PASS":
    raise RuntimeError("Response similarity is not PASS.")


hist_cfg = json.loads(
    HIST_CONFIG.read_text(encoding="utf-8")
)

gold_k = int(hist_cfg.get("gold_neighbor_k", 10))
tie_repeats = int(hist_cfg.get("retrieval_tie_repeats", 5))

components = pd.read_csv(
    VIEW_COMPONENTS,
    sep="\t",
    compression="gzip",
    low_memory=False,
)

catalog = pd.read_csv(
    VIEW_TEXT_CATALOG,
    sep="\t",
    compression="gzip",
    low_memory=False,
)

atoms = pd.read_csv(
    ATOM_AUTHORITY,
    sep="\t",
    low_memory=False,
)

primary_index = pd.read_csv(
    PRIMARY_EMB_INDEX,
    sep="\t",
    low_memory=False,
)

components["response_atom_id"] = components["response_atom_id"].astype(str)
components["component_id"] = components["component_id"].astype(str)
components["view"] = components["view"].astype(str)
components["text_index"] = components["text_index"].astype(np.int64)

catalog["text_index"] = catalog["text_index"].astype(np.int64)
catalog["prompt_text"] = catalog["prompt_text"].astype(str)

atoms["response_atom_id"] = atoms["response_atom_id"].astype(str)
atoms["dataset_id"] = atoms["dataset_id"].astype(str)

primary_index["response_atom_id"] = primary_index["response_atom_id"].astype(str)

if set(components["view"].unique()) != set(VIEWS):
    raise RuntimeError(
        f"Unexpected view set: {sorted(components['view'].unique())}"
    )

catalog = catalog.sort_values("text_index").reset_index(drop=True)

if not np.array_equal(
    catalog["text_index"].to_numpy(),
    np.arange(len(catalog), dtype=np.int64),
):
    raise RuntimeError("View text catalog index not contiguous.")


atom_ids = atoms["response_atom_id"].astype(str).tolist()

if len(atom_ids) != len(set(atom_ids)):
    raise RuntimeError("Duplicate response atom IDs.")

atom_pos = {
    rid: i
    for i, rid in enumerate(atom_ids)
}

if set(primary_index["response_atom_id"].astype(str)) != set(atom_ids):
    raise RuntimeError("Primary P4 embedding index namespace mismatch.")


requested = set(ARGS.models)

if requested:
    known = {x["model_key"] for x in MODEL_SPECS}
    unknown = requested - known

    if unknown:
        raise ValueError(f"Unknown models: {sorted(unknown)}")

    model_specs = [
        x
        for x in MODEL_SPECS
        if x["model_key"] in requested
    ]
else:
    model_specs = list(MODEL_SPECS)


# ======================================================================================
# Encode / aggregate
# ======================================================================================

texts = catalog["prompt_text"].tolist()


def encode_texts_for_model(
    spec: dict[str, Any],
) -> Path:
    key = spec["model_key"]

    model_dir = EMB_ROOT / key
    model_dir.mkdir(parents=True, exist_ok=True)

    path = model_dir / "all_view_text_embeddings.npy"
    done = model_dir / "text_DONE.json"

    expected_shape = (
        len(texts),
        int(spec["expected_dim"]),
    )

    if path.is_file() and done.is_file():
        meta = json.loads(done.read_text(encoding="utf-8"))

        try:
            arr = np.load(path, mmap_mode="r")
            shape_ok = tuple(arr.shape) == expected_shape
        except Exception:
            shape_ok = False

        if (
            shape_ok
            and meta.get("status") == "PASS"
            and meta.get("model_id") == spec["model_id"]
        ):
            log(f"[resume] {key} text embeddings")
            return path

    runner = EncoderRunner(spec)

    try:
        out_arr = np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype="float32",
            shape=expected_shape,
        )

        for start in range(
            0,
            len(texts),
            TEXT_CHUNK_SIZE,
        ):
            stop = min(
                len(texts),
                start + TEXT_CHUNK_SIZE,
            )

            z = runner.encode(
                texts[start:stop]
            )

            if z.shape != (
                stop - start,
                int(spec["expected_dim"]),
            ):
                raise RuntimeError(
                    f"{key}: text encoding shape mismatch"
                )

            if not np.isfinite(z).all():
                raise RuntimeError(
                    f"{key}: non-finite text embeddings"
                )

            out_arr[start:stop] = z
            out_arr.flush()

            log(
                f"[{key}] texts {stop}/{len(texts)}"
            )

        del out_arr

    finally:
        runner.close()

    write_json(
        {
            "status": "PASS",
            "model_key": key,
            "model_id": spec["model_id"],
            "text_n": len(texts),
            "dim": int(spec["expected_dim"]),
        },
        done,
    )

    return path


def aggregate_model_views(
    spec: dict[str, Any],
    text_path: Path,
) -> dict[str, Path]:
    key = spec["model_key"]
    dim = int(spec["expected_dim"])

    model_dir = EMB_ROOT / key

    text_emb = np.load(
        text_path,
        mmap_mode="r",
    )

    outputs = {}

    for view in VIEWS:
        path = (
            model_dir
            / f"{view}_response_atom_embeddings.npy"
        )

        done = (
            model_dir
            / f"{view}_aggregate_DONE.json"
        )

        expected_shape = (
            len(atom_ids),
            dim,
        )

        if path.is_file() and done.is_file():
            meta = json.loads(done.read_text(encoding="utf-8"))

            try:
                arr = np.load(path, mmap_mode="r")
                shape_ok = tuple(arr.shape) == expected_shape
            except Exception:
                shape_ok = False

            if shape_ok and meta.get("status") == "PASS":
                outputs[view] = path
                continue

        sub = components.loc[
            components["view"].eq(view)
        ].copy()

        sums = np.zeros(
            expected_shape,
            dtype=np.float32,
        )

        counts = np.zeros(
            len(atom_ids),
            dtype=np.int32,
        )

        for start in range(
            0,
            len(sub),
            AGG_CHUNK,
        ):
            block = sub.iloc[
                start:start + AGG_CHUNK
            ]

            rows = np.asarray(
                [
                    atom_pos[str(rid)]
                    for rid in block[
                        "response_atom_id"
                    ].astype(str)
                ],
                dtype=np.int64,
            )

            tids = block[
                "text_index"
            ].to_numpy(dtype=np.int64)

            vecs = np.asarray(
                text_emb[tids],
                dtype=np.float32,
            )

            np.add.at(
                sums,
                rows,
                vecs,
            )

            np.add.at(
                counts,
                rows,
                1,
            )

        if np.any(counts <= 0):
            raise RuntimeError(
                f"{key}/{view}: atom without components"
            )

        atom_emb = sums / counts[:, None].astype(np.float32)
        atom_emb = l2_dense(atom_emb)

        np.save(
            path,
            atom_emb.astype(np.float32),
        )

        write_json(
            {
                "status": "PASS",
                "model_key": key,
                "view": view,
                "response_atom_n": len(atom_ids),
                "dim": dim,
                "aggregation": "component_mean_then_l2",
            },
            done,
        )

        outputs[view] = path

    return outputs


if ARGS.preflight_only:
    for spec in model_specs:
        log(f"[preflight] {spec['model_key']}")

        runner = EncoderRunner(spec)

        try:
            sample = runner.encode(
                texts[: min(4, len(texts))]
            )

            if sample.shape[1] != int(spec["expected_dim"]):
                raise RuntimeError(
                    f"{spec['model_key']}: preflight dim mismatch"
                )

            log(
                f"[preflight] PASS {spec['model_key']} "
                f"shape={sample.shape}"
            )

        finally:
            runner.close()

    print("R2_P1P4_EMBEDDING_PREFLIGHT=PASS")
    raise SystemExit(0)


model_view_paths: dict[str, dict[str, Path]] = {}

for spec in model_specs:
    key = spec["model_key"]

    log(f"[model] {key}")

    text_path = encode_texts_for_model(spec)
    model_view_paths[key] = aggregate_model_views(
        spec,
        text_path,
    )


# ======================================================================================
# P4 exact reproduction check against frozen primary embeddings
# ======================================================================================

repro_rows = []

primary_pos = {
    rid: i
    for i, rid in enumerate(
        primary_index["response_atom_id"].astype(str)
    )
}

primary_reorder = np.asarray(
    [primary_pos[rid] for rid in atom_ids],
    dtype=np.int64,
)

for spec in model_specs:
    key = spec["model_key"]

    frozen_path = (
        PRIMARY_EMB_ROOT
        / key
        / "response_atom_embeddings.npy"
    )

    if not frozen_path.is_file():
        raise FileNotFoundError(frozen_path)

    frozen = np.load(
        frozen_path,
        mmap_mode="r",
    )

    new_p4 = np.load(
        model_view_paths[key]["P4"],
        mmap_mode="r",
    )

    frozen_aligned = np.asarray(
        frozen[primary_reorder],
        dtype=np.float32,
    )

    new_arr = np.asarray(
        new_p4,
        dtype=np.float32,
    )

    diff = np.abs(
        frozen_aligned - new_arr
    )

    repro_rows.append({
        "model_key": key,
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "allclose_atol_1e_6": bool(
            np.allclose(
                frozen_aligned,
                new_arr,
                atol=1e-6,
                rtol=1e-6,
            )
        ),
        "status": (
            "PASS"
            if np.allclose(
                frozen_aligned,
                new_arr,
                atol=1e-6,
                rtol=1e-6,
            )
            else "FAIL"
        ),
    })


p4_repro = pd.DataFrame(repro_rows)

p4_repro.to_csv(
    P4_REPRO_TSV,
    sep="\t",
    index=False,
)

if not p4_repro["status"].eq("PASS").all():
    raise RuntimeError(
        "P4 re-encoding does not reproduce frozen primary embeddings."
    )


# ======================================================================================
# Alignment metrics
# ======================================================================================

rows = []

for ds in DATASET_ORDER:
    log(f"[dataset] {ds}")

    resp_path = (
        RESP_ROOT
        / ds
        / "response_similarity_logfc_hvg_spearman.npz"
    )

    response_S, response_ids = load_similarity_bundle(
        resp_path
    )

    n = len(response_ids)

    rows_idx = np.asarray(
        [atom_pos[rid] for rid in response_ids],
        dtype=np.int64,
    )

    response_upper = upper_vec(response_S)

    retrieval_seed = stable_seed(
        SEED,
        ds,
        "p1p4_retrieval",
    )

    uniform10 = exact_random_ndcg_expectation(
        n_atoms=n,
        gold_k=gold_k,
        eval_k=10,
    )

    for spec in model_specs:
        key = spec["model_key"]

        for view in VIEWS:
            emb = np.load(
                model_view_paths[key][view],
                mmap_mode="r",
            )

            Z = np.asarray(
                emb[rows_idx],
                dtype=np.float32,
            )

            S = (Z @ Z.T).astype(np.float32)

            S = np.clip(
                S,
                -1.0,
                1.0,
            )

            np.fill_diagonal(
                S,
                1.0,
            )

            text_upper = upper_vec(S)

            rsa, rsa_status = safe_corr(
                text_upper,
                response_upper,
                "spearman",
            )

            ret = retrieval_metrics(
                text_S=S,
                response_S=response_S,
                gold_k=gold_k,
                tie_repeats=tie_repeats,
                seed=retrieval_seed,
            )

            if rsa_status != "ok":
                raise RuntimeError(
                    f"{ds}/{key}/{view}: RSA invalid {rsa_status}"
                )

            if ret.get("status") != "ok":
                raise RuntimeError(
                    f"{ds}/{key}/{view}: retrieval invalid"
                )

            ndcg10 = float(
                ret["ndcg_at_10"]
            )

            rows.append({
                "dataset_id": ds,
                "model_key": key,
                "model_display": spec["display_name"],
                "model_group": spec["group"],
                "view": view,
                "response_atom_n": n,
                "spearman_rsa": float(rsa),
                "ndcg_at_10": ndcg10,
                "exact_uniform_random_ndcg_at_10": float(uniform10),
                "exact_uniform_excess_ndcg_at_10": float(
                    ndcg10 - uniform10
                ),
                "recall_at_10": float(
                    ret["recall_at_10"]
                ),
                "status": "PASS",
            })


by_dataset = pd.DataFrame(rows).sort_values(
    ["model_key", "dataset_id", "view"]
).reset_index(drop=True)

by_dataset.to_csv(
    BY_DATASET_TSV,
    sep="\t",
    index=False,
)


# ======================================================================================
# Deltas
# ======================================================================================

delta_rows = []

for ds in DATASET_ORDER:
    for spec in model_specs:
        key = spec["model_key"]

        sub = by_dataset.loc[
            by_dataset["dataset_id"].eq(ds)
            & by_dataset["model_key"].eq(key)
        ].set_index("view")

        for delta_name, view_a, view_b in DELTA_DEFS:
            delta_rows.append({
                "dataset_id": ds,
                "model_key": key,
                "model_display": spec["display_name"],
                "model_group": spec["group"],
                "delta_name": delta_name,
                "view_from": view_a,
                "view_to": view_b,
                "delta_rsa": float(
                    sub.loc[view_b, "spearman_rsa"]
                    - sub.loc[view_a, "spearman_rsa"]
                ),
                "delta_ndcg_at_10": float(
                    sub.loc[view_b, "ndcg_at_10"]
                    - sub.loc[view_a, "ndcg_at_10"]
                ),
                "delta_exact_uniform_excess_ndcg_at_10": float(
                    sub.loc[
                        view_b,
                        "exact_uniform_excess_ndcg_at_10",
                    ]
                    - sub.loc[
                        view_a,
                        "exact_uniform_excess_ndcg_at_10",
                    ]
                ),
            })


deltas = pd.DataFrame(delta_rows)

deltas.to_csv(
    DELTA_BY_DATASET_TSV,
    sep="\t",
    index=False,
)


# ======================================================================================
# Summaries
# ======================================================================================

summary_rows = []

for (model_key, delta_name), sub in deltas.groupby(
    ["model_key", "delta_name"],
    sort=False,
):
    first = sub.iloc[0]

    row = {
        "model_key": model_key,
        "model_display": first["model_display"],
        "model_group": first["model_group"],
        "delta_name": delta_name,
        "dataset_n": int(sub["dataset_id"].nunique()),
        "delta_rsa_macro_mean": float(sub["delta_rsa"].mean()),
        "delta_rsa_macro_median": float(sub["delta_rsa"].median()),
        "delta_rsa_positive_dataset_n": int(
            (sub["delta_rsa"] > 0).sum()
        ),
        "delta_ndcg_at_10_macro_mean": float(
            sub["delta_ndcg_at_10"].mean()
        ),
        "delta_ndcg_at_10_macro_median": float(
            sub["delta_ndcg_at_10"].median()
        ),
        "delta_ndcg_at_10_positive_dataset_n": int(
            (sub["delta_ndcg_at_10"] > 0).sum()
        ),
    }

    for metric in [
        "delta_rsa",
        "delta_ndcg_at_10",
    ]:
        lo, hi = bootstrap_ci(
            sub[metric].to_numpy(dtype=float),
            seed=stable_seed(
                SEED,
                model_key,
                delta_name,
                metric,
                "dataset_bootstrap",
            ),
        )

        row[f"{metric}_bootstrap_ci_low"] = lo
        row[f"{metric}_bootstrap_ci_high"] = hi

    summary_rows.append(row)


delta_summary = pd.DataFrame(summary_rows).sort_values(
    ["delta_name", "model_key"]
).reset_index(drop=True)

delta_summary.to_csv(
    DELTA_SUMMARY_TSV,
    sep="\t",
    index=False,
)


absolute_rows = []

for (model_key, view), sub in by_dataset.groupby(
    ["model_key", "view"],
    sort=False,
):
    first = sub.iloc[0]

    absolute_rows.append({
        "model_key": model_key,
        "model_display": first["model_display"],
        "model_group": first["model_group"],
        "view": view,
        "dataset_n": int(sub["dataset_id"].nunique()),
        "rsa_macro_mean": float(sub["spearman_rsa"].mean()),
        "rsa_macro_median": float(sub["spearman_rsa"].median()),
        "ndcg10_macro_mean": float(sub["ndcg_at_10"].mean()),
        "uniform_excess_ndcg10_macro_mean": float(
            sub["exact_uniform_excess_ndcg_at_10"].mean()
        ),
    })


absolute_summary = pd.DataFrame(absolute_rows)

absolute_summary.to_csv(
    ABSOLUTE_SUMMARY_TSV,
    sep="\t",
    index=False,
)


# ======================================================================================
# Audit
# ======================================================================================

full_run = (
    len(model_specs) == 6
)

status = (
    "PASS"
    if full_run
    else "PARTIAL_PASS"
)

lines = [
    "PERTURBCONTEXTALIGN R2 CORRECTED P1-P4 ABLATION AUDIT v1",
    "=" * 120,
    "",
    "UPSTREAM",
    "-" * 120,
    "response_atom_namespace=FROZEN",
    "measured_response_similarity=PASS",
    "P1P4_text_authority=corrected_response_aligned",
    "technical_context_included=FALSE",
    "",
    "ENCODER CONTRACT",
    "-" * 120,
    "same_six_frozen_R1_encoders=TRUE",
    f"max_length={MAX_LENGTH}",
    f"encoder_batch_size={ENCODER_BATCH_SIZE}",
    "component_aggregation=mean_then_L2",
    "",
    "P4 REPRODUCTION",
    "-" * 120,
    p4_repro.to_string(index=False),
    "",
    "COUNTS",
    "-" * 120,
    f"dataset_n={by_dataset['dataset_id'].nunique()}",
    f"model_n={by_dataset['model_key'].nunique()}",
    f"view_n={by_dataset['view'].nunique()}",
    f"dataset_model_view_row_n={len(by_dataset)}",
    f"delta_row_n={len(deltas)}",
    "",
    "DELTA SUMMARY",
    "-" * 120,
    delta_summary.to_string(index=False),
    "",
    "STATUS",
    "-" * 120,
    f"R2_CORRECTED_P1_P4_ABLATION={status}",
]

if full_run:
    lines += [
        "NEXT=R2_WITHIN_CONDITION_CROSS_CONTEXT_ANALYSIS_v1",
    ]
else:
    lines += [
        "NEXT=COMPLETE_REMAINING_ENCODERS",
    ]

AUDIT_TXT.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)

write_json(
    {
        "version": "R2_CORRECTED_P1_P4_ABLATION_v1",
        "status": status,
        "views": VIEWS,
        "delta_definitions": [
            {
                "name": name,
                "from": a,
                "to": b,
            }
            for name, a, b in DELTA_DEFS
        ],
        "method": {
            "text_similarity": "cosine",
            "response_similarity": "logfc_hvg__spearman",
            "rsa": "spearman_upper_triangle",
            "local_metric": "NDCG@10",
            "local_reference": "exact_uniform_random_expectation",
            "component_aggregation": "mean_then_l2",
        },
        "inputs": {
            "view_components": {
                "path": str(VIEW_COMPONENTS),
                "sha256": sha256_file(VIEW_COMPONENTS),
            },
            "response_atom_authority": {
                "path": str(ATOM_AUTHORITY),
                "sha256": sha256_file(ATOM_AUTHORITY),
            },
            "response_similarity_manifest": {
                "path": str(RESP_MANIFEST),
                "sha256": sha256_file(RESP_MANIFEST),
            },
        },
        "outputs": {
            "by_dataset": str(BY_DATASET_TSV),
            "deltas_by_dataset": str(DELTA_BY_DATASET_TSV),
            "delta_summary": str(DELTA_SUMMARY_TSV),
            "absolute_summary": str(ABSOLUTE_SUMMARY_TSV),
            "p4_reproduction": str(P4_REPRO_TSV),
            "audit": str(AUDIT_TXT),
        },
    },
    MANIFEST_JSON,
)

print(
    AUDIT_TXT.read_text(
        encoding="utf-8"
    )
)

if full_run and status != "PASS":
    raise SystemExit(2)
