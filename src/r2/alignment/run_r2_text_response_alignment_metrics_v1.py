from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata


# ======================================================================================
# CLI / paths
# ======================================================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "PerturbContextAlign R2 primary text-response alignment metrics: "
            "Global Spearman RSA, NDCG@10, exact random NDCG baseline, "
            "and excess NDCG@10."
        )
    )
    ap.add_argument(
        "--root",
        default=".",
    )
    ap.add_argument(
        "--models",
        nargs="*",
        default=[],
        help="Optional subset of model_key values. Default: all frozen R1 encoders.",
    )
    ap.add_argument(
        "--datasets",
        nargs="*",
        default=[],
        help="Optional dataset subset. Default: all 9 frozen quantitative datasets.",
    )
    return ap.parse_args()


ARGS = parse_args()

ROOT = Path(ARGS.root).expanduser().resolve()

NEW = (
    ROOT
    / "0_ProjectCodeAndLogs"
    / "PerturbContextAlign_R2_corrected_v1_20261002"
)

EMB_ROOT = NEW / "05_response_aligned_text" / "02_embeddings_v1"
EMB_MANIFEST = EMB_ROOT / "R2_RESPONSE_ALIGNED_TEXT_EMBEDDING_MANIFEST_v1.json"
MODEL_SPECS_TSV = EMB_ROOT / "R2_TEXT_EMBEDDING_MODEL_SPECS_v1.tsv"
ATOM_EMB_INDEX = EMB_ROOT / "R2_RESPONSE_ATOM_EMBEDDING_INDEX_v1.tsv"

RESP_ROOT = NEW / "04_response_geometry" / "02_similarity_v1"
RESP_MANIFEST = RESP_ROOT / "R2_RESPONSE_SIMILARITY_MANIFEST_v1.json"
RESP_AUDIT_TSV = RESP_ROOT / "R2_RESPONSE_SIMILARITY_AUDIT_v1.tsv"

HIST_EFFECTIVE_CONFIG = (
    NEW
    / "01_contract"
    / "R2_HISTORICAL_EFFECTIVE_CONFIG_v1.json"
)

CORRECTED_CONTRACT = (
    NEW
    / "01_contract"
    / "R2_CORRECTED_METHOD_CONTRACT_DRAFT_v1.md"
)

OUT = NEW / "06_alignment_metrics" / "01_primary_v1"
OUT.mkdir(parents=True, exist_ok=True)

BY_DATASET_TSV = OUT / "R2_PRIMARY_ALIGNMENT_BY_DATASET_v1.tsv"
MODEL_SUMMARY_TSV = OUT / "R2_PRIMARY_ALIGNMENT_MODEL_SUMMARY_v1.tsv"
AUDIT_TXT = OUT / "R2_PRIMARY_ALIGNMENT_AUDIT_v1.txt"
MANIFEST_JSON = OUT / "R2_PRIMARY_ALIGNMENT_MANIFEST_v1.json"

SCRIPT_VERSION = "R2_TEXT_RESPONSE_ALIGNMENT_METRICS_v1.0.0"
SEED = 20260711
BOOTSTRAP_REPEATS = 10000


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


def load_similarity_bundle(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=True) as z:
        keys = set(z.files)
        matrix_key = "S" if "S" in keys else "X"

        if matrix_key not in keys:
            raise KeyError(f"{path}: similarity matrix missing")

        S = np.asarray(z[matrix_key], dtype=np.float32)

        id_key = (
            "response_atom_ids"
            if "response_atom_ids" in keys
            else "condition_ids"
        )

        if id_key not in keys:
            raise KeyError(f"{path}: response atom IDs missing")

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
) -> tuple[float, str, int]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]

    if len(x) < 3:
        return np.nan, "invalid_too_few_pairs", int(len(x))

    if np.ptp(x) <= 1e-12:
        return np.nan, "invalid_constant_text", int(len(x))

    if np.ptp(y) <= 1e-12:
        return np.nan, "invalid_constant_response", int(len(x))

    if method == "spearman":
        x = rankdata(x)
        y = rankdata(y)
    elif method != "pearson":
        raise ValueError(method)

    x = x - x.mean()
    y = y - y.mean()

    denom = math.sqrt(
        float(np.dot(x, x))
        * float(np.dot(y, y))
    )

    if denom <= 0 or not np.isfinite(denom):
        return np.nan, "invalid_zero_variance", int(len(x))

    return (
        float(np.dot(x, y) / denom),
        "ok",
        int(len(x)),
    )


def random_tie_ranking(
    S: np.ndarray,
    i: int,
    rng: np.random.Generator,
) -> np.ndarray:
    row = np.asarray(S[i], dtype=np.float64).copy()
    row[~np.isfinite(row)] = -np.inf
    row[i] = -np.inf

    # Historical Result2 tie policy.
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
    recall_ks=(5, 10, 20),
    gold_k: int = 10,
    ndcg_ks=(10, 20),
    tie_repeats: int = 5,
    seed: int = 0,
) -> dict[str, Any]:
    """
    Historical Result2 retrieval definition:
      * self excluded
      * gold = top-k measured-response neighbors
      * binary relevance
      * seeded random tie breaking for both response and text rankings
      * mean over all queries and tie repeats

    Corrected-R2 fairness detail:
      the seed is dataset-specific and shared across models, so response-space
      tie resolution is comparable across encoder models.
    """
    text_S = np.asarray(text_S, dtype=np.float64)
    response_S = np.asarray(response_S, dtype=np.float64)

    n = int(text_S.shape[0])

    if text_S.shape != response_S.shape:
        raise ValueError(
            f"retrieval matrix shape mismatch "
            f"{text_S.shape} vs {response_S.shape}"
        )

    if n < 3:
        return {
            "status": "invalid_too_few_atoms",
            "n_queries": 0,
        }

    rows = []
    repeats = max(1, int(tie_repeats))

    for repeat in range(repeats):
        # Reset from the same deterministic repeat seed for every model.
        # Effect gold rankings therefore use the same random tie-break stream.
        response_rng = np.random.default_rng(
            int(seed) + repeat
        )
        text_rng = np.random.default_rng(
            int(seed) + 1000003 + repeat
        )

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

            g = min(
                int(gold_k),
                n - 1,
            )

            gold = set(
                response_rank[:g].tolist()
            )

            if not gold:
                continue

            row: dict[str, float] = {}

            for k in recall_ks:
                kk = min(int(k), n - 1)

                row[f"recall_at_{k}"] = (
                    len(
                        gold.intersection(
                            text_rank[:kk]
                        )
                    )
                    / len(gold)
                )

            hit_ranks = [
                r
                for r, j in enumerate(
                    text_rank,
                    start=1,
                )
                if j in gold
            ]

            row["mrr"] = (
                1.0 / min(hit_ranks)
                if hit_ranks
                else 0.0
            )

            for k in ndcg_ks:
                kk = min(
                    int(k),
                    n - 1,
                )

                rel = np.asarray(
                    [
                        j in gold
                        for j in text_rank[:kk]
                    ],
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

                dcg = float(
                    np.dot(
                        rel,
                        discounts,
                    )
                )

                ideal = min(
                    len(gold),
                    kk,
                )

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

    if not rows:
        return {
            "status": "invalid_no_queries",
            "n_queries": 0,
        }

    frame = pd.DataFrame(rows)

    out = {
        col: float(frame[col].mean())
        for col in frame.columns
    }

    out.update({
        "n_queries": int(n),
        "tie_repeats": int(repeats),
        "status": "ok",
    })

    return out


def exact_random_ndcg_expectation(
    n_atoms: int,
    gold_k: int,
    eval_k: int,
) -> float:
    """
    Exact expected NDCG for a uniformly random ranking under binary relevance.

    The gold set contains g=min(gold_k,n-1) relevant candidates among n-1
    possible non-self candidates. Each rank has relevance probability g/(n-1).
    """
    candidates = int(n_atoms) - 1

    if candidates <= 0:
        return np.nan

    g = min(
        int(gold_k),
        candidates,
    )

    k = min(
        int(eval_k),
        candidates,
    )

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
        * float(
            discounts.sum()
        )
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

    return (
        expected_dcg / idcg
        if idcg > 0
        else np.nan
    )


def bootstrap_macro_ci(
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

    indices = rng.integers(
        0,
        len(x),
        size=(int(repeats), len(x)),
    )

    means = x[indices].mean(axis=1)

    return (
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


# ======================================================================================
# Frozen upstream authorities
# ======================================================================================

for p in [
    EMB_MANIFEST,
    MODEL_SPECS_TSV,
    ATOM_EMB_INDEX,
    RESP_MANIFEST,
    RESP_AUDIT_TSV,
    HIST_EFFECTIVE_CONFIG,
    CORRECTED_CONTRACT,
]:
    if not p.is_file():
        raise FileNotFoundError(p)


emb_manifest = json.loads(
    EMB_MANIFEST.read_text(encoding="utf-8")
)

resp_manifest = json.loads(
    RESP_MANIFEST.read_text(encoding="utf-8")
)

effective_cfg = json.loads(
    HIST_EFFECTIVE_CONFIG.read_text(encoding="utf-8")
)

if emb_manifest.get("status") != "PASS":
    raise RuntimeError(
        "Frozen response-aligned text embedding manifest is not PASS."
    )

if resp_manifest.get("status") != "PASS":
    raise RuntimeError(
        "Frozen response-similarity manifest is not PASS."
    )

model_specs = pd.read_csv(
    MODEL_SPECS_TSV,
    sep="\t",
    low_memory=False,
)

atom_emb_index = pd.read_csv(
    ATOM_EMB_INDEX,
    sep="\t",
    low_memory=False,
)

resp_audit = pd.read_csv(
    RESP_AUDIT_TSV,
    sep="\t",
    low_memory=False,
)

if not resp_audit["status"].astype(str).eq("PASS").all():
    raise RuntimeError(
        "Not all response-similarity datasets are PASS."
    )

if "response_atom_id" not in atom_emb_index.columns:
    raise KeyError(
        "Embedding index lacks response_atom_id."
    )

atom_emb_index = atom_emb_index.copy()
atom_emb_index["response_atom_id"] = (
    atom_emb_index["response_atom_id"].astype(str)
)

if atom_emb_index["response_atom_id"].duplicated().any():
    raise RuntimeError(
        "Embedding index response_atom_id is not unique."
    )

if "row_index" in atom_emb_index.columns:
    expected_rows = np.arange(
        len(atom_emb_index),
        dtype=np.int64,
    )
    observed_rows = atom_emb_index["row_index"].to_numpy(dtype=np.int64)

    if not np.array_equal(
        expected_rows,
        observed_rows,
    ):
        raise RuntimeError(
            "Embedding index row_index is not contiguous."
        )


# Retrieval contract frozen from historical method.
gold_k = int(
    effective_cfg.get(
        "gold_neighbor_k",
        10,
    )
)

ndcg_ks = tuple(
    int(x)
    for x in effective_cfg.get(
        "ndcg_ks",
        [10, 20],
    )
)

tie_repeats = int(
    effective_cfg.get(
        "retrieval_tie_repeats",
        5,
    )
)

if gold_k != 10:
    raise RuntimeError(
        f"Unexpected frozen gold_neighbor_k={gold_k}"
    )

if 10 not in ndcg_ks:
    raise RuntimeError(
        f"NDCG@10 absent from frozen ndcg_ks={ndcg_ks}"
    )


all_models = model_specs["model_key"].astype(str).tolist()

if len(all_models) != 6:
    raise RuntimeError(
        f"Expected six frozen R1 encoders, got {len(all_models)}"
    )

if ARGS.models:
    unknown = sorted(
        set(ARGS.models) - set(all_models)
    )
    if unknown:
        raise ValueError(
            f"Unknown model keys: {unknown}"
        )
    requested_models = [
        x
        for x in all_models
        if x in set(ARGS.models)
    ]
else:
    requested_models = list(all_models)


all_datasets = resp_audit["dataset_id"].astype(str).tolist()

if len(all_datasets) != 9:
    raise RuntimeError(
        f"Expected 9 response datasets, got {len(all_datasets)}"
    )

if ARGS.datasets:
    unknown = sorted(
        set(ARGS.datasets) - set(all_datasets)
    )
    if unknown:
        raise ValueError(
            f"Unknown datasets: {unknown}"
        )
    requested_datasets = [
        x
        for x in all_datasets
        if x in set(ARGS.datasets)
    ]
else:
    requested_datasets = list(all_datasets)


# ======================================================================================
# Load model embedding memmaps
# ======================================================================================

model_embedding_paths: dict[str, Path] = {}
model_embeddings: dict[str, np.ndarray] = {}

for model_key in requested_models:
    p = (
        EMB_ROOT
        / model_key
        / "response_atom_embeddings.npy"
    )

    done = (
        EMB_ROOT
        / model_key
        / "aggregate_DONE.json"
    )

    if not p.is_file():
        raise FileNotFoundError(p)

    if not done.is_file():
        raise FileNotFoundError(done)

    done_json = json.loads(
        done.read_text(encoding="utf-8")
    )

    if done_json.get("status") != "PASS":
        raise RuntimeError(
            f"{model_key}: aggregate_DONE is not PASS"
        )

    arr = np.load(
        p,
        mmap_mode="r",
    )

    if arr.shape[0] != len(atom_emb_index):
        raise RuntimeError(
            f"{model_key}: embedding rows {arr.shape[0]} "
            f"!= index rows {len(atom_emb_index)}"
        )

    model_embedding_paths[model_key] = p
    model_embeddings[model_key] = arr


atom_pos = {
    rid: i
    for i, rid in enumerate(
        atom_emb_index["response_atom_id"].astype(str)
    )
}


# ======================================================================================
# Primary per-dataset metrics
# ======================================================================================

rows: list[dict[str, Any]] = []

for dataset_id in requested_datasets:
    log(f"[dataset] {dataset_id}")

    resp_path = (
        RESP_ROOT
        / dataset_id
        / "response_similarity_logfc_hvg_spearman.npz"
    )

    resp_index_path = (
        RESP_ROOT
        / dataset_id
        / "response_similarity_index.tsv"
    )

    resp_done_path = (
        RESP_ROOT
        / dataset_id
        / "DONE.json"
    )

    for p in [
        resp_path,
        resp_index_path,
        resp_done_path,
    ]:
        if not p.is_file():
            raise FileNotFoundError(p)

    resp_done = json.loads(
        resp_done_path.read_text(encoding="utf-8")
    )

    if resp_done.get("status") != "PASS":
        raise RuntimeError(
            f"{dataset_id}: response similarity DONE not PASS"
        )

    response_S, response_ids = load_similarity_bundle(
        resp_path
    )

    response_index = pd.read_csv(
        resp_index_path,
        sep="\t",
        low_memory=False,
    )

    if "response_atom_id" not in response_index.columns:
        raise KeyError(
            f"{dataset_id}: response similarity index lacks response_atom_id"
        )

    index_ids = (
        response_index["response_atom_id"]
        .astype(str)
        .to_numpy()
    )

    if not np.array_equal(
        index_ids,
        response_ids,
    ):
        raise RuntimeError(
            f"{dataset_id}: response similarity bundle/index ID mismatch"
        )

    n = len(response_ids)

    if response_S.shape != (n, n):
        raise RuntimeError(
            f"{dataset_id}: response similarity shape {response_S.shape} "
            f"!= ({n},{n})"
        )

    if not np.isfinite(response_S).all():
        raise RuntimeError(
            f"{dataset_id}: response similarity contains non-finite values"
        )

    missing_embedding_ids = [
        rid
        for rid in response_ids
        if rid not in atom_pos
    ]

    if missing_embedding_ids:
        raise RuntimeError(
            f"{dataset_id}: {len(missing_embedding_ids)} "
            "response atoms absent from text embedding index"
        )

    emb_rows = np.asarray(
        [atom_pos[rid] for rid in response_ids],
        dtype=np.int64,
    )

    response_upper = upper_vec(
        response_S
    )

    response_upper_sd = float(
        np.std(response_upper)
    )

    if response_upper_sd <= 1e-12:
        raise RuntimeError(
            f"{dataset_id}: response geometry is constant"
        )

    # Dataset-specific shared seed: identical effect-gold tie resolution across models.
    retrieval_seed = stable_seed(
        SEED,
        dataset_id,
        "primary_retrieval",
    )

    random_ndcg_10 = exact_random_ndcg_expectation(
        n_atoms=n,
        gold_k=gold_k,
        eval_k=10,
    )

    random_ndcg_20 = exact_random_ndcg_expectation(
        n_atoms=n,
        gold_k=gold_k,
        eval_k=20,
    )

    for model_key in requested_models:
        spec = model_specs.loc[
            model_specs["model_key"].astype(str).eq(model_key)
        ].iloc[0]

        Z = np.asarray(
            model_embeddings[model_key][emb_rows],
            dtype=np.float32,
        )

        if Z.shape[0] != n:
            raise RuntimeError(
                f"{dataset_id}/{model_key}: subset embedding rows mismatch"
            )

        if not np.isfinite(Z).all():
            raise RuntimeError(
                f"{dataset_id}/{model_key}: non-finite embeddings"
            )

        norms = np.linalg.norm(
            Z,
            axis=1,
        )

        max_norm_dev = float(
            np.max(
                np.abs(
                    norms - 1.0
                )
            )
        )

        if max_norm_dev > 5e-4:
            raise RuntimeError(
                f"{dataset_id}/{model_key}: embedding norm drift "
                f"{max_norm_dev}"
            )

        # Frozen text embeddings are L2-normalized, so cosine = dot product.
        text_S = (
            Z
            @ Z.T
        ).astype(
            np.float32,
            copy=False,
        )

        text_S = np.clip(
            text_S,
            -1.0,
            1.0,
        )

        np.fill_diagonal(
            text_S,
            1.0,
        )

        text_upper = upper_vec(
            text_S
        )

        text_upper_sd = float(
            np.std(text_upper)
        )

        if text_upper_sd <= 1e-12:
            raise RuntimeError(
                f"{dataset_id}/{model_key}: text geometry is constant"
            )

        spearman_rsa, rsa_status, rsa_pair_n = safe_corr(
            text_upper,
            response_upper,
            method="spearman",
        )

        pearson_rsa, pearson_status, _ = safe_corr(
            text_upper,
            response_upper,
            method="pearson",
        )

        ret = retrieval_metrics(
            text_S=text_S,
            response_S=response_S,
            recall_ks=(5, 10, 20),
            gold_k=gold_k,
            ndcg_ks=ndcg_ks,
            tie_repeats=tie_repeats,
            seed=retrieval_seed,
        )

        if rsa_status != "ok":
            raise RuntimeError(
                f"{dataset_id}/{model_key}: Spearman RSA status={rsa_status}"
            )

        if ret.get("status") != "ok":
            raise RuntimeError(
                f"{dataset_id}/{model_key}: retrieval status={ret.get('status')}"
            )

        ndcg10 = float(
            ret["ndcg_at_10"]
        )

        ndcg20 = (
            float(ret["ndcg_at_20"])
            if "ndcg_at_20" in ret
            else np.nan
        )

        excess10 = (
            ndcg10
            - random_ndcg_10
        )

        random_adjusted10 = (
            excess10
            / (1.0 - random_ndcg_10)
            if np.isfinite(random_ndcg_10)
            and random_ndcg_10 < 1.0
            else np.nan
        )

        rows.append({
            "dataset_id": dataset_id,
            "model_key": model_key,
            "model_display": str(spec["display_name"]),
            "model_group": str(spec["group"]),
            "model_id": str(spec["model_id"]),
            "response_atom_n": int(n),
            "pair_n": int(rsa_pair_n),
            "text_similarity": "cosine",
            "response_similarity": "logfc_hvg__spearman",
            "spearman_rsa": float(spearman_rsa),
            "spearman_rsa_status": rsa_status,
            "pearson_rsa": float(pearson_rsa),
            "pearson_rsa_status": pearson_status,
            "text_upper_sd": text_upper_sd,
            "response_upper_sd": response_upper_sd,
            "gold_neighbor_k": int(gold_k),
            "retrieval_tie_repeats": int(tie_repeats),
            "retrieval_seed": int(retrieval_seed),
            "recall_at_5": float(ret.get("recall_at_5", np.nan)),
            "recall_at_10": float(ret.get("recall_at_10", np.nan)),
            "recall_at_20": float(ret.get("recall_at_20", np.nan)),
            "mrr": float(ret.get("mrr", np.nan)),
            "ndcg_at_10": ndcg10,
            "ndcg_at_20": ndcg20,
            "random_ndcg_at_10_exact": float(random_ndcg_10),
            "random_ndcg_at_20_exact": float(random_ndcg_20),
            "excess_ndcg_at_10": float(excess10),
            "random_adjusted_ndcg_at_10": float(random_adjusted10),
            "embedding_norm_max_deviation": max_norm_dev,
            "status": "PASS",
        })

        log(
            f"  {model_key}: "
            f"RSA={spearman_rsa:.6f} "
            f"NDCG10={ndcg10:.6f} "
            f"excess={excess10:.6f}"
        )


by_dataset = pd.DataFrame(rows)

if by_dataset.empty:
    raise RuntimeError("No alignment rows produced.")

expected_rows = (
    len(requested_datasets)
    * len(requested_models)
)

if len(by_dataset) != expected_rows:
    raise RuntimeError(
        f"Alignment row count {len(by_dataset)} != expected {expected_rows}"
    )

if not by_dataset["status"].eq("PASS").all():
    raise RuntimeError("Not all alignment rows are PASS.")

by_dataset = by_dataset.sort_values(
    ["model_key", "dataset_id"]
).reset_index(drop=True)

by_dataset.to_csv(
    BY_DATASET_TSV,
    sep="\t",
    index=False,
)


# ======================================================================================
# Macro summaries + deterministic dataset bootstrap CIs
# ======================================================================================

summary_rows = []

for model_key, sub in by_dataset.groupby(
    "model_key",
    sort=False,
):
    first = sub.iloc[0]

    row: dict[str, Any] = {
        "model_key": model_key,
        "model_display": first["model_display"],
        "model_group": first["model_group"],
        "model_id": first["model_id"],
        "dataset_n": int(sub["dataset_id"].nunique()),
        "total_response_atom_n": int(sub["response_atom_n"].sum()),
        "total_pair_n": int(sub["pair_n"].sum()),
        "spearman_rsa_macro_mean": float(sub["spearman_rsa"].mean()),
        "spearman_rsa_macro_median": float(sub["spearman_rsa"].median()),
        "spearman_rsa_positive_dataset_n": int((sub["spearman_rsa"] > 0).sum()),
        "ndcg_at_10_macro_mean": float(sub["ndcg_at_10"].mean()),
        "ndcg_at_20_macro_mean": float(sub["ndcg_at_20"].mean()),
        "random_ndcg_at_10_macro_mean": float(
            sub["random_ndcg_at_10_exact"].mean()
        ),
        "excess_ndcg_at_10_macro_mean": float(
            sub["excess_ndcg_at_10"].mean()
        ),
        "excess_ndcg_at_10_positive_dataset_n": int(
            (sub["excess_ndcg_at_10"] > 0).sum()
        ),
        "random_adjusted_ndcg_at_10_macro_mean": float(
            sub["random_adjusted_ndcg_at_10"].mean()
        ),
        "recall_at_10_macro_mean": float(sub["recall_at_10"].mean()),
        "mrr_macro_mean": float(sub["mrr"].mean()),
    }

    for metric in [
        "spearman_rsa",
        "ndcg_at_10",
        "excess_ndcg_at_10",
    ]:
        lo, hi = bootstrap_macro_ci(
            sub[metric].to_numpy(dtype=np.float64),
            seed=stable_seed(
                SEED,
                model_key,
                metric,
                "dataset_bootstrap",
            ),
        )

        row[f"{metric}_dataset_bootstrap_ci_low"] = lo
        row[f"{metric}_dataset_bootstrap_ci_high"] = hi

    # Pair-weighted RSA is secondary/descriptive only.
    weights = sub["pair_n"].to_numpy(dtype=np.float64)
    rsa_vals = sub["spearman_rsa"].to_numpy(dtype=np.float64)

    row["spearman_rsa_pair_weighted_mean_secondary"] = float(
        np.average(
            rsa_vals,
            weights=weights,
        )
    )

    summary_rows.append(row)


summary = pd.DataFrame(summary_rows).sort_values(
    "spearman_rsa_macro_mean",
    ascending=False,
).reset_index(drop=True)

summary.to_csv(
    MODEL_SUMMARY_TSV,
    sep="\t",
    index=False,
)


# ======================================================================================
# Final audit / manifest
# ======================================================================================

full_run = (
    set(requested_models) == set(all_models)
    and set(requested_datasets) == set(all_datasets)
)

status = (
    "PASS"
    if full_run
    else "PARTIAL_PASS"
)

lines = [
    "PERTURBCONTEXTALIGN R2 PRIMARY TEXT-RESPONSE ALIGNMENT AUDIT v1",
    "=" * 120,
    "",
    "UPSTREAM",
    "-" * 120,
    "response_atom_namespace=FROZEN",
    "response_aligned_text_embeddings=PASS",
    "measured_response_similarity=PASS",
    "H5AD_access=FALSE",
    "effect_recomputed=FALSE",
    "",
    "PRIMARY METHOD",
    "-" * 120,
    "text_similarity=cosine(response_atom_embedding_i,response_atom_embedding_j)",
    "response_similarity=Spearman(logFC_HVG_i,logFC_HVG_j)",
    "global_RSA=Spearman(strict_upper_triangle(text_similarity),strict_upper_triangle(response_similarity))",
    f"gold_neighbor_k={gold_k}",
    f"retrieval_tie_repeats={tie_repeats}",
    "retrieval_relevance=binary_top10_response_neighbors",
    "self_neighbor=excluded",
    "primary_retrieval_metric=NDCG@10",
    "random_NDCG_baseline=exact_uniform_random_ranking_expectation",
    "excess_NDCG@10=NDCG@10_model-random_NDCG@10_exact",
    "random_adjusted_NDCG@10=(model-random)/(1-random) [secondary historical-style statistic]",
    "dataset_macro_mean=PRIMARY cross-dataset summary",
    f"dataset_bootstrap_repeats={BOOTSTRAP_REPEATS}",
    "",
    "COUNTS",
    "-" * 120,
    f"dataset_n={by_dataset['dataset_id'].nunique()}",
    f"model_n={by_dataset['model_key'].nunique()}",
    f"dataset_model_row_n={len(by_dataset)}",
    "",
    "MODEL SUMMARY",
    "-" * 120,
    summary.to_string(index=False),
    "",
    "STATUS",
    "-" * 120,
    f"R2_PRIMARY_TEXT_RESPONSE_ALIGNMENT={status}",
]

if full_run:
    lines += [
        "NEXT=R2_PRIMARY_ALIGNMENT_INTERPRETATION_AND_SENSITIVITY",
    ]
else:
    lines += [
        "NEXT=COMPLETE_REMAINING_MODELS_OR_DATASETS",
    ]

AUDIT_TXT.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)

manifest = {
    "version": "R2_PRIMARY_TEXT_RESPONSE_ALIGNMENT_v1",
    "script_version": SCRIPT_VERSION,
    "status": status,
    "seed": SEED,
    "dataset_bootstrap_repeats": BOOTSTRAP_REPEATS,
    "models": requested_models,
    "datasets": requested_datasets,
    "method": {
        "text_similarity": "cosine",
        "response_similarity": "logfc_hvg__spearman",
        "rsa": "spearman_strict_upper_triangle",
        "gold_neighbor_k": gold_k,
        "ndcg_ks": list(ndcg_ks),
        "retrieval_tie_repeats": tie_repeats,
        "random_ndcg_baseline": "exact_uniform_random_ranking_expectation",
        "excess_ndcg_at_10": "model_minus_exact_random",
        "primary_cross_dataset_summary": "unweighted_dataset_macro_mean",
    },
    "inputs": {
        "embedding_manifest": {
            "path": str(EMB_MANIFEST),
            "sha256": sha256_file(EMB_MANIFEST),
        },
        "model_specs": {
            "path": str(MODEL_SPECS_TSV),
            "sha256": sha256_file(MODEL_SPECS_TSV),
        },
        "embedding_index": {
            "path": str(ATOM_EMB_INDEX),
            "sha256": sha256_file(ATOM_EMB_INDEX),
        },
        "response_similarity_manifest": {
            "path": str(RESP_MANIFEST),
            "sha256": sha256_file(RESP_MANIFEST),
        },
        "response_similarity_audit": {
            "path": str(RESP_AUDIT_TSV),
            "sha256": sha256_file(RESP_AUDIT_TSV),
        },
        "historical_effective_config": {
            "path": str(HIST_EFFECTIVE_CONFIG),
            "sha256": sha256_file(HIST_EFFECTIVE_CONFIG),
        },
    },
    "outputs": {
        "by_dataset": str(BY_DATASET_TSV),
        "model_summary": str(MODEL_SUMMARY_TSV),
        "audit": str(AUDIT_TXT),
    },
}

write_json(
    manifest,
    MANIFEST_JSON,
)

print(
    AUDIT_TXT.read_text(
        encoding="utf-8"
    )
)

if full_run and status != "PASS":
    raise SystemExit(2)
