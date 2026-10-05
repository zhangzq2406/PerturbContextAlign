from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata


def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "R2 cross-context context-only control: compare biology-only "
            "representations with identity-conditioned P2/P4 representations."
        )
    )
    ap.add_argument(
        "--root",
        default=".",
    )
    ap.add_argument(
        "--device",
        default="cuda:0",
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
BATCH_SIZE = 64
BOOTSTRAP_REPEATS = 10000

MODEL_SPECS = [
    {
        "model_key": "bge_m3",
        "display_name": "BGE-M3",
        "group": "general_purpose",
        "model_id": "BAAI/bge-m3",
        "backend": "sentence_transformers",
        "pooling": "sentence_transformers",
        "trust_remote_code": False,
        "dim": 1024,
    },
    {
        "model_key": "qwen3_0_6b",
        "display_name": "Qwen3-Embedding-0.6B",
        "group": "general_purpose",
        "model_id": "Qwen/Qwen3-Embedding-0.6B",
        "backend": "sentence_transformers",
        "pooling": "sentence_transformers",
        "trust_remote_code": True,
        "dim": 1024,
    },
    {
        "model_key": "sapbert",
        "display_name": "SapBERT",
        "group": "biomedical",
        "model_id": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        "backend": "transformers",
        "pooling": "mean",
        "trust_remote_code": False,
        "dim": 768,
    },
    {
        "model_key": "biomedbert",
        "display_name": "BiomedBERT",
        "group": "biomedical",
        "model_id": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        "backend": "transformers",
        "pooling": "mean",
        "trust_remote_code": False,
        "dim": 768,
    },
    {
        "model_key": "medcpt_article",
        "display_name": "MedCPT Article",
        "group": "biomedical",
        "model_id": "ncbi/MedCPT-Article-Encoder",
        "backend": "transformers",
        "pooling": "cls",
        "trust_remote_code": False,
        "dim": 768,
    },
    {
        "model_key": "medcpt_query",
        "display_name": "MedCPT Query",
        "group": "biomedical",
        "model_id": "ncbi/MedCPT-Query-Encoder",
        "backend": "transformers",
        "pooling": "cls",
        "trust_remote_code": False,
        "dim": 768,
    },
]

CROSS_DATASETS = [
    "kaggle_cross_patient",
    "mcfarland_2020",
    "srivatsan_sciplex3",
]

P1P4_COMPONENTS = (
    NEW
    / "06_alignment_metrics"
    / "02_paper_closure_v2"
    / "R2_RESPONSE_ALIGNED_P1P4_COMPONENT_TEXTS_v2.tsv.gz"
)

P1P4_EMB_ROOT = (
    NEW
    / "06_alignment_metrics"
    / "03_p1_p4_ablation_v1"
    / "embeddings"
)

CROSS_ROOT = (
    NEW
    / "07_cross_context"
)

PAIR_AUTH = (
    CROSS_ROOT
    / "01_pair_authority_v1"
    / "R2_WITHIN_CONDITION_CROSS_CONTEXT_PAIRS_v1.tsv.gz"
)

GROUP_AUTH = (
    CROSS_ROOT
    / "01_pair_authority_v1"
    / "R2_WITHIN_CONDITION_CROSS_CONTEXT_GROUPS_v1.tsv"
)

EXISTING_CONDITION_METRICS = (
    CROSS_ROOT
    / "02_alignment_v1"
    / "R2_CROSS_CONTEXT_CONDITION_METRICS_v1.tsv.gz"
)

ATOM_AUTHORITY = (
    NEW
    / "02_input_authority"
    / "R2_RESPONSE_ATOM_AUTHORITY_v1.tsv"
)

RESP_ROOT = (
    NEW
    / "04_response_geometry"
    / "02_similarity_v1"
)

OUT = (
    CROSS_ROOT
    / "03_context_only_control_v1"
)

OUT.mkdir(parents=True, exist_ok=True)

TEXT_AUTH_TSV = (
    OUT
    / "R2_CROSS_CONTEXT_CONTEXT_ONLY_TEXT_AUTHORITY_v1.tsv.gz"
)

CONDITION_METRICS_TSV = (
    OUT
    / "R2_CROSS_CONTEXT_CONTEXT_ONLY_CONDITION_METRICS_v1.tsv.gz"
)

SUMMARY_TSV = (
    OUT
    / "R2_CROSS_CONTEXT_CONTEXT_ONLY_SUMMARY_v1.tsv"
)

IDENTITY_GAIN_TSV = (
    OUT
    / "R2_CROSS_CONTEXT_IDENTITY_CONDITIONING_GAIN_v1.tsv"
)

AUDIT_TXT = (
    OUT
    / "R2_CROSS_CONTEXT_CONTEXT_ONLY_CONTROL_AUDIT_v1.txt"
)

MANIFEST_JSON = (
    OUT
    / "R2_CROSS_CONTEXT_CONTEXT_ONLY_CONTROL_MANIFEST_v1.json"
)

BIO_LABEL = " Biological context:"
DOSE_LABEL = " Dose:"
DURATION_LABEL = " Duration:"


def log(msg: str):
    print(msg, flush=True)


def stable_seed(*parts: Any) -> int:
    raw = "|".join(map(str, parts)).encode("utf-8")
    digest = hashlib.sha256(raw).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)

    return h.hexdigest()


def write_json(obj: Any, path: Path):
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


def l2(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(a, axis=1, keepdims=True)

    if np.any(~np.isfinite(n)) or np.any(n <= 1e-12):
        raise RuntimeError("Invalid embedding norm.")

    return a / n


def mean_pool(last_hidden, mask):
    m = mask.unsqueeze(-1).to(last_hidden.dtype)
    return (
        (last_hidden * m).sum(dim=1)
        / m.sum(dim=1).clamp(min=1e-9)
    )


def cleanup():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class Encoder:
    def __init__(self, spec):
        self.spec = spec
        self.model = None
        self.tokenizer = None

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

        else:
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

    def encode(self, texts):
        if self.spec["backend"] == "sentence_transformers":
            return np.asarray(
                self.model.encode(
                    texts,
                    batch_size=BATCH_SIZE,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                ),
                dtype=np.float32,
            )

        chunks = []

        for start in range(0, len(texts), BATCH_SIZE):
            local = texts[start:start + BATCH_SIZE]

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

                if self.spec["pooling"] == "mean":
                    z = mean_pool(
                        out.last_hidden_state,
                        batch["attention_mask"],
                    )
                else:
                    z = out.last_hidden_state[:, 0]

            chunks.append(
                z.float().cpu().numpy()
            )

        return l2(
            np.concatenate(chunks, axis=0)
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

        cleanup()


def parse_context_only(text: str) -> tuple[str, str]:
    text = str(text)

    b = text.find(BIO_LABEL)
    d = text.find(DOSE_LABEL)
    t = text.find(DURATION_LABEL)

    if min(b, d, t) < 0 or not (b < d < t):
        raise ValueError(
            f"Cannot parse response-aligned P4: {text[:300]}"
        )

    bio = text[
        b + 1:d
    ].strip()

    bio_exp = text[
        b + 1:
    ].strip()

    if not bio.startswith("Biological context:"):
        raise ValueError(bio)

    if "Dose:" not in bio_exp or "Duration:" not in bio_exp:
        raise ValueError(bio_exp)

    return bio, bio_exp


def load_similarity(path: Path):
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


def upper_vec(S):
    A = np.asarray(S, dtype=np.float64)
    return A[np.triu_indices(A.shape[0], 1)]


def safe_spearman(x, y):
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

    x = rankdata(x)
    y = rankdata(y)

    x -= x.mean()
    y -= y.mean()

    den = math.sqrt(
        float(np.dot(x, x))
        * float(np.dot(y, y))
    )

    if den <= 0:
        return np.nan, "invalid_zero_variance"

    return float(np.dot(x, y) / den), "ok"


def top_tie(values):
    v = np.asarray(values, dtype=np.float64)

    vmax = float(np.max(v))

    return set(
        np.where(
            np.isclose(
                v,
                vmax,
                atol=1e-12,
                rtol=0,
            )
        )[0].tolist()
    )


def nearest_gain(text_S, response_S):
    n = text_S.shape[0]

    gains = []

    for i in range(n):
        cand = [
            j
            for j in range(n)
            if j != i
        ]

        if len(cand) < 2:
            continue

        r = np.asarray(
            response_S[i, cand],
            dtype=float,
        )

        t = np.asarray(
            text_S[i, cand],
            dtype=float,
        )

        if np.ptp(r) <= 1e-12:
            continue

        gold = top_tie(r)
        pred = top_tie(t)

        if not gold or not pred:
            continue

        hit = (
            len(
                gold.intersection(pred)
            )
            / len(pred)
        )

        rnd = len(gold) / len(cand)

        gains.append(
            hit - rnd
        )

    if not gains:
        return np.nan, 0

    return (
        float(np.mean(gains)),
        len(gains),
    )


def bootstrap_ci(values, seed):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan, np.nan

    if len(x) == 1:
        return float(x[0]), float(x[0])

    rng = np.random.default_rng(seed)

    idx = rng.integers(
        0,
        len(x),
        size=(
            BOOTSTRAP_REPEATS,
            len(x),
        ),
    )

    means = x[idx].mean(axis=1)

    return (
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


# ======================================================================================
# Inputs
# ======================================================================================

for p in [
    P1P4_COMPONENTS,
    PAIR_AUTH,
    GROUP_AUTH,
    EXISTING_CONDITION_METRICS,
    ATOM_AUTHORITY,
]:
    if not p.is_file():
        raise FileNotFoundError(p)


components = pd.read_csv(
    P1P4_COMPONENTS,
    sep="\t",
    compression="gzip",
    low_memory=False,
)

pairs = pd.read_csv(
    PAIR_AUTH,
    sep="\t",
    compression="gzip",
    low_memory=False,
)

groups = pd.read_csv(
    GROUP_AUTH,
    sep="\t",
    low_memory=False,
)

existing = pd.read_csv(
    EXISTING_CONDITION_METRICS,
    sep="\t",
    compression="gzip",
    low_memory=False,
)

atoms = pd.read_csv(
    ATOM_AUTHORITY,
    sep="\t",
    low_memory=False,
)


for df in [
    components,
    pairs,
    groups,
    existing,
    atoms,
]:
    for col in [
        "dataset_id",
        "condition_id",
        "response_atom_id",
        "response_atom_id_a",
        "response_atom_id_b",
        "model_key",
        "view",
    ]:
        if col in df.columns:
            df[col] = df[col].astype(str)


cross_atom_ids = sorted(
    set(
        pairs["response_atom_id_a"]
    ).union(
        set(
            pairs["response_atom_id_b"]
        )
    )
)


p4 = components.loc[
    components["view"].eq("P4")
    & components["response_atom_id"].isin(
        cross_atom_ids
    )
].copy()


text_rows = []

for _, row in p4.iterrows():
    bio, bio_exp = parse_context_only(
        row["prompt_text"]
    )

    text_rows.append({
        "dataset_id": row["dataset_id"],
        "response_atom_id": row["response_atom_id"],
        "component_id": row["component_id"],
        "B_prompt": bio,
        "BE_prompt": bio_exp,
    })


text_auth = pd.DataFrame(text_rows)


# All components of one atom must collapse to the same context-only prompt.
for col in [
    "B_prompt",
    "BE_prompt",
]:
    bad = (
        text_auth.groupby(
            "response_atom_id",
            observed=True,
        )[col]
        .nunique()
        .ne(1)
        .sum()
    )

    if int(bad) != 0:
        raise RuntimeError(
            f"{col}: component conflict atom_n={bad}"
        )


text_auth.to_csv(
    TEXT_AUTH_TSV,
    sep="\t",
    index=False,
    compression="gzip",
)


atom_context = (
    text_auth[
        [
            "response_atom_id",
            "B_prompt",
            "BE_prompt",
        ]
    ]
    .drop_duplicates(
        "response_atom_id"
    )
    .set_index(
        "response_atom_id"
    )
)


if set(atom_context.index) != set(cross_atom_ids):
    raise RuntimeError(
        "Context-only text authority does not cover cross-context atom namespace."
    )


unique_texts = sorted(
    set(
        atom_context[
            "B_prompt"
        ].astype(str)
    ).union(
        set(
            atom_context[
                "BE_prompt"
            ].astype(str)
        )
    )
)

text_pos = {
    t: i
    for i, t in enumerate(unique_texts)
}


if ARGS.preflight_only:
    for spec in MODEL_SPECS:
        log(
            f"[preflight] {spec['model_key']}"
        )

        enc = Encoder(spec)

        try:
            z = enc.encode(
                unique_texts[
                    :min(
                        4,
                        len(unique_texts),
                    )
                ]
            )

            if z.shape[1] != spec["dim"]:
                raise RuntimeError(
                    f"{spec['model_key']}: dim mismatch"
                )

        finally:
            enc.close()

    print(
        "R2_CROSS_CONTEXT_CONTEXT_ONLY_PREFLIGHT=PASS"
    )

    raise SystemExit(0)


# ======================================================================================
# Encode context-only texts
# ======================================================================================

embedding_by_model = {}

for spec in MODEL_SPECS:
    key = spec["model_key"]

    log(
        f"[encode] {key} unique_text_n={len(unique_texts)}"
    )

    enc = Encoder(spec)

    try:
        z = enc.encode(
            unique_texts
        )
    finally:
        enc.close()

    if z.shape != (
        len(unique_texts),
        spec["dim"],
    ):
        raise RuntimeError(
            f"{key}: embedding shape mismatch"
        )

    embedding_by_model[
        key
    ] = z


# ======================================================================================
# Prepare atom row indices and condition atom sets
# ======================================================================================

condition_atoms = {}

for (
    ds,
    condition_id,
), sub in pairs.groupby(
    [
        "dataset_id",
        "condition_id",
    ],
    sort=True,
):
    ids = sorted(
        set(
            sub[
                "response_atom_id_a"
            ]
        ).union(
            set(
                sub[
                    "response_atom_id_b"
                ]
            )
        )
    )

    condition_atoms[
        (
            ds,
            condition_id,
        )
    ] = ids


# ======================================================================================
# Evaluate B / BE
# ======================================================================================

rows = []

for ds in CROSS_DATASETS:
    log(f"[dataset] {ds}")

    response_S, response_ids = load_similarity(
        RESP_ROOT
        / ds
        / "response_similarity_logfc_hvg_spearman.npz"
    )

    rpos = {
        rid: i
        for i, rid in enumerate(response_ids)
    }

    ds_conditions = sorted(
        condition_id
        for (
            dataset_id,
            condition_id,
        ) in condition_atoms
        if dataset_id == ds
    )

    for spec in MODEL_SPECS:
        key = spec["model_key"]

        zall = embedding_by_model[
            key
        ]

        for condition_id in ds_conditions:
            ids = condition_atoms[
                (
                    ds,
                    condition_id,
                )
            ]

            rix = np.asarray(
                [
                    rpos[rid]
                    for rid in ids
                ],
                dtype=np.int64,
            )

            R = np.asarray(
                response_S[
                    np.ix_(
                        rix,
                        rix,
                    )
                ],
                dtype=np.float32,
            )

            for view, col in [
                (
                    "B",
                    "B_prompt",
                ),
                (
                    "BE",
                    "BE_prompt",
                ),
            ]:
                tids = np.asarray(
                    [
                        text_pos[
                            str(
                                atom_context.loc[
                                    rid,
                                    col,
                                ]
                            )
                        ]
                        for rid in ids
                    ],
                    dtype=np.int64,
                )

                Z = np.asarray(
                    zall[tids],
                    dtype=np.float32,
                )

                S = (
                    Z
                    @ Z.T
                ).astype(
                    np.float32
                )

                S = np.clip(
                    S,
                    -1.0,
                    1.0,
                )

                np.fill_diagonal(
                    S,
                    1.0,
                )

                rho, rho_status = safe_spearman(
                    upper_vec(S),
                    upper_vec(R),
                )

                gain, query_n = nearest_gain(
                    S,
                    R,
                )

                rows.append({
                    "dataset_id": ds,
                    "condition_id": condition_id,
                    "model_key": key,
                    "model_display": spec[
                        "display_name"
                    ],
                    "model_group": spec[
                        "group"
                    ],
                    "view": view,
                    "response_atom_n": len(ids),
                    "context_rsa": rho,
                    "context_rsa_status": rho_status,
                    "nearest_context_query_n": query_n,
                    "nearest_context_hit_gain": gain,
                })


context_only = pd.DataFrame(rows)

context_only.to_csv(
    CONDITION_METRICS_TSV,
    sep="\t",
    index=False,
    compression="gzip",
)


# ======================================================================================
# Summary
# ======================================================================================

summary_rows = []

for (
    ds,
    key,
    view,
), sub in context_only.groupby(
    [
        "dataset_id",
        "model_key",
        "view",
    ],
    sort=False,
):
    rsa = sub[
        "context_rsa"
    ].to_numpy(
        dtype=float
    )

    hit = sub[
        "nearest_context_hit_gain"
    ].to_numpy(
        dtype=float
    )

    rsa = rsa[
        np.isfinite(rsa)
    ]

    hit = hit[
        np.isfinite(hit)
    ]

    rsa_lo, rsa_hi = bootstrap_ci(
        rsa,
        stable_seed(
            SEED,
            ds,
            key,
            view,
            "rsa",
        ),
    )

    hit_lo, hit_hi = bootstrap_ci(
        hit,
        stable_seed(
            SEED,
            ds,
            key,
            view,
            "hit",
        ),
    )

    summary_rows.append({
        "dataset_id": ds,
        "model_key": key,
        "model_display": sub[
            "model_display"
        ].iloc[0],
        "model_group": sub[
            "model_group"
        ].iloc[0],
        "view": view,
        "condition_n_total": int(
            sub[
                "condition_id"
            ].nunique()
        ),
        "context_rsa_valid_condition_n": len(rsa),
        "context_rsa_macro_mean": (
            float(rsa.mean())
            if len(rsa)
            else np.nan
        ),
        "context_rsa_bootstrap_ci_low": rsa_lo,
        "context_rsa_bootstrap_ci_high": rsa_hi,
        "nearest_context_valid_condition_n": len(hit),
        "nearest_context_hit_gain_macro_mean": (
            float(hit.mean())
            if len(hit)
            else np.nan
        ),
        "nearest_context_hit_gain_bootstrap_ci_low": hit_lo,
        "nearest_context_hit_gain_bootstrap_ci_high": hit_hi,
    })


summary = pd.DataFrame(summary_rows)

summary.to_csv(
    SUMMARY_TSV,
    sep="\t",
    index=False,
)


# ======================================================================================
# Identity-conditioning gains: P2-B and P4-BE
# ======================================================================================

existing_sub = existing.loc[
    existing[
        "view"
    ].isin(
        [
            "P2",
            "P4",
        ]
    )
].copy()

context_map = {
    "P2": "B",
    "P4": "BE",
}

gain_rows = []

for target_view, base_view in context_map.items():
    target = existing_sub.loc[
        existing_sub[
            "view"
        ].eq(
            target_view
        )
    ][
        [
            "dataset_id",
            "condition_id",
            "model_key",
            "model_display",
            "model_group",
            "context_rsa",
            "nearest_context_hit_gain",
        ]
    ].rename(
        columns={
            "context_rsa":
                "identity_conditioned_context_rsa",
            "nearest_context_hit_gain":
                "identity_conditioned_hit_gain",
        }
    )

    base = context_only.loc[
        context_only[
            "view"
        ].eq(
            base_view
        )
    ][
        [
            "dataset_id",
            "condition_id",
            "model_key",
            "context_rsa",
            "nearest_context_hit_gain",
        ]
    ].rename(
        columns={
            "context_rsa":
                "context_only_context_rsa",
            "nearest_context_hit_gain":
                "context_only_hit_gain",
        }
    )

    merged = target.merge(
        base,
        on=[
            "dataset_id",
            "condition_id",
            "model_key",
        ],
        how="inner",
        validate="one_to_one",
    )

    merged[
        "delta_rsa_identity_conditioning"
    ] = (
        merged[
            "identity_conditioned_context_rsa"
        ]
        - merged[
            "context_only_context_rsa"
        ]
    )

    merged[
        "delta_hit_gain_identity_conditioning"
    ] = (
        merged[
            "identity_conditioned_hit_gain"
        ]
        - merged[
            "context_only_hit_gain"
        ]
    )

    merged[
        "identity_conditioned_view"
    ] = target_view

    merged[
        "context_only_view"
    ] = base_view

    gain_rows.append(
        merged
    )


gain_long = pd.concat(
    gain_rows,
    ignore_index=True,
)


gain_summary_rows = []

for (
    ds,
    key,
    target_view,
), sub in gain_long.groupby(
    [
        "dataset_id",
        "model_key",
        "identity_conditioned_view",
    ],
    sort=False,
):
    dr = sub[
        "delta_rsa_identity_conditioning"
    ].to_numpy(
        dtype=float
    )

    dh = sub[
        "delta_hit_gain_identity_conditioning"
    ].to_numpy(
        dtype=float
    )

    dr = dr[np.isfinite(dr)]
    dh = dh[np.isfinite(dh)]

    rlo, rhi = bootstrap_ci(
        dr,
        stable_seed(
            SEED,
            ds,
            key,
            target_view,
            "identity_rsa_gain",
        ),
    )

    hlo, hhi = bootstrap_ci(
        dh,
        stable_seed(
            SEED,
            ds,
            key,
            target_view,
            "identity_hit_gain",
        ),
    )

    gain_summary_rows.append({
        "dataset_id": ds,
        "model_key": key,
        "model_display": sub[
            "model_display"
        ].iloc[0],
        "model_group": sub[
            "model_group"
        ].iloc[0],
        "identity_conditioned_view": target_view,
        "context_only_view": sub[
            "context_only_view"
        ].iloc[0],
        "rsa_paired_condition_n": len(dr),
        "delta_rsa_identity_conditioning_mean": (
            float(dr.mean())
            if len(dr)
            else np.nan
        ),
        "delta_rsa_identity_conditioning_positive_condition_n": int(
            (dr > 0).sum()
        ),
        "delta_rsa_bootstrap_ci_low": rlo,
        "delta_rsa_bootstrap_ci_high": rhi,
        "hit_paired_condition_n": len(dh),
        "delta_hit_gain_identity_conditioning_mean": (
            float(dh.mean())
            if len(dh)
            else np.nan
        ),
        "delta_hit_gain_identity_conditioning_positive_condition_n": int(
            (dh > 0).sum()
        ),
        "delta_hit_gain_bootstrap_ci_low": hlo,
        "delta_hit_gain_bootstrap_ci_high": hhi,
    })


gain_summary = pd.DataFrame(
    gain_summary_rows
)

gain_summary.to_csv(
    IDENTITY_GAIN_TSV,
    sep="\t",
    index=False,
)


# ======================================================================================
# Audit
# ======================================================================================

lines = [
    "PERTURBCONTEXTALIGN R2 CROSS-CONTEXT CONTEXT-ONLY CONTROL AUDIT v1",
    "=" * 120,
    "",
    "QUESTION",
    "-" * 120,
    "Does identity-conditioned biological-context text outperform generic biological-context text",
    "when perturbation/exposure condition is fixed?",
    "",
    "VIEWS",
    "-" * 120,
    "B=Biological context only",
    "BE=Biological context + fixed dose/duration exposure; perturbation identity omitted",
    "P2=identity + biological context",
    "P4=identity + biological context + exposure",
    "",
    "PRIMARY COMPARISONS",
    "-" * 120,
    "P2_minus_B=identity conditioning of biological-context representation",
    "P4_minus_BE=identity conditioning with exposure present",
    "evaluation_unit=condition",
    "",
    "CONTEXT-ONLY SUMMARY",
    "-" * 120,
    summary.to_string(index=False),
    "",
    "IDENTITY-CONDITIONING GAIN",
    "-" * 120,
    gain_summary.to_string(index=False),
    "",
    "STATUS",
    "-" * 120,
    "R2_CROSS_CONTEXT_CONTEXT_ONLY_CONTROL=PASS",
    "NEXT=DECIDE_MAIN_VS_SUPPLEMENT_AND_FREEZE_CROSS_CONTEXT_STAGE",
]

AUDIT_TXT.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)

write_json(
    {
        "version":
            "R2_CROSS_CONTEXT_CONTEXT_ONLY_CONTROL_v1",
        "status":
            "PASS",
        "views": {
            "B":
                "Biological context only",
            "BE":
                "Biological context + exposure, no identity",
            "P2":
                "identity + biological context",
            "P4":
                "identity + biological context + exposure",
        },
        "comparisons": [
            "P2-B",
            "P4-BE",
        ],
        "inputs": {
            "components_sha256":
                sha256_file(
                    P1P4_COMPONENTS
                ),
            "pair_authority_sha256":
                sha256_file(
                    PAIR_AUTH
                ),
            "existing_condition_metrics_sha256":
                sha256_file(
                    EXISTING_CONDITION_METRICS
                ),
        },
        "outputs": {
            "text_authority":
                str(
                    TEXT_AUTH_TSV
                ),
            "condition_metrics":
                str(
                    CONDITION_METRICS_TSV
                ),
            "summary":
                str(
                    SUMMARY_TSV
                ),
            "identity_conditioning_gain":
                str(
                    IDENTITY_GAIN_TSV
                ),
            "audit":
                str(
                    AUDIT_TXT
                ),
        },
    },
    MANIFEST_JSON,
)

print(
    AUDIT_TXT.read_text(
        encoding="utf-8"
    )
)
