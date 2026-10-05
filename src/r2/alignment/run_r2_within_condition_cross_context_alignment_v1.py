from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata


# ======================================================================================
# CLI / constants
# ======================================================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "PerturbContextAlign R2 within-condition cross-context alignment. "
            "Condition is the primary evaluation unit."
        )
    )
    ap.add_argument(
        "--root",
        default=".",
    )
    return ap.parse_args()


ARGS = parse_args()

ROOT = Path(ARGS.root).expanduser().resolve()

NEW = (
    ROOT
    / "0_ProjectCodeAndLogs"
    / "PerturbContextAlign_R2_corrected_v1_20261002"
)

SEED = 20260711
BOOTSTRAP_REPEATS = 10000

VIEWS = ["P1", "P2", "P3", "P4"]

MODEL_SPECS = [
    ("bge_m3", "BGE-M3", "general_purpose"),
    ("qwen3_0_6b", "Qwen3-Embedding-0.6B", "general_purpose"),
    ("sapbert", "SapBERT", "biomedical"),
    ("biomedbert", "BiomedBERT", "biomedical"),
    ("medcpt_article", "MedCPT Article", "biomedical"),
    ("medcpt_query", "MedCPT Query", "biomedical"),
]

EXPECTED_DATASETS = [
    "kaggle_cross_patient",
    "mcfarland_2020",
    "srivatsan_sciplex3",
]


# ======================================================================================
# Paths
# ======================================================================================

FEAS = (
    NEW
    / "07_cross_context"
    / "00_feasibility_audit_v1"
)

FEAS_AUDIT = (
    FEAS
    / "R2_WITHIN_CONDITION_CROSS_CONTEXT_FEASIBILITY_AUDIT_v1.txt"
)

PAIR_INPUT = (
    FEAS
    / "R2_WITHIN_CONDITION_CROSS_CONTEXT_PAIRS_v1.tsv.gz"
)

GROUP_INPUT = (
    FEAS
    / "R2_WITHIN_CONDITION_CROSS_CONTEXT_GROUPS_v1.tsv"
)

DATASET_INPUT = (
    FEAS
    / "R2_WITHIN_CONDITION_CROSS_CONTEXT_DATASET_SUMMARY_v1.tsv"
)

TEXT_QC_INPUT = (
    FEAS
    / "R2_WITHIN_CONDITION_TEXT_INVARIANCE_v1.tsv"
)

ATOM_AUTHORITY = (
    NEW
    / "02_input_authority"
    / "R2_RESPONSE_ATOM_AUTHORITY_v1.tsv"
)

P1P4_ROOT = (
    NEW
    / "06_alignment_metrics"
    / "03_p1_p4_ablation_v1"
)

P1P4_AUDIT = (
    P1P4_ROOT
    / "R2_CORRECTED_P1_P4_ABLATION_AUDIT_v1.txt"
)

EMB_ROOT = P1P4_ROOT / "embeddings"

RESP_ROOT = (
    NEW
    / "04_response_geometry"
    / "02_similarity_v1"
)

PAIR_AUTH_OUT = (
    NEW
    / "07_cross_context"
    / "01_pair_authority_v1"
)

ALIGN_OUT = (
    NEW
    / "07_cross_context"
    / "02_alignment_v1"
)

PAIR_AUTH_OUT.mkdir(parents=True, exist_ok=True)
ALIGN_OUT.mkdir(parents=True, exist_ok=True)

CONDITION_METRICS_TSV = (
    ALIGN_OUT
    / "R2_CROSS_CONTEXT_CONDITION_METRICS_v1.tsv.gz"
)

DATASET_MODEL_VIEW_TSV = (
    ALIGN_OUT
    / "R2_CROSS_CONTEXT_DATASET_MODEL_VIEW_SUMMARY_v1.tsv"
)

P4_MINUS_P2_TSV = (
    ALIGN_OUT
    / "R2_CROSS_CONTEXT_P4_MINUS_P2_v1.tsv"
)

MODEL_MACRO_TSV = (
    ALIGN_OUT
    / "R2_CROSS_CONTEXT_MODEL_MACRO_DESCRIPTIVE_v1.tsv"
)

PAIR_POOLED_SECONDARY_TSV = (
    ALIGN_OUT
    / "R2_CROSS_CONTEXT_PAIR_POOLED_SECONDARY_v1.tsv"
)

AUDIT_TXT = (
    ALIGN_OUT
    / "R2_WITHIN_CONDITION_CROSS_CONTEXT_ALIGNMENT_AUDIT_v1.txt"
)

MANIFEST_JSON = (
    ALIGN_OUT
    / "R2_WITHIN_CONDITION_CROSS_CONTEXT_ALIGNMENT_MANIFEST_v1.json"
)


# ======================================================================================
# Helpers
# ======================================================================================

def log(msg: str) -> None:
    print(msg, flush=True)


def sha256_file(path: Path, chunk: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)

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
    with np.load(path, allow_pickle=True) as z:
        key = "S" if "S" in z.files else "X"

        S = np.asarray(
            z[key],
            dtype=np.float32,
        )

        id_key = (
            "response_atom_ids"
            if "response_atom_ids" in z.files
            else "condition_ids"
        )

        ids = np.asarray(
            z[id_key]
        ).astype(str)

    return S, ids


def upper_vec(S: np.ndarray) -> np.ndarray:
    A = np.asarray(S, dtype=np.float64)
    idx = np.triu_indices(A.shape[0], k=1)
    return A[idx]


def safe_spearman(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, str]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    keep = np.isfinite(x) & np.isfinite(y)

    x = x[keep]
    y = y[keep]

    if len(x) < 3:
        return np.nan, "invalid_too_few_pairs"

    if np.ptp(x) <= 1e-12:
        return np.nan, "invalid_constant_text"

    if np.ptp(y) <= 1e-12:
        return np.nan, "invalid_constant_response"

    rx = rankdata(
        x,
        method="average",
    )

    ry = rankdata(
        y,
        method="average",
    )

    rx -= rx.mean()
    ry -= ry.mean()

    denom = math.sqrt(
        float(np.dot(rx, rx))
        * float(np.dot(ry, ry))
    )

    if denom <= 0 or not np.isfinite(denom):
        return np.nan, "invalid_zero_variance"

    return (
        float(np.dot(rx, ry) / denom),
        "ok",
    )


def top_tie_set(
    values: np.ndarray,
    atol: float = 1e-12,
) -> set[int]:
    v = np.asarray(values, dtype=np.float64)

    if len(v) == 0:
        return set()

    finite = np.isfinite(v)

    if not finite.any():
        return set()

    vmax = float(np.max(v[finite]))

    return set(
        np.where(
            finite
            & np.isclose(
                v,
                vmax,
                atol=atol,
                rtol=0.0,
            )
        )[0].tolist()
    )


def nearest_context_gain(
    text_S: np.ndarray,
    response_S: np.ndarray,
) -> dict[str, Any]:
    """
    Query-level nearest-context recovery with analytic tie handling.

    Gold = all candidates tied for maximal measured-response similarity.
    Prediction = all candidates tied for maximal text similarity.

    If the text top is tied, expected Hit@1 under uniform selection from the
    text-top tie block is |T ∩ G| / |T|.

    Exact random expectation is |G| / (n_candidates).

    Queries with a completely flat response truth are uninformative and remain
    excluded rather than converted to zero.
    """
    A = np.asarray(text_S, dtype=np.float64)
    R = np.asarray(response_S, dtype=np.float64)

    n = A.shape[0]

    hits = []
    randoms = []

    for i in range(n):
        cand = [
            j
            for j in range(n)
            if j != i
        ]

        if len(cand) < 2:
            # With one candidate, random accuracy is 1 and there is no
            # discriminative context-ranking problem.
            continue

        r = R[i, cand]
        t = A[i, cand]

        if np.ptp(r) <= 1e-12:
            continue

        gold_local = top_tie_set(r)
        pred_local = top_tie_set(t)

        if not gold_local or not pred_local:
            continue

        hit = (
            len(
                gold_local.intersection(
                    pred_local
                )
            )
            / len(pred_local)
        )

        random_expectation = (
            len(gold_local)
            / len(cand)
        )

        hits.append(float(hit))
        randoms.append(
            float(random_expectation)
        )

    if not hits:
        return {
            "status": "invalid_no_informative_queries",
            "query_n": 0,
            "hit_at_1": np.nan,
            "random_hit_at_1": np.nan,
            "hit_gain": np.nan,
        }

    h = np.asarray(hits, dtype=np.float64)
    r = np.asarray(randoms, dtype=np.float64)

    return {
        "status": "ok",
        "query_n": int(len(h)),
        "hit_at_1": float(h.mean()),
        "random_hit_at_1": float(r.mean()),
        "hit_gain": float(
            (h - r).mean()
        ),
    }


def bootstrap_ci(
    values: np.ndarray,
    seed: int,
    repeats: int = BOOTSTRAP_REPEATS,
) -> tuple[float, float]:
    x = np.asarray(
        values,
        dtype=np.float64,
    )

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
            int(repeats),
            len(x),
        ),
    )

    means = x[idx].mean(axis=1)

    return (
        float(
            np.quantile(
                means,
                0.025,
            )
        ),
        float(
            np.quantile(
                means,
                0.975,
            )
        ),
    )


def context_axis_from_pairs(
    sub: pd.DataFrame,
) -> str:
    combos = sorted(
        set(
            sub[
                "bio_fields_different"
            ].astype(str)
        )
    )

    if combos == ["cell_type"]:
        return "cell_type"

    if combos == ["cell_line"]:
        return "cell_line"

    if combos == ["cell_type;cell_line"]:
        return "cell_type+cell_line_composite"

    return "|".join(combos)


# ======================================================================================
# Upstream checks
# ======================================================================================

for p in [
    FEAS_AUDIT,
    PAIR_INPUT,
    GROUP_INPUT,
    DATASET_INPUT,
    TEXT_QC_INPUT,
    ATOM_AUTHORITY,
    P1P4_AUDIT,
]:
    if not p.is_file():
        raise FileNotFoundError(p)


if (
    "R2_WITHIN_CONDITION_CROSS_CONTEXT_FEASIBILITY=PASS"
    not in FEAS_AUDIT.read_text(
        encoding="utf-8",
        errors="replace",
    )
):
    raise RuntimeError(
        "Cross-context feasibility audit is not PASS."
    )


if (
    "R2_CORRECTED_P1_P4_ABLATION=PASS"
    not in P1P4_AUDIT.read_text(
        encoding="utf-8",
        errors="replace",
    )
):
    raise RuntimeError(
        "Corrected P1-P4 ablation is not PASS."
    )


pairs = pd.read_csv(
    PAIR_INPUT,
    sep="\t",
    compression="gzip",
    low_memory=False,
)

groups = pd.read_csv(
    GROUP_INPUT,
    sep="\t",
    low_memory=False,
)

dataset_feas = pd.read_csv(
    DATASET_INPUT,
    sep="\t",
    low_memory=False,
)

atoms = pd.read_csv(
    ATOM_AUTHORITY,
    sep="\t",
    low_memory=False,
)


for col in [
    "dataset_id",
    "condition_id",
    "response_atom_id_a",
    "response_atom_id_b",
]:
    if col in pairs.columns:
        pairs[col] = pairs[col].astype(str)


for col in [
    "dataset_id",
    "condition_id",
]:
    groups[col] = groups[col].astype(str)


atoms["dataset_id"] = atoms["dataset_id"].astype(str)
atoms["condition_id"] = atoms["condition_id"].astype(str)
atoms["response_atom_id"] = atoms["response_atom_id"].astype(str)


cross_datasets = (
    dataset_feas.loc[
        dataset_feas[
            "cross_context_condition_n"
        ] > 0,
        "dataset_id",
    ]
    .astype(str)
    .tolist()
)


if set(cross_datasets) != set(EXPECTED_DATASETS):
    raise RuntimeError(
        "Unexpected cross-context dataset set: "
        f"{cross_datasets}"
    )


# ======================================================================================
# Freeze pair authority
# ======================================================================================

log("step=freeze_pair_authority")

for src in [
    PAIR_INPUT,
    GROUP_INPUT,
    DATASET_INPUT,
    TEXT_QC_INPUT,
    FEAS_AUDIT,
]:
    shutil.copy2(
        src,
        PAIR_AUTH_OUT / src.name,
    )


pair_authority_manifest = {
    "version": "R2_CROSS_CONTEXT_PAIR_AUTHORITY_v1",
    "status": "FROZEN",
    "pair_definition": (
        "same dataset + same condition_id + "
        "different frozen biological-context key"
    ),
    "bio_fields": [
        "species",
        "cell_type",
        "cell_line",
        "tissue",
        "disease",
    ],
    "datasets": cross_datasets,
    "condition_n": int(
        dataset_feas[
            "cross_context_condition_n"
        ].sum()
    ),
    "pair_n": int(
        dataset_feas[
            "within_condition_cross_context_pair_n"
        ].sum()
    ),
    "inputs": {
        src.name: {
            "sha256": sha256_file(src),
        }
        for src in [
            PAIR_INPUT,
            GROUP_INPUT,
            DATASET_INPUT,
            TEXT_QC_INPUT,
            FEAS_AUDIT,
        ]
    },
}

write_json(
    pair_authority_manifest,
    PAIR_AUTH_OUT
    / "R2_CROSS_CONTEXT_PAIR_AUTHORITY_MANIFEST_v1.json",
)

(
    PAIR_AUTH_OUT
    / "R2_CROSS_CONTEXT_PAIR_AUTHORITY_FROZEN_v1.txt"
).write_text(
    "R2_CROSS_CONTEXT_PAIR_AUTHORITY=FROZEN\n",
    encoding="utf-8",
)


# ======================================================================================
# Index authorities / embeddings
# ======================================================================================

atom_ids = atoms["response_atom_id"].tolist()

if len(atom_ids) != len(set(atom_ids)):
    raise RuntimeError(
        "Response atom authority IDs are not unique."
    )

atom_pos = {
    rid: i
    for i, rid in enumerate(atom_ids)
}


model_paths = {}

for model_key, _, _ in MODEL_SPECS:
    model_paths[model_key] = {}

    for view in VIEWS:
        p = (
            EMB_ROOT
            / model_key
            / f"{view}_response_atom_embeddings.npy"
        )

        if not p.is_file():
            raise FileNotFoundError(p)

        arr = np.load(
            p,
            mmap_mode="r",
        )

        if arr.shape[0] != len(atom_ids):
            raise RuntimeError(
                f"{model_key}/{view}: embedding row count mismatch."
            )

        model_paths[
            model_key
        ][
            view
        ] = p


# ======================================================================================
# Primary condition-level metrics
# ======================================================================================

log("step=condition_level_alignment")

condition_rows = []
pooled_rows = []


for ds in cross_datasets:
    log(f"[dataset] {ds}")

    ds_pairs = pairs.loc[
        pairs["dataset_id"].eq(ds)
    ].copy()

    ds_groups = groups.loc[
        groups["dataset_id"].eq(ds)
    ].copy()

    context_axis = context_axis_from_pairs(
        ds_pairs
    )

    response_path = (
        RESP_ROOT
        / ds
        / "response_similarity_logfc_hvg_spearman.npz"
    )

    response_S, response_ids = load_similarity_bundle(
        response_path
    )

    response_pos = {
        rid: i
        for i, rid in enumerate(
            response_ids
        )
    }

    # Frozen cross-context atoms in each condition.
    condition_atoms = {}

    for condition_id, sub in ds_pairs.groupby(
        "condition_id",
        sort=True,
    ):
        ids = sorted(
            set(
                sub[
                    "response_atom_id_a"
                ].astype(str)
            ).union(
                set(
                    sub[
                        "response_atom_id_b"
                    ].astype(str)
                )
            )
        )

        condition_atoms[
            str(condition_id)
        ] = ids

    expected_conditions = set(
        ds_groups[
            "condition_id"
        ].astype(str)
    )

    if set(condition_atoms) != expected_conditions:
        raise RuntimeError(
            f"{ds}: pair/group condition namespace mismatch."
        )

    for model_key, model_display, model_group in MODEL_SPECS:
        embeddings = {
            view: np.load(
                model_paths[
                    model_key
                ][
                    view
                ],
                mmap_mode="r",
            )
            for view in VIEWS
        }

        # Secondary pooled vectors across all condition-pairs.
        pooled_response = []
        pooled_text = {
            view: []
            for view in VIEWS
        }

        for condition_id in sorted(
            condition_atoms
        ):
            ids = condition_atoms[
                condition_id
            ]

            n_context = len(ids)

            group_row = ds_groups.loc[
                ds_groups[
                    "condition_id"
                ].eq(condition_id)
            ]

            if len(group_row) != 1:
                raise RuntimeError(
                    f"{ds}/{condition_id}: expected one group authority row."
                )

            expected_n = int(
                group_row.iloc[0][
                    "response_atom_n"
                ]
            )

            if n_context != expected_n:
                raise RuntimeError(
                    f"{ds}/{condition_id}: atom count {n_context} "
                    f"!= authority {expected_n}"
                )

            missing_response = [
                rid
                for rid in ids
                if rid not in response_pos
            ]

            if missing_response:
                raise RuntimeError(
                    f"{ds}/{condition_id}: atoms absent from response similarity."
                )

            missing_text = [
                rid
                for rid in ids
                if rid not in atom_pos
            ]

            if missing_text:
                raise RuntimeError(
                    f"{ds}/{condition_id}: atoms absent from text embedding authority."
                )

            rix = np.asarray(
                [
                    response_pos[rid]
                    for rid in ids
                ],
                dtype=np.int64,
            )

            aix = np.asarray(
                [
                    atom_pos[rid]
                    for rid in ids
                ],
                dtype=np.int64,
            )

            response_sub = np.asarray(
                response_S[
                    np.ix_(
                        rix,
                        rix,
                    )
                ],
                dtype=np.float32,
            )

            response_upper = upper_vec(
                response_sub
            )

            for view in VIEWS:
                Z = np.asarray(
                    embeddings[
                        view
                    ][
                        aix
                    ],
                    dtype=np.float32,
                )

                text_sub = (
                    Z
                    @ Z.T
                ).astype(
                    np.float32
                )

                text_sub = np.clip(
                    text_sub,
                    -1.0,
                    1.0,
                )

                np.fill_diagonal(
                    text_sub,
                    1.0,
                )

                text_upper = upper_vec(
                    text_sub
                )

                rsa, rsa_status = safe_spearman(
                    text_upper,
                    response_upper,
                )

                local = nearest_context_gain(
                    text_sub,
                    response_sub,
                )

                # P1/P3 should be biologically invariant within a condition.
                # Numerical cosine drift is tolerated at the 1e-5 scale.
                invariant_deviation = float(
                    np.max(
                        np.abs(
                            text_upper
                            - 1.0
                        )
                    )
                ) if len(text_upper) else np.nan

                if (
                    view in {"P1", "P3"}
                    and np.isfinite(
                        invariant_deviation
                    )
                    and invariant_deviation > 1e-5
                ):
                    raise RuntimeError(
                        f"{ds}/{condition_id}/{model_key}/{view}: "
                        f"invariance deviation={invariant_deviation}"
                    )

                condition_rows.append({
                    "dataset_id": ds,
                    "context_axis": context_axis,
                    "condition_id": condition_id,
                    "response_atom_n": n_context,
                    "context_pair_n": int(
                        n_context
                        * (
                            n_context - 1
                        )
                        // 2
                    ),
                    "model_key": model_key,
                    "model_display": model_display,
                    "model_group": model_group,
                    "view": view,
                    "context_rsa": rsa,
                    "context_rsa_status": rsa_status,
                    "nearest_context_query_n": int(
                        local[
                            "query_n"
                        ]
                    ),
                    "nearest_context_hit_at_1": (
                        float(
                            local[
                                "hit_at_1"
                            ]
                        )
                        if np.isfinite(
                            local[
                                "hit_at_1"
                            ]
                        )
                        else np.nan
                    ),
                    "nearest_context_random_hit_at_1": (
                        float(
                            local[
                                "random_hit_at_1"
                            ]
                        )
                        if np.isfinite(
                            local[
                                "random_hit_at_1"
                            ]
                        )
                        else np.nan
                    ),
                    "nearest_context_hit_gain": (
                        float(
                            local[
                                "hit_gain"
                            ]
                        )
                        if np.isfinite(
                            local[
                                "hit_gain"
                            ]
                        )
                        else np.nan
                    ),
                    "nearest_context_status": local[
                        "status"
                    ],
                    "invariant_view_max_offdiag_deviation_from_1": (
                        invariant_deviation
                        if view in {"P1", "P3"}
                        else np.nan
                    ),
                })

                pooled_text[
                    view
                ].extend(
                    text_upper.tolist()
                )

            pooled_response.extend(
                response_upper.tolist()
            )

        # Pair-pooled metric is explicitly secondary because conditions have
        # unequal numbers of biological contexts/pairs.
        pooled_response_arr = np.asarray(
            pooled_response,
            dtype=np.float64,
        )

        for view in VIEWS:
            pooled_text_arr = np.asarray(
                pooled_text[
                    view
                ],
                dtype=np.float64,
            )

            rho, rho_status = safe_spearman(
                pooled_text_arr,
                pooled_response_arr,
            )

            pooled_rows.append({
                "dataset_id": ds,
                "context_axis": context_axis,
                "model_key": model_key,
                "model_display": model_display,
                "model_group": model_group,
                "view": view,
                "pair_n": int(
                    len(
                        pooled_response_arr
                    )
                ),
                "pair_pooled_spearman_secondary": rho,
                "status": rho_status,
                "interpretation": (
                    "secondary_only_unequal_condition_pair_counts"
                ),
            })


condition_metrics = pd.DataFrame(
    condition_rows
)

condition_metrics.to_csv(
    CONDITION_METRICS_TSV,
    sep="\t",
    index=False,
    compression="gzip",
)

pair_pooled = pd.DataFrame(
    pooled_rows
)

pair_pooled.to_csv(
    PAIR_POOLED_SECONDARY_TSV,
    sep="\t",
    index=False,
)


# ======================================================================================
# Condition-balanced dataset summaries
# ======================================================================================

log("step=condition_balanced_summaries")

summary_rows = []

for (
    ds,
    model_key,
    view,
), sub in condition_metrics.groupby(
    [
        "dataset_id",
        "model_key",
        "view",
    ],
    sort=False,
):
    first = sub.iloc[0]

    rsa_values = sub[
        "context_rsa"
    ].to_numpy(
        dtype=np.float64
    )

    hit_gain_values = sub[
        "nearest_context_hit_gain"
    ].to_numpy(
        dtype=np.float64
    )

    valid_rsa = rsa_values[
        np.isfinite(
            rsa_values
        )
    ]

    valid_hit = hit_gain_values[
        np.isfinite(
            hit_gain_values
        )
    ]

    rsa_lo, rsa_hi = bootstrap_ci(
        valid_rsa,
        seed=stable_seed(
            SEED,
            ds,
            model_key,
            view,
            "context_rsa",
        ),
    )

    hit_lo, hit_hi = bootstrap_ci(
        valid_hit,
        seed=stable_seed(
            SEED,
            ds,
            model_key,
            view,
            "nearest_context_hit_gain",
        ),
    )

    summary_rows.append({
        "dataset_id": ds,
        "context_axis": first[
            "context_axis"
        ],
        "model_key": model_key,
        "model_display": first[
            "model_display"
        ],
        "model_group": first[
            "model_group"
        ],
        "view": view,
        "condition_n_total": int(
            sub[
                "condition_id"
            ].nunique()
        ),
        "context_rsa_valid_condition_n": int(
            len(
                valid_rsa
            )
        ),
        "context_rsa_macro_mean": (
            float(
                valid_rsa.mean()
            )
            if len(
                valid_rsa
            )
            else np.nan
        ),
        "context_rsa_macro_median": (
            float(
                np.median(
                    valid_rsa
                )
            )
            if len(
                valid_rsa
            )
            else np.nan
        ),
        "context_rsa_positive_condition_n": int(
            (
                valid_rsa > 0
            ).sum()
        ),
        "context_rsa_bootstrap_ci_low": rsa_lo,
        "context_rsa_bootstrap_ci_high": rsa_hi,
        "nearest_context_valid_condition_n": int(
            len(
                valid_hit
            )
        ),
        "nearest_context_hit_gain_macro_mean": (
            float(
                valid_hit.mean()
            )
            if len(
                valid_hit
            )
            else np.nan
        ),
        "nearest_context_hit_gain_macro_median": (
            float(
                np.median(
                    valid_hit
                )
            )
            if len(
                valid_hit
            )
            else np.nan
        ),
        "nearest_context_hit_gain_positive_condition_n": int(
            (
                valid_hit > 0
            ).sum()
        ),
        "nearest_context_hit_gain_bootstrap_ci_low": hit_lo,
        "nearest_context_hit_gain_bootstrap_ci_high": hit_hi,
    })


summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    DATASET_MODEL_VIEW_TSV,
    sep="\t",
    index=False,
)


# ======================================================================================
# P4 - P2 within-condition paired changes
# ======================================================================================

log("step=p4_minus_p2")

delta_rows = []

for (
    ds,
    model_key,
), sub in condition_metrics.loc[
    condition_metrics[
        "view"
    ].isin(
        [
            "P2",
            "P4",
        ]
    )
].groupby(
    [
        "dataset_id",
        "model_key",
    ],
    sort=False,
):
    pivot_rsa = sub.pivot(
        index="condition_id",
        columns="view",
        values="context_rsa",
    )

    pivot_hit = sub.pivot(
        index="condition_id",
        columns="view",
        values="nearest_context_hit_gain",
    )

    rsa_delta = (
        pivot_rsa[
            "P4"
        ]
        - pivot_rsa[
            "P2"
        ]
    )

    hit_delta = (
        pivot_hit[
            "P4"
        ]
        - pivot_hit[
            "P2"
        ]
    )

    rsa_valid = rsa_delta[
        np.isfinite(
            rsa_delta
        )
    ].to_numpy(
        dtype=float
    )

    hit_valid = hit_delta[
        np.isfinite(
            hit_delta
        )
    ].to_numpy(
        dtype=float
    )

    rsa_lo, rsa_hi = bootstrap_ci(
        rsa_valid,
        seed=stable_seed(
            SEED,
            ds,
            model_key,
            "P4-P2",
            "context_rsa",
        ),
    )

    hit_lo, hit_hi = bootstrap_ci(
        hit_valid,
        seed=stable_seed(
            SEED,
            ds,
            model_key,
            "P4-P2",
            "nearest_context_hit_gain",
        ),
    )

    model_display = (
        sub[
            "model_display"
        ].iloc[0]
    )

    model_group = (
        sub[
            "model_group"
        ].iloc[0]
    )

    context_axis = (
        sub[
            "context_axis"
        ].iloc[0]
    )

    delta_rows.append({
        "dataset_id": ds,
        "context_axis": context_axis,
        "model_key": model_key,
        "model_display": model_display,
        "model_group": model_group,
        "rsa_paired_condition_n": len(
            rsa_valid
        ),
        "P4_minus_P2_context_rsa_mean": (
            float(
                rsa_valid.mean()
            )
            if len(
                rsa_valid
            )
            else np.nan
        ),
        "P4_minus_P2_context_rsa_median": (
            float(
                np.median(
                    rsa_valid
                )
            )
            if len(
                rsa_valid
            )
            else np.nan
        ),
        "P4_minus_P2_context_rsa_positive_condition_n": int(
            (
                rsa_valid > 0
            ).sum()
        ),
        "P4_minus_P2_context_rsa_bootstrap_ci_low": rsa_lo,
        "P4_minus_P2_context_rsa_bootstrap_ci_high": rsa_hi,
        "hit_paired_condition_n": len(
            hit_valid
        ),
        "P4_minus_P2_nearest_hit_gain_mean": (
            float(
                hit_valid.mean()
            )
            if len(
                hit_valid
            )
            else np.nan
        ),
        "P4_minus_P2_nearest_hit_gain_median": (
            float(
                np.median(
                    hit_valid
                )
            )
            if len(
                hit_valid
            )
            else np.nan
        ),
        "P4_minus_P2_nearest_hit_gain_positive_condition_n": int(
            (
                hit_valid > 0
            ).sum()
        ),
        "P4_minus_P2_nearest_hit_gain_bootstrap_ci_low": hit_lo,
        "P4_minus_P2_nearest_hit_gain_bootstrap_ci_high": hit_hi,
    })


p4_minus_p2 = pd.DataFrame(
    delta_rows
)

p4_minus_p2.to_csv(
    P4_MINUS_P2_TSV,
    sep="\t",
    index=False,
)


# ======================================================================================
# Three-dataset descriptive macro
# ======================================================================================

log("step=three_dataset_descriptive_macro")

macro_rows = []

for (
    model_key,
    view,
), sub in summary.groupby(
    [
        "model_key",
        "view",
    ],
    sort=False,
):
    macro_rows.append({
        "model_key": model_key,
        "model_display": sub[
            "model_display"
        ].iloc[0],
        "model_group": sub[
            "model_group"
        ].iloc[0],
        "view": view,
        "dataset_n": int(
            sub[
                "dataset_id"
            ].nunique()
        ),
        "context_rsa_dataset_macro_mean_descriptive": float(
            sub[
                "context_rsa_macro_mean"
            ].mean(
                skipna=True
            )
        ),
        "nearest_context_hit_gain_dataset_macro_mean_descriptive": float(
            sub[
                "nearest_context_hit_gain_macro_mean"
            ].mean(
                skipna=True
            )
        ),
        "interpretation": (
            "descriptive_equal_dataset_macro_only_n3"
        ),
    })


model_macro = pd.DataFrame(
    macro_rows
)

model_macro.to_csv(
    MODEL_MACRO_TSV,
    sep="\t",
    index=False,
)


# ======================================================================================
# QC / audit
# ======================================================================================

# P1/P3 should have no valid context RSA because those views are invariant
# within each fixed intervention/exposure condition.
negative_control = summary.loc[
    summary[
        "view"
    ].isin(
        [
            "P1",
            "P3",
        ]
    )
]

negative_rsa_valid_n = int(
    negative_control[
        "context_rsa_valid_condition_n"
    ].sum()
)

max_invariance_deviation = float(
    condition_metrics.loc[
        condition_metrics[
            "view"
        ].isin(
            [
                "P1",
                "P3",
            ]
        ),
        "invariant_view_max_offdiag_deviation_from_1",
    ]
    .dropna()
    .max()
)


# P1/P3 local gain is expected to be ~0 under analytic tie handling.
negative_hit_gain_max_abs = float(
    np.nanmax(
        np.abs(
            summary.loc[
                summary[
                    "view"
                ].isin(
                    [
                        "P1",
                        "P3",
                    ]
                ),
                "nearest_context_hit_gain_macro_mean",
            ].to_numpy(
                dtype=float
            )
        )
    )
)


if negative_rsa_valid_n != 0:
    raise RuntimeError(
        f"P1/P3 unexpectedly yielded valid context RSA n={negative_rsa_valid_n}"
    )


if max_invariance_deviation > 1e-5:
    raise RuntimeError(
        f"P1/P3 embedding invariance deviation too large: "
        f"{max_invariance_deviation}"
    )


if negative_hit_gain_max_abs > 1e-8:
    raise RuntimeError(
        "P1/P3 nearest-context gain is not zero under tie-aware analytic scoring: "
        f"{negative_hit_gain_max_abs}"
    )


lines = [
    "PERTURBCONTEXTALIGN R2 WITHIN-CONDITION CROSS-CONTEXT ALIGNMENT AUDIT v1",
    "=" * 120,
    "",
    "PAIR AUTHORITY",
    "-" * 120,
    "status=FROZEN",
    "pair_definition=same dataset + same condition_id + different biological context",
    f"dataset_n={len(cross_datasets)}",
    f"condition_n={pair_authority_manifest['condition_n']}",
    f"pair_n={pair_authority_manifest['pair_n']}",
    "",
    "PRIMARY EVALUATION UNIT",
    "-" * 120,
    "unit=condition",
    "reason=avoid pair-count domination by conditions/datasets with many contexts",
    "primary_global_metric=condition-wise Spearman(text-context similarity,response similarity)",
    "primary_local_metric=condition-wise nearest-context Hit@1 gain over exact random expectation",
    "P1_P3_role=negative controls; biological context absent and within-condition text invariant",
    "P2_P4_role=context-aware representations",
    "",
    "DATASET CONTEXT AXES",
    "-" * 120,
    dataset_feas.loc[
        dataset_feas[
            "cross_context_condition_n"
        ] > 0,
        [
            "dataset_id",
            "cross_context_condition_n",
            "cross_context_response_atom_n",
            "within_condition_cross_context_pair_n",
            "single_bio_axis_pair_n",
            "pair_diff_cell_type_n",
            "pair_diff_cell_line_n",
        ],
    ].to_string(index=False),
    "",
    "NEGATIVE-CONTROL QC",
    "-" * 120,
    f"P1_P3_valid_context_RSA_condition_n={negative_rsa_valid_n}",
    f"P1_P3_max_offdiag_similarity_deviation_from_1={max_invariance_deviation:.10g}",
    f"P1_P3_max_abs_nearest_context_hit_gain={negative_hit_gain_max_abs:.10g}",
    "",
    "CONDITION-BALANCED SUMMARY",
    "-" * 120,
    summary.to_string(index=False),
    "",
    "P4 MINUS P2",
    "-" * 120,
    p4_minus_p2.to_string(index=False),
    "",
    "STATUS",
    "-" * 120,
    "R2_WITHIN_CONDITION_CROSS_CONTEXT_ALIGNMENT=PASS",
    "NEXT=INTERPRET_CROSS_CONTEXT_RESULTS_AND_DECIDE_MAIN_VS_SUPPLEMENT",
]

AUDIT_TXT.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)


write_json(
    {
        "version": "R2_WITHIN_CONDITION_CROSS_CONTEXT_ALIGNMENT_v1",
        "status": "PASS",
        "primary_unit": "condition",
        "datasets": cross_datasets,
        "metrics": {
            "global": "condition-wise context RSA",
            "local": "nearest-context Hit@1 gain over exact random",
            "pair_pooled": "secondary only",
        },
        "negative_controls": [
            "P1",
            "P3",
        ],
        "context_aware_views": [
            "P2",
            "P4",
        ],
        "inputs": {
            "pair_authority": pair_authority_manifest,
            "p1p4_audit_sha256": sha256_file(
                P1P4_AUDIT
            ),
            "atom_authority_sha256": sha256_file(
                ATOM_AUTHORITY
            ),
        },
        "outputs": {
            "condition_metrics": str(
                CONDITION_METRICS_TSV
            ),
            "dataset_model_view_summary": str(
                DATASET_MODEL_VIEW_TSV
            ),
            "p4_minus_p2": str(
                P4_MINUS_P2_TSV
            ),
            "model_macro_descriptive": str(
                MODEL_MACRO_TSV
            ),
            "pair_pooled_secondary": str(
                PAIR_POOLED_SECONDARY_TSV
            ),
            "audit": str(
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
