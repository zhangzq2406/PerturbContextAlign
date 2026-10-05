from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default=".",
        help="Project root containing result2/, result3/, and 0_ProjectCodeAndLogs/.",
    )
    return ap.parse_args()


ARGS = parse_args()
ROOT = Path(ARGS.root).expanduser().resolve()

NEW = (
    ROOT
    / "0_ProjectCodeAndLogs"
    / "PerturbContextAlign_R2_corrected_v1_20261002"
)

RESULT3 = ROOT / "result3"
OLD_R2 = ROOT / "result2"

RESULT3_SRC = (
    RESULT3
    / "code"
    / "src"
)

RESULT3_SIMILARITY_PY = (
    RESULT3_SRC
    / "result3"
    / "similarity.py"
)

if not RESULT3_SIMILARITY_PY.is_file():
    # Public release fallback: identical frozen helper, no change to fusion math.
    RESULT3_SRC = Path(__file__).resolve().parents[2] / "vendor"
    RESULT3_SIMILARITY_PY = RESULT3_SRC / "result3" / "similarity.py"
    if not RESULT3_SIMILARITY_PY.is_file():
        raise FileNotFoundError(RESULT3_SIMILARITY_PY)

sys.path.insert(
    0,
    str(
        RESULT3_SRC
    ),
)

from result3.similarity import hybrid_similarity  # noqa: E402


NATIVE_CATALOG = (
    RESULT3
    / "cache"
    / "native_prior_similarity"
    / "native_similarity_catalog.tsv"
)

HIST_LANG_CATALOG = (
    OLD_R2
    / "outputs"
    / "metrics"
    / "prior_similarity_catalog_v2.tsv"
)

ATOM_AUTHORITY = (
    NEW
    / "02_input_authority"
    / "R2_RESPONSE_ATOM_AUTHORITY_v1.tsv"
)

BGE_EMB = (
    NEW
    / "05_response_aligned_text"
    / "02_embeddings_v1"
    / "bge_m3"
    / "response_atom_embeddings.npy"
)

BGE_INDEX = (
    NEW
    / "05_response_aligned_text"
    / "02_embeddings_v1"
    / "R2_RESPONSE_ATOM_EMBEDDING_INDEX_v1.tsv"
)

RESP_ROOT = (
    NEW
    / "04_response_geometry"
    / "02_similarity_v1"
)

OUT = (
    NEW
    / "08_knowledge_native"
    / "04_corrected_native_fusion_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

HIST_VALIDATION = (
    OUT
    / "R2_HISTORICAL_HYBRID_IMPLEMENTATION_VALIDATION_v1.tsv"
)

BY_COMPARISON = (
    OUT
    / "R2_CORRECTED_NATIVE_FUSION_BY_COMPARISON_v1.tsv"
)

SUMMARY = (
    OUT
    / "R2_CORRECTED_NATIVE_FUSION_SUMMARY_v1.tsv"
)

AUDIT = (
    OUT
    / "R2_CORRECTED_NATIVE_FUSION_AUDIT_v1.txt"
)

MANIFEST = (
    OUT
    / "R2_CORRECTED_NATIVE_FUSION_MANIFEST_v1.json"
)

MIN_CONDITIONS = 8


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write_json(obj, path: Path):
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


def load_npz(path: Path):
    with np.load(path, allow_pickle=True) as z:
        keys = list(z.files)

        matrix = None
        matrix_key = None

        for key in [
            "X",
            "S",
            "similarity",
            "matrix",
        ]:
            if key in keys:
                arr = np.asarray(z[key])
                if (
                    arr.ndim == 2
                    and arr.shape[0] == arr.shape[1]
                ):
                    matrix = arr
                    matrix_key = key
                    break

        if matrix is None:
            for key in keys:
                arr = np.asarray(z[key])
                if (
                    arr.ndim == 2
                    and arr.shape[0] == arr.shape[1]
                ):
                    matrix = arr
                    matrix_key = key
                    break

        if matrix is None:
            raise RuntimeError(
                f"No square matrix in {path}"
            )

        n = matrix.shape[0]

        ids = None
        id_key = None

        for key in [
            "condition_ids",
            "response_atom_ids",
            "row_ids",
            "ids",
            "feature_names",
        ]:
            if key in keys:
                arr = np.asarray(
                    z[key]
                ).astype(str)

                if arr.ndim == 1 and len(arr) == n:
                    ids = arr
                    id_key = key
                    break

        if ids is None:
            one_d = []

            for key in keys:
                arr = np.asarray(
                    z[key]
                )

                if arr.ndim == 1 and len(arr) == n:
                    one_d.append(
                        (
                            key,
                            arr.astype(str),
                        )
                    )

            if len(one_d) == 1:
                id_key, ids = one_d[0]

        if ids is None:
            raise RuntimeError(
                f"No ID vector in {path}; keys={keys}"
            )

    return (
        np.asarray(
            matrix,
            dtype=np.float32,
        ),
        np.asarray(
            ids
        ).astype(str),
        matrix_key,
        id_key,
    )


def load_response(path: Path):
    return load_npz(path)[:2]


def upper(S):
    A = np.asarray(
        S,
        dtype=np.float64,
    )

    return A[
        np.triu_indices(
            A.shape[0],
            1,
        )
    ]


def spearman(x, y):
    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    keep = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[keep]
    y = y[keep]

    if len(x) < 3:
        return np.nan

    if (
        np.ptp(x) <= 1e-12
        or np.ptp(y) <= 1e-12
    ):
        return np.nan

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

    den = math.sqrt(
        float(
            np.dot(
                rx,
                rx,
            )
        )
        * float(
            np.dot(
                ry,
                ry,
            )
        )
    )

    return (
        float(
            np.dot(
                rx,
                ry,
            )
            / den
        )
        if den > 0
        else np.nan
    )


for p in [
    NATIVE_CATALOG,
    HIST_LANG_CATALOG,
    ATOM_AUTHORITY,
    BGE_EMB,
    BGE_INDEX,
]:
    if not p.is_file():
        raise FileNotFoundError(p)


catalog = pd.read_csv(
    NATIVE_CATALOG,
    sep="\t",
    low_memory=False,
)

native = catalog.loc[
    catalog[
        "prior_group"
    ].astype(str).eq(
        "native"
    )
].copy()

hybrid_catalog = catalog.loc[
    catalog[
        "prior_group"
    ].astype(str).eq(
        "hybrid"
    )
].copy()

lang_catalog = pd.read_csv(
    HIST_LANG_CATALOG,
    sep="\t",
    low_memory=False,
)

atoms = pd.read_csv(
    ATOM_AUTHORITY,
    sep="\t",
    low_memory=False,
)

bge_index = pd.read_csv(
    BGE_INDEX,
    sep="\t",
    low_memory=False,
)


for c in [
    "dataset_id",
    "condition_id",
    "response_atom_id",
]:
    if c in atoms.columns:
        atoms[c] = atoms[c].astype(str)


bge_index[
    "response_atom_id"
] = bge_index[
    "response_atom_id"
].astype(str)

bge_pos = {
    rid: i
    for i, rid in enumerate(
        bge_index[
            "response_atom_id"
        ]
    )
}

bge = np.load(
    BGE_EMB,
    mmap_mode="r",
)


# -----------------------------------------------------------------------------
# Validate frozen historical hybrid implementation and alignment
# -----------------------------------------------------------------------------

validation_rows = []


for _, nrow in native.iterrows():
    ds = str(
        nrow[
            "dataset_id"
        ]
    )

    prior = str(
        nrow[
            "prior_id"
        ]
    )

    N, nids, _, _ = load_npz(
        Path(
            str(
                nrow[
                    "path"
                ]
            )
        )
    )

    expected_hybrid_id = (
        f"hybrid_bge_m3_P4__{prior}"
    )

    hrow = hybrid_catalog.loc[
        hybrid_catalog[
            "dataset_id"
        ].astype(str).eq(ds)
        & hybrid_catalog[
            "prior_id"
        ].astype(str).eq(
            expected_hybrid_id
        )
    ]

    if len(hrow) != 1:
        raise RuntimeError(
            f"{ds}/{prior}: expected one historical hybrid row"
        )

    H_hist, hids, _, _ = load_npz(
        Path(
            str(
                hrow.iloc[0][
                    "path"
                ]
            )
        )
    )

    lrow = lang_catalog.loc[
        lang_catalog[
            "dataset_id"
        ].astype(str).eq(ds)
        & lang_catalog[
            "model_key"
        ].astype(str).eq(
            "bge_m3"
        )
        & lang_catalog[
            "prompt_type"
        ].astype(str).eq(
            "P4"
        )
        & lang_catalog[
            "prompt_variant"
        ].astype(str).eq(
            "raw"
        )
    ]

    if len(lrow) != 1:
        raise RuntimeError(
            f"{ds}: expected one historical BGE-M3/P4/raw similarity row"
        )

    L_hist, lids, _, _ = load_npz(
        Path(
            str(
                lrow.iloc[0][
                    "path"
                ]
            )
        )
    )

    lpos = {
        rid: i
        for i, rid in enumerate(
            lids
        )
    }

    missing = [
        rid
        for rid in nids
        if rid not in lpos
    ]

    if missing:
        raise RuntimeError(
            f"{ds}/{prior}: native IDs absent from historical language n={len(missing)}"
        )

    lix = np.asarray(
        [
            lpos[rid]
            for rid in nids
        ],
        dtype=np.int64,
    )

    L = np.asarray(
        L_hist[
            np.ix_(
                lix,
                lix,
            )
        ],
        dtype=np.float32,
    )

    H_re = np.asarray(
        hybrid_similarity(
            L,
            N,
            0.5,
            "rank",
        ),
        dtype=np.float32,
    )

    hpos = {
        rid: i
        for i, rid in enumerate(
            hids
        )
    }

    if set(hids) != set(nids):
        raise RuntimeError(
            f"{ds}/{prior}: historical hybrid/native ID sets differ"
        )

    hix = np.asarray(
        [
            hpos[rid]
            for rid in nids
        ],
        dtype=np.int64,
    )

    H_ref = np.asarray(
        H_hist[
            np.ix_(
                hix,
                hix,
            )
        ],
        dtype=np.float32,
    )

    max_diff = float(
        np.max(
            np.abs(
                H_re - H_ref
            )
        )
    )

    validation_rows.append({
        "dataset_id": ds,
        "prior_id": prior,
        "condition_n": len(nids),
        "max_abs_diff": max_diff,
        "status": (
            "PASS"
            if max_diff <= 1e-7
            else "FAIL"
        ),
    })


validation = pd.DataFrame(
    validation_rows
)

validation.to_csv(
    HIST_VALIDATION,
    sep="\t",
    index=False,
)

if not validation[
    "status"
].eq(
    "PASS"
).all():
    raise RuntimeError(
        "Historical hybrid implementation reproduction failed."
    )


# -----------------------------------------------------------------------------
# Corrected analysis
# -----------------------------------------------------------------------------

rows = []


for _, nrow in native.iterrows():
    ds = str(
        nrow[
            "dataset_id"
        ]
    )

    prior = str(
        nrow[
            "prior_id"
        ]
    )

    condition_n = int(
        nrow[
            "n_conditions"
        ]
    )

    if condition_n < MIN_CONDITIONS:
        continue

    N_cond, nids, _, _ = load_npz(
        Path(
            str(
                nrow[
                    "path"
                ]
            )
        )
    )

    npos = {
        cid: i
        for i, cid in enumerate(
            nids
        )
    }

    ds_atoms = atoms.loc[
        atoms[
            "dataset_id"
        ].eq(ds)
        & atoms[
            "condition_id"
        ].isin(
            set(
                nids
            )
        )
    ][
        [
            "condition_id",
            "response_atom_id",
        ]
    ].drop_duplicates()

    # Restrict to the frozen corrected response-similarity atom namespace.
    E_full, eids = load_response(
        RESP_ROOT
        / ds
        / "response_similarity_logfc_hvg_spearman.npz"
    )

    epos = {
        rid: i
        for i, rid in enumerate(
            eids
        )
    }

    ds_atoms = ds_atoms.loc[
        ds_atoms[
            "response_atom_id"
        ].isin(
            set(
                eids
            )
        )
    ].copy()

    matched_condition_n = int(
        ds_atoms[
            "condition_id"
        ].nunique()
    )

    if matched_condition_n < MIN_CONDITIONS:
        raise RuntimeError(
            f"{ds}/{prior}: corrected matched distinct conditions < {MIN_CONDITIONS}"
        )

    # Every historical native condition was proven recoverable in v3.1.
    if matched_condition_n != condition_n:
        raise RuntimeError(
            f"{ds}/{prior}: corrected condition n={matched_condition_n} "
            f"!= historical native n={condition_n}"
        )

    ds_atoms = ds_atoms.sort_values(
        [
            "condition_id",
            "response_atom_id",
        ]
    ).reset_index(drop=True)

    atom_ids = ds_atoms[
        "response_atom_id"
    ].astype(str).tolist()

    cond_ids = ds_atoms[
        "condition_id"
    ].astype(str).tolist()

    cix = np.asarray(
        [
            npos[cid]
            for cid in cond_ids
        ],
        dtype=np.int64,
    )

    N = np.asarray(
        N_cond[
            np.ix_(
                cix,
                cix,
            )
        ],
        dtype=np.float32,
    )

    erows = np.asarray(
        [
            epos[rid]
            for rid in atom_ids
        ],
        dtype=np.int64,
    )

    E = np.asarray(
        E_full[
            np.ix_(
                erows,
                erows,
            )
        ],
        dtype=np.float32,
    )

    brows = np.asarray(
        [
            bge_pos[rid]
            for rid in atom_ids
        ],
        dtype=np.int64,
    )

    Z = np.asarray(
        bge[
            brows
        ],
        dtype=np.float32,
    )

    L = np.clip(
        Z @ Z.T,
        -1.0,
        1.0,
    )

    np.fill_diagonal(
        L,
        1.0,
    )

    H = np.asarray(
        hybrid_similarity(
            L,
            N,
            0.5,
            "rank",
        ),
        dtype=np.float32,
    )

    ev = upper(E)

    rsa_native = spearman(
        upper(N),
        ev,
    )

    rsa_language = spearman(
        upper(L),
        ev,
    )

    rsa_hybrid = spearman(
        upper(H),
        ev,
    )

    if not all(
        np.isfinite(
            x
        )
        for x in [
            rsa_native,
            rsa_language,
            rsa_hybrid,
        ]
    ):
        raise RuntimeError(
            f"{ds}/{prior}: non-finite corrected RSA"
        )

    best_single = max(
        rsa_native,
        rsa_language,
    )

    rows.append({
        "dataset_id": ds,
        "family_group": str(
            nrow[
                "family_group"
            ]
        ),
        "native_prior_id": prior,
        "distinct_condition_n": matched_condition_n,
        "corrected_response_atom_n": len(atom_ids),
        "corrected_pair_n": int(
            len(ev)
        ),
        "historical_condition_coverage": float(
            nrow[
                "condition_coverage"
            ]
        ),
        "native_rsa": rsa_native,
        "language_bge_p4_rsa": rsa_language,
        "fusion_rsa": rsa_hybrid,
        "native_minus_language_rsa": (
            rsa_native
            - rsa_language
        ),
        "best_single_rsa": best_single,
        "fusion_minus_best_single_rsa": (
            rsa_hybrid
            - best_single
        ),
        "fusion_improves_best_single": bool(
            rsa_hybrid
            > best_single
        ),
        "status": "PASS",
    })


result = pd.DataFrame(
    rows
)

if len(result) != 33:
    raise RuntimeError(
        f"Expected 33 corrected manuscript comparisons, got {len(result)}"
    )

result.to_csv(
    BY_COMPARISON,
    sep="\t",
    index=False,
)


summary_rows = []

for family, sub in result.groupby(
    "family_group",
    sort=False,
):
    summary_rows.append({
        "scope": family,
        "comparison_n": len(sub),
        "native_minus_language_macro_mean": float(
            sub[
                "native_minus_language_rsa"
            ].mean()
        ),
        "fusion_minus_best_single_macro_mean": float(
            sub[
                "fusion_minus_best_single_rsa"
            ].mean()
        ),
        "fusion_improves_best_single_n": int(
            sub[
                "fusion_improves_best_single"
            ].sum()
        ),
    })


summary_rows.append({
    "scope": "all_33",
    "comparison_n": len(result),
    "native_minus_language_macro_mean": float(
        result[
            "native_minus_language_rsa"
        ].mean()
    ),
    "fusion_minus_best_single_macro_mean": float(
        result[
            "fusion_minus_best_single_rsa"
        ].mean()
    ),
    "fusion_improves_best_single_n": int(
        result[
            "fusion_improves_best_single"
        ].sum()
    ),
})


summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    SUMMARY,
    sep="\t",
    index=False,
)


lines = [
    "PERTURBCONTEXTALIGN R2 CORRECTED NATIVE / FUSION AUDIT v1",
    "=" * 120,
    "",
    "HISTORICAL FUSION IMPLEMENTATION",
    "-" * 120,
    f"source={RESULT3_SIMILARITY_PY}",
    f"source_sha256={sha256_file(RESULT3_SIMILARITY_PY)}",
    f"historical_validation_row_n={len(validation)}",
    f"historical_validation_max_abs_diff={validation['max_abs_diff'].max():.12g}",
    "",
    "CORRECTED CONTRACT",
    "-" * 120,
    "language_anchor=corrected_BGE-M3_P4",
    "native_source=frozen_Result3_condition_level_similarity",
    "native_lift=S_atom[a,b]=S_condition[c(a),c(b)]",
    "fusion=frozen_Result3_rank_hybrid_weight_0.5",
    "best_single=max(language_RSA,native_RSA)_same_corrected_atom_set",
    "minimum_support=8_distinct_conditions",
    "missing_zero_fill=FALSE",
    "",
    "COUNTS",
    "-" * 120,
    f"comparison_n={len(result)}",
    f"fusion_improves_best_single_n={int(result['fusion_improves_best_single'].sum())}",
    f"fusion_not_improve_best_single_n={int((~result['fusion_improves_best_single']).sum())}",
    "",
    "SUMMARY",
    "-" * 120,
    summary.to_string(index=False),
    "",
    "STATUS",
    "-" * 120,
    "R2_CORRECTED_NATIVE_FUSION=PASS",
]

AUDIT.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)

write_json(
    {
        "version": "R2_CORRECTED_NATIVE_FUSION_v1",
        "status": "PASS",
        "comparison_n": 33,
        "historical_hybrid_source": {
            "path": str(
                RESULT3_SIMILARITY_PY
            ),
            "sha256": sha256_file(
                RESULT3_SIMILARITY_PY
            ),
        },
        "outputs": {
            "historical_validation": str(
                HIST_VALIDATION
            ),
            "by_comparison": str(
                BY_COMPARISON
            ),
            "summary": str(
                SUMMARY
            ),
            "audit": str(
                AUDIT
            ),
        },
    },
    MANIFEST,
)

print(
    AUDIT.read_text(
        encoding="utf-8"
    )
)
