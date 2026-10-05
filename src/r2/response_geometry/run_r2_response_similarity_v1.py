from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata


# ======================================================================================
# Paths / CLI
# ======================================================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Build PerturbContextAlign R2 primary measured-response similarity "
            "matrices from frozen logFC-HVG effects. No H5AD access."
        )
    )
    ap.add_argument(
        "--root",
        default=".",
    )
    ap.add_argument(
        "--datasets",
        nargs="*",
        default=[],
        help="Optional subset. Default: all datasets in the frozen effect audit.",
    )
    return ap.parse_args()


ARGS = parse_args()

ROOT = Path(ARGS.root).expanduser().resolve()

NEW = (
    ROOT
    / "0_ProjectCodeAndLogs"
    / "PerturbContextAlign_R2_corrected_v1_20261002"
)

EFFECT_ROOT = NEW / "04_response_geometry" / "01_effect_v1"
EFFECT_AUDIT_TSV = EFFECT_ROOT / "R2_MEASURED_RESPONSE_EFFECT_AUDIT_v1.tsv"
EFFECT_AUDIT_TXT = EFFECT_ROOT / "R2_MEASURED_RESPONSE_EFFECT_AUDIT_v1.txt"
EFFECT_MANIFEST = EFFECT_ROOT / "R2_MEASURED_RESPONSE_EFFECT_MANIFEST_v1.json"

OUT = NEW / "04_response_geometry" / "02_similarity_v1"
OUT.mkdir(parents=True, exist_ok=True)

GLOBAL_AUDIT_TSV = OUT / "R2_RESPONSE_SIMILARITY_AUDIT_v1.tsv"
GLOBAL_AUDIT_TXT = OUT / "R2_RESPONSE_SIMILARITY_AUDIT_v1.txt"
GLOBAL_MANIFEST = OUT / "R2_RESPONSE_SIMILARITY_MANIFEST_v1.json"

SCRIPT_VERSION = "R2_RESPONSE_SIMILARITY_v1.0.0"


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


def load_effect_bundle(path: Path):
    """
    Frozen effect bundle format:
      X
      condition_ids
      response_atom_ids
      feature_names
      metadata
    """
    if not path.is_file():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=True) as z:
        keys = set(z.files)

        if "X" not in keys:
            raise KeyError(f"{path}: X missing")

        X = np.asarray(z["X"], dtype=np.float32)

        id_key = (
            "response_atom_ids"
            if "response_atom_ids" in keys
            else "condition_ids"
        )

        if id_key not in keys:
            raise KeyError(f"{path}: response atom IDs missing")

        ids = np.asarray(z[id_key]).astype(str)

        feature_names = (
            np.asarray(z["feature_names"]).astype(str)
            if "feature_names" in keys
            else np.arange(X.shape[1]).astype(str)
        )

        metadata = {}
        if "metadata" in keys:
            raw = z["metadata"]
            try:
                value = raw.item()
            except Exception:
                value = raw
            try:
                metadata = json.loads(str(value))
            except Exception:
                metadata = {"raw_metadata": str(value)}

    return X, ids, feature_names, metadata


def save_similarity_bundle(
    path: Path,
    S: np.ndarray,
    response_atom_ids: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        path,
        S=np.asarray(S, dtype=np.float32),
        X=np.asarray(S, dtype=np.float32),  # compatibility alias
        response_atom_ids=np.asarray(response_atom_ids, dtype=object),
        condition_ids=np.asarray(response_atom_ids, dtype=object),
        metadata=np.asarray(
            json.dumps(
                metadata,
                ensure_ascii=False,
                sort_keys=True,
            ),
            dtype=object,
        ),
    )


def spearman_similarity_exact_historical(
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Historical Result2 definition:
      rankdata across genes within each response vector
      -> Pearson similarity between ranked rows

    Corrected R2 hardening:
      constant/zero-variance response rows are marked invalid and excluded
      rather than silently converted to zero similarity.

    Returns
    -------
    S : float32 [n_valid, n_valid]
    valid_mask : bool [n_original]
    """
    A = np.asarray(X, dtype=np.float64)

    if A.ndim != 2:
        raise ValueError(f"Expected 2D response matrix, got {A.shape}")

    if not np.isfinite(A).all():
        raise RuntimeError("Primary logFC matrix contains non-finite values.")

    n, p = A.shape

    if n < 2:
        raise RuntimeError("Need at least two response atoms.")
    if p < 3:
        raise RuntimeError("Need at least three genes for Spearman geometry.")

    # rankdata uses average ranks for ties, matching historical Result2.
    R = np.empty_like(A, dtype=np.float64)

    for i in range(n):
        R[i] = rankdata(A[i])

    R -= R.mean(axis=1, keepdims=True)

    norms = np.linalg.norm(R, axis=1)

    valid = (
        np.isfinite(norms)
        & (norms > 1e-12)
    )

    if valid.sum() < 2:
        raise RuntimeError(
            f"Too few nonconstant response vectors: {int(valid.sum())}/{n}"
        )

    Z = R[valid] / norms[valid, None]

    S = Z @ Z.T

    # Numerical cleanup only; no biological missingness is encoded as zero.
    S = np.clip(S, -1.0, 1.0)

    # Exact self similarity.
    np.fill_diagonal(S, 1.0)

    return np.asarray(S, dtype=np.float32), valid


def matrix_qc(S: np.ndarray) -> dict[str, Any]:
    A = np.asarray(S, dtype=np.float64)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Similarity must be square, got {A.shape}")

    n = A.shape[0]
    tri = np.triu_indices(n, 1)

    upper = A[tri]

    finite_fraction = (
        float(np.isfinite(upper).mean())
        if len(upper)
        else 1.0
    )

    symmetry_max_abs = float(
        np.max(
            np.abs(A - A.T)
        )
    )

    diagonal_max_abs_from_one = float(
        np.max(
            np.abs(np.diag(A) - 1.0)
        )
    )

    if len(upper):
        upper_min = float(np.min(upper))
        upper_max = float(np.max(upper))
        upper_mean = float(np.mean(upper))
        upper_sd = float(np.std(upper))
        constant_geometry = bool(np.ptp(upper) <= 1e-12)
    else:
        upper_min = np.nan
        upper_max = np.nan
        upper_mean = np.nan
        upper_sd = np.nan
        constant_geometry = True

    return {
        "n_atoms": int(n),
        "n_pairs": int(len(upper)),
        "finite_fraction": finite_fraction,
        "symmetry_max_abs": symmetry_max_abs,
        "diagonal_max_abs_from_one": diagonal_max_abs_from_one,
        "upper_min": upper_min,
        "upper_max": upper_max,
        "upper_mean": upper_mean,
        "upper_sd": upper_sd,
        "constant_geometry": constant_geometry,
    }


# ======================================================================================
# Frozen upstream checks
# ======================================================================================

for p in [
    EFFECT_AUDIT_TSV,
    EFFECT_AUDIT_TXT,
    EFFECT_MANIFEST,
]:
    if not p.is_file():
        raise FileNotFoundError(
            f"Frozen measured-response effect authority missing: {p}"
        )

effect_manifest = json.loads(
    EFFECT_MANIFEST.read_text(encoding="utf-8")
)

if effect_manifest.get("status") != "PASS":
    raise RuntimeError(
        f"Measured-response effect manifest is not PASS: "
        f"{effect_manifest.get('status')!r}"
    )

effect_audit = pd.read_csv(
    EFFECT_AUDIT_TSV,
    sep="\t",
    low_memory=False,
)

if effect_audit.empty:
    raise RuntimeError("Measured-response effect audit is empty.")

if not effect_audit["status"].astype(str).eq("PASS").all():
    raise RuntimeError("Not all measured-response effect datasets are PASS.")

all_datasets = effect_audit["dataset_id"].astype(str).tolist()

if len(all_datasets) != 9:
    raise RuntimeError(
        f"Expected 9 frozen quantitative datasets, got {len(all_datasets)}"
    )

if ARGS.datasets:
    requested = list(map(str, ARGS.datasets))
    unknown = sorted(set(requested) - set(all_datasets))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
else:
    requested = list(all_datasets)


# ======================================================================================
# Dataset build
# ======================================================================================

def build_dataset(dataset_id: str) -> dict[str, Any]:
    t0 = time.time()

    ds_effect = EFFECT_ROOT / dataset_id
    effect_path = ds_effect / "effect_logfc_hvg.npz"
    effect_qc_path = ds_effect / "effect_qc.tsv"
    effect_index_path = ds_effect / "response_atom_index.tsv"
    effect_done_path = ds_effect / "DONE.json"

    for p in [
        effect_path,
        effect_qc_path,
        effect_index_path,
        effect_done_path,
    ]:
        if not p.is_file():
            raise FileNotFoundError(f"{dataset_id}: missing {p}")

    effect_done = json.loads(
        effect_done_path.read_text(encoding="utf-8")
    )

    if effect_done.get("status") != "PASS":
        raise RuntimeError(f"{dataset_id}: effect DONE is not PASS")

    ds_out = OUT / dataset_id
    ds_out.mkdir(parents=True, exist_ok=True)

    similarity_path = ds_out / "response_similarity_logfc_hvg_spearman.npz"
    index_path = ds_out / "response_similarity_index.tsv"
    qc_path = ds_out / "response_similarity_qc.json"
    done_path = ds_out / "DONE.json"

    input_hashes = {
        "effect_logfc_hvg": sha256_file(effect_path),
        "effect_qc": sha256_file(effect_qc_path),
        "response_atom_index": sha256_file(effect_index_path),
        "effect_done": sha256_file(effect_done_path),
    }

    # Resume only if all exact upstream hashes match.
    if done_path.is_file() and similarity_path.is_file() and index_path.is_file():
        old = json.loads(done_path.read_text(encoding="utf-8"))
        if (
            old.get("status") == "PASS"
            and old.get("script_version") == SCRIPT_VERSION
            and old.get("input_hashes") == input_hashes
        ):
            log(f"[resume] {dataset_id}")
            return old["audit_row"]

    log(f"[similarity] {dataset_id}")

    X, ids, features, metadata = load_effect_bundle(effect_path)

    effect_qc = pd.read_csv(
        effect_qc_path,
        sep="\t",
        low_memory=False,
    )

    effect_index = pd.read_csv(
        effect_index_path,
        sep="\t",
        low_memory=False,
    )

    if len(X) != len(ids):
        raise RuntimeError(
            f"{dataset_id}: effect rows {len(X)} != ID rows {len(ids)}"
        )

    if X.shape[1] != len(features):
        raise RuntimeError(
            f"{dataset_id}: effect feature count mismatch"
        )

    if len(effect_qc) != len(ids):
        raise RuntimeError(
            f"{dataset_id}: effect_qc rows {len(effect_qc)} != effect rows {len(ids)}"
        )

    if len(effect_index) != len(ids):
        raise RuntimeError(
            f"{dataset_id}: response_atom_index rows {len(effect_index)} != effect rows {len(ids)}"
        )

    if "response_atom_id" not in effect_index.columns:
        raise KeyError(
            f"{dataset_id}: response_atom_index lacks response_atom_id"
        )

    index_ids = effect_index["response_atom_id"].astype(str).to_numpy()

    if not np.array_equal(index_ids, ids):
        raise RuntimeError(
            f"{dataset_id}: effect bundle and response_atom_index ID order mismatch"
        )

    if "response_atom_id" in effect_qc.columns:
        qc_ids = effect_qc["response_atom_id"].astype(str).to_numpy()
        if not np.array_equal(qc_ids, ids):
            raise RuntimeError(
                f"{dataset_id}: effect_qc ID order mismatch"
            )

    # Frozen effect audit already reported zero constant rows, but re-check here.
    row_sd = np.std(
        np.asarray(X, dtype=np.float64),
        axis=1,
    )

    preconstant = (
        ~np.isfinite(row_sd)
        | (row_sd <= 1e-12)
    )

    S, valid_mask = spearman_similarity_exact_historical(X)

    if not np.array_equal(
        valid_mask,
        ~preconstant,
    ):
        raise RuntimeError(
            f"{dataset_id}: constant-vector QC disagreement"
        )

    valid_ids = ids[valid_mask]
    valid_index = effect_index.loc[valid_mask].copy().reset_index(drop=True)

    valid_index.insert(
        0,
        "similarity_row",
        np.arange(len(valid_index), dtype=np.int64),
    )

    valid_index["primary_response_similarity_status"] = "ELIGIBLE"

    # Preserve excluded rows separately through counts/provenance; current frozen
    # build has zero exclusions in all nine datasets.
    excluded_n = int((~valid_mask).sum())

    qc = matrix_qc(S)

    if qc["finite_fraction"] != 1.0:
        raise RuntimeError(
            f"{dataset_id}: non-finite values in response similarity"
        )

    if qc["symmetry_max_abs"] > 5e-6:
        raise RuntimeError(
            f"{dataset_id}: similarity symmetry failure "
            f"{qc['symmetry_max_abs']}"
        )

    if qc["diagonal_max_abs_from_one"] > 5e-6:
        raise RuntimeError(
            f"{dataset_id}: similarity diagonal failure "
            f"{qc['diagonal_max_abs_from_one']}"
        )

    if qc["constant_geometry"]:
        raise RuntimeError(
            f"{dataset_id}: response similarity geometry is constant"
        )

    expected_row = effect_audit.loc[
        effect_audit["dataset_id"].astype(str).eq(dataset_id)
    ]

    if len(expected_row) != 1:
        raise RuntimeError(
            f"{dataset_id}: expected exactly one upstream audit row"
        )

    expected = expected_row.iloc[0]

    expected_effect_n = int(expected["eligible_response_atom_n"])
    expected_similarity_n = int(
        expected["primary_similarity_eligible_atom_n"]
    )
    expected_constant_n = int(expected["constant_logfc_atom_n"])

    if len(ids) != expected_effect_n:
        raise RuntimeError(
            f"{dataset_id}: effect atom count {len(ids)} != frozen "
            f"{expected_effect_n}"
        )

    if len(valid_ids) != expected_similarity_n:
        raise RuntimeError(
            f"{dataset_id}: similarity atom count {len(valid_ids)} != frozen "
            f"{expected_similarity_n}"
        )

    if excluded_n != expected_constant_n:
        raise RuntimeError(
            f"{dataset_id}: constant/excluded count {excluded_n} != frozen "
            f"{expected_constant_n}"
        )

    output_metadata = {
        "dataset_id": dataset_id,
        "version": "R2_RESPONSE_SIMILARITY_v1",
        "row_identity": "response_atom_id",
        "effect_representation": "logfc_hvg",
        "similarity_metric": "spearman",
        "implementation": (
            "scipy.stats.rankdata per response vector -> row-center -> "
            "L2 -> dot product"
        ),
        "tie_method": "rankdata_average",
        "constant_response_policy": "exclude_explicitly",
        "n_source_effect_atoms": int(len(ids)),
        "n_similarity_atoms": int(len(valid_ids)),
        "n_excluded_constant_atoms": excluded_n,
        "n_hvg": int(X.shape[1]),
        "source_effect_metadata": metadata,
    }

    save_similarity_bundle(
        similarity_path,
        S,
        valid_ids,
        output_metadata,
    )

    valid_index.to_csv(
        index_path,
        sep="\t",
        index=False,
    )

    write_json(
        {
            **qc,
            "dataset_id": dataset_id,
            "n_source_effect_atoms": int(len(ids)),
            "n_similarity_atoms": int(len(valid_ids)),
            "n_excluded_constant_atoms": excluded_n,
            "n_hvg": int(X.shape[1]),
            "similarity_min": float(np.min(S)),
            "similarity_max": float(np.max(S)),
            "status": "PASS",
        },
        qc_path,
    )

    audit_row = {
        "dataset_id": dataset_id,
        "status": "PASS",
        "effect_representation": "logfc_hvg",
        "similarity_metric": "spearman",
        "n_hvg": int(X.shape[1]),
        "source_effect_atom_n": int(len(ids)),
        "excluded_constant_atom_n": excluded_n,
        "response_similarity_atom_n": int(len(valid_ids)),
        "pair_n": int(qc["n_pairs"]),
        "finite_fraction": float(qc["finite_fraction"]),
        "symmetry_max_abs": float(qc["symmetry_max_abs"]),
        "diagonal_max_abs_from_one": float(
            qc["diagonal_max_abs_from_one"]
        ),
        "upper_min": float(qc["upper_min"]),
        "upper_max": float(qc["upper_max"]),
        "upper_mean": float(qc["upper_mean"]),
        "upper_sd": float(qc["upper_sd"]),
        "constant_geometry": bool(qc["constant_geometry"]),
        "elapsed_seconds": float(time.time() - t0),
    }

    done = {
        "status": "PASS",
        "version": "R2_RESPONSE_SIMILARITY_v1",
        "script_version": SCRIPT_VERSION,
        "dataset_id": dataset_id,
        "input_hashes": input_hashes,
        "outputs": {
            "similarity": str(similarity_path),
            "index": str(index_path),
            "qc": str(qc_path),
        },
        "audit_row": audit_row,
    }

    write_json(done, done_path)

    log(
        f"[done] {dataset_id} "
        f"atoms={len(valid_ids)} "
        f"pairs={qc['n_pairs']} "
        f"upper_sd={qc['upper_sd']:.6f}"
    )

    return audit_row


# ======================================================================================
# Run / consolidated audit
# ======================================================================================

rows = []

for ds in requested:
    rows.append(build_dataset(ds))

# Consolidate all already completed datasets so partial reruns still report global state.
completed_rows = []

for ds in all_datasets:
    p = OUT / ds / "DONE.json"
    if p.is_file():
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("status") == "PASS":
            completed_rows.append(d["audit_row"])

audit = (
    pd.DataFrame(completed_rows).sort_values("dataset_id")
    if completed_rows
    else pd.DataFrame(rows).sort_values("dataset_id")
)

audit.to_csv(
    GLOBAL_AUDIT_TSV,
    sep="\t",
    index=False,
)

complete_set = set(audit["dataset_id"].astype(str))
all_complete = (
    complete_set == set(all_datasets)
    and audit["status"].astype(str).eq("PASS").all()
)

total_atoms = int(
    audit["response_similarity_atom_n"].sum()
) if len(audit) else 0

total_pairs = int(
    audit["pair_n"].sum()
) if len(audit) else 0

total_excluded = int(
    audit["excluded_constant_atom_n"].sum()
) if len(audit) else 0

status = "PASS" if all_complete else "PARTIAL_PASS"

lines = [
    "PERTURBCONTEXTALIGN R2 RESPONSE SIMILARITY AUDIT v1",
    "=" * 120,
    "",
    "METHOD",
    "-" * 120,
    "source=R2 frozen effect_logfc_hvg.npz",
    "H5AD_access=FALSE",
    "effect_recomputed=FALSE",
    "similarity=Spearman(logFC_HVG_i, logFC_HVG_j)",
    "implementation=rankdata(avg ties) -> row-center -> L2 -> dot product",
    "historical_definition_reproduced=TRUE",
    "constant_response_policy=explicit_exclusion_not_silent_zero",
    "",
    "DATASET AUDIT",
    "-" * 120,
    audit.to_string(index=False),
    "",
    "OVERALL",
    "-" * 120,
    f"completed_dataset_n={len(audit)}",
    f"expected_dataset_n={len(all_datasets)}",
    f"response_similarity_atom_n={total_atoms}",
    f"excluded_constant_atom_n={total_excluded}",
    f"within_dataset_pair_n={total_pairs}",
    "",
    "STATUS",
    "-" * 120,
    f"R2_RESPONSE_SIMILARITY={status}",
]

if all_complete:
    lines += [
        "NEXT=R2_TEXT_RESPONSE_ALIGNMENT_METRICS_v1",
    ]
else:
    lines += [
        "NEXT=COMPLETE_REMAINING_RESPONSE_SIMILARITY_DATASETS",
    ]

GLOBAL_AUDIT_TXT.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)

write_json(
    {
        "version": "R2_RESPONSE_SIMILARITY_v1",
        "script_version": SCRIPT_VERSION,
        "status": status,
        "completed_datasets": audit["dataset_id"].astype(str).tolist(),
        "expected_datasets": all_datasets,
        "input_hashes": {
            "effect_audit": sha256_file(EFFECT_AUDIT_TSV),
            "effect_manifest": sha256_file(EFFECT_MANIFEST),
        },
        "method": {
            "effect_representation": "logfc_hvg",
            "similarity_metric": "spearman",
            "rank_tie_method": "average",
            "strict_constant_response_policy": "exclude",
        },
    },
    GLOBAL_MANIFEST,
)

print(
    GLOBAL_AUDIT_TXT.read_text(
        encoding="utf-8"
    )
)

if not all_complete:
    raise SystemExit(3)
