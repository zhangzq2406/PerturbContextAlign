from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy import sparse


ROOT = Path(".").resolve()

NEW = (
    ROOT
    / "0_ProjectCodeAndLogs"
    / "PerturbContextAlign_R2_corrected_v1_20261002"
)

HIST_RESULT2 = ROOT / "result2"

DATASET_CATALOG = (
    HIST_RESULT2
    / "configs"
    / "result2_dataset_catalog.tsv"
)

EFFECT_CONFIG = (
    HIST_RESULT2
    / "configs"
    / "result2_effect_cache_config.yaml"
)

ATOM_AUTHORITY = (
    NEW
    / "02_input_authority"
    / "R2_RESPONSE_ATOM_AUTHORITY_v1.tsv"
)

BIO_KEY_MAP = (
    NEW
    / "02_input_authority"
    / "R2_BIOLOGICAL_CONTEXT_KEY_MAP_v1.tsv"
)

PATH_AUTHORITY = (
    NEW
    / "02_input_authority"
    / "R2_DATASET_AUTHORITY_WITH_H5AD_v1.tsv"
)

NUISANCE_POLICY = (
    NEW
    / "01_contract"
    / "R2_NUISANCE_CONTROL_POLICY_v1.json"
)

FROZEN_SELECTION = (
    NEW
    / "03_response_atom_audit"
    / "R2_NUISANCE_HIERARCHY_SELECTION_v1.tsv"
)

OUT = NEW / "04_response_geometry" / "00_preflight_v1"
OUT.mkdir(parents=True, exist_ok=True)

PLAN_TSV = OUT / "R2_MEASURED_RESPONSE_CONTROL_PLAN_PREFLIGHT_v1.tsv"
DATASET_TSV = OUT / "R2_MEASURED_RESPONSE_PREFLIGHT_DATASET_SUMMARY_v1.tsv"
EXPR_TSV = OUT / "R2_EXPRESSION_INPUT_PREFLIGHT_v1.tsv"
REPORT_TXT = OUT / "R2_MEASURED_RESPONSE_EFFECT_PREFLIGHT_v1.txt"
MANIFEST_JSON = OUT / "R2_MEASURED_RESPONSE_EFFECT_PREFLIGHT_MANIFEST_v1.json"


MISSING = "__NA__"
MISSING_STRINGS = {"", "nan", "none", "null", "<na>", "na"}

TRUE_SET = {"1", "true", "t", "yes", "y", "control", "ctrl"}
FALSE_SET = {"0", "false", "f", "no", "n", "perturbed", "perturb", "case"}

BIO_FIELDS = ["species", "cell_type", "cell_line", "tissue", "disease"]

OBS_BIO = {
    "species": "C_context_species",
    "cell_type": "C_context_cell_type",
    "cell_line": "C_context_cell_line",
    "tissue": "C_context_tissue",
    "disease": "C_context_disease",
}

DONOR_COL = "C_context_donor"
BATCH_COL = "C_batch"
PLATFORM_COL = "C_platform"

DEFAULT_CONDITION_COL = "C_condition_id"
DEFAULT_CONTROL_COL = "C_control_indicator"

RNG = np.random.default_rng(20260711)


# Dataset-specific expression authorities that must never be inferred away.
# These are hard assertions, not silent overrides.
DATASET_EXPRESSION_AUTHORITY = {
    "combo_sciplex": {
        "expression_state": "raw_counts",
        "source_category": "raw_counts_in_layer",
        "layer": "counts",
        "description": (
            "source category is raw_counts_in_layer; the actual AnnData raw-count "
            "matrix is adata.layers['counts']; adata.X is processed/log1p-like "
            "and must not be treated as raw counts"
        ),
    },
    "kaggle_cross_patient": {
        "expression_state": "log1p_library_normalized",
        "source_category": "log1p_library_normalized",
        "layer": None,  # adata.X
        "description": (
            "adata.X is already library-normalized log1p expression; raw counts "
            "are unavailable; do not apply a second library-size normalization "
            "or log1p transform"
        ),
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_value(v: Any) -> str:
    if pd.isna(v):
        return MISSING
    s = str(v).strip()
    if s.lower() in MISSING_STRINGS:
        return MISSING
    return s


def norm_series(s: pd.Series) -> pd.Series:
    return s.map(norm_value)


def parse_bool_like(v: Any) -> bool | None:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if pd.isna(v):
        return None
    s = str(v).strip().lower()
    if s in TRUE_SET:
        return True
    if s in FALSE_SET:
        return False
    try:
        x = float(s)
        if x == 1:
            return True
        if x == 0:
            return False
    except Exception:
        pass
    return None


def control_mask_from_series(
    series: pd.Series,
    control_value: Any = True,
) -> np.ndarray:
    target_bool = parse_bool_like(control_value)
    parsed = series.map(parse_bool_like)

    if target_bool is not None and parsed.notna().mean() >= 0.95:
        return parsed.fillna(False).eq(target_bool).to_numpy()

    target = norm_value(control_value)
    return norm_series(series).eq(target).to_numpy()


def get_expr_source(adata, layer: str | None):
    if layer is None:
        return adata.X
    if layer not in adata.layers:
        raise KeyError(
            f"Layer {layer!r} not found; available={list(adata.layers.keys())}"
        )
    return adata.layers[layer]


def read_small_expression_sample(
    adata,
    X,
    layer: str | None,
    n_rows: int,
    n_cols: int,
) -> np.ndarray:
    """
    Read a small expression block without fancy/scalar indexing on backed
    sparse datasets.

    Strategy
    --------
    * For backed adata.X, use AnnData.chunked_X(), which is the official
      row-chunk interface for backed X.
    * For layers or ordinary in-memory matrices, read a contiguous row slice
      X[:row_n] (single slice, no tuple/fancy index).
    * Only after the row block is materialized do we select a random subset of
      genes in memory.

    This avoids the AnnData-backed-sparse -> scipy indexing failures triggered
    by X[rows, :] and X[int_row, :].
    """
    nr, nc = X.shape

    row_n = min(int(n_rows), int(nr))
    col_n = min(int(n_cols), int(nc))

    cols = np.sort(
        RNG.choice(
            nc,
            size=col_n,
            replace=False,
        )
    )

    if row_n <= 0 or col_n <= 0:
        return np.empty(
            (row_n, col_n),
            dtype=np.float64,
        )

    # For backed X, use AnnData's chunked row interface. This is specifically
    # designed to iterate over rows of X in backed mode.
    if layer is None and bool(getattr(adata, "isbacked", False)):
        iterator = adata.chunked_X(
            chunk_size=row_n
        )

        try:
            first = next(iterator)
        except StopIteration:
            return np.empty(
                (0, col_n),
                dtype=np.float64,
            )

        # AnnData.chunked_X yields (X_chunk, start, end).
        if isinstance(first, tuple):
            block = first[0]
        else:
            block = first

    else:
        # Important: use a single row slice rather than X[:row_n, :].
        # The single-slice path is compatible with scipy sparse, dense arrays,
        # h5py-like matrices, and AnnData layer objects.
        block = X[:row_n]

    if sparse.issparse(block):
        block = block.tocsr()
        sub = block[:, cols]
        sample = sub.toarray().astype(
            np.float64,
            copy=False,
        )

    else:
        block = np.asarray(block)

        if block.ndim == 1:
            block = block.reshape(1, -1)

        if block.ndim != 2:
            raise RuntimeError(
                f"Unexpected materialized expression block ndim={block.ndim}"
            )

        if block.shape[1] != nc:
            raise RuntimeError(
                f"Unexpected materialized block width={block.shape[1]} "
                f"for expression matrix n_genes={nc}"
            )

        sample = np.asarray(
            block[:, cols],
            dtype=np.float64,
        )

    expected_shape = (
        min(row_n, sample.shape[0]),
        col_n,
    )

    if sample.shape[1] != col_n:
        raise RuntimeError(
            f"Unexpected sampled gene count={sample.shape[1]}; expected={col_n}"
        )

    if sample.shape[0] <= 0:
        raise RuntimeError(
            "Expression sample returned zero rows."
        )

    return sample


def tuple_series(
    frame: pd.DataFrame,
    cols: list[str],
) -> pd.Series:
    if not cols:
        return pd.Series(
            [("__ALL__",)] * len(frame),
            index=frame.index,
            dtype=object,
        )

    arrays = [
        frame[c].astype(str).to_numpy()
        for c in cols
    ]

    return pd.Series(
        list(zip(*arrays)),
        index=frame.index,
        dtype=object,
    )


def build_match_plan(
    pert: pd.DataFrame,
    ctrl: pd.DataFrame,
    keys: list[str],
    min_controls_total: int,
    min_controls_per_stratum: int,
) -> dict[str, Any]:
    missing_columns = [
        k
        for k in keys
        if k not in pert.columns or k not in ctrl.columns
    ]

    if missing_columns:
        return {
            "feasible": False,
            "reason": "keys_unavailable",
            "n_strata": 0,
            "n_controls_union": 0,
            "min_controls_per_observed_stratum": np.nan,
            "strata_json": "{}",
        }

    pert_missing = np.zeros(
        len(pert),
        dtype=bool,
    )

    for key in keys:
        pert_missing |= pert[key].eq(MISSING).to_numpy()

    if pert_missing.any():
        return {
            "feasible": False,
            "reason": "perturbed_key_missing",
            "n_strata": 0,
            "n_controls_union": 0,
            "min_controls_per_observed_stratum": np.nan,
            "strata_json": "{}",
        }

    pert_key = tuple_series(pert, keys)
    ctrl_key = tuple_series(ctrl, keys)

    pert_counts = pert_key.value_counts(sort=False)
    ctrl_counts = ctrl_key.value_counts(sort=False)

    rows = []
    control_union_positions: set[int] = set()
    failed = False
    min_ctrl_observed = None

    for stratum, n_pert in pert_counts.items():
        ctrl_positions = np.flatnonzero(
            ctrl_key.map(lambda x: x == stratum).to_numpy(dtype=bool)
        )
        n_ctrl = int(len(ctrl_positions))

        if min_ctrl_observed is None:
            min_ctrl_observed = n_ctrl
        else:
            min_ctrl_observed = min(
                min_ctrl_observed,
                n_ctrl,
            )

        if n_ctrl < min_controls_per_stratum:
            failed = True

        control_union_positions.update(
            map(int, ctrl_positions.tolist())
        )

        rows.append({
            "stratum": list(map(str, stratum)),
            "n_perturbed": int(n_pert),
            "n_controls": n_ctrl,
        })

    n_controls_union = len(control_union_positions)

    feasible = bool(
        len(rows)
        and not failed
        and n_controls_union >= min_controls_total
    )

    if feasible:
        reason = "ok"
    elif failed:
        reason = "insufficient_controls_per_stratum"
    else:
        reason = "insufficient_total_controls"

    return {
        "feasible": feasible,
        "reason": reason,
        "n_strata": int(len(rows)),
        "n_controls_union": int(n_controls_union),
        "min_controls_per_observed_stratum": (
            int(min_ctrl_observed)
            if min_ctrl_observed is not None
            else np.nan
        ),
        "strata_json": json.dumps(
            rows,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    }


def effective_catalog_row(
    catalog: pd.DataFrame,
    dataset_id: str,
) -> pd.Series:
    if "dataset_id" not in catalog.columns:
        raise KeyError("Historical result2 dataset catalog lacks dataset_id.")

    sub = catalog.loc[
        catalog["dataset_id"].astype(str).eq(dataset_id)
    ]

    if len(sub) != 1:
        raise RuntimeError(
            f"{dataset_id}: expected exactly one historical catalog row, got {len(sub)}"
        )

    return sub.iloc[0]


for p in [
    DATASET_CATALOG,
    EFFECT_CONFIG,
    ATOM_AUTHORITY,
    BIO_KEY_MAP,
    PATH_AUTHORITY,
    NUISANCE_POLICY,
    FROZEN_SELECTION,
]:
    if not p.is_file():
        raise FileNotFoundError(p)


catalog = pd.read_csv(
    DATASET_CATALOG,
    sep="\t",
    low_memory=False,
)

atoms = pd.read_csv(
    ATOM_AUTHORITY,
    sep="\t",
    low_memory=False,
)

bio_key_map = pd.read_csv(
    BIO_KEY_MAP,
    sep="\t",
    low_memory=False,
)

path_auth = pd.read_csv(
    PATH_AUTHORITY,
    sep="\t",
    low_memory=False,
)

frozen_selection = pd.read_csv(
    FROZEN_SELECTION,
    sep="\t",
    low_memory=False,
)

with EFFECT_CONFIG.open() as f:
    cfg = yaml.safe_load(f)

policy = json.loads(
    NUISANCE_POLICY.read_text(
        encoding="utf-8"
    )
)

if policy.get("status") != "FROZEN":
    raise RuntimeError("Nuisance policy is not FROZEN.")

min_pert = int(
    policy["support_thresholds"][
        "min_perturbed_cells_per_response_atom"
    ]
)

min_ctrl = int(
    policy["support_thresholds"][
        "min_biological_context_control_cells"
    ]
)

min_ctrl_stratum = int(
    policy["support_thresholds"][
        "min_controls_per_nuisance_stratum"
    ]
)

if [
    x["name"]
    for x in policy["primary_nuisance_hierarchy"]
] != ["donor+batch", "donor"]:
    raise RuntimeError(
        "Unexpected frozen nuisance hierarchy."
    )

atoms = atoms.copy()
atoms["dataset_id"] = atoms["dataset_id"].astype(str)
atoms["response_atom_id"] = atoms["response_atom_id"].astype(str)
atoms["condition_id"] = atoms["condition_id"].astype(str)

for f in BIO_FIELDS:
    atoms[f] = norm_series(atoms[f])

if atoms["response_atom_id"].duplicated().any():
    raise RuntimeError("Response atom authority IDs are not unique.")

path_auth = path_auth.copy()
path_auth["dataset_id"] = path_auth["dataset_id"].astype(str)

if "h5ad_path" not in path_auth.columns:
    raise KeyError("Path authority lacks h5ad_path.")

path_map = path_auth.set_index("dataset_id")["h5ad_path"].astype(str).to_dict()

# Frozen preflight selected levels are stored under provisional IDs.
# The final atom authority retains response_atom_id_provisional.
if "response_atom_id_provisional" not in atoms.columns:
    raise KeyError(
        "Final atom authority lacks response_atom_id_provisional."
    )

frozen_selection = frozen_selection.copy()
frozen_selection["response_atom_id"] = frozen_selection["response_atom_id"].astype(str)

frozen_level_by_provisional = frozen_selection.set_index(
    "response_atom_id"
)["selected_historical_collapsed"].astype(str).to_dict()


# ======================================================================
# Dataset loop
# ======================================================================

try:
    import anndata as ad
except Exception as exc:
    raise RuntimeError(
        "anndata is required in the project py311 environment."
    ) from exc


plan_rows: list[dict[str, Any]] = []
dataset_rows: list[dict[str, Any]] = []
expr_rows: list[dict[str, Any]] = []

all_pass = True


for dataset_id in sorted(
    atoms["dataset_id"].unique().tolist()
):
    print(f"[dataset] {dataset_id}", flush=True)

    if dataset_id not in path_map:
        raise RuntimeError(
            f"{dataset_id}: missing resolved h5ad path."
        )

    h5ad_path = Path(
        path_map[dataset_id]
    )

    if not h5ad_path.is_file():
        raise FileNotFoundError(h5ad_path)

    cat = effective_catalog_row(
        catalog,
        dataset_id,
    )

    condition_col = str(
        cat.get(
            "condition_id_column",
            cfg.get("schema", {}).get(
                "condition_id_column",
                DEFAULT_CONDITION_COL,
            ),
        )
    )

    if (
        not condition_col
        or condition_col.lower()
        in {"nan", "none"}
    ):
        condition_col = DEFAULT_CONDITION_COL

    control_col = str(
        cat.get(
            "control_column",
            cfg.get("schema", {}).get(
                "control_column",
                DEFAULT_CONTROL_COL,
            ),
        )
    )

    if (
        not control_col
        or control_col.lower()
        in {"nan", "none"}
    ):
        control_col = DEFAULT_CONTROL_COL

    control_value = cat.get(
        "control_value",
        cfg.get("schema", {}).get(
            "control_value",
            True,
        ),
    )

    raw_layer = cat.get(
        "layer",
        cfg.get("expression", {}).get(
            "layer",
            None,
        ),
    )

    layer = None

    if pd.notna(raw_layer):
        s = str(raw_layer).strip()
        if s.lower() not in {
            "",
            "none",
            "nan",
            "x",
        }:
            layer = s

    expression_state = str(
        cat.get(
            "expression_state",
            cfg.get("expression", {}).get(
                "data_state",
                "raw_counts",
            ),
        )
    ).strip().lower()

    if expression_state in {
        "",
        "nan",
        "none",
    }:
        expression_state = str(
            cfg.get("expression", {}).get(
                "data_state",
                "raw_counts",
            )
        ).strip().lower()

    if expression_state not in {
        "raw_counts",
        "log1p_library_normalized",
    }:
        raise RuntimeError(
            f"{dataset_id}: unsupported expression_state={expression_state!r}"
        )

    # ------------------------------------------------------------------
    # Dataset-specific expression authority override.
    #
    # Historical result2 catalog values are retained as provenance, but the
    # current corrected-R2 authority supplied for these datasets takes
    # precedence.  This is explicit, recorded, and later verified against the
    # actual h5ad structure; it is not a silent override.
    # ------------------------------------------------------------------
    catalog_resolved_layer = layer
    catalog_resolved_expression_state = expression_state

    authority_override_applied = False
    authority_description = ""

    if dataset_id in DATASET_EXPRESSION_AUTHORITY:
        authority = DATASET_EXPRESSION_AUTHORITY[dataset_id]
        authority_override_applied = True
        authority_description = authority["description"]

        expression_state = authority["expression_state"]
        layer = authority["layer"]

    use_hvg = cat.get(
        "use_hvg",
        cfg.get("effect_cache", {}).get(
            "use_hvg",
            True,
        ),
    )

    try:
        use_hvg_bool = bool(
            parse_bool_like(use_hvg)
            if parse_bool_like(use_hvg) is not None
            else bool(use_hvg)
        )
    except Exception:
        use_hvg_bool = True

    n_hvg = int(
        cat.get(
            "n_hvg",
            cfg.get("effect_cache", {}).get(
                "n_hvg",
                3000,
            ),
        )
        if pd.notna(
            cat.get(
                "n_hvg",
                np.nan,
            )
        )
        else cfg.get(
            "effect_cache",
            {},
        ).get(
            "n_hvg",
            3000,
        )
    )

    target_sum = float(
        cfg.get("expression", {}).get(
            "target_sum",
            10000.0,
        )
    )

    pseudocount = float(
        cfg.get("effect_cache", {}).get(
            "pseudocount",
            1.0,
        )
    )

    adata = ad.read_h5ad(
        h5ad_path,
        backed="r",
    )

    obs = adata.obs.copy()

    # Verify that the corrected-R2 expression authority is physically present
    # in the h5ad before any expression values are sampled.
    if authority_override_applied:
        if layer is None:
            if adata.X is None:
                raise RuntimeError(
                    f"{dataset_id}: authority requires adata.X, but X is missing."
                )
        else:
            if layer not in adata.layers:
                raise RuntimeError(
                    f"{dataset_id}: authority requires layer {layer!r}, "
                    f"but available layers are {list(adata.layers.keys())}"
                )

    if condition_col not in obs.columns:
        raise KeyError(
            f"{dataset_id}: missing {condition_col}"
        )

    if control_col not in obs.columns:
        raise KeyError(
            f"{dataset_id}: missing {control_col}"
        )

    X = get_expr_source(
        adata,
        layer,
    )

    # --------------------------------------------------------------
    # Expression input QC: light but real matrix access.
    # --------------------------------------------------------------

    sample = read_small_expression_sample(
        adata=adata,
        X=X,
        layer=layer,
        n_rows=64,
        n_cols=512,
    )

    finite_fraction = float(
        np.isfinite(sample).mean()
    )

    nonnegative_fraction = float(
        (sample >= 0).mean()
    )

    nonzero = sample[
        np.isfinite(sample)
        & (sample != 0)
    ]

    if len(nonzero):
        integer_like_fraction = float(
            (
                np.abs(
                    nonzero
                    - np.rint(nonzero)
                )
                <= 1e-6
            ).mean()
        )
    else:
        integer_like_fraction = 1.0

    sample_positive_row_fraction = float(
        (
            np.nansum(
                sample,
                axis=1,
            )
            > 0
        ).mean()
    )

    if expression_state == "raw_counts":
        expr_status = (
            "PASS"
            if (
                finite_fraction == 1.0
                and nonnegative_fraction == 1.0
                and integer_like_fraction >= 0.99
                and sample_positive_row_fraction >= 0.90
            )
            else "FAIL"
        )
    else:
        expr_status = (
            "PASS"
            if (
                finite_fraction == 1.0
                and nonnegative_fraction == 1.0
                and sample_positive_row_fraction >= 0.90
            )
            else "FAIL"
        )

    if expr_status != "PASS":
        all_pass = False

    expr_rows.append({
        "dataset_id": dataset_id,
        "h5ad_path": str(h5ad_path),
        "expression_source": (
            "X"
            if layer is None
            else f"layers[{layer}]"
        ),
        "expression_state": expression_state,
        "catalog_resolved_expression_source": (
            "X"
            if catalog_resolved_layer is None
            else f"layers[{catalog_resolved_layer}]"
        ),
        "catalog_resolved_expression_state": catalog_resolved_expression_state,
        "dataset_specific_expression_authority": authority_description,
        "dataset_specific_source_category": (
            DATASET_EXPRESSION_AUTHORITY.get(dataset_id, {}).get(
                "source_category", ""
            )
        ),
        "dataset_specific_authority_override_applied": authority_override_applied,
        "authority_differs_from_catalog": bool(
            authority_override_applied
            and (
                layer != catalog_resolved_layer
                or expression_state != catalog_resolved_expression_state
            )
        ),
        "n_cells": int(X.shape[0]),
        "n_genes": int(X.shape[1]),
        "var_names_unique": bool(
            pd.Index(
                adata.var_names.astype(str)
            ).is_unique
        ),
        "sample_rows": int(sample.shape[0]),
        "sample_genes": int(sample.shape[1]),
        "finite_fraction": finite_fraction,
        "nonnegative_fraction": nonnegative_fraction,
        "integer_like_nonzero_fraction": integer_like_fraction,
        "sample_positive_row_fraction": sample_positive_row_fraction,
        "use_hvg": use_hvg_bool,
        "n_hvg": n_hvg,
        "target_sum": target_sum,
        "pseudocount": pseudocount,
        "status": expr_status,
    })

    # --------------------------------------------------------------
    # Normalize metadata required for exact frozen control plans.
    # --------------------------------------------------------------

    work = pd.DataFrame(
        index=obs.index
    )

    work["condition_id"] = norm_series(
        obs[condition_col]
    )

    control_mask = control_mask_from_series(
        obs[control_col],
        control_value,
    )

    work["is_control"] = control_mask

    for col in [
        DONOR_COL,
        BATCH_COL,
        PLATFORM_COL,
    ]:
        if col in obs.columns:
            work[col] = norm_series(
                obs[col]
            )
        else:
            work[col] = MISSING

    # Available biological fields are dataset-specific and frozen in
    # the R1->obs biological key map.
    ds_bio_key_map = bio_key_map.loc[
        bio_key_map["dataset_id"]
        .astype(str)
        .eq(dataset_id)
    ].copy()

    if ds_bio_key_map.empty:
        raise RuntimeError(
            f"{dataset_id}: biological key map is empty."
        )

    available_field_strings = (
        ds_bio_key_map[
            "available_obs_bio_fields"
        ]
        .astype(str)
        .unique()
        .tolist()
    )

    if len(available_field_strings) != 1:
        raise RuntimeError(
            f"{dataset_id}: inconsistent available biological fields."
        )

    available_fields = [
        x
        for x in available_field_strings[0].split(";")
        if x
    ]

    available_obs_cols = [
        OBS_BIO[x]
        for x in available_fields
    ]

    for short, col in zip(
        available_fields,
        available_obs_cols,
    ):
        if col not in obs.columns:
            raise KeyError(
                f"{dataset_id}: biological key map expects {col}, "
                "but it is absent from obs."
            )
        work[col] = norm_series(
            obs[col]
        )

    # Build exact observed-key -> full authoritative biological tuple.
    key_lookup: dict[tuple[str, ...], tuple[str, ...]] = {}

    for _, row in ds_bio_key_map.iterrows():
        observed_key = tuple(
            norm_value(
                row[f"key_{field}"]
            )
            for field in available_fields
        )

        full_tuple = tuple(
            norm_value(
                row[f"authority_{field}"]
            )
            for field in BIO_FIELDS
        )

        if observed_key in key_lookup:
            if key_lookup[observed_key] != full_tuple:
                raise RuntimeError(
                    f"{dataset_id}: observed biological key maps to "
                    "multiple full authority tuples."
                )
        else:
            key_lookup[observed_key] = full_tuple

    observed_keys = tuple_series(
        work,
        available_obs_cols,
    )

    full_bio_values = []

    missing_control_key_n = 0

    for is_control, key in zip(
        control_mask,
        observed_keys.tolist(),
    ):
        if key in key_lookup:
            full_bio_values.append(
                key_lookup[key]
            )
        else:
            # Perturbed cells should also map. Keep explicit missing.
            full_bio_values.append(
                None
            )
            if is_control:
                missing_control_key_n += 1

    if missing_control_key_n:
        raise RuntimeError(
            f"{dataset_id}: {missing_control_key_n} control cells "
            "lack authoritative biological-key crosswalk."
        )

    full_bio_series = pd.Series(
        full_bio_values,
        index=work.index,
        dtype=object,
    )

    # --------------------------------------------------------------
    # Final response atom -> exact cell and control plan.
    # --------------------------------------------------------------

    ds_atoms = atoms.loc[
        atoms["dataset_id"].eq(dataset_id)
    ].copy()

    selected_level_counts = Counter()

    dataset_eligible_n = 0
    dataset_plan_pass_n = 0

    for _, atom in ds_atoms.iterrows():
        rid = str(
            atom["response_atom_id"]
        )

        provisional_id = str(
            atom[
                "response_atom_id_provisional"
            ]
        )

        condition_id = str(
            atom["condition_id"]
        )

        full_bio_tuple = tuple(
            norm_value(
                atom[field]
            )
            for field in BIO_FIELDS
        )

        pert_mask = (
            ~control_mask
            & work["condition_id"]
            .eq(condition_id)
            .to_numpy()
            & full_bio_series
            .map(lambda x: x == full_bio_tuple)
            .to_numpy()
        )

        ctrl_mask = (
            control_mask
            & full_bio_series
            .map(lambda x: x == full_bio_tuple)
            .to_numpy()
        )

        pert_df = work.loc[
            pert_mask
        ].copy()

        ctrl_df = work.loc[
            ctrl_mask
        ].copy()

        n_pert_cells = int(
            len(pert_df)
        )

        n_bio_controls = int(
            len(ctrl_df)
        )

        frozen_support = str(
            atom[
                "support_status_pre_nuisance"
            ]
        )

        frozen_level = frozen_level_by_provisional.get(
            provisional_id,
            "",
        )

        selected_level = ""
        selected_result = None

        donor_batch_result = build_match_plan(
            pert_df,
            ctrl_df,
            [DONOR_COL, BATCH_COL],
            min_ctrl,
            min_ctrl_stratum,
        )

        donor_result = build_match_plan(
            pert_df,
            ctrl_df,
            [DONOR_COL],
            min_ctrl,
            min_ctrl_stratum,
        )

        if frozen_support == "BIO_CONTROL_ELIGIBLE":
            dataset_eligible_n += 1

            if donor_batch_result["feasible"]:
                selected_level = "donor+batch"
                selected_result = donor_batch_result

            elif donor_result["feasible"]:
                selected_level = "donor"
                selected_result = donor_result

            else:
                selected_level = "UNSUPPORTED_NUISANCE_MATCH"
                selected_result = donor_result

            # This must exactly reproduce the frozen preflight selection.
            if selected_level != frozen_level:
                all_pass = False
                plan_status = "FAIL_FROZEN_LEVEL_MISMATCH"
            else:
                plan_status = "PASS"
                dataset_plan_pass_n += 1

        else:
            selected_level = "NOT_ELIGIBLE_PRE_NUISANCE"
            selected_result = {
                "n_strata": 0,
                "n_controls_union": 0,
                "min_controls_per_observed_stratum": np.nan,
                "strata_json": "{}",
            }

            plan_status = "PASS_NOT_ELIGIBLE"

        selected_level_counts[
            selected_level
        ] += 1

        plan_rows.append({
            "dataset_id": dataset_id,
            "response_atom_id": rid,
            "response_atom_id_provisional": provisional_id,
            "condition_id": condition_id,
            "support_status_pre_nuisance": frozen_support,
            "n_perturbed_cells_reconstructed": n_pert_cells,
            "n_bio_context_controls_reconstructed": n_bio_controls,
            "frozen_selected_level": frozen_level,
            "selected_level_reconstructed": selected_level,
            "donor_batch_feasible": donor_batch_result["feasible"],
            "donor_batch_reason": donor_batch_result["reason"],
            "donor_batch_n_strata": donor_batch_result["n_strata"],
            "donor_batch_n_controls_union": donor_batch_result["n_controls_union"],
            "donor_feasible": donor_result["feasible"],
            "donor_reason": donor_result["reason"],
            "donor_n_strata": donor_result["n_strata"],
            "donor_n_controls_union": donor_result["n_controls_union"],
            "selected_n_strata": selected_result["n_strata"],
            "selected_n_controls_union": selected_result["n_controls_union"],
            "selected_min_controls_per_observed_stratum": selected_result[
                "min_controls_per_observed_stratum"
            ],
            "selected_strata_json": selected_result["strata_json"],
            "status": plan_status,
        })

    ds_plan_rows = [
        x
        for x in plan_rows
        if x["dataset_id"] == dataset_id
    ]

    ds_fail_n = sum(
        x["status"].startswith("FAIL")
        for x in ds_plan_rows
    )

    if ds_fail_n:
        all_pass = False

    dataset_rows.append({
        "dataset_id": dataset_id,
        "response_atom_n": int(
            len(ds_atoms)
        ),
        "eligible_atom_n": int(
            dataset_eligible_n
        ),
        "eligible_plan_pass_n": int(
            dataset_plan_pass_n
        ),
        "plan_fail_n": int(
            ds_fail_n
        ),
        "selected_donor_batch_n": int(
            selected_level_counts[
                "donor+batch"
            ]
        ),
        "selected_donor_n": int(
            selected_level_counts[
                "donor"
            ]
        ),
        "selected_unsupported_nuisance_n": int(
            selected_level_counts[
                "UNSUPPORTED_NUISANCE_MATCH"
            ]
        ),
        "not_eligible_pre_nuisance_n": int(
            selected_level_counts[
                "NOT_ELIGIBLE_PRE_NUISANCE"
            ]
        ),
        "expression_state": expression_state,
        "expression_source": (
            "X"
            if layer is None
            else f"layers[{layer}]"
        ),
        "n_cells": int(
            X.shape[0]
        ),
        "n_genes": int(
            X.shape[1]
        ),
        "expression_qc_status": expr_status,
        "status": (
            "PASS"
            if (
                ds_fail_n == 0
                and expr_status == "PASS"
            )
            else "FAIL"
        ),
    })

    try:
        adata.file.close()
    except Exception:
        pass


plan = pd.DataFrame(
    plan_rows
).sort_values(
    [
        "dataset_id",
        "condition_id",
        "response_atom_id",
    ]
).reset_index(
    drop=True
)

dataset_summary = pd.DataFrame(
    dataset_rows
).sort_values(
    "dataset_id"
).reset_index(
    drop=True
)

expr_summary = pd.DataFrame(
    expr_rows
).sort_values(
    "dataset_id"
).reset_index(
    drop=True
)

plan.to_csv(
    PLAN_TSV,
    sep="\t",
    index=False,
)

dataset_summary.to_csv(
    DATASET_TSV,
    sep="\t",
    index=False,
)

expr_summary.to_csv(
    EXPR_TSV,
    sep="\t",
    index=False,
)


# ======================================================================
# Global frozen-contract checks
# ======================================================================

eligible_plan = plan.loc[
    plan[
        "support_status_pre_nuisance"
    ].astype(str).eq(
        "BIO_CONTROL_ELIGIBLE"
    )
].copy()

eligible_n = int(
    len(eligible_plan)
)

selected_donor_batch_n = int(
    eligible_plan[
        "selected_level_reconstructed"
    ].eq(
        "donor+batch"
    ).sum()
)

selected_donor_n = int(
    eligible_plan[
        "selected_level_reconstructed"
    ].eq(
        "donor"
    ).sum()
)

unsupported_nuisance_n = int(
    eligible_plan[
        "selected_level_reconstructed"
    ].eq(
        "UNSUPPORTED_NUISANCE_MATCH"
    ).sum()
)

frozen_level_mismatch_n = int(
    eligible_plan[
        "status"
    ].eq(
        "FAIL_FROZEN_LEVEL_MISMATCH"
    ).sum()
)

expected_eligible_n = int(
    policy["coverage"][
        "eligible_response_atom_n"
    ]
)

expected_db_n = int(
    policy["coverage"][
        "donor_batch_selected_n"
    ]
)

expected_donor_n = int(
    policy["coverage"][
        "donor_selected_n"
    ]
)

global_contract_pass = (
    eligible_n == expected_eligible_n
    and selected_donor_batch_n == expected_db_n
    and selected_donor_n == expected_donor_n
    and unsupported_nuisance_n == 0
    and frozen_level_mismatch_n == 0
    and dataset_summary["status"].eq("PASS").all()
    and expr_summary["status"].eq("PASS").all()
)

if not global_contract_pass:
    all_pass = False

status = (
    "PASS"
    if all_pass
    else "FAIL"
)


lines = [
    "PERTURBCONTEXTALIGN R2 MEASURED-RESPONSE EFFECT PREFLIGHT v1",
    "=" * 120,
    "",
    "MODE",
    "-" * 120,
    "expression_access=LIGHTWEIGHT_SAMPLE_QC_ONLY",
    "formal_hvg_computation=FALSE",
    "formal_logfc_computation=FALSE",
    "response_similarity_computed=FALSE",
    "historical_numerical_cache_reused=FALSE",
    "",
    "FROZEN CONTRACT",
    "-" * 120,
    f"min_perturbed_cells={min_pert}",
    "expression_authority_precedence=corrected_R2_dataset_specific_authority > historical_result2_catalog > config_default",
    "combo_sciplex_source_category=raw_counts_in_layer; actual_expression=layers['counts']; state=raw_counts; X_is_not_raw_counts=TRUE",
    "kaggle_cross_patient_source_category=log1p_library_normalized; actual_expression=X; state=log1p_library_normalized; renormalize=FALSE; relog1p=FALSE",
    f"min_bio_context_controls={min_ctrl}",
    f"min_controls_per_nuisance_stratum={min_ctrl_stratum}",
    "primary_nuisance_hierarchy=donor+batch -> donor -> unsupported",
    "global_control_fallback=PROHIBITED",
    "",
    "OVERALL CONTROL PLAN",
    "-" * 120,
    f"response_atom_n={len(plan)}",
    f"eligible_response_atom_n={eligible_n}",
    f"selected_donor_batch_n={selected_donor_batch_n}",
    f"selected_donor_n={selected_donor_n}",
    f"unsupported_nuisance_n={unsupported_nuisance_n}",
    f"frozen_level_mismatch_n={frozen_level_mismatch_n}",
    "",
    "EXPECTED FROZEN COUNTS",
    "-" * 120,
    f"expected_eligible_response_atom_n={expected_eligible_n}",
    f"expected_donor_batch_selected_n={expected_db_n}",
    f"expected_donor_selected_n={expected_donor_n}",
    "",
    "DATASET SUMMARY",
    "-" * 120,
    dataset_summary.to_string(index=False),
    "",
    "EXPRESSION INPUT QC",
    "-" * 120,
    expr_summary[
        [
            "dataset_id",
            "expression_source",
            "expression_state",
            "catalog_resolved_expression_source",
            "catalog_resolved_expression_state",
            "dataset_specific_source_category",
            "dataset_specific_authority_override_applied",
            "authority_differs_from_catalog",
            "n_cells",
            "n_genes",
            "var_names_unique",
            "finite_fraction",
            "nonnegative_fraction",
            "integer_like_nonzero_fraction",
            "sample_positive_row_fraction",
            "use_hvg",
            "n_hvg",
            "target_sum",
            "pseudocount",
            "status",
        ]
    ].to_string(index=False),
    "",
    "STATUS",
    "-" * 120,
    f"R2_MEASURED_RESPONSE_EFFECT_PREFLIGHT={status}",
]

if status == "PASS":
    lines += [
        "CONTROL_PLAN_REPRODUCTION=PASS",
        "EXPRESSION_INPUT_PREFLIGHT=PASS",
        "NEXT=R2_MEASURED_RESPONSE_EFFECT_CONSTRUCTION_v1",
    ]
else:
    lines += [
        "NEXT=REVIEW_PREFLIGHT_FAILURE_BEFORE_FORMAL_EFFECT_CONSTRUCTION",
    ]

REPORT_TXT.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)

manifest = {
    "version": "R2_MEASURED_RESPONSE_EFFECT_PREFLIGHT_v1",
    "script_revision": "v1.6_chunked_backed_X_sampling_plus_prior_bugfixes",
    "status": status,
    "mode": {
        "expression_access": "lightweight_sample_qc_only",
        "formal_hvg_computation": False,
        "formal_logfc_computation": False,
        "historical_numerical_cache_reused": False,
    },
    "inputs": {
        "dataset_catalog": {
            "path": str(DATASET_CATALOG),
            "sha256": sha256_file(DATASET_CATALOG),
        },
        "effect_config": {
            "path": str(EFFECT_CONFIG),
            "sha256": sha256_file(EFFECT_CONFIG),
        },
        "response_atom_authority": {
            "path": str(ATOM_AUTHORITY),
            "sha256": sha256_file(ATOM_AUTHORITY),
        },
        "biological_key_map": {
            "path": str(BIO_KEY_MAP),
            "sha256": sha256_file(BIO_KEY_MAP),
        },
        "path_authority": {
            "path": str(PATH_AUTHORITY),
            "sha256": sha256_file(PATH_AUTHORITY),
        },
        "nuisance_policy": {
            "path": str(NUISANCE_POLICY),
            "sha256": sha256_file(NUISANCE_POLICY),
        },
        "frozen_selection": {
            "path": str(FROZEN_SELECTION),
            "sha256": sha256_file(FROZEN_SELECTION),
        },
    },
    "outputs": {
        "control_plan": str(PLAN_TSV),
        "dataset_summary": str(DATASET_TSV),
        "expression_input_qc": str(EXPR_TSV),
        "report": str(REPORT_TXT),
    },
}

MANIFEST_JSON.write_text(
    json.dumps(
        manifest,
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)

print(
    REPORT_TXT.read_text(
        encoding="utf-8"
    )
)

if status != "PASS":
    raise SystemExit(2)
