from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pandas as pd
import yaml
from scipy import sparse


# ======================================================================================
# CLI / paths
# ======================================================================================


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--mode", choices=["preflight", "build"], required=True)
    ap.add_argument("--datasets", nargs="*", default=[])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--row-block", type=int, default=2048)
    ap.add_argument("--csc-col-block", type=int, default=128)
    return ap.parse_args()


ARGS = parse_args()
ROOT = Path(ARGS.root).expanduser().resolve()
NEW = ROOT / "0_ProjectCodeAndLogs" / "PerturbContextAlign_R2_corrected_v1_20261002"
OLD = ROOT / "result2"

PATH_AUTH = NEW / "02_input_authority" / "R2_DATASET_AUTHORITY_WITH_H5AD_v1.tsv"
ATOM_AUTH = NEW / "02_input_authority" / "R2_RESPONSE_ATOM_AUTHORITY_v1.tsv"
BIO_KEY_MAP = NEW / "02_input_authority" / "R2_BIOLOGICAL_CONTEXT_KEY_MAP_v1.tsv"
SELECTION_TSV = NEW / "03_response_atom_audit" / "R2_NUISANCE_HIERARCHY_SELECTION_v1.tsv"
NUISANCE_POLICY = NEW / "01_contract" / "R2_NUISANCE_CONTROL_POLICY_v1.json"
DATASET_CATALOG = OLD / "configs" / "result2_dataset_catalog.tsv"
EFFECT_CONFIG = OLD / "configs" / "result2_effect_cache_config.yaml"

PREFLIGHT_ROOT = NEW / "04_response_geometry" / "00_preflight_v2"
EFFECT_ROOT = NEW / "04_response_geometry" / "01_effect_v1"
PREFLIGHT_REPORT = PREFLIGHT_ROOT / "R2_MEASURED_RESPONSE_EFFECT_PREFLIGHT_v2.txt"
PREFLIGHT_TSV = PREFLIGHT_ROOT / "R2_MEASURED_RESPONSE_EFFECT_PREFLIGHT_v2.tsv"
PREFLIGHT_MANIFEST = PREFLIGHT_ROOT / "R2_MEASURED_RESPONSE_EFFECT_PREFLIGHT_MANIFEST_v2.json"
GLOBAL_AUDIT_TSV = EFFECT_ROOT / "R2_MEASURED_RESPONSE_EFFECT_AUDIT_v1.tsv"
GLOBAL_AUDIT_TXT = EFFECT_ROOT / "R2_MEASURED_RESPONSE_EFFECT_AUDIT_v1.txt"
GLOBAL_MANIFEST = EFFECT_ROOT / "R2_MEASURED_RESPONSE_EFFECT_MANIFEST_v1.json"

SEED = 20260711
SCRIPT_VERSION = "R2_DIRECT_H5_EFFECT_BUILDER_v1"
MISSING = "__NA__"
MISSING_STRINGS = {"", "nan", "none", "null", "<na>", "na"}
TRUE_SET = {"1", "true", "t", "yes", "y", "control", "ctrl"}
FALSE_SET = {"0", "false", "f", "no", "n", "perturbed", "perturb", "case"}
SEP = "\x1f"
ATOM_SEP = "\x1e"
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

# These are corrected-R2 authorities, not literal guesses derived from defaults.
DATASET_EXPRESSION_AUTHORITY = {
    "combo_sciplex": {
        "source_category": "raw_counts_in_layer",
        "actual_source": "layers/counts",
        "expression_state": "raw_counts",
        "notes": "raw counts are in adata.layers['counts']; X is not raw-count authority",
    },
    "kaggle_cross_patient": {
        "source_category": "log1p_library_normalized",
        "actual_source": "X",
        "expression_state": "log1p_library_normalized",
        "notes": "X is released library-normalized log1p; no second normalization/log1p",
    },
}


# ======================================================================================
# Generic helpers
# ======================================================================================


def log(msg: str) -> None:
    print(msg, flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def norm_value(v: Any) -> str:
    if pd.isna(v):
        return MISSING
    s = str(v).strip()
    return MISSING if s.lower() in MISSING_STRINGS else s


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


def control_mask_from_series(series: pd.Series, control_value: Any = True) -> np.ndarray:
    target_bool = parse_bool_like(control_value)
    parsed = series.map(parse_bool_like)
    if target_bool is not None and parsed.notna().mean() >= 0.95:
        return parsed.fillna(False).eq(target_bool).to_numpy(dtype=bool)
    return norm_series(series).eq(norm_value(control_value)).to_numpy(dtype=bool)


def join_key(values: Iterable[Any]) -> str:
    return SEP.join(norm_value(v) for v in values)


def full_bio_key_from_row(row: pd.Series) -> str:
    return join_key(row[f] for f in BIO_FIELDS)


def parse_catalog_layer(v: Any) -> str | None:
    if pd.isna(v):
        return None
    s = str(v).strip()
    if s.lower() in {"", "none", "nan", "x"}:
        return None
    return s


def expression_authority(dataset_id: str, catalog_row: pd.Series, cfg: dict[str, Any]) -> dict[str, Any]:
    hist_layer = parse_catalog_layer(catalog_row.get("layer", cfg.get("expression", {}).get("layer")))
    hist_state = str(catalog_row.get("expression_state", cfg.get("expression", {}).get("data_state", "raw_counts"))).strip().lower()
    if hist_state in {"", "nan", "none"}:
        hist_state = str(cfg.get("expression", {}).get("data_state", "raw_counts")).strip().lower()
    historical_source = "X" if hist_layer is None else f"layers/{hist_layer}"
    if dataset_id in DATASET_EXPRESSION_AUTHORITY:
        a = dict(DATASET_EXPRESSION_AUTHORITY[dataset_id])
        a.update({
            "authority_source": "explicit_corrected_R2",
            "historical_catalog_source": historical_source,
            "historical_catalog_state": hist_state,
        })
        return a
    return {
        "source_category": "historical_catalog",
        "actual_source": historical_source,
        "expression_state": hist_state,
        "notes": "",
        "authority_source": "historical_result2_catalog",
        "historical_catalog_source": historical_source,
        "historical_catalog_state": hist_state,
    }


def save_dense_bundle(path: Path, X: np.ndarray, row_ids: list[str], feature_names: list[str], metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=np.asarray(X, dtype=np.float32),
        condition_ids=np.asarray([str(x) for x in row_ids], dtype=object),
        response_atom_ids=np.asarray([str(x) for x in row_ids], dtype=object),
        feature_names=np.asarray([str(x) for x in feature_names], dtype=object),
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True), dtype=object),
    )


# ======================================================================================
# Direct HDF5 expression reader. AnnData is never used to index X/layers.
# ======================================================================================


class H5ExpressionSource:
    def __init__(self, h5ad_path: Path, source: str):
        self.path = Path(h5ad_path)
        self.source = str(source)
        self.h5 = h5py.File(self.path, "r")
        if self.source == "X":
            if "X" not in self.h5:
                raise KeyError(f"{self.path}: X missing")
            self.node = self.h5["X"]
        elif self.source.startswith("layers/"):
            key = self.source.split("/", 1)[1]
            if "layers" not in self.h5 or key not in self.h5["layers"]:
                avail = sorted(self.h5["layers"].keys()) if "layers" in self.h5 else []
                raise KeyError(f"{self.path}: layer {key!r} missing; available={avail}")
            self.node = self.h5["layers"][key]
        else:
            raise ValueError(f"Unsupported source={source!r}")

        if isinstance(self.node, h5py.Dataset):
            self.kind = "dense"
            self.shape = tuple(map(int, self.node.shape))
            self.dtype = str(self.node.dtype)
            self.indptr = None
        elif isinstance(self.node, h5py.Group):
            enc = self.node.attrs.get("encoding-type", "")
            if isinstance(enc, bytes):
                enc = enc.decode()
            if enc not in {"csr_matrix", "csc_matrix"}:
                raise ValueError(f"Unsupported sparse encoding {enc!r} at {source}")
            self.kind = "csr" if enc == "csr_matrix" else "csc"
            shape = self.node.attrs.get("shape")
            if shape is None:
                raise ValueError(f"Sparse source {source} lacks shape attr")
            self.shape = tuple(map(int, list(shape)))
            self.dtype = str(self.node["data"].dtype)
            self.indptr = np.asarray(self.node["indptr"][:], dtype=np.int64)
        else:
            raise TypeError(type(self.node))

        if len(self.shape) != 2:
            raise ValueError(f"Expression source must be 2D; shape={self.shape}")
        self.n_rows, self.n_cols = self.shape

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def close(self):
        try:
            self.h5.close()
        except Exception:
            pass

    @property
    def nnz(self) -> int | None:
        if self.kind in {"csr", "csc"}:
            return int(self.node["data"].shape[0])
        return None

    def metadata(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "storage_kind": self.kind,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "nnz": self.nnz,
        }

    def stored_value_sample(self, n: int = 200000, seed: int = SEED) -> np.ndarray:
        rng = np.random.default_rng(seed)
        if self.kind in {"csr", "csc"}:
            ds = self.node["data"]
            total = int(ds.shape[0])
            if total <= n:
                return np.asarray(ds[:], dtype=np.float64)
            # Contiguous sampling avoids expensive HDF5 point-selection on
            # hundreds of thousands of random offsets. This sample is QC only
            # and never changes effect values.
            start = int(rng.integers(0, total - int(n) + 1))
            return np.asarray(ds[start : start + int(n)], dtype=np.float64)
        # Dense: take a random contiguous row window, then sample its flattened values.
        rows_needed = min(self.n_rows, max(1, int(math.ceil(n / max(self.n_cols, 1)))))
        start_max = max(0, self.n_rows - rows_needed)
        start = int(rng.integers(0, start_max + 1)) if start_max else 0
        block = np.asarray(self.node[start : start + rows_needed, :], dtype=np.float64).reshape(-1)
        if block.size > n:
            take = np.sort(rng.choice(block.size, size=int(n), replace=False))
            block = block[take]
        return block

    def read_small_block(self, n_rows: int = 64, n_cols: int = 512) -> np.ndarray:
        r = min(int(n_rows), self.n_rows)
        c = min(int(n_cols), self.n_cols)
        if self.kind == "dense":
            return np.asarray(self.node[:r, :c], dtype=np.float64)
        if self.kind == "csr":
            block = self._csr_row_block(0, r)
            return block[:, :c].toarray().astype(np.float64, copy=False)
        block = self._csc_contiguous_col_block(0, c)
        return block[:r, :].toarray().astype(np.float64, copy=False)

    def _csr_row_block(self, start: int, stop: int) -> sparse.csr_matrix:
        if self.kind != "csr":
            raise TypeError("not CSR")
        start = int(start); stop = int(stop)
        lo = int(self.indptr[start]); hi = int(self.indptr[stop])
        data = np.asarray(self.node["data"][lo:hi])
        indices = np.asarray(self.node["indices"][lo:hi], dtype=np.int32)
        ptr = np.asarray(self.indptr[start : stop + 1] - lo, dtype=np.int64)
        return sparse.csr_matrix((data, indices, ptr), shape=(stop - start, self.n_cols))

    def _csr_selected_rows(self, rows: np.ndarray) -> sparse.csr_matrix:
        rows = np.asarray(rows, dtype=np.int64)
        if rows.size == 0:
            return sparse.csr_matrix((0, self.n_cols), dtype=np.float32)
        if np.any(np.diff(rows) < 0) or len(np.unique(rows)) != len(rows):
            raise ValueError("CSR selected rows must be sorted unique")
        start = int(rows[0]); stop = int(rows[-1]) + 1
        span = self._csr_row_block(start, stop)
        return span[(rows - start).astype(np.int64)]

    def _csc_contiguous_col_block(self, start: int, stop: int) -> sparse.csr_matrix:
        if self.kind != "csc":
            raise TypeError("not CSC")
        start = int(start); stop = int(stop)
        lo = int(self.indptr[start]); hi = int(self.indptr[stop])
        data = np.asarray(self.node["data"][lo:hi])
        indices = np.asarray(self.node["indices"][lo:hi], dtype=np.int32)
        ptr = np.asarray(self.indptr[start : stop + 1] - lo, dtype=np.int64)
        csc = sparse.csc_matrix((data, indices, ptr), shape=(self.n_rows, stop - start))
        return csc.tocsr()

    def _csc_selected_cols(self, cols: np.ndarray) -> sparse.csr_matrix:
        if self.kind != "csc":
            raise TypeError("not CSC")
        cols = np.asarray(cols, dtype=np.int64)
        if cols.size == 0:
            return sparse.csr_matrix((self.n_rows, 0), dtype=np.float32)
        if np.any(np.diff(cols) < 0) or len(np.unique(cols)) != len(cols):
            raise ValueError("CSC selected cols must be sorted unique")
        data_parts = []
        row_parts = []
        col_parts = []
        for j, col in enumerate(cols.tolist()):
            lo = int(self.indptr[col]); hi = int(self.indptr[col + 1])
            if hi <= lo:
                continue
            vals = np.asarray(self.node["data"][lo:hi])
            rows = np.asarray(self.node["indices"][lo:hi], dtype=np.int32)
            data_parts.append(vals)
            row_parts.append(rows)
            col_parts.append(np.full(len(rows), j, dtype=np.int32))
        if not data_parts:
            return sparse.csr_matrix((self.n_rows, len(cols)), dtype=np.float32)
        data = np.concatenate(data_parts)
        rows = np.concatenate(row_parts)
        cidx = np.concatenate(col_parts)
        return sparse.coo_matrix((data, (rows, cidx)), shape=(self.n_rows, len(cols))).tocsr()

    def row_sums(self, row_block: int = 4096, nnz_block: int = 5_000_000) -> np.ndarray:
        sums = np.zeros(self.n_rows, dtype=np.float64)
        if self.kind == "dense":
            max_elems = 12_000_000
            rb = min(int(row_block), max(1, max_elems // max(self.n_cols, 1)))
            for start in range(0, self.n_rows, rb):
                stop = min(self.n_rows, start + rb)
                block = np.asarray(self.node[start:stop, :], dtype=np.float64)
                sums[start:stop] = block.sum(axis=1)
            return sums
        if self.kind == "csr":
            rb = int(row_block)
            for start in range(0, self.n_rows, rb):
                stop = min(self.n_rows, start + rb)
                lo = int(self.indptr[start]); hi = int(self.indptr[stop])
                vals = np.asarray(self.node["data"][lo:hi], dtype=np.float64)
                local_ptr = self.indptr[start : stop + 1] - lo
                cs = np.empty(len(vals) + 1, dtype=np.float64)
                cs[0] = 0.0
                np.cumsum(vals, out=cs[1:])
                sums[start:stop] = cs[local_ptr[1:]] - cs[local_ptr[:-1]]
            return sums
        # CSC: stream stored nonzeros and bincount by row index.
        total = int(self.node["data"].shape[0])
        for lo in range(0, total, int(nnz_block)):
            hi = min(total, lo + int(nnz_block))
            rows = np.asarray(self.node["indices"][lo:hi], dtype=np.int64)
            vals = np.asarray(self.node["data"][lo:hi], dtype=np.float64)
            sums += np.bincount(rows, weights=vals, minlength=self.n_rows)
        return sums

    @staticmethod
    def _logged_block(block, rows: np.ndarray, library_sizes: np.ndarray, expression_state: str, target_sum: float):
        rows = np.asarray(rows, dtype=np.int64)
        if expression_state == "raw_counts":
            libs = library_sizes[rows]
            scale = np.divide(float(target_sum), libs, out=np.zeros_like(libs, dtype=np.float64), where=libs > 0)
            if sparse.issparse(block):
                linear = block.astype(np.float64).tocsr(copy=True)
                linear = sparse.diags(scale) @ linear
                logged = linear.copy().tocsr()
                logged.data = np.log1p(logged.data)
                return linear.tocsr(), logged
            linear = np.asarray(block, dtype=np.float64) * scale[:, None]
            logged = np.log1p(linear)
            return linear, logged
        if expression_state == "log1p_library_normalized":
            if sparse.issparse(block):
                logged = block.astype(np.float64).tocsr(copy=True)
                linear = logged.copy().tocsr()
                linear.data = np.expm1(linear.data)
                return linear, logged
            logged = np.asarray(block, dtype=np.float64)
            return np.expm1(logged), logged
        raise ValueError(expression_state)

    def control_hvg_variance(
        self,
        control_idx: np.ndarray,
        library_sizes: np.ndarray,
        expression_state: str,
        target_sum: float,
        n_hvg: int,
        max_cells: int = 20000,
        seed: int = SEED,
        row_batch: int = 256,
        csc_col_block: int = 256,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        control_idx = np.asarray(control_idx, dtype=np.int64)
        if len(control_idx) == 0:
            raise ValueError("No controls for HVG selection")
        rng = np.random.default_rng(int(seed))
        if max_cells and len(control_idx) > max_cells:
            selected = np.sort(rng.choice(control_idx, size=int(max_cells), replace=False))
        else:
            selected = np.sort(control_idx)
        n = len(selected)
        sums = np.zeros(self.n_cols, dtype=np.float64)
        sumsq = np.zeros(self.n_cols, dtype=np.float64)

        if self.kind == "dense":
            for start in range(0, n, int(row_batch)):
                rows = selected[start : start + int(row_batch)]
                raw = np.asarray(self.node[rows, :], dtype=np.float64)
                _, logged = self._logged_block(raw, rows, library_sizes, expression_state, target_sum)
                sums += np.asarray(logged.sum(axis=0)).ravel()
                sumsq += np.asarray((logged * logged).sum(axis=0)).ravel()

        elif self.kind == "csr":
            for start in range(0, n, int(row_batch)):
                rows = selected[start : start + int(row_batch)]
                raw = self._csr_selected_rows(rows)
                _, logged = self._logged_block(raw, rows, library_sizes, expression_state, target_sum)
                sums += np.asarray(logged.sum(axis=0)).ravel()
                sumsq += np.asarray(logged.power(2).sum(axis=0)).ravel()

        else:  # CSC is naturally gene-oriented.
            for c0 in range(0, self.n_cols, int(csc_col_block)):
                c1 = min(self.n_cols, c0 + int(csc_col_block))
                raw_all = self._csc_contiguous_col_block(c0, c1)
                raw = raw_all[selected]
                _, logged = self._logged_block(raw, selected, library_sizes, expression_state, target_sum)
                sums[c0:c1] = np.asarray(logged.sum(axis=0)).ravel()
                sumsq[c0:c1] = np.asarray(logged.power(2).sum(axis=0)).ravel()

        mean = sums / n
        mean_sq = sumsq / n
        var_pop = np.maximum(mean_sq - mean * mean, 0.0)
        variance = var_pop * n / (n - 1) if n > 1 else np.zeros_like(mean)
        keep = min(max(int(n_hvg), 1), self.n_cols)
        idx = np.sort(np.argsort(-variance, kind="stable")[:keep])
        return idx.astype(np.int64), variance[idx], int(n)

    @staticmethod
    def _membership(gids: np.ndarray, n_groups: int) -> sparse.csr_matrix | None:
        if n_groups <= 0:
            return None
        gids = np.asarray(gids, dtype=np.int64)
        rows = np.flatnonzero(gids >= 0)
        if len(rows) == 0:
            return sparse.csr_matrix((len(gids), n_groups), dtype=np.float32)
        return sparse.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, gids[rows])),
            shape=(len(gids), n_groups),
        )

    @staticmethod
    def _to_dense(A) -> np.ndarray:
        return A.toarray() if sparse.issparse(A) else np.asarray(A)

    def aggregate_hvg_sums(
        self,
        hvg_idx: np.ndarray,
        library_sizes: np.ndarray,
        expression_state: str,
        target_sum: float,
        group_systems: dict[str, tuple[np.ndarray, int]],
        row_block: int = 2048,
        csc_col_block: int = 128,
    ) -> dict[str, dict[str, np.ndarray]]:
        hvg_idx = np.asarray(hvg_idx, dtype=np.int64)
        n_feat = len(hvg_idx)
        result: dict[str, dict[str, np.ndarray]] = {}
        memberships = {}
        for name, (gids, n_groups) in group_systems.items():
            gids = np.asarray(gids, dtype=np.int64)
            counts = np.bincount(gids[gids >= 0], minlength=int(n_groups)).astype(np.int64)
            result[name] = {
                "linear_sum": np.zeros((int(n_groups), n_feat), dtype=np.float64),
                "log_sum": np.zeros((int(n_groups), n_feat), dtype=np.float64),
                "counts": counts,
            }
            memberships[name] = self._membership(gids, int(n_groups))

        if self.kind in {"dense", "csr"}:
            rb = int(row_block)
            for r0 in range(0, self.n_rows, rb):
                r1 = min(self.n_rows, r0 + rb)
                rows = np.arange(r0, r1, dtype=np.int64)
                if self.kind == "dense":
                    # Read a contiguous dense row block first. This avoids large
                    # h5py fancy-column selections and is efficient for the two
                    # ~8.5k-gene Replogle dense matrices.
                    raw_full = np.asarray(self.node[r0:r1, :], dtype=np.float64)
                    raw = raw_full[:, hvg_idx]
                else:
                    raw = self._csr_row_block(r0, r1)[:, hvg_idx]
                linear, logged = self._logged_block(raw, rows, library_sizes, expression_state, target_sum)
                for name, G in memberships.items():
                    if G is None or result[name]["linear_sum"].shape[0] == 0:
                        continue
                    Gb = G[r0:r1]
                    if Gb.nnz == 0:
                        continue
                    a = Gb.T @ linear
                    b = Gb.T @ logged
                    result[name]["linear_sum"] += self._to_dense(a)
                    result[name]["log_sum"] += self._to_dense(b)
        else:
            # CSC: only read selected HVG columns, in batches, for all rows.
            for p0 in range(0, n_feat, int(csc_col_block)):
                p1 = min(n_feat, p0 + int(csc_col_block))
                cols = hvg_idx[p0:p1]
                raw = self._csc_selected_cols(cols)
                rows = np.arange(self.n_rows, dtype=np.int64)
                linear, logged = self._logged_block(raw, rows, library_sizes, expression_state, target_sum)
                for name, G in memberships.items():
                    if G is None or result[name]["linear_sum"].shape[0] == 0:
                        continue
                    a = G.T @ linear
                    b = G.T @ logged
                    result[name]["linear_sum"][:, p0:p1] += self._to_dense(a)
                    result[name]["log_sum"][:, p0:p1] += self._to_dense(b)
        return result


# ======================================================================================
# Frozen metadata/control plan construction
# ======================================================================================


@dataclass
class DatasetPlan:
    dataset_id: str
    eligible_atoms: pd.DataFrame
    cond_gid: np.ndarray
    ctrl_db_gid: np.ndarray
    ctrl_donor_gid: np.ndarray
    pair_plans: list[list[dict[str, Any]]]
    selected_levels: list[str]
    control_mask: np.ndarray
    condition_counts: np.ndarray
    ctrl_db_keys: list[str]
    ctrl_donor_keys: list[str]
    summary: dict[str, Any]
    plan_table: pd.DataFrame


def build_cell_bio_keys(obs: pd.DataFrame, bio_map_ds: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    available_values = bio_map_ds["available_obs_bio_fields"].astype(str).unique().tolist()
    if len(available_values) != 1:
        raise RuntimeError(f"Inconsistent available_obs_bio_fields: {available_values}")
    available_fields = [x for x in available_values[0].split(";") if x]
    observed_cols = [OBS_BIO[x] for x in available_fields]
    for col in observed_cols:
        if col not in obs.columns:
            raise KeyError(f"Required biological obs column missing: {col}")
    lookup: dict[str, str] = {}
    for _, r in bio_map_ds.iterrows():
        observed_key = join_key(r[f"key_{f}"] for f in available_fields)
        full_key = join_key(r[f"authority_{f}"] for f in BIO_FIELDS)
        if observed_key in lookup and lookup[observed_key] != full_key:
            raise RuntimeError("Observed biological key maps to multiple full keys")
        lookup[observed_key] = full_key
    if observed_cols:
        normalized = [norm_series(obs[c]).to_numpy(dtype=object) for c in observed_cols]
        observed_keys = np.asarray([SEP.join(vals) for vals in zip(*normalized)], dtype=object)
    else:
        observed_keys = np.asarray([""] * len(obs), dtype=object)
    full = np.asarray([lookup.get(str(k), "") for k in observed_keys], dtype=object)
    return full, available_fields


def build_dataset_plan(
    dataset_id: str,
    obs: pd.DataFrame,
    atoms_ds: pd.DataFrame,
    bio_map_ds: pd.DataFrame,
    selection: pd.DataFrame,
    control_value: Any,
    condition_col: str,
    valid_mask: np.ndarray,
    min_ctrl_total: int,
    min_ctrl_stratum: int,
) -> DatasetPlan:
    n_cells = len(obs)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if len(valid_mask) != n_cells:
        raise ValueError("valid mask length mismatch")
    if condition_col not in obs.columns or "C_control_indicator" not in obs.columns:
        raise KeyError("condition/control metadata missing")
    condition = norm_series(obs[condition_col]).to_numpy(dtype=object)
    donor = norm_series(obs[DONOR_COL]) if DONOR_COL in obs.columns else pd.Series([MISSING] * n_cells, index=obs.index)
    batch = norm_series(obs[BATCH_COL]) if BATCH_COL in obs.columns else pd.Series([MISSING] * n_cells, index=obs.index)
    donor_arr = donor.to_numpy(dtype=object)
    batch_arr = batch.to_numpy(dtype=object)
    control_mask = control_mask_from_series(obs["C_control_indicator"], control_value) & valid_mask
    full_bio, available_fields = build_cell_bio_keys(obs, bio_map_ds)
    if np.any((full_bio == "") & control_mask):
        raise RuntimeError(f"{dataset_id}: controls without biological authority mapping")

    atoms_ds = atoms_ds.copy().reset_index(drop=True)
    atoms_ds["response_atom_id"] = atoms_ds["response_atom_id"].astype(str)
    atoms_ds["response_atom_id_provisional"] = atoms_ds["response_atom_id_provisional"].astype(str)
    atoms_ds["condition_id"] = atoms_ds["condition_id"].astype(str)
    eligible = atoms_ds.loc[atoms_ds["support_status_pre_nuisance"].astype(str).eq("BIO_CONTROL_ELIGIBLE")].copy().reset_index(drop=True)
    selection_map = selection.set_index("response_atom_id")["selected_historical_collapsed"].astype(str).to_dict()

    # Pre-group valid perturbation cells by condition + biological context.
    pert_df = pd.DataFrame({
        "row": np.arange(n_cells, dtype=np.int64),
        "condition": condition,
        "bio": full_bio,
        "donor": donor_arr,
        "batch": batch_arr,
        "is_control": control_mask_from_series(obs["C_control_indicator"], control_value),
        "valid": valid_mask,
    })
    pert_df = pert_df.loc[(~pert_df["is_control"]) & pert_df["valid"] & pert_df["bio"].ne("")]
    pert_groups = {
        (str(c), str(b)): g["row"].to_numpy(dtype=np.int64)
        for (c, b), g in pert_df.groupby(["condition", "bio"], sort=False, dropna=False)
    }

    ctrl_df = pd.DataFrame({
        "row": np.arange(n_cells, dtype=np.int64),
        "bio": full_bio,
        "donor": donor_arr,
        "batch": batch_arr,
        "control": control_mask,
    })
    ctrl_df = ctrl_df.loc[ctrl_df["control"] & ctrl_df["bio"].ne("")]
    ctrl_db_indices = {
        (str(bio), str(d), str(ba)): g["row"].to_numpy(dtype=np.int64)
        for (bio, d, ba), g in ctrl_df.groupby(["bio", "donor", "batch"], sort=False, dropna=False)
    }
    ctrl_donor_indices = {
        (str(bio), str(d)): g["row"].to_numpy(dtype=np.int64)
        for (bio, d), g in ctrl_df.groupby(["bio", "donor"], sort=False, dropna=False)
    }

    ctrl_db_map: dict[tuple[str, str, str], int] = {}
    ctrl_donor_map: dict[tuple[str, str], int] = {}
    ctrl_db_keys: list[str] = []
    ctrl_donor_keys: list[str] = []
    pair_plans: list[list[dict[str, Any]]] = []
    selected_levels: list[str] = []
    cond_gid = np.full(n_cells, -1, dtype=np.int32)
    plan_rows = []
    selected_db = 0
    selected_donor = 0

    for effect_row, atom in eligible.iterrows():
        rid = str(atom["response_atom_id"])
        provisional = str(atom["response_atom_id_provisional"])
        condition_id = str(atom["condition_id"])
        bio = full_bio_key_from_row(atom)
        pidx = np.sort(pert_groups.get((condition_id, bio), np.asarray([], dtype=np.int64)))
        if len(pidx) == 0:
            raise RuntimeError(f"{dataset_id} {rid}: no valid perturbed cells reconstructed")
        if len(pidx) < min_pert:
            raise RuntimeError(
                f"{dataset_id} {rid}: valid perturbed cells {len(pidx)} < frozen minimum {min_pert}"
            )
        frozen_level = selection_map.get(provisional, "")
        if frozen_level not in {"donor+batch", "donor"}:
            raise RuntimeError(f"{dataset_id} {rid}: invalid frozen level {frozen_level!r}")

        # Evaluate both levels after applying valid-library filtering; the frozen level may not change.
        level_results = {}
        for level in ["donor+batch", "donor"]:
            strata: dict[tuple[str, ...], np.ndarray] = {}
            if level == "donor+batch":
                vals = [(str(donor_arr[i]), str(batch_arr[i])) for i in pidx]
            else:
                vals = [(str(donor_arr[i]),) for i in pidx]
            tmp: dict[tuple[str, ...], list[int]] = {}
            for i, val in zip(pidx.tolist(), vals):
                tmp.setdefault(val, []).append(int(i))
            for val, rows in tmp.items():
                strata[val] = np.asarray(rows, dtype=np.int64)
            ok = True
            total_controls = 0
            details = []
            for val, rows in strata.items():
                if level == "donor+batch":
                    cidx = ctrl_db_indices.get((bio, val[0], val[1]), np.asarray([], dtype=np.int64))
                else:
                    cidx = ctrl_donor_indices.get((bio, val[0]), np.asarray([], dtype=np.int64))
                n_ctrl = len(cidx)
                if n_ctrl < min_ctrl_stratum:
                    ok = False
                total_controls += n_ctrl
                details.append((val, rows, cidx))
            feasible = bool(ok and len(details) and total_controls >= min_ctrl_total)
            level_results[level] = (feasible, details, total_controls)

        expected_level = "donor+batch" if level_results["donor+batch"][0] else "donor" if level_results["donor"][0] else "UNSUPPORTED"
        if expected_level != frozen_level:
            raise RuntimeError(
                f"{dataset_id} {rid}: valid-row reconstruction changes frozen level: "
                f"frozen={frozen_level}, reconstructed={expected_level}"
            )

        details = level_results[frozen_level][1]
        pairs = []
        weight_sum = 0
        for val, crows, zrows in details:
            weight = int(len(crows))
            weight_sum += weight
            if frozen_level == "donor+batch":
                key = (bio, val[0], val[1])
                if key not in ctrl_db_map:
                    ctrl_db_map[key] = len(ctrl_db_map)
                    ctrl_db_keys.append(SEP.join(key))
                gid = ctrl_db_map[key]
                system = "ctrl_db"
            else:
                key = (bio, val[0])
                if key not in ctrl_donor_map:
                    ctrl_donor_map[key] = len(ctrl_donor_map)
                    ctrl_donor_keys.append(SEP.join(key))
                gid = ctrl_donor_map[key]
                system = "ctrl_donor"
            pairs.append({
                "control_system": system,
                "control_gid": int(gid),
                "weight": weight,
                "n_controls": int(len(zrows)),
                "stratum": SEP.join(val),
            })
        if weight_sum != len(pidx):
            raise RuntimeError(f"{dataset_id} {rid}: pair weights != condition cell count")
        cond_gid[pidx] = int(effect_row)
        pair_plans.append(pairs)
        selected_levels.append(frozen_level)
        selected_db += int(frozen_level == "donor+batch")
        selected_donor += int(frozen_level == "donor")
        plan_rows.append({
            "dataset_id": dataset_id,
            "effect_row": int(effect_row),
            "response_atom_id": rid,
            "response_atom_id_provisional": provisional,
            "condition_id": condition_id,
            "selected_level": frozen_level,
            "n_perturbed_cells": int(len(pidx)),
            "n_strata": int(len(pairs)),
            "n_controls_union": int(sum(p["n_controls"] for p in pairs)),
            "min_controls_per_stratum": int(min(p["n_controls"] for p in pairs)),
            "pair_plan_json": json.dumps(pairs, ensure_ascii=False, separators=(",", ":")),
        })

    ctrl_db_gid = np.full(n_cells, -1, dtype=np.int32)
    for key, gid in ctrl_db_map.items():
        ctrl_db_gid[ctrl_db_indices[key]] = int(gid)
    ctrl_donor_gid = np.full(n_cells, -1, dtype=np.int32)
    for key, gid in ctrl_donor_map.items():
        ctrl_donor_gid[ctrl_donor_indices[key]] = int(gid)
    condition_counts = np.bincount(cond_gid[cond_gid >= 0], minlength=len(eligible)).astype(np.int64)
    if np.any(condition_counts <= 0):
        raise RuntimeError(f"{dataset_id}: eligible atom with zero condition cells")

    summary = {
        "dataset_id": dataset_id,
        "response_atom_n": int(len(atoms_ds)),
        "eligible_atom_n": int(len(eligible)),
        "selected_donor_batch_n": int(selected_db),
        "selected_donor_n": int(selected_donor),
        "control_cell_n": int(control_mask.sum()),
        "valid_cell_n": int(valid_mask.sum()),
        "available_obs_bio_fields": ";".join(available_fields),
        "condition_group_n": int(len(eligible)),
        "ctrl_db_group_n": int(len(ctrl_db_map)),
        "ctrl_donor_group_n": int(len(ctrl_donor_map)),
    }
    return DatasetPlan(
        dataset_id=dataset_id,
        eligible_atoms=eligible,
        cond_gid=cond_gid,
        ctrl_db_gid=ctrl_db_gid,
        ctrl_donor_gid=ctrl_donor_gid,
        pair_plans=pair_plans,
        selected_levels=selected_levels,
        control_mask=control_mask,
        condition_counts=condition_counts,
        ctrl_db_keys=ctrl_db_keys,
        ctrl_donor_keys=ctrl_donor_keys,
        summary=summary,
        plan_table=pd.DataFrame(plan_rows),
    )


# ======================================================================================
# Load frozen authorities
# ======================================================================================


for p in [PATH_AUTH, ATOM_AUTH, BIO_KEY_MAP, SELECTION_TSV, NUISANCE_POLICY, DATASET_CATALOG, EFFECT_CONFIG]:
    if not p.is_file():
        raise FileNotFoundError(p)

path_auth = pd.read_csv(PATH_AUTH, sep="\t", low_memory=False)
atoms = pd.read_csv(ATOM_AUTH, sep="\t", low_memory=False)
bio_map = pd.read_csv(BIO_KEY_MAP, sep="\t", low_memory=False)
selection = pd.read_csv(SELECTION_TSV, sep="\t", low_memory=False)
catalog = pd.read_csv(DATASET_CATALOG, sep="\t", low_memory=False)
policy = json.loads(NUISANCE_POLICY.read_text(encoding="utf-8"))
with EFFECT_CONFIG.open() as f:
    cfg = yaml.safe_load(f)

if policy.get("status") != "FROZEN":
    raise RuntimeError("Nuisance policy is not frozen")
if [x["name"] for x in policy["primary_nuisance_hierarchy"]] != ["donor+batch", "donor"]:
    raise RuntimeError("Unexpected frozen nuisance hierarchy")

min_pert = int(policy["support_thresholds"]["min_perturbed_cells_per_response_atom"])
min_ctrl = int(policy["support_thresholds"]["min_biological_context_control_cells"])
min_ctrl_stratum = int(policy["support_thresholds"]["min_controls_per_nuisance_stratum"])
target_sum = float(cfg["expression"].get("target_sum", 10000.0))
pseudocount = float(cfg["effect_cache"].get("pseudocount", 1.0))
n_hvg_default = int(cfg["effect_cache"].get("n_hvg", 3000))
max_hvg_cells = int(cfg.get("memory", {}).get("max_hvg_cells", 20000))

path_auth["dataset_id"] = path_auth["dataset_id"].astype(str)
atoms["dataset_id"] = atoms["dataset_id"].astype(str)
bio_map["dataset_id"] = bio_map["dataset_id"].astype(str)
selection["response_atom_id"] = selection["response_atom_id"].astype(str)
catalog["dataset_id"] = catalog["dataset_id"].astype(str)

all_datasets = path_auth["dataset_id"].astype(str).tolist()
requested = ARGS.datasets if ARGS.datasets else all_datasets
unknown = sorted(set(requested) - set(all_datasets))
if unknown:
    raise ValueError(f"Unknown datasets: {unknown}")

path_map = path_auth.set_index("dataset_id")["h5ad_path"].astype(str).to_dict()
catalog_map = {str(r["dataset_id"]): r for _, r in catalog.iterrows()}

try:
    import anndata as ad
except Exception as exc:
    raise RuntimeError("anndata is required for obs/var metadata only") from exc


# ======================================================================================
# Preflight
# ======================================================================================


def preflight_dataset(dataset_id: str) -> dict[str, Any]:
    h5ad_path = Path(path_map[dataset_id])
    cat = catalog_map[dataset_id]
    auth = expression_authority(dataset_id, cat, cfg)
    condition_col = str(cat.get("condition_id_column", cfg["schema"].get("condition_id_column", "C_condition_id")))
    control_col = str(cat.get("control_column", cfg["schema"].get("control_column", "C_control_indicator")))
    control_value = cat.get("control_value", cfg["schema"].get("control_value", True))
    if control_col != "C_control_indicator":
        raise RuntimeError(f"{dataset_id}: corrected R2 expects C_control_indicator, got {control_col}")

    adata = ad.read_h5ad(h5ad_path, backed="r")
    obs = adata.obs.copy()
    genes = np.asarray(adata.var_names.astype(str))
    if len(set(genes.tolist())) != len(genes):
        raise RuntimeError(f"{dataset_id}: var_names not unique")

    with H5ExpressionSource(h5ad_path, auth["actual_source"]) as reader:
        if reader.shape != (adata.n_obs, adata.n_vars):
            raise RuntimeError(f"{dataset_id}: source shape {reader.shape} != AnnData {(adata.n_obs, adata.n_vars)}")
        vals = reader.stored_value_sample(n=200000, seed=SEED)
        finite = vals[np.isfinite(vals)]
        finite_fraction = float(len(finite) / len(vals)) if len(vals) else 0.0
        nonnegative = bool(len(finite) and np.min(finite) >= 0)
        integer_fraction = float(np.mean(np.isclose(finite, np.round(finite), atol=1e-6))) if len(finite) else np.nan
        small = reader.read_small_block(64, 512)
        small_finite = bool(np.isfinite(small).all())
        if auth["expression_state"] == "raw_counts":
            expr_ok = bool(nonnegative and finite_fraction == 1.0 and integer_fraction >= 0.99 and small_finite)
        else:
            expr_ok = bool(nonnegative and finite_fraction == 1.0 and small_finite)
        atoms_ds = atoms.loc[atoms["dataset_id"].eq(dataset_id)].copy()
        bio_ds = bio_map.loc[bio_map["dataset_id"].eq(dataset_id)].copy()
        # Metadata-only reproduction of the frozen hierarchy; expression-positive filtering is formal-build QC.
        plan = build_dataset_plan(
            dataset_id, obs, atoms_ds, bio_ds, selection, control_value, condition_col,
            np.ones(len(obs), dtype=bool), min_ctrl, min_ctrl_stratum,
        )
        expected_db = int((selection.loc[
            selection["response_atom_id"].isin(atoms_ds.loc[atoms_ds["support_status_pre_nuisance"].astype(str).eq("BIO_CONTROL_ELIGIBLE"), "response_atom_id_provisional"].astype(str)),
            "selected_historical_collapsed"
        ].astype(str) == "donor+batch").sum())
        expected_donor = int((selection.loc[
            selection["response_atom_id"].isin(atoms_ds.loc[atoms_ds["support_status_pre_nuisance"].astype(str).eq("BIO_CONTROL_ELIGIBLE"), "response_atom_id_provisional"].astype(str)),
            "selected_historical_collapsed"
        ].astype(str) == "donor").sum())
        plan_ok = plan.summary["selected_donor_batch_n"] == expected_db and plan.summary["selected_donor_n"] == expected_donor
        out = {
            "dataset_id": dataset_id,
            "h5ad_path": str(h5ad_path),
            "authority_source": auth["authority_source"],
            "source_category": auth["source_category"],
            "actual_source": auth["actual_source"],
            "expression_state": auth["expression_state"],
            "historical_catalog_source": auth["historical_catalog_source"],
            "historical_catalog_state": auth["historical_catalog_state"],
            **reader.metadata(),
            "stored_sample_n": int(len(vals)),
            "stored_finite_fraction": finite_fraction,
            "stored_nonnegative": nonnegative,
            "stored_integer_like_fraction": integer_fraction,
            "small_block_shape": json.dumps(list(small.shape)),
            "small_block_finite": small_finite,
            "response_atom_n": int(plan.summary["response_atom_n"]),
            "eligible_atom_n": int(plan.summary["eligible_atom_n"]),
            "selected_donor_batch_n": int(plan.summary["selected_donor_batch_n"]),
            "selected_donor_n": int(plan.summary["selected_donor_n"]),
            "expected_donor_batch_n": expected_db,
            "expected_donor_n": expected_donor,
            "expression_source_qc": "PASS" if expr_ok else "FAIL",
            "control_plan_qc": "PASS" if plan_ok else "FAIL",
            "status": "PASS" if expr_ok and plan_ok else "FAIL",
        }
    try:
        adata.file.close()
    except Exception:
        pass
    return out


def run_preflight() -> None:
    PREFLIGHT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in requested:
        log(f"[preflight] {ds}")
        rows.append(preflight_dataset(ds))
    df = pd.DataFrame(rows)
    df.to_csv(PREFLIGHT_TSV, sep="\t", index=False)
    status = "PASS" if len(df) and df["status"].eq("PASS").all() else "FAIL"
    lines = [
        "PERTURBCONTEXTALIGN R2 MEASURED-RESPONSE EFFECT PREFLIGHT v2",
        "=" * 120,
        "",
        "READER CONTRACT",
        "-" * 120,
        "AnnData_expression_indexing=FALSE",
        "AnnData_usage=obs_and_var_names_only",
        "expression_reader=direct_h5py_by_storage_encoding",
        "supported_storage=dense,CSR,CSC",
        "formal_HVG_computation=FALSE",
        "formal_effect_computation=FALSE",
        "",
        "EXPRESSION AUTHORITIES",
        "-" * 120,
        "combo_sciplex=layers/counts; source_category=raw_counts_in_layer; state=raw_counts",
        "kaggle_cross_patient=X; state=log1p_library_normalized; renormalize=FALSE; relog1p=FALSE",
        "",
        "DATASET AUDIT",
        "-" * 120,
        df.to_string(index=False),
        "",
        "STATUS",
        "-" * 120,
        f"R2_MEASURED_RESPONSE_EFFECT_PREFLIGHT_V2={status}",
    ]
    if status == "PASS":
        lines.append("NEXT=R2_MEASURED_RESPONSE_EFFECT_BUILD_V1")
    else:
        lines.append("NEXT=REVIEW_PREFLIGHT_V2_FAILURE")
    PREFLIGHT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json({
        "version": "R2_MEASURED_RESPONSE_EFFECT_PREFLIGHT_v2",
        "script_version": SCRIPT_VERSION,
        "status": status,
        "datasets": requested,
        "reader": "direct_h5py_by_storage_encoding",
        "inputs": {
            "path_authority_sha256": sha256_file(PATH_AUTH),
            "response_atom_authority_sha256": sha256_file(ATOM_AUTH),
            "bio_key_map_sha256": sha256_file(BIO_KEY_MAP),
            "selection_sha256": sha256_file(SELECTION_TSV),
            "nuisance_policy_sha256": sha256_file(NUISANCE_POLICY),
        },
    }, PREFLIGHT_MANIFEST)
    print(PREFLIGHT_REPORT.read_text(encoding="utf-8"))
    if status != "PASS":
        raise SystemExit(2)


# ======================================================================================
# Formal build
# ======================================================================================


def build_dataset(dataset_id: str) -> dict[str, Any]:
    t0 = time.time()
    h5ad_path = Path(path_map[dataset_id])
    cat = catalog_map[dataset_id]
    auth = expression_authority(dataset_id, cat, cfg)
    condition_col = str(cat.get("condition_id_column", cfg["schema"].get("condition_id_column", "C_condition_id")))
    control_value = cat.get("control_value", cfg["schema"].get("control_value", True))
    n_hvg = int(cat.get("n_hvg", n_hvg_default)) if pd.notna(cat.get("n_hvg", np.nan)) else n_hvg_default
    ds_out = EFFECT_ROOT / dataset_id
    ds_out.mkdir(parents=True, exist_ok=True)
    done_path = ds_out / "DONE.json"

    input_hashes = {
        "atom_authority": sha256_file(ATOM_AUTH),
        "bio_key_map": sha256_file(BIO_KEY_MAP),
        "selection": sha256_file(SELECTION_TSV),
        "policy": sha256_file(NUISANCE_POLICY),
        "effect_config": sha256_file(EFFECT_CONFIG),
    }
    if done_path.is_file() and not ARGS.force:
        old = json.loads(done_path.read_text(encoding="utf-8"))
        if old.get("status") == "PASS" and old.get("input_hashes") == input_hashes:
            log(f"[resume] {dataset_id}: DONE")
            return old["audit_row"]

    adata = ad.read_h5ad(h5ad_path, backed="r")
    obs = adata.obs.copy()
    genes = np.asarray(adata.var_names.astype(str))
    if len(set(genes.tolist())) != len(genes):
        raise RuntimeError(f"{dataset_id}: var_names not unique")
    atoms_ds = atoms.loc[atoms["dataset_id"].eq(dataset_id)].copy()
    bio_ds = bio_map.loc[bio_map["dataset_id"].eq(dataset_id)].copy()

    with H5ExpressionSource(h5ad_path, auth["actual_source"]) as reader:
        if reader.shape != (adata.n_obs, adata.n_vars):
            raise RuntimeError(f"{dataset_id}: expression shape mismatch")
        log(f"[{dataset_id}] storage={reader.kind} shape={reader.shape} source={auth['actual_source']} state={auth['expression_state']}")

        # 1. Exact per-cell library/observed row sums from direct HDF5 storage.
        log(f"[{dataset_id}] step=row_sums")
        library_sizes = reader.row_sums(row_block=max(ARGS.row_block, 2048))
        valid_library = np.isfinite(library_sizes) & (library_sizes > 0)
        positive_fraction = float(valid_library.mean())
        if positive_fraction < 0.99:
            raise RuntimeError(f"{dataset_id}: positive row fraction {positive_fraction:.6f} < 0.99")

        # 2. Reconstruct the frozen response/control plan after valid-row filtering.
        log(f"[{dataset_id}] step=control_plan")
        plan = build_dataset_plan(
            dataset_id, obs, atoms_ds, bio_ds, selection, control_value, condition_col,
            valid_library, min_ctrl, min_ctrl_stratum,
        )
        expected_eligible = int((atoms_ds["support_status_pre_nuisance"].astype(str) == "BIO_CONTROL_ELIGIBLE").sum())
        if plan.summary["eligible_atom_n"] != expected_eligible:
            raise RuntimeError(f"{dataset_id}: eligible atom count changed")

        # 3. Validate the selected source directly.
        vals = reader.stored_value_sample(200000, SEED)
        finite = vals[np.isfinite(vals)]
        integer_fraction = float(np.mean(np.isclose(finite, np.round(finite), atol=1e-6))) if len(finite) else np.nan
        if auth["expression_state"] == "raw_counts" and integer_fraction < 0.99:
            raise RuntimeError(f"{dataset_id}: raw-count integer-like validation failed")
        if not len(finite) or np.min(finite) < 0:
            raise RuntimeError(f"{dataset_id}: expression source is empty/negative")

        # 4. Dataset-level control-only HVGs, exactly on normalized/log1p control expression.
        log(f"[{dataset_id}] step=control_only_hvg")
        hvg_idx, hvg_var, n_hvg_controls = reader.control_hvg_variance(
            control_idx=np.flatnonzero(plan.control_mask),
            library_sizes=library_sizes,
            expression_state=auth["expression_state"],
            target_sum=target_sum,
            n_hvg=n_hvg,
            max_cells=max_hvg_cells,
            seed=SEED,
            row_batch=256,
            csc_col_block=max(64, ARGS.csc_col_block),
        )
        genes_hvg = genes[hvg_idx]
        pd.DataFrame({
            "hvg_index": hvg_idx,
            "gene_name": genes_hvg,
            "control_log1p_variance": hvg_var,
        }).to_csv(ds_out / "hvg_genes.tsv", sep="\t", index=False)

        # 5. One streaming aggregation over 3000 HVGs into response/control sufficient statistics.
        log(f"[{dataset_id}] step=aggregate_hvg_sufficient_statistics")
        group_systems = {
            "condition": (plan.cond_gid, len(plan.eligible_atoms)),
            "ctrl_db": (plan.ctrl_db_gid, len(plan.ctrl_db_keys)),
            "ctrl_donor": (plan.ctrl_donor_gid, len(plan.ctrl_donor_keys)),
        }
        agg = reader.aggregate_hvg_sums(
            hvg_idx=hvg_idx,
            library_sizes=library_sizes,
            expression_state=auth["expression_state"],
            target_sum=target_sum,
            group_systems=group_systems,
            row_block=max(256, ARGS.row_block),
            csc_col_block=max(32, ARGS.csc_col_block),
        )

        cond_counts = agg["condition"]["counts"]
        if not np.array_equal(cond_counts, plan.condition_counts):
            raise RuntimeError(f"{dataset_id}: condition aggregation counts mismatch")
        cond_linear_mean = agg["condition"]["linear_sum"] / cond_counts[:, None]
        cond_log_mean = agg["condition"]["log_sum"] / cond_counts[:, None]

        ctrl_db_counts = agg["ctrl_db"]["counts"]
        ctrl_donor_counts = agg["ctrl_donor"]["counts"]
        ctrl_db_linear_mean = np.divide(
            agg["ctrl_db"]["linear_sum"], ctrl_db_counts[:, None],
            out=np.zeros_like(agg["ctrl_db"]["linear_sum"]), where=ctrl_db_counts[:, None] > 0,
        )
        ctrl_db_log_mean = np.divide(
            agg["ctrl_db"]["log_sum"], ctrl_db_counts[:, None],
            out=np.zeros_like(agg["ctrl_db"]["log_sum"]), where=ctrl_db_counts[:, None] > 0,
        )
        ctrl_donor_linear_mean = np.divide(
            agg["ctrl_donor"]["linear_sum"], ctrl_donor_counts[:, None],
            out=np.zeros_like(agg["ctrl_donor"]["linear_sum"]), where=ctrl_donor_counts[:, None] > 0,
        )
        ctrl_donor_log_mean = np.divide(
            agg["ctrl_donor"]["log_sum"], ctrl_donor_counts[:, None],
            out=np.zeros_like(agg["ctrl_donor"]["log_sum"]), where=ctrl_donor_counts[:, None] > 0,
        )

        # 6. Exact historical weighting of matched-control stratum means by perturbed-cell counts.
        n_atoms = len(plan.eligible_atoms)
        logfc = np.zeros((n_atoms, len(hvg_idx)), dtype=np.float32)
        mean_delta = np.zeros_like(logfc)
        qc_rows = []
        for i in range(n_atoms):
            pairs = plan.pair_plans[i]
            total_weight = float(sum(p["weight"] for p in pairs))
            if total_weight != float(cond_counts[i]):
                raise RuntimeError(f"{dataset_id} row {i}: weight sum != condition count")
            zlin = np.zeros(len(hvg_idx), dtype=np.float64)
            zlog = np.zeros(len(hvg_idx), dtype=np.float64)
            control_cells_sum = 0
            for p in pairs:
                w = float(p["weight"] / total_weight)
                gid = int(p["control_gid"])
                if p["control_system"] == "ctrl_db":
                    zlin += w * ctrl_db_linear_mean[gid]
                    zlog += w * ctrl_db_log_mean[gid]
                    control_cells_sum += int(ctrl_db_counts[gid])
                else:
                    zlin += w * ctrl_donor_linear_mean[gid]
                    zlog += w * ctrl_donor_log_mean[gid]
                    control_cells_sum += int(ctrl_donor_counts[gid])
            c_lin = cond_linear_mean[i]
            c_log = cond_log_mean[i]
            lf = np.log2((c_lin + pseudocount) / (zlin + pseudocount))
            md = c_log - zlog
            if not np.isfinite(lf).all() or not np.isfinite(md).all():
                raise RuntimeError(f"{dataset_id} row {i}: nonfinite effect")
            logfc[i] = lf.astype(np.float32)
            mean_delta[i] = md.astype(np.float32)
            constant = bool(np.ptp(lf) <= 1e-12)
            qc_rows.append({
                "dataset_id": dataset_id,
                "effect_row": int(i),
                "response_atom_id": str(plan.eligible_atoms.loc[i, "response_atom_id"]),
                "condition_id": str(plan.eligible_atoms.loc[i, "condition_id"]),
                "selected_level": plan.selected_levels[i],
                "n_condition_cells": int(cond_counts[i]),
                "n_matched_strata": int(len(pairs)),
                "sum_control_cells_across_strata": int(control_cells_sum),
                "effect_norm_logfc": float(np.linalg.norm(lf)),
                "effect_logfc_sd": float(np.std(lf)),
                "constant_logfc": constant,
                "primary_similarity_eligible": not constant,
            })

        response_ids = plan.eligible_atoms["response_atom_id"].astype(str).tolist()
        metadata = {
            "dataset_id": dataset_id,
            "row_identity": "response_atom_id",
            "expression_authority": auth,
            "expression_state": auth["expression_state"],
            "normalization": "library_size_from_raw_counts" if auth["expression_state"] == "raw_counts" else "released_library_normalized_log1p",
            "target_sum": target_sum if auth["expression_state"] == "raw_counts" else None,
            "logfc_linear_scale": "normalized_counts" if auth["expression_state"] == "raw_counts" else "expm1_released_normalized_count",
            "hvg_source": "dataset_level_controls_only",
            "n_hvg": int(len(hvg_idx)),
            "pseudocount": pseudocount,
            "control_policy": "same_bio_context -> donor+batch -> donor -> unsupported",
        }
        save_dense_bundle(ds_out / "effect_logfc_hvg.npz", logfc, response_ids, genes_hvg.tolist(), {**metadata, "effect_representation": "logfc_hvg"})
        save_dense_bundle(ds_out / "effect_mean_delta_hvg.npz", mean_delta, response_ids, genes_hvg.tolist(), {**metadata, "effect_representation": "mean_delta_hvg"})
        plan.plan_table.to_csv(ds_out / "control_plan.tsv", sep="\t", index=False)
        pd.DataFrame(qc_rows).to_csv(ds_out / "effect_qc.tsv", sep="\t", index=False)
        plan.eligible_atoms.assign(effect_row=np.arange(n_atoms, dtype=np.int64)).to_csv(ds_out / "response_atom_index.tsv", sep="\t", index=False)
        write_json({
            "dataset_id": dataset_id,
            "h5ad_path": str(h5ad_path),
            "expression_authority": auth,
            "storage": reader.metadata(),
            "positive_row_fraction": positive_fraction,
            "median_observed_row_sum": float(np.median(library_sizes)),
            "n_hvg_controls": int(n_hvg_controls),
            "n_hvg": int(len(hvg_idx)),
            "integer_like_fraction_sample": integer_fraction,
        }, ds_out / "expression_qc.json")

        constant_n = int(sum(r["constant_logfc"] for r in qc_rows))
        audit_row = {
            "dataset_id": dataset_id,
            "status": "PASS",
            "storage_kind": reader.kind,
            "actual_source": auth["actual_source"],
            "expression_state": auth["expression_state"],
            "n_cells": int(reader.n_rows),
            "n_genes": int(reader.n_cols),
            "positive_row_fraction": positive_fraction,
            "eligible_response_atom_n": int(n_atoms),
            "selected_donor_batch_n": int(plan.summary["selected_donor_batch_n"]),
            "selected_donor_n": int(plan.summary["selected_donor_n"]),
            "n_hvg": int(len(hvg_idx)),
            "n_hvg_controls": int(n_hvg_controls),
            "constant_logfc_atom_n": constant_n,
            "primary_similarity_eligible_atom_n": int(n_atoms - constant_n),
            "elapsed_seconds": float(time.time() - t0),
        }
        done = {
            "status": "PASS",
            "version": "R2_MEASURED_RESPONSE_EFFECT_v1",
            "script_version": SCRIPT_VERSION,
            "dataset_id": dataset_id,
            "input_hashes": input_hashes,
            "audit_row": audit_row,
        }
        write_json(done, done_path)
        try:
            adata.file.close()
        except Exception:
            pass
        return audit_row


def run_build() -> None:
    if not PREFLIGHT_MANIFEST.is_file():
        raise RuntimeError(f"Preflight manifest missing: {PREFLIGHT_MANIFEST}. Run --mode preflight first.")
    pre = json.loads(PREFLIGHT_MANIFEST.read_text(encoding="utf-8"))
    if pre.get("status") != "PASS":
        raise RuntimeError("Preflight v2 is not PASS")
    EFFECT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in requested:
        log(f"[build] {ds}")
        rows.append(build_dataset(ds))
    # Include previously completed datasets for a consolidated status table.
    completed_rows = []
    for ds in all_datasets:
        p = EFFECT_ROOT / ds / "DONE.json"
        if p.is_file():
            d = json.loads(p.read_text(encoding="utf-8"))
            if d.get("status") == "PASS":
                completed_rows.append(d["audit_row"])
    audit = pd.DataFrame(completed_rows).sort_values("dataset_id") if completed_rows else pd.DataFrame(rows)
    audit.to_csv(GLOBAL_AUDIT_TSV, sep="\t", index=False)
    all_complete = set(audit["dataset_id"].astype(str)) == set(all_datasets) and audit["status"].eq("PASS").all()
    status = "PASS" if all_complete else "PARTIAL_PASS"
    lines = [
        "PERTURBCONTEXTALIGN R2 MEASURED-RESPONSE EFFECT AUDIT v1",
        "=" * 120,
        "",
        "METHOD",
        "-" * 120,
        "expression_reader=direct_h5py_by_dense_CSR_CSC_storage",
        "AnnData_expression_indexing=FALSE",
        "HVG=dataset-level control-only top-3000 variance on normalized/log1p expression",
        "primary_effect=log2((weighted_perturbed_mean+1)/(weighted_matched_control_mean+1))",
        "matched_control=same biological context; donor+batch -> donor; no broader fallback",
        "condition_mean=all eligible condition cells (equivalent to stratum means weighted by condition-cell counts)",
        "control_mean=stratum means weighted by matched perturbed-stratum cell counts",
        "",
        "DATASET AUDIT",
        "-" * 120,
        audit.to_string(index=False),
        "",
        "STATUS",
        "-" * 120,
        f"R2_MEASURED_RESPONSE_EFFECT={status}",
        f"completed_dataset_n={len(audit)}",
        f"expected_dataset_n={len(all_datasets)}",
    ]
    if all_complete:
        lines.append("NEXT=R2_RESPONSE_SIMILARITY_v1")
    else:
        lines.append("NEXT=COMPLETE_REMAINING_DATASETS")
    GLOBAL_AUDIT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json({
        "version": "R2_MEASURED_RESPONSE_EFFECT_v1",
        "script_version": SCRIPT_VERSION,
        "status": status,
        "completed_datasets": audit["dataset_id"].astype(str).tolist(),
        "expected_datasets": all_datasets,
        "input_hashes": {
            "atom_authority": sha256_file(ATOM_AUTH),
            "bio_key_map": sha256_file(BIO_KEY_MAP),
            "selection": sha256_file(SELECTION_TSV),
            "policy": sha256_file(NUISANCE_POLICY),
            "effect_config": sha256_file(EFFECT_CONFIG),
        },
    }, GLOBAL_MANIFEST)
    print(GLOBAL_AUDIT_TXT.read_text(encoding="utf-8"))


if __name__ == "__main__":
    if ARGS.mode == "preflight":
        run_preflight()
    else:
        run_build()
