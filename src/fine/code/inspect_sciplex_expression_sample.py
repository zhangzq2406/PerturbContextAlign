#!/usr/bin/env python3
"""Bounded, metadata-selected sci-Plex expression QC; never estimates effects.

Default: two distinct wells per plate and role, one cell per selected well,
up to 192 cells total. Only selected CSR row slices are read from X. All source
feature columns, including repeated legacy IDs and mouse features, are retained.
Existing output directories are refused. Inputs are opened read-only.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import platform
import resource
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = Path("${PCA_DATA_ROOT}/2_DrugOrSmallmoleculePerturbationDatasets/1_SrivatsanTrapnell2020_sciPlex3/0_raw/SrivatsanTrapnell2020_sciplex3.h5ad")
MEMBERSHIP = ROOT / "metadata/sciplex_preflight_v1"
OUTPUT = ROOT / "qa/expression_sample_v1"
SEED = "sciplex-expression-sample-v1-20260914"
LIMITATION = ("Deterministic metadata-stratified sample only; this is not full-matrix "
              "QC approval. No cell/feature filtering, normalization, effects, HVG "
              "fitting, model fitting, or training was performed.")


def file_stat(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns,
            "device": stat.st_dev, "inode": stat.st_ino,
            "atime_ns": stat.st_atime_ns}


def content_stat_equal(left, right):
    # Access time can change from a read; it is not a content-mutation signal.
    keys = ["path", "size_bytes", "mtime_ns", "ctime_ns", "device", "inode"]
    return all(left[key] == right[key] for key in keys)


def stable_hash(*values):
    text = json.dumps([SEED, *values], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json_exclusive(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def write_table_exclusive(path, rows):
    if not rows:
        raise ValueError(f"Refusing empty table: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def read_h5_column(group, key, selected=None):
    node = group[key]
    if isinstance(node, h5py.Group):
        categories = node["categories"]
        vals = categories.asstr()[:] if categories.dtype.kind in "OSU" else categories[:]
        codes = node["codes"][:] if selected is None else node["codes"][selected]
        if np.any(codes < 0) or np.any(codes >= len(vals)):
            raise ValueError(f"Missing/invalid categorical code in selected {key}")
        return vals[codes]
    access = node.asstr() if node.dtype.kind in "OSU" else node
    return access[:] if selected is None else access[selected]


def select_cells(membership, max_rows):
    """Stream metadata; keep one hash-ranked cell per well, then rank wells."""
    required = {"source_row", "source_cell_id", "plate", "well", "cell_line", "time", "replicate"}
    best_per_well = {}
    scanned = {}
    for role in ("treated", "control"):
        path = membership / f"{role}_cell_membership.tsv.gz"
        count = 0
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if not required <= set(reader.fieldnames or []):
                raise ValueError(f"Missing membership columns: {path}")
            for row in reader:
                count += 1
                if not row["plate"] or not row["well"]:
                    raise ValueError("Admitted membership lacks plate/well")
                row["source_row"] = int(row["source_row"])
                key = (role, row["plate"], row["well"])
                cell_rank = stable_hash("cell", role, row["source_cell_id"], row["source_row"])
                previous = best_per_well.get(key)
                if previous is None or cell_rank < previous[0]:
                    best_per_well[key] = (cell_rank, row)
        scanned[role] = count
    grouped = {}
    for (role, plate, well), (cell_rank, row) in best_per_well.items():
        selected = {"role": role, **row, "cell_selection_sha256": cell_rank,
                    "well_selection_sha256": stable_hash("well", role, plate, well)}
        grouped.setdefault((plate, role), []).append(selected)
    plate_sets = {role: {plate for plate, r in grouped if r == role} for role in ("treated", "control")}
    if not plate_sets["treated"] or plate_sets["treated"] != plate_sets["control"]:
        raise ValueError("Treated/control plate coverage differs")
    if max_rows < len(grouped):
        raise ValueError(f"max_rows={max_rows} cannot cover all {len(grouped)} plate-role groups")
    for candidates in grouped.values():
        candidates.sort(key=lambda row: (row["well_selection_sha256"], row["source_row"]))
    group_order = sorted(grouped, key=lambda pair: (int(pair[0].removeprefix("plate")), pair[1]))
    selected = []
    for depth in range(2):
        for key in group_order:
            if len(selected) < max_rows and len(grouped[key]) > depth:
                selected.append(grouped[key][depth])
    selected.sort(key=lambda row: row["source_row"])
    if len({row["source_row"] for row in selected}) != len(selected):
        raise ValueError("A source row appears twice in selected memberships")
    if len({row["source_cell_id"] for row in selected}) != len(selected):
        raise ValueError("A source cell ID appears twice in selected memberships")
    return selected, {"membership_rows_scanned": scanned,
                      "n_plates": len(plate_sets["treated"]),
                      "n_plate_role_groups": len(grouped),
                      "well_candidates_by_role": dict(Counter(key[0] for key in best_per_well)),
                      "selection_seed": SEED,
                      "selection_uses_expression": False,
                      "method": "SHA256-ranked distinct wells per plate/role; SHA256-ranked cell within well; up to two wells per group"}


def inspect_rows(raw_path, feature_table, selected):
    selected_indices = np.array([row["source_row"] for row in selected], dtype=np.int64)
    rows = []
    with h5py.File(raw_path, "r") as source:
        matrix = source["X"]
        if not isinstance(matrix, h5py.Group) or matrix.attrs.get("encoding-type") != "csr_matrix":
            raise ValueError("Only on-disk CSR input is supported; no conversion is allowed")
        shape = tuple(int(n) for n in matrix.attrs["shape"])
        if selected_indices.min() < 0 or selected_indices.max() >= shape[0]:
            raise ValueError("Membership source row outside input dimensions")
        if len(matrix["indptr"]) != shape[0] + 1:
            raise ValueError("CSR indptr length differs from n_rows + 1")
        if len(matrix["data"]) != len(matrix["indices"]):
            raise ValueError("CSR data/indices length mismatch")
        obs = source["obs"]
        index_name = obs.attrs["_index"]
        local_ids = read_h5_column(obs, index_name, selected_indices)
        if list(local_ids) != [row["source_cell_id"] for row in selected]:
            raise ValueError("Selected source row/cell ID mapping does not match raw H5AD")
        for field in ("plate", "well", "cell_line", "replicate", "time"):
            values = read_h5_column(obs, field, selected_indices)
            if field == "time":
                matches = all(float(value) == float(row[field]) for value, row in zip(values, selected))
            else:
                matches = all(str(value) == row[field] for value, row in zip(values, selected))
            if not matches:
                raise ValueError(f"Selected membership/raw obs mismatch: {field}")
        labels = read_h5_column(obs, "perturbation", selected_indices)
        if not all((label == "control") == (row["role"] == "control") for label, row in zip(labels, selected)):
            raise ValueError("Selected role differs from raw exact control label")

        var = source["var"]
        identifiers = read_h5_column(var, "ensembl_id")
        symbols = read_h5_column(var, var.attrs["_index"])
        if len(identifiers) != shape[1] or len(symbols) != shape[1]:
            raise ValueError("Feature metadata/X shape mismatch")
        feature_rows = 0
        with feature_table.open("r", encoding="utf-8", newline="") as handle:
            for idx, feature in enumerate(csv.DictReader(handle, delimiter="\t")):
                if idx >= len(identifiers) or int(feature["source_feature_row"]) != idx:
                    raise ValueError("S1 feature row sequence differs from raw var")
                if feature["ensembl_id"] != identifiers[idx] or feature["gene_symbol"] != symbols[idx]:
                    raise ValueError(f"S1 feature identity mismatch at row {idx}")
                feature_rows += 1
        if feature_rows != shape[1]:
            raise ValueError("S1 feature table does not cover raw var")
        human = np.fromiter((str(value).startswith("ENSG") for value in identifiers), bool, count=shape[1])
        mouse = np.fromiter((str(value).startswith("ENSMUSG") for value in identifiers), bool, count=shape[1])
        other = ~(human | mouse)
        counts = Counter(identifiers)
        duplicates = {key: value for key, value in counts.items() if value > 1}
        total_entries = 0
        expression_start = time.perf_counter()
        for selected_row in selected:
            row_index = selected_row["source_row"]
            # Exactly one row's pointer pair and associated stored entries.
            start, stop = (int(n) for n in matrix["indptr"][row_index:row_index + 2])
            if not (0 <= start <= stop <= len(matrix["data"])):
                raise ValueError(f"Invalid CSR pointer bounds at row {row_index}")
            data = matrix["data"][start:stop]
            indices = matrix["indices"][start:stop]
            if len(data) != stop - start or len(indices) != len(data):
                raise ValueError("Truncated CSR slice")
            if np.any(indices < 0) or np.any(indices >= shape[1]):
                raise ValueError(f"CSR column index outside feature axis: {row_index}")
            finite = np.isfinite(data)
            nonnegative = bool(finite.all() and np.all(data >= 0))
            integer = np.zeros(len(data), dtype=bool)
            integer[finite] = np.isclose(data[finite], np.rint(data[finite]), atol=1e-8, rtol=0)
            library = float(np.sum(data, dtype=np.float64)) if finite.all() else None
            species_sums = {}
            for species, mask in (("human", human), ("mouse", mouse), ("other", other)):
                subtotal = float(np.sum(data[mask[indices]], dtype=np.float64)) if finite.all() else None
                species_sums[f"{species}_sum_observed"] = subtotal
                species_sums[f"{species}_fraction_observed"] = subtotal / library if nonnegative and library and library > 0 else None
            rows.append({"source_row": row_index, "source_cell_id": selected_row["source_cell_id"],
                         "role": selected_row["role"], "cell_line": selected_row["cell_line"],
                         "time": selected_row["time"], "replicate": selected_row["replicate"],
                         "plate": selected_row["plate"], "well": selected_row["well"],
                         "csr_start": start, "csr_stop": stop, "stored_entries_read": len(data),
                         "nonzero_entries_observed": int(np.count_nonzero(data)),
                         "nonfinite_entries": int((~finite).sum()),
                         "negative_finite_entries": int(np.sum(data[finite] < 0)),
                         "noninteger_finite_entries": int(np.sum(~integer[finite])),
                         "all_finite": bool(finite.all()), "all_nonnegative": nonnegative,
                         "all_integer_compatible": bool(integer.all()),
                         "observed_library_sum": library,
                         "duplicate_csr_column_entries": len(indices) - len(np.unique(indices)),
                         **species_sums})
            total_entries += len(data)
        expression_seconds = time.perf_counter() - expression_start
        axes = {"matrix_shape": list(shape), "matrix_dtype": str(matrix["data"].dtype),
                "matrix_encoding": "csr_matrix", "stored_entries_in_full_matrix": len(matrix["data"]),
                "selected_rows": len(selected), "stored_entries_read": total_entries,
                "csr_pointer_values_requested": 2 * len(selected),
                "logical_expression_bytes_requested": int(total_entries * (matrix["data"].dtype.itemsize + matrix["indices"].dtype.itemsize) + 2 * len(selected) * matrix["indptr"].dtype.itemsize),
                "physical_io_bytes": "NOT_MEASURED: HDF5 chunk decompression/cache may read additional bytes",
                "expression_read_and_row_qc_seconds": expression_seconds,
                "feature_rows_preserved": shape[1], "human_prefix_features": int(human.sum()),
                "mouse_prefix_features": int(mouse.sum()), "other_prefix_features": int(other.sum()),
                "duplicate_legacy_id_groups_preserved": len(duplicates),
                "feature_rows_in_duplicate_legacy_id_groups": sum(duplicates.values()),
                "feature_axis_deduplicated": False, "species_used_for_cell_filtering": False,
                "raw_obs_membership_checks": "PASS", "s1_feature_order_checks": "PASS"}
    return rows, axes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-h5ad", type=Path, default=RAW)
    parser.add_argument("--membership-dir", type=Path, default=MEMBERSHIP)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--max-rows", type=int, default=192)
    args = parser.parse_args()
    if not 1 <= args.max_rows <= 192:
        parser.error("--max-rows must be between 1 and 192")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {args.output_dir}")
    paths = [args.raw_h5ad, args.membership_dir / "treated_cell_membership.tsv.gz",
             args.membership_dir / "control_cell_membership.tsv.gz",
             args.membership_dir / "feature_axis.tsv"]
    before = [file_stat(path) for path in paths]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    audit = {"started_utc": datetime.now(timezone.utc).isoformat(),
             "limitations": LIMITATION, "full_matrix_qc_passed": False,
             "effect_construction_performed": False, "training_performed": False,
             "hvg_fitted": False, "normalization_performed": False,
             "input_hash_policy": "Raw H5AD and membership inputs not rehashed; pre/post content-relevant stat checked",
             "software": {"python": platform.python_version(), "h5py": h5py.__version__, "numpy": np.__version__},
             "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    failure = None
    try:
        metadata_start = time.perf_counter()
        selected, selection = select_cells(args.membership_dir, args.max_rows)
        audit["metadata_selection_seconds"] = time.perf_counter() - metadata_start
        row_qc, axes = inspect_rows(args.raw_h5ad, args.membership_dir / "feature_axis.tsv", selected)
        write_table_exclusive(args.output_dir / "selected_cells.tsv", selected)
        write_table_exclusive(args.output_dir / "row_qc.tsv", row_qc)
        compatible = all(row["all_finite"] and row["all_nonnegative"] and row["all_integer_compatible"]
                         and row["observed_library_sum"] > 0 for row in row_qc)
        audit.update({"status": "COMPLETED", "selection": selection, "matrix_sample": axes,
                      "sample_count_scale_conclusion": "raw_counts_compatible" if compatible else "unresolved",
                      "count_scale_basis": "Selected expression values are tested for finite, nonnegative, integer-compatible entries and positive row sums; no conclusion is drawn from the filename.",
                      "sample_qc_compatible_rows": sum(row["all_finite"] and row["all_nonnegative"] and row["all_integer_compatible"] and row["observed_library_sum"] > 0 for row in row_qc),
                      "count_state_identity_limit": "Integer-compatible sampled values support raw-count compatibility, not proof of every upstream operation or every matrix entry."})
        for metric in ("observed_library_sum", "human_fraction_observed", "mouse_fraction_observed", "other_fraction_observed"):
            values = np.array([row[metric] for row in row_qc if row[metric] is not None], dtype=float)
            audit.setdefault("sample_descriptive_ranges", {})[metric] = ({"min": float(values.min()), "median": float(np.median(values)), "max": float(values.max())} if len(values) else None)
    except Exception as exc:
        failure = exc
        audit.update({"status": "FAILED", "sample_count_scale_conclusion": "unresolved",
                      "error_type": type(exc).__name__, "error": str(exc)})
    after = [file_stat(path) for path in paths]
    stat_audit = [{"before": left, "after": right,
                   "content_relevant_stat_unchanged": content_stat_equal(left, right),
                   "access_time_unchanged": left["atime_ns"] == right["atime_ns"]}
                  for left, right in zip(before, after)]
    unchanged = all(item["content_relevant_stat_unchanged"] for item in stat_audit)
    audit.update({"input_content_relevant_stat_unchanged": unchanged,
                  "input_stat_comparison_excludes_atime": True,
                  "elapsed_seconds_before_final_report_write": time.perf_counter() - started,
                  "peak_process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
                  "peak_rss_semantics": "Linux ru_maxrss high-water mark for this process; bytes",
                  "finished_utc": datetime.now(timezone.utc).isoformat()})
    if not unchanged:
        audit["status"] = "FAILED_INPUT_STAT_CHANGED"
        audit["sample_count_scale_conclusion"] = "unresolved"
    write_json_exclusive(args.output_dir / "input_stat_audit.json", stat_audit)
    write_json_exclusive(args.output_dir / "audit.json", audit)
    with (args.output_dir / "README.md").open("x", encoding="utf-8") as handle:
        handle.write("# sci-Plex 3 有界表达抽样预检\n\n")
        handle.write(f"状态：`{audit['status']}`；矩阵尺度结论：`{audit['sample_count_scale_conclusion']}`。\n\n")
        handle.write("样本由membership metadata、固定SHA-256规则和plate/role分层确定，最多192行；仅逐选定row读取CSR指针及data/indices。selected_cells.tsv保存source row与cell ID，row_qc.tsv保存每行库大小、人鼠前缀计数比例及数值检查，audit.json保存读取量、耗时、RAM与结论。\n\n")
        handle.write("本次仅为小样本QC，不代表全矩阵QC通过。所有feature行均保留，包括45组重复legacy Ensembl ID；未做归一化、筛细胞、删鼠features、效应构建、HVG或训练。人鼠比例只描述选定行的矩阵尺度。\n\n")
        handle.write("输入前后比较size、mtime、ctime、device和inode；atime单独记录，读取可能更新atime。stat未变不等于全文件重新checksum认证。\n")
    print(json.dumps({"output_dir": str(args.output_dir), "status": audit["status"],
                      "selected_rows": audit.get("matrix_sample", {}).get("selected_rows"),
                      "stored_entries_read": audit.get("matrix_sample", {}).get("stored_entries_read"),
                      "count_scale": audit["sample_count_scale_conclusion"],
                      "input_stat_unchanged": unchanged,
                      "elapsed_seconds": audit["elapsed_seconds_before_final_report_write"],
                      "peak_rss_bytes": audit["peak_process_rss_bytes"]}, ensure_ascii=False))
    if failure is not None:
        raise failure
    if not unchanged:
        raise RuntimeError("Input content-relevant stat changed during read-only QC")


if __name__ == "__main__":
    main()
