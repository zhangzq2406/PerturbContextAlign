#!/usr/bin/env python3
"""Build a read-only-input, metadata-only sci-Plex v2 preflight.

Reads obs/var and X attributes, never X values. Candidate control matches and
splits are NOT authorized effect construction or training. Existing outputs are
refused. Raw, minimal-schema, pilot and manuscript files are never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def stable_id(namespace, values):
    content = json.dumps([namespace, *values], ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False)
    return namespace + ":" + hashlib.sha256(content.encode()).hexdigest()[:24]


def read_column(group, key):
    if key not in group:
        raise KeyError(f"Required H5AD metadata field absent: {key}")
    node = group[key]
    if isinstance(node, h5py.Group):
        if not {"categories", "codes"} <= set(node):
            raise ValueError(f"Unsupported metadata encoding: {key}")
        cats = node["categories"]
        values = cats.asstr()[:] if cats.dtype.kind in "OSU" else cats[:]
        codes = node["codes"][:]
        if np.any(codes < -1) or np.any(codes >= len(values)):
            raise ValueError(f"Invalid categorical code: {key}")
        out = np.full(len(codes), None, dtype=object)
        valid = codes >= 0
        out[valid] = values[codes[valid]]
        return out
    return node.asstr()[:] if node.dtype.kind in "OSU" else node[:]


def normalize_name(value):
    if pd.isna(value):
        return ""
    return " ".join(str(value).split())


def file_record(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "device": stat.st_dev,
            "inode": stat.st_ino, "sha256_status": "NOT_REHASHED_LARGE_INPUT"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_table(frame, path):
    if path.exists():
        raise FileExistsError(path)
    frame.to_csv(path, sep="\t", index=False, float_format="%.17g")


def make_atoms(treated, dataset_id):
    keys = ["source_entity_key", "cell_line", "dose_value", "dose_unit", "time"]
    t = treated.copy()
    grouped = t.groupby(keys, dropna=False, sort=True)
    atoms = grouped.agg(n_treated_cells=("source_row", "size"),
                        n_replicate_labels=("replicate", "nunique"),
                        n_plate_labels=("plate", "nunique"),
                        n_well_labels=("well", "nunique"),
                        n_legacy_condition_ids=("legacy_condition_id", "nunique"),
                        legacy_condition_id=("legacy_condition_id", "first"),
                        source_entity_name=("source_entity_name", "first")).reset_index()
    if not atoms.n_legacy_condition_ids.eq(1).all():
        raise ValueError("A new atom maps to multiple legacy condition IDs")
    atoms.insert(0, "atomic_id", [stable_id("atom", [dataset_id, *row])
                                  for row in atoms[keys].itertuples(index=False, name=None)])
    if atoms.atomic_id.duplicated().any():
        raise ValueError("Atomic ID collision")
    t = t.merge(atoms[keys + ["atomic_id"]], on=keys, validate="many_to_one")
    return atoms, t


def measurements_and_controls(treated, controls, policies, thresholds):
    keys = ["atomic_id", "cell_line", "time", "replicate", "plate", "well"]
    measurements = treated.groupby(keys, dropna=False, sort=True).size().rename(
        "n_treated_cells").reset_index()
    measurements.insert(0, "measurement_id", [stable_id("measurement", list(row))
                                              for row in measurements[keys].itertuples(index=False, name=None)])
    all_pools, all_links, summaries = [], [], []
    for policy, fields in policies.items():
        if controls[fields].isna().any().any() or measurements[fields].isna().any().any():
            raise ValueError(f"Missing proposed matching keys: {policy}")
        pools = controls.groupby(fields, sort=True).size().rename("n_control_cells").reset_index()
        pools.insert(0, "control_pool_id", [stable_id("controlpool", [policy, *row])
                                            for row in pools[fields].itertuples(index=False, name=None)])
        pools.insert(0, "candidate_policy", policy)
        all_pools.append(pools)
        links = measurements.merge(pools, on=fields, how="left", validate="many_to_one")
        links["candidate_policy"] = policy
        links["n_control_cells"] = links.n_control_cells.fillna(0).astype(int)
        links["control_pool_id"] = links.control_pool_id.fillna("")
        links["match_status"] = np.where(links.n_control_cells.gt(0),
                                         "CANDIDATE_MATCH_NOT_APPROVED", "NO_MATCH")
        all_links.append(links)
        for threshold in thresholds:
            per_atom = links.groupby("atomic_id").n_control_cells.min()
            summaries.append({"candidate_policy": policy, "control_cell_threshold": threshold,
                              "n_measurements": len(links),
                              "n_measurements_meeting_threshold": int(links.n_control_cells.ge(threshold).sum()),
                              "n_atoms": len(per_atom),
                              "n_atoms_all_measurements_meet_threshold": int(per_atom.ge(threshold).sum()),
                              "threshold_status": "AUDIT_ONLY_NOT_INCLUSION_RULE"})
    return measurements, pd.concat(all_pools, ignore_index=True), pd.concat(all_links, ignore_index=True), pd.DataFrame(summaries)


def candidate_splits(atoms, lines):
    key = ["source_entity_key", "dose_value", "dose_unit", "time"]
    records = []
    for target in lines:
        source = atoms.loc[atoms.cell_line.ne(target)]
        support = source.groupby(key).cell_line.nunique().rename("n_source_contexts").reset_index()
        frame = atoms.merge(support, on=key, how="left", validate="many_to_one")
        frame["n_source_contexts"] = frame.n_source_contexts.fillna(0).astype(int)
        frame["target_context"] = target
        frame["candidate_role"] = np.where(frame.cell_line.eq(target), "test", "source")
        frame["split_status"] = "CANDIDATE_METADATA_ONLY_NOT_FORMAL_TRAINING_SPLIT"
        test = frame.loc[frame.candidate_role.eq("test")]
        if not test.n_source_contexts.ge(1).all():
            raise ValueError(f"Seen-entity contract missing source support: {target}")
        records.append(frame[["atomic_id", "source_entity_key", "cell_line", "dose_value",
                              "time", "target_context", "candidate_role", "n_source_contexts", "split_status"]])
    return pd.concat(records, ignore_index=True)


def feature_audit(h5):
    var = h5["var"]
    index_key = var.attrs["_index"]
    ids = pd.Series(read_column(var, "ensembl_id"), dtype="string")
    symbols = pd.Series(read_column(var, index_key), dtype="string")
    base = ids.str.replace(r"\.\d+$", "", regex=True)
    species = np.select([base.str.startswith("ENSG", na=False),
                         base.str.startswith("ENSMUSG", na=False)],
                        ["human_ensembl", "mouse_ensembl"], default="other_or_missing")
    axis = pd.DataFrame({"source_feature_row": np.arange(len(ids)), "ensembl_id": ids,
                         "ensembl_id_without_version": base, "gene_symbol": symbols,
                         "identifier_species_prefix": species})
    axis["duplicate_versionless_id"] = base.duplicated(keep=False)
    summary = {"n_features": len(axis), "species_prefix_counts": axis.identifier_species_prefix.value_counts().to_dict(),
               "n_duplicate_versionless_id_rows": int(axis.duplicate_versionless_id.sum()),
               "n_duplicate_versionless_id_excess": int(base.duplicated().sum()),
               "n_missing_ensembl_id": int(ids.isna().sum()),
               "feature_filtering_performed": False,
               "human_axis_policy": "REQUIRES_REVIEW_BEFORE_EXPRESSION_PROCESSING"}
    return axis, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/stage1_contract.json")
    parser.add_argument("--output", type=Path, default=ROOT / "metadata/sciplex_preflight_v1")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refuse to overwrite existing run: {args.output}")
    config = json.loads(args.config.read_text())
    if config["training_allowed"] or config["expression_values_allowed_in_this_stage"]:
        raise ValueError("This program is exclusively a metadata-only preflight")
    paths = [config[k] for k in ("raw_h5ad", "schema_h5ad", "legacy_pilot_cohort")]
    before = [file_record(p) for p in paths]
    fields = ["perturbation", "cell_line", "dose_value", "dose_unit", "time", "replicate", "plate", "well"]
    with h5py.File(config["raw_h5ad"], "r") as raw, h5py.File(config["schema_h5ad"], "r") as schema:
        raw_ids = read_column(raw["obs"], raw["obs"].attrs["_index"])
        schema_ids = read_column(schema["obs"], schema["obs"].attrs["_index"])
        if not np.array_equal(raw_ids, schema_ids):
            raise ValueError("Raw/schema obs indices differ; no implicit positional join")
        if len(np.unique(raw_ids)) != len(raw_ids):
            raise ValueError("Nonunique raw cell ID")
        obs = pd.DataFrame({field: read_column(raw["obs"], field) for field in fields})
        obs.insert(0, "source_row", np.arange(len(obs)))
        obs["source_cell_id"] = raw_ids
        obs["legacy_condition_id"] = read_column(schema["obs"], "C_condition_id")
        obs["legacy_control"] = read_column(schema["obs"], "C_control_indicator").astype(bool)
        axis, feature_summary = feature_audit(raw)
        x_shape = list(map(int, raw["X"].attrs["shape"]))
    obs["source_entity_name"] = obs.perturbation.map(normalize_name)
    obs["source_entity_key"] = obs.source_entity_name.map(
        lambda name: stable_id("localentity", [config["dataset_id"], name]) if name else "")
    # Vehicle is identified from an exact reviewed label, not by substring or missingness.
    vehicle = obs.source_entity_name.isin(config["vehicle_names_exact"])
    annotated = obs.source_entity_name.ne("")
    if not np.array_equal(vehicle[annotated].to_numpy(), obs.loc[annotated, "legacy_control"].to_numpy()):
        raise ValueError("Raw vehicle label and minimal-schema control classification disagree")
    context = obs.cell_line.isin(config["cell_lines"])
    duration = obs.time.eq(config["duration_raw"])
    exposure = obs.dose_value.isin(config["dose_values_nM"]) & obs.dose_unit.eq("nM")
    selected_treated = annotated & ~vehicle & context & duration & exposure
    selected_control = vehicle & context & duration
    obs["selection_status"] = "OUTSIDE_SELECTED_DURATION_OR_EXPOSURE"
    obs.loc[~annotated | ~context, "selection_status"] = "MISSING_ENTITY_OR_SUPPORTED_CONTEXT"
    obs.loc[selected_treated, "selection_status"] = "CANDIDATE_TREATED"
    obs.loc[selected_control, "selection_status"] = "CANDIDATE_CONTROL"
    treated = obs.loc[selected_treated].copy()
    controls = obs.loc[selected_control].copy()
    if treated.empty or controls.empty:
        raise ValueError("Empty treated/control candidate; check exact vehicle and duration")
    if treated[["replicate", "plate", "well"]].isna().any().any():
        raise ValueError("Missing measurement keys in selected treated")
    atoms, treated = make_atoms(treated, config["dataset_id"])
    measurements, pools, links, control_summary = measurements_and_controls(
        treated, controls, config["candidate_control_policies"], config["support_audit_thresholds"])
    splits = candidate_splits(atoms, config["cell_lines"])
    legacy = pd.read_csv(config["legacy_pilot_cohort"], sep="\t", keep_default_na=False)
    mapping = atoms[["atomic_id", "source_entity_key", "source_entity_name", "legacy_condition_id", "cell_line"]].merge(
        legacy[["aggregate_uid", "condition_id", "cell_line"]],
        left_on=["legacy_condition_id", "cell_line"], right_on=["condition_id", "cell_line"],
        how="left", validate="one_to_one")
    if mapping.aggregate_uid.isna().any() or len(mapping) != len(legacy):
        raise ValueError("New atom/legacy pilot coverage mismatch")
    counts = treated.groupby("atomic_id").size()
    if not atoms.set_index("atomic_id").n_treated_cells.sort_index().equals(counts.sort_index()):
        raise ValueError("Atom membership count mismatch")
    measurement_keys = ["atomic_id", "cell_line", "time", "replicate", "plate", "well"]
    treated = treated.merge(measurements[measurement_keys + ["measurement_id"]],
                            on=measurement_keys, validate="many_to_one")
    membership_columns = ["source_row", "source_cell_id", "legacy_condition_id", "source_entity_key",
                          "cell_line", "time", "replicate", "plate", "well"]
    source_ledger = obs.selection_status.value_counts().rename_axis("selection_status").reset_index(name="n_cells")
    support = pd.DataFrame([{"treated_cell_threshold": n, "n_atoms": len(atoms),
                             "n_atoms_meeting_threshold": int(atoms.n_treated_cells.ge(n).sum()),
                             "status": "AUDIT_ONLY_NOT_INCLUSION_RULE"}
                            for n in config["support_audit_thresholds"]])
    after = [file_record(p) for p in paths]
    if before != after:
        raise RuntimeError("Input metadata changed during read-only preflight")
    args.output.mkdir(parents=True, exist_ok=False)
    tables = {"atomic_conditions.tsv": atoms, "measurements.tsv": measurements,
              "candidate_control_pools.tsv": pools, "candidate_control_links.tsv": links,
              "candidate_control_support.tsv": control_summary, "candidate_loco_splits.tsv": splits,
              "legacy_to_atomic_mapping.tsv": mapping, "source_selection_ledger.tsv": source_ledger,
              "treated_support_thresholds.tsv": support,
              "feature_axis.tsv": axis, "duplicate_feature_ids.tsv": axis.loc[axis.duplicate_versionless_id],
              "source_entity_catalog.tsv": obs.loc[annotated, ["perturbation", "source_entity_name", "source_entity_key"]].drop_duplicates(),
              "treated_cell_membership.tsv.gz": treated[membership_columns + ["atomic_id", "measurement_id"]].sort_values("source_row"),
              "control_cell_membership.tsv.gz": controls[membership_columns].sort_values("source_row")}
    for name, frame in tables.items():
        save_table(frame, args.output / name)
    report = {"stage": "S1_METADATA_PREFLIGHT", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "metadata_audit_pass": True, "formal_effects_ready": False, "training_ready": False,
              "expression_values_read": False, "raw_inputs_unchanged_stat_identity": before == after,
              "large_input_hashes_recomputed": False, "source_records": before,
              "config_sha256": digest(args.config), "code_sha256": digest(Path(__file__)),
              "n_raw_cells": len(obs), "n_candidate_treated_cells": len(treated),
              "n_candidate_control_cells": len(controls), "n_atoms": len(atoms),
              "n_entities": int(atoms.source_entity_key.nunique()),
              "n_measurements": len(measurements), "n_candidate_split_rows": len(splits),
              "raw_X_shape": x_shape, "feature_axis": feature_summary,
              "pending_gates": config["gates_before_effects_or_training"],
              "interpretation": "Candidate matches/splits only. No final threshold, effect vector, model, or manuscript result was produced."}
    (args.output / "preflight_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    output_manifest = [{"file": p.name, "size_bytes": p.stat().st_size, "sha256": digest(p)}
                       for p in sorted(args.output.iterdir()) if p.is_file()]
    (args.output / "output_manifest.json").write_text(json.dumps(output_manifest, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("metadata_audit_pass", "formal_effects_ready", "training_ready",
                     "n_raw_cells", "n_candidate_treated_cells", "n_candidate_control_cells", "n_atoms",
                     "n_entities", "n_measurements")}, ensure_ascii=False, indent=2))
    print(f"Saved metadata-only run: {args.output}")


if __name__ == "__main__":
    main()
