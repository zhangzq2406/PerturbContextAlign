#!/usr/bin/env python3
"""Verify official GEO design/gene tables against local metadata; never read X."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import urllib.request

import h5py
import numpy as np
import pandas as pd

from prepare_sciplex_metadata import ROOT, read_column, file_record, save_table, digest

SOURCE_ROOT = "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4150nnn/GSM4150378/suppl/"
SOURCES = {
    "design": ("GSM4150378_sciPlex3_A549_MCF7_K562_hashTable_metadata.txt.gz",
               "98a65191d343d149c5742cf6e7b734d403e853d165da800fa2daea41b978a2ff"),
    "genes": ("GSM4150378_sciPlex3_A549_MCF7_K562_screen_gene.annotations.txt.gz",
              "fbe43028cfb75dc5ebf383cbbbb100da6b51e7be8e2e74f6354f7c59b9fda7c2")
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/stage1_contract.json")
    ap.add_argument("--output", type=Path, default=ROOT / "inputs/sciplex_geo_verified_v1")
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refuse to overwrite source-verification run: {args.output}")
    config = json.loads(args.config.read_text())
    compressed, manifest = {}, []
    for kind, (filename, expected) in SOURCES.items():
        url = SOURCE_ROOT + filename
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read(20 * 1024 * 1024)
            if response.read(1):
                raise ValueError("Unexpectedly large official metadata table")
        observed = hashlib.sha256(data).hexdigest()
        if observed != expected:
            raise ValueError(f"Official source checksum differs from reviewed source: {filename}")
        compressed[kind] = data
        manifest.append({"kind": kind, "url": url, "filename": filename, "bytes": len(data),
                         "sha256": observed, "retrieved_at_utc": datetime.now(timezone.utc).isoformat()})
    design = pd.read_csv(io.BytesIO(gzip.decompress(compressed["design"])), sep="\t")
    genes = pd.read_csv(io.BytesIO(gzip.decompress(compressed["genes"])), sep=r"\s+")
    source_before = file_record(config["raw_h5ad"])
    with h5py.File(config["raw_h5ad"], "r") as h5:
        obs = pd.DataFrame({k: read_column(h5["obs"], k) for k in
                            ["well", "plate", "cell_line", "replicate", "time", "dose_value", "perturbation"]})
        local_gene_ids = pd.Series(read_column(h5["var"], "ensembl_id"), dtype="string")
        local_symbols = pd.Series(read_column(h5["var"], h5["var"].attrs["_index"]), dtype="string")
    well_key = ["well", "plate"]
    missing = obs[well_key].isna().any(axis=1)
    grouped = obs.loc[~missing].groupby(well_key, sort=True)
    for field in ["cell_line", "replicate", "time", "dose_value", "perturbation"]:
        if not grouped[field].nunique(dropna=False).eq(1).all():
            raise ValueError(f"Local well has ambiguous metadata: {field}")
    wells = grouped.agg(n_cells=("time", "size"), cell_line=("cell_line", "first"),
                        replicate=("replicate", "first"), time=("time", "first"),
                        dose_value=("dose_value", "first"), perturbation=("perturbation", "first")).reset_index()
    if design.duplicated(["well_oligo", "plate_oligo"]).any():
        raise ValueError("Official design has nonunique matching keys")
    joined = wells.merge(design, left_on=well_key, right_on=["well_oligo", "plate_oligo"],
                          how="outer", suffixes=("_local", "_geo"), validate="one_to_one", indicator=True)
    if not joined["_merge"].eq("both").all():
        raise ValueError("Local/source well sets do not match exactly")
    comparisons = {"cell_line": joined.cell_line.eq(joined.cell_type),
                   "replicate": joined.replicate_local.eq(joined.replicate_geo),
                   "time": joined.time.eq(joined.time_point),
                   "dose": joined.dose_value.eq(joined.dose),
                   "control": joined.perturbation.eq("control").eq(joined.vehicle)}
    for label, equal in comparisons.items():
        if not equal.all():
            raise ValueError(f"Official/local metadata mismatch: {label}, n={int((~equal).sum())}")
    if len(genes) != len(local_gene_ids):
        raise ValueError("Official/local gene-axis lengths differ")
    expected_truncated = genes.id.astype("string").str.split(".").str[0]
    if not np.array_equal(expected_truncated.to_numpy(), local_gene_ids.to_numpy()):
        raise ValueError("Official gene IDs do not reproduce local truncated IDs row-by-row")
    # Reproduce the known anndata symbol uniquification; do not change any H5AD.
    from anndata.utils import make_index_unique
    expected_symbols = pd.Series(make_index_unique(pd.Index(genes.gene_short_name.astype(str)), join=":"), dtype="string")
    if not np.array_equal(expected_symbols.to_numpy(), local_symbols.to_numpy()):
        raise ValueError("Official gene symbols do not reproduce local symbols row-by-row")
    axis = pd.DataFrame({"source_feature_row": np.arange(len(genes)),
                         "original_ensembl_id": genes.id,
                         "original_gene_symbol": genes.gene_short_name,
                         "legacy_truncated_ensembl_id": local_gene_ids,
                         "legacy_unique_gene_symbol": local_symbols})
    axis["is_human_identifier"] = axis.original_ensembl_id.str.startswith("ENSG")
    axis["is_mouse_identifier"] = axis.original_ensembl_id.str.startswith("ENSMUSG")
    axis["is_PAR_Y_feature"] = axis.original_ensembl_id.str.endswith("_PAR_Y")
    axis["legacy_id_collided"] = axis.legacy_truncated_ensembl_id.duplicated(keep=False)
    if axis.original_ensembl_id.duplicated().any():
        raise ValueError("Official full feature IDs are not unique; must review before processing")
    source_after = file_record(config["raw_h5ad"])
    if source_before != source_after:
        raise RuntimeError("Local source changed during verification")
    report = {"status": "SOURCE_METADATA_VERIFIED", "n_design_wells": len(design),
              "n_local_wells": len(wells), "n_joined_wells": len(joined),
              "n_cells_without_local_well_mapping": int(missing.sum()),
              "mismatches": {k: int((~v).sum()) for k, v in comparisons.items()},
              "n_vehicle_wells": int(design.vehicle.sum()),
              "n_plates": int(design.plate_oligo.nunique()),
              "vehicle_wells_per_plate": design.groupby("plate_oligo").vehicle.sum().value_counts().to_dict(),
              "n_human_feature_rows": int(axis.is_human_identifier.sum()),
              "n_mouse_feature_rows": int(axis.is_mouse_identifier.sum()),
              "n_PAR_Y_rows": int(axis.is_PAR_Y_feature.sum()),
              "n_original_feature_ids_unique": int(axis.original_ensembl_id.nunique()),
              "gene_axis_original_order_verified": True,
              "raw_file_unchanged_stat_identity": source_before == source_after,
              "X_values_read": False, "feature_aggregation_performed": False,
              "config_sha256": digest(args.config), "code_sha256": digest(Path(__file__)),
              "remaining_expression_gate": "Inspect count scale; explicitly freeze human feature/PAR_Y handling, controls, weighting and train-only transformations before effects."}
    args.output.mkdir(parents=True, exist_ok=False)
    for kind, (filename, _) in SOURCES.items():
        (args.output / filename).write_bytes(compressed[kind])
    save_table(joined, args.output / "local_wells_to_geo.tsv")
    save_table(axis, args.output / "restored_feature_axis.tsv")
    save_table(axis.loc[axis.legacy_id_collided], args.output / "legacy_feature_collisions.tsv")
    (args.output / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "source_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
