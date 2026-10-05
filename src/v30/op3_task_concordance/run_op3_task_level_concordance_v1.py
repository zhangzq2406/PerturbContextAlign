#!/usr/bin/env python3
"""Descriptive within-task alignment-prediction concordance for frozen OP3 results.

For each frozen OP3 task, this script computes Spearman correlation across the
same six encoders between: global RSA or excess NDCG@10, and -MAE or DrugOrder.
The analysis is descriptive: it emits no p-values, confidence intervals, or
meta-analysis across tasks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

EXPECTED_ENCODERS = 6
EXPECTED_TASKS = 6
EXPECTED_ROWS = 36
COMPARISONS = [
    ("Global RSA vs -MAE", "rsa", "negative_mae"),
    ("Global RSA vs DrugOrder", "rsa", "drugorder"),
    ("Excess NDCG@10 vs -MAE", "excess_ndcg_mean", "negative_mae"),
    ("Excess NDCG@10 vs DrugOrder", "excess_ndcg_mean", "drugorder"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True, help="Frozen l2_l3_task_join.tsv")
    p.add_argument("--outdir", type=Path, required=True)
    a = p.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(a.input, sep="\t")
    if len(df) != EXPECTED_ROWS or df["model_key"].nunique() != EXPECTED_ENCODERS or df["task_id"].nunique() != EXPECTED_TASKS:
        raise RuntimeError("Unexpected frozen OP3 task/encoder cardinality")

    rows = []
    for (task_id, cell_type, donor_id), g in df.groupby(["task_id", "cell_type", "donor_id"], sort=False):
        if len(g) != EXPECTED_ENCODERS or g["model_key"].nunique() != EXPECTED_ENCODERS:
            raise RuntimeError(f"Task does not contain the same six unique encoders: {task_id}")
        values = {
            "rsa": g["rsa"].to_numpy(float),
            "excess_ndcg_mean": g["excess_ndcg_mean"].to_numpy(float),
            "negative_mae": -g["mae_mean"].to_numpy(float),
            "drugorder": g["gene_spearman_mean"].to_numpy(float),
        }
        for label, xname, yname in COMPARISONS:
            rho = float(spearmanr(values[xname], values[yname]).statistic)
            rows.append({
                "task_id": task_id,
                "cell_type": cell_type,
                "donor_id": donor_id,
                "comparison": label,
                "x_metric": xname,
                "y_metric": yname,
                "n_encoders": EXPECTED_ENCODERS,
                "spearman_rho": rho,
            })

    out = pd.DataFrame(rows)
    out.to_csv(a.outdir / "op3_task_level_concordance.tsv", sep="\t", index=False)

    summary = []
    for label, g in out.groupby("comparison", sort=False):
        v = g["spearman_rho"].to_numpy(float)
        summary.append({
            "comparison": label,
            "n_tasks": len(v),
            "min_rho": float(v.min()),
            "max_rho": float(v.max()),
            "mean_rho": float(v.mean()),
            "median_rho": float(np.median(v)),
            "positive_tasks": int((v > 0).sum()),
            "negative_tasks": int((v < 0).sum()),
            "zero_tasks": int((v == 0).sum()),
        })
    pd.DataFrame(summary).to_csv(a.outdir / "op3_concordance_summary.tsv", sep="\t", index=False)

    audit = {
        "status": "PASS_DESCRIPTIVE_RECOMPUTATION",
        "input_sha256": sha256(a.input),
        "input_rows": len(df),
        "task_n": int(df["task_id"].nunique()),
        "encoder_n": int(df["model_key"].nunique()),
        "definition": "Within each frozen OP3 task, Spearman correlation across the same six encoders; -MAE is used so higher is better; DrugOrder is gene_spearman_mean.",
        "inference": "descriptive only; no p-values, no task-independence assumption, no meta-analysis",
        "output_rows": len(out),
    }
    (a.outdir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
