#!/usr/bin/env python3
"""Generate the prespecified response-blind length-balanced mechanism derangements.

The input is a 57-row frozen mechanism corpus with one row per drug/entity.
No response, prediction, embedding, or model output is read. The algorithm
constructs deterministic one-to-one derangements that prohibit self and
identical-text assignments, minimize a model-independent text-length cost, and
select 20 diverse assignments from a fixed candidate pool.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

SEEDS = list(range(2026100521, 2026100541))
AMPS = [0.02, 0.05, 0.10, 0.20, 0.40]
REPS = 80
MAX_RATIO = 1.10
N_SELECT = 20


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mechanism-corpus", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    a = p.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)

    corpus = pd.read_csv(a.mechanism_corpus, sep="\t")
    required = ["source_entity_key", "text_sha256", "n_chars", "n_whitespace_tokens"]
    if list(corpus.columns) != required or len(corpus) != 57 or not corpus["source_entity_key"].is_unique:
        raise RuntimeError("Unexpected frozen 57-drug mechanism corpus")
    corpus = corpus.sort_values("source_entity_key").reset_index(drop=True)
    ci = corpus["n_chars"].to_numpy(float)
    cj = ci.copy()
    ti = corpus["n_whitespace_tokens"].to_numpy(float)
    tj = ti.copy()
    base = 0.5 * np.abs(np.log((cj[None, :] + 1) / (ci[:, None] + 1))) + 0.5 * np.abs(np.log((tj[None, :] + 1) / (ti[:, None] + 1)))
    valid = np.ones(base.shape, dtype=bool)
    for i, x in corpus.iterrows():
        for j, y in corpus.iterrows():
            if x.source_entity_key == y.source_entity_key or x.text_sha256 == y.text_sha256:
                valid[i, j] = False
    c0 = base.copy(); c0[~valid] = 1e6
    rr, cc = linear_sum_assignment(c0)
    if not (c0[rr, cc] < 1e5).all():
        raise RuntimeError("No valid complete derangement exists")
    optimum = float(base[rr, cc].sum())

    pool = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        for amp in AMPS:
            for rep in range(REPS):
                cost = base + amp * rng.random(base.shape)
                cost[~valid] = 1e6
                r, c = linear_sum_assignment(cost)
                if not (cost[r, c] < 1e5).all():
                    continue
                sig = tuple(corpus["text_sha256"].iloc[c].tolist())
                rec = (float(base[r, c].sum()), seed, amp, rep, c.copy())
                if sig not in pool or rec[:4] < pool[sig][:4]:
                    pool[sig] = rec

    eligible = []
    for sig, (cost, seed, amp, rep, c) in pool.items():
        if cost <= optimum * MAX_RATIO + 1e-12:
            eligible.append((sig, cost, seed, amp, rep, c))
    if len(eligible) < N_SELECT:
        raise RuntimeError("Insufficient eligible derangements")
    eligible.sort(key=lambda x: (x[1], x[0], x[2], x[3], x[4]))

    selected = [eligible[0]]
    remaining = eligible[1:]
    while len(selected) < N_SELECT:
        best_i = None; best_key = None
        for i, x in enumerate(remaining):
            sig = x[0]
            min_hamming = min(sum(a != b for a, b in zip(sig, s[0])) for s in selected)
            key = (min_hamming, -x[1], tuple(sig), -x[2], -x[3], -x[4])
            if best_key is None or key > best_key:
                best_key, best_i = key, i
        selected.append(remaining.pop(best_i))

    manifest, quality, text_sigs = [], [], []
    for pidx, (sig, total_cost, seed, amp, rep, c) in enumerate(selected):
        text_sigs.append(sig)
        pid = f"length_balanced_{pidx:02d}"
        char_rel, token_rel, char_ratio, token_ratio, char_abs, token_abs = [], [], [], [], [], []
        for i, j in enumerate(c):
            x, y = corpus.iloc[i], corpus.iloc[j]
            cr = abs(float(y.n_chars) - float(x.n_chars)) / float(x.n_chars)
            tr = abs(float(y.n_whitespace_tokens) - float(x.n_whitespace_tokens)) / float(x.n_whitespace_tokens)
            crat = max(float(y.n_chars)/float(x.n_chars), float(x.n_chars)/float(y.n_chars))
            trat = max(float(y.n_whitespace_tokens)/float(x.n_whitespace_tokens), float(x.n_whitespace_tokens)/float(y.n_whitespace_tokens))
            ca, ta = abs(int(y.n_chars)-int(x.n_chars)), abs(int(y.n_whitespace_tokens)-int(x.n_whitespace_tokens))
            char_rel.append(cr); token_rel.append(tr); char_ratio.append(crat); token_ratio.append(trat); char_abs.append(ca); token_abs.append(ta)
            manifest.append({
                "perm_index": pidx, "perm_id": pid, "design_seed": seed, "jitter_amplitude": amp, "candidate_rep": rep,
                "target_entity": x.source_entity_key, "shuffled_from_entity": y.source_entity_key,
                "original_text_sha256": x.text_sha256, "shuffled_text_sha256": y.text_sha256,
                "original_n_chars": int(x.n_chars), "shuffled_n_chars": int(y.n_chars),
                "original_n_whitespace_tokens": int(x.n_whitespace_tokens), "shuffled_n_whitespace_tokens": int(y.n_whitespace_tokens),
                "char_abs_diff": ca, "token_abs_diff": ta, "char_abs_relative_diff": cr, "token_abs_relative_diff": tr
            })
        quality.append({
            "perm_index": pidx, "perm_id": pid, "design_seed": seed, "jitter_amplitude": amp, "candidate_rep": rep,
            "total_length_cost": total_cost, "cost_ratio_to_global_optimum": total_cost/optimum,
            "mean_char_abs_relative_diff": float(np.mean(char_rel)), "median_char_length_ratio": float(np.median(char_ratio)), "max_char_length_ratio": float(np.max(char_ratio)),
            "mean_token_abs_relative_diff": float(np.mean(token_rel)), "median_token_length_ratio": float(np.median(token_ratio)), "max_token_length_ratio": float(np.max(token_ratio)),
            "mean_char_abs_diff": float(np.mean(char_abs)), "mean_token_abs_diff": float(np.mean(token_abs)),
        })

    man = pd.DataFrame(manifest); qual = pd.DataFrame(quality)
    man_path = a.outdir / "length_balanced_permutation_manifest.tsv"
    qual_path = a.outdir / "length_balanced_permutation_quality.tsv"
    man.to_csv(man_path, sep="\t", index=False)
    qual.to_csv(qual_path, sep="\t", index=False)

    hamming = []
    for i in range(N_SELECT):
        for j in range(i):
            n = sum(x != y for x, y in zip(text_sigs[i], text_sigs[j]))
            hamming.append({"perm_i": i, "perm_j": j, "different_target_assignments": n, "fraction_different": n/57})
    pd.DataFrame(hamming).to_csv(a.outdir / "pairwise_assignment_diversity.tsv", sep="\t", index=False)

    audit = {
        "status": "PASS_RESPONSE_BLIND_LENGTH_BALANCED_MAPPING",
        "input_sha256": sha256(a.mechanism_corpus),
        "response_or_prediction_values_read": False,
        "encoding_run": False,
        "prediction_run": False,
        "truth_scoring_run": False,
        "n_selected": N_SELECT,
        "n_drugs": 57,
        "global_optimum_total_cost": optimum,
        "eligible_cost_ratio_threshold": MAX_RATIO,
        "output_mapping_sha256": sha256(man_path),
    }
    (a.outdir / "preflight_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
