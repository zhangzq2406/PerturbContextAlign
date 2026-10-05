# Methods and release scope

This repository contains the analysis and result-generation code used for the current PerturbContextAlign manuscript. It is intentionally **analysis-only**: per-panel figure rendering code is distributed separately, and third-party raw data/model weights are not redistributed here.

## Scientific scope

PerturbContextAlign asks two questions:

1. How do different pretrained text encoders represent perturbation context?
2. How do these context representations correspond to experimentally measured perturbation responses?

The released analyses cover context retention, broad text-response correspondence, fixed-perturbation cross-context correspondence, controlled mechanism-text perturbations, and downstream prediction-oriented tests.

## Context views

- **P1:** perturbation identity and intervention structure.
- **P2:** P1 plus biological context.
- **P3:** P1 plus experimental/exposure context.
- **P4:** identity, biological context, and experimental/exposure context.

Exact fields are encoded by the released scripts/configuration and should not be inferred from these short labels alone.

## Broad measured-response analysis

The primary broad response unit is condition × biological context. Biological context uses species, cell type, cell line, tissue, and disease. Donor and batch are nuisance-matching variables rather than response identity. The corrected control hierarchy uses exact biological context with donor+batch matching when feasible, otherwise donor matching; broader fallback is not allowed.

For the OP3 broad R2 workflow, released `X` is treated as log1p library-normalized expression. The accepted implementation converts each cell to the corresponding linear normalized scale with `expm1`, averages treated and matched-control cells using the frozen control weights, and computes a pseudocount-1 log2 ratio. This does not recover raw UMI counts or perform a second library normalization. The independent fine-grained OP3 donor-transfer analysis retains its own released-log-scale mean-difference definition.

## Mechanism controls

The final sci-Plex mechanism analysis compares correct drug-mechanism matching with 20 prespecified length-balanced mismatched controls. These mappings are deterministic negative controls, not independent biological replicates and not Monte Carlo samples from a formal null distribution. Public-safe mapping metadata are under `inputs/v30/mechanism_length_balanced/`.

## Downstream prediction

Prediction analyses are downstream tests of representation utility. They are not intended as a new perturbation-prediction benchmark or state-of-the-art predictor. OP3 uses cross-donor transfer/prediction as a downstream test; target-donor identity is not supplied as text input in the main analysis.
