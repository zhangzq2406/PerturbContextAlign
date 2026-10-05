# PerturbContextAlign analysis runbook

## 1. Install the lightweight verification environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For the broader scientific stack:

```bash
pip install -r requirements-analysis.txt
```

The analysis requirements are not a complete historical GPU lockfile. Pretrained model weights must be obtained separately.

## 2. Run repository tests

```bash
python -m unittest discover -s tests -v
```

These tests do not open the full expression datasets or rerun encoders.

## 3. Verify companion public source data

After downloading the Figshare source-data archive:

```bash
python tools/verify_release.py \
  --data-root /path/to/PerturbContextAlign_Figshare/sources
```

Verify summary derivations:

```bash
python tools/verify_summary_derivations.py \
  --figshare-root /path/to/PerturbContextAlign_Figshare \
  --report /tmp/summary_derivations.json
```

The report path should not already exist.

## 4. Inspect external-input readiness

```bash
python tools/preflight_inputs.py /path/to/resolved_config.json
```

This is a path/readiness check, not a full scientific validation gate.

## 5. Broad R1/R2 analyses

The broad scripts expect a reconstructed study-root layout with the required external datasets and frozen metadata. Inspect command-line help before running:

```bash
python src/r1/r1_full_pipeline.py --help
python src/r2/response_geometry/run_r2_measured_response_effect_directh5_v1.py --help
python src/r2/alignment/run_r2_primary_paper_closure_v2.py --help
```

Use the released R2 nuisance-control configuration and the method boundaries in `docs/METHODS_AND_SCOPE.md`.

## 6. Fine-grained analyses

`src/fine/` contains the retained sci-Plex/OP3 analysis code and configurations. Configure paths into a new writable workspace rather than editing frozen inputs in place. `tools/configure.py` can resolve environment-tokenized JSON files into a new destination.

## 7. Table-only OP3 concordance

```bash
python src/v30/op3_task_concordance/run_op3_task_level_concordance_v1.py \
  --input /path/to/l2_l3_task_join.tsv \
  --outdir /path/to/output/op3_concordance
```

This recomputes descriptive within-task encoder concordance from frozen task-level results and does not refit predictors.

## 8. Length-balanced mechanism mappings

```bash
python src/v30/mechanism_length_balanced_control/generate_length_balanced_derangements_v1.py \
  --mechanism-corpus inputs/v30/mechanism_length_balanced/mechanism_corpus_public.tsv \
  --outdir /path/to/output/length_balanced_mapping
```

For the frozen manuscript mapping, `length_balanced_permutation_manifest.tsv` has SHA256:

```text
3cfd507b621a739980c405844c2e3331ac3a8cedf105c86c078e664dcbbee375
```

Running the full mechanism-control encoder/prediction workflow is a scientific computation and requires the external fine-analysis/E08 inputs, model snapshots, and frozen response arrays.
