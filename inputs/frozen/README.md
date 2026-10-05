# Exact recovered small inputs

These two files were extracted as UTF-8 bytes from the supplied historical `01_exact_sources.json`; their original byte lengths and SHA256 match. No values or paths were reconstructed.

`result2_effect_cache_config.yaml` matches all nine effect `DONE.json` seals. The recovered dataset catalog matches its historical export, but the effect seal does not record a dataset-catalog hash. Its use as the exact October catalog still needs local confirmation.

The YAML contains historical fallback keys. They are **not** permission to restore broad fallbacks: the corrected R2 builder uses the separate frozen nuisance policy and selection tables. The corrected H5AD path authority overrides old catalog paths. ComboSciPlex counts are `layers/counts`; OP3 is the accepted per-cell expm1 / log2-ratio definition. Do not run the historical result2 pipeline as a substitute for the corrected R2 runner.

The two files alone do not make R1/R2 fully runnable. Consult `docs/INPUT_RESTORATION.md` and `manifests/MISSING_EXTERNAL_INPUTS.tsv`.
