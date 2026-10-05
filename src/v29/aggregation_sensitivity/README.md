# v29 broad aggregation sensitivity

This lightweight analysis operates only on the frozen nine-dataset encoder summary table. It preserves the primary equal-dataset estimand and adds two sensitivity summaries:

1. nine leave-one-dataset-out summaries;
2. response-atom-weighted summaries.

Response-atom weighting changes the estimand by giving large screens more influence and is not a replacement for the prespecified equal-dataset primary analysis.

```bash
python run_aggregation_sensitivity_v1.py \
  --input /path/to/dataset_model_summary.tsv \
  --outdir /new/output_directory
```
