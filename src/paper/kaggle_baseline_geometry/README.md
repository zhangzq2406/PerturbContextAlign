# OP3 baseline input-geometry producer

This directory restores the historical analysis that generated `baseline_l2_task_metrics.tsv` and `baseline_l2_macro_summary.tsv`. It contains analysis/QA code only; no plotting code is included.

Historical source hashes:

- producer: `42d07e6e1e21ad56c9c6b8fd0cac77b8ba2244694ddd0e220e683efd5f41d289`
- independent QA: `80e72a6bda0598198e95269dd7189de43862b4f0e6ff4eacf82b3552aca42ac9`
- accepted metric helper: `7719b593c53ed556750d60e6416fb3ef2b8876869e7ee924693e1ea1fabb2749`

The public producer and QA files differ from the historical snapshots only in path/bootstrap assignments: absolute local roots were replaced by environment variables and an optional output-root variable. Scientific scoring, cohort selection, aggregation and output-writing statements are unchanged. This recovery did not rerun the producer.

Required variables:

```bash
export PERTURBCONTEXTALIGN_FINE_ANALYSIS_ROOT=/path/to/analysis/benchmark_context_v2_20260914
export PERTURBCONTEXTALIGN_R3_ROOT=/path/to/R3_kaggle_alignment_prediction_20260925T014917531347Z
export PERTURBCONTEXTALIGN_BASELINE_GEOMETRY_OUT=/path/to/new/output
```

Create `results`, `geometry`, `provenance`, and `qa` under the output root before execution. The historical script writes both released baseline tables directly. The recovered independent-QA source reconstructs the task and macro values from frozen kernels. Historical execution-log binding was not recovered; output hashes and producer source coexist in the historical result package.
