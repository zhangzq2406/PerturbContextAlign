# OP3 task-level alignment-prediction concordance

This v30 analysis reads the frozen 36-row OP3 encoder-by-task result table and computes, **within each task**, Spearman correlation across the same six encoders for four prespecified pairs: global RSA or excess NDCG@10 versus `-MAE` or DrugOrder. The six task-level correlations are descriptive and are not treated as independent biological replicates; the script performs no significance test or meta-analysis.

Example:

```bash
python run_op3_task_level_concordance_v1.py \
  --input /path/to/l2_l3_task_join.tsv \
  --outdir /path/to/output
```
