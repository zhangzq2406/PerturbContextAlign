# Result 4: manuscript dataset names and approved final export

2026-09-28 export amendment: the baseline geometry repair below is already accepted. This turn reuses all those values without recalculation. The only scientific-display text changes are `Kaggle` → `Kaggle cross-patient` and `sci-Plex` → `sci-Plex 3` in dataset labels. Existing data keys, task definitions and role labels do not change. Supplement a uses the approved grouped-bar version; supplement b/c values remain intact. Reference reuse level: exact reuse (labels/output paths only), with stand-alone ownership and shared legends restored for self-contained panel exports.

Final contract: Python/Matplotlib, original 24-inch working canvas (25 inches high main, 24 inches high supplement), Arial 22 pt, black text, no outer frames. Two assembled figures retain only a–d/a–c panel letters; seven stand-alone panels omit letters. Direct-rendered 600 dpi PNG and editable PDF/SVG; small previews retained. Shared donor/type or cell-line legend is included in stand-alone panels where necessary. No source resizing or upsampling. All new final PDFs must pass fresh glyph/collision/alignment QA and source equality checks; no known errors may pass into exports.

The following section records the inherited baseline-repair rationale. Its preview-only restrictions were superseded by the user's explicit final-export approval.

- Backend: Python/Matplotlib, existing reference-style layout. Arial 22 pt; 120 dpi preview only. No new main-panel titles or outer frames.
- Purpose: complete the applicable input-representation comparisons in panel a without changing accepted prediction or calibration evidence. No claim of universal superiority or a required representation–response gap.
- New evidence: `results/baseline_l2_task_metrics.tsv` (3 native representation baselines × 6 accepted Kaggle tasks); `results/baseline_l2_macro_summary.tsv` (3 macro summaries). Source definitions and metric permissions: `ANALYSIS_CONTRACT.md`.
- Mapping: panel a first two metric columns; rows Identity, TF-IDF, Morgan. Task dots are donor/type tasks, not independent experiments. The macro bar equally averages donors within type and then the two types. All finite values and all tasks are retained. No new CI or test.
- Geometry uses the original native input kernels, not cosine similarity of landmark-feature rows. Morgan uses its frozen fingerprint/dose kernel, not a new text representation.
- Missingness: Identity RSA is undefined because off-diagonal similarities are constant; label `NA`. Its tie-averaged excess NDCG is exactly zero and must be drawn numerically. Zero, Source mean, Source median and Same-drug source are prediction-only references; their input-geometry block is `Not applicable`, not missing or failed evaluation. No predicted-response geometry is substituted.
- All six accepted LLM geometry rows, all prediction values, and panels b–d remain unchanged. Only the geometry-column limits expand to include new task values. Approved supplementary preview is not edited or regenerated.
- Audit: exact source hashes and original-key preservation; independent numerical recalculation; all plotted numbers checked against artists; no clipping or text collisions; compare unchanged panel pixels; rerun alignment and PDF glyph checks.
- Export: one revised main preview plus analysis/plot scripts, necessary small source tables, provenance and QA. Final 600 dpi export awaits user approval.

## Panel evidence contract

| Panel | Evidence | Center and variability | Dependencies | This revision |
|---|---|---|---|---|
| a | Kaggle alignment and prediction; sci-Plex prediction | Frozen/new macro means plus all task dots; no CI | Repeated tasks/models are dependent | Add only applicable Kaggle input baselines |
| b | Accepted baseline-paired prediction differences | Box quartiles and all task dots | Same-task paired differences, not biological replicates | Unchanged |
| c | Accepted strict unseen-entity calibration | Frozen mean, dataset bootstrap interval, six dataset points | Existing statistical definition retained | Unchanged |
| d | Accepted single/joint holdout | Box quartiles, all task points, paired connectors | Paired entity folds/cell lines | Unchanged |
