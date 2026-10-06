# Current public-code release QA

This file summarizes checks applied to the current public GitHub package aligned to the final scientific manuscript (v36). It is a current release note, not a development history.

## Scope

- Analysis/result-generation code only; no panel plotting tree.
- Scientific analysis code is unchanged by the v36 manuscript presentation update.
- The v36 sync updates only public documentation and the panel-to-source analysis map for Fig. 2c, Fig. 2d and Fig. S5e.
- Historical QA folders, superseded changelogs, recovery logs and old release snapshots remain excluded from the public GitHub package.

## Current inventory

- Current manuscript panel map: 34 panels.
- Fig. 2c now contrasts four simple drug-discriminating references with six pretrained text encoders for OP3 MAE and DrugOrder.
- Fig. 2d shows four-endpoint encoder rank discordance across global RSA, excess NDCG@10, MAE and DrugOrder.
- Fig. S5e retains the six-task within-task alignment-prediction concordance analysis.
- Companion source data are associated with Figshare DOI `10.6084/m9.figshare.34071051`.

## Validation

The current package is checked for:

- Python syntax/import-independent compilation of public `.py` files;
- repository unit tests;
- 34-panel analysis-source mapping integrity;
- absence of known private workstation/user-data path prefixes in readable public files;
- regenerated SHA256 checksums.

No model was retrained, no embedding was regenerated, and no perturbation-response value was recomputed for the v36 sync.
