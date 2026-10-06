# PerturbContextAlign

**Pretrained text representations of single-cell perturbation context and their correspondence with measured perturbation responses**

PerturbContextAlign is an analysis framework for studying how pretrained text encoders represent **single-cell perturbation context** and how the resulting semantic relationships correspond to **experimentally measured perturbation responses**.

**Manuscript status:** Manuscript in preparation.

**Authors:** Zhiqiang Zhang, Jianbo Qiao, Zhihui Zhuang, Youzhuo Zhu, Hengwei Zhu, Chen Su, and Leyi Wei  
**Correspondence:** Leyi Wei — [weileyi@mpu.edu.mo](mailto:weileyi@mpu.edu.mo)

---

## Overview

Perturbation responses depend not only on perturbation identity, but also on biological and experimental context, including cell background, dose, treatment duration, and other recorded conditions. PerturbContextAlign asks two main questions:

1. **How do different pretrained text encoders represent perturbation context?**
2. **How do these context representations correspond to measured perturbation responses?**

The analyses separate three related but non-equivalent aspects of representation quality:

- **Context retention:** whether recorded perturbation and context information can be recovered or retrieved from the representation.
- **Text–response correspondence:** whether relationships in text-representation space agree with relationships in measured perturbation-response space.
- **Downstream utility:** whether representation differences translate into changes in fixed prediction tasks.

The repository is intentionally focused on **analysis and result generation**, rather than proposing a new perturbation-prediction model or a leaderboard of text encoders.

---

## Study scope

### Quantitative datasets

The manuscript analyzes nine quantitative single-cell perturbation datasets spanning genetic, small-molecule, and combination perturbations:

| Dataset | Perturbation setting |
|---|---|
| Norman 2019 | Genetic perturbation |
| Replogle 2022 K562-essential | Genetic perturbation |
| Replogle 2022 RPE1-essential | Genetic perturbation |
| Tian 2021 CRISPRa | Genetic perturbation |
| Tian 2021 CRISPRi | Genetic perturbation |
| sci-Plex 3 | Small-molecule perturbation |
| McFarland 2020 | Small-molecule perturbation |
| Open Problems–Single-Cell Perturbations (OP3; NeurIPS 2023) | Small-molecule perturbation |
| ComboSciPlex | Combination perturbation |

Internal dataset identifiers are kept stable in the code and may differ from the display names used in the manuscript.

### Pretrained text encoders

Six frozen pretrained text encoders are compared:

- BGE-M3
- Qwen3-Embedding-0.6B
- SapBERT
- BiomedBERT
- MedCPT Article Encoder
- MedCPT Query Encoder

The analyses also include simple reference representations such as TF–IDF, structured metadata, and random controls where applicable.

---

## Perturbation-context representations

Recorded perturbation metadata are converted into structured text descriptions. Four primary views are used:

- **P1 — Identity:** perturbation identity, perturbation mode, and combination structure.
- **P2 — Identity + biological context:** P1 plus species, cell type, cell line, tissue, disease, and related biological-background fields.
- **P3 — Identity + experimental context:** P1 plus exposure information such as dose and duration, together with recorded technical context used by the corresponding analysis.
- **P4 — Full context:** identity, biological context, and experimental context.

The exact field definitions, exclusions, and analysis-specific contracts are documented in the repository configuration and provenance files.

---

## Main analyses

The code in this repository supports the principal analysis layers reported in the manuscript:

### 1. Context retention

- Field recovery from frozen text representations
- Context and perturbation retrieval
- sci-Plex 3 dose and perturbation-identity readout
- Anonymous-name and context-view controls

### 2. Measured perturbation-response correspondence

- Broad response-effect construction
- Response-similarity geometry
- Representational similarity analysis (RSA)
- Local response-neighbour recovery using NDCG@10
- P1–P4 context ablations
- Fixed-perturbation, cross-context analyses
- Context-only and identity-conditioning controls

### 3. Response-relevant information controls

- Mechanism-description augmentation
- Prespecified shuffled-mechanism negative controls
- Length-balanced drug–mechanism mismatches
- Structured knowledge and native-prior comparisons
- Aggregation-sensitivity analyses

### 4. Downstream prediction analyses

- sci-Plex matched prediction analyses
- OP3 cross-donor prediction analyses
- MAE and perturbation-specific ordering metrics
- Task-level concordance between response alignment and downstream prediction objectives

These prediction analyses are used as **downstream tests of representation utility**. They are not intended to establish a new state-of-the-art perturbation-prediction method.

---

## Repository structure

```text
PerturbContextAlign/
├── src/
│   ├── r1/                  # Broad context-retention analyses
│   ├── r2/                  # Broad measured-response analyses
│   ├── fine/                # Fine-grained sci-Plex / OP3 analyses
│   ├── paper/               # Matched manuscript-result producers
│   ├── v29/                 # Aggregation and global-shuffle analyses
│   ├── v30/                 # Length-balanced mechanism controls and OP3 concordance
│   ├── context/             # Context-processing utilities
│   └── vendor/              # Small retained helper modules
├── configs/                 # Public analysis configurations
├── inputs/
│   ├── frozen/              # Lightweight recovered/frozen inputs
│   └── v30/                 # Inputs for final mechanism-control analyses
├── manifests/               # Current source, naming, and input-availability maps
├── provenance/              # Minimal verification anchors used by public checks
├── tests/                   # Lightweight unit and packaging tests
├── tools/                   # Release verification and configuration utilities
└── docs/                    # Runbook, contracts, methods notes, and release documentation
```

### What is intentionally not included

This analysis-only repository does **not** redistribute:

- third-party raw single-cell datasets or H5AD files;
- pretrained model weights or local model caches;
- large prediction arrays and internal development caches;
- private development logs or machine-specific paths;
- per-panel plotting and figure-assembly code.

Per-panel plotting/assembly code is maintained as a separate release artifact so that scientific analysis code and figure-rendering code remain clearly separated.

---

## Installation

For lightweight release verification:

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

For the broader scientific analysis stack:

```bash
pip install -r requirements-analysis.txt
```

`requirements-analysis.txt` lists the main analysis dependencies, but it is **not an exact lockfile for every historical CPU/GPU run**. Large pretrained models must be obtained from their original providers.

---

## Quick validation

Run the released unit and packaging checks:

```bash
python -m unittest discover -s tests -v
```

Verify a deposited Figshare source-data directory after the public source-data archive becomes available:

```bash
python tools/verify_release.py \
  --data-root /path/to/PerturbContextAlign_Figshare/sources
```

Verify released summary-table derivations:

```bash
python tools/verify_summary_derivations.py \
  --figshare-root /path/to/PerturbContextAlign_Figshare \
  --report /tmp/summary_derivations.json
```

These commands verify released tables, fixtures, and documented numerical identities. They do **not** reconstruct all raw-data processing or historical GPU runs.

---

## Selected reproducible analyses

### OP3 task-level alignment–prediction concordance

This analysis uses already frozen task-level results and does not refit a predictor:

```bash
python src/v30/op3_task_concordance/run_op3_task_level_concordance_v1.py \
  --input /path/to/l2_l3_task_join.tsv \
  --outdir /path/to/output/op3_concordance
```

### Length-balanced mechanism-control mappings

The prespecified length-balanced drug–mechanism mappings can be regenerated without response outcomes:

```bash
python src/v30/mechanism_length_balanced_control/generate_length_balanced_derangements_v1.py \
  --mechanism-corpus inputs/v30/mechanism_length_balanced/mechanism_corpus_public.tsv \
  --outdir /path/to/output/length_balanced_mapping
```

For the frozen manuscript mapping, the generated `length_balanced_permutation_manifest.tsv` should have SHA256:

```text
3cfd507b621a739980c405844c2e3331ac3a8cedf105c86c078e664dcbbee375
```

For broader scientific reruns, read [`docs/RUNBOOK.md`](docs/RUNBOOK.md) and the relevant method/provenance records before executing production scripts.

---

## Data availability

Raw datasets are **not redistributed** in this repository. They should be obtained from their original public repositories or study resources, as described in the manuscript and accompanying documentation.

Public figure-linked source data, compact analysis outputs, data dictionaries, panel-to-source mappings, checksums, and public-safe provenance records are available from the accompanying **Figshare** record.

**Figshare DOI:** [10.6084/m9.figshare.34071051](https://doi.org/10.6084/m9.figshare.34071051).

---

## Reproducibility and provenance

This release distinguishes between:

1. **Released-source verification** — checking that deposited tables and summaries have the expected identities and numerical values.
2. **Software-fixture verification** — checking selected implementation components on lightweight or synthetic inputs.
3. **Full scientific reproduction** — regenerating all results from raw third-party data, exact pretrained-model snapshots, and complete historical runtime environments.

The repository provides extensive provenance records for the first two levels and the available inputs for scientific reruns. Some historical execution details, including a complete original CPU/GPU environment lock and certain process-level execution bindings, are not reconstructed and are documented explicitly rather than inferred.

See:

- [`docs/RUNBOOK.md`](docs/RUNBOOK.md)
- [`docs/METHODS_AND_SCOPE.md`](docs/METHODS_AND_SCOPE.md)
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)
- [`docs/RELEASE_QA.md`](docs/RELEASE_QA.md)

---

## Manuscript

**Pretrained text representations of single-cell perturbation context and their correspondence with measured perturbation responses**

Zhiqiang Zhang, Jianbo Qiao, Zhihui Zhuang, Youzhuo Zhu, Hengwei Zhu, Chen Su, and Leyi Wei.

**Status:** Manuscript in preparation.

A preprint/publication link will be added when available.

---

## Citation

Until a preprint or peer-reviewed version is available, please cite the manuscript as:

> Zhang Z, Qiao J, Zhuang Z, Zhu Y, Zhu H, Su C, Wei L. *Pretrained text representations of single-cell perturbation context and their correspondence with measured perturbation responses.* Manuscript in preparation, 2026.

BibTeX:

```bibtex
@unpublished{zhang2026perturbcontextalign,
  author = {Zhang, Zhiqiang and Qiao, Jianbo and Zhuang, Zhihui and Zhu, Youzhuo and Zhu, Hengwei and Su, Chen and Wei, Leyi},
  title  = {Pretrained text representations of single-cell perturbation context and their correspondence with measured perturbation responses},
  year   = {2026},
  note   = {Manuscript in preparation}
}
```

Citation metadata will be updated after a preprint or journal DOI becomes available.

---

## License

The software in this repository is released under the **MIT License**. See [`LICENSE`](LICENSE).

Third-party datasets, pretrained model weights, annotations, and external resources remain subject to the licenses and terms of their original providers.

---

## Contact

For questions about the study or repository:

**Leyi Wei**  
Email: [weileyi@mpu.edu.mo](mailto:weileyi@mpu.edu.mo)

---

## Repository

https://github.com/zhangzq2406/PerturbContextAlign
