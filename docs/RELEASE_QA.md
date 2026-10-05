# Current public-code release QA

This file summarizes checks applied to the cleaned public GitHub package. It is a current release note, not a development history.

## Scope

- Analysis/result-generation code only; no panel plotting tree.
- Scientific source code copied from the frozen public analysis package without modification, except for the packaging test that now references the version-neutral current manifest name.
- Historical QA folders, superseded changelogs, recovery logs, prior release snapshots, and old current-status documents are intentionally excluded.

## Current inventory

- Scientific `src/` tree: 106 files.
- Current manuscript panel map: 34 panels.
- Companion public source-data inventory: 122 source files, including 105 TSV files.
- Current public manifests are version-neutral and limited to source mapping, input availability, dataset naming, and table-producer status.

## Validation

The cleaned package is checked for:

- absence of `history/`, legacy QA archives, plotting/panel trees, and obsolete release documents;
- Python syntax/import-independent compilation of public `.py` files;
- repository unit tests;
- current 34-panel analysis-source mapping integrity;
- absence of known private workstation/user-data path prefixes in readable public files;
- archive integrity and regenerated SHA256 checksums.

Scientific values are not recomputed during this cleanup pass.
