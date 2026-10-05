# Reproducibility boundaries

The public release separates three levels of reproducibility.

## 1. Released-table verification

The code repository and companion Figshare archive support verification of deposited source tables, summary derivations, fixed mappings, and selected producer identities.

## 2. Lightweight software verification

`tests/` and the verification utilities exercise packaging invariants and selected numerical/software contracts without opening the full expression datasets or rerunning large encoders.

## 3. Full scientific rerun

A complete raw-to-result rerun additionally requires:

- third-party single-cell datasets in their documented representations;
- pretrained encoder weights/model snapshots from their original providers;
- large intermediate effect/prediction caches where the original workflow expects them;
- a compatible scientific software environment.

The repository does not claim that the lightweight dependency files reproduce every historical CPU/GPU runtime exactly. Where an exact historical process-level binding is unavailable, that limitation is left explicit rather than reconstructed retrospectively.

## Included lightweight identities

- R2 dataset/effect configuration: `inputs/frozen/result2/configs/`
- Fine-stage model snapshot records: `inputs/frozen/fine_models/`
- Phase A/B lightweight freeze manifest: `inputs/frozen/phase_ab/`
- Final length-balanced mechanism-control mapping metadata: `inputs/v30/mechanism_length_balanced/`

See `manifests/INPUT_AVAILABILITY.tsv` for the current inclusion boundary.
