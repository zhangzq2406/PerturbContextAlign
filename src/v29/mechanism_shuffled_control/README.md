# v29 mechanism-shuffled control

This analysis is the matched negative control added for manuscript v29. It keeps the frozen 57-drug sci-Plex 3 cohort, six encoder snapshots, response/evaluation definitions, source/query splits, landmarks, alpha and gene panels, while replacing each drug's admitted mechanism text with another admitted mechanism text using 20 deterministic derangements.

The 20 mappings are fixed negative controls, not independent biological replicates. A mapping cannot assign a drug its own mechanism text or an exactly identical mechanism string. Identical prompt strings that recur across permutations are encoded once and mapped back to all assignment rows; this is only computational deduplication and does not modify assignments.

The script itself is path-agnostic. It expects a compatibility root passed as `--deep-root` that provides the following layout:

- `code/`, `configs/`, `metadata/`, `representations/`, `effects/`, `metrics/`
- `extensions/e08_prediction/`
- `experiments/e08_prediction_v1/`

The original project stored the fine-analysis assets and the E08 prediction assets in separate workspaces. Public users should construct a read-only overlay from their local copies of those external assets. Large expression/effect arrays, model weights, and historical prediction arrays are not redistributed in this repository.

Stages are run in order:

```bash
python mechanism_shuffled_control_v1.py preflight  --deep-root /path/to/overlay --run-root /new/run
python mechanism_shuffled_control_v1.py encode     --deep-root /path/to/overlay --run-root /new/run
python mechanism_shuffled_control_v1.py geometry   --deep-root /path/to/overlay --run-root /new/run
python mechanism_shuffled_control_v1.py predict    --deep-root /path/to/overlay --run-root /new/run
python mechanism_shuffled_control_v1.py score      --deep-root /path/to/overlay --run-root /new/run
python mechanism_shuffled_control_v1.py summarize  --deep-root /path/to/overlay --run-root /new/run
```

Predictions are sealed before query truth is scored. No encoder, permutation, split, gene panel, threshold or hyperparameter is selected using the shuffled-control outcomes.
