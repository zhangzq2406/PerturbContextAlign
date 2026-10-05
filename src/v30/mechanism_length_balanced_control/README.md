# Length-balanced mechanism negative control

v30 retains the original 20 global mechanism derangements as a supplementary sensitivity analysis and adds 20 prespecified **length-balanced** derangements as the primary negative control. Every mapping preserves the same 57 mechanism descriptions as a multiset, prohibits self and exact-text assignments, and is generated without reading perturbation-response or prediction outcomes. The matching cost uses model-independent character count and whitespace-token count.

- `generate_length_balanced_derangements_v1.py` reproduces the **public assignment/length view** from the 57-row hash/length corpus `inputs/v30/mechanism_length_balanced/mechanism_corpus_public.tsv`.
- `mechanism_length_balanced_control_v1.py` runs the frozen encoder/alignment/prediction/scoring workflow once the historical fine-analysis and E08 inputs are made available under the expected layout.
- The frozen mapping and mapping audit used in the manuscript are distributed under `inputs/v30/mechanism_length_balanced/`.

The 20 mappings are deterministic negative controls, not independent biological replicates. Main figures display the full 20-control distribution/range rather than treating 2.5-97.5% quantiles from 20 fixed controls as stable tail estimates.

The public candidate does not redistribute the full third-party mechanism-description strings. The public mapping therefore omits `shuffled_mechanism_text`; entity assignments, text hashes, lengths, seeds and matching diagnostics are retained. To execute the scientific runner, rehydrate the mechanism texts from the original external annotation snapshot and reconstruct the full mapping; the executed full mapping SHA256 is `3cfd507b621a739980c405844c2e3331ac3a8cedf105c86c078e664dcbbee375`.
