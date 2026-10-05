from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default=".",
    )
    ap.add_argument(
        "--device",
        default="cuda:0",
    )
    ap.add_argument(
        "--preflight-only",
        action="store_true",
    )
    return ap.parse_args()


ARGS = parse_args()

ROOT = Path(ARGS.root).expanduser().resolve()

NEW = (
    ROOT
    / "0_ProjectCodeAndLogs"
    / "PerturbContextAlign_R2_corrected_v1_20261002"
)

DEVICE = ARGS.device
MAX_LENGTH = 512
BATCH_SIZE = 64
BOOTSTRAP_REPEATS = 2000
BOOTSTRAP_SEED = 20260713

DATASETS = [
    "norman_2019",
    "replogle_k562_essential",
    "replogle_rpe1",
    "tian_activation",
    "tian_inhibition",
    "srivatsan_sciplex3",
    "kaggle_cross_patient",
    "combo_sciplex",
]

MODEL_SPECS = [
    {
        "model_key": "bge_m3",
        "display": "BGE-M3",
        "group": "general_purpose",
        "model_id": "BAAI/bge-m3",
        "backend": "sentence_transformers",
        "pooling": "sentence",
        "trust_remote_code": False,
        "dim": 1024,
    },
    {
        "model_key": "qwen3_0_6b",
        "display": "Qwen3-Embedding-0.6B",
        "group": "general_purpose",
        "model_id": "Qwen/Qwen3-Embedding-0.6B",
        "backend": "sentence_transformers",
        "pooling": "sentence",
        "trust_remote_code": True,
        "dim": 1024,
    },
    {
        "model_key": "sapbert",
        "display": "SapBERT",
        "group": "biomedical",
        "model_id": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        "backend": "transformers",
        "pooling": "mean",
        "trust_remote_code": False,
        "dim": 768,
    },
    {
        "model_key": "biomedbert",
        "display": "BiomedBERT",
        "group": "biomedical",
        "model_id": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        "backend": "transformers",
        "pooling": "mean",
        "trust_remote_code": False,
        "dim": 768,
    },
    {
        "model_key": "medcpt_article",
        "display": "MedCPT Article",
        "group": "biomedical",
        "model_id": "ncbi/MedCPT-Article-Encoder",
        "backend": "transformers",
        "pooling": "cls",
        "trust_remote_code": False,
        "dim": 768,
    },
    {
        "model_key": "medcpt_query",
        "display": "MedCPT Query",
        "group": "biomedical",
        "model_id": "ncbi/MedCPT-Query-Encoder",
        "backend": "transformers",
        "pooling": "cls",
        "trust_remote_code": False,
        "dim": 768,
    },
]

PREFLIGHT = (
    NEW
    / "08_knowledge_native"
    / "02_contract_preflight_v3_1"
)

P5A_AUTH = (
    PREFLIGHT
    / "R2_P5A_RAW_SUFFIX_AUTHORITY_PREFLIGHT_v3_1.tsv"
)

P5A_COMPONENT_FEAS = (
    PREFLIGHT
    / "R2_P5A_COMPONENT_KNOWLEDGE_FEASIBILITY_v3_1.tsv"
)

ENTITY_KNOWLEDGE_CACHE = (
    ROOT
    / "result1"
    / "outputs_v2_1_2"
    / "knowledge"
    / "entity_knowledge_cache.tsv"
)

COMPONENT_P4 = (
    NEW
    / "05_response_aligned_text"
    / "R2_RESPONSE_ALIGNED_COMPONENT_PROMPTS_v1.tsv.gz"
)

ATOM_AUTHORITY = (
    NEW
    / "02_input_authority"
    / "R2_RESPONSE_ATOM_AUTHORITY_v1.tsv"
)

P4_EMB_ROOT = (
    NEW
    / "05_response_aligned_text"
    / "02_embeddings_v1"
)

P4_INDEX = (
    P4_EMB_ROOT
    / "R2_RESPONSE_ATOM_EMBEDDING_INDEX_v1.tsv"
)

RESP_ROOT = (
    NEW
    / "04_response_geometry"
    / "02_similarity_v1"
)

OUT = (
    NEW
    / "08_knowledge_native"
    / "03_corrected_p5a_v1_2"
)

AUTH_OUT = OUT / "00_authority"
EMB_OUT = OUT / "01_embeddings"
METRIC_OUT = OUT / "02_metrics"

for d in [AUTH_OUT, EMB_OUT, METRIC_OUT]:
    d.mkdir(parents=True, exist_ok=True)

P5A_COMPONENT_OUT = (
    AUTH_OUT
    / "R2_CORRECTED_P5A_COMPONENT_PROMPTS_v1_2.tsv.gz"
)

P5A_ATOM_INDEX = (
    AUTH_OUT
    / "R2_CORRECTED_P5A_ATOM_INDEX_v1_2.tsv"
)

MAPPING_AUDIT = (
    AUTH_OUT
    / "R2_CORRECTED_P5A_COMPONENT_MAPPING_AUDIT_v1_2.tsv"
)

ELIGIBLE_ATOM_AUDIT = (
    AUTH_OUT
    / "R2_CORRECTED_P5A_ELIGIBLE_ATOM_COUNT_AUDIT_v1_2.tsv"
)

BY_DATASET = (
    METRIC_OUT
    / "R2_CORRECTED_P5A_ALIGNMENT_BY_DATASET_v1_2.tsv"
)

SUMMARY = (
    METRIC_OUT
    / "R2_CORRECTED_P5A_DELTA_SUMMARY_v1_2.tsv"
)

AUDIT = (
    OUT
    / "R2_CORRECTED_P5A_AUDIT_v1_2.txt"
)

MANIFEST = (
    OUT
    / "R2_CORRECTED_P5A_MANIFEST_v1_2.json"
)

KNOWLEDGE_PREFIX = "External structured knowledge."

BLOCK_SPLIT_RX = re.compile(
    r"\s+\|\s+(?=canonical name:)",
    re.I,
)

CANONICAL_RX = re.compile(
    r"^\s*canonical name:\s*([^;]+)",
    re.I,
)


def clean(x: Any) -> str:
    if x is None or pd.isna(x):
        return ""
    return str(x).strip()


def norm(x: Any) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "",
        clean(x).lower(),
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def stable_seed(*parts: Any) -> int:
    raw = "|".join(map(str, parts)).encode()
    return int.from_bytes(
        hashlib.sha256(raw).digest()[:4],
        "little",
    )


def write_json(obj, path: Path):
    path.write_text(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def l2(x):
    a = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(a, axis=1, keepdims=True)

    if np.any(~np.isfinite(n)) or np.any(n <= 1e-12):
        raise RuntimeError("Invalid embedding norm.")

    return a / n


def mean_pool(hidden, mask):
    m = mask.unsqueeze(-1).to(hidden.dtype)
    return (
        (hidden * m).sum(dim=1)
        / m.sum(dim=1).clamp(min=1e-9)
    )


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class Encoder:
    def __init__(self, spec):
        self.spec = spec
        self.model = None
        self.tokenizer = None

        if spec["backend"] == "sentence_transformers":
            from sentence_transformers import SentenceTransformer

            kwargs = {
                "device": DEVICE,
                "trust_remote_code": spec["trust_remote_code"],
            }

            try:
                self.model = SentenceTransformer(
                    spec["model_id"],
                    local_files_only=True,
                    **kwargs,
                )
            except TypeError:
                self.model = SentenceTransformer(
                    spec["model_id"],
                    **kwargs,
                )

            if hasattr(self.model, "max_seq_length"):
                self.model.max_seq_length = MAX_LENGTH

        else:
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                spec["model_id"],
                local_files_only=True,
                trust_remote_code=spec["trust_remote_code"],
            )

            self.model = AutoModel.from_pretrained(
                spec["model_id"],
                local_files_only=True,
                trust_remote_code=spec["trust_remote_code"],
            ).to(DEVICE)

            self.model.eval()

    def encode(self, texts):
        if self.spec["backend"] == "sentence_transformers":
            return np.asarray(
                self.model.encode(
                    texts,
                    batch_size=BATCH_SIZE,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                ),
                dtype=np.float32,
            )

        chunks = []

        for start in range(0, len(texts), BATCH_SIZE):
            local = texts[start:start + BATCH_SIZE]

            batch = self.tokenizer(
                local,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )

            batch = {
                k: v.to(DEVICE)
                for k, v in batch.items()
            }

            with torch.inference_mode():
                out = self.model(**batch)

                if self.spec["pooling"] == "mean":
                    z = mean_pool(
                        out.last_hidden_state,
                        batch["attention_mask"],
                    )
                else:
                    z = out.last_hidden_state[:, 0]

            chunks.append(
                z.float().cpu().numpy()
            )

        return l2(
            np.concatenate(chunks, axis=0)
        )

    def close(self):
        try:
            del self.model
        except Exception:
            pass

        try:
            del self.tokenizer
        except Exception:
            pass

        cleanup()


def load_response(path: Path):
    with np.load(path, allow_pickle=True) as z:
        key = "S" if "S" in z.files else "X"
        S = np.asarray(z[key], dtype=np.float32)
        id_key = (
            "response_atom_ids"
            if "response_atom_ids" in z.files
            else "condition_ids"
        )
        ids = np.asarray(z[id_key]).astype(str)
    return S, ids


def upper(S):
    A = np.asarray(S, dtype=np.float64)
    return A[np.triu_indices(A.shape[0], 1)]


def spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]

    if len(x) < 3:
        return np.nan

    if np.ptp(x) <= 1e-12 or np.ptp(y) <= 1e-12:
        return np.nan

    rx = rankdata(x)
    ry = rankdata(y)

    rx -= rx.mean()
    ry -= ry.mean()

    den = math.sqrt(
        float(np.dot(rx, rx))
        * float(np.dot(ry, ry))
    )

    return (
        float(np.dot(rx, ry) / den)
        if den > 0
        else np.nan
    )


def bootstrap_ci(values, seed):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)

    idx = rng.integers(
        0,
        len(x),
        size=(BOOTSTRAP_REPEATS, len(x)),
    )

    means = x[idx].mean(axis=1)

    return (
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


# -----------------------------------------------------------------------------
# Authority
# -----------------------------------------------------------------------------

for p in [
    P5A_AUTH,
    P5A_COMPONENT_FEAS,
    ENTITY_KNOWLEDGE_CACHE,
    COMPONENT_P4,
    ATOM_AUTHORITY,
    P4_INDEX,
]:
    if not p.is_file():
        raise FileNotFoundError(p)


auth = pd.read_csv(
    P5A_AUTH,
    sep="\t",
    low_memory=False,
)

feas = pd.read_csv(
    P5A_COMPONENT_FEAS,
    sep="\t",
    low_memory=False,
)

entity_cache = pd.read_csv(
    ENTITY_KNOWLEDGE_CACHE,
    sep="\t",
    low_memory=False,
)

required_entity_cache_cols = {
    "entity_id",
    "entity_name",
    "canonical_name",
    "query_name",
    "resolved_name",
    "match_type",
    "status",
}

missing_entity_cache_cols = (
    required_entity_cache_cols
    - set(entity_cache.columns)
)

if missing_entity_cache_cols:
    raise KeyError(
        "Entity knowledge cache missing required columns: "
        f"{sorted(missing_entity_cache_cols)}"
    )

entity_cache["entity_id"] = (
    entity_cache["entity_id"].astype(str)
)

if entity_cache["entity_id"].duplicated().any():
    raise RuntimeError(
        "Entity knowledge cache entity_id is not unique."
    )

entity_by_id = (
    entity_cache
    .set_index("entity_id")
)

components = pd.read_csv(
    COMPONENT_P4,
    sep="\t",
    compression="gzip",
    low_memory=False,
)

atoms = pd.read_csv(
    ATOM_AUTHORITY,
    sep="\t",
    low_memory=False,
)

p4_index = pd.read_csv(
    P4_INDEX,
    sep="\t",
    low_memory=False,
)


for df in [auth, feas, atoms]:
    for c in [
        "dataset_id",
        "condition_id",
        "response_atom_id",
    ]:
        if c in df.columns:
            df[c] = df[c].astype(str)


components["dataset_id"] = components["dataset_id"].astype(str)
components["response_atom_id"] = components["response_atom_id"].astype(str)
components["component_id"] = components["component_id"].astype(str)

atoms["response_atom_id"] = atoms["response_atom_id"].astype(str)
atoms["condition_id"] = atoms["condition_id"].astype(str)
atoms["dataset_id"] = atoms["dataset_id"].astype(str)

p4_index["response_atom_id"] = p4_index["response_atom_id"].astype(str)


# ------------------------------------------------------------------
# Frozen corrected eligibility namespace
# ------------------------------------------------------------------
eligible_response_atom_ids = set()

eligible_response_atom_ids_by_dataset = {}

for ds in DATASETS:
    index_path = (
        RESP_ROOT
        / ds
        / "response_similarity_index.tsv"
    )

    if not index_path.is_file():
        raise FileNotFoundError(
            index_path
        )

    idx = pd.read_csv(
        index_path,
        sep="\t",
        low_memory=False,
    )

    if "response_atom_id" not in idx.columns:
        raise KeyError(
            f"{index_path}: response_atom_id missing"
        )

    ids = set(
        idx["response_atom_id"].astype(str)
    )

    eligible_response_atom_ids_by_dataset[
        ds
    ] = ids

    eligible_response_atom_ids.update(
        ids
    )


authority_atom_ids = set(
    atoms["response_atom_id"].astype(str)
)

missing_eligible_authority_ids = (
    eligible_response_atom_ids
    - authority_atom_ids
)

if missing_eligible_authority_ids:
    raise RuntimeError(
        "Frozen response-similarity atom IDs absent from response-atom authority: "
        f"n={len(missing_eligible_authority_ids)}"
    )


# IMPORTANT:
# P5A is evaluated only on response atoms already admitted to the frozen
# corrected response-similarity namespace. A knowledge-covered condition can
# have additional response atoms that failed response-side eligibility; those
# atoms must not re-enter P5A simply because the condition has knowledge.
atoms_all_authority = atoms.copy()

atoms = atoms.loc[
    atoms["response_atom_id"].isin(
        eligible_response_atom_ids
    )
].copy()


covered = auth.loc[
    auth["dataset_id"].isin(DATASETS)
    & auth["eligible_knowledge_covered"].astype(bool)
].copy()


cond_suffix = {
    (
        str(r["dataset_id"]),
        str(r["condition_id"]),
    ): str(r["knowledge_suffix"])
    for _, r in covered.iterrows()
}

required_auth_cols = {
    "dataset_id",
    "condition_id",
    "knowledge_suffix",
    "knowledge_entity_ids",
}

missing_auth_cols = (
    required_auth_cols
    - set(covered.columns)
)

if missing_auth_cols:
    raise KeyError(
        "P5A preflight authority missing columns: "
        f"{sorted(missing_auth_cols)}"
    )

covered_condition_keys = set(
    zip(
        covered["dataset_id"].astype(str),
        covered["condition_id"].astype(str),
    )
)

expected_p5a_atom_ids = set(
    atoms.loc[
        [
            (
                str(ds),
                str(cid),
            )
            in covered_condition_keys
            for ds, cid in zip(
                atoms["dataset_id"],
                atoms["condition_id"],
            )
        ],
        "response_atom_id",
    ].astype(str)
)


cond_entity_ids = {}

for _, r in covered.iterrows():
    key = (
        str(r["dataset_id"]),
        str(r["condition_id"]),
    )

    ids = [
        x.strip()
        for x in str(
            r["knowledge_entity_ids"]
        ).split("|")
        if x.strip()
        and x.strip().lower()
        not in {
            "nan",
            "none",
            "null",
        }
    ]

    if not ids:
        raise RuntimeError(
            f"{key}: empty knowledge_entity_ids"
        )

    if len(ids) != len(set(ids)):
        raise RuntimeError(
            f"{key}: duplicate knowledge_entity_ids"
        )

    cond_entity_ids[key] = ids


atom_cond = (
    atoms[
        [
            "response_atom_id",
            "dataset_id",
            "condition_id",
        ]
    ]
    .set_index("response_atom_id")
)


components = components.loc[
    components["response_atom_id"].isin(
        atom_cond.index
    )
].copy()

components["condition_id"] = [
    str(
        atom_cond.loc[
            rid,
            "condition_id",
        ]
    )
    for rid in components["response_atom_id"]
]


components = components.loc[
    [
        (
            str(ds),
            str(cid),
        )
        in cond_suffix
        for ds, cid in zip(
            components["dataset_id"],
            components["condition_id"],
        )
    ]
].copy()


def parse_blocks(suffix: str):
    payload = str(suffix).strip()

    if payload.startswith(KNOWLEDGE_PREFIX):
        payload = payload[
            len(KNOWLEDGE_PREFIX):
        ].strip()

    raw_blocks = BLOCK_SPLIT_RX.split(payload)

    result = []

    for block in raw_blocks:
        block = block.strip()

        m = CANONICAL_RX.search(block)

        if not m:
            raise RuntimeError(
                f"Knowledge block lacks canonical name: {block[:300]}"
            )

        name = m.group(1).strip()

        result.append(
            (
                name,
                block,
            )
        )

    return result


mapping_rows = []
component_suffix = {}


def component_match_score(
    entity_name: str,
    component_id: str,
    prompt: str,
) -> tuple[int, str]:
    """
    Match the historical original entity_name to a corrected component.

    entity_name is authoritative because it is reached through the exact
    historical entity_id. canonical_name is explicitly NOT used for linkage:
    it may be an alias-resolved external-database symbol (e.g. C19orf26 -> CBARP).
    """
    en = norm(entity_name)

    if not en:
        return 0, ""

    comp_tokens = [
        norm(x)
        for x in str(
            component_id
        ).split(":")
        if norm(x)
    ]

    identity_prefix = str(prompt).split(
        " Biological context:",
        1,
    )[0]

    if en in comp_tokens:
        return 100, "entity_name_exact_component_token"

    if en == norm(component_id):
        return 90, "entity_name_exact_component_id"

    if en in norm(component_id):
        return 70, "entity_name_substring_component_id"

    if en in norm(identity_prefix):
        return 50, "entity_name_in_identity_prefix"

    if en in norm(prompt):
        return 30, "entity_name_in_full_component_prompt"

    return 0, ""


for (
    ds,
    cid,
), sub in components.groupby(
    [
        "dataset_id",
        "condition_id",
    ],
    observed=True,
):
    ds = str(ds)
    cid = str(cid)

    key = (
        ds,
        cid,
    )

    comp_ids = sorted(
        sub["component_id"].astype(str).unique()
    )

    rep = (
        sub.sort_values(
            [
                "component_id",
                "response_atom_id",
            ]
        )
        .drop_duplicates(
            "component_id"
        )
        .set_index(
            "component_id"
        )
    )

    blocks = parse_blocks(
        cond_suffix[
            key
        ]
    )

    entity_ids = cond_entity_ids[
        key
    ]

    if (
        len(blocks)
        != len(entity_ids)
        or len(entity_ids)
        != len(comp_ids)
    ):
        raise RuntimeError(
            f"{ds}/{cid}: "
            f"block_n={len(blocks)} "
            f"entity_id_n={len(entity_ids)} "
            f"component_n={len(comp_ids)}"
        )

    used = set()

    # v3.1 established that knowledge_entity_ids and P5A blocks have the
    # same ordered canonical-name sequence. Re-check that contract here.
    for block_index, (
        entity_id,
        (
            suffix_canonical,
            block,
        ),
    ) in enumerate(
        zip(
            entity_ids,
            blocks,
        )
    ):
        if entity_id not in entity_by_id.index:
            raise RuntimeError(
                f"{ds}/{cid}: entity_id absent from entity cache: {entity_id}"
            )

        erow = entity_by_id.loc[
            entity_id
        ]

        entity_name = clean(
            erow["entity_name"]
        )

        cache_canonical = clean(
            erow["canonical_name"]
        )

        query_name = clean(
            erow["query_name"]
        )

        resolved_name = clean(
            erow["resolved_name"]
        )

        match_type = clean(
            erow["match_type"]
        )

        entity_status = clean(
            erow["status"]
        )

        if entity_status != "ok":
            raise RuntimeError(
                f"{ds}/{cid}: knowledge entity status is not ok: "
                f"{entity_id} status={entity_status}"
            )

        if (
            norm(cache_canonical)
            != norm(suffix_canonical)
        ):
            raise RuntimeError(
                f"{ds}/{cid}: ordered entity-id/block canonical mismatch: "
                f"entity_id={entity_id} "
                f"cache={cache_canonical!r} "
                f"suffix={suffix_canonical!r}"
            )

        candidates = []

        for comp_id in comp_ids:
            if comp_id in used:
                continue

            prompt = str(
                rep.loc[
                    comp_id,
                    "response_aligned_prompt",
                ]
            )

            score, method = component_match_score(
                entity_name,
                comp_id,
                prompt,
            )

            if score > 0:
                candidates.append(
                    (
                        score,
                        comp_id,
                        method,
                    )
                )

        # One entity ID + one component is a fully determined linkage even
        # when naming punctuation prevents a textual match.
        if (
            len(comp_ids) == 1
            and len(entity_ids) == 1
            and not candidates
        ):
            only = comp_ids[0]

            candidates = [
                (
                    1,
                    only,
                    "single_entity_single_component_by_exact_entity_id_order",
                )
            ]

        if not candidates:
            raise RuntimeError(
                f"{ds}/{cid}: cannot map exact entity_id={entity_id!r} "
                f"entity_name={entity_name!r} "
                f"canonical_name={cache_canonical!r} "
                f"to corrected components={comp_ids}"
            )

        candidates.sort(
            key=lambda x: (
                -x[0],
                x[1],
            )
        )

        best_score = candidates[0][0]

        best = [
            x
            for x in candidates
            if x[0] == best_score
        ]

        if len(best) != 1:
            raise RuntimeError(
                f"{ds}/{cid}: ambiguous entity-ID-authoritative mapping "
                f"entity_id={entity_id!r} "
                f"entity_name={entity_name!r} "
                f"candidates={best}"
            )

        (
            _,
            comp_id,
            match_method,
        ) = best[0]

        used.add(
            comp_id
        )

        full_suffix = (
            KNOWLEDGE_PREFIX
            + " "
            + block
        )

        component_suffix[
            (
                ds,
                cid,
                comp_id,
            )
        ] = full_suffix

        mapping_rows.append({
            "dataset_id": ds,
            "condition_id": cid,
            "block_index": int(
                block_index
            ),
            "knowledge_entity_id": entity_id,
            "historical_entity_name": entity_name,
            "historical_query_name": query_name,
            "historical_resolved_name": resolved_name,
            "historical_canonical_name": cache_canonical,
            "historical_match_type": match_type,
            "suffix_canonical_name": suffix_canonical,
            "corrected_component_id": comp_id,
            "mapping_method": match_method,
            "mapping_score": int(
                best_score
            ),
            "knowledge_block": full_suffix,
        })

    if used != set(comp_ids):
        raise RuntimeError(
            f"{ds}/{cid}: not all corrected components were assigned exactly once; "
            f"used={sorted(used)} components={sorted(comp_ids)}"
        )


mapping_audit = pd.DataFrame(mapping_rows)

mapping_audit.to_csv(
    MAPPING_AUDIT,
    sep="\t",
    index=False,
)


components["knowledge_block"] = [
    component_suffix[
        (
            str(ds),
            str(cid),
            str(comp),
        )
    ]
    for ds, cid, comp in zip(
        components["dataset_id"],
        components["condition_id"],
        components["component_id"],
    )
]

components["p5a_text"] = (
    components["response_aligned_prompt"].astype(str).str.rstrip()
    + " "
    + components["knowledge_block"].astype(str)
)


components[
    [
        "dataset_id",
        "condition_id",
        "response_atom_id",
        "component_id",
        "response_aligned_prompt",
        "knowledge_block",
        "p5a_text",
    ]
].to_csv(
    P5A_COMPONENT_OUT,
    sep="\t",
    index=False,
    compression="gzip",
)


atom_index = (
    components[
        [
            "dataset_id",
            "condition_id",
            "response_atom_id",
        ]
    ]
    .drop_duplicates()
    .sort_values(
        [
            "dataset_id",
            "condition_id",
            "response_atom_id",
        ]
    )
    .reset_index(drop=True)
)

atom_index.insert(
    0,
    "p5a_row_index",
    np.arange(
        len(atom_index),
        dtype=np.int64,
    ),
)

atom_index.to_csv(
    P5A_ATOM_INDEX,
    sep="\t",
    index=False,
)


# Expect exactly the preflight-covered frozen eligible response atoms.
expected_n = int(
    auth.loc[
        auth["dataset_id"].isin(DATASETS)
        & auth["eligible_knowledge_covered"].astype(bool),
        "covered_response_atom_n",
    ].sum()
)

actual_p5a_atom_ids = set(
    atom_index["response_atom_id"].astype(str)
)

missing_expected_atoms = (
    expected_p5a_atom_ids
    - actual_p5a_atom_ids
)

unexpected_atoms = (
    actual_p5a_atom_ids
    - expected_p5a_atom_ids
)

if missing_expected_atoms or unexpected_atoms:
    raise RuntimeError(
        "P5A exact eligible-atom namespace mismatch: "
        f"missing_expected_n={len(missing_expected_atoms)} "
        f"unexpected_n={len(unexpected_atoms)}"
    )

if len(atom_index) != expected_n:
    raise RuntimeError(
        f"P5A response atom count mismatch after exact eligibility filtering: "
        f"{len(atom_index)} != {expected_n}"
    )


# Per-dataset audit: distinguishes all knowledge-covered authority atoms from
# the final frozen eligible response atoms.
eligible_audit_rows = []

for ds in DATASETS:
    ds_covered_conditions = set(
        covered.loc[
            covered["dataset_id"].astype(str).eq(ds),
            "condition_id",
        ].astype(str)
    )

    all_authority_atoms_under_covered_conditions = set(
        atoms_all_authority.loc[
            atoms_all_authority["dataset_id"].astype(str).eq(ds)
            & atoms_all_authority["condition_id"].astype(str).isin(
                ds_covered_conditions
            ),
            "response_atom_id",
        ].astype(str)
    )

    eligible_atoms_under_covered_conditions = (
        all_authority_atoms_under_covered_conditions
        & eligible_response_atom_ids_by_dataset[
            ds
        ]
    )

    actual_atoms = set(
        atom_index.loc[
            atom_index["dataset_id"].astype(str).eq(ds),
            "response_atom_id",
        ].astype(str)
    )

    expected_from_preflight = int(
        auth.loc[
            auth["dataset_id"].astype(str).eq(ds)
            & auth["eligible_knowledge_covered"].astype(bool),
            "covered_response_atom_n",
        ].sum()
    )

    eligible_audit_rows.append({
        "dataset_id": ds,
        "knowledge_covered_condition_n": len(
            ds_covered_conditions
        ),
        "all_authority_atom_n_under_covered_conditions": len(
            all_authority_atoms_under_covered_conditions
        ),
        "frozen_eligible_atom_n_under_covered_conditions": len(
            eligible_atoms_under_covered_conditions
        ),
        "excluded_noneligible_atom_n": len(
            all_authority_atoms_under_covered_conditions
            - eligible_atoms_under_covered_conditions
        ),
        "preflight_expected_atom_n": expected_from_preflight,
        "final_p5a_atom_n": len(
            actual_atoms
        ),
        "exact_match": bool(
            actual_atoms
            == eligible_atoms_under_covered_conditions
            and len(actual_atoms)
            == expected_from_preflight
        ),
    })


eligible_atom_audit = pd.DataFrame(
    eligible_audit_rows
)

eligible_atom_audit.to_csv(
    ELIGIBLE_ATOM_AUDIT,
    sep="\t",
    index=False,
)

if not eligible_atom_audit["exact_match"].all():
    raise RuntimeError(
        "Per-dataset P5A eligible-atom audit failed."
    )


unique_texts = sorted(
    components["p5a_text"].astype(str).unique()
)

text_pos = {
    t: i
    for i, t in enumerate(unique_texts)
}


if ARGS.preflight_only:
    print(
        f"P5A_COMPONENT_MAPPING=PASS mapped_component_n={len(component_suffix)}"
    )
    print(
        f"P5A_RESPONSE_ATOM_N={len(atom_index)} unique_text_n={len(unique_texts)}"
    )

    print(
        "P5A_ELIGIBLE_ATOM_FILTER=PASS "
        f"excluded_noneligible_atom_n={int(eligible_atom_audit['excluded_noneligible_atom_n'].sum())}"
    )

    for spec in MODEL_SPECS:
        enc = Encoder(spec)

        try:
            z = enc.encode(
                unique_texts[
                    :min(
                        4,
                        len(unique_texts),
                    )
                ]
            )

            if z.shape[1] != spec["dim"]:
                raise RuntimeError(
                    f"{spec['model_key']}: dimension mismatch"
                )

            print(
                f"P5A_ENCODER_PREFLIGHT=PASS model={spec['model_key']} shape={z.shape}"
            )

        finally:
            enc.close()

    print("R2_CORRECTED_P5A_PREFLIGHT=PASS")
    raise SystemExit(0)


# -----------------------------------------------------------------------------
# Encode and aggregate
# -----------------------------------------------------------------------------

p5a_pos = {
    rid: int(row)
    for rid, row in zip(
        atom_index["response_atom_id"].astype(str),
        atom_index["p5a_row_index"].astype(int),
    )
}

model_atom_paths = {}


for spec in MODEL_SPECS:
    key = spec["model_key"]

    print(
        f"[encode] {key} unique_text_n={len(unique_texts)}",
        flush=True,
    )

    model_dir = EMB_OUT / key
    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    atom_path = (
        model_dir
        / "response_atom_embeddings.npy"
    )

    done_path = (
        model_dir
        / "DONE.json"
    )

    if atom_path.is_file() and done_path.is_file():
        try:
            old = json.loads(
                done_path.read_text(
                    encoding="utf-8"
                )
            )

            arr = np.load(
                atom_path,
                mmap_mode="r",
            )

            if (
                old.get("status") == "PASS"
                and tuple(arr.shape)
                == (
                    len(atom_index),
                    spec["dim"],
                )
            ):
                model_atom_paths[
                    key
                ] = atom_path

                print(
                    f"[resume] {key}",
                    flush=True,
                )

                continue
        except Exception:
            pass

    enc = Encoder(spec)

    try:
        text_emb = enc.encode(
            unique_texts
        )
    finally:
        enc.close()

    sums = np.zeros(
        (
            len(atom_index),
            spec["dim"],
        ),
        dtype=np.float32,
    )

    counts = np.zeros(
        len(atom_index),
        dtype=np.int32,
    )

    for _, row in components.iterrows():
        rid = str(
            row["response_atom_id"]
        )

        ai = p5a_pos[rid]
        ti = text_pos[
            str(
                row["p5a_text"]
            )
        ]

        sums[ai] += text_emb[ti]
        counts[ai] += 1

    if np.any(counts <= 0):
        raise RuntimeError(
            f"{key}: P5A atom without component"
        )

    atom_emb = l2(
        sums
        / counts[:, None].astype(
            np.float32
        )
    )

    np.save(
        atom_path,
        atom_emb.astype(
            np.float32
        ),
    )

    write_json(
        {
            "status": "PASS",
            "model_key": key,
            "model_id": spec["model_id"],
            "response_atom_n": len(atom_index),
            "dim": spec["dim"],
            "aggregation": "component_mean_then_l2",
        },
        done_path,
    )

    model_atom_paths[
        key
    ] = atom_path


# -----------------------------------------------------------------------------
# Alignment
# -----------------------------------------------------------------------------

p4_pos = {
    rid: i
    for i, rid in enumerate(
        p4_index[
            "response_atom_id"
        ].astype(str)
    )
}

rows = []


for ds in DATASETS:
    ds_atoms = atom_index.loc[
        atom_index[
            "dataset_id"
        ].eq(ds)
    ].copy()

    ids = ds_atoms[
        "response_atom_id"
    ].astype(str).tolist()

    response_S, response_ids = load_response(
        RESP_ROOT
        / ds
        / "response_similarity_logfc_hvg_spearman.npz"
    )

    rpos = {
        rid: i
        for i, rid in enumerate(
            response_ids
        )
    }

    missing = [
        rid
        for rid in ids
        if rid not in rpos
    ]

    if missing:
        raise RuntimeError(
            f"{ds}: P5A atoms absent from response geometry n={len(missing)}"
        )

    rix = np.asarray(
        [
            rpos[rid]
            for rid in ids
        ],
        dtype=np.int64,
    )

    E = np.asarray(
        response_S[
            np.ix_(
                rix,
                rix,
            )
        ],
        dtype=np.float32,
    )

    ev = upper(E)

    p5rows = np.asarray(
        [
            p5a_pos[rid]
            for rid in ids
        ],
        dtype=np.int64,
    )

    p4rows = np.asarray(
        [
            p4_pos[rid]
            for rid in ids
        ],
        dtype=np.int64,
    )

    for spec in MODEL_SPECS:
        key = spec["model_key"]

        Z5 = np.asarray(
            np.load(
                model_atom_paths[key],
                mmap_mode="r",
            )[p5rows],
            dtype=np.float32,
        )

        p4_path = (
            P4_EMB_ROOT
            / key
            / "response_atom_embeddings.npy"
        )

        Z4 = np.asarray(
            np.load(
                p4_path,
                mmap_mode="r",
            )[p4rows],
            dtype=np.float32,
        )

        S5 = np.clip(
            Z5 @ Z5.T,
            -1.0,
            1.0,
        )

        S4 = np.clip(
            Z4 @ Z4.T,
            -1.0,
            1.0,
        )

        np.fill_diagonal(
            S5,
            1.0,
        )

        np.fill_diagonal(
            S4,
            1.0,
        )

        rsa5 = spearman(
            upper(S5),
            ev,
        )

        rsa4 = spearman(
            upper(S4),
            ev,
        )

        if not np.isfinite(rsa5) or not np.isfinite(rsa4):
            raise RuntimeError(
                f"{ds}/{key}: invalid P4/P5A RSA"
            )

        rows.append({
            "dataset_id": ds,
            "model_key": key,
            "model_display": spec["display"],
            "model_group": spec["group"],
            "knowledge_condition_n": int(
                ds_atoms[
                    "condition_id"
                ].nunique()
            ),
            "response_atom_n": len(ids),
            "pair_n": int(
                len(ev)
            ),
            "p4_rsa": rsa4,
            "p5a_rsa": rsa5,
            "delta_rsa_p5a_minus_p4": (
                rsa5 - rsa4
            ),
            "status": "PASS",
        })


by_dataset = pd.DataFrame(rows)

by_dataset.to_csv(
    BY_DATASET,
    sep="\t",
    index=False,
)


summary_rows = []


for key, sub in by_dataset.groupby(
    "model_key",
    sort=False,
):
    delta = sub[
        "delta_rsa_p5a_minus_p4"
    ].to_numpy(
        dtype=float
    )

    lo, hi = bootstrap_ci(
        delta,
        stable_seed(
            BOOTSTRAP_SEED,
            "corrected_P5A",
            key,
            "delta_rsa",
        ),
    )

    summary_rows.append({
        "model_key": key,
        "model_display": sub[
            "model_display"
        ].iloc[0],
        "model_group": sub[
            "model_group"
        ].iloc[0],
        "dataset_n": int(
            sub[
                "dataset_id"
            ].nunique()
        ),
        "delta_rsa_macro_mean": float(
            delta.mean()
        ),
        "delta_rsa_macro_median": float(
            np.median(
                delta
            )
        ),
        "delta_rsa_positive_dataset_n": int(
            (
                delta > 0
            ).sum()
        ),
        "delta_rsa_bootstrap_ci_low": lo,
        "delta_rsa_bootstrap_ci_high": hi,
        "bootstrap_unit": "dataset",
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
    })


summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    SUMMARY,
    sep="\t",
    index=False,
)


lines = [
    "PERTURBCONTEXTALIGN R2 CORRECTED P5A AUDIT v1.2",
    "=" * 120,
    "",
    "CONTRACT",
    "-" * 120,
    "dataset_scope=8_historical_manuscript_datasets_excluding_McFarland",
    "knowledge_source=frozen_historical_entity_knowledge_blocks",
    "corrected_construction=corrected_P4_component + matched_frozen_entity_knowledge_block",
    "component_aggregation=mean_then_L2",
    "missing_knowledge_zero_fill=FALSE",
    "",
    "COUNTS",
    "-" * 120,
    f"response_atom_n={len(atom_index)}",
    f"knowledge_condition_n={atom_index[['dataset_id','condition_id']].drop_duplicates().shape[0]}",
    f"mapped_component_n={len(component_suffix)}",
    f"excluded_noneligible_response_atom_n={int(eligible_atom_audit['excluded_noneligible_atom_n'].sum())}",
    "response_atom_scope=frozen_response_similarity_eligible_namespace_only",
    "",
    "ELIGIBLE ATOM AUDIT",
    "-" * 120,
    eligible_atom_audit.to_string(index=False),
    "",
    "MODEL SUMMARY",
    "-" * 120,
    summary.to_string(index=False),
    "",
    "STATUS",
    "-" * 120,
    "knowledge_component_linkage=historical_entity_id_to_entity_name_to_corrected_component",
    "canonical_name_used_for_component_linkage=FALSE",
    "R2_CORRECTED_P5A=PASS",
]

AUDIT.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)

write_json(
    {
        "version": "R2_CORRECTED_P5A_v1.2",
        "status": "PASS",
        "datasets": DATASETS,
        "response_atom_n": len(atom_index),
        "input_hashes": {
            "p5a_preflight_authority": sha256_file(
                P5A_AUTH
            ),
            "component_p4": sha256_file(
                COMPONENT_P4
            ),
            "entity_knowledge_cache": sha256_file(
                ENTITY_KNOWLEDGE_CACHE
            ),
        },
        "outputs": {
            "component_authority": str(
                P5A_COMPONENT_OUT
            ),
            "atom_index": str(
                P5A_ATOM_INDEX
            ),
            "eligible_atom_audit": str(
                ELIGIBLE_ATOM_AUDIT
            ),
            "by_dataset": str(
                BY_DATASET
            ),
            "summary": str(
                SUMMARY
            ),
            "audit": str(
                AUDIT
            ),
        },
    },
    MANIFEST,
)

print(
    AUDIT.read_text(
        encoding="utf-8"
    )
)
