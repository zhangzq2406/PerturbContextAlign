#!/usr/bin/env python3
# R1_FULL_PATCH_V1_1_SPARSE_RETRIEVAL_CHECKPOINT
# - fixes scipy sparse retrieval conversion
# - adds retrieval checkpoint/resume
# - fixes 9-dataset semantic-instance total to 186319
from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import math
import os
import platform
import re
import shutil
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize as sk_normalize


SEED = 20260711
MAX_LENGTH = 512
ENCODER_BATCH_SIZE = 64
TEXT_CHUNK_SIZE = 2048
AGG_COMPONENT_CHUNK = 4096

RETRIEVAL_QUERY_CAP = 2500
RETRIEVAL_QUERY_BATCH = 32
RETRIEVAL_SCORE_DECIMALS = 8

CLASSIFICATION_ROW_CAP = 5000
CLASSIFICATION_K = 5
CLASSIFICATION_FOLDS = 5
MIN_CLASS_COUNT = 5

TFIDF_MAX_FEATURES = 20000
BASELINE_DIM = 256

VIEWS = ("P1", "P2", "P3", "P4")
TRANSITIONS = (("P1", "P4"), ("P2", "P4"), ("P3", "P4"))

MODEL_SPECS = [
    {
        "model_key": "bge_m3",
        "display_name": "BGE-M3",
        "group": "general_purpose",
        "model_id": "BAAI/bge-m3",
        "backend": "sentence_transformers",
        "pooling": "sentence_transformers",
        "trust_remote_code": False,
        "expected_dim": 1024,
    },
    {
        "model_key": "qwen3_0_6b",
        "display_name": "Qwen3-Embedding-0.6B",
        "group": "general_purpose",
        "model_id": "Qwen/Qwen3-Embedding-0.6B",
        "backend": "sentence_transformers",
        "pooling": "sentence_transformers",
        "trust_remote_code": True,
        "expected_dim": 1024,
    },
    {
        "model_key": "sapbert",
        "display_name": "SapBERT",
        "group": "biomedical",
        "model_id": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        "backend": "transformers",
        "pooling": "mean",
        "trust_remote_code": False,
        "expected_dim": 768,
    },
    {
        "model_key": "biomedbert",
        "display_name": "BiomedBERT",
        "group": "biomedical",
        "model_id": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        "backend": "transformers",
        "pooling": "mean",
        "trust_remote_code": False,
        "expected_dim": 768,
    },
    {
        "model_key": "medcpt_query",
        "display_name": "MedCPT Query",
        "group": "biomedical",
        "model_id": "ncbi/MedCPT-Query-Encoder",
        "backend": "transformers",
        "pooling": "cls",
        "trust_remote_code": False,
        "expected_dim": 768,
    },
    {
        "model_key": "medcpt_article",
        "display_name": "MedCPT Article",
        "group": "biomedical",
        "model_id": "ncbi/MedCPT-Article-Encoder",
        "backend": "transformers",
        "pooling": "cls",
        "trust_remote_code": False,
        "expected_dim": 768,
    },
]

CONTROL_SPECS = [
    {
        "model_key": "tfidf_svd",
        "display_name": "TF-IDF + SVD",
        "group": "lexical_control",
    },
    {
        "model_key": "metadata_onehot",
        "display_name": "Metadata one-hot",
        "group": "structured_control",
    },
    {
        "model_key": "random",
        "display_name": "Random",
        "group": "random_control",
    },
]

METHOD_SPECS = MODEL_SPECS + CONTROL_SPECS
METHOD_KEYS = tuple(x["model_key"] for x in METHOD_SPECS)

DATASET_CONFIGS = {
    "norman_2019": {
        "freeze_name": "Norman",
        "expected_semantic_n": 1888,
        "expected_signatures": {"P1": 236, "P2": 236, "P3": 1888, "P4": 1888},
    },
    "replogle_k562_essential": {
        "freeze_name": "K562_essential",
        "expected_semantic_n": 84393,
        "expected_signatures": {"P1": 2057, "P2": 2057, "P3": 84393, "P4": 84393},
    },
    "replogle_rpe1": {
        "freeze_name": "RPE1",
        "expected_semantic_n": 90285,
        "expected_signatures": {"P1": 2393, "P2": 2393, "P3": 90285, "P4": 90285},
    },
    "tian_activation": {
        "freeze_name": "TianActivation",
        "expected_semantic_n": 200,
        "expected_signatures": {"P1": 100, "P2": 100, "P3": 200, "P4": 200},
    },
    "tian_inhibition": {
        "freeze_name": "TianInhibition",
        "expected_semantic_n": 736,
        "expected_signatures": {"P1": 184, "P2": 184, "P3": 736, "P4": 736},
    },
    "srivatsan_sciplex3": {
        "freeze_name": "sciPlex3",
        "expected_semantic_n": 4889,
        "expected_signatures": {"P1": 189, "P2": 565, "P3": 4889, "P4": 4889},
    },
    "mcfarland_2020": {
        "freeze_name": "McFarland",
        "expected_semantic_n": 1627,
        "expected_signatures": {"P1": 17, "P2": 1370, "P3": 32, "P4": 1627},
    },
    "kaggle_cross_patient": {
        "freeze_name": "KaggleCrossPatient",
        "expected_semantic_n": 2270,
        "expected_signatures": {"P1": 146, "P2": 614, "P3": 528, "P4": 2270},
    },
    "combo_sciplex": {
        "freeze_name": None,
        "expected_semantic_n": 31,
        "expected_signatures": {"P1": None, "P2": None, "P3": None, "P4": 31},
    },
}

BIO_FIELDS = ["species", "cell_type", "cell_line", "tissue", "disease"]
TECH_FIELDS = ["platform", "batch"]
EXPOSURE_FIELDS = ["dose", "duration"]
INTERVENTION_FIELDS = ["family", "mode", "entity", "combo_n"]

VISIBLE_FIELDS = {
    "P1": INTERVENTION_FIELDS,
    "P2": INTERVENTION_FIELDS + BIO_FIELDS,
    "P3": INTERVENTION_FIELDS + TECH_FIELDS + EXPOSURE_FIELDS,
    "P4": INTERVENTION_FIELDS + BIO_FIELDS + TECH_FIELDS + EXPOSURE_FIELDS,
}

LABEL_VIEWS = {
    "family": ("P1", "P4"),
    "mode": ("P1", "P4"),
    "combo_n": ("P1", "P4"),
    "species": ("P1", "P2", "P4"),
    "cell_type": ("P1", "P2", "P4"),
    "cell_line": ("P1", "P2", "P4"),
    "tissue": ("P1", "P2", "P4"),
    "disease": ("P1", "P2", "P4"),
    "platform": ("P1", "P3", "P4"),
    "batch": ("P1", "P3", "P4"),
    "dose": ("P1", "P3", "P4"),
    "duration": ("P1", "P3", "P4"),
}

PLACEHOLDERS = {
    "",
    "unknown",
    "unknown_donor",
    "unknown_target",
    "not_applicable",
    "not applicable",
    "not_applicable_cell_line",
    "not_applicable_primary_cell",
    "__mixed__",
    "nan",
    "none",
    "null",
    "<na>",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Corrected PerturbContextAlign Result 1 full 9-dataset run."
    )
    ap.add_argument("--root", required=True)
    ap.add_argument("--launcher", default="")
    return ap.parse_args()


ARGS = parse_args()
ROOT = Path(ARGS.root).expanduser().resolve()
WORK = ROOT / "0_ProjectCodeAndLogs" / "PerturbContextAlign_corrected_broad_v1_20261002"
OUT = WORK / "01_result1_retention" / "05_full_v1"
AUTH_OUT = OUT / "00_authority"
TEXT_OUT = OUT / "01_text_catalog"
TEXT_EMB_OUT = OUT / "02_text_embeddings"
SEM_OUT = OUT / "03_semantic_embeddings"
CONTROL_OUT = OUT / "04_controls"
METRIC_OUT = OUT / "05_metrics"
SUMMARY_OUT = OUT / "06_summary"
AUDIT_OUT = OUT / "07_audit"

FREEZE = ROOT / "0_ProjectCodeAndLogs" / "PHASE_AB_FREEZE_v1_20261001"
COMBO_AUTH = WORK / "00_contract" / "combo_sciplex_p4_authority_v1"

RELEASE = WORK / "08_release_staging"
GH = RELEASE / "github_candidate"
FIG = RELEASE / "figshare_candidate"
PROV = RELEASE / "internal_provenance"

for d in [
    AUTH_OUT,
    TEXT_OUT,
    TEXT_EMB_OUT,
    SEM_OUT,
    CONTROL_OUT,
    METRIC_OUT,
    SUMMARY_OUT,
    AUDIT_OUT,
    GH / "scripts" / "result1",
    GH / "docs",
    GH / "manifests",
    FIG / "result1" / "final_tables",
    FIG / "manifests",
    PROV / "result1" / "full_v1",
]:
    d.mkdir(parents=True, exist_ok=True)


os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def log(msg: str) -> None:
    print(msg, flush=True)


def step(name: str) -> None:
    log("")
    log("=" * 120)
    log(f"step={name}")
    log("=" * 120)


def sha256_file(path: Path, chunk: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def stable_hash(*parts: Any, length: int = 32) -> str:
    payload = "\x1f".join(str(x) for x in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def clean_value(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"", "nan", "none", "null", "<na>", "na"}:
        return ""
    return s


def supported_value(x: Any) -> bool:
    s = clean_value(x)
    return bool(s and s.lower() not in PLACEHOLDERS)


def bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    return (
        s.astype("string")
        .fillna("")
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "control", "ctrl"})
    )


def l2_dense(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-12)
    return x / denom


def family_class(family: Any) -> str:
    x = clean_value(family).lower()
    if any(k in x for k in ["genetic", "crispr", "gene perturb"]):
        return "genetic"
    if any(k in x for k in ["drug", "small molecule", "small-molecule", "compound"]):
        return "drug"
    if any(k in x for k in ["cytokine", "ligand", "growth factor"]):
        return "cytokine"
    if any(k in x for k in ["environment", "culture", "exposure"]):
        return "environment"
    if any(k in x for k in ["immune", "infection", "stimulus"]):
        return "immune"
    return x or "other"


def family_aware_entity(row: pd.Series) -> str:
    cls = family_class(row.get("family", ""))
    agent = clean_value(row.get("agent", ""))
    target = clean_value(row.get("target", ""))
    if cls == "genetic":
        return target
    return agent or target


def canonical_entity_key(row: pd.Series) -> str:
    cls = family_class(row.get("family", ""))
    entity = family_aware_entity(row)
    parts = [
        clean_value(x).lower()
        for x in re.split(r"\s*[|;,]\s*", entity)
        if clean_value(x)
    ]
    if not parts:
        return f"{cls}::condition_fallback::{clean_value(row.get('condition_id', ''))}"
    return f"{cls}::" + "|".join(sorted(set(parts)))


def parse_p4_ablation(text: str) -> dict[str, str]:
    text = str(text).strip()
    labels = [
        " Biological context:",
        " Technical context:",
        " Dose:",
        " Duration:",
    ]
    pos = [text.find(x) for x in labels]
    if any(x < 0 for x in pos):
        raise ValueError(f"missing P4 section: {text[:240]}")
    if not (pos[0] < pos[1] < pos[2] < pos[3]):
        raise ValueError(f"invalid P4 section order: {text[:240]}")

    prefix = text[: pos[0]].strip()
    bio = text[pos[0] : pos[1]].strip()
    tech = text[pos[1] : pos[2]].strip()
    dose = text[pos[2] : pos[3]].strip()
    duration = text[pos[3] :].strip()

    if not all([prefix, bio, tech, dose, duration]):
        raise ValueError(f"blank P4 section: {text[:240]}")

    return {
        "P1": prefix,
        "P2": f"{prefix} {bio}",
        "P3": f"{prefix} {tech} {dose} {duration}",
        "P4": text,
    }


def semantic_signature(values: Iterable[str]) -> tuple[str, str]:
    canonical = json.dumps(sorted(map(str, values)), ensure_ascii=False)
    return stable_hash(canonical, length=40), canonical


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def normalize_memmap_rows(path: Path, row_chunk: int = 4096) -> None:
    arr = np.load(path, mmap_mode="r+")
    for start in range(0, arr.shape[0], row_chunk):
        stop = min(arr.shape[0], start + row_chunk)
        block = np.asarray(arr[start:stop], dtype=np.float32)
        denom = np.linalg.norm(block, axis=1, keepdims=True)
        if np.any(denom <= 1e-12):
            raise RuntimeError(f"zero-norm row in {path} [{start}:{stop}]")
        arr[start:stop] = block / denom
    arr.flush()


def write_json(obj: Any, path: Path) -> None:
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


# ==================================================================================================
# Authority resolution
# ==================================================================================================

step("01_authority_resolution")

manifest_path = FREEZE / "FREEZE_MANIFEST.json"
if not manifest_path.is_file():
    raise FileNotFoundError(manifest_path)

freeze_manifest = json.loads(manifest_path.read_text())
freeze_records = {
    str(x["dataset"]): x
    for x in freeze_manifest["dataset_records"]
}

file_records = freeze_manifest.get("file_records", [])
file_hash_index = {
    (str(x["dataset"]), str(x["role"])): str(x.get("sha256", ""))
    for x in file_records
}

generator_record = next(
    (
        x
        for x in file_records
        if str(x.get("dataset")) == "GLOBAL"
        and str(x.get("role")) == "corrected_p4_generator"
    ),
    None,
)
if not generator_record:
    raise RuntimeError("corrected P4 generator missing from freeze manifest")

generator_path = Path(generator_record["absolute_path"])
generator_sha = sha256_file(generator_path)
if generator_sha != str(generator_record["sha256"]):
    raise RuntimeError("corrected P4 generator SHA mismatch")

combo_manifest_path = COMBO_AUTH / "COMBO_SCIPLEX_P4_AUTHORITY_MANIFEST_v1.json"
if not combo_manifest_path.is_file():
    raise FileNotFoundError(combo_manifest_path)

combo_manifest = json.loads(combo_manifest_path.read_text())
if combo_manifest.get("status") != "PASS":
    raise RuntimeError("ComboSciPlex current P4 authority is not PASS")


def resolve_dataset_authority(dataset_id: str, cfg: dict[str, Any]) -> dict[str, Any]:
    freeze_name = cfg["freeze_name"]

    if freeze_name is None:
        semantic_csv = COMBO_AUTH / "semantic_instance_table_bgem3_p4.csv"
        prompt_csv = COMBO_AUTH / "p4_component_prompts.csv"
        return {
            "dataset_id": dataset_id,
            "freeze_name": "",
            "semantic_csv": semantic_csv,
            "prompt_csv": prompt_csv,
            "frozen_h5ad": None,
            "pooled_key": "",
            "expected_prompt_sha": sha256_file(prompt_csv),
            "expected_condition_sha": sha256_file(semantic_csv),
            "authority_source": "combo_sciplex_p4_authority_v1",
        }

    if freeze_name not in freeze_records:
        raise RuntimeError(f"{dataset_id}: freeze record missing: {freeze_name}")

    rec = freeze_records[freeze_name]
    frozen_h5ad = Path(rec["canonical_embedding_h5ad"])
    parent = frozen_h5ad.parent

    semantic_csv = parent / "condition_table_bgem3_p4.csv"
    prompt_csv = parent / "p4_component_prompts.csv"

    return {
        "dataset_id": dataset_id,
        "freeze_name": freeze_name,
        "semantic_csv": semantic_csv,
        "prompt_csv": prompt_csv,
        "frozen_h5ad": frozen_h5ad,
        "pooled_key": str(rec["pooled_key"]),
        "expected_prompt_sha": file_hash_index.get((freeze_name, "p4_component_prompts"), ""),
        "expected_condition_sha": file_hash_index.get((freeze_name, "condition_table_bgem3_p4"), ""),
        "authority_source": "PHASE_AB_FREEZE_v1_20261001",
    }


authorities = {
    ds: resolve_dataset_authority(ds, cfg)
    for ds, cfg in DATASET_CONFIGS.items()
}

authority_rows = []
dataset_meta: dict[str, pd.DataFrame] = {}
dataset_components: dict[str, pd.DataFrame] = {}
dataset_uids: dict[str, list[str]] = {}
signature_maps: dict[tuple[str, str], dict[str, str]] = {}

all_text_set: set[str] = set()

for dataset_id, cfg in DATASET_CONFIGS.items():
    auth = authorities[dataset_id]

    semantic_csv = Path(auth["semantic_csv"])
    prompt_csv = Path(auth["prompt_csv"])

    for p in [semantic_csv, prompt_csv]:
        if not p.is_file():
            raise FileNotFoundError(p)

    prompt_sha = sha256_file(prompt_csv)
    condition_sha = sha256_file(semantic_csv)

    prompt_hash_status = (
        "PASS"
        if not auth["expected_prompt_sha"] or prompt_sha == auth["expected_prompt_sha"]
        else "FAIL"
    )
    condition_hash_status = (
        "PASS"
        if not auth["expected_condition_sha"] or condition_sha == auth["expected_condition_sha"]
        else "FAIL"
    )

    s = pd.read_csv(semantic_csv)
    p = pd.read_csv(prompt_csv)

    required_s = {
        "semantic_instance_id",
        "condition_id",
        "is_control",
        "family",
        "mode",
        "agent",
        "target",
        "combo_n",
        "dose",
        "duration",
        "species",
        "cell_type",
        "cell_line",
        "donor",
        "tissue",
        "disease",
        "platform",
        "batch",
    }
    required_p = {
        "semantic_instance_id",
        "condition_id",
        "component_id",
        "prompt_type",
        "model_name",
        "prompt",
    }

    missing_s = required_s - set(s.columns)
    missing_p = required_p - set(p.columns)
    if missing_s:
        raise RuntimeError(f"{dataset_id}: semantic table missing {sorted(missing_s)}")
    if missing_p:
        raise RuntimeError(f"{dataset_id}: prompt table missing {sorted(missing_p)}")

    s["is_control"] = bool_series(s["is_control"])
    s = s.loc[~s["is_control"]].copy()
    s["semantic_instance_id"] = s["semantic_instance_id"].astype(str)
    s["condition_id"] = s["condition_id"].astype(str)

    p["semantic_instance_id"] = p["semantic_instance_id"].astype(str)
    p["component_id"] = p["component_id"].astype(str)
    p["prompt"] = p["prompt"].astype(str)

    non_ids = set(s["semantic_instance_id"].astype(str))
    p = p.loc[p["semantic_instance_id"].isin(non_ids)].copy()

    semantic_n = int(s["semantic_instance_id"].nunique())
    condition_n = int(s["condition_id"].nunique())

    if semantic_n != int(cfg["expected_semantic_n"]):
        raise RuntimeError(
            f"{dataset_id}: semantic count mismatch {semantic_n} != {cfg['expected_semantic_n']}"
        )

    duplicate_pair_n = int(
        p.duplicated(["semantic_instance_id", "component_id"], keep=False).sum()
    )
    if duplicate_pair_n:
        raise RuntimeError(f"{dataset_id}: duplicate semantic/component pairs={duplicate_pair_n}")

    missing_prompt_sid = non_ids - set(p["semantic_instance_id"])
    extra_prompt_sid = set(p["semantic_instance_id"]) - non_ids
    if missing_prompt_sid or extra_prompt_sid:
        raise RuntimeError(
            f"{dataset_id}: prompt semantic-ID mismatch "
            f"missing={len(missing_prompt_sid)} extra={len(extra_prompt_sid)}"
        )

    parsed_rows = []
    for text in p["prompt"].astype(str):
        parsed_rows.append(parse_p4_ablation(text))

    parsed = pd.DataFrame(parsed_rows)
    p = pd.concat([p.reset_index(drop=True), parsed], axis=1)

    if not p["P4"].astype(str).equals(p["prompt"].astype(str)):
        raise RuntimeError(f"{dataset_id}: exact P4 identity failed")

    if not p["prompt_type"].astype(str).eq("p4_normalized_full_context").all():
        raise RuntimeError(f"{dataset_id}: unexpected prompt_type")

    if not p["model_name"].astype(str).eq("BAAI/bge-m3").all():
        raise RuntimeError(f"{dataset_id}: unexpected prompt model_name")

    s["entity"] = s.apply(family_aware_entity, axis=1)
    s["entity_key"] = s.apply(canonical_entity_key, axis=1)

    condition_entity_n = (
        s.groupby("condition_id", observed=True)["entity_key"].nunique()
    )
    if int(condition_entity_n.gt(1).sum()) != 0:
        raise RuntimeError(
            f"{dataset_id}: a condition maps to multiple entity keys"
        )

    semantic_ids = sorted(s["semantic_instance_id"].astype(str).unique())
    s = (
        s.drop_duplicates("semantic_instance_id")
        .set_index("semantic_instance_id")
        .loc[semantic_ids]
        .reset_index()
    )

    # Literal donor leakage check. Donor is schema/provenance only, not P1-P4 text.
    donor_map = dict(
        zip(
            s["semantic_instance_id"].astype(str),
            s["donor"].astype(str),
        )
    )
    donor_checked_n = 0
    donor_leak_n = 0

    for sid, g in p.groupby("semantic_instance_id", sort=False):
        donor = clean_value(donor_map.get(str(sid), ""))
        if not supported_value(donor):
            continue
        donor_checked_n += 1
        if any(donor in text for text in g["P4"].astype(str)):
            donor_leak_n += 1

    if donor_leak_n:
        raise RuntimeError(f"{dataset_id}: donor literal leak n={donor_leak_n}")

    # Signatures.
    sig_counts = {}
    for view in VIEWS:
        sig_map = {}
        for sid, g in p.groupby("semantic_instance_id", sort=False):
            sig, _ = semantic_signature(g[view].astype(str).tolist())
            sig_map[str(sid)] = sig

        signature_maps[(dataset_id, view)] = sig_map
        s[f"signature_{view}"] = s["semantic_instance_id"].map(sig_map)

        observed = int(len(set(sig_map.values())))
        expected = cfg["expected_signatures"].get(view)
        status = "PASS" if expected is None or observed == int(expected) else "FAIL"

        if status != "PASS":
            raise RuntimeError(
                f"{dataset_id} {view}: signature count {observed} != {expected}"
            )

        sig_counts[view] = observed

    for view in VIEWS:
        all_text_set.update(p[view].astype(str).tolist())

    ds_auth_dir = AUTH_OUT / dataset_id
    ds_auth_dir.mkdir(parents=True, exist_ok=True)

    s.to_csv(
        ds_auth_dir / "semantic_metadata.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    p[
        [
            "semantic_instance_id",
            "condition_id",
            "component_id",
            "P1",
            "P2",
            "P3",
            "P4",
        ]
    ].to_csv(
        ds_auth_dir / "component_prompts.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )

    np.save(
        ds_auth_dir / "semantic_uids.npy",
        np.asarray(semantic_ids, dtype=str),
    )

    authority_rows.append(
        {
            "dataset_id": dataset_id,
            "freeze_name": auth["freeze_name"],
            "authority_source": auth["authority_source"],
            "semantic_instance_n": semantic_n,
            "condition_n": condition_n,
            "component_prompt_n": int(len(p)),
            "unique_p1_signature_n": sig_counts["P1"],
            "unique_p2_signature_n": sig_counts["P2"],
            "unique_p3_signature_n": sig_counts["P3"],
            "unique_p4_signature_n": sig_counts["P4"],
            "donor_checked_n": donor_checked_n,
            "donor_literal_leak_n": donor_leak_n,
            "prompt_sha256": prompt_sha,
            "prompt_hash_status": prompt_hash_status,
            "condition_table_sha256": condition_sha,
            "condition_hash_status": condition_hash_status,
            "status": (
                "PASS"
                if prompt_hash_status == "PASS"
                and condition_hash_status == "PASS"
                and donor_leak_n == 0
                else "FAIL"
            ),
        }
    )

    dataset_meta[dataset_id] = s
    dataset_components[dataset_id] = p
    dataset_uids[dataset_id] = semantic_ids

authority_df = pd.DataFrame(authority_rows)
authority_df.to_csv(
    AUTH_OUT / "R1_DATASET_AUTHORITY.tsv",
    sep="\t",
    index=False,
)

if not authority_df["status"].eq("PASS").all():
    raise RuntimeError("R1 authority audit failed")

log(authority_df.to_string(index=False))


# ==================================================================================================
# Exact text catalog
# ==================================================================================================

step("02_exact_text_catalog")

all_texts = sorted(all_text_set)
text_to_index = {text: i for i, text in enumerate(all_texts)}
text_hashes = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in all_texts]

logical_text_catalog_sha = hashlib.sha256(
    "\n".join(text_hashes).encode("ascii")
).hexdigest()

text_catalog = pd.DataFrame(
    {
        "text_index": np.arange(len(all_texts), dtype=np.int64),
        "text_sha256": text_hashes,
        "prompt_text": all_texts,
    }
)
text_catalog.to_csv(
    TEXT_OUT / "all_component_texts.tsv.gz",
    sep="\t",
    index=False,
    compression="gzip",
)

(TEXT_OUT / "TEXT_CATALOG_LOGICAL_SHA256.txt").write_text(
    logical_text_catalog_sha + "\n",
    encoding="utf-8",
)

for dataset_id, p in dataset_components.items():
    for view in VIEWS:
        p[f"text_index_{view}"] = p[view].map(text_to_index).astype(np.int64)

    p[
        [
            "semantic_instance_id",
            "condition_id",
            "component_id",
            "text_index_P1",
            "text_index_P2",
            "text_index_P3",
            "text_index_P4",
        ]
    ].to_csv(
        AUTH_OUT / dataset_id / "component_text_indices.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )

log(f"text_n={len(all_texts)}")
log(f"text_catalog_logical_sha256={logical_text_catalog_sha}")


# ==================================================================================================
# Encoder implementation
# ==================================================================================================

def mean_pool(last_hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom


class EncoderRunner:
    def __init__(self, spec: dict[str, Any]):
        self.spec = spec
        self.model = None
        self.tokenizer = None
        self.dim = int(spec["expected_dim"])

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

            try:
                observed_dim = int(self.model.get_sentence_embedding_dimension())
            except Exception:
                observed_dim = self.dim

            if observed_dim != self.dim:
                raise RuntimeError(
                    f"{spec['model_key']}: dim {observed_dim} != expected {self.dim}"
                )

        elif spec["backend"] == "transformers":
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

            observed_dim = int(getattr(self.model.config, "hidden_size", self.dim))
            if observed_dim != self.dim:
                raise RuntimeError(
                    f"{spec['model_key']}: dim {observed_dim} != expected {self.dim}"
                )
        else:
            raise ValueError(spec["backend"])

    def encode(self, texts: list[str]) -> np.ndarray:
        spec = self.spec

        if spec["backend"] == "sentence_transformers":
            z = self.model.encode(
                texts,
                batch_size=ENCODER_BATCH_SIZE,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return np.asarray(z, dtype=np.float32)

        chunks = []
        for start in range(0, len(texts), ENCODER_BATCH_SIZE):
            local = texts[start : start + ENCODER_BATCH_SIZE]
            batch = self.tokenizer(
                local,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            with torch.inference_mode():
                out = self.model(**batch)
                if spec["pooling"] == "mean":
                    z = mean_pool(out.last_hidden_state, batch["attention_mask"])
                elif spec["pooling"] == "cls":
                    z = out.last_hidden_state[:, 0]
                else:
                    raise ValueError(spec["pooling"])

            chunks.append(z.float().cpu().numpy())

        return l2_dense(np.concatenate(chunks, axis=0))

    def close(self) -> None:
        try:
            del self.model
        except Exception:
            pass
        try:
            del self.tokenizer
        except Exception:
            pass
        self.model = None
        self.tokenizer = None
        cleanup_cuda()


def encode_text_cache(spec: dict[str, Any]) -> Path:
    key = spec["model_key"]
    out_dir = TEXT_EMB_OUT / key
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / "text_embeddings.npy"
    progress_path = out_dir / "progress.json"
    done_path = out_dir / "DONE.json"

    if done_path.is_file() and path.is_file():
        done = json.loads(done_path.read_text())
        arr = np.load(path, mmap_mode="r")
        if (
            done.get("text_catalog_logical_sha256") == logical_text_catalog_sha
            and done.get("model_id") == spec["model_id"]
            and tuple(arr.shape) == (len(all_texts), int(spec["expected_dim"]))
        ):
            log(f"[resume] {key} text embeddings already DONE")
            return path

    expected_shape = (len(all_texts), int(spec["expected_dim"]))

    next_index = 0
    reuse = False

    if path.is_file() and progress_path.is_file():
        progress = json.loads(progress_path.read_text())
        try:
            arr = np.load(path, mmap_mode="r")
            shape_ok = tuple(arr.shape) == expected_shape
        except Exception:
            shape_ok = False

        reuse = bool(
            shape_ok
            and progress.get("text_catalog_logical_sha256") == logical_text_catalog_sha
            and progress.get("model_id") == spec["model_id"]
        )

        if reuse:
            next_index = int(progress.get("next_index", 0))
            log(f"[resume] {key} from text index {next_index}")

    if not reuse:
        if path.exists():
            path.unlink()
        if progress_path.exists():
            progress_path.unlink()

        arr = np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype="float32",
            shape=expected_shape,
        )
        arr.flush()
        del arr

        next_index = 0
        write_json(
            {
                "model_key": key,
                "model_id": spec["model_id"],
                "text_catalog_logical_sha256": logical_text_catalog_sha,
                "next_index": 0,
                "shape": list(expected_shape),
            },
            progress_path,
        )

    runner = EncoderRunner(spec)
    try:
        arr = np.load(path, mmap_mode="r+")

        for start in range(next_index, len(all_texts), TEXT_CHUNK_SIZE):
            stop = min(len(all_texts), start + TEXT_CHUNK_SIZE)
            t0 = time.time()

            z = runner.encode(all_texts[start:stop])

            if z.shape != (stop - start, int(spec["expected_dim"])):
                raise RuntimeError(
                    f"{key}: encode shape {z.shape} != {(stop - start, int(spec['expected_dim']))}"
                )

            norms = np.linalg.norm(z, axis=1)
            if not np.isfinite(z).all():
                raise RuntimeError(f"{key}: nonfinite text embeddings")
            if float(np.max(np.abs(norms - 1.0))) > 5e-4:
                raise RuntimeError(
                    f"{key}: text embedding norm failure max_dev={float(np.max(np.abs(norms - 1.0)))}"
                )

            arr[start:stop] = z
            arr.flush()

            write_json(
                {
                    "model_key": key,
                    "model_id": spec["model_id"],
                    "text_catalog_logical_sha256": logical_text_catalog_sha,
                    "next_index": stop,
                    "shape": list(expected_shape),
                },
                progress_path,
            )

            log(
                f"[{key}] texts {stop}/{len(all_texts)} "
                f"chunk_seconds={time.time() - t0:.1f}"
            )

        del arr

    finally:
        runner.close()

    write_json(
        {
            "status": "PASS",
            "model_key": key,
            "model_id": spec["model_id"],
            "backend": spec["backend"],
            "pooling": spec["pooling"],
            "dimension": int(spec["expected_dim"]),
            "normalize": True,
            "max_length": MAX_LENGTH,
            "text_n": len(all_texts),
            "text_catalog_logical_sha256": logical_text_catalog_sha,
        },
        done_path,
    )

    return path


# ==================================================================================================
# Semantic aggregation
# ==================================================================================================

def aggregate_dense_text_embeddings(
    text_embedding_path: Path,
    model_key: str,
    dim: int,
) -> None:
    model_dir = SEM_OUT / model_key
    model_dir.mkdir(parents=True, exist_ok=True)
    done_path = model_dir / "DONE.json"

    if done_path.is_file():
        done = json.loads(done_path.read_text())
        if done.get("text_catalog_logical_sha256") == logical_text_catalog_sha:
            all_exist = all(
                (model_dir / ds / f"{view}.npy").is_file()
                for ds in DATASET_CONFIGS
                for view in VIEWS
            )
            if all_exist:
                log(f"[resume] semantic aggregation DONE: {model_key}")
                return

    text_emb = np.load(text_embedding_path, mmap_mode="r")

    for dataset_id in DATASET_CONFIGS:
        ds_dir = model_dir / dataset_id
        ds_dir.mkdir(parents=True, exist_ok=True)

        p = dataset_components[dataset_id]
        semantic_ids = dataset_uids[dataset_id]
        sid_to_row = {sid: i for i, sid in enumerate(semantic_ids)}

        component_semantic_rows = (
            p["semantic_instance_id"].astype(str).map(sid_to_row).to_numpy(dtype=np.int64)
        )

        for view in VIEWS:
            path = ds_dir / f"{view}.npy"
            out = np.lib.format.open_memmap(
                path,
                mode="w+",
                dtype="float32",
                shape=(len(semantic_ids), dim),
            )
            out[:] = 0
            out.flush()

            counts = np.zeros(len(semantic_ids), dtype=np.int32)
            text_indices = p[f"text_index_{view}"].to_numpy(dtype=np.int64)

            for start in range(0, len(p), AGG_COMPONENT_CHUNK):
                stop = min(len(p), start + AGG_COMPONENT_CHUNK)
                rows = component_semantic_rows[start:stop]
                tids = text_indices[start:stop]
                vecs = np.asarray(text_emb[tids], dtype=np.float32)

                np.add.at(out, rows, vecs)
                np.add.at(counts, rows, 1)

            if np.any(counts <= 0):
                raise RuntimeError(
                    f"{model_key} {dataset_id} {view}: semantic row without component"
                )

            for start in range(0, len(semantic_ids), AGG_COMPONENT_CHUNK):
                stop = min(len(semantic_ids), start + AGG_COMPONENT_CHUNK)
                block = np.asarray(out[start:stop], dtype=np.float32)
                block = block / counts[start:stop, None].astype(np.float32)
                block = l2_dense(block)
                out[start:stop] = block

            out.flush()
            del out

            log(
                f"[aggregate] {model_key} {dataset_id} {view} "
                f"n={len(semantic_ids)} dim={dim}"
            )

    write_json(
        {
            "status": "PASS",
            "model_key": model_key,
            "text_catalog_logical_sha256": logical_text_catalog_sha,
            "pooling": "masked_mean_over_component_embeddings_then_l2",
        },
        done_path,
    )


# ==================================================================================================
# Six text encoders
# ==================================================================================================

step("03_six_language_encoders")

for spec in MODEL_SPECS:
    key = spec["model_key"]
    log(f"model_start={key}")
    text_path = encode_text_cache(spec)
    aggregate_dense_text_embeddings(
        text_path,
        key,
        int(spec["expected_dim"]),
    )
    log(f"model_done={key}")


# ==================================================================================================
# TF-IDF + SVD control
# ==================================================================================================

step("04_tfidf_svd_control")

tfidf_dir = CONTROL_OUT / "tfidf_svd"
tfidf_dir.mkdir(parents=True, exist_ok=True)
tfidf_text_path = tfidf_dir / "text_embeddings.npy"
tfidf_done = tfidf_dir / "DONE.json"

if not tfidf_done.is_file():
    log("TF-IDF fit_transform start")
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=(1, 2),
        lowercase=True,
    )
    X_sparse = vectorizer.fit_transform(all_texts)

    n_components = int(
        min(
            BASELINE_DIM,
            X_sparse.shape[0] - 1,
            X_sparse.shape[1] - 1,
        )
    )
    if n_components < 2:
        raise RuntimeError(f"TF-IDF/SVD rank too small: {X_sparse.shape}")

    svd = TruncatedSVD(
        n_components=n_components,
        random_state=SEED,
        n_iter=5,
    )
    X_tfidf = svd.fit_transform(X_sparse)
    X_tfidf = l2_dense(X_tfidf)

    np.save(tfidf_text_path, X_tfidf.astype(np.float32))

    write_json(
        {
            "status": "PASS",
            "text_catalog_logical_sha256": logical_text_catalog_sha,
            "tfidf_max_features": TFIDF_MAX_FEATURES,
            "ngram_range": [1, 2],
            "svd_dim": n_components,
            "explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum()),
        },
        tfidf_done,
    )

    del X_sparse, X_tfidf, svd, vectorizer
    gc.collect()
else:
    info = json.loads(tfidf_done.read_text())
    n_components = int(info["svd_dim"])
    log("[resume] TF-IDF/SVD text embeddings already DONE")

aggregate_dense_text_embeddings(
    tfidf_text_path,
    "tfidf_svd",
    n_components,
)


# ==================================================================================================
# Metadata one-hot sparse control
# ==================================================================================================

step("05_metadata_onehot_control")

metadata_dir = SEM_OUT / "metadata_onehot"
metadata_dir.mkdir(parents=True, exist_ok=True)
metadata_done = metadata_dir / "DONE.json"
metadata_feature_path = CONTROL_OUT / "metadata_onehot_feature_names.tsv"

if not metadata_done.is_file():
    feature_set: set[str] = set()

    def visible_features(row: pd.Series, view: str) -> list[str]:
        feats = []
        for field in VISIBLE_FIELDS[view]:
            if field == "entity":
                value = family_aware_entity(row)
            else:
                value = clean_value(row.get(field, ""))
            if value:
                feats.append(f"{field}={value}")
        return feats

    for dataset_id, meta in dataset_meta.items():
        for _, row in meta.iterrows():
            for view in VIEWS:
                feature_set.update(visible_features(row, view))

    feature_names = sorted(feature_set)
    feature_pos = {x: i for i, x in enumerate(feature_names)}

    pd.DataFrame(
        {
            "feature_index": np.arange(len(feature_names), dtype=np.int64),
            "feature_name": feature_names,
        }
    ).to_csv(
        metadata_feature_path,
        sep="\t",
        index=False,
    )

    for dataset_id, meta in dataset_meta.items():
        ds_dir = metadata_dir / dataset_id
        ds_dir.mkdir(parents=True, exist_ok=True)

        for view in VIEWS:
            indptr = [0]
            indices = []
            data = []

            for _, row in meta.iterrows():
                feats = visible_features(row, view)
                cols = sorted({feature_pos[x] for x in feats})

                if cols:
                    value = 1.0 / math.sqrt(len(cols))
                    indices.extend(cols)
                    data.extend([value] * len(cols))

                indptr.append(len(indices))

            mat = sparse.csr_matrix(
                (
                    np.asarray(data, dtype=np.float32),
                    np.asarray(indices, dtype=np.int32),
                    np.asarray(indptr, dtype=np.int64),
                ),
                shape=(len(meta), len(feature_names)),
                dtype=np.float32,
            )

            norms = np.sqrt(np.asarray(mat.multiply(mat).sum(axis=1)).ravel())
            if float(np.max(np.abs(norms - 1.0))) > 5e-6:
                raise RuntimeError(
                    f"metadata onehot norm failure {dataset_id} {view}"
                )

            sparse.save_npz(
                ds_dir / f"{view}.npz",
                mat,
                compressed=True,
            )

            log(
                f"[metadata] {dataset_id} {view} "
                f"n={mat.shape[0]} dim={mat.shape[1]} nnz={mat.nnz}"
            )

    write_json(
        {
            "status": "PASS",
            "dimension": len(feature_names),
            "feature_name_file": str(metadata_feature_path),
            "visible_fields": VISIBLE_FIELDS,
        },
        metadata_done,
    )

else:
    log("[resume] metadata one-hot already DONE")


# ==================================================================================================
# Direct metadata visibility audit
# ==================================================================================================

step("06_metadata_visibility_audit")

feature_table = pd.read_csv(metadata_feature_path, sep="\t")
feature_pos = dict(
    zip(
        feature_table["feature_name"].astype(str),
        feature_table["feature_index"].astype(int),
    )
)

visibility_rows = []

for dataset_id, meta in dataset_meta.items():
    for view in VIEWS:
        mat = sparse.load_npz(metadata_dir / dataset_id / f"{view}.npz").tocsr()

        for field in [
            "family",
            "mode",
            "combo_n",
            "species",
            "cell_type",
            "cell_line",
            "tissue",
            "disease",
            "platform",
            "batch",
            "dose",
            "duration",
        ]:
            expected_visible = field in VISIBLE_FIELDS[view]
            checked = 0
            fail_n = 0

            for i, row in meta.iterrows():
                value = clean_value(row.get(field, ""))
                if not supported_value(value):
                    continue

                checked += 1
                col = feature_pos.get(f"{field}={value}")

                active = bool(
                    col is not None
                    and mat[i, col] > 0
                )

                if active != expected_visible:
                    fail_n += 1

            if checked > 0:
                visibility_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "prompt_type": view,
                        "field": field,
                        "expected_visible": expected_visible,
                        "checked_n": checked,
                        "fail_n": fail_n,
                        "status": "PASS" if fail_n == 0 else "FAIL",
                    }
                )

visibility_df = pd.DataFrame(visibility_rows)
visibility_df.to_csv(
    AUDIT_OUT / "metadata_field_visibility_audit.tsv",
    sep="\t",
    index=False,
)

if not visibility_df["status"].eq("PASS").all():
    raise RuntimeError("metadata visibility audit failed")


# ==================================================================================================
# Random control
# ==================================================================================================

step("07_random_control")

random_dir = SEM_OUT / "random"
random_dir.mkdir(parents=True, exist_ok=True)
random_done = random_dir / "DONE.json"

if not random_done.is_file():
    for dataset_id, semantic_ids in dataset_uids.items():
        ds_dir = random_dir / dataset_id
        ds_dir.mkdir(parents=True, exist_ok=True)

        for view in VIEWS:
            seed = int(
                stable_hash("random", dataset_id, view, SEED, length=16),
                16,
            ) % (2**63 - 1)

            rng = np.random.default_rng(seed)
            X = rng.normal(
                size=(len(semantic_ids), BASELINE_DIM)
            ).astype(np.float32)
            X = l2_dense(X)

            np.save(ds_dir / f"{view}.npy", X)

            log(
                f"[random] {dataset_id} {view} "
                f"n={len(semantic_ids)} dim={BASELINE_DIM}"
            )

    write_json(
        {
            "status": "PASS",
            "dimension": BASELINE_DIM,
            "seed": SEED,
            "seed_rule": "stable_hash(global_seed,dataset_id,prompt_type)",
        },
        random_done,
    )
else:
    log("[resume] random control already DONE")


# ==================================================================================================
# Representation audit / catalog
# ==================================================================================================

step("08_representation_catalog_and_norm_audit")

representation_rows = []

for spec in METHOD_SPECS:
    method = spec["model_key"]

    for dataset_id, semantic_ids in dataset_uids.items():
        for view in VIEWS:
            if method == "metadata_onehot":
                path = metadata_dir / dataset_id / f"{view}.npz"
                mat = sparse.load_npz(path).tocsr()
                norms = np.sqrt(np.asarray(mat.multiply(mat).sum(axis=1)).ravel())
                n, dim = mat.shape
                sparse_flag = True
                finite = bool(np.isfinite(mat.data).all())
            else:
                path = SEM_OUT / method / dataset_id / f"{view}.npy"
                X = np.load(path, mmap_mode="r")
                n, dim = X.shape
                sparse_flag = False
                finite = bool(np.isfinite(X).all())

                norm_min = float("inf")
                norm_max = float("-inf")
                norm_values = []

                for start in range(0, n, AGG_COMPONENT_CHUNK):
                    stop = min(n, start + AGG_COMPONENT_CHUNK)
                    local = np.linalg.norm(
                        np.asarray(X[start:stop], dtype=np.float32),
                        axis=1,
                    )
                    norm_min = min(norm_min, float(local.min()))
                    norm_max = max(norm_max, float(local.max()))
                    norm_values.append(local)

                norms = np.concatenate(norm_values)

            status = (
                "PASS"
                if (
                    n == len(semantic_ids)
                    and finite
                    and float(np.max(np.abs(norms - 1.0))) <= 5e-4
                )
                else "FAIL"
            )

            representation_rows.append(
                {
                    "method_key": method,
                    "dataset_id": dataset_id,
                    "prompt_type": view,
                    "path": str(path),
                    "is_sparse": sparse_flag,
                    "n": int(n),
                    "dim": int(dim),
                    "finite": finite,
                    "norm_min": float(norms.min()),
                    "norm_median": float(np.median(norms)),
                    "norm_max": float(norms.max()),
                    "status": status,
                }
            )

representation_df = pd.DataFrame(representation_rows)
representation_df.to_csv(
    AUDIT_OUT / "representation_catalog.tsv",
    sep="\t",
    index=False,
)

if not representation_df["status"].eq("PASS").all():
    raise RuntimeError("representation audit failed")


# ==================================================================================================
# BGE-M3 frozen corrected P4 reproduction for 8 corrected datasets
# ==================================================================================================

step("09_bge_frozen_p4_reproduction")

bge_repro_rows = []

for dataset_id, cfg in DATASET_CONFIGS.items():
    auth = authorities[dataset_id]
    frozen_h5ad = auth["frozen_h5ad"]

    if frozen_h5ad is None:
        continue

    frozen_h5ad = Path(frozen_h5ad)
    pooled_key = auth["pooled_key"]

    new_path = SEM_OUT / "bge_m3" / dataset_id / "P4.npy"
    new_X = np.load(new_path, mmap_mode="r")

    a = ad.read_h5ad(frozen_h5ad, backed="r")
    try:
        if "_semantic_instance_id" not in a.obs.columns:
            raise RuntimeError(
                f"{dataset_id}: frozen H5AD missing _semantic_instance_id"
            )

        obs_ids = a.obs["_semantic_instance_id"].astype(str).to_numpy()
        first_idx: dict[str, int] = {}

        for i, sid in enumerate(obs_ids):
            if sid not in first_idx:
                first_idx[sid] = i

        semantic_ids = dataset_uids[dataset_id]

        missing = [sid for sid in semantic_ids if sid not in first_idx]
        if missing:
            raise RuntimeError(
                f"{dataset_id}: {len(missing)} semantic IDs missing in frozen H5AD"
            )

        pairs = sorted(
            [(first_idx[sid], j) for j, sid in enumerate(semantic_ids)],
            key=lambda x: x[0],
        )

        max_abs = 0.0
        abs_sum = 0.0
        elem_n = 0
        cosine_values = []

        for start in range(0, len(pairs), AGG_COMPONENT_CHUNK):
            local_pairs = pairs[start : start + AGG_COMPONENT_CHUNK]
            cell_idx = [x[0] for x in local_pairs]
            new_idx = [x[1] for x in local_pairs]

            frozen = np.asarray(
                a.obsm[pooled_key][cell_idx],
                dtype=np.float32,
            )
            frozen = l2_dense(frozen)

            new = np.asarray(
                new_X[new_idx],
                dtype=np.float32,
            )

            diff = np.abs(new - frozen)

            max_abs = max(max_abs, float(diff.max()))
            abs_sum += float(diff.sum())
            elem_n += int(diff.size)
            cosine_values.append(np.sum(new * frozen, axis=1))

        cos = np.concatenate(cosine_values)

    finally:
        try:
            a.file.close()
        except Exception:
            pass

    row = {
        "dataset_id": dataset_id,
        "n": len(dataset_uids[dataset_id]),
        "max_abs_diff": max_abs,
        "mean_abs_diff": abs_sum / elem_n,
        "cosine_min": float(cos.min()),
        "cosine_median": float(np.median(cos)),
        "status": (
            "PASS"
            if max_abs <= 2e-3 and float(cos.min()) >= 0.9999
            else "FAIL"
        ),
    }
    bge_repro_rows.append(row)
    log(str(row))

bge_repro_df = pd.DataFrame(bge_repro_rows)
bge_repro_df.to_csv(
    AUDIT_OUT / "bge_frozen_p4_reproduction.tsv",
    sep="\t",
    index=False,
)

if len(bge_repro_df) != 8 or not bge_repro_df["status"].eq("PASS").all():
    raise RuntimeError("BGE frozen P4 reproduction failed")


# ==================================================================================================
# Representation loader
# ==================================================================================================

def load_representation(method: str, dataset_id: str, view: str):
    if method == "metadata_onehot":
        return sparse.load_npz(
            metadata_dir / dataset_id / f"{view}.npz"
        ).tocsr()

    return np.load(
        SEM_OUT / method / dataset_id / f"{view}.npy",
        mmap_mode="r",
    )


# ==================================================================================================
# Retrieval
# ==================================================================================================

step("10_set_valued_retrieval")

def deterministic_signature_subset(signatures: list[str], max_n: int) -> list[str]:
    signatures = sorted(signatures)
    if len(signatures) <= max_n:
        return signatures

    return sorted(
        signatures,
        key=lambda x: stable_hash("retrieval_query", x, SEED, length=64),
    )[:max_n]


def rank_from_scores(
    scores: np.ndarray,
    positive_idx: np.ndarray,
    candidate_ids: np.ndarray,
) -> tuple[int, float, str]:
    sr = np.round(
        np.asarray(scores, dtype=np.float64),
        RETRIEVAL_SCORE_DECIMALS,
    )

    best_score = float(sr[positive_idx].max())

    tied_positive_ids = [
        str(candidate_ids[i])
        for i in positive_idx
        if float(sr[i]) == best_score
    ]
    first_positive_uid = min(tied_positive_ids)

    rank = (
        int(np.sum(sr > best_score))
        + int(
            np.sum(
                (sr == best_score)
                & (candidate_ids < first_positive_uid)
            )
        )
        + 1
    )

    return rank, best_score, first_positive_uid


def retrieval_dense(
    source,
    target,
    semantic_ids: list[str],
    signature_by_uid: dict[str, str],
) -> tuple[dict[str, Any], pd.DataFrame]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, uid in enumerate(semantic_ids):
        groups[signature_by_uid[uid]].append(i)

    query_signatures = deterministic_signature_subset(
        list(groups.keys()),
        RETRIEVAL_QUERY_CAP,
    )

    candidate_ids = np.asarray(semantic_ids, dtype=str)
    source_rep_idx = np.asarray(
        [groups[sig][0] for sig in query_signatures],
        dtype=np.int64,
    )

    details = []

    use_cuda = bool(torch.cuda.is_available())

    if use_cuda:
        target_t = torch.from_numpy(
            np.array(target, dtype=np.float32, copy=True)
        ).to(DEVICE)

        try:
            for start in range(0, len(query_signatures), RETRIEVAL_QUERY_BATCH):
                stop = min(
                    len(query_signatures),
                    start + RETRIEVAL_QUERY_BATCH,
                )

                q_np = np.asarray(
                    source[source_rep_idx[start:stop]],
                    dtype=np.float32,
                )

                q_t = torch.from_numpy(q_np).to(DEVICE)

                with torch.inference_mode():
                    scores_block = (
                        q_t @ target_t.T
                    ).float().cpu().numpy()

                for j, sig in enumerate(query_signatures[start:stop]):
                    positives = np.asarray(groups[sig], dtype=np.int64)

                    rank, best_score, first_uid = rank_from_scores(
                        scores_block[j],
                        positives,
                        candidate_ids,
                    )

                    details.append(
                        {
                            "query_signature": sig,
                            "representative_uid": semantic_ids[groups[sig][0]],
                            "positive_n": len(positives),
                            "first_positive_uid": first_uid,
                            "first_positive_rank": rank,
                            "reciprocal_rank": 1.0 / rank,
                            "hit_at_1": float(rank <= 1),
                            "hit_at_5": float(rank <= 5),
                            "hit_at_10": float(rank <= 10),
                            "best_positive_score": best_score,
                        }
                    )
        finally:
            del target_t
            cleanup_cuda()

    else:
        for sig in query_signatures:
            rep_idx = groups[sig][0]
            q = np.asarray(source[rep_idx], dtype=np.float32)
            scores = np.asarray(target, dtype=np.float32) @ q
            positives = np.asarray(groups[sig], dtype=np.int64)

            rank, best_score, first_uid = rank_from_scores(
                scores,
                positives,
                candidate_ids,
            )

            details.append(
                {
                    "query_signature": sig,
                    "representative_uid": semantic_ids[rep_idx],
                    "positive_n": len(positives),
                    "first_positive_uid": first_uid,
                    "first_positive_rank": rank,
                    "reciprocal_rank": 1.0 / rank,
                    "hit_at_1": float(rank <= 1),
                    "hit_at_5": float(rank <= 5),
                    "hit_at_10": float(rank <= 10),
                    "best_positive_score": best_score,
                }
            )

    d = pd.DataFrame(details)

    return (
        {
            "query_n_total": int(len(groups)),
            "query_n_evaluated": int(len(d)),
            "query_capped": bool(len(groups) > len(d)),
            "mean_positive_n": float(d["positive_n"].mean()),
            "mrr": float(d["reciprocal_rank"].mean()),
            "hit_at_1": float(d["hit_at_1"].mean()),
            "hit_at_5": float(d["hit_at_5"].mean()),
            "hit_at_10": float(d["hit_at_10"].mean()),
            "max_rank": int(d["first_positive_rank"].max()),
        },
        d,
    )


def retrieval_sparse(
    source: sparse.csr_matrix,
    target: sparse.csr_matrix,
    semantic_ids: list[str],
    signature_by_uid: dict[str, str],
) -> tuple[dict[str, Any], pd.DataFrame]:
    # Exact set-valued retrieval for sparse representations.
    #
    # scipy sparse @ sparse.T returns a sparse matrix, so the numeric score block is
    # explicitly densified with .toarray(). Batching avoids one sparse multiply per query.
    groups: dict[str, list[int]] = defaultdict(list)

    for i, uid in enumerate(semantic_ids):
        groups[signature_by_uid[uid]].append(i)

    query_signatures = deterministic_signature_subset(
        list(groups.keys()),
        RETRIEVAL_QUERY_CAP,
    )

    candidate_ids = np.asarray(semantic_ids, dtype=str)

    source_rep_idx = np.asarray(
        [groups[sig][0] for sig in query_signatures],
        dtype=np.int64,
    )

    details = []

    for start in range(0, len(query_signatures), RETRIEVAL_QUERY_BATCH):
        stop = min(
            len(query_signatures),
            start + RETRIEVAL_QUERY_BATCH,
        )

        q_block = source[source_rep_idx[start:stop]]

        # [candidate_n, query_batch] -> [query_batch, candidate_n]
        score_block = (target @ q_block.T).T.toarray()

        for j, sig in enumerate(query_signatures[start:stop]):
            positives = np.asarray(
                groups[sig],
                dtype=np.int64,
            )

            rank, best_score, first_uid = rank_from_scores(
                score_block[j],
                positives,
                candidate_ids,
            )

            details.append(
                {
                    "query_signature": sig,
                    "representative_uid": semantic_ids[groups[sig][0]],
                    "positive_n": len(positives),
                    "first_positive_uid": first_uid,
                    "first_positive_rank": rank,
                    "reciprocal_rank": 1.0 / rank,
                    "hit_at_1": float(rank <= 1),
                    "hit_at_5": float(rank <= 5),
                    "hit_at_10": float(rank <= 10),
                    "best_positive_score": best_score,
                }
            )

    d = pd.DataFrame(details)

    return (
        {
            "query_n_total": int(len(groups)),
            "query_n_evaluated": int(len(d)),
            "query_capped": bool(len(groups) > len(d)),
            "mean_positive_n": float(d["positive_n"].mean()),
            "mrr": float(d["reciprocal_rank"].mean()),
            "hit_at_1": float(d["hit_at_1"].mean()),
            "hit_at_5": float(d["hit_at_5"].mean()),
            "hit_at_10": float(d["hit_at_10"].mean()),
            "max_rank": int(d["first_positive_rank"].max()),
        },
        d,
    )


# --------------------------------------------------------------------------------------------------
# v1.1 retrieval checkpointing
# --------------------------------------------------------------------------------------------------

retrieval_checkpoint_dir = METRIC_OUT / "_retrieval_checkpoints_v1_1"
retrieval_checkpoint_dir.mkdir(parents=True, exist_ok=True)

retrieval_detail_checkpoint_dir = (
    PROV
    / "result1"
    / "full_v1"
    / "_retrieval_detail_parts_v1_1"
)
retrieval_detail_checkpoint_dir.mkdir(parents=True, exist_ok=True)

retrieval_rows = []
retrieval_detail_paths: list[Path] = []
retrieval_done_keys: set[str] = set()


def retrieval_checkpoint_key(
    method: str,
    dataset_id: str,
    source_view: str,
    target_view: str,
) -> str:
    return f"{method}__{dataset_id}__{source_view}_to_{target_view}"


# Recover only complete summary/detail checkpoint pairs.
for summary_path in sorted(
    retrieval_checkpoint_dir.glob("*.summary.json")
):
    payload = json.loads(summary_path.read_text())

    key = str(payload.get("checkpoint_key", ""))

    detail_path = (
        retrieval_detail_checkpoint_dir
        / f"{key}.detail.tsv.gz"
    )

    row = payload.get("row")

    if (
        key
        and isinstance(row, dict)
        and detail_path.is_file()
    ):
        retrieval_rows.append(row)
        retrieval_detail_paths.append(detail_path)
        retrieval_done_keys.add(key)


for method in METHOD_KEYS:
    for dataset_id in DATASET_CONFIGS:
        semantic_ids = dataset_uids[dataset_id]

        for source_view, target_view in TRANSITIONS:
            checkpoint_key = retrieval_checkpoint_key(
                method,
                dataset_id,
                source_view,
                target_view,
            )

            if checkpoint_key in retrieval_done_keys:
                log(
                    f"[retrieval-resume] {method} {dataset_id} "
                    f"{source_view}->{target_view}"
                )
                continue

            source = load_representation(
                method,
                dataset_id,
                source_view,
            )

            target = load_representation(
                method,
                dataset_id,
                target_view,
            )

            sig = signature_maps[
                (dataset_id, source_view)
            ]

            if sparse.issparse(source):
                summary, detail = retrieval_sparse(
                    source,
                    target,
                    semantic_ids,
                    sig,
                )
            else:
                summary, detail = retrieval_dense(
                    source,
                    target,
                    semantic_ids,
                    sig,
                )

            row = {
                "method_key": method,
                "dataset_id": dataset_id,
                "source_prompt": source_view,
                "target_prompt": target_view,
                **summary,
            }

            detail.insert(0, "target_prompt", target_view)
            detail.insert(0, "source_prompt", source_view)
            detail.insert(0, "dataset_id", dataset_id)
            detail.insert(0, "method_key", method)

            summary_path = (
                retrieval_checkpoint_dir
                / f"{checkpoint_key}.summary.json"
            )

            detail_path = (
                retrieval_detail_checkpoint_dir
                / f"{checkpoint_key}.detail.tsv.gz"
            )

            # Detail is written first; summary JSON acts as the completion marker.
            detail.to_csv(
                detail_path,
                sep="	",
                index=False,
                compression="gzip",
            )

            write_json(
                {
                    "checkpoint_key": checkpoint_key,
                    "row": row,
                    "detail_path_relative_to_work": str(
                        detail_path.relative_to(WORK)
                    ),
                },
                summary_path,
            )

            retrieval_rows.append(row)
            retrieval_detail_paths.append(detail_path)
            retrieval_done_keys.add(checkpoint_key)

            pd.DataFrame(retrieval_rows).sort_values(
                [
                    "method_key",
                    "dataset_id",
                    "source_prompt",
                    "target_prompt",
                ]
            ).to_csv(
                METRIC_OUT
                / "result1_retrieval_by_dataset.partial.tsv",
                sep="	",
                index=False,
            )

            log(
                f"[retrieval] {method} {dataset_id} "
                f"{source_view}->{target_view} "
                f"q={summary['query_n_evaluated']}/"
                f"{summary['query_n_total']} "
                f"mrr={summary['mrr']:.6f}"
            )


retrieval_df = (
    pd.DataFrame(retrieval_rows)
    .drop_duplicates(
        [
            "method_key",
            "dataset_id",
            "source_prompt",
            "target_prompt",
        ],
        keep="last",
    )
    .sort_values(
        [
            "method_key",
            "dataset_id",
            "source_prompt",
            "target_prompt",
        ]
    )
    .reset_index(drop=True)
)

retrieval_df.to_csv(
    METRIC_OUT / "result1_retrieval_by_dataset.tsv",
    sep="	",
    index=False,
)

detail_frames = []

for detail_path in sorted(
    set(retrieval_detail_paths)
):
    detail_frames.append(
        pd.read_csv(
            detail_path,
            sep="	",
        )
    )

retrieval_detail_df = pd.concat(
    detail_frames,
    ignore_index=True,
)

retrieval_detail_df.to_csv(
    PROV
    / "result1"
    / "full_v1"
    / "result1_retrieval_query_detail.tsv.gz",
    sep="	",
    index=False,
    compression="gzip",
)


# ==================================================================================================
# Classification plan: entity-grouped, donor excluded
# ==================================================================================================

step("11_entity_grouped_classification")

def deterministic_balanced_sample(
    frame: pd.DataFrame,
    label: str,
    max_n: int,
) -> pd.DataFrame:
    if len(frame) <= max_n:
        return frame.copy()

    classes = sorted(frame[label].astype(str).unique())
    per_class = max(1, math.ceil(max_n / len(classes)))

    pieces = []
    for cls in classes:
        x = frame[frame[label].astype(str).eq(cls)].copy()
        x["_hash"] = x["semantic_instance_id"].astype(str).map(
            lambda uid: stable_hash(
                "classification_sample",
                label,
                cls,
                uid,
                SEED,
                length=64,
            )
        )
        pieces.append(
            x.sort_values("_hash")
            .head(per_class)
            .drop(columns="_hash")
        )

    out = pd.concat(pieces, ignore_index=True)

    if len(out) > max_n:
        out["_hash"] = out["semantic_instance_id"].astype(str).map(
            lambda uid: stable_hash(
                "classification_global",
                label,
                uid,
                SEED,
                length=64,
            )
        )
        out = (
            out.sort_values("_hash")
            .head(max_n)
            .drop(columns="_hash")
        )

    return out.reset_index(drop=True)


classification_plans: dict[tuple[str, str, str], dict[str, Any]] = {}
plan_rows = []
fold_audit_rows = []

for dataset_id, meta0 in dataset_meta.items():
    uid_to_row = {
        uid: i
        for i, uid in enumerate(dataset_uids[dataset_id])
    }

    for label, views in LABEL_VIEWS.items():
        m0 = meta0.copy()
        m0[label] = m0[label].astype(str).map(clean_value)
        m0 = m0[m0[label].map(supported_value)].copy()

        counts = m0[label].value_counts()
        supported_classes = counts[counts >= MIN_CLASS_COUNT].index
        m0 = m0[m0[label].isin(supported_classes)].copy()

        if m0[label].nunique() < 2:
            continue

        entity_support = (
            m0.groupby(label, observed=True)["entity_key"].nunique()
        )

        supported_by_group = entity_support[entity_support >= 2].index
        m0 = m0[m0[label].isin(supported_by_group)].copy()

        if m0[label].nunique() < 2:
            continue

        m0 = deterministic_balanced_sample(
            m0,
            label,
            CLASSIFICATION_ROW_CAP,
        )

        group_support = (
            m0.groupby(label, observed=True)["entity_key"].nunique()
        )

        n_splits = int(min(CLASSIFICATION_FOLDS, group_support.min()))
        if n_splits < 2:
            continue

        y = m0[label].astype(str).to_numpy()
        groups = m0["entity_key"].astype(str).to_numpy()

        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=SEED,
        )

        folds = []
        for fold_i, (tr, te) in enumerate(
            splitter.split(np.zeros(len(m0)), y, groups),
            start=1,
        ):
            overlap = set(groups[tr]) & set(groups[te])
            if overlap:
                raise RuntimeError(
                    f"{dataset_id} {label}: entity leakage in fold {fold_i}"
                )

            folds.append((tr, te))

            fold_audit_rows.append(
                {
                    "dataset_id": dataset_id,
                    "label": label,
                    "fold": fold_i,
                    "train_n": len(tr),
                    "test_n": len(te),
                    "train_entity_group_n": len(set(groups[tr])),
                    "test_entity_group_n": len(set(groups[te])),
                    "entity_group_overlap_n": len(overlap),
                }
            )

        semantic_rows = np.asarray(
            [
                uid_to_row[uid]
                for uid in m0["semantic_instance_id"].astype(str)
            ],
            dtype=np.int64,
        )

        for view in views:
            classification_plans[(dataset_id, view, label)] = {
                "meta": m0,
                "semantic_rows": semantic_rows,
                "y": y,
                "groups": groups,
                "folds": folds,
                "n_splits": n_splits,
            }

            plan_rows.append(
                {
                    "dataset_id": dataset_id,
                    "prompt_type": view,
                    "label": label,
                    "n": len(m0),
                    "n_classes": int(m0[label].nunique()),
                    "n_entity_groups": int(m0["entity_key"].nunique()),
                    "n_folds": n_splits,
                    "status": "READY",
                }
            )

plan_df = pd.DataFrame(plan_rows)
plan_df.to_csv(
    AUDIT_OUT / "classification_plan.tsv",
    sep="\t",
    index=False,
)

fold_audit_df = pd.DataFrame(fold_audit_rows)
fold_audit_df.to_csv(
    AUDIT_OUT / "classification_fold_leakage_audit.tsv",
    sep="\t",
    index=False,
)

if len(fold_audit_df) and not fold_audit_df["entity_group_overlap_n"].eq(0).all():
    raise RuntimeError("classification entity-group leakage")


classification_rows = []

for method in METHOD_KEYS:
    by_dataset_view: dict[tuple[str, str], list[tuple[str, dict[str, Any]]]] = defaultdict(list)

    for (dataset_id, view, label), plan in classification_plans.items():
        by_dataset_view[(dataset_id, view)].append((label, plan))

    for (dataset_id, view), tasks in by_dataset_view.items():
        Xfull = load_representation(method, dataset_id, view)

        for label, plan in tasks:
            rows = plan["semantic_rows"]

            if sparse.issparse(Xfull):
                z = Xfull[rows]
            else:
                z = np.asarray(Xfull[rows], dtype=np.float32)

            y = plan["y"]

            y_true = []
            y_pred = []

            for tr, te in plan["folds"]:
                k = int(min(CLASSIFICATION_K, len(tr)))

                clf = KNeighborsClassifier(
                    n_neighbors=k,
                    metric="cosine",
                    algorithm="brute",
                    weights="uniform",
                    n_jobs=-1,
                )

                clf.fit(z[tr], y[tr])
                pred = clf.predict(z[te])

                y_true.extend(y[te].tolist())
                y_pred.extend(pred.tolist())

            classification_rows.append(
                {
                    "method_key": method,
                    "dataset_id": dataset_id,
                    "prompt_type": view,
                    "label": label,
                    "cv_group": "entity_key",
                    "n": int(len(y)),
                    "n_classes": int(len(set(y))),
                    "n_entity_groups": int(len(set(plan["groups"]))),
                    "n_folds": int(plan["n_splits"]),
                    "macro_f1": float(
                        f1_score(y_true, y_pred, average="macro")
                    ),
                    "balanced_accuracy": float(
                        balanced_accuracy_score(y_true, y_pred)
                    ),
                    "status": "PASS",
                }
            )

            log(
                f"[classify] {method} {dataset_id} {view} {label} "
                f"n={len(y)} bal_acc={classification_rows[-1]['balanced_accuracy']:.6f}"
            )

classification_df = pd.DataFrame(classification_rows)
classification_df.to_csv(
    METRIC_OUT / "result1_classification_by_dataset.tsv",
    sep="\t",
    index=False,
)


# ==================================================================================================
# Summaries: dataset macro only
# ==================================================================================================

step("12_dataset_macro_summaries")

retrieval_macro = (
    retrieval_df.groupby(
        ["method_key", "source_prompt", "target_prompt"],
        observed=True,
        as_index=False,
    )
    .agg(
        dataset_n=("dataset_id", "nunique"),
        mrr_macro_mean=("mrr", "mean"),
        hit_at_1_macro_mean=("hit_at_1", "mean"),
        hit_at_5_macro_mean=("hit_at_5", "mean"),
        hit_at_10_macro_mean=("hit_at_10", "mean"),
    )
)

retrieval_macro.to_csv(
    SUMMARY_OUT / "result1_retrieval_macro.tsv",
    sep="\t",
    index=False,
)

classification_macro = (
    classification_df.groupby(
        ["method_key", "prompt_type", "label"],
        observed=True,
        as_index=False,
    )
    .agg(
        dataset_n=("dataset_id", "nunique"),
        macro_f1_dataset_macro_mean=("macro_f1", "mean"),
        balanced_accuracy_dataset_macro_mean=("balanced_accuracy", "mean"),
    )
)

classification_macro.to_csv(
    SUMMARY_OUT / "result1_classification_macro.tsv",
    sep="\t",
    index=False,
)

# Paired prompt-view deltas, always paired within dataset/method/label.
paired_rows = []

for method in METHOD_KEYS:
    sub_m = classification_df[classification_df["method_key"].eq(method)]

    for dataset_id in DATASET_CONFIGS:
        sub_d = sub_m[sub_m["dataset_id"].eq(dataset_id)]

        for label in LABEL_VIEWS:
            sub_l = sub_d[sub_d["label"].eq(label)].set_index("prompt_type")

            for source_view in ("P1", "P2", "P3"):
                if source_view in sub_l.index and "P4" in sub_l.index:
                    paired_rows.append(
                        {
                            "method_key": method,
                            "dataset_id": dataset_id,
                            "label": label,
                            "source_prompt": source_view,
                            "target_prompt": "P4",
                            "delta_balanced_accuracy": float(
                                sub_l.loc["P4", "balanced_accuracy"]
                                - sub_l.loc[source_view, "balanced_accuracy"]
                            ),
                            "delta_macro_f1": float(
                                sub_l.loc["P4", "macro_f1"]
                                - sub_l.loc[source_view, "macro_f1"]
                            ),
                        }
                    )

paired_df = pd.DataFrame(paired_rows)
paired_df.to_csv(
    SUMMARY_OUT / "result1_classification_paired_view_deltas.tsv",
    sep="\t",
    index=False,
)

if len(paired_df):
    paired_macro = (
        paired_df.groupby(
            ["method_key", "label", "source_prompt", "target_prompt"],
            observed=True,
            as_index=False,
        )
        .agg(
            dataset_n=("dataset_id", "nunique"),
            delta_balanced_accuracy_macro_mean=("delta_balanced_accuracy", "mean"),
            delta_macro_f1_macro_mean=("delta_macro_f1", "mean"),
        )
    )
else:
    paired_macro = pd.DataFrame()

paired_macro.to_csv(
    SUMMARY_OUT / "result1_classification_paired_view_deltas_macro.tsv",
    sep="\t",
    index=False,
)


# ==================================================================================================
# Final audit
# ==================================================================================================

step("13_final_audit")

expected_dataset_n = 9
expected_method_n = 9
expected_rep_rows = expected_dataset_n * expected_method_n * len(VIEWS)
expected_retrieval_rows = expected_dataset_n * expected_method_n * len(TRANSITIONS)

checks = {
    "dataset_n_9": int(authority_df["dataset_id"].nunique()) == expected_dataset_n,
    "authority_all_pass": bool(authority_df["status"].eq("PASS").all()),
    "semantic_instance_total_186319": (
        int(authority_df["semantic_instance_n"].sum()) == 186319
    ),
    "representation_row_count": len(representation_df) == expected_rep_rows,
    "representation_all_pass": bool(representation_df["status"].eq("PASS").all()),
    "bge_reproduction_8_of_8_pass": (
        len(bge_repro_df) == 8
        and bool(bge_repro_df["status"].eq("PASS").all())
    ),
    "metadata_visibility_all_pass": bool(visibility_df["status"].eq("PASS").all()),
    "retrieval_row_count": len(retrieval_df) == expected_retrieval_rows,
    "retrieval_all_finite": bool(
        np.isfinite(
            retrieval_df[
                ["mrr", "hit_at_1", "hit_at_5", "hit_at_10"]
            ].to_numpy(dtype=float)
        ).all()
        and (retrieval_df["query_n_evaluated"] > 0).all()
    ),
    "classification_nonempty": len(classification_df) > 0,
    "classification_all_pass": bool(
        len(classification_df) > 0
        and classification_df["status"].eq("PASS").all()
    ),
    "classification_entity_grouped_only": bool(
        len(classification_df) > 0
        and classification_df["cv_group"].astype(str).eq("entity_key").all()
    ),
    "classification_fold_leakage_zero": bool(
        len(fold_audit_df) > 0
        and fold_audit_df["entity_group_overlap_n"].eq(0).all()
    ),
    "donor_not_primary_label": bool(
        not classification_df["label"].astype(str).eq("donor").any()
    ),
    "retrieval_dataset_macro_summary_nonempty": len(retrieval_macro) > 0,
    "classification_dataset_macro_summary_nonempty": len(classification_macro) > 0,
}

overall_pass = all(checks.values())

audit_lines = [
    "PERTURBCONTEXTALIGN — RESULT 1 CORRECTED FULL 9-DATASET AUDIT v1",
    "=" * 120,
    "",
    f"root = {ROOT}",
    f"work = {WORK}",
    f"device = {DEVICE}",
    f"python = {sys.version.split()[0]}",
    f"platform = {platform.platform()}",
    f"torch = {torch.__version__}",
    f"text_catalog_logical_sha256 = {logical_text_catalog_sha}",
    f"unique_component_text_n = {len(all_texts)}",
    f"dataset_n = {authority_df['dataset_id'].nunique()}",
    f"semantic_instance_total = {int(authority_df['semantic_instance_n'].sum())}",
    f"method_n = {len(METHOD_KEYS)}",
    f"methods = {','.join(METHOD_KEYS)}",
    f"retrieval_query_cap = {RETRIEVAL_QUERY_CAP}",
    f"classification_row_cap = {CLASSIFICATION_ROW_CAP}",
    "",
    "DATASET AUTHORITY",
    "-" * 120,
    authority_df[
        [
            "dataset_id",
            "semantic_instance_n",
            "condition_n",
            "component_prompt_n",
            "unique_p1_signature_n",
            "unique_p2_signature_n",
            "unique_p3_signature_n",
            "unique_p4_signature_n",
            "prompt_hash_status",
            "condition_hash_status",
            "status",
        ]
    ].to_string(index=False),
    "",
    "BGE FROZEN P4 REPRODUCTION",
    "-" * 120,
    bge_repro_df.to_string(index=False),
    "",
    "RETRIEVAL MACRO",
    "-" * 120,
    retrieval_macro.to_string(index=False),
    "",
    "CLASSIFICATION MACRO",
    "-" * 120,
    classification_macro.to_string(index=False),
    "",
    "FINAL CHECKS",
    "-" * 120,
]

for k, v in checks.items():
    audit_lines.append(f"{k} = {v}")

audit_lines += [
    "",
    "=" * 120,
    "R1 FULL CORRECTED 9-DATASET RUN = " + ("PASS" if overall_pass else "REVIEW"),
    "=" * 120,
]

audit_text = "\n".join(audit_lines) + "\n"

final_audit_path = AUDIT_OUT / "R1_FULL_CORRECTED_AUDIT_v1.txt"
final_audit_path.write_text(audit_text, encoding="utf-8")

log(audit_text)

if not overall_pass:
    raise SystemExit(2)


# ==================================================================================================
# Release staging: corrected lineage only
# ==================================================================================================

step("14_release_staging")

# Stage public GitHub code candidate.
shutil.copy2(
    Path(__file__),
    GH / "scripts" / "result1" / "r1_full_pipeline.py",
)

if ARGS.launcher:
    launcher = Path(ARGS.launcher)
    if launcher.is_file():
        shutil.copy2(
            launcher,
            GH / "scripts" / "result1" / "run_r1_full.sh",
        )

# Public-friendly compact Result 1 tables.
public_tables = [
    AUTH_OUT / "R1_DATASET_AUTHORITY.tsv",
    METRIC_OUT / "result1_retrieval_by_dataset.tsv",
    METRIC_OUT / "result1_classification_by_dataset.tsv",
    SUMMARY_OUT / "result1_retrieval_macro.tsv",
    SUMMARY_OUT / "result1_classification_macro.tsv",
    SUMMARY_OUT / "result1_classification_paired_view_deltas.tsv",
    SUMMARY_OUT / "result1_classification_paired_view_deltas_macro.tsv",
    AUDIT_OUT / "bge_frozen_p4_reproduction.tsv",
    AUDIT_OUT / "metadata_field_visibility_audit.tsv",
]

for src in public_tables:
    shutil.copy2(
        src,
        FIG / "result1" / "final_tables" / src.name,
    )

# Internal provenance.
for src in [
    final_audit_path,
    AUDIT_OUT / "representation_catalog.tsv",
    AUDIT_OUT / "classification_plan.tsv",
    AUDIT_OUT / "classification_fold_leakage_audit.tsv",
]:
    shutil.copy2(
        src,
        PROV / "result1" / "full_v1" / src.name,
    )

# Sanitized public manifest: no local absolute paths.
model_public = []
for spec in MODEL_SPECS:
    model_public.append(
        {
            "model_key": spec["model_key"],
            "display_name": spec["display_name"],
            "group": spec["group"],
            "model_id": spec["model_id"],
            "backend": spec["backend"],
            "pooling": spec["pooling"],
            "dimension": spec["expected_dim"],
            "max_length": MAX_LENGTH,
            "normalized": True,
        }
    )

for spec in CONTROL_SPECS:
    model_public.append(
        {
            "model_key": spec["model_key"],
            "display_name": spec["display_name"],
            "group": spec["group"],
        }
    )

table_checksums = {}
for src in public_tables:
    table_checksums[src.name] = sha256_file(
        FIG / "result1" / "final_tables" / src.name
    )

public_manifest = {
    "project": "PerturbContextAlign",
    "result": "Result 1 — context representation retention",
    "analysis_version": "corrected_full_v1",
    "status": "PASS",
    "seed": SEED,
    "dataset_ids": list(DATASET_CONFIGS.keys()),
    "semantic_instance_total": int(authority_df["semantic_instance_n"].sum()),
    "prompt_views": list(VIEWS),
    "prompt_transitions": [list(x) for x in TRANSITIONS],
    "component_pooling": "masked mean over component embeddings, followed by L2 normalization",
    "donor_in_prompt": False,
    "classification": {
        "classifier": "5-nearest neighbors",
        "metric": "cosine",
        "cross_validation": "StratifiedGroupKFold",
        "group_key": "family-aware perturbation entity",
        "folds_max": CLASSIFICATION_FOLDS,
        "row_cap_per_dataset_label": CLASSIFICATION_ROW_CAP,
        "dataset_summary": "macro average across eligible datasets",
    },
    "retrieval": {
        "query_unit": "unique source-prompt equivalence group",
        "positive_set": "all same-dataset P4 semantic instances consistent with source prompt",
        "candidate_pool": "all non-control P4 semantic instances in the same dataset",
        "metrics": ["MRR", "Hit@1", "Hit@5", "Hit@10"],
        "query_cap_per_dataset_transition": RETRIEVAL_QUERY_CAP,
        "tie_policy": f"score rounded to {RETRIEVAL_SCORE_DECIMALS} decimals, then semantic_instance_id ascending",
        "dataset_summary": "macro average across datasets",
    },
    "models": model_public,
    "public_table_sha256": table_checksums,
    "note": "Absolute machine paths and raw H5AD files are intentionally excluded from this public manifest.",
}

public_manifest_path = FIG / "manifests" / "R1_PUBLIC_MANIFEST_v1.json"
write_json(public_manifest, public_manifest_path)

shutil.copy2(
    public_manifest_path,
    GH / "manifests" / "R1_PUBLIC_MANIFEST_v1.json",
)

readme = """# Result 1 — corrected full run

This directory contains release candidates for PerturbContextAlign Result 1.

## Analysis unit

The corrected semantic instance is perturbation condition × P4 biological context × P4 technical context.

Donor is retained in the standardized schema but is not exposed in P1–P4 prompt text.

## Prompt views

- P1: intervention only
- P2: intervention + biological context
- P3: intervention + technical context + dose + duration
- P4: intervention + biological context + technical context + dose + duration

P1–P3 are exact sentence-section ablations of the frozen corrected P4 prompt. P4 wording is not rewritten.

## Component perturbations

Each perturbation component is encoded separately. Semantic-instance embeddings are the masked mean of component embeddings followed by L2 normalization.

## Evaluation

Retrieval is set-valued because reduced prompt views may correspond to multiple P4 semantic instances.

Field retention uses entity-grouped cross-validation to prevent repeated use of the same perturbation entity across contexts/exposures from leaking between folds.

Cross-dataset summaries are dataset-macro summaries; large datasets do not receive larger weight solely because they contain more semantic instances.

## Release state

This is a staging candidate. Final GitHub/Figshare publication still requires the project-level release gate after Results 1–4 and manuscript/figure/source-data reconciliation.
"""

(GH / "docs" / "RESULT1_METHOD_AND_RELEASE_NOTES.md").write_text(
    readme,
    encoding="utf-8",
)

(FIG / "result1" / "README.md").write_text(
    readme,
    encoding="utf-8",
)

# Refresh staging checksums.
staging_hashes = []
for p in sorted(RELEASE.rglob("*")):
    if not p.is_file():
        continue
    if p.name == "STAGING_SHA256SUMS.txt":
        continue
    staging_hashes.append(
        f"{sha256_file(p)}  {p.relative_to(RELEASE)}"
    )

(RELEASE / "STAGING_SHA256SUMS.txt").write_text(
    "\n".join(staging_hashes) + "\n",
    encoding="utf-8",
)

# Compact handoff.
handoff = f"""# PerturbContextAlign Result 1 corrected full-run handoff

Status: **PASS**

- Datasets: {len(DATASET_CONFIGS)}
- Corrected semantic instances: {int(authority_df["semantic_instance_n"].sum())}
- Methods: {len(METHOD_KEYS)}
- Prompt views: P1–P4
- Retrieval: set-valued, same-dataset candidate pool, deterministic cap {RETRIEVAL_QUERY_CAP}
- Classification: 5-NN cosine, entity-grouped CV, donor excluded
- Cross-dataset aggregation: dataset macro
- BGE frozen P4 reproduction: 8/8 PASS
- ComboSciPlex current P4 authority: PASS

Primary outputs:

- `05_metrics/result1_retrieval_by_dataset.tsv`
- `05_metrics/result1_classification_by_dataset.tsv`
- `06_summary/result1_retrieval_macro.tsv`
- `06_summary/result1_classification_macro.tsv`
- `06_summary/result1_classification_paired_view_deltas_macro.tsv`
- `07_audit/R1_FULL_CORRECTED_AUDIT_v1.txt`

Release staging is under `08_release_staging/`.

Do not substitute pre-correction broad Result 1 numerical outputs for this corrected lineage.
"""

(WORK / "01_result1_retention" / "R1_FINAL_HANDOFF_v1.md").write_text(
    handoff,
    encoding="utf-8",
)

log("")
log("R1_RELEASE_STAGING=PASS")
log(f"FINAL_AUDIT={final_audit_path}")
log(
    "HANDOFF="
    + str(WORK / "01_result1_retention" / "R1_FINAL_HANDOFF_v1.md")
)
log(f"FIGSHARE_STAGE={FIG / 'result1'}")
log(f"GITHUB_STAGE={GH / 'scripts' / 'result1'}")
