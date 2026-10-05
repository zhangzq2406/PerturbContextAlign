from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


# ======================================================================================
# CLI / constants
# ======================================================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "PerturbContextAlign R2 primary paper closure: corrected simple "
            "references, matched-Gaussian-random NDCG gain, paired language-encoder "
            "closure, rank concordance, and Fig.1d-ready source data."
        )
    )
    ap.add_argument(
        "--root",
        default=".",
    )
    return ap.parse_args()


ARGS = parse_args()

ROOT = Path(ARGS.root).expanduser().resolve()

NEW = (
    ROOT
    / "0_ProjectCodeAndLogs"
    / "PerturbContextAlign_R2_corrected_v1_20261002"
)

SEED = 20260711
BOOTSTRAP_REPEATS = 10000
TFIDF_MAX_FEATURES = 20000
BASELINE_DIM = 256

MODEL_KEYS = [
    "bge_m3",
    "qwen3_0_6b",
    "sapbert",
    "biomedbert",
    "medcpt_article",
    "medcpt_query",
]

MODEL_SHORT = {
    "bge_m3": "BGE",
    "qwen3_0_6b": "Qwn",
    "sapbert": "Sap",
    "biomedbert": "Bio",
    "medcpt_article": "Art",
    "medcpt_query": "Qry",
    "tfidf_svd": "TF",
    "metadata_onehot": "OH",
    "random": "RND",
}

MODEL_DISPLAY = {
    "bge_m3": "BGE-M3",
    "qwen3_0_6b": "Qwen3-Embedding-0.6B",
    "sapbert": "SapBERT",
    "biomedbert": "BiomedBERT",
    "medcpt_article": "MedCPT Article",
    "medcpt_query": "MedCPT Query",
    "tfidf_svd": "TF-IDF + SVD",
    "metadata_onehot": "Structured one-hot",
    "random": "Random",
}

MODEL_GROUP = {
    "bge_m3": "general_purpose",
    "qwen3_0_6b": "general_purpose",
    "sapbert": "biomedical",
    "biomedbert": "biomedical",
    "medcpt_article": "biomedical",
    "medcpt_query": "biomedical",
    "tfidf_svd": "lexical_control",
    "metadata_onehot": "structured_control",
    "random": "random_control",
}

DATASET_ORDER = [
    "norman_2019",
    "replogle_k562_essential",
    "replogle_rpe1",
    "tian_activation",
    "tian_inhibition",
    "srivatsan_sciplex3",
    "mcfarland_2020",
    "kaggle_cross_patient",
    "combo_sciplex",
]

DATASET_DISPLAY = {
    "norman_2019": "Norman",
    "replogle_k562_essential": "Rep. K562",
    "replogle_rpe1": "Rep. RPE1",
    "tian_activation": "Tian CRISPRa",
    "tian_inhibition": "Tian CRISPRi",
    "srivatsan_sciplex3": "sci-Plex",
    "mcfarland_2020": "McFarland",
    "kaggle_cross_patient": "Kaggle",
    "combo_sciplex": "ComboSciPlex",
}

INTERVENTION_FIELDS = ["family", "mode", "entity", "combo_n"]
BIO_FIELDS = ["species", "cell_type", "cell_line", "tissue", "disease"]
EXPOSURE_FIELDS = ["dose", "duration"]

# Corrected response-aligned broad views: technical context is intentionally absent.
VISIBLE_FIELDS = {
    "P1": INTERVENTION_FIELDS,
    "P2": INTERVENTION_FIELDS + BIO_FIELDS,
    "P3": INTERVENTION_FIELDS + EXPOSURE_FIELDS,
    "P4": INTERVENTION_FIELDS + BIO_FIELDS + EXPOSURE_FIELDS,
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
    "__na__",
    "nan",
    "none",
    "null",
    "<na>",
}

TECH_LABEL = " Technical context:"
BIO_LABEL = " Biological context:"
DOSE_LABEL = " Dose:"
DURATION_LABEL = " Duration:"


# ======================================================================================
# Paths
# ======================================================================================

PRIMARY_ROOT = NEW / "06_alignment_metrics" / "01_primary_v1"
PRIMARY_BY_DATASET = PRIMARY_ROOT / "R2_PRIMARY_ALIGNMENT_BY_DATASET_v1.tsv"
PRIMARY_AUDIT = PRIMARY_ROOT / "R2_PRIMARY_ALIGNMENT_AUDIT_v1.txt"
PRIMARY_MANIFEST = PRIMARY_ROOT / "R2_PRIMARY_ALIGNMENT_MANIFEST_v1.json"

RESP_ROOT = NEW / "04_response_geometry" / "02_similarity_v1"
RESP_AUDIT = RESP_ROOT / "R2_RESPONSE_SIMILARITY_AUDIT_v1.tsv"
RESP_MANIFEST = RESP_ROOT / "R2_RESPONSE_SIMILARITY_MANIFEST_v1.json"

TEXT_ROOT = NEW / "05_response_aligned_text"
COMPONENT_PROMPTS = TEXT_ROOT / "R2_RESPONSE_ALIGNED_COMPONENT_PROMPTS_v1.tsv.gz"
TEXT_CATALOG = TEXT_ROOT / "R2_RESPONSE_ALIGNED_TEXT_CATALOG_v1.tsv.gz"
TEXT_AUTHORITY = TEXT_ROOT / "R2_RESPONSE_ALIGNED_TEXT_AUTHORITY_v1.tsv"

ATOM_AUTHORITY = (
    NEW
    / "02_input_authority"
    / "R2_RESPONSE_ATOM_AUTHORITY_v1.tsv"
)

HIST_CONFIG = (
    NEW
    / "01_contract"
    / "R2_HISTORICAL_EFFECTIVE_CONFIG_v1.json"
)

OUT = NEW / "06_alignment_metrics" / "02_paper_closure_v2"
OUT.mkdir(parents=True, exist_ok=True)

BASELINE_ROOT = OUT / "baselines"
BASELINE_ROOT.mkdir(parents=True, exist_ok=True)

VIEW_COMPONENT_TSV = OUT / "R2_RESPONSE_ALIGNED_P1P4_COMPONENT_TEXTS_v2.tsv.gz"
VIEW_TEXT_CATALOG_TSV = OUT / "R2_RESPONSE_ALIGNED_P1P4_TEXT_CATALOG_v2.tsv.gz"

BASELINE_BY_DATASET_TSV = OUT / "R2_BASELINE_ALIGNMENT_BY_DATASET_v2.tsv"
ALL_METHOD_BY_DATASET_TSV = OUT / "R2_ALL_METHOD_ALIGNMENT_BY_DATASET_v2.tsv"
ALL_METHOD_SUMMARY_TSV = OUT / "R2_ALL_METHOD_ALIGNMENT_SUMMARY_v2.tsv"

TFIDF_SENS_TSV = OUT / "R2_TFIDF_SPEC_SENSITIVITY_v2.tsv"
PAIRWISE_TSV = OUT / "R2_LANGUAGE_PAIRWISE_CLOSURE_v2.tsv"
RANK_BY_DATASET_TSV = OUT / "R2_LANGUAGE_RANK_CONCORDANCE_BY_DATASET_v2.tsv"
MODEL_STABILITY_TSV = OUT / "R2_LANGUAGE_MODEL_STABILITY_v2.tsv"
GROUP_BY_DATASET_TSV = OUT / "R2_MODEL_GROUP_DESCRIPTIVE_BY_DATASET_v2.tsv"
GROUP_SUMMARY_TSV = OUT / "R2_MODEL_GROUP_DESCRIPTIVE_SUMMARY_v2.tsv"

FIG1D_RSA_TSV = OUT / "R2_FIG1D_RSA_MATRIX_v2.tsv"
FIG1D_GAIN_TSV = OUT / "R2_FIG1D_NDCG_GAIN_MATCHED_RANDOM_MATRIX_v2.tsv"
FIG1D_RAW_NDCG_TSV = OUT / "R2_FIG1D_RAW_NDCG_MATRIX_v2.tsv"
UNIFORM_EXCESS_TSV = OUT / "R2_EXACT_UNIFORM_EXCESS_NDCG_SENSITIVITY_MATRIX_v2.tsv"

METHODS_MD = OUT / "R2_PRIMARY_PAPER_CLOSURE_METHODS_v2.md"
AUDIT_TXT = OUT / "R2_PRIMARY_PAPER_CLOSURE_AUDIT_v2.txt"
MANIFEST_JSON = OUT / "R2_PRIMARY_PAPER_CLOSURE_MANIFEST_v2.json"


# ======================================================================================
# Helpers
# ======================================================================================

def log(msg: str) -> None:
    print(msg, flush=True)


def sha256_file(path: Path, chunk: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def stable_hash(*parts: Any, length: int = 32) -> str:
    payload = "\x1f".join(str(x) for x in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def stable_seed(*parts: Any) -> int:
    raw = "|".join(map(str, parts)).encode("utf-8")
    digest = hashlib.sha256(raw).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def write_json(obj: Any, path: Path) -> None:
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


def l2_dense(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float32)
    denom = np.linalg.norm(a, axis=1, keepdims=True)
    if np.any(~np.isfinite(denom)):
        raise RuntimeError("Non-finite embedding norm.")
    if np.any(denom <= 1e-12):
        bad = int((denom <= 1e-12).sum())
        raise RuntimeError(f"Zero-norm embedding rows={bad}")
    return a / denom


def parse_response_aligned_p4(text: str) -> dict[str, str]:
    """
    Corrected R2 response-aligned P4 no longer contains Technical context.
    Derive P1–P4 using the same view semantics while excluding technical fields.
    """
    text = str(text).strip()

    if TECH_LABEL in text:
        raise ValueError("Technical context unexpectedly present in response-aligned P4.")

    p_bio = text.find(BIO_LABEL)
    p_dose = text.find(DOSE_LABEL)
    p_duration = text.find(DURATION_LABEL)

    if min(p_bio, p_dose, p_duration) < 0:
        raise ValueError(f"Missing response-aligned P4 section: {text[:300]}")

    if not (p_bio < p_dose < p_duration):
        raise ValueError(f"Invalid response-aligned P4 section order: {text[:300]}")

    prefix = text[:p_bio].strip()
    bio = text[p_bio:p_dose].strip()
    dose = text[p_dose:p_duration].strip()
    duration = text[p_duration:].strip()

    if not all([prefix, bio, dose, duration]):
        raise ValueError(f"Blank response-aligned P4 section: {text[:300]}")

    return {
        "P1": prefix,
        "P2": f"{prefix} {bio}",
        "P3": f"{prefix} {dose} {duration}",
        "P4": text,
    }


def load_similarity_bundle(path: Path):
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


def upper_vec(S: np.ndarray) -> np.ndarray:
    A = np.asarray(S, dtype=np.float64)
    idx = np.triu_indices(A.shape[0], k=1)
    return A[idx]


def matrix_qc(S: np.ndarray) -> dict[str, Any]:
    A = np.asarray(S, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Similarity matrix must be square, got {A.shape}")
    upper = upper_vec(A)
    finite = upper[np.isfinite(upper)]
    return {
        "n_atoms": int(A.shape[0]),
        "n_pairs": int(len(upper)),
        "finite_fraction": float(len(finite) / len(upper)) if len(upper) else 1.0,
        "upper_sd": float(np.std(finite)) if len(finite) else np.nan,
        "is_constant": bool(len(finite) < 2 or np.ptp(finite) <= 1e-12),
    }


def safe_corr(
    x: np.ndarray,
    y: np.ndarray,
    method: str = "spearman",
) -> tuple[float, str]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]

    if len(x) < 3:
        return np.nan, "invalid_too_few_pairs"
    if np.ptp(x) <= 1e-12:
        return np.nan, "invalid_constant_representation"
    if np.ptp(y) <= 1e-12:
        return np.nan, "invalid_constant_response"

    if method == "spearman":
        x = rankdata(x)
        y = rankdata(y)
    elif method != "pearson":
        raise ValueError(method)

    x -= x.mean()
    y -= y.mean()

    denom = math.sqrt(float(np.dot(x, x)) * float(np.dot(y, y)))

    if denom <= 0 or not np.isfinite(denom):
        return np.nan, "invalid_zero_variance"

    return float(np.dot(x, y) / denom), "ok"


def random_tie_ranking(
    S: np.ndarray,
    i: int,
    rng: np.random.Generator,
) -> np.ndarray:
    row = np.asarray(S[i], dtype=np.float64).copy()
    row[~np.isfinite(row)] = -np.inf
    row[i] = -np.inf

    order = np.lexsort(
        (
            rng.random(len(row)),
            -row,
        )
    )

    return order[order != i]


def retrieval_metrics(
    representation_S: np.ndarray,
    response_S: np.ndarray,
    gold_k: int,
    tie_repeats: int,
    seed: int,
) -> dict[str, float | int | str]:
    n = int(representation_S.shape[0])

    if representation_S.shape != response_S.shape:
        raise ValueError("Representation/response matrix shape mismatch.")

    representation_qc = matrix_qc(representation_S)
    response_qc = matrix_qc(response_S)

    if representation_qc["is_constant"]:
        return {
            "status": "invalid_constant_representation",
            "n_queries": 0,
            "tie_repeats": int(tie_repeats),
        }

    if response_qc["is_constant"]:
        return {
            "status": "invalid_constant_response",
            "n_queries": 0,
            "tie_repeats": int(tie_repeats),
        }

    rows = []

    for repeat in range(max(1, int(tie_repeats))):
        response_rng = np.random.default_rng(int(seed) + repeat)
        representation_rng = np.random.default_rng(
            int(seed) + 1000003 + repeat
        )

        for i in range(n):
            response_rank = random_tie_ranking(
                response_S,
                i,
                response_rng,
            )

            representation_rank = random_tie_ranking(
                representation_S,
                i,
                representation_rng,
            )

            g = min(int(gold_k), n - 1)

            gold = set(
                response_rank[:g].tolist()
            )

            if not gold:
                continue

            row = {}

            for k in [5, 10, 20]:
                kk = min(k, n - 1)
                row[f"recall_at_{k}"] = (
                    len(
                        gold.intersection(
                            representation_rank[:kk]
                        )
                    )
                    / len(gold)
                )

            hit_ranks = [
                rank
                for rank, j in enumerate(
                    representation_rank,
                    start=1,
                )
                if j in gold
            ]

            row["mrr"] = (
                1.0 / min(hit_ranks)
                if hit_ranks
                else 0.0
            )

            for k in [10, 20]:
                kk = min(k, n - 1)

                rel = np.asarray(
                    [
                        j in gold
                        for j in representation_rank[:kk]
                    ],
                    dtype=np.float64,
                )

                discounts = (
                    1.0
                    / np.log2(
                        np.arange(
                            2,
                            kk + 2,
                            dtype=np.float64,
                        )
                    )
                )

                dcg = float(np.dot(rel, discounts))

                ideal = min(len(gold), kk)

                idcg = float(
                    np.sum(
                        1.0
                        / np.log2(
                            np.arange(
                                2,
                                ideal + 2,
                                dtype=np.float64,
                            )
                        )
                    )
                )

                row[f"ndcg_at_{k}"] = (
                    dcg / idcg
                    if idcg > 0
                    else np.nan
                )

            rows.append(row)

    if not rows:
        return {
            "status": "invalid_no_queries",
            "n_queries": 0,
        }

    frame = pd.DataFrame(rows)

    out = {
        col: float(frame[col].mean())
        for col in frame.columns
    }

    out.update({
        "status": "ok",
        "n_queries": n,
        "tie_repeats": int(tie_repeats),
    })

    return out


def exact_random_ndcg_expectation(
    n_atoms: int,
    gold_k: int,
    eval_k: int,
) -> float:
    candidates = int(n_atoms) - 1

    if candidates <= 0:
        return np.nan

    g = min(int(gold_k), candidates)
    k = min(int(eval_k), candidates)

    if g <= 0 or k <= 0:
        return np.nan

    discounts = (
        1.0
        / np.log2(
            np.arange(
                2,
                k + 2,
                dtype=np.float64,
            )
        )
    )

    expected_dcg = (
        float(g)
        / float(candidates)
        * float(discounts.sum())
    )

    ideal = min(g, k)

    idcg = float(
        np.sum(
            1.0
            / np.log2(
                np.arange(
                    2,
                    ideal + 2,
                    dtype=np.float64,
                )
            )
        )
    )

    return expected_dcg / idcg


def bootstrap_mean_ci(
    values: np.ndarray,
    seed: int,
    repeats: int = BOOTSTRAP_REPEATS,
) -> tuple[float, float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan, np.nan
    if len(x) == 1:
        return float(x[0]), float(x[0])

    rng = np.random.default_rng(seed)

    idx = rng.integers(
        0,
        len(x),
        size=(int(repeats), len(x)),
    )

    means = x[idx].mean(axis=1)

    return (
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def exact_signflip_p_two_sided(diffs: np.ndarray) -> tuple[float, int]:
    x = np.asarray(diffs, dtype=np.float64)
    x = x[np.isfinite(x)]

    # Exact zero deltas carry no sign information.
    x = x[np.abs(x) > 1e-15]

    n = len(x)

    if n == 0:
        return np.nan, 0

    observed = abs(float(x.mean()))

    values = []

    for signs in itertools.product([-1.0, 1.0], repeat=n):
        signed = x * np.asarray(signs)
        values.append(abs(float(signed.mean())))

    arr = np.asarray(values)

    p = float(
        np.mean(
            arr >= observed - 1e-15
        )
    )

    return p, n


def make_matrix(
    frame: pd.DataFrame,
    value_col: str,
    method_order: list[str],
    out_path: Path,
) -> pd.DataFrame:
    rows = []

    for ds in DATASET_ORDER:
        sub = frame.loc[
            frame["dataset_id"].astype(str).eq(ds)
        ].copy()

        if sub.empty:
            raise RuntimeError(f"Missing dataset in matrix source: {ds}")

        n_values = sub["response_atom_n"].astype(int).unique()

        if len(n_values) != 1:
            raise RuntimeError(f"{ds}: inconsistent response_atom_n")

        row = {
            "dataset_id": ds,
            "dataset_display": DATASET_DISPLAY[ds],
            "n_response_atoms": int(n_values[0]),
        }

        for method in method_order:
            hit = sub.loc[
                sub["model_key"].astype(str).eq(method),
                value_col,
            ]

            if len(hit) != 1:
                raise RuntimeError(
                    f"{ds}/{method}: expected one {value_col}, got {len(hit)}"
                )

            row[MODEL_SHORT[method]] = float(hit.iloc[0])

        rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv(out_path, sep="\t", index=False)
    return result


# ======================================================================================
# Upstream validation
# ======================================================================================

for p in [
    PRIMARY_BY_DATASET,
    PRIMARY_AUDIT,
    PRIMARY_MANIFEST,
    RESP_AUDIT,
    RESP_MANIFEST,
    COMPONENT_PROMPTS,
    TEXT_CATALOG,
    TEXT_AUTHORITY,
    ATOM_AUTHORITY,
    HIST_CONFIG,
]:
    if not p.is_file():
        raise FileNotFoundError(p)


if "R2_PRIMARY_TEXT_RESPONSE_ALIGNMENT=PASS" not in PRIMARY_AUDIT.read_text(
    encoding="utf-8",
    errors="replace",
):
    raise RuntimeError("Primary six-encoder alignment is not PASS.")


primary = pd.read_csv(
    PRIMARY_BY_DATASET,
    sep="\t",
    low_memory=False,
)

resp_audit = pd.read_csv(
    RESP_AUDIT,
    sep="\t",
    low_memory=False,
)

components = pd.read_csv(
    COMPONENT_PROMPTS,
    sep="\t",
    compression="gzip",
    low_memory=False,
)

atoms = pd.read_csv(
    ATOM_AUTHORITY,
    sep="\t",
    low_memory=False,
)

hist_cfg = json.loads(
    HIST_CONFIG.read_text(encoding="utf-8")
)


if set(primary["model_key"].astype(str).unique()) != set(MODEL_KEYS):
    raise RuntimeError("Primary language-model set differs from frozen six encoders.")

if set(primary["dataset_id"].astype(str).unique()) != set(DATASET_ORDER):
    raise RuntimeError("Primary dataset set differs from frozen 9 datasets.")

if len(primary) != 54:
    raise RuntimeError(f"Expected 54 primary rows, got {len(primary)}")


gold_k = int(hist_cfg.get("gold_neighbor_k", 10))
tie_repeats = int(hist_cfg.get("retrieval_tie_repeats", 5))

if gold_k != 10:
    raise RuntimeError(f"Unexpected gold_neighbor_k={gold_k}")

if tie_repeats != 5:
    raise RuntimeError(f"Unexpected retrieval_tie_repeats={tie_repeats}")


# ======================================================================================
# Corrected P1–P4 component text catalog for reproducible lexical controls
# ======================================================================================

log("step=corrected_view_text_catalog")

required_component_cols = {
    "dataset_id",
    "response_atom_id",
    "component_id",
    "response_aligned_prompt",
}

missing = required_component_cols - set(components.columns)

if missing:
    raise KeyError(f"Component prompt authority missing {sorted(missing)}")


view_rows = []

for _, row in components.iterrows():
    parsed = parse_response_aligned_p4(
        row["response_aligned_prompt"]
    )

    for view in ["P1", "P2", "P3", "P4"]:
        text = parsed[view]

        view_rows.append({
            "dataset_id": str(row["dataset_id"]),
            "response_atom_id": str(row["response_atom_id"]),
            "component_id": str(row["component_id"]),
            "view": view,
            "prompt_text": text,
            "prompt_sha256": sha256_text(text),
        })


view_components = pd.DataFrame(view_rows).sort_values(
    [
        "dataset_id",
        "response_atom_id",
        "component_id",
        "view",
    ]
).reset_index(drop=True)


# Strict collapse: response_atom × component × view must have one exact text.
conflicts = (
    view_components.groupby(
        ["response_atom_id", "component_id", "view"],
        observed=True,
    )["prompt_text"]
    .nunique()
)

if int(conflicts.ne(1).sum()) != 0:
    raise RuntimeError("Corrected P1–P4 component-text conflict.")


all_view_texts = sorted(
    view_components["prompt_text"].astype(str).unique().tolist()
)

view_text_index = {
    text: i
    for i, text in enumerate(all_view_texts)
}

view_components["text_index"] = (
    view_components["prompt_text"]
    .map(view_text_index)
    .astype(np.int64)
)

view_text_catalog = pd.DataFrame({
    "text_index": np.arange(
        len(all_view_texts),
        dtype=np.int64,
    ),
    "text_sha256": [
        sha256_text(text)
        for text in all_view_texts
    ],
    "prompt_text": all_view_texts,
})

view_components.to_csv(
    VIEW_COMPONENT_TSV,
    sep="\t",
    index=False,
    compression="gzip",
)

view_text_catalog.to_csv(
    VIEW_TEXT_CATALOG_TSV,
    sep="\t",
    index=False,
    compression="gzip",
)


# ======================================================================================
# Atom index
# ======================================================================================

atoms = atoms.copy()
atoms["dataset_id"] = atoms["dataset_id"].astype(str)
atoms["response_atom_id"] = atoms["response_atom_id"].astype(str)

if atoms["response_atom_id"].duplicated().any():
    raise RuntimeError("Final response atom authority IDs are not unique.")

atom_ids = atoms["response_atom_id"].tolist()

atom_pos = {
    rid: i
    for i, rid in enumerate(atom_ids)
}

if set(view_components["response_atom_id"].astype(str)) != set(atom_ids):
    raise RuntimeError("View-component atom namespace differs from final atom authority.")


# ======================================================================================
# TF-IDF baselines: frozen R1 code exact + manuscript specification sensitivity
# ======================================================================================

log("step=tfidf_controls")


def fit_tfidf_atom_embeddings(
    variant: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    if variant == "frozen_r1":
        vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=(1, 2),
            lowercase=True,
        )
        spec = {
            "variant": "frozen_r1",
            "min_df": 1,
            "sublinear_tf": False,
            "max_features": TFIDF_MAX_FEATURES,
            "ngram_range": [1, 2],
            "lowercase": True,
        }

    elif variant == "manuscript":
        vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=(1, 2),
            lowercase=True,
            min_df=2,
            sublinear_tf=True,
        )
        spec = {
            "variant": "manuscript",
            "min_df": 2,
            "sublinear_tf": True,
            "max_features": TFIDF_MAX_FEATURES,
            "ngram_range": [1, 2],
            "lowercase": True,
        }

    else:
        raise ValueError(variant)

    X_sparse = vectorizer.fit_transform(all_view_texts)

    n_components = int(
        min(
            BASELINE_DIM,
            X_sparse.shape[0] - 1,
            X_sparse.shape[1] - 1,
        )
    )

    if n_components < 2:
        raise RuntimeError(
            f"TF-IDF/SVD rank too small for {variant}: {X_sparse.shape}"
        )

    svd = TruncatedSVD(
        n_components=n_components,
        random_state=SEED,
        n_iter=5,
    )

    text_emb = svd.fit_transform(X_sparse)
    text_emb = l2_dense(text_emb)

    # Aggregate P4 component embeddings -> response atoms, mean -> L2.
    p4 = view_components.loc[
        view_components["view"].eq("P4")
    ].copy()

    sums = np.zeros(
        (len(atom_ids), n_components),
        dtype=np.float32,
    )

    counts = np.zeros(
        len(atom_ids),
        dtype=np.int32,
    )

    for _, row in p4.iterrows():
        rid = str(row["response_atom_id"])
        ai = atom_pos[rid]
        ti = int(row["text_index"])
        sums[ai] += text_emb[ti]
        counts[ai] += 1

    if np.any(counts <= 0):
        raise RuntimeError(f"{variant}: response atom without P4 component.")

    atom_emb = sums / counts[:, None].astype(np.float32)
    atom_emb = l2_dense(atom_emb)

    spec.update({
        "fit_corpus": "corrected_response_aligned_P1_P2_P3_P4_unique_component_texts",
        "fit_text_n": int(len(all_view_texts)),
        "vocabulary_n": int(X_sparse.shape[1]),
        "svd_dim": int(n_components),
        "svd_n_iter": 5,
        "seed": SEED,
        "explained_variance_ratio_sum": float(
            svd.explained_variance_ratio_.sum()
        ),
        "component_aggregation": "mean_then_l2",
    })

    variant_dir = BASELINE_ROOT / f"tfidf_{variant}"
    variant_dir.mkdir(parents=True, exist_ok=True)

    np.save(
        variant_dir / "response_atom_embeddings.npy",
        atom_emb.astype(np.float32),
    )

    write_json(
        spec,
        variant_dir / "contract.json",
    )

    return atom_emb, spec


tfidf_frozen, tfidf_frozen_spec = fit_tfidf_atom_embeddings(
    "frozen_r1"
)

tfidf_manuscript, tfidf_manuscript_spec = fit_tfidf_atom_embeddings(
    "manuscript"
)


# ======================================================================================
# Structured one-hot corrected P4 baseline
# ======================================================================================

log("step=structured_onehot")

atoms_for_onehot = atoms.copy()
atoms_for_onehot["entity"] = atoms_for_onehot.apply(
    family_aware_entity,
    axis=1,
)


def visible_features(row: pd.Series, view: str) -> list[str]:
    feats = []

    for field in VISIBLE_FIELDS[view]:
        value = clean_value(row.get(field, ""))

        if field == "entity":
            value = clean_value(row.get("entity", ""))

        if supported_value(value):
            feats.append(f"{field}={value}")

    return feats


feature_set = set()

for _, row in atoms_for_onehot.iterrows():
    for view in ["P1", "P2", "P3", "P4"]:
        feature_set.update(
            visible_features(row, view)
        )


onehot_features = sorted(feature_set)

feature_pos = {
    feat: i
    for i, feat in enumerate(onehot_features)
}

indptr = [0]
indices = []
data = []

for _, row in atoms_for_onehot.iterrows():
    feats = visible_features(row, "P4")

    cols = sorted(
        {
            feature_pos[x]
            for x in feats
        }
    )

    if not cols:
        raise RuntimeError(
            f"One-hot P4 row has no visible features: {row['response_atom_id']}"
        )

    value = 1.0 / math.sqrt(len(cols))

    indices.extend(cols)
    data.extend(
        [value] * len(cols)
    )

    indptr.append(
        len(indices)
    )


onehot = sparse.csr_matrix(
    (
        np.asarray(data, dtype=np.float32),
        np.asarray(indices, dtype=np.int32),
        np.asarray(indptr, dtype=np.int64),
    ),
    shape=(
        len(atoms_for_onehot),
        len(onehot_features),
    ),
    dtype=np.float32,
)

onehot_norms = np.sqrt(
    np.asarray(
        onehot.multiply(onehot).sum(axis=1)
    ).ravel()
)

if float(np.max(np.abs(onehot_norms - 1.0))) > 5e-6:
    raise RuntimeError("Structured one-hot L2 norm failure.")


onehot_dir = BASELINE_ROOT / "metadata_onehot"
onehot_dir.mkdir(parents=True, exist_ok=True)

sparse.save_npz(
    onehot_dir / "response_atom_embeddings.npz",
    onehot,
    compressed=True,
)

pd.DataFrame({
    "feature_index": np.arange(len(onehot_features), dtype=np.int64),
    "feature_name": onehot_features,
}).to_csv(
    onehot_dir / "feature_names.tsv",
    sep="\t",
    index=False,
)

write_json(
    {
        "status": "PASS",
        "view": "corrected_response_aligned_P4",
        "visible_fields": VISIBLE_FIELDS["P4"],
        "technical_context_included": False,
        "dimension": len(onehot_features),
        "normalization": "binary_features_scaled_by_1/sqrt(n_visible_features)",
        "source_contract": "frozen_R1_metadata_onehot_adapted_to_response_aligned_P4",
    },
    onehot_dir / "contract.json",
)


# ======================================================================================
# Primary simple-reference metrics
# ======================================================================================

log("step=simple_reference_metrics")

baseline_rows = []
tfidf_manuscript_rows = []

# The primary six-encoder table provides exact-uniform excess and raw NDCG.
primary = primary.copy()
primary["dataset_id"] = primary["dataset_id"].astype(str)
primary["model_key"] = primary["model_key"].astype(str)

# Response atom count lookup.
resp_count = {
    str(r["dataset_id"]): int(r["response_similarity_atom_n"])
    for _, r in resp_audit.iterrows()
}


for dataset_id in DATASET_ORDER:
    log(f"[baseline] {dataset_id}")

    resp_path = (
        RESP_ROOT
        / dataset_id
        / "response_similarity_logfc_hvg_spearman.npz"
    )

    response_S, response_ids = load_similarity_bundle(resp_path)

    n = len(response_ids)

    if n != resp_count[dataset_id]:
        raise RuntimeError(
            f"{dataset_id}: response count mismatch {n} != {resp_count[dataset_id]}"
        )

    missing = [
        rid
        for rid in response_ids
        if rid not in atom_pos
    ]

    if missing:
        raise RuntimeError(
            f"{dataset_id}: {len(missing)} response IDs absent from atom authority."
        )

    atom_rows = np.asarray(
        [atom_pos[rid] for rid in response_ids],
        dtype=np.int64,
    )

    response_upper = upper_vec(response_S)

    retrieval_seed = stable_seed(
        SEED,
        dataset_id,
        "primary_retrieval",
    )

    exact_uniform_10 = exact_random_ndcg_expectation(
        n_atoms=n,
        gold_k=gold_k,
        eval_k=10,
    )

    # ------------------------------------------------------------------
    # Frozen-R1-exact TF-IDF
    # ------------------------------------------------------------------
    representations: list[tuple[str, np.ndarray | sparse.csr_matrix, str]] = [
        (
            "tfidf_svd",
            np.asarray(tfidf_frozen[atom_rows], dtype=np.float32),
            "tfidf_frozen_r1",
        ),
        (
            "metadata_onehot",
            onehot[atom_rows],
            "structured_onehot",
        ),
    ]

    # ------------------------------------------------------------------
    # Fixed Gaussian random matched representation
    # Historical R1 rule: stable_hash("random",dataset_id,view,SEED)
    # adapted to corrected response-atom P4 row order.
    # ------------------------------------------------------------------
    random_seed = int(
        stable_hash(
            "random",
            dataset_id,
            "P4",
            SEED,
            length=16,
        ),
        16,
    ) % (2**63 - 1)

    rng = np.random.default_rng(random_seed)

    random_Z = rng.normal(
        size=(n, BASELINE_DIM)
    ).astype(np.float32)

    random_Z = l2_dense(random_Z)

    random_dir = BASELINE_ROOT / "random"
    random_dir.mkdir(parents=True, exist_ok=True)

    np.save(
        random_dir / f"{dataset_id}.npy",
        random_Z,
    )

    representations.append(
        (
            "random",
            random_Z,
            "matched_gaussian_random",
        )
    )

    random_ndcg_10 = None

    for model_key, Z, baseline_variant in representations:
        if sparse.issparse(Z):
            S = (Z @ Z.T).toarray().astype(np.float32)
        else:
            A = np.asarray(Z, dtype=np.float32)

            norms = np.linalg.norm(A, axis=1)

            if float(np.max(np.abs(norms - 1.0))) > 5e-4:
                raise RuntimeError(
                    f"{dataset_id}/{model_key}: embedding norm failure."
                )

            S = (A @ A.T).astype(np.float32)

        S = np.clip(S, -1.0, 1.0)
        np.fill_diagonal(S, 1.0)

        rep_upper = upper_vec(S)
        rep_qc = matrix_qc(S)

        rsa, rsa_status = safe_corr(
            rep_upper,
            response_upper,
            "spearman",
        )

        pearson_rsa, pearson_status = safe_corr(
            rep_upper,
            response_upper,
            "pearson",
        )

        ret = retrieval_metrics(
            representation_S=S,
            response_S=response_S,
            gold_k=gold_k,
            tie_repeats=tie_repeats,
            seed=retrieval_seed,
        )

        # Corrected R2 preserves historical Result2 constant-prior handling:
        # an uninformative constant representation is NA, not zero and not a
        # randomly tie-broken pseudo-score. This is expected for structured
        # one-hot in fixed-context single-perturbation datasets.
        if rep_qc["is_constant"]:
            if model_key != "metadata_onehot":
                raise RuntimeError(
                    f"{dataset_id}/{model_key}: unexpected constant representation"
                )
            ndcg10 = np.nan
            ndcg20 = np.nan
            recall10 = np.nan
            mrr = np.nan
            uniform_excess = np.nan
            row_status = "NA_UNINFORMATIVE_CONSTANT_REPRESENTATION"
        else:
            if rsa_status != "ok":
                raise RuntimeError(
                    f"{dataset_id}/{model_key}: RSA invalid: {rsa_status}"
                )

            if ret.get("status") != "ok":
                raise RuntimeError(
                    f"{dataset_id}/{model_key}: retrieval invalid: {ret.get('status')}"
                )

            ndcg10 = float(ret["ndcg_at_10"])
            ndcg20 = float(ret["ndcg_at_20"])
            recall10 = float(ret["recall_at_10"])
            mrr = float(ret["mrr"])
            uniform_excess = ndcg10 - exact_uniform_10
            row_status = "PASS"

        row = {
            "dataset_id": dataset_id,
            "model_key": model_key,
            "model_display": MODEL_DISPLAY[model_key],
            "model_group": MODEL_GROUP[model_key],
            "baseline_variant": baseline_variant,
            "response_atom_n": n,
            "pair_n": int(len(response_upper)),
            "representation_upper_sd": float(rep_qc["upper_sd"]),
            "representation_is_constant": bool(rep_qc["is_constant"]),
            "spearman_rsa": float(rsa) if np.isfinite(rsa) else np.nan,
            "spearman_rsa_status": rsa_status,
            "pearson_rsa": float(pearson_rsa) if np.isfinite(pearson_rsa) else np.nan,
            "pearson_rsa_status": pearson_status,
            "ndcg_at_10": ndcg10,
            "ndcg_at_20": ndcg20,
            "recall_at_10": recall10,
            "mrr": mrr,
            "retrieval_status": str(ret.get("status")),
            "exact_uniform_random_ndcg_at_10": float(exact_uniform_10),
            "exact_uniform_excess_ndcg_at_10": float(uniform_excess) if np.isfinite(uniform_excess) else np.nan,
            "matched_random_ndcg_at_10": np.nan,
            "ndcg_gain_vs_matched_random": np.nan,
            "retrieval_seed": int(retrieval_seed),
            "random_representation_seed": (
                int(random_seed)
                if model_key == "random"
                else np.nan
            ),
            "status": row_status,
        }

        baseline_rows.append(row)

        if model_key == "random":
            random_ndcg_10 = ndcg10

    if random_ndcg_10 is None:
        raise RuntimeError(f"{dataset_id}: random NDCG was not computed.")

    # Fill dataset matched-random baseline/gain for all primary baseline rows.
    for row in baseline_rows:
        if row["dataset_id"] == dataset_id:
            row["matched_random_ndcg_at_10"] = float(random_ndcg_10)
            if np.isfinite(row["ndcg_at_10"]):
                row["ndcg_gain_vs_matched_random"] = float(
                    row["ndcg_at_10"] - random_ndcg_10
                )
            else:
                row["ndcg_gain_vs_matched_random"] = np.nan

    # ------------------------------------------------------------------
    # Manuscript-spec TF-IDF sensitivity
    # ------------------------------------------------------------------
    Zm = np.asarray(
        tfidf_manuscript[atom_rows],
        dtype=np.float32,
    )

    Sm = (Zm @ Zm.T).astype(np.float32)
    Sm = np.clip(Sm, -1.0, 1.0)
    np.fill_diagonal(Sm, 1.0)

    rsa_m, rsa_m_status = safe_corr(
        upper_vec(Sm),
        response_upper,
        "spearman",
    )

    ret_m = retrieval_metrics(
        representation_S=Sm,
        response_S=response_S,
        gold_k=gold_k,
        tie_repeats=tie_repeats,
        seed=retrieval_seed,
    )

    if rsa_m_status != "ok" or ret_m.get("status") != "ok":
        raise RuntimeError(
            f"{dataset_id}: manuscript-spec TF-IDF sensitivity failed."
        )

    ndcg_m = float(ret_m["ndcg_at_10"])

    tfidf_manuscript_rows.append({
        "dataset_id": dataset_id,
        "response_atom_n": n,
        "spearman_rsa_manuscript": float(rsa_m),
        "ndcg_at_10_manuscript": ndcg_m,
        "ndcg_gain_vs_matched_random_manuscript": float(
            ndcg_m - random_ndcg_10
        ),
        "exact_uniform_excess_ndcg_at_10_manuscript": float(
            ndcg_m - exact_uniform_10
        ),
    })


baseline = pd.DataFrame(baseline_rows).sort_values(
    ["model_key", "dataset_id"]
).reset_index(drop=True)

baseline.to_csv(
    BASELINE_BY_DATASET_TSV,
    sep="\t",
    index=False,
)


# ======================================================================================
# Combine six encoders + corrected simple references
# ======================================================================================

log("step=all_method_primary_table")

random_baseline = baseline.loc[
    baseline["model_key"].eq("random"),
    ["dataset_id", "ndcg_at_10"],
].rename(
    columns={
        "ndcg_at_10": "matched_random_ndcg_at_10"
    }
)

language = primary.copy()

language = language.merge(
    random_baseline,
    on="dataset_id",
    how="left",
    validate="many_to_one",
)

language["ndcg_gain_vs_matched_random"] = (
    language["ndcg_at_10"]
    - language["matched_random_ndcg_at_10"]
)

# Primary runner already contains exact-uniform excess under this name.
language["exact_uniform_excess_ndcg_at_10"] = (
    language["excess_ndcg_at_10"]
)

language["exact_uniform_random_ndcg_at_10"] = (
    language["random_ndcg_at_10_exact"]
)

language_keep = [
    "dataset_id",
    "model_key",
    "model_display",
    "model_group",
    "response_atom_n",
    "pair_n",
    "spearman_rsa",
    "pearson_rsa",
    "ndcg_at_10",
    "ndcg_at_20",
    "recall_at_10",
    "mrr",
    "exact_uniform_random_ndcg_at_10",
    "exact_uniform_excess_ndcg_at_10",
    "matched_random_ndcg_at_10",
    "ndcg_gain_vs_matched_random",
    "status",
]

language_primary = language[language_keep].copy()

baseline_primary = baseline[language_keep].copy()

all_primary = pd.concat(
    [
        language_primary,
        baseline_primary,
    ],
    ignore_index=True,
)

primary_method_order = MODEL_KEYS + [
    "tfidf_svd",
    "metadata_onehot",
    "random",
]

if len(all_primary) != 9 * len(primary_method_order):
    raise RuntimeError(
        f"Expected 81 primary method rows, got {len(all_primary)}"
    )

for ds in DATASET_ORDER:
    methods = set(
        all_primary.loc[
            all_primary["dataset_id"].eq(ds),
            "model_key",
        ].astype(str)
    )

    if methods != set(primary_method_order):
        raise RuntimeError(
            f"{ds}: incomplete primary method set: {sorted(methods)}"
        )

all_primary = all_primary.sort_values(
    ["dataset_id", "model_key"]
).reset_index(drop=True)

all_primary.to_csv(
    ALL_METHOD_BY_DATASET_TSV,
    sep="\t",
    index=False,
)


summary_rows = []

for model_key, sub in all_primary.groupby("model_key"):
    row = {
        "model_key": model_key,
        "model_display": MODEL_DISPLAY[model_key],
        "model_group": MODEL_GROUP[model_key],
        "dataset_n_total": int(sub["dataset_id"].nunique()),
        "spearman_rsa_valid_dataset_n": int(sub["spearman_rsa"].notna().sum()),
        "ndcg_gain_valid_dataset_n": int(sub["ndcg_gain_vs_matched_random"].notna().sum()),
        "spearman_rsa_macro_mean": float(sub["spearman_rsa"].mean()),
        "spearman_rsa_macro_median": float(sub["spearman_rsa"].median()),
        "spearman_rsa_positive_dataset_n": int(
            (sub["spearman_rsa"] > 0).sum()
        ),
        "ndcg_at_10_macro_mean": float(sub["ndcg_at_10"].mean()),
        "ndcg_gain_vs_matched_random_macro_mean": float(
            sub["ndcg_gain_vs_matched_random"].mean()
        ),
        "ndcg_gain_vs_matched_random_positive_dataset_n": int(
            (sub["ndcg_gain_vs_matched_random"] > 0).sum()
        ),
        "exact_uniform_excess_ndcg_at_10_macro_mean": float(
            sub["exact_uniform_excess_ndcg_at_10"].mean()
        ),
    }

    for metric in [
        "spearman_rsa",
        "ndcg_gain_vs_matched_random",
        "exact_uniform_excess_ndcg_at_10",
    ]:
        lo, hi = bootstrap_mean_ci(
            sub[metric].to_numpy(dtype=float),
            seed=stable_seed(
                SEED,
                "paper_closure",
                model_key,
                metric,
            ),
        )

        row[f"{metric}_bootstrap_ci_low"] = lo
        row[f"{metric}_bootstrap_ci_high"] = hi

    summary_rows.append(row)


all_summary = pd.DataFrame(summary_rows).sort_values(
    "spearman_rsa_macro_mean",
    ascending=False,
).reset_index(drop=True)

all_summary.to_csv(
    ALL_METHOD_SUMMARY_TSV,
    sep="\t",
    index=False,
)


# ======================================================================================
# TF-IDF frozen-code vs manuscript-method sensitivity
# ======================================================================================

log("step=tfidf_spec_discrepancy")

tfidf_primary = baseline.loc[
    baseline["model_key"].eq("tfidf_svd"),
    [
        "dataset_id",
        "response_atom_n",
        "spearman_rsa",
        "ndcg_at_10",
        "ndcg_gain_vs_matched_random",
        "exact_uniform_excess_ndcg_at_10",
    ],
].rename(
    columns={
        "spearman_rsa": "spearman_rsa_frozen_r1",
        "ndcg_at_10": "ndcg_at_10_frozen_r1",
        "ndcg_gain_vs_matched_random": (
            "ndcg_gain_vs_matched_random_frozen_r1"
        ),
        "exact_uniform_excess_ndcg_at_10": (
            "exact_uniform_excess_ndcg_at_10_frozen_r1"
        ),
    }
)

tfidf_sens = tfidf_primary.merge(
    pd.DataFrame(tfidf_manuscript_rows),
    on=[
        "dataset_id",
        "response_atom_n",
    ],
    how="inner",
    validate="one_to_one",
)

tfidf_sens["delta_rsa_manuscript_minus_frozen"] = (
    tfidf_sens["spearman_rsa_manuscript"]
    - tfidf_sens["spearman_rsa_frozen_r1"]
)

tfidf_sens["delta_ndcg10_manuscript_minus_frozen"] = (
    tfidf_sens["ndcg_at_10_manuscript"]
    - tfidf_sens["ndcg_at_10_frozen_r1"]
)

tfidf_sens[
    "delta_gain_manuscript_minus_frozen"
] = (
    tfidf_sens[
        "ndcg_gain_vs_matched_random_manuscript"
    ]
    - tfidf_sens[
        "ndcg_gain_vs_matched_random_frozen_r1"
    ]
)

tfidf_sens.to_csv(
    TFIDF_SENS_TSV,
    sep="\t",
    index=False,
)


# ======================================================================================
# Stage A: paired language-encoder closure
# ======================================================================================

log("step=language_paired_closure")

language_closure = all_primary.loc[
    all_primary["model_key"].isin(MODEL_KEYS)
].copy()

pair_rows = []

for model_a, model_b in itertools.combinations(MODEL_KEYS, 2):
    a = language_closure.loc[
        language_closure["model_key"].eq(model_a)
    ].set_index("dataset_id")

    b = language_closure.loc[
        language_closure["model_key"].eq(model_b)
    ].set_index("dataset_id")

    if set(a.index) != set(DATASET_ORDER) or set(b.index) != set(DATASET_ORDER):
        raise RuntimeError(
            f"Pair {model_a}/{model_b}: dataset mismatch"
        )

    for metric in [
        "spearman_rsa",
        "ndcg_gain_vs_matched_random",
    ]:
        diffs = np.asarray(
            [
                float(a.loc[ds, metric])
                - float(b.loc[ds, metric])
                for ds in DATASET_ORDER
            ],
            dtype=np.float64,
        )

        lo, hi = bootstrap_mean_ci(
            diffs,
            seed=stable_seed(
                SEED,
                "paired_closure",
                model_a,
                model_b,
                metric,
            ),
        )

        p, n_eff = exact_signflip_p_two_sided(diffs)

        eps = 1e-12

        pair_rows.append({
            "model_a": model_a,
            "model_a_display": MODEL_DISPLAY[model_a],
            "model_b": model_b,
            "model_b_display": MODEL_DISPLAY[model_b],
            "metric": metric,
            "dataset_n": len(diffs),
            "mean_delta_a_minus_b": float(diffs.mean()),
            "median_delta_a_minus_b": float(np.median(diffs)),
            "bootstrap_ci_low": lo,
            "bootstrap_ci_high": hi,
            "a_win_dataset_n": int((diffs > eps).sum()),
            "b_win_dataset_n": int((diffs < -eps).sum()),
            "tie_dataset_n": int((np.abs(diffs) <= eps).sum()),
            "exact_signflip_n_nonzero": int(n_eff),
            "exact_signflip_p_two_sided_diagnostic": p,
            "dataset_deltas_json": json.dumps(
                {
                    ds: float(diff)
                    for ds, diff in zip(DATASET_ORDER, diffs)
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        })


pairwise = pd.DataFrame(pair_rows)

pairwise.to_csv(
    PAIRWISE_TSV,
    sep="\t",
    index=False,
)


# ======================================================================================
# Dataset-level rank concordance + model rank stability
# ======================================================================================

log("step=rank_concordance")

rank_rows = []
rank_long_rows = []

for ds in DATASET_ORDER:
    sub = language_closure.loc[
        language_closure["dataset_id"].eq(ds)
    ].copy()

    if len(sub) != 6:
        raise RuntimeError(f"{ds}: expected 6 language rows")

    rsa_values = sub["spearman_rsa"].to_numpy(dtype=float)
    gain_values = sub["ndcg_gain_vs_matched_random"].to_numpy(dtype=float)

    rsa_rank = rankdata(
        -rsa_values,
        method="average",
    )

    gain_rank = rankdata(
        -gain_values,
        method="average",
    )

    rank_corr, rank_status = safe_corr(
        rsa_rank,
        gain_rank,
        "spearman",
    )

    # Pairwise metric direction discordance.
    metric_discordant_pair_n = 0
    metric_concordant_pair_n = 0
    metric_tied_pair_n = 0

    val = {
        str(r["model_key"]): (
            float(r["spearman_rsa"]),
            float(r["ndcg_gain_vs_matched_random"]),
        )
        for _, r in sub.iterrows()
    }

    for a, b in itertools.combinations(MODEL_KEYS, 2):
        dr = val[a][0] - val[b][0]
        dg = val[a][1] - val[b][1]

        if abs(dr) <= 1e-12 or abs(dg) <= 1e-12:
            metric_tied_pair_n += 1
        elif dr * dg > 0:
            metric_concordant_pair_n += 1
        else:
            metric_discordant_pair_n += 1

    rsa_max = float(np.max(rsa_values))
    gain_max = float(np.max(gain_values))

    top_rsa = sorted(
        sub.loc[
            np.isclose(
                sub["spearman_rsa"].to_numpy(dtype=float),
                rsa_max,
                atol=1e-12,
                rtol=0,
            ),
            "model_key",
        ].astype(str).tolist()
    )

    top_gain = sorted(
        sub.loc[
            np.isclose(
                sub["ndcg_gain_vs_matched_random"].to_numpy(dtype=float),
                gain_max,
                atol=1e-12,
                rtol=0,
            ),
            "model_key",
        ].astype(str).tolist()
    )

    rank_rows.append({
        "dataset_id": ds,
        "dataset_display": DATASET_DISPLAY[ds],
        "response_atom_n": int(sub["response_atom_n"].iloc[0]),
        "rsa_ndcg_rank_spearman": float(rank_corr),
        "rank_correlation_status": rank_status,
        "top_rsa_model_keys": ";".join(top_rsa),
        "top_ndcg_gain_model_keys": ";".join(top_gain),
        "same_top_model": bool(set(top_rsa) & set(top_gain)),
        "metric_concordant_encoder_pair_n": metric_concordant_pair_n,
        "metric_discordant_encoder_pair_n": metric_discordant_pair_n,
        "metric_tied_encoder_pair_n": metric_tied_pair_n,
    })

    for i, (_, r) in enumerate(sub.reset_index(drop=True).iterrows()):
        rank_long_rows.append({
            "dataset_id": ds,
            "model_key": str(r["model_key"]),
            "rsa_rank": float(rsa_rank[i]),
            "ndcg_gain_rank": float(gain_rank[i]),
            "spearman_rsa": float(r["spearman_rsa"]),
            "ndcg_gain_vs_matched_random": float(
                r["ndcg_gain_vs_matched_random"]
            ),
        })


rank_by_dataset = pd.DataFrame(rank_rows)
rank_by_dataset.to_csv(
    RANK_BY_DATASET_TSV,
    sep="\t",
    index=False,
)

rank_long = pd.DataFrame(rank_long_rows)

stability_rows = []

for model_key in MODEL_KEYS:
    sub = rank_long.loc[
        rank_long["model_key"].eq(model_key)
    ]

    stability_rows.append({
        "model_key": model_key,
        "model_display": MODEL_DISPLAY[model_key],
        "model_group": MODEL_GROUP[model_key],
        "rsa_mean_rank": float(sub["rsa_rank"].mean()),
        "rsa_median_rank": float(sub["rsa_rank"].median()),
        "rsa_rank_sd": float(sub["rsa_rank"].std(ddof=0)),
        "rsa_rank1_dataset_n": int(
            np.isclose(sub["rsa_rank"], 1.0).sum()
        ),
        "ndcg_gain_mean_rank": float(sub["ndcg_gain_rank"].mean()),
        "ndcg_gain_median_rank": float(sub["ndcg_gain_rank"].median()),
        "ndcg_gain_rank_sd": float(
            sub["ndcg_gain_rank"].std(ddof=0)
        ),
        "ndcg_gain_rank1_dataset_n": int(
            np.isclose(sub["ndcg_gain_rank"], 1.0).sum()
        ),
    })


model_stability = pd.DataFrame(stability_rows).sort_values(
    "rsa_mean_rank"
)

model_stability.to_csv(
    MODEL_STABILITY_TSV,
    sep="\t",
    index=False,
)


# Dataset-centered RSA–NDCG concordance over all 54 language rows.
centered = language_closure.copy()

centered["rsa_centered"] = (
    centered["spearman_rsa"]
    - centered.groupby("dataset_id")["spearman_rsa"].transform("mean")
)

centered["gain_centered"] = (
    centered["ndcg_gain_vs_matched_random"]
    - centered.groupby("dataset_id")[
        "ndcg_gain_vs_matched_random"
    ].transform("mean")
)

centered_metric_corr, centered_metric_corr_status = safe_corr(
    centered["rsa_centered"].to_numpy(dtype=float),
    centered["gain_centered"].to_numpy(dtype=float),
    "spearman",
)


# ======================================================================================
# Descriptive general-purpose vs biomedical group comparison
# ======================================================================================

log("step=model_group_descriptive")

group_rows = []

for ds in DATASET_ORDER:
    sub = language_closure.loc[
        language_closure["dataset_id"].eq(ds)
    ]

    general = sub.loc[
        sub["model_group"].eq("general_purpose")
    ]

    biomedical = sub.loc[
        sub["model_group"].eq("biomedical")
    ]

    if len(general) != 2 or len(biomedical) != 4:
        raise RuntimeError(
            f"{ds}: unexpected model-group sizes "
            f"general={len(general)} biomedical={len(biomedical)}"
        )

    group_rows.append({
        "dataset_id": ds,
        "dataset_display": DATASET_DISPLAY[ds],
        "general_rsa_mean": float(general["spearman_rsa"].mean()),
        "biomedical_rsa_mean": float(biomedical["spearman_rsa"].mean()),
        "biomedical_minus_general_rsa": float(
            biomedical["spearman_rsa"].mean()
            - general["spearman_rsa"].mean()
        ),
        "general_ndcg_gain_mean": float(
            general["ndcg_gain_vs_matched_random"].mean()
        ),
        "biomedical_ndcg_gain_mean": float(
            biomedical["ndcg_gain_vs_matched_random"].mean()
        ),
        "biomedical_minus_general_ndcg_gain": float(
            biomedical["ndcg_gain_vs_matched_random"].mean()
            - general["ndcg_gain_vs_matched_random"].mean()
        ),
    })


group_by_dataset = pd.DataFrame(group_rows)

group_by_dataset.to_csv(
    GROUP_BY_DATASET_TSV,
    sep="\t",
    index=False,
)


group_summary_rows = []

for metric in [
    "biomedical_minus_general_rsa",
    "biomedical_minus_general_ndcg_gain",
]:
    vals = group_by_dataset[metric].to_numpy(dtype=float)

    lo, hi = bootstrap_mean_ci(
        vals,
        seed=stable_seed(
            SEED,
            "model_group",
            metric,
        ),
    )

    group_summary_rows.append({
        "metric": metric,
        "dataset_n": len(vals),
        "macro_mean": float(vals.mean()),
        "macro_median": float(np.median(vals)),
        "bootstrap_ci_low": lo,
        "bootstrap_ci_high": hi,
        "biomedical_higher_dataset_n": int((vals > 1e-12).sum()),
        "general_higher_dataset_n": int((vals < -1e-12).sum()),
        "tie_dataset_n": int((np.abs(vals) <= 1e-12).sum()),
        "interpretation_unit": (
            "dataset-level descriptive mean of 4 biomedical encoders "
            "minus mean of 2 general-purpose encoders; encoders are not "
            "treated as independent biological replicates"
        ),
    })


group_summary = pd.DataFrame(group_summary_rows)

group_summary.to_csv(
    GROUP_SUMMARY_TSV,
    sep="\t",
    index=False,
)


# ======================================================================================
# Fig.1d-ready source matrices
# ======================================================================================

log("step=fig1d_source_data")

fig_rsa = make_matrix(
    all_primary,
    "spearman_rsa",
    primary_method_order,
    FIG1D_RSA_TSV,
)

fig_gain = make_matrix(
    all_primary,
    "ndcg_gain_vs_matched_random",
    primary_method_order,
    FIG1D_GAIN_TSV,
)

fig_raw = make_matrix(
    all_primary,
    "ndcg_at_10",
    primary_method_order,
    FIG1D_RAW_NDCG_TSV,
)

fig_uniform = make_matrix(
    all_primary,
    "exact_uniform_excess_ndcg_at_10",
    primary_method_order,
    UNIFORM_EXCESS_TSV,
)


# ======================================================================================
# Paper/release methods note + public staging
# ======================================================================================

tfidf_max_abs_rsa_delta = float(
    tfidf_sens["delta_rsa_manuscript_minus_frozen"].abs().max()
)

tfidf_max_abs_gain_delta = float(
    tfidf_sens["delta_gain_manuscript_minus_frozen"].abs().max()
)

methods_text = f"""# PerturbContextAlign R2 — Primary paper closure v2

## Scope

This stage closes the corrected broad Result 2 primary panel before P1–P4
ablation and deeper cross-context analysis.

## Primary methods included

- BGE-M3
- Qwen3-Embedding-0.6B
- SapBERT
- BiomedBERT
- MedCPT Article
- MedCPT Query
- TF-IDF + SVD
- structured metadata one-hot
- matched Gaussian Random

All methods are aligned to the same frozen corrected response atoms.

## Broad local-alignment baseline

The paper-facing broad local metric is:

`NDCG@10 gain = method NDCG@10 - matched Gaussian-Random NDCG@10`

The matched Random representation is a fixed 256-dimensional Gaussian
representation generated independently per dataset with the frozen R1 seed rule:

`stable_hash("random", dataset_id, "P4", 20260711)`

and L2-normalized row vectors.

The exact uniform-random expected NDCG remains in the source data as a
sensitivity/reference statistic and is not substituted for matched Random in
the Fig.1d primary panel.

## Corrected response-aligned one-hot

The structured P4 control uses visible response-aligned fields only:

`family, mode, entity, combo_n, species, cell_type, cell_line, tissue, disease, dose, duration`

Platform and batch are excluded because corrected R2 removes Technical context
from the primary response-aligned representation.

Each visible `field=value` is one binary feature and each row is L2-normalized.

For fixed-context single-perturbation datasets, this representation can be
mathematically uninformative even though every atom has a unique identity. If
all rows share the same non-identity fields and each row contributes one unique
entity category, every off-diagonal cosine is identical. Corrected R2 preserves
the frozen Result2 constant-prior rule: RSA and neighbourhood metrics are then
reported as NA rather than silently set to zero or randomized.

The input audit predicts this for Replogle K562 essential, Replogle RPE1, Tian
CRISPRa and Tian CRISPRi. This is an expected property of the baseline under the
corrected response-atom unit, not loss of perturbation identity.

## TF-IDF implementation discrepancy

The frozen R1 code used:

- word uni/bigrams
- max_features=20,000
- lowercase=True
- sklearn defaults `min_df=1`, `sublinear_tf=False`
- TruncatedSVD up to 256 dimensions, n_iter=5
- L2 normalization

The current manuscript Supplement instead states `min_df=2` and
`sublinear_tf=True`.

This stage therefore computes both variants:

1. `tfidf_svd` primary = frozen R1 code exact;
2. manuscript-spec TF-IDF = sensitivity only.

Maximum observed differences across datasets in this run:

- |ΔRSA| max = {tfidf_max_abs_rsa_delta:.8f}
- |Δmatched-random NDCG gain| max = {tfidf_max_abs_gain_delta:.8f}

The manuscript/code discrepancy must be resolved explicitly before final public
release.

## Language-encoder paired closure

For each pair among the six language encoders, differences are calculated on
the same nine datasets for:

- Spearman RSA
- NDCG@10 gain versus matched Random

Reported closure statistics include:

- mean and median paired difference;
- dataset bootstrap interval;
- dataset win/loss/tie counts;
- exact sign-flip p value as an internal diagnostic only.

The exact sign-flip value is not automatically promoted to the manuscript.

## Model-family descriptive comparison

General-purpose and biomedical encoder group means are descriptive only.
Encoders are not treated as independent biological replicates.
"""

METHODS_MD.write_text(
    methods_text,
    encoding="utf-8",
)


# Public candidate staging.
figshare_stage = (
    NEW
    / "09_release_staging"
    / "figshare_candidate"
    / "result2"
    / "primary_paper_closure_v2"
)

figshare_source = (
    NEW
    / "09_release_staging"
    / "figshare_candidate"
    / "source_data"
    / "primary_paper_closure_v2"
)

github_scripts = (
    NEW
    / "09_release_staging"
    / "github_candidate"
    / "scripts"
    / "alignment"
)

github_docs = (
    NEW
    / "09_release_staging"
    / "github_candidate"
    / "docs"
)

for d in [
    figshare_stage,
    figshare_source,
    github_scripts,
    github_docs,
]:
    d.mkdir(parents=True, exist_ok=True)


# ======================================================================================
# Final audit
# ======================================================================================

language_summary = all_summary.loc[
    all_summary["model_key"].isin(MODEL_KEYS)
].copy()

rank_same_top_n = int(
    rank_by_dataset["same_top_model"].sum()
)

rank_discordant_pair_n = int(
    rank_by_dataset["metric_discordant_encoder_pair_n"].sum()
)

rank_all_pair_n = int(
    (
        rank_by_dataset["metric_discordant_encoder_pair_n"]
        + rank_by_dataset["metric_concordant_encoder_pair_n"]
        + rank_by_dataset["metric_tied_encoder_pair_n"]
    ).sum()
)

random_gain_max_abs = float(
    all_primary.loc[
        all_primary["model_key"].eq("random"),
        "ndcg_gain_vs_matched_random",
    ].abs().max()
)

if random_gain_max_abs > 1e-12:
    raise RuntimeError(
        f"Matched Random gain is not zero: max abs={random_gain_max_abs}"
    )

status = "PASS"

lines = [
    "PERTURBCONTEXTALIGN R2 PRIMARY PAPER CLOSURE AUDIT v2",
    "=" * 120,
    "",
    "UPSTREAM",
    "-" * 120,
    "primary_six_encoder_alignment=PASS",
    "response_similarity=PASS",
    "response_atom_namespace=FROZEN",
    "response_aligned_text=FROZEN",
    "",
    "PRIMARY PAPER BASELINES",
    "-" * 120,
    "TFIDF_primary=frozen_R1_code_exact",
    "TFIDF_manuscript_spec=sensitivity_only",
    "onehot=corrected_response_aligned_P4_visible_fields",
    "random=fixed_256d_Gaussian_matched_per_dataset",
    "broad_NDCG_gain=method_NDCG@10-minus-matched_Gaussian_Random_NDCG@10",
    "exact_uniform_excess=sensitivity_reference",
    f"tfidf_max_abs_rsa_delta_manuscript_vs_frozen={tfidf_max_abs_rsa_delta:.10f}",
    f"tfidf_max_abs_gain_delta_manuscript_vs_frozen={tfidf_max_abs_gain_delta:.10f}",
    "onehot_constant_policy=NA_not_zero_not_randomized",
    "onehot_constant_datasets=" + ";".join(
        sorted(
            baseline.loc[
                baseline["model_key"].eq("metadata_onehot")
                & baseline["representation_is_constant"].fillna(False),
                "dataset_id",
            ].astype(str).tolist()
        )
    ),
    "",
    "COUNTS",
    "-" * 120,
    f"dataset_n={len(DATASET_ORDER)}",
    f"language_encoder_n={len(MODEL_KEYS)}",
    f"primary_method_n={len(primary_method_order)}",
    f"primary_dataset_method_row_n={len(all_primary)}",
    f"paired_language_comparison_n={len(pairwise)}",
    "",
    "LANGUAGE MODEL SUMMARY",
    "-" * 120,
    language_summary.to_string(index=False),
    "",
    "RANK CONCORDANCE",
    "-" * 120,
    f"dataset_same_top_encoder_for_RSA_and_NDCG_gain_n={rank_same_top_n}",
    f"dataset_n={len(rank_by_dataset)}",
    f"encoder_pair_metric_discordance_n={rank_discordant_pair_n}",
    f"encoder_pair_metric_comparison_total_n={rank_all_pair_n}",
    f"dataset_centered_RSA_vs_NDCG_gain_spearman={centered_metric_corr:.10f}",
    f"dataset_centered_metric_correlation_status={centered_metric_corr_status}",
    "",
    "MODEL STABILITY",
    "-" * 120,
    model_stability.to_string(index=False),
    "",
    "MODEL GROUP DESCRIPTIVE",
    "-" * 120,
    group_summary.to_string(index=False),
    "",
    "STATUS",
    "-" * 120,
    f"R2_PRIMARY_PAPER_CLOSURE={status}",
    "NEXT=R2_CORRECTED_P1_P4_ABLATION_v1",
]

AUDIT_TXT.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)


manifest = {
    "version": "R2_PRIMARY_PAPER_CLOSURE_v2",
    "status": status,
    "primary_tfidf": "frozen_R1_code_exact",
    "manuscript_tfidf": "sensitivity_only",
    "matched_random": {
        "dimension": BASELINE_DIM,
        "distribution": "standard_normal_then_L2",
        "seed": SEED,
        "seed_rule": 'stable_hash("random",dataset_id,"P4",SEED)',
    },
    "onehot_visible_fields": VISIBLE_FIELDS["P4"],
    "onehot_constant_geometry_policy": "NA_per_frozen_Result2_constant_prior_rule",
    "broad_ndcg_primary": "method_minus_matched_gaussian_random",
    "exact_uniform_ndcg": "sensitivity_reference",
    "inputs": {
        "primary_alignment": {
            "path": str(PRIMARY_BY_DATASET),
            "sha256": sha256_file(PRIMARY_BY_DATASET),
        },
        "response_similarity_manifest": {
            "path": str(RESP_MANIFEST),
            "sha256": sha256_file(RESP_MANIFEST),
        },
        "component_prompts": {
            "path": str(COMPONENT_PROMPTS),
            "sha256": sha256_file(COMPONENT_PROMPTS),
        },
        "response_atom_authority": {
            "path": str(ATOM_AUTHORITY),
            "sha256": sha256_file(ATOM_AUTHORITY),
        },
    },
    "tfidf_specs": {
        "frozen_r1": tfidf_frozen_spec,
        "manuscript": tfidf_manuscript_spec,
    },
    "outputs": {
        "all_method_by_dataset": str(ALL_METHOD_BY_DATASET_TSV),
        "all_method_summary": str(ALL_METHOD_SUMMARY_TSV),
        "pairwise_language_closure": str(PAIRWISE_TSV),
        "rank_concordance": str(RANK_BY_DATASET_TSV),
        "model_stability": str(MODEL_STABILITY_TSV),
        "group_descriptive": str(GROUP_SUMMARY_TSV),
        "fig1d_rsa": str(FIG1D_RSA_TSV),
        "fig1d_ndcg_gain": str(FIG1D_GAIN_TSV),
        "tfidf_sensitivity": str(TFIDF_SENS_TSV),
        "audit": str(AUDIT_TXT),
    },
}

write_json(
    manifest,
    MANIFEST_JSON,
)


# ======================================================================================
# Public staging after PASS
# ======================================================================================

public_result_files = [
    ALL_METHOD_BY_DATASET_TSV,
    ALL_METHOD_SUMMARY_TSV,
    TFIDF_SENS_TSV,
    PAIRWISE_TSV,
    RANK_BY_DATASET_TSV,
    MODEL_STABILITY_TSV,
    GROUP_BY_DATASET_TSV,
    GROUP_SUMMARY_TSV,
    FIG1D_RSA_TSV,
    FIG1D_GAIN_TSV,
    FIG1D_RAW_NDCG_TSV,
    UNIFORM_EXCESS_TSV,
    METHODS_MD,
    AUDIT_TXT,
    MANIFEST_JSON,
]

source_data_files = [
    ALL_METHOD_BY_DATASET_TSV,
    PAIRWISE_TSV,
    RANK_BY_DATASET_TSV,
    MODEL_STABILITY_TSV,
    GROUP_BY_DATASET_TSV,
    FIG1D_RSA_TSV,
    FIG1D_GAIN_TSV,
    FIG1D_RAW_NDCG_TSV,
    UNIFORM_EXCESS_TSV,
    TFIDF_SENS_TSV,
]

for src in public_result_files:
    shutil.copy2(
        src,
        figshare_stage / src.name,
    )

for src in source_data_files:
    shutil.copy2(
        src,
        figshare_source / src.name,
    )

# Current running script becomes GitHub candidate.
shutil.copy2(
    Path(__file__).resolve(),
    github_scripts / Path(__file__).name,
)

shutil.copy2(
    METHODS_MD,
    github_docs / METHODS_MD.name,
)

source_registry = pd.DataFrame([
    {
        "file": src.name,
        "source_stage": "R2_PRIMARY_PAPER_CLOSURE_v2",
        "description": {
            ALL_METHOD_BY_DATASET_TSV.name:
                "9 datasets × 9 primary methods with RSA, raw NDCG, matched-Random gain, and exact-uniform sensitivity",
            PAIRWISE_TSV.name:
                "paired six-language-encoder closure across nine datasets",
            RANK_BY_DATASET_TSV.name:
                "dataset-level RSA versus local-neighbourhood encoder rank concordance",
            MODEL_STABILITY_TSV.name:
                "encoder rank stability across datasets",
            GROUP_BY_DATASET_TSV.name:
                "descriptive general-purpose versus biomedical group means",
            FIG1D_RSA_TSV.name:
                "Fig.1d-ready corrected RSA matrix",
            FIG1D_GAIN_TSV.name:
                "Fig.1d-ready corrected NDCG@10 gain versus matched Gaussian Random",
            FIG1D_RAW_NDCG_TSV.name:
                "raw NDCG@10 matrix",
            UNIFORM_EXCESS_TSV.name:
                "exact-uniform excess NDCG sensitivity matrix",
            TFIDF_SENS_TSV.name:
                "frozen-R1-code versus current-manuscript TF-IDF specification sensitivity",
        }.get(src.name, ""),
        "public_candidate": True,
        "sha256": sha256_file(src),
    }
    for src in source_data_files
])

source_registry.to_csv(
    figshare_source / "SOURCE_DATA_REGISTRY_primary_paper_closure_v2.tsv",
    sep="\t",
    index=False,
)


print(
    AUDIT_TXT.read_text(
        encoding="utf-8"
    )
)
