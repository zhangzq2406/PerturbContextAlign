#!/usr/bin/env python3
from __future__ import annotations

"""
Build BGE-M3 + P4 normalized full-context semantic embeddings for perturbation AnnData (v4 context-instance corrected).

Main design:
  - Fixed model: BAAI/bge-m3 by default.
  - Fixed prompt: P4 normalized full context prompt.
  - Component-level tokens for compatibility with cross-attention training code.
  - No semantic-response evaluation in this script.
  - Saves pert_key flattened tokens/mask/pooled into adata.obsm.
  - Biological and technical context are folded into each P4 component prompt.

P4 component-level prompt example:
  This is a genetic perturbation condition.
  Mode: CRISPR activation.
  Target gene: KLF1.
  Combination size: 1.
  Biological context: human K562 myelogenous leukemia cell line.
  Technical context: Perturb-seq.
  Dose: not applicable.
  Duration: unknown.

Optional knowledge:
  Use --knowledge-source schema or schema_online to add a short "Additional context" line.
  schema_online can query MyGene.info/PubChem when available, but failures are skipped.
"""

import argparse
import gc
import hashlib
import json
import os
import re
import time
import traceback
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = lambda x, **kwargs: x


# =========================
# CLI / runtime
# =========================


def _normalize_proxy(proxy: str | None) -> str | None:
    if proxy is None:
        return None
    proxy = str(proxy).strip()
    if not proxy:
        return None
    if "://" not in proxy:
        proxy = "http://" + proxy
    return proxy


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build BGE-M3 + P4 normalized full-context component tokens for perturbation AnnData."
    )
    ap.add_argument("--input-h5ad", required=True, help="Input AnnData .h5ad file.")
    ap.add_argument("--output-dir", default="./semantic_embedding_bgem3_p4_output", help="Output directory.")
    ap.add_argument(
        "--output-h5ad",
        default=None,
        help="Output .h5ad path. Default: {output_dir}/adata_with_bgem3_p4_semantic_embeddings.h5ad",
    )

    ap.add_argument("--model-name", default="BAAI/bge-m3", help="Embedding model name; default BAAI/bge-m3.")
    ap.add_argument(
        "--backend",
        choices=["auto", "flagembedding", "sentence_transformers", "transformers"],
        default="auto",
        help="BGE-M3 backend. auto tries FlagEmbedding, then sentence-transformers, then transformers.",
    )
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--cuda-id", default=None, help="Physical CUDA id to expose before torch/model import.")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--save-dtype", choices=["float16", "float32"], default="float16")
    ap.add_argument("--normalize", action="store_true", default=True, help="L2-normalize saved dense embeddings.")
    ap.add_argument("--no-normalize", dest="normalize", action="store_false")
    ap.add_argument("--use-fp16", action="store_true", default=True, help="Use fp16 model inference when backend supports it.")
    ap.add_argument("--no-use-fp16", dest="use_fp16", action="store_false")

    ap.add_argument("--proxy", default=None, help="Optional proxy, e.g. http://127.0.0.1:7897 or 127.0.0.1:7897.")
    ap.add_argument("--no-proxy", action="store_true", help="Do not set proxy env vars even if --proxy is provided.")

    ap.add_argument(
        "--knowledge-source",
        choices=["none", "schema", "schema_online"],
        default="schema_online",
        help=(
            "Additional context source. none=P4 fields only; schema=use existing schema descriptions; "
            "schema_online=also try online MyGene/PubChem enrichment with cache and skip failures."
        ),
    )
    ap.add_argument(
        "--online-timeout",
        type=float,
        default=8.0,
        help="HTTP timeout for optional online enrichment.",
    )
    ap.add_argument(
        "--max-knowledge-chars",
        type=int,
        default=320,
        help="Maximum characters for Additional context line per component.",
    )
    ap.add_argument(
        "--no-online-enrichment",
        action="store_true",
        help="Shortcut: disable online enrichment even if knowledge-source=schema_online.",
    )

    ap.add_argument(
        "--include-batch-in-tech-context",
        action="store_true",
        default=True,
        help="Include C_batch in Technical context if non-empty. Default True.",
    )
    ap.add_argument("--no-include-batch-in-tech-context", dest="include_batch_in_tech_context", action="store_false")
    ap.add_argument(
        "--include-donor-in-bio-context",
        action="store_true",
        default=False,
        help="Include C_context_donor in biological context. Default False because donor IDs are often not semantic.",
    )

    ap.add_argument(
        "--condition-id-key",
        default="C_condition_id",
        help="obs key for condition id.",
    )
    ap.add_argument(
        "--control-condition-id",
        default="control",
        help="Canonical control condition id.",
    )
    ap.add_argument(
        "--write-h5ad",
        action="store_true",
        default=True,
        help="Write output AnnData file. Default True.",
    )
    ap.add_argument("--no-write-h5ad", dest="write_h5ad", action="store_false")
    ap.add_argument(
        "--dry-run-prompts",
        action="store_true",
        help="Build condition/component tables and P4 prompts, save prompt table, then exit before model loading.",
    )
    return ap.parse_args()


ARGS = parse_args()

if ARGS.device == "cuda" and ARGS.cuda_id is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(ARGS.cuda_id)

_proxy = _normalize_proxy(ARGS.proxy)
if _proxy and not ARGS.no_proxy:
    os.environ["http_proxy"] = _proxy
    os.environ["https_proxy"] = _proxy
    os.environ["HTTP_PROXY"] = _proxy
    os.environ["HTTPS_PROXY"] = _proxy

INPUT_H5AD = Path(ARGS.input_h5ad)
OUTPUT_DIR = Path(ARGS.output_dir)
OUTPUT_H5AD = Path(ARGS.output_h5ad) if ARGS.output_h5ad else OUTPUT_DIR / "adata_with_bgem3_p4_semantic_embeddings.h5ad"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("[bgem3-p4] input_h5ad:", INPUT_H5AD, flush=True)
print("[bgem3-p4] output_dir:", OUTPUT_DIR, flush=True)
print("[bgem3-p4] output_h5ad:", OUTPUT_H5AD, flush=True)
print("[bgem3-p4] model:", ARGS.model_name, "backend:", ARGS.backend, flush=True)
print("[bgem3-p4] device:", ARGS.device, "cuda_id:", ARGS.cuda_id, "CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)
print("[bgem3-p4] knowledge_source:", ARGS.knowledge_source, "online_disabled:", ARGS.no_online_enrichment, flush=True)


# =========================
# AnnData schema / obs keys
# =========================

SCHEMA_KEY_CANDIDATES = {
    "condition": ["C_condition_schema", "C_condition_schema_json"],
    "component": ["C_component_schema", "C_component_schema_json"],
    "entity": ["C_entity_schema", "C_entity_schema_json"],
    "prompt": ["C_prompt_schema", "C_prompt_schema_json"],
}

OBS_KEYS = {
    "control": "C_control_indicator",
    "condition_id": ARGS.condition_id_key,
    "condition_label": "C_condition_label",
    "component_ids": "C_perturbation_component_ids",
    "family": "C_perturbation_family",
    "mode": "C_perturbation_mode",
    "agent": "C_perturbation_agent",
    "target": "C_perturbation_target",
    "combo": "C_perturbation_combo",
    "combo_n": "C_perturbation_combo_n",
    "dose": "C_perturbation_dose",
    "duration": "C_perturbation_duration",
    "species": "C_context_species",
    "cell_type": "C_context_cell_type",
    "cell_line": "C_context_cell_line",
    "donor": "C_context_donor",
    "tissue": "C_context_tissue",
    "disease": "C_context_disease",
    "batch": "C_batch",
    "platform": "C_platform",
}

BIO_CONTEXT_FIELDS = [
    "C_context_species",
    "C_context_cell_line",
    "C_context_cell_type",
    "C_context_tissue",
    "C_context_disease",
]
if ARGS.include_donor_in_bio_context:
    BIO_CONTEXT_FIELDS.append("C_context_donor")


# =========================
# Utilities
# =========================


def is_missing(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and np.isnan(x):
        return True
    if isinstance(x, (np.floating,)) and np.isnan(float(x)):
        return True
    s = str(x).strip()
    return s == "" or s.lower() in {"nan", "none", "na", "n/a", "null", "unknown", "unk"}


def as_str(x: Any, default: str = "") -> str:
    return default if is_missing(x) else str(x).strip()


def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def parse_json_like(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, (dict, list)):
        return x
    if isinstance(x, bytes):
        x = x.decode("utf-8")
    if isinstance(x, np.ndarray):
        if x.shape == ():
            x = x.item()
        elif x.size == 1:
            x = x.ravel()[0]
        else:
            return x.tolist()
    if isinstance(x, np.bytes_):
        x = x.decode("utf-8")
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except Exception:
            try:
                return json.loads(s.replace("'", '"'))
            except Exception:
                return s
    return x


def get_uns_json(adata: ad.AnnData, key: str) -> Any:
    return parse_json_like(adata.uns[key]) if key in adata.uns else None


def get_uns_json_any(adata: ad.AnnData, keys: Iterable[str]) -> Any:
    for key in keys:
        if key in adata.uns:
            return parse_json_like(adata.uns[key])
    return None


def normalize_schema_mapping(schema: Any, id_candidates: List[str]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if schema is None:
        return out
    if isinstance(schema, dict):
        for nested_key in ["items", "records", "components", "conditions", "entities", "data"]:
            if nested_key in schema and isinstance(schema[nested_key], (list, dict)):
                return normalize_schema_mapping(schema[nested_key], id_candidates)
        for k, v in schema.items():
            if isinstance(v, dict):
                rec = dict(v)
                rec.setdefault("id", str(k))
                out[str(k)] = rec
            else:
                out[str(k)] = {"id": str(k), "value": v}
        return out
    if isinstance(schema, list):
        for i, rec in enumerate(schema):
            if not isinstance(rec, dict):
                out[str(i)] = {"id": str(i), "value": rec}
                continue
            rid = None
            for c in id_candidates:
                if c in rec and not is_missing(rec[c]):
                    rid = str(rec[c])
                    break
            if rid is None:
                rid = str(i)
            new_rec = dict(rec)
            new_rec.setdefault("id", rid)
            out[rid] = new_rec
    return out


def split_component_ids(x: Any) -> List[str]:
    if is_missing(x):
        return []
    if isinstance(x, (list, tuple, np.ndarray, pd.Series)):
        return [str(v).strip() for v in list(x) if not is_missing(v)]
    s = str(x).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            arr = json.loads(s)
            if isinstance(arr, list):
                return [str(v).strip() for v in arr if not is_missing(v)]
        except Exception:
            pass
    for sep in ["|", ";", ","]:
        if sep in s:
            return [v.strip() for v in s.split(sep) if v.strip()]
    if "+" in s:
        return [v.strip() for v in s.split("+") if v.strip()]
    return [s]


def bool_from_any(x: Any) -> bool:
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if is_missing(x):
        return False
    s = str(x).strip().lower()
    return s in {"1", "true", "t", "yes", "y", "control", "ctrl"}


def first_present(d: Dict[str, Any], keys: Iterable[str], default=None):
    for k in keys:
        if k in d and not is_missing(d[k]):
            return d[k]
    return default


def get_row_value(row: pd.Series, key: str, default=None):
    return row[key] if key in row.index else default


# =========================
# Optional online knowledge cache
# =========================


class OnlineKnowledgeCache:
    def __init__(self, path: Path, timeout: float = 8.0, enabled: bool = True):
        self.path = path
        self.timeout = float(timeout)
        self.enabled = bool(enabled)
        self.cache: Dict[str, Any] = {}
        if self.path.exists():
            try:
                self.cache = json.loads(self.path.read_text())
            except Exception:
                self.cache = {}

    def save(self):
        try:
            self.path.write_text(json.dumps(self.cache, indent=2, ensure_ascii=False))
        except Exception as e:
            print("[WARN] failed to save online cache:", repr(e), flush=True)

    def fetch_json(self, key: str, url: str) -> Optional[Any]:
        if not self.enabled:
            return None
        if key in self.cache:
            val = self.cache[key]
            return None if val == "__FAILED__" else val
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "perturb-semantic-bgem3-p4/1.0"})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            self.cache[key] = data
            # be gentle with public APIs
            time.sleep(0.05)
            return data
        except Exception as e:
            self.cache[key] = "__FAILED__"
            print(f"[WARN] online enrichment failed for {key}: {repr(e)}", flush=True)
            return None


online_cache = OnlineKnowledgeCache(
    OUTPUT_DIR / "online_knowledge_cache.json",
    timeout=ARGS.online_timeout,
    enabled=(ARGS.knowledge_source == "schema_online" and not ARGS.no_online_enrichment),
)


def short_text(s: str, max_chars: int) -> str:
    s = clean_text(s)
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3].rstrip() + "..."


def query_mygene_summary(symbol: str, species: str = "human") -> str:
    symbol = as_str(symbol, "")
    if not symbol:
        return ""
    q = urllib.parse.quote(f"symbol:{symbol}")
    spname = "human" if not species or "human" in str(species).lower() else str(species).lower()
    url = f"https://mygene.info/v3/query?q={q}&species={urllib.parse.quote(spname)}&fields=symbol,name,summary,entrezgene&size=1"
    data = online_cache.fetch_json(f"mygene::{spname}::{symbol}", url)
    try:
        hits = data.get("hits", []) if isinstance(data, dict) else []
        if hits:
            hit = hits[0]
            parts = []
            if hit.get("name"):
                parts.append(str(hit["name"]))
            if hit.get("summary"):
                parts.append(str(hit["summary"]))
            return clean_text(" ".join(parts))
    except Exception:
        pass
    return ""


def query_pubchem_description(name: str) -> str:
    name = as_str(name, "")
    if not name:
        return ""
    enc = urllib.parse.quote(name)
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{enc}/description/JSON"
    data = online_cache.fetch_json(f"pubchem::{name}", url)
    try:
        infos = data.get("InformationList", {}).get("Information", []) if isinstance(data, dict) else []
        descs = [str(x.get("Description", "")) for x in infos if x.get("Description")]
        if descs:
            return clean_text(descs[0])
    except Exception:
        pass
    return ""


# =========================
# Condition/component parsing
# =========================


adata = ad.read_h5ad(INPUT_H5AD)
print(adata, flush=True)
print("obs columns:", list(adata.obs.columns), flush=True)
print("uns keys:", list(adata.uns.keys()), flush=True)
print("obsm keys:", list(adata.obsm.keys()), flush=True)
print("layers:", list(adata.layers.keys()), flush=True)

condition_schema = get_uns_json_any(adata, SCHEMA_KEY_CANDIDATES["condition"])
component_schema = get_uns_json_any(adata, SCHEMA_KEY_CANDIDATES["component"])
entity_schema = get_uns_json_any(adata, SCHEMA_KEY_CANDIDATES["entity"])
prompt_schema = get_uns_json_any(adata, SCHEMA_KEY_CANDIDATES["prompt"])

condition_schema_map = normalize_schema_mapping(condition_schema, ["condition_id", "id", "name", "label", "condition"])
component_schema_map = normalize_schema_mapping(component_schema, ["component_id", "id", "name", "label", "component"])
entity_schema_map = normalize_schema_mapping(entity_schema, ["entity_id", "id", "name", "label", "symbol"])

print("condition_schema_map:", len(condition_schema_map), flush=True)
print("component_schema_map:", len(component_schema_map), flush=True)
print("entity_schema_map:", len(entity_schema_map), flush=True)


def infer_condition_id_from_row(row: pd.Series) -> str:
    control_key = OBS_KEYS["control"]
    if control_key in row.index and bool_from_any(row[control_key]):
        return ARGS.control_condition_id
    for key in [OBS_KEYS["condition_id"], OBS_KEYS["condition_label"], OBS_KEYS["target"], OBS_KEYS["agent"]]:
        if key in row.index and not is_missing(row[key]):
            return str(row[key]).strip()
    family = as_str(get_row_value(row, OBS_KEYS["family"], "unknown"), "unknown")
    mode = as_str(get_row_value(row, OBS_KEYS["mode"], "unknown"), "unknown")
    target = as_str(get_row_value(row, OBS_KEYS["target"], "unknown"), "unknown")
    return f"{family}:{mode}:{target}"


def condition_record_component_ids(cond_rec: Dict[str, Any]) -> List[str]:
    for key in [
        "component_ids",
        "components",
        "component_list",
        "C_perturbation_component_ids",
        "perturbation_component_ids",
        "targets",
        "target_genes",
    ]:
        if key in cond_rec and not is_missing(cond_rec[key]):
            val = cond_rec[key]
            if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
                out = []
                for i, r in enumerate(val):
                    out.append(str(first_present(r, ["component_id", "id", "name", "label", "component"], f"component_{i}")))
                return out
            return split_component_ids(val)
    return []


def parse_combo_n(x: Any, component_ids: List[str]) -> int:
    if not is_missing(x):
        try:
            return int(float(str(x)))
        except Exception:
            pass
    return int(len(component_ids))


def _stable_semantic_instance_id(condition_id: str, row: pd.Series, is_control: bool) -> str:
    if is_control:
        return ARGS.control_condition_id

    fields = list(BIO_CONTEXT_FIELDS) + [OBS_KEYS["platform"]]
    if ARGS.include_batch_in_tech_context:
        fields.append(OBS_KEYS["batch"])

    payload_parts = [f"condition_id={condition_id}"]
    for field in fields:
        value = row[field] if field in row.index else ""
        payload_parts.append(f"{field}={as_str(value, '')}")
    payload = "\x1f".join(payload_parts)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"{condition_id}::p4ctx::{digest}"


def build_condition_table_from_adata(adata: ad.AnnData) -> pd.DataFrame:
    """Build one row per semantic P4 instance rather than per C_condition_id.

    C_condition_id remains the perturbation identity. A semantic instance is the
    perturbation condition crossed with the biological/technical fields actually
    used by P4. Control remains one all-zero semantic instance.
    """
    obs = adata.obs.copy()
    condition_ids_per_cell = obs.apply(infer_condition_id_from_row, axis=1).astype(str)
    adata.obs["_semantic_condition_id"] = condition_ids_per_cell.values
    obs["_semantic_condition_id"] = condition_ids_per_cell.values

    control_flags = []
    instance_ids = []
    for _, row in obs.iterrows():
        cond_id = str(row["_semantic_condition_id"])
        is_control = False
        if OBS_KEYS["control"] in row.index:
            is_control = bool_from_any(row[OBS_KEYS["control"]])
        if cond_id.strip().lower() in {"control", "ctrl", "non-targeting", "non_targeting", "nt"}:
            is_control = True
        control_flags.append(bool(is_control))
        instance_ids.append(_stable_semantic_instance_id(cond_id, row, is_control))

    obs["_semantic_is_control"] = np.asarray(control_flags, dtype=bool)
    obs["_semantic_instance_id"] = np.asarray(instance_ids, dtype=object)
    adata.obs["_semantic_instance_id"] = obs["_semantic_instance_id"].astype(str).values

    rows = []
    for instance_id, sub_obs in obs.groupby("_semantic_instance_id", sort=True):
        first_row = sub_obs.iloc[0]
        cond_id = str(first_row["_semantic_condition_id"])
        is_control = bool(first_row["_semantic_is_control"])

        cond_rec = condition_schema_map.get(cond_id, {})
        component_ids = condition_record_component_ids(cond_rec) if cond_rec else []

        if not component_ids and OBS_KEYS["component_ids"] in first_row.index:
            component_ids = split_component_ids(first_row[OBS_KEYS["component_ids"]])

        if not component_ids and not is_control:
            component_ids = (
                split_component_ids(get_row_value(first_row, OBS_KEYS["target"], None))
                or split_component_ids(get_row_value(first_row, OBS_KEYS["agent"], None))
                or [cond_id]
            )

        if is_control:
            component_ids = []

        combo_n = parse_combo_n(get_row_value(first_row, OBS_KEYS["combo_n"], None), component_ids)
        if is_control:
            combo_n = 0

        rows.append(
            {
                "semantic_instance_id": str(instance_id),
                "condition_id": cond_id,
                "is_control": bool(is_control),
                "n_cells": int(len(sub_obs)),
                "component_ids": component_ids,
                "condition_schema_found": bool(cond_rec),
                "family": as_str(get_row_value(first_row, OBS_KEYS["family"], None)),
                "mode": as_str(get_row_value(first_row, OBS_KEYS["mode"], None)),
                "agent": as_str(get_row_value(first_row, OBS_KEYS["agent"], None)),
                "target": as_str(get_row_value(first_row, OBS_KEYS["target"], None)),
                "combo_n": int(combo_n),
                "dose": as_str(get_row_value(first_row, OBS_KEYS["dose"], None)),
                "duration": as_str(get_row_value(first_row, OBS_KEYS["duration"], None)),
                "species": as_str(get_row_value(first_row, OBS_KEYS["species"], None)),
                "cell_type": as_str(get_row_value(first_row, OBS_KEYS["cell_type"], None)),
                "cell_line": as_str(get_row_value(first_row, OBS_KEYS["cell_line"], None)),
                "donor": as_str(get_row_value(first_row, OBS_KEYS["donor"], None)),
                "tissue": as_str(get_row_value(first_row, OBS_KEYS["tissue"], None)),
                "disease": as_str(get_row_value(first_row, OBS_KEYS["disease"], None)),
                "platform": as_str(get_row_value(first_row, OBS_KEYS["platform"], None)),
                "batch": as_str(get_row_value(first_row, OBS_KEYS["batch"], None)),
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["is_control", "condition_id", "semantic_instance_id"]
    ).reset_index(drop=True)


condition_df = build_condition_table_from_adata(adata)
condition_df.to_csv(OUTPUT_DIR / "semantic_instance_table_bgem3_p4.csv", index=False)
condition_df.to_csv(OUTPUT_DIR / "condition_table_bgem3_p4.csv", index=False)
print(condition_df.head().to_string(max_cols=120), flush=True)
print("n_semantic_instances:", len(condition_df), flush=True)
print("n_unique_conditions:", int(condition_df["condition_id"].astype(str).nunique()), flush=True)
print("n_control_instances:", int(condition_df["is_control"].sum()), flush=True)
print("max components:", max([len(x) for x in condition_df["component_ids"]], default=0), flush=True)


def infer_component_record(component_id: str, condition_row: Optional[pd.Series] = None) -> Dict[str, Any]:
    cid = str(component_id)
    rec: Dict[str, Any] = {}

    if cid in component_schema_map:
        rec.update(component_schema_map[cid])
    elif cid in entity_schema_map:
        rec.update(entity_schema_map[cid])
    else:
        lower_comp = {str(k).lower(): k for k in component_schema_map.keys()}
        lower_ent = {str(k).lower(): k for k in entity_schema_map.keys()}
        if cid.lower() in lower_comp:
            rec.update(component_schema_map[lower_comp[cid.lower()]])
        elif cid.lower() in lower_ent:
            rec.update(entity_schema_map[lower_ent[cid.lower()]])

    if condition_row is not None:
        # Prefer schema values, fill missing from condition-level obs values.
        rec.setdefault("family", condition_row.get("family", ""))
        rec.setdefault("mode", condition_row.get("mode", ""))
        rec.setdefault("agent", condition_row.get("agent", ""))
        rec.setdefault("target", cid if not is_missing(cid) else condition_row.get("target", ""))
        rec.setdefault("dose", condition_row.get("dose", ""))
        rec.setdefault("duration", condition_row.get("duration", ""))

    rec.setdefault("component_id", cid)
    rec.setdefault("id", cid)

    # Common aliases.
    if "gene" in rec and "target" not in rec:
        rec["target"] = rec["gene"]
    if "gene_symbol" in rec and "target" not in rec:
        rec["target"] = rec["gene_symbol"]
    if "symbol" in rec and "target" not in rec:
        rec["target"] = rec["symbol"]
    if "entity_name" in rec and "agent" not in rec:
        rec["agent"] = rec["entity_name"]
    return rec


all_component_ids: List[str] = []
for _, row in condition_df.iterrows():
    all_component_ids.extend(list(map(str, row["component_ids"])))
all_component_ids = sorted(set(all_component_ids))

cond_row_containing_component: Dict[str, pd.Series] = {}
for _, row in condition_df.iterrows():
    for cid in list(map(str, row["component_ids"])):
        cond_row_containing_component.setdefault(cid, row)

component_records = {
    cid: infer_component_record(cid, cond_row_containing_component.get(cid))
    for cid in all_component_ids
}

print("n_unique_components:", len(component_records), flush=True)
for cid in list(component_records.keys())[:5]:
    print("--- component", cid, component_records[cid], flush=True)


# =========================
# P4 prompt construction
# =========================


def component_family(rec: Dict[str, Any], condition_row: Optional[pd.Series] = None) -> str:
    val = first_present(rec, ["family", "perturbation_family", "C_perturbation_family", "type", "entity_type"], None)
    if is_missing(val) and condition_row is not None:
        val = condition_row.get("family", "")
    return str(val or "unknown").lower().strip()


def component_mode(rec: Dict[str, Any], condition_row: Optional[pd.Series] = None) -> str:
    val = first_present(rec, ["mode", "perturbation_mode", "C_perturbation_mode", "operation"], None)
    if is_missing(val) and condition_row is not None:
        val = condition_row.get("mode", "")
    return as_str(val, "perturbation")


_COMPONENT_PREFIXES = {
    "genetic", "gene", "crispr", "rnai", "grna", "guide",
    "drug", "compound", "small_molecule", "small-molecule",
    "cytokine", "ligand", "growth_factor", "growth-factor",
    "environment", "environmental", "stress", "exposure",
    "cytokine_ligand_growth_factor", "environmental_toxicant",
    "infection", "immune_stimulus", "vaccination", "immune_checkpoint_blockade",
}

_MODE_SUFFIXES = {
    "activation", "inhibition", "knockout", "ko", "oe", "overexpression",
    "crispr_activation", "crispr_inhibition", "treatment", "stimulation",
    "perturbation", "exposure",
}


def clean_prompt_value(x: Any) -> str:
    """Human-readable scalar value normalization."""
    s = as_str(x, "")
    if not s:
        return s
    low = s.lower().strip()
    replacements = {
        "not_applicable": "not applicable",
        "not-applicable": "not applicable",
        "not applicable": "not applicable",
        "n/a": "not applicable",
        "na": "not applicable",
        "none": "not applicable",
        "nan": "unknown",
        "unk": "unknown",
    }
    return replacements.get(low, s.replace("not_applicable", "not applicable"))


def clean_context_value(x: Any) -> str:
    """
    Normalize schema-derived context strings for natural-language prompts.

    Many context-modeled datasets use placeholders such as
    `not_applicable_primary_cell` or compact strings such as
    `species=mouse|tissue=bone_marrow`. These should not be embedded
    literally as biomedical text.
    """
    s = as_str(x, "")
    if not s:
        return ""
    low = s.lower().strip()
    if low in {"not_applicable", "not-applicable", "not applicable", "n/a", "na", "none", "nan", "unknown", "unk"}:
        return ""

    # Remove placeholder prefix while preserving useful descriptors after it.
    s = re.sub(r"(?i)^not[_ -]?applicable[_ -]*", "", s).strip(" _-;|")
    s = re.sub(r"(?i)\bnot[_ -]?applicable[_ -]*", "", s).strip(" _-;|")

    # Humanize compact key/value and underscore notation for context fields.
    s = s.replace("|", "; ")
    s = re.sub(r"\s*=\s*", ": ", s)
    s = re.sub(r"_+", " ", s)
    s = re.sub(r"\s*;\s*", "; ", s)
    s = clean_text(s)
    return s.strip(" ;,")


def clean_entity_display_name(name: Any, fam: str = "") -> str:
    """Return a human-readable entity name for prompt text.

    Gene symbols are preserved exactly. Non-gene intervention names are
    lightly humanized so structured IDs such as `Heligmosomoides_polygyrus`
    become `Heligmosomoides polygyrus`, while names like anti-PD-1 remain intact.
    """
    s = as_str(name, "unknown")
    if not s:
        return "unknown"
    if is_genetic_family(fam):
        return s
    s = re.sub(r"_+", " ", s)
    return clean_text(s)


def parse_entity_from_component_id(x: Any) -> str:
    """
    Extract the intervention entity from unified component IDs.

    Examples:
      genetic:AHR:activation                                      -> AHR
      drug:trametinib:treatment                                   -> trametinib
      cytokine_ligand_growth_factor:IFN-beta:stimulation          -> IFN-beta
      environmental_toxicant:TCDD:AHR_ligand:30_ug_per_kg:28_days -> TCDD
      immune_stimulus:LPS:TLR4:duration_2h                        -> LPS
      infection:Salmonella:bacterial_infection:dose_unknown       -> Salmonella
      immune_checkpoint_blockade:anti-PD-1:PD-1_blockade          -> anti-PD-1
      genetic__AHR__activation                                    -> AHR
      AHR_activation                                              -> AHR
    """
    s = as_str(x, "")
    if not s:
        return ""

    # General colon grammar: family:entity:...
    # Use the second field as the biological intervention entity.
    parts = [p.strip() for p in s.split(":") if p.strip()]
    if len(parts) >= 2:
        first = parts[0].lower()
        # Accept both known and newly added perturbation families. This is safe
        # because component IDs in this project are intentionally schema-structured.
        if first in _COMPONENT_PREFIXES or re.fullmatch(r"[a-z][a-z0-9_\-]*", first):
            return parts[1]

    # double-underscore grammar: family__entity__mode
    parts2 = [p.strip() for p in s.split("__") if p.strip()]
    if len(parts2) >= 3 and (parts2[0].lower() in _COMPONENT_PREFIXES or re.fullmatch(r"[a-z][a-z0-9_\-]*", parts2[0].lower())):
        return parts2[1]

    s2 = s

    # Remove common mode suffixes from target-like strings, but avoid changing
    # gene symbols with underscores unless the suffix is known.
    for suffix in sorted(_MODE_SUFFIXES, key=len, reverse=True):
        for sep in ["_", "-"]:
            tail = sep + suffix
            if s2.lower().endswith(tail):
                return s2[: -len(tail)]

    return s2


def component_name(rec: Dict[str, Any], condition_row: Optional[pd.Series] = None) -> str:
    """Resolve intervention entity with family-aware semantics."""
    fam = component_family(rec, condition_row)

    if is_genetic_family(fam):
        preferred = ["target", "gene", "gene_symbol", "symbol", "name", "label", "entity_name", "agent"]
    elif is_drug_family(fam) or is_cytokine_family(fam) or is_environment_family(fam) or any(
        x in fam.lower() for x in ["immune", "infection", "checkpoint"]
    ):
        preferred = ["agent", "drug", "ligand", "factor", "entity_name", "name", "label", "target"]
    else:
        preferred = ["agent", "target", "name", "label", "symbol", "gene", "gene_symbol", "drug", "ligand", "factor", "entity_name"]

    val = first_present(rec, preferred, None)
    cid = first_present(rec, ["component_id", "id"], None)

    if not is_missing(val):
        val_s = as_str(val)
        if not is_missing(cid) and val_s == as_str(cid):
            parsed = parse_entity_from_component_id(cid)
            if parsed:
                return parsed
        parsed_val = parse_entity_from_component_id(val_s)
        return as_str(parsed_val, val_s)

    if not is_missing(cid):
        parsed = parse_entity_from_component_id(cid)
        if parsed:
            return parsed

    if condition_row is not None:
        row_val = condition_row.get("target", None) if is_genetic_family(fam) else condition_row.get("agent", None)
        if is_missing(row_val):
            row_val = condition_row.get("agent", None) or condition_row.get("target", None)
        if not is_missing(row_val):
            return as_str(parse_entity_from_component_id(row_val), as_str(row_val))

    return "unknown"


def schema_knowledge_text(rec: Dict[str, Any]) -> str:
    keys = [
        "summary",
        "description",
        "gene_summary",
        "function",
        "gene_function",
        "biological_function",
        "mechanism",
        "mechanism_of_action",
        "moa",
        "pathway_summary",
        "drug_targets",
        "targets",
        "known_targets",
        "molecular_targets",
        "receptor",
        "receptors",
        "pathway",
        "signaling_pathway",
        "response_pathway",
    ]
    vals = []
    for k in keys:
        if k in rec and not is_missing(rec[k]):
            v = rec[k]
            if isinstance(v, list):
                v = ", ".join(map(str, v))
            vals.append(str(v))
    return clean_text(" ".join(vals))


def is_genetic_family(fam: str) -> bool:
    f = fam.lower()
    return f in {"genetic", "gene", "crispr", "rnai", "grna", "guide"} or "gene" in f or "crispr" in f


def is_drug_family(fam: str) -> bool:
    f = fam.lower()
    return f in {"drug", "compound", "small_molecule", "small-molecule"} or "drug" in f or "compound" in f


def is_cytokine_family(fam: str) -> bool:
    f = fam.lower()
    return f in {"cytokine", "ligand", "growth_factor", "growth-factor"} or "cytokine" in f or "ligand" in f


def is_environment_family(fam: str) -> bool:
    f = fam.lower()
    return f in {"environment", "environmental", "stress", "exposure"} or "environment" in f or "stress" in f


def family_sentence(fam: str) -> str:
    if is_genetic_family(fam):
        return "This is a genetic perturbation condition."
    if is_drug_family(fam):
        return "This is a drug perturbation condition."
    if is_cytokine_family(fam):
        return "This is a cytokine or ligand stimulation condition."
    if is_environment_family(fam):
        return "This is an environmental perturbation condition."
    fam_txt = clean_context_value(fam) or "unknown"
    article = "an" if fam_txt[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
    return f"This is {article} {fam_txt} perturbation condition."


def target_line(fam: str, name: str) -> str:
    if is_genetic_family(fam):
        return f"Target gene: {name}."
    if is_drug_family(fam):
        return f"Drug or compound: {name}."
    if is_cytokine_family(fam):
        return f"Cytokine or ligand: {name}."
    if is_environment_family(fam):
        return f"Environmental factor: {name}."
    return f"Intervention entity: {name}."


def build_biological_context(condition_row: pd.Series) -> str:
    species = clean_context_value(condition_row.get("species", ""))
    cell_line = clean_context_value(condition_row.get("cell_line", ""))
    cell_type = clean_context_value(condition_row.get("cell_type", ""))
    tissue = clean_context_value(condition_row.get("tissue", ""))
    disease = clean_context_value(condition_row.get("disease", ""))
    donor = clean_context_value(condition_row.get("donor", "")) if ARGS.include_donor_in_bio_context else ""

    chunks: List[str] = []
    # Compact natural order: human K562 cell line / T cell / tissue / disease.
    if species:
        chunks.append(species)
    if cell_line:
        chunks.append(cell_line)
    if cell_type and cell_type.lower() != cell_line.lower():
        chunks.append(cell_type)
    if tissue:
        chunks.append(f"from {tissue}")
    if disease:
        chunks.append(f"disease context: {disease}")
    if donor:
        chunks.append(f"donor: {donor}")
    return clean_text(" ".join(chunks)) if chunks else "unknown biological context"


def build_technical_context(condition_row: pd.Series) -> str:
    platform = clean_context_value(condition_row.get("platform", ""))
    batch = as_str(condition_row.get("batch", ""))
    chunks = []
    if platform:
        chunks.append(platform)
    if ARGS.include_batch_in_tech_context and batch:
        # Batch IDs are identifiers; keep them exact except for not_applicable placeholders.
        batch_clean = clean_prompt_value(batch)
        if batch_clean and batch_clean != "not applicable":
            chunks.append(f"batch: {batch_clean}")
    return clean_text("; ".join(chunks)) if chunks else "unknown technical context"


def dose_value(fam: str, condition_row: pd.Series, rec: Dict[str, Any]) -> str:
    val = first_present(rec, ["dose", "C_perturbation_dose"], None)
    if is_missing(val):
        val = condition_row.get("dose", "")
    if not is_missing(val):
        return clean_prompt_value(val)
    if is_genetic_family(fam):
        return "not applicable"
    return "unknown"


def duration_value(condition_row: pd.Series, rec: Dict[str, Any]) -> str:
    val = first_present(rec, ["duration", "C_perturbation_duration"], None)
    if is_missing(val):
        val = condition_row.get("duration", "")
    return clean_prompt_value(val) or "unknown"


def online_knowledge_text(rec: Dict[str, Any], condition_row: pd.Series) -> str:
    if ARGS.knowledge_source != "schema_online" or ARGS.no_online_enrichment:
        return ""
    fam = component_family(rec, condition_row)
    name = component_name(rec, condition_row)
    species = as_str(condition_row.get("species", "human"), "human")
    if is_genetic_family(fam):
        return query_mygene_summary(name, species=species)
    if is_drug_family(fam):
        return query_pubchem_description(name)
    return ""


def additional_knowledge(rec: Dict[str, Any], condition_row: pd.Series) -> str:
    if ARGS.knowledge_source == "none":
        return ""
    texts = []
    schema_txt = schema_knowledge_text(rec)
    if schema_txt:
        texts.append(schema_txt)
    online_txt = online_knowledge_text(rec, condition_row)
    if online_txt:
        texts.append(online_txt)
    return short_text(" ".join(texts), ARGS.max_knowledge_chars)


def build_p4_component_prompt(component_id: str, rec: Dict[str, Any], condition_row: pd.Series) -> str:
    fam = component_family(rec, condition_row)
    mode = clean_prompt_value(component_mode(rec, condition_row))
    name = clean_entity_display_name(component_name(rec, condition_row), fam)
    combo_n = int(condition_row.get("combo_n", 0) or 0)
    bio = build_biological_context(condition_row)
    tech = build_technical_context(condition_row)
    dose = dose_value(fam, condition_row, rec)
    duration = duration_value(condition_row, rec)

    lines = [
        family_sentence(fam),
        f"Mode: {mode}.",
        target_line(fam, name),
        f"Combination size: {combo_n}.",
        f"Biological context: {bio}.",
        f"Technical context: {tech}.",
        f"Dose: {dose}.",
        f"Duration: {duration}.",
    ]
    know = additional_knowledge(rec, condition_row)
    if know:
        lines.append(f"Additional context: {know}.")
    return clean_text(" ".join(lines))


prompt_rows: List[Dict[str, Any]] = []
component_prompt_by_instance: Dict[Tuple[str, str], str] = {}
all_prompts: List[str] = []

for _, row in tqdm(condition_df.iterrows(), total=len(condition_df), desc="build P4 prompts"):
    instance_id = str(row["semantic_instance_id"])
    cond_id = str(row["condition_id"])
    for cid in list(map(str, row["component_ids"])):
        rec = component_records.get(cid) or infer_component_record(cid, row)
        prompt = build_p4_component_prompt(cid, rec, row)
        component_prompt_by_instance[(instance_id, cid)] = prompt
        all_prompts.append(prompt)
        prompt_rows.append(
            {
                "semantic_instance_id": instance_id,
                "condition_id": cond_id,
                "component_id": cid,
                "prompt_type": "p4_normalized_full_context",
                "model_name": ARGS.model_name,
                "prompt": prompt,
            }
        )

prompt_df = pd.DataFrame(prompt_rows)
prompt_df.to_csv(OUTPUT_DIR / "p4_component_prompts.csv", index=False)
print("n_component_prompts:", len(prompt_df), "n_unique_texts:", len(set(all_prompts)), flush=True)
print(prompt_df.head(10).to_string(max_colwidth=180), flush=True)

if ARGS.dry_run_prompts:
    print("[bgem3-p4] dry run done; exiting before model loading.", flush=True)
    online_cache.save()
    raise SystemExit(0)


# =========================
# BGE-M3 encoder
# =========================


class BGEM3Encoder:
    def __init__(self):
        self.model_name = ARGS.model_name
        self.backend_requested = ARGS.backend
        self.backend_used = None
        self.model = None
        self.tokenizer = None
        self.torch = None
        self.device = "cpu"
        self.dim: Optional[int] = None
        self._load()

    def _load(self):
        backends = [self.backend_requested] if self.backend_requested != "auto" else ["flagembedding", "sentence_transformers", "transformers"]
        errors = []
        for backend in backends:
            try:
                if backend == "flagembedding":
                    from FlagEmbedding import BGEM3FlagModel

                    use_fp16 = bool(ARGS.use_fp16 and ARGS.device == "cuda")
                    self.model = BGEM3FlagModel(self.model_name, use_fp16=use_fp16)
                    self.backend_used = "flagembedding"
                    # Dimension known after first encode.
                    print("[bgem3-p4] loaded with FlagEmbedding", flush=True)
                    return

                if backend == "sentence_transformers":
                    from sentence_transformers import SentenceTransformer
                    import torch

                    dev = ARGS.device if ARGS.device == "cuda" and torch.cuda.is_available() else "cpu"
                    self.device = dev
                    self.model = SentenceTransformer(self.model_name, device=dev)
                    self.dim = int(self.model.get_sentence_embedding_dimension())
                    self.backend_used = "sentence_transformers"
                    print("[bgem3-p4] loaded with sentence-transformers on", dev, flush=True)
                    return

                if backend == "transformers":
                    import torch
                    from transformers import AutoModel, AutoTokenizer

                    dev = ARGS.device if ARGS.device == "cuda" and torch.cuda.is_available() else "cpu"
                    self.torch = torch
                    self.device = dev
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                    self.model = AutoModel.from_pretrained(self.model_name)
                    self.model.to(dev)
                    self.model.eval()
                    self.dim = int(self.model.config.hidden_size)
                    self.backend_used = "transformers"
                    print("[bgem3-p4] loaded with transformers mean pooling on", dev, flush=True)
                    return

                raise ValueError(f"unknown backend {backend}")
            except Exception as e:
                errors.append((backend, repr(e), traceback.format_exc()))
                print(f"[WARN] failed loading backend={backend}: {repr(e)}", flush=True)

        msg = "Could not load BGE-M3 with requested backends.\n" + "\n".join([f"{b}: {e}" for b, e, _ in errors])
        raise RuntimeError(msg)

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        denom = np.linalg.norm(arr, axis=1, keepdims=True).clip(min=1e-8)
        return arr / denom

    def encode(self, texts: List[str]) -> np.ndarray:
        texts = [str(t) for t in texts]
        if len(texts) == 0:
            return np.zeros((0, int(self.dim or 0)), dtype="float32")

        if self.backend_used == "flagembedding":
            outputs = []
            for start in tqdm(range(0, len(texts), ARGS.batch_size), desc="encode:bge-m3"):
                batch = texts[start : start + ARGS.batch_size]
                out = self.model.encode(
                    batch,
                    batch_size=ARGS.batch_size,
                    max_length=ARGS.max_length,
                    return_dense=True,
                    return_sparse=False,
                    return_colbert_vecs=False,
                )
                arr = np.asarray(out["dense_vecs"], dtype="float32")
                outputs.append(arr)
            arr = np.concatenate(outputs, axis=0).astype("float32")
            if ARGS.normalize:
                arr = self._normalize(arr).astype("float32")
            self.dim = int(arr.shape[1])
            return arr

        if self.backend_used == "sentence_transformers":
            arr = self.model.encode(
                texts,
                batch_size=ARGS.batch_size,
                normalize_embeddings=ARGS.normalize,
                convert_to_numpy=True,
                show_progress_bar=True,
            ).astype("float32")
            self.dim = int(arr.shape[1])
            return arr

        if self.backend_used == "transformers":
            torch = self.torch
            outputs = []
            with torch.no_grad():
                for start in tqdm(range(0, len(texts), ARGS.batch_size), desc="encode:bge-m3-transformers"):
                    batch = texts[start : start + ARGS.batch_size]
                    enc = self.tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=ARGS.max_length,
                        return_tensors="pt",
                    )
                    enc = {k: v.to(self.device) for k, v in enc.items()}
                    out = self.model(**enc)
                    last = out.last_hidden_state
                    attn = enc["attention_mask"].unsqueeze(-1).float()
                    emb = (last * attn).sum(dim=1) / attn.sum(dim=1).clamp_min(1.0)
                    if ARGS.normalize:
                        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                    outputs.append(emb.detach().cpu().float().numpy())
            arr = np.concatenate(outputs, axis=0).astype("float32")
            self.dim = int(arr.shape[1])
            return arr

        raise RuntimeError("Encoder not loaded correctly.")

    def release(self):
        try:
            del self.model
            del self.tokenizer
        except Exception:
            pass
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


# =========================
# Encode unique prompts and assemble condition tokens
# =========================


encoder = BGEM3Encoder()
unique_prompts = sorted(set(all_prompts))
print("Encoding unique P4 prompts:", len(unique_prompts), flush=True)
emb_arr = encoder.encode(unique_prompts)
prompt_to_embedding = {t: emb_arr[i] for i, t in enumerate(unique_prompts)}
D = int(emb_arr.shape[1])
print("Embedding dim:", D, "backend_used:", encoder.backend_used, flush=True)

max_components = max(1, max([len(x) for x in condition_df["component_ids"]], default=1))
print("max_components:", max_components, flush=True)

token_by_instance: Dict[str, np.ndarray] = {}
mask_by_instance: Dict[str, np.ndarray] = {}
prompt_texts_by_instance: Dict[str, List[str]] = {}

for _, row in condition_df.iterrows():
    instance_id = str(row["semantic_instance_id"])
    tokens = np.zeros((max_components, D), dtype="float32")
    mask = np.zeros((max_components,), dtype="float32")
    texts = []
    if not row["is_control"]:
        for j, cid in enumerate(list(map(str, row["component_ids"]))[:max_components]):
            prompt = component_prompt_by_instance.get((instance_id, cid), "")
            if prompt and prompt in prompt_to_embedding:
                tokens[j] = prompt_to_embedding[prompt]
                mask[j] = 1.0
                texts.append(prompt)
    token_by_instance[instance_id] = tokens
    mask_by_instance[instance_id] = mask
    prompt_texts_by_instance[instance_id] = texts


def mask_aware_pool(tokens: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask_f = mask.astype("float32")[..., None]
    pooled = (tokens * mask_f).sum(axis=1) / mask_f.sum(axis=1).clip(min=1.0)
    pooled[mask.sum(axis=1) == 0] = 0.0
    return pooled.astype("float32")


def broadcast_to_cells(adata: ad.AnnData) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    instance_ids = adata.obs["_semantic_instance_id"].astype(str).values
    N = adata.n_obs
    tokens = np.zeros((N, max_components, D), dtype="float32")
    mask = np.zeros((N, max_components), dtype="float32")
    for i, instance_id in enumerate(tqdm(instance_ids, desc="broadcast P4 tokens to cells")):
        if instance_id in token_by_instance:
            tokens[i] = token_by_instance[instance_id]
            mask[i] = mask_by_instance[instance_id]
    pooled = mask_aware_pool(tokens, mask)
    return tokens, mask, pooled


def to_save_dtype(x: np.ndarray) -> np.ndarray:
    if ARGS.save_dtype == "float16":
        return x.astype("float16")
    return x.astype("float32")


tokens, mask, pooled = broadcast_to_cells(adata)

# Sanity checks
noncontrol_instances = condition_df.loc[~condition_df["is_control"], "semantic_instance_id"].astype(str).tolist()
noncontrol_zero = [x for x in noncontrol_instances if float(mask_by_instance[x].sum()) == 0.0]
if noncontrol_zero:
    print("[WARN] non-control semantic instances with all-zero mask:", noncontrol_zero[:20], "n=", len(noncontrol_zero), flush=True)

control_instances = condition_df.loc[condition_df["is_control"], "semantic_instance_id"].astype(str).tolist()
print("control semantic instances:", control_instances, flush=True)
print("cell token shape:", tokens.shape, "mask shape:", mask.shape, "pooled shape:", pooled.shape, flush=True)
print("mask valid count min/median/max:", float(mask.sum(axis=1).min()), float(np.median(mask.sum(axis=1))), float(mask.sum(axis=1).max()), flush=True)

# =========================
# Save to AnnData
# =========================


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(s)).strip("_")


embedding_name = "sem_pert_key__bge_m3__p4_normalized_full_context__component__dense"
tokens_key = f"X_{embedding_name}_tokens"
mask_key = f"X_{embedding_name}_mask"
pooled_key = f"X_{embedding_name}_pooled"

adata.obsm[tokens_key] = to_save_dtype(tokens.reshape(adata.n_obs, max_components * D))
adata.obsm[mask_key] = mask.astype("float32")
adata.obsm[pooled_key] = to_save_dtype(pooled)

if "semantic_embedding_meta" not in adata.uns:
    adata.uns["semantic_embedding_meta"] = {}
if "C_embedding_schema" not in adata.uns:
    adata.uns["C_embedding_schema"] = {}

meta = {
    "embedding_name": embedding_name,
    "semantic_role": "intervention_identity_full_context",
    "semantic_instance_granularity": "condition_x_p4_context",
    "role_note": (
        "BGE-M3 P4 component-level prompt. Each perturbation component token includes perturbation, "
        "biological context, and technical context. Tokens are generated per unique condition x P4-context "
        "instance and broadcast only within that semantic instance. ctx_bio/ctx_protocol are not saved separately."
    ),
    "model_name": "bge_m3",
    "hf_model_name": ARGS.model_name,
    "backend_used": encoder.backend_used,
    "prompt_type": "p4_normalized_full_context",
    "token_level": "component",
    "token_type": "component_tokens",
    "tokens_key": tokens_key,
    "mask_key": mask_key,
    "pooled_key": pooled_key,
    "tokens_are_flattened": True,
    "max_components": int(max_components),
    "max_tokens": int(max_components),
    "raw_token_dim": int(D),
    "flattened_dim": int(max_components * D),
    "mask_semantics": "1=valid component token, 0=padding/control/no component",
    "pooling": "dense",
    "normalize": bool(ARGS.normalize),
    "save_dtype": ARGS.save_dtype,
    "knowledge_source": ARGS.knowledge_source,
    "online_enrichment_enabled": bool(ARGS.knowledge_source == "schema_online" and not ARGS.no_online_enrichment),
    "bio_context_fields": BIO_CONTEXT_FIELDS,
    "technical_context_fields": ["C_platform"] + (["C_batch"] if ARGS.include_batch_in_tech_context else []),
    "dose_rule": "genetic missing dose -> not applicable; non-genetic missing dose -> unknown",
    "duration_rule": "missing duration -> unknown",
    "combination_rule": "combo_n from C_perturbation_combo_n if available, else number of component_ids",
    "evaluation_performed": False,
    "created_at": datetime.now().isoformat(timespec="seconds"),
}
adata.uns["semantic_embedding_meta"][embedding_name] = meta
adata.uns["C_embedding_schema"][embedding_name] = {
    "semantic_role": meta["semantic_role"],
    "prompt_type": meta["prompt_type"],
    "model_name": meta["model_name"],
    "hf_model_name": meta["hf_model_name"],
    "token_level": meta["token_level"],
    "tokens_key": tokens_key,
    "mask_key": mask_key,
    "pooled_key": pooled_key,
    "role_note": meta["role_note"],
}
adata.uns["selected_role_aware_semantic_embeddings"] = {
    "pert_key": {
        "embedding_name": embedding_name,
        "saved_keys": {
            "tokens_key": tokens_key,
            "mask_key": mask_key,
            "pooled_key": pooled_key,
        },
    }
}
adata.uns["selected_role_aware_semantic_embeddings_meta"] = {
    "selection_scope": "Fixed BGE-M3 + P4 normalized full-context prompt; no semantic-response evaluation was performed in this script.",
    "background_context_note": "Biological and technical context are folded into each P4 pert_key component prompt rather than saved as separate ctx_bio/ctx_protocol embeddings.",
    "evaluation_performed": False,
    "created_at": datetime.now().isoformat(timespec="seconds"),
}

# Compact semantic-instance tables for debugging and reproducibility.
compact = {}
semantic_instance_ids = condition_df["semantic_instance_id"].astype(str).tolist()
condition_ids = condition_df["condition_id"].astype(str).tolist()
compact[embedding_name] = {
    "semantic_instance_ids": semantic_instance_ids,
    "condition_ids": condition_ids,
    "tokens_by_semantic_instance_flat": np.stack(
        [token_by_instance[x].reshape(-1) for x in semantic_instance_ids]
    ).astype(ARGS.save_dtype),
    "mask_by_semantic_instance": np.stack(
        [mask_by_instance[x] for x in semantic_instance_ids]
    ).astype("float32"),
    "prompt_table_csv": str(OUTPUT_DIR / "p4_component_prompts.csv"),
    "semantic_instance_table_csv": str(OUTPUT_DIR / "semantic_instance_table_bgem3_p4.csv"),
    "meta": meta,
}
adata.uns["semantic_embedding_compact"] = adata.uns.get("semantic_embedding_compact", {})
adata.uns["semantic_embedding_compact"][embedding_name] = compact[embedding_name]

# Save prompt and metadata sidecars.
with open(OUTPUT_DIR / "selected_embedding_meta.json", "w") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
with open(OUTPUT_DIR / "selected_role_aware_semantic_embeddings.json", "w") as f:
    json.dump(adata.uns["selected_role_aware_semantic_embeddings"], f, indent=2, ensure_ascii=False)

online_cache.save()
encoder.release()

if ARGS.write_h5ad:
    print("Writing:", OUTPUT_H5AD, flush=True)
    adata.write_h5ad(OUTPUT_H5AD)
    print("Done.", flush=True)
else:
    print("--no-write-h5ad set; output AnnData was not written.", flush=True)


# =========================
# Reader helper copied into the script output for downstream reference
# =========================


def load_semantic_tokens_from_adata(
    adata: ad.AnnData,
    embedding_name: str = "sem_pert_key__bge_m3__p4_normalized_full_context__component__dense",
    to_float32: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Recover [N,K,D] semantic tokens from flattened AnnData obsm storage."""
    meta = adata.uns["semantic_embedding_meta"][embedding_name]
    tokens_key = meta["tokens_key"]
    mask_key = meta["mask_key"]
    tokens_flat = np.asarray(adata.obsm[tokens_key])
    mask = np.asarray(adata.obsm[mask_key])
    K = int(meta.get("max_components", meta.get("max_tokens")))
    D = int(meta["raw_token_dim"])
    tokens = tokens_flat.reshape(tokens_flat.shape[0], K, D)
    if to_float32:
        tokens = tokens.astype("float32")
        mask = mask.astype("float32")
    return tokens, mask, meta


def stabilize_all_zero_mask_for_attention_torch(tokens, mask):
    """For cross-attention: keep zero tokens, but make all-zero rows attend to the first zero token to avoid NaNs."""
    stable_mask = mask.clone()
    all_zero = stable_mask.sum(dim=1) == 0
    if all_zero.any():
        stable_mask[all_zero, 0] = 1
    return tokens, stable_mask


def key_padding_mask_from_valid_mask_torch(mask):
    """Convert 1=valid, 0=padding to PyTorch MultiheadAttention key_padding_mask where True=ignore."""
    return mask <= 0
