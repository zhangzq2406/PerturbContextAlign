#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise RuntimeError(f"{label}: expected exactly 1 match, found {n}")
    return text.replace(old, new, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="Historical preprocess2_embedding_bgem3_p4_v3.py")
    ap.add_argument("output", help="Output corrected v4 script")
    args = ap.parse_args()

    src = Path(args.source)
    out = Path(args.output)
    s = src.read_text()

    s = replace_once(
        s,
        '''SCHEMA_KEYS = {\n    "condition": "C_condition_schema_json",\n    "component": "C_component_schema_json",\n    "entity": "C_entity_schema_json",\n    "prompt": "C_prompt_schema_json",\n}\n''',
        '''SCHEMA_KEY_CANDIDATES = {\n    "condition": ["C_condition_schema", "C_condition_schema_json"],\n    "component": ["C_component_schema", "C_component_schema_json"],\n    "entity": ["C_entity_schema", "C_entity_schema_json"],\n    "prompt": ["C_prompt_schema", "C_prompt_schema_json"],\n}\n''',
        "schema key block",
    )

    s = replace_once(
        s,
        '''def get_uns_json(adata: ad.AnnData, key: str) -> Any:\n    return parse_json_like(adata.uns[key]) if key in adata.uns else None\n''',
        '''def get_uns_json(adata: ad.AnnData, key: str) -> Any:\n    return parse_json_like(adata.uns[key]) if key in adata.uns else None\n\n\ndef get_uns_json_any(adata: ad.AnnData, keys: Iterable[str]) -> Any:\n    for key in keys:\n        if key in adata.uns:\n            return parse_json_like(adata.uns[key])\n    return None\n''',
        "schema helper",
    )

    s = replace_once(
        s,
        '''condition_schema = get_uns_json(adata, SCHEMA_KEYS["condition"])\ncomponent_schema = get_uns_json(adata, SCHEMA_KEYS["component"])\nentity_schema = get_uns_json(adata, SCHEMA_KEYS["entity"])\nprompt_schema = get_uns_json(adata, SCHEMA_KEYS["prompt"])\n''',
        '''condition_schema = get_uns_json_any(adata, SCHEMA_KEY_CANDIDATES["condition"])\ncomponent_schema = get_uns_json_any(adata, SCHEMA_KEY_CANDIDATES["component"])\nentity_schema = get_uns_json_any(adata, SCHEMA_KEY_CANDIDATES["entity"])\nprompt_schema = get_uns_json_any(adata, SCHEMA_KEY_CANDIDATES["prompt"])\n''',
        "schema load",
    )

    start = s.index("def build_condition_table_from_adata(adata: ad.AnnData) -> pd.DataFrame:")
    end = s.index("\n\ndef infer_component_record", start)
    replacement = r'''def _stable_semantic_instance_id(condition_id: str, row: pd.Series, is_control: bool) -> str:
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
'''
    s = s[:start] + replacement + s[end:]

    start = s.index("def component_name(rec: Dict[str, Any], condition_row: Optional[pd.Series] = None) -> str:")
    end = s.index("\n\ndef schema_knowledge_text", start)
    replacement = r'''def component_name(rec: Dict[str, Any], condition_row: Optional[pd.Series] = None) -> str:
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
'''
    s = s[:start] + replacement + s[end:]

    s = replace_once(
        s,
        '''prompt_rows: List[Dict[str, Any]] = []\ncomponent_prompt_by_condition: Dict[Tuple[str, str], str] = {}\nall_prompts: List[str] = []\n\nfor _, row in tqdm(condition_df.iterrows(), total=len(condition_df), desc="build P4 prompts"):\n    cond_id = row["condition_id"]\n    for cid in list(map(str, row["component_ids"])):\n        rec = component_records.get(cid) or infer_component_record(cid, row)\n        prompt = build_p4_component_prompt(cid, rec, row)\n        component_prompt_by_condition[(cond_id, cid)] = prompt\n        all_prompts.append(prompt)\n        prompt_rows.append(\n            {\n                "condition_id": cond_id,\n                "component_id": cid,\n                "prompt_type": "p4_normalized_full_context",\n                "model_name": ARGS.model_name,\n                "prompt": prompt,\n            }\n        )\n''',
        '''prompt_rows: List[Dict[str, Any]] = []\ncomponent_prompt_by_instance: Dict[Tuple[str, str], str] = {}\nall_prompts: List[str] = []\n\nfor _, row in tqdm(condition_df.iterrows(), total=len(condition_df), desc="build P4 prompts"):\n    instance_id = str(row["semantic_instance_id"])\n    cond_id = str(row["condition_id"])\n    for cid in list(map(str, row["component_ids"])):\n        rec = component_records.get(cid) or infer_component_record(cid, row)\n        prompt = build_p4_component_prompt(cid, rec, row)\n        component_prompt_by_instance[(instance_id, cid)] = prompt\n        all_prompts.append(prompt)\n        prompt_rows.append(\n            {\n                "semantic_instance_id": instance_id,\n                "condition_id": cond_id,\n                "component_id": cid,\n                "prompt_type": "p4_normalized_full_context",\n                "model_name": ARGS.model_name,\n                "prompt": prompt,\n            }\n        )\n''',
        "prompt mapping",
    )

    start = s.index("token_by_condition: Dict[str, np.ndarray] = {}")
    end = s.index("\ndef to_save_dtype", start)
    replacement = r'''token_by_instance: Dict[str, np.ndarray] = {}
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

'''
    s = s[:start] + replacement + s[end:]

    s = replace_once(
        s,
        '''noncontrol_cond = condition_df.loc[~condition_df["is_control"], "condition_id"].astype(str).tolist()\nnoncontrol_zero = [c for c in noncontrol_cond if float(mask_by_condition[c].sum()) == 0.0]\nif noncontrol_zero:\n    print("[WARN] non-control conditions with all-zero mask:", noncontrol_zero[:20], "n=", len(noncontrol_zero), flush=True)\n\ncontrol_cond = condition_df.loc[condition_df["is_control"], "condition_id"].astype(str).tolist()\nprint("control conditions:", control_cond, flush=True)\n''',
        '''noncontrol_instances = condition_df.loc[~condition_df["is_control"], "semantic_instance_id"].astype(str).tolist()\nnoncontrol_zero = [x for x in noncontrol_instances if float(mask_by_instance[x].sum()) == 0.0]\nif noncontrol_zero:\n    print("[WARN] non-control semantic instances with all-zero mask:", noncontrol_zero[:20], "n=", len(noncontrol_zero), flush=True)\n\ncontrol_instances = condition_df.loc[condition_df["is_control"], "semantic_instance_id"].astype(str).tolist()\nprint("control semantic instances:", control_instances, flush=True)\n''',
        "mask sanity",
    )

    s = replace_once(
        s,
        '''    "semantic_role": "intervention_identity_full_context",\n    "role_note": (\n        "BGE-M3 P4 component-level prompt. Each perturbation component token includes perturbation, "\n        "biological context, and technical context. ctx_bio/ctx_protocol are not saved separately in this script."\n    ),\n''',
        '''    "semantic_role": "intervention_identity_full_context",\n    "semantic_instance_granularity": "condition_x_p4_context",\n    "role_note": (\n        "BGE-M3 P4 component-level prompt. Each perturbation component token includes perturbation, "\n        "biological context, and technical context. Tokens are generated per unique condition x P4-context "\n        "instance and broadcast only within that semantic instance. ctx_bio/ctx_protocol are not saved separately."\n    ),\n''',
        "metadata",
    )

    start = s.index("# Compact condition-level tables for debugging and reproducibility.")
    end = s.index("\n# Save prompt and metadata sidecars.", start)
    replacement = r'''# Compact semantic-instance tables for debugging and reproducibility.
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
'''
    s = s[:start] + replacement + s[end:]

    s = s.replace(
        "Build BGE-M3 + P4 normalized full-context semantic embeddings for perturbation AnnData.",
        "Build BGE-M3 + P4 normalized full-context semantic embeddings for perturbation AnnData (v4 context-instance corrected).",
        1,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(s)

    compile(s, str(out), "exec")
    print(f"PASS: {out}")
    print(f"bytes={out.stat().st_size}")


if __name__ == "__main__":
    main()
