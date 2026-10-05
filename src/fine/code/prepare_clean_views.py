#!/usr/bin/env python3
"""Prepare metadata-only clean text views; never load expression or run a model.

Only the atomic metadata TSV, source-verification JSON and provenance-review
Markdown are inputs. Output directories must not already exist. Templates and
anonymous seeds are fixed below, independently of outcomes or model scores.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (20260914, 20260915, 20260916)
CELL_LINES = ("A549", "K562", "MCF7")
DOSES_NM = ("10", "100", "1000", "10000")
VIEWS = ("entity", "entity_exposure", "entity_context", "complete_metadata")
REQUIRED_FIELDS = (
    "atomic_id", "source_entity_key", "source_entity_name", "cell_line",
    "dose_value", "dose_unit", "time",
)
TEMPLATES = {
    "entity": "{entity}",
    "entity_exposure": "{entity}; dose: {dose} nM; duration: 24 h.",
    "entity_context": "{entity}; cell line: {cell_line}.",
    "complete_metadata": "{entity}; dose: {dose} nM; duration: 24 h; cell line: {cell_line}.",
}
PRIMARY_TIME_SOURCES = (
    "https://doi.org/10.1126/science.aax6234",
    "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4150378",
    "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4150nnn/GSM4150378/suppl/"
    "GSM4150378_sciPlex3_A549_MCF7_K562_hashTable_metadata.txt.gz",
)


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _decimal_text(value, field: str) -> str:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{field} is not a valid decimal") from exc
    if not number.is_finite():
        raise ValueError(f"{field} must be finite")
    return format(number.normalize(), "f")


def validate_time_evidence(verification: dict, review_text: str) -> None:
    """Require numerical source agreement plus the separately reviewed unit.

    The JSON alone supplies no time unit. The review contains the human-audited
    Science/GEO evidence; anchors identify that evidence rather than deriving
    hours from the raw number 24. Its complete bytes are recorded in the output.
    """
    if verification.get("status") != "SOURCE_METADATA_VERIFIED":
        raise ValueError("source verification is not SOURCE_METADATA_VERIFIED")
    mismatches = verification.get("mismatches", {})
    for field in ("cell_line", "replicate", "time", "dose", "control"):
        if field not in mismatches or mismatches[field] != 0:
            raise ValueError(f"source verification has missing/nonzero {field} mismatches")
    for anchor in ("duration_hours=24", "10.1126/science.aax6234", "GSM4150378"):
        if anchor not in review_text:
            raise ValueError(f"provenance review lacks the required time evidence anchor: {anchor}")


def validate_atoms(rows: list[dict], *, expected_entity_count: int = 188) -> list[dict]:
    """Validate the locked 24-hour, three-line, four-dose complete metadata grid."""
    if not rows:
        raise ValueError("atomic metadata is empty")
    if expected_entity_count < 1:
        raise ValueError("expected_entity_count must be positive")
    atoms, by_key, by_name, atom_ids, conditions = [], {}, {}, set(), set()
    for row in rows:
        missing = [field for field in REQUIRED_FIELDS if field not in row or row[field] is None]
        if missing:
            raise ValueError(f"atomic metadata lacks required fields: {missing}")
        # Upstream keys use dataset-local whitespace-normalized source names.
        # Preserve those names exactly; do not merge synonyms or resolve a
        # global canonical chemical identity here.
        selected = {field: str(row[field]) for field in REQUIRED_FIELDS}
        for field in ("atomic_id", "source_entity_key", "source_entity_name", "cell_line"):
            value = selected[field]
            if not value or value != " ".join(value.split()):
                raise ValueError(f"{field} must be nonempty and already whitespace-normalized")
        if selected["atomic_id"] in atom_ids:
            raise ValueError("duplicate atomic_id")
        atom_ids.add(selected["atomic_id"])
        key, name = selected["source_entity_key"], selected["source_entity_name"]
        if key in by_key and by_key[key] != name:
            raise ValueError("one source_entity_key maps to multiple source names")
        if name in by_name and by_name[name] != key:
            raise ValueError("one source name maps to multiple source_entity_keys")
        by_key[key], by_name[name] = name, key
        if selected["cell_line"] not in CELL_LINES:
            raise ValueError("cell_line is outside the locked three-line cohort")
        if selected["dose_unit"] != "nM":
            raise ValueError("dose_unit must be nM")
        selected["dose_value"] = _decimal_text(selected["dose_value"], "dose_value")
        if selected["dose_value"] not in DOSES_NM:
            raise ValueError("dose is outside the locked four-dose cohort")
        if _decimal_text(selected["time"], "time") != "24":
            raise ValueError("only the source-verified 24-hour cohort is allowed")
        selected["time"] = "24"
        condition = (key, selected["cell_line"], selected["dose_value"])
        if condition in conditions:
            raise ValueError("multiple atoms share one entity/cell_line/dose condition")
        conditions.add(condition)
        atoms.append(selected)
    if len(by_key) != expected_entity_count:
        raise ValueError(f"expected {expected_entity_count} entities, found {len(by_key)}")
    expected = {(key, line, dose) for key in by_key for line in CELL_LINES for dose in DOSES_NM}
    if conditions != expected:
        raise ValueError("incomplete or unexpected entity/cell_line/dose grid")
    return sorted(atoms, key=lambda row: row["atomic_id"])


def build_clean_views(
    rows: list[dict], verification: dict, review_text: str, *, expected_entity_count: int = 188
) -> dict:
    """Build deterministic tables in memory without any model or expression IO."""
    validate_time_evidence(verification, review_text)
    atoms = validate_atoms(rows, expected_entity_count=expected_entity_count)
    names = {row["source_entity_key"]: row["source_entity_name"] for row in atoms}
    ordered_keys = sorted(names)
    variants = [("source_name", "", {key: names[key] for key in ordered_keys})]
    for seed in SEEDS:
        permutation = np.random.default_rng(seed).permutation(len(ordered_keys)) + 1
        mapping = {key: f"Entity_{int(number):04d}" for key, number in zip(ordered_keys, permutation)}
        variants.append((f"anonymous_seed_{seed}", str(seed), mapping))

    mappings, registry = [], []
    unique = {}
    groups = defaultdict(list)
    for variant, seed, mapping in variants:
        for key in ordered_keys:
            mappings.append({
                "variant": variant, "anonymous_seed": seed, "source_entity_key": key,
                "source_entity_name": names[key], "display_entity_name": mapping[key],
            })
        for atom in atoms:
            entity = mapping[atom["source_entity_key"]]
            for view in VIEWS:
                text = TEMPLATES[view].format(
                    entity=entity, dose=atom["dose_value"], cell_line=atom["cell_line"]
                )
                digest = sha256_bytes(text.encode("utf-8"))
                text_id = "text:" + digest
                if digest in unique and unique[digest]["prompt_text"] != text:
                    raise RuntimeError("SHA-256 collision between unequal text strings")
                record = unique.setdefault(digest, {
                    "text_id": text_id, "prompt_sha256": digest, "prompt_text": text,
                    "n_registry_rows": 0, "views": set(), "variants": set(),
                })
                record["n_registry_rows"] += 1
                record["views"].add(view)
                record["variants"].add(variant)
                row = {
                    "atomic_id": atom["atomic_id"], "source_entity_key": atom["source_entity_key"],
                    "view": view, "variant": variant, "anonymous_seed": seed,
                    "display_entity_name": entity, "text_id": text_id, "prompt_sha256": digest,
                }
                registry.append(row)
                groups[(variant, view)].append(row)

    summary = []
    for (variant, view), group in groups.items():
        counts = Counter(row["text_id"] for row in group)
        entities = defaultdict(set)
        for row in group:
            entities[row["text_id"]].add(row["source_entity_key"])
        collision_groups = sum(count > 1 for count in counts.values())
        summary.append({
            "variant": variant, "view": view, "n_atomic_rows": len(group),
            "n_unique_texts": len(counts), "collision_excess_rows": len(group) - len(counts),
            "n_collision_groups": collision_groups,
            "n_rows_in_collision_groups": sum(count for count in counts.values() if count > 1),
            "max_atoms_per_text": max(counts.values()),
            "n_cross_entity_collision_groups": sum(len(keys) > 1 for keys in entities.values()),
            "exact_atomic_retrieval_identifiable": collision_groups == 0,
        })
    unique_rows = []
    for digest in sorted(unique):
        row = unique[digest].copy()
        row["views"] = "|".join(sorted(row["views"]))
        row["variants"] = "|".join(sorted(row["variants"]))
        unique_rows.append(row)

    config = {
        "version": "clean_views_v1", "dataset": "srivatsan_sciplex3",
        "source_name_policy": "Inherited dataset-local whitespace-normalized source names; no global canonical resolution or synonym merging.",
        "prompt_input_fields": ["source_entity_name", "dose_value", "dose_unit", "time", "cell_line"],
        "linkage_fields_not_rendered": ["atomic_id", "source_entity_key"],
        "excluded_prompt_fields": ["target", "MoA", "platform", "batch", "plate", "library", "component_ids", "legacy_condition_id"],
        "templates": TEMPLATES,
        "duration_hours": 24,
        "duration_unit_authority": "Science and GEO primary sources documented in the hashed provenance review; source-verification JSON confirms raw/design time agreement only.",
        "time_primary_sources": list(PRIMARY_TIME_SOURCES),
        "anonymous_seeds": list(SEEDS), "anonymous_format": "Entity_{number:04d}",
        "mapping_order": "Lexicographically sorted source_entity_key; one permutation per seed shared across every atom/view.",
        "rng": "numpy.random.default_rng(seed).permutation", "numpy_version": np.__version__,
        "python_version": sys.version.split()[0],
        "dedup_key": "SHA-256 of exact UTF-8 prompt text, shared across views and variants",
        "entity_retrieval_relevance": "All atoms with the same source_entity_key are relevant, even when context/exposure differs.",
        "condition_retrieval_policy": "Views with multiple atoms per text cannot be used for unrestricted exact-atomic retrieval; restrict a predeclared identifiable candidate pool or report the ambiguity.",
        "annotation": "Same anonymous string can denote different entities under different seeds; variant is mandatory in the registry join. Shared text embeddings may still be reused exactly.",
        "expression_data_read": False, "model_encoding_performed": False,
    }
    qa = {
        "status": "PASS_METADATA_ONLY_NO_MODEL_ENCODING",
        "n_atoms": len(atoms), "n_entities": len(names), "n_views": len(VIEWS),
        "n_variants": len(variants), "n_mapping_rows": len(mappings),
        "n_registry_rows": len(registry), "n_global_unique_texts": len(unique_rows),
        "n_sum_unique_texts_across_view_variants": sum(s["n_unique_texts"] for s in summary),
        "n_duplicate_registry_rows_avoided_by_global_text_dedup": len(registry) - len(unique_rows),
        "per_view_variant": summary,
        "n_cross_entity_collision_groups_within_view_variant": sum(s["n_cross_entity_collision_groups"] for s in summary),
        "all_text_hashes_verified": all(sha256_bytes(row["prompt_text"].encode()) == row["prompt_sha256"] for row in unique_rows),
        "source_name_scope": config["source_name_policy"],
        "time_unit_supported_by_reviewed_primary_sources": True,
        "model_encoding_performed": False, "expression_data_read": False,
        "exact_atomic_retrieval_warning": config["condition_retrieval_policy"],
    }
    return {"config": config, "mappings": mappings, "registry": registry,
            "unique_texts": unique_rows, "qa": qa}


def _write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _write_tsv(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def prepare(
    atomic_path: Path, verification_path: Path, review_path: Path, output_path: Path,
    *, expected_entity_count: int = 188,
) -> dict:
    """Save a new auditable metadata package; refuse any existing output path."""
    output_path = Path(output_path)
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
    sources = []
    content = {}
    for role, path in (("atomic_metadata", atomic_path), ("source_verification", verification_path),
                       ("time_unit_provenance_review", review_path)):
        path = Path(path).resolve()
        raw = path.read_bytes()
        content[role] = raw.decode("utf-8")
        sources.append({"role": role, "path": str(path), "bytes": len(raw), "sha256": sha256_bytes(raw)})
    atoms = list(csv.DictReader(content["atomic_metadata"].splitlines(), delimiter="\t"))
    payload = build_clean_views(
        atoms, json.loads(content["source_verification"]), content["time_unit_provenance_review"],
        expected_entity_count=expected_entity_count,
    )
    payload["config"]["source_inputs"] = sources
    payload["config"]["producer_script_sha256"] = sha256_bytes(Path(__file__).read_bytes())
    output_path.mkdir(parents=True, exist_ok=False)
    _write_json(output_path / "config.json", payload["config"])
    _write_tsv(output_path / "source_sha256.tsv", sources)
    _write_tsv(output_path / "entity_mapping.tsv", payload["mappings"])
    _write_tsv(output_path / "row_to_text_registry.tsv", payload["registry"])
    _write_tsv(output_path / "unique_texts.tsv", payload["unique_texts"])
    for source in sources:
        if sha256_bytes(Path(source["path"]).read_bytes()) != source["sha256"]:
            raise RuntimeError(f"Input changed during preparation; preserve incomplete output: {source['path']}")
    records = []
    for path in sorted(output_path.iterdir()):
        records.append({"file": path.name, "bytes": path.stat().st_size,
                        "sha256": sha256_bytes(path.read_bytes())})
    _write_tsv(output_path / "output_sha256.tsv", records)
    payload["qa"]["source_files_unchanged_during_run"] = True
    payload["qa"]["output_sha256"] = {record["file"]: record["sha256"] for record in records}
    payload["qa"]["output_sha256"]["output_sha256.tsv"] = sha256_bytes((output_path / "output_sha256.tsv").read_bytes())
    _write_json(output_path / "qa_summary.json", payload["qa"])
    return payload["qa"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atoms", type=Path, default=ROOT / "metadata/sciplex_preflight_v1/atomic_conditions.tsv")
    parser.add_argument("--source-verification", type=Path, default=ROOT / "inputs/sciplex_geo_verified_v1/source_verification.json")
    parser.add_argument("--provenance-review", type=Path, default=ROOT / "qa/sciplex_provenance_review.md")
    parser.add_argument("--output", type=Path, default=ROOT / "representations/clean_views_v1")
    args = parser.parse_args()
    qa = prepare(args.atoms, args.source_verification, args.provenance_review, args.output)
    print(json.dumps({key: value for key, value in qa.items() if key not in ("per_view_variant", "output_sha256")},
                     ensure_ascii=False, indent=2))
    for row in qa["per_view_variant"]:
        print(row["variant"], row["view"], "unique=", row["n_unique_texts"],
              "collision_excess=", row["collision_excess_rows"],
              "exact_atomic_identifiable=", row["exact_atomic_retrieval_identifiable"])


if __name__ == "__main__":
    main()
