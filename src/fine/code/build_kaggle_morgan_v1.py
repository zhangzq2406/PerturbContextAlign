#!/usr/bin/env python3
"""Source-provided full-structure Morgan prior, on the frozen Kaggle cohort."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import rdFingerprintGenerator
from build_morgan_prior import fingerprint_from_source, sha256

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "metadata/kaggle_atomic_preflight_v1"
OUT = ROOT / "representations/kaggle_morgan_v1"


def run():
    if OUT.exists():
        raise FileExistsError("Refuse existing output " + str(OUT))
    independent = ROOT / "qa/kaggle_atomic_metadata_independent_20260920.json"
    qa = json.loads(independent.read_text())
    if "PASS" not in qa.get("status", ""):
        raise ValueError("independent metadata QA not PASS")
    manifest = json.loads((META / "output_manifest.json").read_text())
    for row in manifest["files"]:
        if sha256(META / row["path"]) != row["sha256"]:
            raise ValueError("metadata payload drift " + row["path"])
    atoms = pd.read_csv(META / "atomic_index.tsv", sep="\t", keep_default_na=False)
    compounds = pd.read_csv(META / "compound_index.tsv", sep="\t", keep_default_na=False)
    compounds = compounds[compounds.sm_lincs_id.isin(atoms.sm_lincs_id)].sort_values("sm_lincs_id").reset_index(drop=True)
    if len(compounds) != 137 or not compounds.sm_lincs_id.is_unique or set(compounds.sm_lincs_id) != set(atoms.sm_lincs_id):
        raise ValueError("frozen cohort compound coverage drift")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    vectors, canonical, fragments, charges = [], [], [], []
    for row in compounds.itertuples():
        vector, text = fingerprint_from_source(row.SMILES, generator)
        mol = Chem.MolFromSmiles(row.SMILES)
        vectors.append(vector)
        canonical.append(text)
        fragments.append(len(Chem.GetMolFrags(mol)))
        charges.append(Chem.GetFormalCharge(mol))
    matrix = np.stack(vectors)
    compounds["canonical_isomeric_smiles"] = canonical
    compounds["n_fragments"], compounds["formal_charge"] = fragments, charges
    compounds["fingerprint_sha256"] = [hashlib.sha256(v.tobytes()).hexdigest() for v in matrix]
    compounds["n_on_bits"] = matrix.sum(axis=1)
    compounds["fingerprint_collided"] = compounds.fingerprint_sha256.duplicated(keep=False)
    sources = [META / "atomic_index.tsv", META / "compound_index.tsv", META / "output_manifest.json", independent,
        ROOT / "configs/kaggle_independent_study_contract_v1.json", ROOT / "code/build_morgan_prior.py", Path(__file__)]
    input_hash = {str(path): sha256(path) for path in sources}
    OUT.mkdir()
    with (OUT / "arrays.npz").open("xb") as stream:
        np.savez_compressed(stream, source_entity_key=compounds.sm_lincs_id.to_numpy(dtype=str),
            sm_lincs_id=compounds.sm_lincs_id.to_numpy(dtype=str), fingerprints=matrix)
    compounds.to_csv(OUT / "entity_fingerprint_manifest.tsv", sep="\t", index=False)
    if any(sha256(path) != value for path, value in input_hash.items()):
        raise ValueError("source changed during prior generation")
    audit = dict(status="PASS", completed_utc=datetime.now(timezone.utc).isoformat(), n_entities=137,
        n_unique_fingerprints=int(compounds.fingerprint_sha256.nunique()), n_collided_entities=int(compounds.fingerprint_collided.sum()),
        radius=2, fp_size=2048, include_chirality=True, binary_not_count=True, rdkit_version=rdBase.rdkitVersion,
        structure_policy="FULL_SOURCE_STRUCTURE_NO_DESALTING_NO_NEUTRALIZATION",
        chemical_structures_are_additional_information=True, source="release-provided SMILES, exact source LINCS mapping",
        kernel_contract="0.5*(Tanimoto(full-source Morgan)+dose equality); optional product with control-A cosine",
        expression_or_responses_read=False, input_sha256=input_hash,
        output_sha256={p.name: sha256(p) for p in OUT.iterdir() if p.is_file()})
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(dict(status="PASS", n_entities=137, n_unique_fingerprints=audit["n_unique_fingerprints"])))


if __name__ == "__main__":
    run()
