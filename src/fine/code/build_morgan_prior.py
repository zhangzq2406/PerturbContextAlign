#!/usr/bin/env python3
"""Build the predeclared full-source Morgan prior, not a textual representation."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import rdFingerprintGenerator

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def fingerprint_from_source(smiles, generator):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise ValueError('Invalid or empty source structure; do not drop silently')
    # Preserve salts/fragments, charges and supplied stereochemistry. No new
    # stereochemistry, desalting, neutralization or score-dependent cleanup.
    vector = np.fromiter((int(v) for v in generator.GetFingerprint(molecule).ToBitString()), dtype=np.uint8)
    if vector.shape != (2048,) or not vector.any():
        raise ValueError('Wrong-size or zero Morgan fingerprint')
    return vector, Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def main():
    start = time.perf_counter()
    mapping = ROOT / 'representations/morgan_preflight_v1/source_structure_mapping.tsv'
    provenance = ROOT / 'qa/morgan_provenance_preflight.md'
    source = ROOT / 'inputs/sciplex_geo_verified_v1/GSM4150378_sciPlex3_A549_MCF7_K562_hashTable_metadata.txt.gz'
    output = ROOT / 'representations/morgan_v1'
    if output.exists():
        raise FileExistsError(f'Refusing existing output {output}')
    inputs = {str(path): sha256(path) for path in (mapping, provenance, source, Path(__file__))}
    table = pd.read_csv(mapping, sep='\t', keep_default_na=False).sort_values('source_entity_key').reset_index(drop=True)
    if len(table) != 188 or not table.source_entity_key.is_unique:
        raise ValueError('Expected all 188 uniquely mapped source entities')
    if not table.mapping_status.eq('UNIQUE_SOURCE_STRUCTURE_PARSE_VALID').all():
        raise ValueError('Unreviewed source mapping')
    if not table.source_file_sha256.eq(inputs[str(source)]).all():
        raise ValueError('Source archive hash mismatch')
    if not table.rdkit_version.eq(rdBase.rdkitVersion).all():
        raise ValueError('RDKit version differs from parsing audit')
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    rows = []
    for row in table.itertuples():
        bits, canonical = fingerprint_from_source(row.source_smiles_raw, generator)
        if canonical != row.canonical_isomeric_smiles:
            raise ValueError(f'Canonical structure mismatch {row.source_entity_key}')
        rows.append(bits)
    fingerprints = np.stack(rows)
    signatures = [hashlib.sha256(row.tobytes()).hexdigest() for row in fingerprints]
    table['fingerprint_sha256'] = signatures
    table['n_on_bits'] = fingerprints.sum(axis=1)
    table['fingerprint_collided'] = table.fingerprint_sha256.duplicated(keep=False)
    output.mkdir(parents=True)
    with (output / 'arrays.npz').open('xb') as stream:
        np.savez_compressed(stream, source_entity_key=table.source_entity_key.to_numpy(dtype=str), fingerprints=fingerprints)
    table.to_csv(output / 'entity_fingerprint_manifest.tsv', sep='\t', index=False)
    if any(sha256(path) != expected for path, expected in inputs.items()):
        raise RuntimeError('Prior source changed during generation')
    audit = {'status': 'PASS', 'n_entities': len(table), 'n_unique_fingerprints': len(set(signatures)),
        'n_zero_fingerprints': 0, 'n_collided_entities': int(table.fingerprint_collided.sum()),
        'radius': 2, 'fp_size': 2048, 'include_chirality': True, 'binary_not_count': True,
        'structure_policy': 'FULL_SOURCE_STRUCTURE_NO_DESALTING_NO_NEUTRALIZATION',
        'rdkit_version': rdBase.rdkitVersion, 'input_sha256': inputs,
        'output_sha256': {p.name: sha256(p) for p in output.iterdir()},
        'seconds': time.perf_counter() - start,
        'effect_values_read': False,
        'kernel_contract': '(Tanimoto(full source Morgan) + numeric-nM-dose-equality)/2; optional product with A-state cosine',
        'limits': ['Additional molecular prior, not same-information language encoder comparison.',
                   'Additive structural and dose kernel does not itself estimate drug-by-dose interactions.',
                   'Source stereochemistry is preserved, not completed when absent.']}
    with (output / 'audit.json').open('x', encoding='utf-8') as stream:
        json.dump(audit, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': 'PASS', 'output': str(output), 'n_unique_fingerprints': len(set(signatures))}))


if __name__ == '__main__':
    main()
