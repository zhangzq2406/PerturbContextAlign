#!/usr/bin/env python3
"""E08 metadata-only, exact-structure, raw-snapshot knowledge admission."""
from pathlib import Path
import hashlib
import json
import re
import sys
sys.dont_write_bytecode = True
import pandas as pd
from rdkit import Chem, rdBase

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/knowledge_admission_v1.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False)


def canonical(smiles):
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None or not mol.GetNumAtoms():
        return ''
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def mechanism_reason(record, chosen_id, targets):
    moa = record.get('mechanism_of_action')
    if not isinstance(moa, str) or not moa.strip() or moa.strip().lower() in {'unknown', 'none', 'n/a', 'na'}:
        return 'UNINFORMATIVE_MECHANISM'
    if record.get('molecule_chembl_id') != chosen_id:
        return 'MOLECULE_RECORD_ID_MISMATCH'
    target_id = record.get('target_chembl_id')
    if target_id:
        if target_id not in targets:
            return 'MISSING_TARGET_PAYLOAD'
        target = targets[target_id]
        if str(target.get('tax_id')) != '9606' or target.get('organism') != 'Homo sapiens':
            return 'NONHUMAN_TARGET'
    return 'ALLOWED_MECHANISM'


def make_knowledge(records):
    phrases = sorted({r['mechanism_of_action'] for r in records})
    assert phrases and all(p == p.strip() and not any(c in p for c in '\r\n\t') for p in phrases)
    return 'Mechanisms of action: ' + '; '.join(phrases) + '.'


def write_json(path, data):
    with Path(path).open('x') as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')


def run():
    config = json.loads(CONFIG.read_text())
    out = ROOT / config['output_root']
    assert not out.exists(), 'Refuse existing or partial output'
    assert rdBase.rdkitVersion == '2025.09.1'
    paths = [CONFIG, Path(__file__).resolve(),
             ROOT / 'metadata/existing_prior_sources_preflight_v1/entity_coverage_inventory.tsv',
             ROOT / 'representations/morgan_preflight_v1/source_structure_mapping.tsv',
             ROOT / 'effects/atomic_effects_v1/atomic_index.tsv',
             ROOT / 'representations/clean_views_v1/unique_texts.tsv',
             ROOT / 'representations/clean_views_v1/row_to_text_registry.tsv']
    inventory = read(paths[2]); structures = read(paths[3]).set_index('source_entity_key')
    atoms = read(paths[4]); atoms = atoms[atoms.main_eligible.eq('True')].copy()
    old_texts = read(paths[5]).set_index('text_id')
    old_rows = read(paths[6]); old_rows = old_rows[old_rows.view.eq('complete_metadata') & old_rows.variant.eq('source_name')].set_index('atomic_id')
    assert len(inventory) == 188 and inventory.source_entity_key.is_unique
    assert len(atoms) == 2250 and atoms.atomic_id.is_unique and old_rows.index.is_unique
    paths += [Path(p) for p in inventory.raw_json_path]
    before = {str(p): sha(p) for p in paths}
    coverage, mechanisms, target_rows = [], [], []
    texts_by_entity = {}
    for row in inventory.itertuples():
        assert before[row.raw_json_path] == row.raw_json_sha256
        raw = json.loads(Path(row.raw_json_path).read_text())
        assert raw['entity']['entity_id'] == row.knowledge_entity_id
        release = raw['source_statuses']['ChEMBL']
        # Preserve the full raw declaration; do not infer release from cache TSV.
        assert 'ChEMBL_37' in json.dumps(release)
        ch = raw['online'].get('ChEMBL') or {}
        chosen = ch.get('chosen_molecule') or {}
        source_smiles = canonical(structures.loc[row.source_entity_key, 'source_smiles_raw'])
        assert source_smiles == structures.loc[row.source_entity_key, 'canonical_isomeric_smiles']
        knowledge_smiles = canonical((chosen.get('molecule_structures') or {}).get('canonical_smiles', ''))
        identity = 'EXACT_FULL_STRUCTURE' if knowledge_smiles and source_smiles == knowledge_smiles else ('NO_CHOSEN_STRUCTURE' if not knowledge_smiles else 'DIFFERENT_FULL_STRUCTURE')
        chosen_id = chosen.get('molecule_chembl_id', '')
        payload = ch.get('mechanism') or {}
        records = payload.get('mechanisms', [])
        if chosen:
            page = payload.get('page_meta') or {}
            assert page.get('next') is None and page.get('total_count') == len(records), 'Incomplete mechanism snapshot'
        target_payload = (ch.get('targets') or {}).get('targets', [])
        targets = {t['target_chembl_id']: t for t in target_payload}
        assert len(targets) == len(target_payload), 'Ambiguous target IDs'
        allowed = []
        for index, record in enumerate(records):
            reason = mechanism_reason(record, chosen_id, targets)
            admit = identity == 'EXACT_FULL_STRUCTURE' and reason == 'ALLOWED_MECHANISM'
            target_id = record.get('target_chembl_id') or ''
            target = targets.get(target_id, {})
            mechanisms.append(dict(source_entity_key=row.source_entity_key, raw_json_path=row.raw_json_path,
                raw_json_sha256=row.raw_json_sha256, raw_mechanism_index=index,
                mec_id=record.get('mec_id'), chosen_chembl_id=chosen_id,
                mechanism_molecule_chembl_id=record.get('molecule_chembl_id'),
                mechanism_parent_chembl_id=record.get('parent_molecule_chembl_id'),
                mechanism_of_action=record.get('mechanism_of_action'), target_chembl_id=target_id,
                target_name=target.get('pref_name', ''), target_type=target.get('target_type', ''),
                target_tax_id=target.get('tax_id'), target_organism=target.get('organism', ''),
                direct_interaction=record.get('direct_interaction'), action_type=record.get('action_type'),
                molecular_mechanism=record.get('molecular_mechanism'),
                n_reference_records=len(record.get('mechanism_refs') or []),
                mechanism_reason=reason, identity_status=identity, admitted=admit))
            if admit:
                allowed.append(record)
        ids = sorted({r['target_chembl_id'] for r in allowed if r.get('target_chembl_id')})
        for tid in ids:
            t = targets[tid]
            target_rows.append(dict(source_entity_key=row.source_entity_key, target_chembl_id=tid,
                target_name=t['pref_name'], target_type=t['target_type'], organism=t['organism'], tax_id=t['tax_id']))
        text = make_knowledge(allowed) if allowed else ''
        if text:
            texts_by_entity[row.source_entity_key] = text
        reason = identity if identity != 'EXACT_FULL_STRUCTURE' else ('ADMITTED_MECHANISM' if allowed else 'NO_ALLOWED_INFORMATIVE_MECHANISM')
        coverage.append(dict(source_entity_key=row.source_entity_key, source_entity_name=row.source_entity_name,
            knowledge_entity_id=row.knowledge_entity_id, chosen_chembl_id=chosen_id,
            source_canonical_isomeric_smiles=source_smiles, knowledge_canonical_isomeric_smiles=knowledge_smiles,
            identity_status=identity, admission_status=reason, admitted_mechanism=bool(allowed),
            admitted_human_target=bool(ids), n_raw_mechanisms=len(records), n_allowed_mechanisms=len(allowed),
            n_human_target_ids=len(ids), knowledge_text=text, raw_json_path=row.raw_json_path,
            raw_json_sha256=row.raw_json_sha256, source_declaration_json=json.dumps(release, sort_keys=True),
            retrieval_date=raw['retrieval_date']))
    rows = []
    for atom in atoms.itertuples():
        if atom.source_entity_key not in texts_by_entity:
            continue
        k = texts_by_entity[atom.source_entity_key]
        prior_row = old_rows.loc[atom.atomic_id]; prior = old_texts.loc[prior_row.text_id]
        assert prior_row.prompt_sha256 == prior.prompt_sha256
        for view, prompt in [('complete_metadata+knowledge', prior.prompt_text + ' ' + k), ('knowledge_only', k)]:
            digest = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
            rows.append(dict(atomic_id=atom.atomic_id, source_entity_key=atom.source_entity_key,
                cell_line=atom.cell_line, dose_value=atom.dose_value, view=view, variant=config['variant'],
                text_id='text:' + digest, prompt_sha256=digest, prompt_text=prompt,
                original_complete_text_id=prior_row.text_id))
    registry = pd.DataFrame(rows).sort_values(['atomic_id', 'view']).reset_index(drop=True)
    unique = registry[['text_id', 'prompt_sha256', 'prompt_text']].drop_duplicates().sort_values('text_id').reset_index(drop=True)
    assert unique.text_id.is_unique and unique.prompt_text.is_unique
    unique.insert(0, 'text_row', range(len(unique)))
    unique['views'] = unique.text_id.map(registry.groupby('text_id').view.agg(lambda x: '|'.join(sorted(set(x)))))
    unique['variants'] = config['variant']
    registry['text_row'] = registry.text_id.map(unique.set_index('text_id').text_row)
    cov = pd.DataFrame(coverage)
    # Identity-free here means no deliberately supplied entity ID/name, not proof
    # that pharmacological descriptions cannot identify a compound indirectly.
    out.mkdir(parents=True)
    for name, frame in [('entity_coverage.tsv', cov), ('mechanism_records.tsv', pd.DataFrame(mechanisms)),
                        ('human_target_membership.tsv', pd.DataFrame(target_rows)),
                        ('unique_texts.tsv', unique), ('rowmap.tsv', registry)]:
        with (out / name).open('x') as handle:
            frame.to_csv(handle, sep='\t', index=False, lineterminator='\n')
    write_json(out / 'input_manifest.json', before)
    assert before == {str(p): sha(p) for p in paths}, 'Input changed'
    audit = dict(status='PASS_KNOWLEDGE_METADATA_PREPARATION', frozen_contract_sha256=sha(CONFIG),
        n_entities=188, identity_counts=cov.identity_status.value_counts().to_dict(),
        n_mechanism_entities=int(cov.admitted_mechanism.sum()), n_target_entities=int(cov.admitted_human_target.sum()),
        n_admitted_atoms=registry.atomic_id.nunique(), n_registry_rows=len(registry), n_unique_texts=len(unique),
        n_unique_knowledge_texts=registry[registry.view.eq('knowledge_only')].text_id.nunique(),
        n_unique_target_ids=pd.DataFrame(target_rows).target_chembl_id.nunique(),
        no_effect_values_read=True, no_new_download=True, input_hashes_unchanged=True,
        rdkit_version=rdBase.rdkitVersion, independent_validation_pending=True)
    write_json(out / 'audit.json', audit)
    write_json(out / 'output_manifest.json', dict(frozen_contract_sha256=sha(CONFIG),
        files=[dict(path=p.name, sha256=sha(p), size_bytes=p.stat().st_size) for p in sorted(out.iterdir())]))
    print(json.dumps(audit))


if __name__ == '__main__':
    run()
