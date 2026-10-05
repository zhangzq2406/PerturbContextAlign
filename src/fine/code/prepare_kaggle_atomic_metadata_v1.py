#!/usr/bin/env python3
"""Freeze Kaggle source metadata and control roles before any expression read.

Consumes the reviewed contract and checked metadata-only support artifacts.
Never reads X/var/layers, estimates effects, selects genes, or runs models.
"""
from __future__ import annotations

from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

sys.dont_write_bytecode = True
import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/kaggle_independent_study_contract_v1.json'
OUT = ROOT / 'metadata/kaggle_atomic_preflight_v1'
PKEY = ['donor_id', 'cell_type', 'library_id', 'plate_name']
AKEY = ['sm_lincs_id', 'dose_uM', 'timepoint_hr', 'donor_id', 'cell_type']
QKEY = ['sm_lincs_id', 'dose_uM', 'timepoint_hr', 'cell_type']
CKEY = ['donor_id', 'cell_type']


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def stat(path):
    item = Path(path).stat()
    return dict(size=item.st_size, mtime_ns=item.st_mtime_ns, ctime_ns=item.st_ctime_ns,
                inode=item.st_ino, device=item.st_dev)


def key_id(frame, columns, prefix):
    return frame[columns].astype(str).apply(lambda row: prefix + hashlib.sha256(
        json.dumps(row.tolist(), ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()[:20], axis=1)


def read_table(path):
    return pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False)


def save_table(name, frame):
    if name.endswith('.gz'):
        with (OUT / name).open('xb') as raw:
            with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as stream:
                stream.write(frame.to_csv(sep='\t', index=False, lineterminator='\n', na_rep='').encode())
    else:
        with (OUT / name).open('x') as stream:
            frame.to_csv(stream, sep='\t', index=False, lineterminator='\n', na_rep='')


def save_json(name, value):
    with (OUT / name).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def run():
    started = time.perf_counter()
    require(not OUT.exists(), f'Refuse existing output: {OUT}')
    contract_bytes = CONFIG.read_bytes()
    contract_sha = hashlib.sha256(contract_bytes).hexdigest()
    contract = json.loads(contract_bytes)
    require(contract['vehicle']['sm_name'] == 'Dimethyl Sulfoxide' and
            contract['vehicle']['sm_lincs_id'] == 'LSM-36361', 'Exact source vehicle identity differs')
    require(contract['vehicle']['exact_both_fields_required'], 'Vehicle dual-key requirement missing')
    require(contract['control_pool_key'] == PKEY and contract['atomic_key'] == AKEY, 'Unhandled contract keys')
    require(contract['minimum_treated_cells_per_measurement'] == 20 and
            contract['control_partition']['minimum_total_cells'] == 40 and
            contract['control_partition']['minimum_cells_per_part'] == 20, 'Unhandled support thresholds')
    require(contract['control_partition']['allocate_once_globally'] and
            not contract['control_partition']['independent_wells'], 'Control dependence contract differs')
    require('SHA256(seed|obs_id)' in contract['control_partition']['algorithm'] and
            'alternating' in contract['control_partition']['algorithm'], 'Unhandled partition algorithm')
    require(contract['measurement_scale'] == 'unfiltered_source_provided_log_normalized_difference',
            'Unhandled measurement definition')
    require(contract['endpoints']['gene_min_n'] == 20, 'Unhandled gene endpoint minimum')
    donors = sorted(contract['tasks']['donors'])
    require(donors == ['donor_0', 'donor_1', 'donor_2'], 'Unexpected donor contract')
    seed = int(contract['seed'])
    pre = ROOT / contract['source_task_support']
    prior_manifest = json.loads((pre / 'output_manifest.json').read_text())
    prior_manifest_sha = sha(pre / 'output_manifest.json')
    for entry in prior_manifest['files']:
        file = pre / entry['path']
        require(file.stat().st_size == entry['size_bytes'] and sha(file) == entry['sha256'],
                f'Preflight output identity mismatch: {entry["path"]}')
    prior = json.loads((pre / 'audit.json').read_text())
    provenance = json.loads((pre / 'source_manifest.json').read_text())
    require(prior['status'] == 'PASS_METADATA_ONLY_CANDIDATE_SUPPORT' and
            prior['csv_h5ad_all_field_mismatches'] == 0, 'Unverified candidate support')
    raw_csv, raw_h5 = Path(contract['obs_csv']), Path(contract['raw_h5ad'])
    for raw in [raw_csv, raw_h5]:
        require(stat(raw) == provenance['input_stat'][str(raw)], 'Source metadata stat drift')
    for path, digest in provenance['full_sha256_csv_and_script'].items():
        require(sha(path) == digest, f'Preflight source/code hash drift: {path}')
    before = {str(path): stat(path) for path in [raw_csv, raw_h5, CONFIG, Path(__file__)]}
    input_hashes = {str(path): sha(path) for path in [raw_csv, CONFIG, Path(__file__)]}
    cells = read_table(pre / 'candidate_source_cell_index.tsv.gz')
    cells['raw_obs_position_zero_based'] = cells.raw_obs_position_zero_based.astype(np.int64)
    require(len(cells) == 240090 and cells.obs_id.is_unique and
            cells.raw_obs_position_zero_based.tolist() == list(range(len(cells))), 'Source cell index drift')
    obs_hash = hashlib.sha256(('\n'.join(cells.obs_id) + '\n').encode()).hexdigest()
    require(obs_hash == provenance['ordered_obs_id_sha256'], 'Candidate obs checksum drift')
    # Prior all-15-field equality reused under unchanged source stat; refresh axes only.
    with h5py.File(raw_h5, 'r') as handle:
        hobs = handle['obs']
        hids = hobs['obs_id'].asstr()[:]
        hindex = hobs[hobs.attrs['_index']].asstr()[:]
    require(np.array_equal(hids, cells.obs_id.to_numpy()) and np.array_equal(hindex, hids),
            'H5AD obs ID/order no longer matches source index')
    vehicle_mask = cells.sm_name.eq(contract['vehicle']['sm_name']) & cells.sm_lincs_id.eq(contract['vehicle']['sm_lincs_id'])
    require(vehicle_mask.equals(cells.source_role.eq('vehicle_DMSO')), 'Vehicle role disagreement')
    require(cells.loc[cells.sm_name.isin(contract['vehicle']['positive_controls_remain_active_treatments']),
                      'source_role'].eq('active_positive_control').all(), 'Positive controls became vehicle')
    m = read_table(pre / 'source_measurements.tsv').rename(columns={'vehicle_pool_id': 'control_pool_id'})
    m['n_source_cells'] = m.n_source_cells.astype(int)
    m['n_vehicle_cells'] = m.n_vehicle_cells.astype(int)
    panel = read_table(pre / 'panel_support_comparison.tsv')
    main_panel = panel[panel.threshold.eq('t20_v40')]
    selected_types = sorted(main_panel.loc[main_panel.n_common_supported_compounds.astype(int).ge(20), 'cell_type'].unique())
    require(len(selected_types) > 0, 'No cell type meets fixed-exposure gene endpoint support')
    queries = read_table(pre / 'source_queries_common_support.tsv')
    queries['type_admitted'] = queries.cell_type.isin(selected_types)
    queries['formal_query_admitted'] = queries.type_admitted & queries.common_any_supported_t20_v40.eq('True')
    queries['condition_endpoint_supported'] = queries.formal_query_admitted
    query_keys = queries[QKEY + ['source_query_id', 'type_admitted', 'formal_query_admitted']]
    require(not query_keys.duplicated(QKEY).any(), 'Ambiguous query admission')
    m = m.merge(query_keys, on=QKEY, how='left', validate='many_to_one')
    m['type_admitted'] = m.cell_type.isin(selected_types)
    m['formal_query_admitted'] = m.formal_query_admitted.fillna(False).astype(bool)
    m['measurement_count_eligible'] = m.eligible_t20_v40.eq('True')
    m['formal_measurement_admitted'] = m.measurement_count_eligible & m.formal_query_admitted & m.type_admitted
    def exclusion(row):
        if row.source_role == 'vehicle_DMSO':
            return 'vehicle_not_treated_measurement'
        reasons = []
        if not row.type_admitted: reasons.append('type_below_20_common_compounds_at_any_fixed_exposure')
        if not row.measurement_count_eligible: reasons.append(row.reason_t20_v40)
        if not row.formal_query_admitted: reasons.append('not_admitted_three_donor_common_query')
        return '|'.join(reasons) if reasons else 'ADMITTED'
    m['formal_exclusion_reasons'] = m.apply(exclusion, axis=1)
    admitted = m[m.formal_measurement_admitted].copy()
    require(not admitted.empty, 'No admitted measurements')

    contexts = cells[CKEY].drop_duplicates().sort_values(CKEY).reset_index(drop=True)
    contexts['context_id'] = key_id(contexts, CKEY, 'kctx_')
    contexts['type_admitted'] = contexts.cell_type.isin(selected_types)
    pools = read_table(pre / 'vehicle_pool_dependencies.tsv').rename(columns={
        'vehicle_pool_id': 'control_pool_id', 'n_vehicle_cells': 'n_source_cells', 'vehicle_well': 'well'})
    pools['n_source_cells'] = pools.n_source_cells.astype(int)
    # Remove the misleading inherited label; physical cardinality is not independence.
    pools = pools.drop(columns=['independent_vehicle_wells', 'cell_split_assignment'])
    pools['n_vehicle_wells'] = pools.n_vehicle_wells.astype(int)
    require(pools.n_vehicle_wells.eq(1).all(), 'Vehicle pool has multiple physical source wells')
    pools = pools.merge(contexts, on=CKEY, validate='many_to_one')
    pools['count_eligible'] = pools.n_source_cells.ge(40)
    pools['included_in_state'] = pools.count_eligible & pools.type_admitted
    state_counts = pools[pools.included_in_state].groupby('context_id').size()
    pools['state_weight'] = [1 / state_counts[cid] if include else 0.0
                             for cid, include in zip(pools.context_id, pools.included_in_state)]
    pools['independent_wells_claimed'] = False
    controls = cells[vehicle_mask][['obs_id', 'raw_obs_position_zero_based', *PKEY, 'well',
        'source_role', 'vehicle_pool_id']].rename(columns={'vehicle_pool_id': 'control_pool_id'})
    controls = controls.merge(pools[['control_pool_id', 'context_id', 'count_eligible', 'included_in_state']],
        on='control_pool_id', validate='many_to_one')
    controls['partition_hash'] = controls.obs_id.map(lambda obs: hashlib.sha256(f'{seed}|{obs}'.encode()).hexdigest())
    controls['partition_rank_zero_based'] = -1
    controls['control_role'] = 'EXCLUDED_POOL_LT40'
    for pool_id, group in controls[controls.count_eligible].groupby('control_pool_id', sort=True):
        ordered = group.sort_values(['partition_hash', 'obs_id']).index
        controls.loc[ordered, 'partition_rank_zero_based'] = np.arange(len(ordered), dtype=int)
        controls.loc[ordered, 'control_role'] = np.where(np.arange(len(ordered)) % 2 == 0, 'A', 'B')
    require(controls.obs_id.is_unique, 'Vehicle cell assigned multiple times')
    for role in ['A', 'B']:
        counts = controls[controls.control_role.eq(role)].groupby('control_pool_id').size()
        pools[f'n_{role}'] = pools.control_pool_id.map(counts).fillna(0).astype(int)
    require((pools.loc[pools.count_eligible, 'n_A'] == (pools.loc[pools.count_eligible, 'n_source_cells'] + 1)//2).all(), 'A not ceil(N/2)')
    require((pools.loc[pools.count_eligible, 'n_B'] == pools.loc[pools.count_eligible, 'n_source_cells']//2).all(), 'B not floor(N/2)')
    require(pools.loc[pools.count_eligible, ['n_A', 'n_B']].ge(20).all().all(), 'Insufficient A/B support')
    require(not (set(controls.loc[controls.control_role.eq('A'), 'obs_id']) &
                 set(controls.loc[controls.control_role.eq('B'), 'obs_id'])), 'A/B overlap')
    require(int(pools.loc[pools.count_eligible, ['n_A', 'n_B']].to_numpy().sum()) ==
            int(pools.loc[pools.count_eligible, 'n_source_cells'].sum()), 'Eligible vehicle cells lost')

    atoms = admitted.groupby(AKEY, sort=True).agg(sm_name=('sm_name', 'first'), source_role=('source_role', 'first'),
        n_admitted_measurements=('measurement_id', 'size'), n_treated_cells=('n_source_cells', 'sum')).reset_index()
    atoms = atoms.sort_values(AKEY).reset_index(drop=True)
    atoms.insert(0, 'atomic_row', np.arange(len(atoms), dtype=int))
    atoms['atomic_id'] = key_id(atoms, AKEY, 'ka_')
    require(atoms.atomic_id.is_unique, 'Atomic ID collision')
    observed_counts = m[m.source_role.ne('vehicle_DMSO')].groupby(AKEY).size().rename('n_observed_measurements').reset_index()
    atoms = atoms.merge(observed_counts, on=AKEY, validate='one_to_one').merge(contexts[CKEY + ['context_id']], on=CKEY, validate='many_to_one')
    atoms['study_id'] = 'kaggle_cross_patient'
    atoms['measurement_scale'] = contract['measurement_scale']
    atoms['condition_endpoint_supported'] = True
    admitted = admitted.merge(atoms[AKEY + ['atomic_row', 'atomic_id', 'context_id', 'n_admitted_measurements']],
        on=AKEY, validate='many_to_one').sort_values(['atomic_row', 'plate_name', 'library_id', 'well']).reset_index(drop=True)
    admitted.insert(0, 'measurement_row', np.arange(len(admitted), dtype=int))
    admitted = admitted.rename(columns={'n_source_cells': 'n_treated_cells'})
    admitted['atomic_measurement_weight'] = 1 / admitted.n_admitted_measurements
    admitted = admitted.merge(pools[['control_pool_id', 'n_A', 'n_B', 'included_in_state']], on='control_pool_id', validate='many_to_one').rename(columns={'n_A': 'n_control_A', 'n_B': 'n_control_B'})
    require(admitted.included_in_state.all(), 'Matched B pool outside eligible type state')
    require(np.allclose(admitted.groupby('atomic_id').atomic_measurement_weight.sum(), 1), 'Atom weights not one')
    require(atoms.groupby(QKEY).donor_id.nunique().eq(3).all(), 'Admitted query does not have all donors')
    endpoint = atoms.groupby(['cell_type', 'donor_id', 'dose_uM', 'timepoint_hr']).agg(
        n_unique_source_compounds=('sm_lincs_id', 'nunique'), n_atomic_conditions=('atomic_id', 'size')).reset_index()
    endpoint['gene_endpoint_supported'] = endpoint.n_unique_source_compounds.ge(20)
    endpoint['condition_endpoint_supported'] = endpoint.n_atomic_conditions.gt(0)
    endpoint['gene_min_unique_compounds'] = 20
    atoms = atoms.merge(endpoint[['cell_type', 'donor_id', 'dose_uM', 'timepoint_hr', 'gene_endpoint_supported']],
        on=['cell_type', 'donor_id', 'dose_uM', 'timepoint_hr'], validate='many_to_one')

    context_counts = pools.groupby('context_id').agg(n_all_source_vehicle_pools=('control_pool_id', 'size'),
        n_count_eligible_vehicle_pools=('count_eligible', 'sum'), n_state_vehicle_pools=('included_in_state', 'sum'),
        n_all_source_vehicle_cells=('n_source_cells', 'sum')).reset_index()
    state_cells = controls[controls.included_in_state].groupby(['context_id', 'control_role']).size().unstack(fill_value=0)
    contexts = contexts.merge(context_counts, on='context_id', validate='one_to_one')
    contexts['n_state_A_cells'] = contexts.context_id.map(state_cells.get('A', pd.Series(dtype=int))).fillna(0).astype(int)
    contexts['n_state_B_cells'] = contexts.context_id.map(state_cells.get('B', pd.Series(dtype=int))).fillna(0).astype(int)
    contexts['included_in_tasks'] = contexts.type_admitted
    contexts['state_definition'] = 'EQUAL_WELL_MEAN_OF_ALL_ELIGIBLE_CONTROL_A_WELLS_IN_DONOR_TYPE'
    contexts['source_type_annotation'] = 'SOURCE_RELEASE_RNA_DERIVED_RETROSPECTIVE'
    context_lookup = contexts.set_index(['donor_id', 'cell_type'])
    task_records, membership_records, pair_records = [], [], []
    for cell_type in selected_types:
        type_atoms = atoms[atoms.cell_type.eq(cell_type)]
        for target in donors:
            sources = [donor for donor in donors if donor != target]
            task_id = 'kt_' + hashlib.sha256(f'{cell_type}|{target}'.encode()).hexdigest()[:20]
            source_atoms, target_atoms = type_atoms[type_atoms.donor_id.isin(sources)], type_atoms[type_atoms.donor_id.eq(target)]
            n_source_a = sum(int(context_lookup.loc[(donor, cell_type), 'n_state_A_cells']) for donor in sources)
            task_records.append(dict(task_id=task_id, cell_type=cell_type, heldout_donor=target,
                source_donors='|'.join(sources), target_context_id=context_lookup.loc[(target, cell_type), 'context_id'],
                source_context_ids='|'.join(context_lookup.loc[(donor, cell_type), 'context_id'] for donor in sources),
                n_source_atomic_conditions=len(source_atoms), n_query_atomic_conditions=len(target_atoms),
                n_source_A_cells_for_feature_selection=n_source_a, required_feature_count=3000,
                source_only_gene_selection_done=False, expression_read=False,
                task_scope='DONOR_HELDOUT_WITHIN_FIXED_TYPE_WITH_PLATE_LIBRARY_SHIFT'))
            for row in type_atoms.itertuples(index=False):
                membership_records.append(dict(task_id=task_id, atomic_row=row.atomic_row, atomic_id=row.atomic_id,
                    role='query' if row.donor_id == target else 'source', donor_id=row.donor_id, cell_type=cell_type,
                    context_id=row.context_id, sm_lincs_id=row.sm_lincs_id, dose_uM=row.dose_uM, timepoint_hr=row.timepoint_hr))
            source_lookup = {tuple(getattr(row, col) for col in QKEY + ['donor_id']): row
                             for row in source_atoms.itertuples(index=False)}
            for target_row in target_atoms.itertuples(index=False):
                for donor in sources:
                    match_key = tuple(getattr(target_row, col) for col in QKEY) + (donor,)
                    require(match_key in source_lookup, 'Source same-drug-dose-time coverage missing')
                    source_row = source_lookup[match_key]
                    pair_records.append(dict(task_id=task_id, query_atomic_row=target_row.atomic_row,
                        query_atomic_id=target_row.atomic_id, query_donor=target,
                        source_atomic_row=source_row.atomic_row, source_atomic_id=source_row.atomic_id,
                        source_donor=donor, cell_type=cell_type, sm_lincs_id=target_row.sm_lincs_id,
                        dose_uM=target_row.dose_uM, timepoint_hr=target_row.timepoint_hr, same_drug_source_weight=0.5))
    tasks, membership, pairs = pd.DataFrame(task_records), pd.DataFrame(membership_records), pd.DataFrame(pair_records)
    require(len(tasks) == 3 * len(selected_types), 'Wrong leave-one-donor-out task count')
    for task in tasks.itertuples(index=False):
        rows = membership[membership.task_id.eq(task.task_id)]
        require(rows.loc[rows.role.eq('source'), 'donor_id'].ne(task.heldout_donor).all(), 'Heldout donor leaked to training')
        require(rows.loc[rows.role.eq('query'), 'donor_id'].eq(task.heldout_donor).all(), 'Query contains source donor')
    require(pairs.groupby(['task_id', 'query_atomic_id']).source_donor.nunique().eq(2).all(), 'Not exactly two source atoms')

    cells = cells.rename(columns={'measurement_id': 'source_measurement_id', 'vehicle_pool_id': 'control_pool_id'})
    cells = cells.drop(columns=['cell_split_assignment', 'support_purpose'])
    cells = cells.merge(contexts[CKEY + ['context_id']], on=CKEY, validate='many_to_one')
    admission_lookup = admitted[['measurement_id', 'measurement_row', 'atomic_id', 'atomic_row']].rename(columns={'measurement_id': 'source_measurement_id'})
    cells = cells.merge(admission_lookup, on='source_measurement_id', how='left', validate='many_to_one')
    cells['measurement_id'] = np.where(cells.atomic_id.notna(), cells.source_measurement_id, '')
    cells['atomic_id'] = cells.atomic_id.fillna('')
    for field in ['atomic_row', 'measurement_row']:
        cells[field] = cells[field].fillna(-1).astype(int)
    cells = cells.merge(controls[['obs_id', 'control_role', 'partition_hash', 'partition_rank_zero_based', 'included_in_state']],
        on='obs_id', how='left', validate='one_to_one')
    cells['control_role'] = cells.control_role.fillna('NOT_VEHICLE')
    cells['included_in_state'] = cells.included_in_state.fillna(False).astype(bool)
    cells['partition_rank_zero_based'] = cells.partition_rank_zero_based.fillna(-1).astype(int)
    cells['partition_hash'] = cells.partition_hash.fillna('')
    cells['treated_admitted'] = cells.atomic_row.ge(0)
    cells['cell_role'] = np.select([cells.treated_admitted, cells.control_role.eq('A'), cells.control_role.eq('B'),
        cells.control_role.eq('EXCLUDED_POOL_LT40')], ['treated', 'vehicle_A', 'vehicle_B', 'vehicle_pool_below_support'],
        default='active_not_admitted')
    cells['formal_measurement_reason'] = cells.source_measurement_id.map(m.set_index('measurement_id').formal_exclusion_reasons)
    cells = cells.sort_values('raw_obs_position_zero_based').reset_index(drop=True)
    require(not (cells.treated_admitted & cells.control_role.isin(['A', 'B'])).any(), 'Treated/control role overlap')
    require(cells.loc[cells.treated_admitted, 'source_role'].ne('vehicle_DMSO').all(), 'Vehicle entered treated atom')
    require(int(cells.treated_admitted.sum()) == int(atoms.n_treated_cells.sum()), 'Treated cell index count mismatch')
    require(cells.obs_id.is_unique and len(cells) == prior['source_csv_rows'], 'Source cells lost/duplicated')
    pool_usage = admitted.groupby('control_pool_id').agg(n_formal_measurements=('measurement_id', 'size'),
        n_formal_atoms=('atomic_id', 'nunique'), n_formal_compounds=('sm_lincs_id', 'nunique')).reset_index()
    pools = pools.merge(pool_usage, on='control_pool_id', how='left', validate='one_to_one')
    for field in ['n_formal_measurements', 'n_formal_atoms', 'n_formal_compounds']:
        pools[field] = pools[field].fillna(0).astype(int)
    wells = read_table(pre / 'source_well_dependencies.tsv')
    formal_wells = admitted.groupby(['plate_name', 'well']).agg(n_formal_measurement_strata=('measurement_id', 'size'),
        n_formal_source_types=('cell_type', 'nunique')).reset_index()
    wells = wells.merge(formal_wells, on=['plate_name', 'well'], how='left', validate='one_to_one')
    for field in ['n_formal_measurement_strata', 'n_formal_source_types']:
        wells[field] = wells[field].fillna(0).astype(int)
    compounds = cells[cells.source_role.ne('vehicle_DMSO')][['sm_lincs_id', 'sm_name', 'SMILES', 'dose_uM', 'timepoint_hr', 'source_role']].drop_duplicates().sort_values(['sm_lincs_id', 'dose_uM', 'timepoint_hr'])
    require(compounds.sm_lincs_id.is_unique, 'Expected one released active exposure per compound')
    type_admission = main_panel.copy()
    type_admission['type_admitted'] = type_admission.cell_type.isin(selected_types)

    after = {str(path): stat(path) for path in [raw_csv, raw_h5, CONFIG, Path(__file__)]}
    require(before == after and sha(CONFIG) == contract_sha, 'Source/config changed before freeze')
    require(all(sha(path) == digest for path, digest in input_hashes.items()), 'Source/config/script hash changed')
    require(sha(pre / 'output_manifest.json') == prior_manifest_sha, 'Preflight manifest changed')
    OUT.mkdir(parents=True)
    tables = {'atomic_index.tsv': atoms, 'measurement_index.tsv': admitted,
        'measurement_exclusions.tsv': m, 'control_pool_index.tsv': pools,
        'control_cell_index.tsv.gz': controls.sort_values('raw_obs_position_zero_based'),
        'cell_index.tsv.gz': cells, 'context_index.tsv': contexts, 'task_index.tsv': tasks,
        'fold_atomic_membership.tsv': membership, 'same_drug_source_pairs.tsv': pairs,
        'gene_endpoint_support.tsv': endpoint, 'source_query_admission.tsv': queries,
        'type_admission.tsv': type_admission, 'source_well_dependencies.tsv': wells,
        'compound_index.tsv': compounds}
    for name, frame in tables.items(): save_table(name, frame)
    with (OUT / 'frozen_contract.json').open('xb') as stream:
        stream.write(contract_bytes)
    audit = dict(status='PASS_FORMAL_METADATA_AND_CONTROL_ROLES_ONLY', completed_utc=datetime.now(timezone.utc).isoformat(),
        contract_path=str(CONFIG), contract_sha256=contract_sha, contract_frozen_before_expression=True,
        selected_types=selected_types, source_cells=len(cells), admitted_treated_cells=int(cells.treated_admitted.sum()),
        source_measurements=len(m), admitted_measurements=len(admitted), atomic_conditions=len(atoms),
        admitted_queries=int(queries.formal_query_admitted.sum()), tasks=len(tasks),
        all_source_contexts=len(contexts), task_contexts=int(contexts.included_in_tasks.sum()),
        all_source_vehicle_pools=len(pools), globally_count_eligible_partitioned_pools=int(pools.count_eligible.sum()),
        state_vehicle_pools=int(pools.included_in_state.sum()),
        globally_assigned_A_cells=int(controls.control_role.eq('A').sum()),
        globally_assigned_B_cells=int(controls.control_role.eq('B').sum()),
        state_A_cells=int((controls.included_in_state & controls.control_role.eq('A')).sum()),
        state_B_cells=int((controls.included_in_state & controls.control_role.eq('B')).sum()),
        control_partition_seed=seed, partition_payload='decimal_seed|source_obs_id UTF-8',
        partition_order='sha256 hex ascending then obs_id ascending; zero-based even=A odd=B',
        A_B_disjoint=True, all_eligible_control_cells_assigned_exactly_once=True,
        globally_partitioned_unsupported_types_excluded_from_tasks_and_state=True,
        state_control_pools_independent_of_retained_drugs=True, no_query_donor_in_source_training=True,
        exact_two_source_atoms_per_same_drug_dose_time_query=True, equal_measurement_weights_sum_to_one=True,
        control_A_B_are_independent_wells=False, biological_independence_claimed=False,
        expression_read=False, effects_created=False, features_selected=False, embeddings_created=False, models_run=False,
        h5ad_paths_read=['/obs/obs_id', '/obs/_index'], all_15_source_obs_fields_reused_from_verified_preflight=True,
        full_h5ad_hash_computed=False, source_obs_id_sha256=obs_hash,
        source_input_stat_before=before, source_input_stat_after=after, source_input_hashes=input_hashes,
        preflight_output_manifest_sha256=prior_manifest_sha,
        elapsed_seconds=time.perf_counter()-started, peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        versions=dict(python=sys.version.split()[0], pandas=pd.__version__, numpy=np.__version__, h5py=h5py.__version__),
        executable=sys.executable)
    save_json('audit.json', audit)
    save_json('source_manifest.json', dict(contract_sha256=contract_sha, input_stat=before, full_sha256=input_hashes,
        preflight_output_manifest_sha256=prior_manifest_sha, preflight_source_manifest_sha256=sha(pre/'source_manifest.json'),
        h5ad_full_sha256=None, ordered_obs_id_sha256=obs_hash,
        source_data_page=prior['source_data_page'], expression_read=False))
    save_json('output_manifest.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        contract_sha256=contract_sha, files=[dict(path=file.name, size_bytes=file.stat().st_size, sha256=sha(file))
          for file in sorted(OUT.iterdir()) if file.is_file()]))
    print(json.dumps(audit, indent=2))
    print(tasks.to_string(index=False))


if __name__ == '__main__':
    run()
