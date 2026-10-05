#!/usr/bin/env python3
"""Metadata-only candidate support; never read expression or freeze a cohort.

Read the released observation CSV and /obs fields in the local H5AD only.
Keep all source cells/measurement keys. Thresholds describe feasibility; no
cell partition, feature selection, effect, model, or final task is constructed.
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
RAW = Path('${PCA_DATA_ROOT}/2_DrugOrSmallmoleculePerturbationDatasets/3_KaggleCrossPatient/0_raw')
CSV = RAW / 'open-problems-single-cell-perturbations/adata_obs_meta.csv'
H5 = RAW / 'raw.h5ad'
OUT = ROOT / 'metadata/kaggle_task_support_v1'
ROLE_MAP = {('Dimethyl Sulfoxide', 'LSM-36361'): 'vehicle_DMSO',
            ('Belinostat', 'LSM-43181'): 'active_positive_control',
            ('Dabrafenib', 'LSM-6303'): 'active_positive_control'}
THRESHOLDS = [(20, 40), (10, 20), (20, 20), (10, 40)]
MKEY = ['sm_lincs_id', 'sm_name', 'dose_uM', 'timepoint_hr', 'donor_id',
        'cell_type', 'library_id', 'plate_name', 'well']
PKEY = ['donor_id', 'cell_type', 'library_id', 'plate_name']
QKEY = ['cell_type', 'sm_lincs_id', 'sm_name', 'dose_uM', 'timepoint_hr']
PURPOSE = 'CANDIDATE_SUPPORT_ONLY_NOT_FORMAL_COHORT'


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


def read_obs(node):
    if isinstance(node, h5py.Dataset):
        return node.asstr()[:] if h5py.check_string_dtype(node.dtype) else node[:]
    require(set(node.keys()) == {'categories', 'codes'}, 'Unexpected obs encoding')
    categories, codes = read_obs(node['categories']), node['codes'][:]
    require(bool((codes >= 0).all()), 'Missing categorical metadata')
    return categories[codes]


def ids(frame, columns, prefix):
    return frame[columns].astype(str).apply(lambda row: prefix + hashlib.sha256(
        json.dumps(row.tolist(), ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()[:20], axis=1)


def save_table(name, frame):
    with (OUT / name).open('x') as stream:
        frame.to_csv(stream, sep='\t', index=False, lineterminator='\n', na_rep='NA')


def save_json(name, value):
    with (OUT / name).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def run():
    started = time.perf_counter()
    require(not OUT.exists(), f'Refuse existing output: {OUT}')
    source_paths = [CSV, H5, Path(__file__)]
    before = {str(path): stat(path) for path in source_paths}
    source_hashes = {str(path): sha(path) for path in [CSV, Path(__file__)]}
    obs = pd.read_csv(CSV, dtype=str, keep_default_na=False)
    source_columns = list(obs.columns)
    require(len(obs) == 240090 and obs.obs_id.is_unique, 'Unexpected/nonunique source cells')
    require(not obs.eq('').any().any(), 'Missing source metadata')
    identity = []
    numeric = {'col', 'dose_uM', 'timepoint_hr'}
    with h5py.File(H5, 'r') as handle:
        hobs = handle['obs']
        require(set(source_columns).issubset(hobs), 'Missing H5AD obs fields')
        raw_index = read_obs(hobs[hobs.attrs['_index']])
        require(np.array_equal(raw_index, obs.obs_id.to_numpy()), 'H5AD obs index/order mismatch')
        for field in source_columns:
            values = read_obs(hobs[field])
            if field in numeric:
                same = pd.to_numeric(obs[field]).to_numpy(dtype=float) == np.asarray(values, dtype=float)
                rule = 'numeric_exact_equality'
            elif field == 'control':
                require(obs[field].str.lower().isin(['true', 'false']).all(), 'Unexpected control encoding')
                same = obs[field].str.lower().to_numpy() == np.char.lower(np.asarray(values).astype(str))
                rule = 'boolean_case_canonicalization_only'
            else:
                same = obs[field].to_numpy(dtype=str) == np.asarray(values).astype(str)
                rule = 'exact_string_equality'
            identity.append(dict(field=field, compared_cells=len(obs), mismatches=int((~same).sum()), rule=rule))
            require(bool(same.all()), f'H5AD/CSV mismatch: {field}')
    obs['raw_obs_position_zero_based'] = np.arange(len(obs), dtype=np.int64)
    obs['source_role'] = [ROLE_MAP.get(pair, 'active_other') for pair in zip(obs.sm_name, obs.sm_lincs_id)]
    require(obs.loc[obs.control.str.lower().eq('true'), 'source_role'].isin(
        ['vehicle_DMSO', 'active_positive_control']).all(), 'Unmapped source control')
    require(obs.loc[obs.source_role.ne('active_other'), 'control'].str.lower().eq('true').all(),
            'Expected mapped source-control flag')
    require(obs.groupby('sm_lincs_id').sm_name.nunique().max() == 1 and
            obs.groupby('sm_name').sm_lincs_id.nunique().max() == 1, 'Compound identity not one-to-one')
    donors, types = sorted(obs.donor_id.unique()), sorted(obs.cell_type.unique())
    require(len(donors) == 3 and len(types) == 6, 'Unexpected released donor/type axes')

    measurement = obs.groupby(MKEY, sort=True, dropna=False).agg(
        n_source_cells=('obs_id', 'size'), source_role=('source_role', 'first'),
        source_control_flag=('control', 'first'), source_row=('row', 'first')).reset_index()
    measurement['measurement_id'] = ids(measurement, MKEY, 'km_')
    require(measurement.measurement_id.is_unique, 'Measurement ID collision')
    require(int(measurement.n_source_cells.sum()) == len(obs), 'Cell count conservation failed')
    vehicle = obs[obs.source_role.eq('vehicle_DMSO')]
    pools = vehicle.groupby(PKEY, sort=True).agg(n_vehicle_cells=('obs_id', 'size'),
        n_vehicle_wells=('well', 'nunique'), vehicle_well=('well', 'first'),
        vehicle_dose_source=('dose_uM', 'first'), vehicle_timepoint_source=('timepoint_hr', 'first')).reset_index()
    pools['vehicle_pool_id'] = ids(pools, PKEY, 'kv_')
    require(pools.vehicle_pool_id.is_unique and pools.n_vehicle_wells.eq(1).all(),
            'Matched vehicle pool is not exactly one source well')
    measurement = measurement.merge(pools, on=PKEY, how='left', validate='many_to_one')
    measurement['matched_vehicle_present'] = measurement.vehicle_pool_id.notna()
    measurement['n_vehicle_cells'] = measurement.n_vehicle_cells.fillna(0).astype(int)
    measurement['active_measurement'] = measurement.source_role.ne('vehicle_DMSO')
    for treated, control in THRESHOLDS:
        label = f't{treated}_v{control}'
        measurement[f'treated_support_{treated}'] = measurement.active_measurement & measurement.n_source_cells.ge(treated)
        measurement[f'vehicle_support_{control}'] = measurement.n_vehicle_cells.ge(control)
        measurement[f'eligible_{label}'] = (measurement[f'treated_support_{treated}'] &
            measurement[f'vehicle_support_{control}'] & measurement.matched_vehicle_present)
        measurement[f'reason_{label}'] = np.select(
            [~measurement.active_measurement, ~measurement.matched_vehicle_present,
             ~measurement.n_source_cells.ge(treated) & ~measurement.n_vehicle_cells.ge(control),
             ~measurement.n_source_cells.ge(treated), ~measurement.n_vehicle_cells.ge(control)],
            ['vehicle_not_treated_candidate', 'no_strictly_matched_vehicle', 'treated_and_vehicle_below_threshold',
             'treated_below_threshold', 'vehicle_below_threshold'], default='eligible_candidate')
    measurement['support_purpose'] = PURPOSE
    active = measurement[measurement.active_measurement].copy()
    require(active.matched_vehicle_present.all(), 'Some source treated measurement lacks matched DMSO')
    obs = obs.merge(measurement[MKEY + ['measurement_id', 'vehicle_pool_id']], on=MKEY, how='left', validate='many_to_one')
    require(obs.measurement_id.notna().all() and obs.raw_obs_position_zero_based.is_monotonic_increasing,
            'Lost source cell lineage/order')
    obs['support_purpose'] = PURPOSE
    obs['cell_split_assignment'] = 'NOT_ASSIGNED'

    # Cross-source compound identity is fixed by released name+LINCS+dose+time.
    exposures = active[['sm_lincs_id', 'sm_name', 'dose_uM', 'timepoint_hr', 'source_role']].drop_duplicates()
    queries = pd.DataFrame({'cell_type': types}).merge(exposures, how='cross')
    queries['source_query_id'] = ids(queries, QKEY, 'kq_')
    require(queries.source_query_id.is_unique, 'Source query ID collision')
    qdonor = queries.merge(pd.DataFrame({'donor_id': donors}), how='cross')
    key = QKEY + ['donor_id']
    counts = active.groupby(key).agg(n_observed_measurements=('measurement_id', 'size'),
        n_observed_treated_cells=('n_source_cells', 'sum'),
        min_treated_cells_per_measurement=('n_source_cells', 'min'),
        max_treated_cells_per_measurement=('n_source_cells', 'max'),
        n_observed_libraries=('library_id', 'nunique'), n_observed_plates=('plate_name', 'nunique')).reset_index()
    qdonor = qdonor.merge(counts, on=key, how='left', validate='one_to_one')
    countcols = [column for column in counts if column not in key]
    qdonor[countcols] = qdonor[countcols].fillna(0).astype(int)
    qdonor['observed'] = qdonor.n_observed_measurements.gt(0)
    qdonor['zero_count_means'] = 'NO_RELEASED_METADATA_RECORD_NOT_ZERO_EXPRESSION'
    for treated, control in THRESHOLDS:
        label = f't{treated}_v{control}'
        ok = active[active[f'eligible_{label}']]
        elig = ok.groupby(key).agg(**{
            f'n_eligible_measurements_{label}': ('measurement_id', 'size'),
            f'n_eligible_treated_cells_{label}': ('n_source_cells', 'sum'),
            f'n_eligible_vehicle_pools_{label}': ('vehicle_pool_id', 'nunique')}).reset_index()
        qdonor = qdonor.merge(elig, on=key, how='left', validate='one_to_one')
        ecols = [column for column in elig if column not in key]
        qdonor[ecols] = qdonor[ecols].fillna(0).astype(int)
        qdonor[f'any_measurement_supported_{label}'] = qdonor[f'n_eligible_measurements_{label}'].gt(0)
        qdonor[f'all_observed_measurements_supported_{label}'] = qdonor.observed & (
            qdonor[f'n_eligible_measurements_{label}'] == qdonor.n_observed_measurements)

    common_records, fold_records, support_records = [], [], []
    for _, query in queries.iterrows():
        rows = qdonor[qdonor.source_query_id.eq(query.source_query_id)]
        record = query.to_dict()
        record.update(n_observed_donors=int(rows.observed.sum()), common_observed_all_3=bool(rows.observed.all()),
                      n_observed_measurements=int(rows.n_observed_measurements.sum()))
        for treated, control in THRESHOLDS:
            label = f't{treated}_v{control}'
            record[f'common_any_supported_{label}'] = bool(rows[f'any_measurement_supported_{label}'].all())
            record[f'common_all_observed_supported_{label}'] = bool(rows[f'all_observed_measurements_supported_{label}'].all())
            record[f'n_eligible_measurements_{label}'] = int(rows[f'n_eligible_measurements_{label}'].sum())
        common_records.append(record)
    common = pd.DataFrame(common_records)
    panelcols = ['cell_type', 'dose_uM', 'timepoint_hr']
    for panel, group in common.groupby(panelcols, sort=True):
        panel_dict = dict(zip(panelcols, panel))
        for treated, control in THRESHOLDS:
            label = f't{treated}_v{control}'
            supported = group[group[f'common_any_supported_{label}']]
            all_observed = group[group[f'common_all_observed_supported_{label}']]
            qids = set(supported.source_query_id)
            subset = qdonor[qdonor.source_query_id.isin(qids)]
            base = dict(**panel_dict, threshold=label, min_treated_per_measurement=treated,
                min_vehicle_total=control, implied_min_cells_per_disjoint_vehicle_half=control // 2,
                n_released_source_compounds=len(group), n_common_observed_compounds=int(group.common_observed_all_3.sum()),
                n_common_supported_compounds=len(supported),
                n_common_all_observed_supported_compounds=len(all_observed),
                gene_metric_min_20_compounds_met=len(supported) >= 20,
                gene_metric_min_20_compounds_all_observed_met=len(all_observed) >= 20,
                n_common_supported_positive_control_compounds=int(supported.source_role.eq('active_positive_control').sum()),
                support_purpose=PURPOSE)
            support_records.append(base)
            for target in donors:
                sources = [donor for donor in donors if donor != target]
                source_rows, target_rows = subset[subset.donor_id.isin(sources)], subset[subset.donor_id.eq(target)]
                fold_records.append(dict(**base, heldout_donor=target, source_donors='|'.join(sources),
                    n_source_supported_measurements=int(source_rows[f'n_eligible_measurements_{label}'].sum()),
                    n_query_supported_measurements=int(target_rows[f'n_eligible_measurements_{label}'].sum()),
                    n_source_treated_cells=int(source_rows[f'n_eligible_treated_cells_{label}'].sum()),
                    n_query_treated_cells=int(target_rows[f'n_eligible_treated_cells_{label}'].sum()),
                    independence_label='DONOR_HELDOUT_WITH_PLATE_LIBRARY_SHIFT_NOT_PURE_DONOR_EFFECT'))
    panels, folds = pd.DataFrame(support_records), pd.DataFrame(fold_records)

    # Explicit dependency indices preserve dense controls and common physical wells.
    wells = obs.groupby(['donor_id', 'plate_name', 'well', 'library_id', 'row', 'sm_lincs_id', 'sm_name',
                         'dose_uM', 'timepoint_hr', 'source_role']).agg(
        n_source_cells=('obs_id', 'size'), n_source_cell_types=('cell_type', 'nunique'),
        n_measurement_strata=('measurement_id', 'nunique')).reset_index()
    require(len(wells) == 576, 'Unexpected source well keys')
    require(obs.groupby('library_id').donor_id.nunique().eq(1).all() and
            obs.groupby('library_id').plate_name.nunique().eq(1).all() and
            obs.groupby('library_id')['row'].nunique().eq(1).all() and
            obs.groupby('plate_name').donor_id.nunique().eq(1).all(), 'Nesting changed')
    require(obs.groupby(['plate_name', 'well']).library_id.nunique().eq(1).all() and
            obs.groupby(['plate_name', 'well']).sm_lincs_id.nunique().eq(1).all(),
            'Physical source well has ambiguous library or compound')
    pooldep = active.groupby('vehicle_pool_id').agg(n_matched_active_measurements=('measurement_id', 'size'),
        n_matched_source_compounds=('sm_lincs_id', 'nunique'),
        n_matched_active_cells=('n_source_cells', 'sum')).reset_index()
    pooldep = pools.merge(pooldep, on='vehicle_pool_id', how='left', validate='one_to_one')
    pooldep['independent_vehicle_wells'] = 1
    pooldep['cell_split_assignment'] = 'NOT_ASSIGNED'
    for treated, control in THRESHOLDS:
        label = f't{treated}_v{control}'
        elig = active[active[f'eligible_{label}']].groupby('vehicle_pool_id').size()
        pooldep[f'n_eligible_matched_measurements_{label}'] = pooldep.vehicle_pool_id.map(elig).fillna(0).astype(int)
    dense = wells[wells.source_role.eq('active_positive_control')].groupby(
        ['sm_lincs_id', 'sm_name', 'dose_uM', 'timepoint_hr', 'donor_id']).agg(
        n_source_wells=('well', 'size'), n_source_libraries=('library_id', 'nunique'),
        n_source_plates=('plate_name', 'nunique'), n_source_cells=('n_source_cells', 'sum')).reset_index()
    require(len(dense) == 6 and dense.n_source_wells.eq(16).all(), 'Dense positive controls not 16 wells/donor')
    summaries = active.groupby(['cell_type', 'donor_id', 'dose_uM', 'timepoint_hr']).agg(
        n_source_compounds=('sm_lincs_id', 'nunique'), n_source_measurements=('measurement_id', 'size'),
        n_treated_cells=('n_source_cells', 'sum'), n_matched_vehicle_pools=('vehicle_pool_id', 'nunique')).reset_index()
    for treated, control in THRESHOLDS:
        label = f't{treated}_v{control}'
        sub = active[active[f'eligible_{label}']].groupby(['cell_type', 'donor_id', 'dose_uM', 'timepoint_hr']).agg(**{
            f'n_eligible_compounds_{label}': ('sm_lincs_id', 'nunique'),
            f'n_eligible_measurements_{label}': ('measurement_id', 'size')}).reset_index()
        summaries = summaries.merge(sub, how='left', on=['cell_type', 'donor_id', 'dose_uM', 'timepoint_hr'], validate='one_to_one')
    summaries = summaries.fillna(0)

    # Final read-only checks happen before exclusive output creation.
    after = {str(path): stat(path) for path in source_paths}
    require(before == after, 'Input stat changed during metadata audit')
    require(all(sha(path) == digest for path, digest in source_hashes.items()), 'CSV/script hash changed')
    require(len(obs) == int(measurement.n_source_cells.sum()), 'Final source row conservation failed')
    require(qdonor.groupby('source_query_id').size().eq(3).all(), 'Missing explicit donor candidate row')
    OUT.mkdir(parents=True)
    tables = {'source_measurements.tsv': measurement, 'vehicle_pool_dependencies.tsv': pooldep,
              'source_queries_by_donor.tsv': qdonor, 'source_queries_common_support.tsv': common,
              'panel_support_comparison.tsv': panels, 'donor_heldout_fold_support.tsv': folds,
              'counts_by_type_donor_dose_time.tsv': summaries, 'source_well_dependencies.tsv': wells,
              'dense_positive_control_multiplicity.tsv': dense, 'csv_h5ad_obs_identity.tsv': pd.DataFrame(identity)}
    for name, frame in tables.items():
        save_table(name, frame)
    with (OUT / 'candidate_source_cell_index.tsv.gz').open('xb') as raw_stream:
        with gzip.GzipFile(fileobj=raw_stream, mode='wb', mtime=0) as zipped:
            zipped.write(obs.to_csv(sep='\t', index=False, lineterminator='\n').encode())
    audit = dict(status='PASS_METADATA_ONLY_CANDIDATE_SUPPORT', purpose=PURPOSE,
        completed_utc=datetime.now(timezone.utc).isoformat(), source_csv_rows=len(obs),
        source_metadata_columns=len(source_columns), csv_h5ad_all_field_mismatches=0,
        h5ad_obs_order_identical=True, obs_id_sha256=hashlib.sha256(('\n'.join(obs.obs_id) + '\n').encode()).hexdigest(),
        source_roles={str(k): int(v) for k, v in obs.source_role.value_counts().items()},
        n_measurements=len(measurement), n_active_measurements=len(active), n_vehicle_pools=len(pools),
        n_source_wells=len(wells), n_source_libraries=int(obs.library_id.nunique()), n_source_plates=int(obs.plate_name.nunique()),
        n_source_queries=len(queries), n_source_query_donor_rows=len(qdonor),
        n_common_observed_source_queries=int(common.common_observed_all_3.sum()),
        thresholds_descriptive_only=[dict(treated=t, vehicle_total=v, vehicle_half_min=v//2) for t,v in THRESHOLDS],
        measurement_key=MKEY, matched_vehicle_key=PKEY, source_query_key=QKEY,
        compound_roles={'vehicle': 'exact name Dimethyl Sulfoxide AND LINCS LSM-36361',
                        'active_positive_controls': ['Belinostat/LSM-43181', 'Dabrafenib/LSM-6303']},
        h5ad_paths_read=['/obs/_index'] + ['/obs/' + col for col in source_columns],
        expression_read=False, actual_cell_split_made=False, formal_cohort_created=False,
        independent_vehicle_halves=False, cell_type_annotation='SOURCE_RELEASE_RNA_DERIVED_RETROSPECTIVE',
        source_input_stat_before=before, source_input_stat_after=after, source_input_hashes=source_hashes,
        full_h5ad_hash_computed=False, source_data_page='https://www.kaggle.com/competitions/open-problems-single-cell-perturbations/data',
        elapsed_seconds=time.perf_counter()-started, peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        versions=dict(python=sys.version.split()[0], pandas=pd.__version__, numpy=np.__version__, h5py=h5py.__version__),
        executable=sys.executable,
        interpretation='Counts of released measurement keys and candidate support only; not independent biological replicates. '
        'Any-supported versus all-observed-supported policies are both retained; no aggregation policy is selected. '
        'No source-versus-heldout effects, feature values, gene metrics or model performance were examined.')
    save_json('audit.json', audit)
    save_json('source_manifest.json', dict(retrieved_local_utc=audit['completed_utc'],
        canonical_source_url=audit['source_data_page'], input_stat=before,
        full_sha256_csv_and_script=source_hashes, h5ad_full_sha256=None,
        h5ad_verification='ALL_15_OBS_FIELDS_AND_ORDER_PLUS_CONTENT_STAT_ONLY_NO_X',
        ordered_obs_id_sha256=audit['obs_id_sha256'], csv_h5ad_mismatches=0,
        original_source_files_modified=False))
    save_json('output_manifest.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        files=[dict(path=path.name, size_bytes=path.stat().st_size, sha256=sha(path))
               for path in sorted(OUT.iterdir()) if path.is_file()]))
    print(json.dumps({k: audit[k] for k in ['status', 'source_csv_rows', 'n_active_measurements', 'n_vehicle_pools',
        'n_common_observed_source_queries', 'elapsed_seconds', 'peak_rss_mib']}, indent=2))
    print(panels.to_string(index=False))


if __name__ == '__main__':
    run()
