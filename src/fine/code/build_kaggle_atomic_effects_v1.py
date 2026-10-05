#!/usr/bin/env python3
"""Frozen Kaggle provided-scale effects; no raw-count, normalization or model claim.

Only selected A/B control value slices are requested until source-A gene panels
are sealed. Byte-level SHA256 and full sparse-index validation are provenance /
structure I/O, not expression fitting. HDF5 can physically fetch chunks spanning
unselected cells; their values are not returned to statistical aggregation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import resource
import shutil
import sys
import time

sys.dont_write_bytecode = True
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]


def require(value, message):
    if not bool(value):
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def file_stat(path):
    s = Path(path).stat()
    return dict(size=s.st_size, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns,
                inode=s.st_ino, device=s.st_dev)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def tsv(path, **kwargs):
    return pd.read_csv(path, sep='\t', keep_default_na=False, float_precision='round_trip', **kwargs)


def write_tsv(path, frame):
    require(not Path(path).exists(), 'Refuse output overwrite: ' + str(path))
    frame.to_csv(path, sep='\t', index=False, float_format='%.17g', na_rep='NA')


def save_npz(path, **arrays):
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, **arrays)


def boolean(values):
    text = np.asarray(values).astype(str)
    require(np.isin(text, ['True', 'False', 'true', 'false', '1', '0']).all(), 'Invalid Boolean metadata')
    return np.isin(text, ['True', 'true', '1'])


def variance_and_selection(total, squared_total, n_cells, source_rows, n_features):
    """Cell-pooled provided-scale ddof=0 variance; deterministic tie handling."""
    total, squared_total = np.asarray(total, np.float64), np.asarray(squared_total, np.float64)
    source_rows = np.asarray(source_rows)
    require(total.ndim == 1 and squared_total.shape == total.shape and source_rows.shape == total.shape,
            'Variance axes differ')
    require(np.isfinite(total).all() and np.isfinite(squared_total).all() and n_cells >= 2, 'Invalid source A moments/count')
    require(len(np.unique(source_rows)) == len(source_rows), 'Duplicate source gene rows')
    mean, second = total / n_cells, squared_total / n_cells
    raw_variance = second - mean * mean
    tolerance = 64 * np.finfo(np.float64).eps * np.maximum(1.0, second + mean * mean)
    require(np.isfinite(raw_variance).all() and np.all(raw_variance >= -tolerance), 'Materially negative source A variance')
    variance = np.maximum(raw_variance, 0)
    eligible = np.flatnonzero(variance > 0)
    require(len(eligible) >= n_features, 'Fewer positive-variance source A genes than frozen requirement')
    rank = np.lexsort((source_rows[eligible], -variance[eligible]))
    selected = eligible[rank[:n_features]]
    diagnostic = dict(n_positive_variance=int(len(eligible)), n_negative_roundoff_clipped=int((raw_variance < 0).sum()),
                      min_raw_variance=float(raw_variance.min()), maximum_roundoff_tolerance=float(tolerance.max()))
    return selected, variance, diagnostic


def validate_csr_layout(x, expected_shape, rows_per_chunk):
    """Full structural validation, without requesting X/data values."""
    encoding = x.attrs.get('encoding-type')
    if isinstance(encoding, bytes):
        encoding = encoding.decode()
    require(encoding == 'csr_matrix', 'X must be CSR')
    shape = tuple(int(v) for v in x.attrs['shape'])
    require(shape == tuple(expected_shape), 'Frozen source X shape changed')
    require(x['data'].dtype == np.dtype('float32'), 'Expected released float32 X')
    require(x['indices'].dtype.kind in 'iu' and x['indptr'].dtype.kind in 'iu', 'CSR indices must be integers')
    pointers = x['indptr'][:].astype(np.int64)
    require(len(pointers) == shape[0] + 1 and pointers[0] == 0 and np.all(np.diff(pointers) >= 0), 'Invalid CSR indptr')
    require(pointers[-1] == len(x['data']) == len(x['indices']), 'CSR nnz/axis mismatch')
    for start in range(0, shape[0], rows_per_chunk):
        stop = min(shape[0], start + rows_per_chunk)
        lo, hi = int(pointers[start]), int(pointers[stop])
        indices = x['indices'][lo:hi]
        require(np.all((indices >= 0) & (indices < shape[1])), 'CSR gene index out of bounds')
        bad = indices[1:] <= indices[:-1]
        boundaries = pointers[start + 1:stop] - lo
        boundaries = boundaries[(boundaries > 0) & (boundaries < len(indices))]
        bad[boundaries - 1] = False
        require(not bad.any(), 'CSR source rows contain unsorted or duplicate gene entries; no silent repair')
    return pointers, dict(shape=list(shape), nnz=int(pointers[-1]), encoding=encoding,
        data_dtype=str(x['data'].dtype), index_dtype=str(x['indices'].dtype), canonical_all_source_rows=True,
        full_sparse_indices_validated=True, data_values_read_for_structure=False,
        data_chunk_shape=list(x['data'].chunks) if x['data'].chunks else None)


def selected_csr(x, pointers, selected_rows, n_genes):
    """Literal selected-row data requests; no surrounding row-block values."""
    rows = np.asarray(selected_rows, dtype=np.int64)
    require(rows.ndim == 1 and len(rows) and rows[0] >= 0 and rows[-1] < len(pointers) - 1 and np.all(np.diff(rows) > 0),
            'Selected raw rows must be unique, sorted and in range')
    lengths = pointers[rows + 1] - pointers[rows]
    local = np.r_[0, np.cumsum(lengths)].astype(np.int64)
    values = np.empty(int(local[-1]), dtype=np.float64)
    columns = np.empty(int(local[-1]), dtype=np.int32)
    for i, row in enumerate(rows):
        a, b = int(pointers[row]), int(pointers[row + 1])
        values[local[i]:local[i + 1]] = x['data'][a:b]
        columns[local[i]:local[i + 1]] = x['indices'][a:b]
    return sparse.csr_matrix((values, columns, local), shape=(len(rows), n_genes))


def validate_provided_values(matrix):
    require(sparse.isspmatrix_csr(matrix) and matrix.has_canonical_format, 'Noncanonical selected CSR')
    require(np.isfinite(matrix.data).all() and np.all(matrix.data >= 0), 'Selected provided X is nonfinite or negative')
    expression_sum = np.asarray(matrix.sum(axis=1)).ravel()
    require(np.isfinite(expression_sum).all() and np.all(expression_sum > 0), 'Selected cell has no positive provided expression')
    stored = np.diff(matrix.indptr)
    positive = np.zeros(matrix.shape[0], dtype=np.int64)
    nonempty = stored > 0
    positive[nonempty] = np.add.reduceat((matrix.data > 0).astype(np.int64), matrix.indptr[:-1][nonempty])
    return expression_sum, stored, positive


def grouped_reduce(matrix, ids, n_groups):
    ids = np.asarray(ids)
    require(ids.ndim == 1 and len(ids) == matrix.shape[0] and ids.dtype.kind in 'iu', 'Unaligned group IDs')
    require(np.all((ids >= 0) & (ids < n_groups)), 'Group ID outside declared axis')
    present, inverse = np.unique(ids, return_inverse=True)
    reducer = sparse.csr_matrix((np.ones(len(ids)), (inverse, np.arange(len(ids)))), shape=(len(present), len(ids)))
    return present, (reducer @ matrix).toarray()


def aggregate_selected(x, pointers, members, group_column, n_groups, n_genes, batch_size, *, columns=None, squares=False):
    members = members.sort_values('raw_obs_position_zero_based').reset_index(drop=True)
    rows = members.raw_obs_position_zero_based.to_numpy(np.int64)
    require(members.obs_id.is_unique and np.all(np.diff(rows) > 0), 'Duplicate selected cell membership')
    width = n_genes if columns is None else len(columns)
    total = np.zeros((n_groups, width), np.float64)
    squared_total = np.zeros_like(total) if squares else None
    counts = np.zeros(n_groups, np.int64)
    qc_parts, n_values = [], 0
    for offset in range(0, len(members), batch_size):
        chunk = members.iloc[offset:offset + batch_size]
        matrix = selected_csr(x, pointers, chunk.raw_obs_position_zero_based.to_numpy(np.int64), n_genes)
        expression_sum, stored, positive = validate_provided_values(matrix)
        n_values += int(matrix.nnz)
        groups = chunk[group_column].to_numpy(np.int64)
        counts += np.bincount(groups, minlength=n_groups)
        qc = chunk[['obs_id', 'raw_obs_position_zero_based', group_column]].copy()
        qc['provided_expression_sum_all_source_genes'] = expression_sum
        qc['n_stored_source_entries'], qc['n_positive_source_genes'] = stored, positive
        qc['finite_nonnegative_positive_expression'] = True
        qc_parts.append(qc)
        selected = matrix if columns is None else matrix[:, columns]
        present, reduced = grouped_reduce(selected, groups, n_groups)
        total[present] += reduced
        if squares:
            squared = selected.copy()
            squared.data *= squared.data
            require(np.isfinite(squared.data).all(), 'Squared provided values overflow')
            present, reduced = grouped_reduce(squared, groups, n_groups)
            squared_total[present] += reduced
    require(np.all(counts > 0) and np.isfinite(total).all(), 'Empty or invalid aggregation group')
    return total, squared_total, counts, pd.concat(qc_parts, ignore_index=True), dict(
        selected_cells=len(members), selected_stored_values=n_values,
        full_21255_gene_values_checked=True, numerical_transform='NONE', literal_selected_row_requests=True)


def equal_measurement_effects(treated_mean, reference_mean, measurement_to_atom, n_atoms):
    treated, control = np.asarray(treated_mean, np.float64), np.asarray(reference_mean, np.float64)
    ids = np.asarray(measurement_to_atom)
    require(treated.ndim == 2 and control.shape == treated.shape and np.isfinite(treated).all() and np.isfinite(control).all(), 'Invalid measurement means')
    require(np.all(treated >= 0) and np.all(control >= 0), 'Provided-scale measurement means must be nonnegative')
    require(ids.dtype.kind in 'iu' and ids.shape == (len(treated),) and np.all((ids >= 0) & (ids < n_atoms)), 'Invalid atom membership')
    n = np.bincount(ids, minlength=n_atoms)
    require(np.all(n > 0), 'Atom has no admitted measurement')
    measurement_effect = treated - control
    atom_effect = np.zeros((n_atoms, treated.shape[1]), np.float64)
    atom_treated, atom_reference = np.zeros_like(atom_effect), np.zeros_like(atom_effect)
    np.add.at(atom_effect, ids, measurement_effect)
    np.add.at(atom_treated, ids, treated)
    np.add.at(atom_reference, ids, control)
    atom_effect /= n[:, None]
    atom_treated /= n[:, None]
    atom_reference /= n[:, None]
    return measurement_effect, atom_effect, atom_treated, atom_reference, n


def equal_pool_context_states(pool_means, pool_context_ids, context_ids):
    """Each physical pool contributes one mean, independently of cell count."""
    means = np.asarray(pool_means, np.float64)
    pool_context_ids, context_ids = np.asarray(pool_context_ids), np.asarray(context_ids)
    require(means.ndim == 2 and len(means) == len(pool_context_ids) and np.isfinite(means).all(), 'Invalid pool state means')
    require(len(np.unique(context_ids)) == len(context_ids) and set(pool_context_ids) == set(context_ids), 'Pool/context state coverage differs')
    states, counts = [], []
    for context in context_ids:
        selected = means[pool_context_ids == context]
        require(len(selected) > 0, 'Empty context state')
        states.append(selected.mean(axis=0)); counts.append(len(selected))
    return np.stack(states), np.asarray(counts, np.int64)


def validate_gate(config):
    contract_path, metadata, independent = map(Path, [config['scientific_contract'], config['metadata_root'], config['independent_metadata_audit']])
    require(sha256(contract_path) == config['scientific_contract_sha256'], 'Scientific contract hash changed')
    require(sha256(independent) == config['independent_metadata_audit_sha256'], 'Independent metadata audit hash changed')
    gate = read_json(independent)
    require(gate['status'] == 'PASS_INDEPENDENT_METADATA_ROLES_AND_FOLDS', 'Independent metadata PASS required')
    require(gate['frozen_contract_sha256'] == config['scientific_contract_sha256'], 'Independent metadata contract binding mismatch')
    require(gate['producer_output_manifest_sha256'] == sha256(metadata / 'output_manifest.json'), 'Independent metadata output binding mismatch')
    formal = read_json(metadata / 'audit.json')
    require(formal['status'] == 'PASS_FORMAL_METADATA_AND_CONTROL_ROLES_ONLY', 'Formal metadata PASS required')
    for item in read_json(metadata / 'output_manifest.json')['files']:
        require(sha256(metadata / item['path']) == item['sha256'], 'Metadata artifact changed: ' + item['path'])
    return read_json(contract_path), metadata, formal


def run(config_path):
    started = time.perf_counter()
    config_path = Path(config_path).resolve()
    config = read_json(config_path)
    output = Path(config['output_root'])
    require(not output.exists(), 'Refuse existing effects output, including partial runs')
    contract, metadata, formal = validate_gate(config)
    require(contract['feature_selection']['n_genes'] == 3000 and contract['feature_selection']['candidate_axis'].startswith('all21255'), 'Unexpected scientific feature contract')
    require(config['exclusion_mask_applied'] is False and config['normalization_applied'] is False and config['log_or_inverse_log_applied'] is False, 'Prohibited expression transform')
    atoms, measurements = tsv(metadata / 'atomic_index.tsv'), tsv(metadata / 'measurement_index.tsv')
    pools_all, contexts_all, tasks = tsv(metadata / 'control_pool_index.tsv'), tsv(metadata / 'context_index.tsv'), tsv(metadata / 'task_index.tsv')
    cells = tsv(metadata / 'cell_index.tsv.gz', usecols=['obs_id', 'raw_obs_position_zero_based', 'measurement_row', 'atomic_row', 'measurement_id', 'treated_admitted', 'cell_role', 'context_id', 'control_pool_id'])
    controls_all = tsv(metadata / 'control_cell_index.tsv.gz')
    pools = pools_all.loc[boolean(pools_all.included_in_state)].reset_index(drop=True).copy()
    contexts = contexts_all.loc[boolean(contexts_all.included_in_tasks)].reset_index(drop=True).copy()
    controls = controls_all.loc[boolean(controls_all.included_in_state)].reset_index(drop=True).copy()
    treated = cells.loc[boolean(cells.treated_admitted)].reset_index(drop=True).copy()
    sizes = config['expected_sizes']
    require((len(pools), len(contexts), len(tasks), len(atoms), len(measurements), len(treated)) ==
            (sizes['control_pools'], sizes['contexts'], sizes['tasks'], sizes['atoms'], sizes['measurements'], sizes['treated_cells']), 'Frozen metadata sizes changed')
    require(np.array_equal(atoms.atomic_row, np.arange(len(atoms))) and atoms.atomic_id.is_unique, 'Immutable atom axis invalid')
    require(np.array_equal(measurements.measurement_row, np.arange(len(measurements))) and measurements.measurement_id.is_unique, 'Immutable measurement axis invalid')
    require(set(controls.control_role) == {'A', 'B'} and controls.control_role.eq('A').sum() == sizes['control_A'] and controls.control_role.eq('B').sum() == sizes['control_B'], 'Wrong selected-control role counts')
    require(not set(controls.raw_obs_position_zero_based) & set(treated.raw_obs_position_zero_based), 'Treated/control overlap')
    pools['control_pool_row'] = np.arange(len(pools), dtype=np.int64)
    contexts['context_row'] = np.arange(len(contexts), dtype=np.int64)
    pool_lookup = pd.Series(pools.control_pool_row.to_numpy(), index=pools.control_pool_id)
    controls['control_pool_row'] = controls.control_pool_id.map(pool_lookup)
    require(controls.control_pool_row.notna().all(), 'Control outside selected 96 pools')
    controls['control_group_row'] = 2 * controls.control_pool_row + controls.control_role.eq('B').astype(int)
    measurement_pool_rows = measurements.control_pool_id.map(pool_lookup).to_numpy(np.int64)
    require(np.array_equal(measurements.atomic_id, atoms.atomic_id.to_numpy()[measurements.atomic_row]), 'Measurement/atomic IDs differ')
    require(np.array_equal(treated.measurement_id, measurements.measurement_id.to_numpy()[treated.measurement_row]), 'Cell/measurement IDs differ')
    source = Path(contract['raw_h5ad'])
    before_stat = file_stat(source)
    require(before_stat == formal['source_input_stat_after'][str(source)], 'Raw H5AD changed since metadata audit')
    print('HASH_SOURCE', source, flush=True)
    raw_sha = sha256(source)
    require(file_stat(source) == before_stat, 'Raw source changed during full SHA256')
    input_paths = [config_path, Path(config['scientific_contract']), Path(config['independent_metadata_audit']), metadata / 'output_manifest.json', Path(__file__), Path(contract['excluded_pairs_csv'])]
    input_paths += [metadata / item['path'] for item in read_json(metadata / 'output_manifest.json')['files']]
    input_paths = list(dict.fromkeys(input_paths))
    inputs = [{'path': str(p), 'sha256': sha256(p), 'size_bytes': p.stat().st_size} for p in input_paths]
    inputs.append(dict(path=str(source), sha256=raw_sha, size_bytes=source.stat().st_size, complete_raw_file_hash=True))
    output.mkdir(parents=True)
    try:
        write_json(output / 'RUN_STARTED.json', dict(started_utc=datetime.now(timezone.utc).isoformat(), config=config, scientific_contract=contract))
        write_json(output / 'input_manifest.json', inputs)
        for name in ['atomic_index.tsv', 'measurement_index.tsv', 'context_index.tsv', 'task_index.tsv', 'fold_atomic_membership.tsv', 'same_drug_source_pairs.tsv', 'source_well_dependencies.tsv', 'gene_endpoint_support.tsv', 'measurement_exclusions.tsv']:
            shutil.copyfile(metadata / name, output / name)
        write_tsv(output / 'selected_control_pool_index.tsv', pools)
        write_tsv(output / 'selected_context_index.tsv', contexts)
        with h5py.File(source, 'r') as h5:
            raw_obs = h5['obs']['obs_id'].asstr()[:]
            ordered_cells = cells.sort_values('raw_obs_position_zero_based')
            require(np.array_equal(ordered_cells.raw_obs_position_zero_based, np.arange(config['expected_shape'][0])) and np.array_equal(raw_obs, ordered_cells.obs_id), 'Raw obs/cell-index order mismatch')
            var = h5['var']; index_key = var.attrs['_index']
            if isinstance(index_key, bytes): index_key = index_key.decode()
            genes = var[index_key].asstr()[:]; symbols = var['gene_symbol'].asstr()[:]
            require(len(genes) == 21255 and len(set(genes)) == len(genes) and len(symbols) == len(genes), 'Provided source gene axis invalid')
            gene_axis = pd.DataFrame(dict(source_feature_row=np.arange(len(genes)), source_gene_id=genes, gene_symbol=symbols))
            write_tsv(output / 'source_gene_axis.tsv', gene_axis)
            pointers, structural = validate_csr_layout(h5['X'], config['expected_shape'], config['structure_rows_per_chunk'])
            require(structural['nnz'] == config['expected_nnz'], 'Frozen source nnz changed')
            write_json(output / 'source_sparse_structure.json', structural)
            print('CONTROL_PASS', len(controls), 'selected cells; provided scale unchanged', flush=True)
            control_sum, control_squares, control_n, control_qc, control_read = aggregate_selected(h5['X'], pointers, controls, 'control_group_row', 2 * len(pools), len(genes), config['selected_cells_per_chunk'], squares=True)
            n_A, n_B = control_n[::2], control_n[1::2]
            require(np.array_equal(n_A, pools.n_A) and np.array_equal(n_B, pools.n_B), 'Actual control group counts differ from frozen membership')
            sum_A, sum_B = control_sum[::2], control_sum[1::2]
            squared_sum_A, squared_sum_B = control_squares[::2], control_squares[1::2]
            mean_A, mean_B = sum_A / n_A[:, None], sum_B / n_B[:, None]
            mean_all = (sum_A + sum_B) / (n_A + n_B)[:, None]
            save_npz(output / 'control_pool_statistics.npz', control_pool_id=pools.control_pool_id.to_numpy(dtype=str), source_feature_row=gene_axis.source_feature_row.to_numpy(np.int64),
                n_A=n_A, n_B=n_B, sum_A=sum_A, sum_B=sum_B, squared_sum_A=squared_sum_A, squared_sum_B=squared_sum_B, mean_A=mean_A, mean_B=mean_B, mean_all_control=mean_all)
            write_tsv(output / 'control_cell_qc.tsv.gz', control_qc)
            for context in contexts.itertuples():
                take = np.flatnonzero(pools.context_id.eq(context.context_id))
                require(len(take) == context.n_state_vehicle_pools and np.allclose(pools.iloc[take].state_weight, 1 / len(take)), 'Invalid equal-pool context state weights')
            state_A, state_pool_counts = equal_pool_context_states(mean_A, pools.context_id.to_numpy(), contexts.context_id.to_numpy())
            state_B, state_B_pool_counts = equal_pool_context_states(mean_B, pools.context_id.to_numpy(), contexts.context_id.to_numpy())
            require(np.array_equal(state_pool_counts, state_B_pool_counts) and np.array_equal(state_pool_counts, contexts.n_state_vehicle_pools), 'State pool count mismatch')
            save_npz(output / 'context_control_states.npz', context_id=contexts.context_id.to_numpy(dtype=str), source_feature_row=gene_axis.source_feature_row.to_numpy(np.int64),
                n_pools=contexts.n_state_vehicle_pools.to_numpy(np.int64), state_A_full=state_A, state_B_full=state_B)
            panel_parts, feature_records, feature_totals, feature_squares, feature_variances, feature_counts = [], [], [], [], [], []
            for task in tasks.itertuples():
                source_contexts = task.source_context_ids.split('|')
                take = np.flatnonzero(pools.context_id.isin(source_contexts))
                require(len(set(pools.iloc[take].donor_id)) == 2 and task.heldout_donor not in set(pools.iloc[take].donor_id) and set(pools.iloc[take].cell_type) == {task.cell_type}, 'Feature source contexts violate held-out donor/type')
                count = int(n_A[take].sum())
                require(count == task.n_source_A_cells_for_feature_selection, 'Frozen source A count mismatch')
                total, squared = sum_A[take].sum(axis=0), squared_sum_A[take].sum(axis=0)
                selected, variance, diag = variance_and_selection(total, squared, count, gene_axis.source_feature_row.to_numpy(), 3000)
                frame = gene_axis.iloc[selected].copy()
                frame.insert(0, 'task_id', task.task_id); frame.insert(1, 'rank', np.arange(1, 3001))
                frame['cell_type'], frame['heldout_donor'] = task.cell_type, task.heldout_donor
                frame['source_control_variance'], frame['n_source_A_cells'] = variance[selected], count
                panel_parts.append(frame)
                feature_records.append(dict(task_id=task.task_id, cell_type=task.cell_type, heldout_donor=task.heldout_donor, source_context_ids=source_contexts,
                    source_control_pool_ids=pools.iloc[take].control_pool_id.tolist(), n_source_A_cells=count, selected_features=3000, **diag))
                feature_totals.append(total); feature_squares.append(squared); feature_variances.append(variance); feature_counts.append(count)
            panels = pd.concat(panel_parts, ignore_index=True)
            union_rows = np.sort(panels.source_feature_row.unique()).astype(np.int64)
            panels['union_column'] = pd.Index(union_rows).get_indexer(panels.source_feature_row)
            require(len(panels) == 18000 and panels.union_column.ge(0).all(), 'Feature panel union mapping failed')
            write_tsv(output / 'fold_gene_panels.tsv', panels)
            save_npz(output / 'fold_feature_statistics.npz', task_id=tasks.task_id.to_numpy(dtype=str), source_feature_row=gene_axis.source_feature_row.to_numpy(np.int64),
                n_source_A=np.asarray(feature_counts, np.int64), sum_A=np.stack(feature_totals), squared_sum_A=np.stack(feature_squares), variance=np.stack(feature_variances))
            write_json(output / 'feature_selection_sealed.json', dict(stage='COMPLETED_BEFORE_ANY_ADMITTED_TREATED_VALUE_SLICE_REQUEST', created_utc=datetime.now(timezone.utc).isoformat(),
                tasks=feature_records, feature_selection_uses_target_A=False, feature_selection_uses_any_B=False, feature_selection_uses_any_treated=False,
                selected_gene_union=len(union_rows), fold_gene_panels_sha256=sha256(output / 'fold_gene_panels.tsv'), fold_feature_statistics_sha256=sha256(output / 'fold_feature_statistics.npz'),
                full_raw_sha256_previously_read_as_uninterpreted_bytes=True, physical_HDF5_chunk_overread_possible=True, variance_roundoff=config['variance_roundoff']))
            print('FEATURES_SEALED', len(union_rows), 'union genes across six source-A panels', flush=True)
            treated_pass_started_utc = datetime.now(timezone.utc).isoformat()
            feature_seal_sha256 = sha256(output / 'feature_selection_sealed.json')
            write_json(output / 'treated_pass_started.json', dict(started_utc=treated_pass_started_utc,
                feature_selection_sealed_sha256=feature_seal_sha256, fold_gene_panels_sha256=sha256(output / 'fold_gene_panels.tsv'),
                admitted_treated_value_slices_requested_before_this_event=False, n_admitted_treated_cells=len(treated)))
            print('TREATED_PASS', len(treated), 'selected cells; all source genes checked before union reduction', flush=True)
            treatment_sum, _, treatment_n, treatment_qc, treatment_read = aggregate_selected(h5['X'], pointers, treated, 'measurement_row', len(measurements), len(genes), config['selected_cells_per_chunk'], columns=union_rows)
            require(np.array_equal(treatment_n, measurements.n_treated_cells), 'Actual measurement cell counts differ from metadata')
            treatment_mean = treatment_sum / treatment_n[:, None]
            reference_mean = mean_B[np.ix_(measurement_pool_rows, union_rows)]
            measurement_effect, atom_effect, atom_treated, atom_reference, atom_n = equal_measurement_effects(treatment_mean, reference_mean, measurements.atomic_row.to_numpy(np.int64), len(atoms))
            require(np.array_equal(atom_n, atoms.n_admitted_measurements), 'Atomic measurement counts changed')
            require(np.allclose(measurements.atomic_measurement_weight.to_numpy(), 1 / atom_n[measurements.atomic_row.to_numpy()]), 'Metadata equal-measurement weights differ')
            save_npz(output / 'measurement_means.npz', measurement_id=measurements.measurement_id.to_numpy(dtype=str), atomic_row=measurements.atomic_row.to_numpy(np.int64),
                control_pool_row=measurement_pool_rows, source_feature_row=union_rows, n_cells=treatment_n,
                treated_sum=treatment_sum, treated_mean=treatment_mean, control_B_mean=reference_mean, measurement_effect=measurement_effect)
            save_npz(output / 'atomic_effects_float64.npz', atomic_id=atoms.atomic_id.to_numpy(dtype=str), source_feature_row=union_rows,
                effect_provided_scale=atom_effect, treated_mean_equal_measurement=atom_treated, control_B_mean_equal_measurement=atom_reference)
            export_effect = atom_effect.astype(np.float32)
            require(np.isfinite(export_effect).all(), 'Float32 effect export is nonfinite')
            save_npz(output / 'arrays.npz', atomic_id=atoms.atomic_id.to_numpy(dtype=str), source_feature_row=union_rows,
                context_id=contexts.context_id.to_numpy(dtype=str), effect_provided_scale=export_effect,
                state_A=state_A[:, union_rows], state_B=state_B[:, union_rows])
            write_tsv(output / 'treated_cell_qc.tsv.gz', treatment_qc)
            shared = measurements[['measurement_row', 'measurement_id', 'atomic_row', 'atomic_id', 'context_id', 'control_pool_id', 'atomic_measurement_weight']].copy()
            shared['control_pool_row'] = measurement_pool_rows
            shared['reference_role'] = 'B_SHARED_MEAN_OF_ONE_PHYSICAL_VEHICLE_WELL_CELL_SUBSET'
            write_tsv(output / 'measurement_control_links.tsv', shared)
        after_stat = file_stat(source)
        require(after_stat == before_stat, 'Raw source stat changed during aggregation')
        for item in inputs:
            if item['path'] != str(source):
                require(sha256(item['path']) == item['sha256'], 'Input changed during effect construction: ' + item['path'])
        audit = dict(status='PASS', stage='KAGGLE_ATOMIC_EFFECTS_NOT_PREDICTION_RESULT', created_utc=datetime.now(timezone.utc).isoformat(),
            config=config, scientific_contract_sha256=config['scientific_contract_sha256'], independent_metadata_gate_sha256=config['independent_metadata_audit_sha256'],
            raw_sha256=raw_sha, raw_sha256_policy=config['raw_hash'], raw_stat_before=before_stat, raw_stat_after=after_stat, raw_stat_unchanged=True,
            sizes={**sizes, 'source_genes': len(genes), 'union_genes': len(union_rows), 'global_partitioned_pools_not_all_analyzed': 199},
            controls=control_read, treated=treatment_read, sparse_structure=structural,
            numerical_units='Direct difference of means of the unfiltered source-provided log-normalized X; not logFC or raw counts',
            exclusion_mask_applied=False, no_log_exp_or_renormalization=True, float64_aggregation_caches_retained=True,
            effect_export_dtype=str(export_effect.dtype), state_export_dtype=str(state_A.dtype),
            effect_float32_max_abs_error=float(np.max(np.abs(atom_effect - export_effect))), effect_min=float(atom_effect.min()), effect_max=float(atom_effect.max()),
            control_pool_order='Selected included_in_state rows in original metadata order, control_pool_row0..95',
            context_order='Selected included_in_tasks rows in original metadata order, context_row0..5',
            source_gene_identity='Unique provided var index; gene_symbol retained, no Ensembl identity inferred',
            A_B_independent_wells=False, B_state_is_diagnostic_reference_not_prediction_input=True,
            feature_selection_sealed_before_treated_values=True, metadata_indexes_copied_byte_for_byte=True,
            feature_selection_sealed_sha256=feature_seal_sha256, treated_pass_started_utc=treated_pass_started_utc,
            old_metadata_stage_flags_preserved='Task-index flags describe their original metadata-only stage; feature_selection_sealed.json is the effect-stage authority',
            excluded_pairs_csv_hashed_for_provenance_only_not_applied=True,
            metadata_and_effect_exclusions='No additional numerical/QC cell exclusions; fail on any invalid selected cell',
            future_scientific_scope='Two fixed source-annotated cell types; within-type donor plus plate/library joint shift; no cross-study MAE unit pooling',
            seconds=time.perf_counter() - started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            versions={name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scipy', 'h5py']},
            python=sys.version, executable=sys.executable,
            output_sha256={p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()})
        write_json(output / 'audit.json', audit)
        write_json(output / 'output_manifest.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
            files=[dict(path=p.name, size_bytes=p.stat().st_size, sha256=sha256(p)) for p in sorted(output.iterdir()) if p.is_file()]))
        print('KAGGLE_EFFECTS_PASS', json.dumps(audit['sizes']), 'seconds', audit['seconds'], flush=True)
    except Exception as exc:
        write_json(output / 'failure.json', dict(status='FAILED', error=repr(exc), failed_utc=datetime.now(timezone.utc).isoformat(), partial_outputs_not_complete=True))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/kaggle_effects_v1.json')
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        run(args.config)
