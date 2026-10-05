#!/usr/bin/env python3
"""Streaming, source-control-fitted sci-Plex effect reconstruction.

Raw H5AD and metadata are read-only. A and B control wells never overlap.
Feature ranking is completed before treated expression is read. Existing output
directories are refused, including partial runs. No prediction or test selection
is performed here. The full feature trace is preserved in the verified source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from inspect_sciplex_expression_sample import (file_stat, content_stat_equal,
                                               read_h5_column)

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def write_tsv(path, frame):
    with Path(path).open('x', encoding='utf-8') as stream:
        frame.to_csv(stream, sep='\t', index=False)


def normalize_human(matrix, human_rows, total):
    """Validate all selected-cell counts, then normalize by all human features."""
    if matrix.ndim != 2 or not sparse.isspmatrix_csr(matrix):
        raise ValueError('Expected CSR count matrix')
    if not matrix.has_canonical_format:
        raise ValueError('Noncanonical selected CSR rows; do not silently merge duplicate entries')
    values = matrix.data
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('Nonfinite or negative selected-cell count')
    if not np.equal(values, np.floor(values)).all():
        raise ValueError('Selected expression is not integer-compatible counts')
    if not np.isfinite(total) or total <= 0:
        raise ValueError('Invalid library normalization total')
    human = matrix[:, human_rows].astype(np.float64)
    library = np.asarray(human.sum(axis=1)).ravel()
    if not np.isfinite(library).all() or np.any(library <= 0):
        raise ValueError('Zero or invalid human library; no silent cell removal allowed')
    normalized = human.multiply((total / library)[:, None]).tocsr()
    if not np.isfinite(normalized.data).all():
        raise ValueError('Nonfinite normalized counts')
    return normalized, library


def reduce_groups(matrix, group_indices, n_groups):
    """Sum rows by explicit integer group ID without changing their weights."""
    ids = np.asarray(group_indices)
    if ids.ndim != 1 or len(ids) != matrix.shape[0] or ids.dtype.kind not in 'iu':
        raise ValueError('Group IDs must be row-aligned integers')
    if np.any(ids < 0) or np.any(ids >= n_groups):
        raise ValueError('Group ID outside declared axis')
    present, local = np.unique(ids, return_inverse=True)
    reducer = sparse.csr_matrix((np.ones(len(ids)), (local, np.arange(len(ids)))),
                                shape=(len(present), len(ids)))
    return present, (reducer @ matrix).toarray()


def select_features(sum_log, sum_square_log, n_cells, candidate_mask, source_rows, n):
    if n_cells < 2:
        raise ValueError('Insufficient source control cells')
    mean = np.asarray(sum_log, dtype=float) / n_cells
    variance = np.asarray(sum_square_log, dtype=float) / n_cells - mean ** 2
    if not np.isfinite(variance).all() or np.any(variance < -1e-10):
        raise ValueError('Invalid control variance')
    variance = np.maximum(variance, 0)
    candidates = np.flatnonzero(np.asarray(candidate_mask, bool) & (variance > 0))
    if len(candidates) < n:
        raise ValueError('Fewer nonconstant candidate features than requested')
    order = np.lexsort((np.asarray(source_rows)[candidates], -variance[candidates]))
    return candidates[order[:n]], variance


def atomic_effects(treatment_means, reference_means, measurement_to_atom, n_atoms, pseudocount):
    """Combine linear treatment/reference means equally before taking log-ratio."""
    treated, reference = np.asarray(treatment_means), np.asarray(reference_means)
    if treated.shape != reference.shape or treated.ndim != 2:
        raise ValueError('Treatment and matched-reference matrices must match')
    if not np.isfinite(treated).all() or not np.isfinite(reference).all():
        raise ValueError('Nonfinite measurement mean')
    if np.any(treated < 0) or np.any(reference < 0) or not np.isfinite(pseudocount) or pseudocount <= 0:
        raise ValueError('Invalid effect domain')
    ids = np.asarray(measurement_to_atom)
    if ids.ndim != 1 or ids.dtype.kind not in 'iu':
        raise ValueError('Measurement-to-atom IDs must be one-dimensional integers')
    ids = ids.astype(np.int64, copy=False)
    if len(ids) != len(treated) or np.any(ids < 0) or np.any(ids >= n_atoms):
        raise ValueError('Invalid measurement-to-atom mapping')
    count = np.bincount(ids, minlength=n_atoms)
    if np.any(count != 2):
        raise ValueError('This contract requires exactly two measurement wells per atom')
    t = np.zeros((n_atoms, treated.shape[1]), dtype=np.float64)
    c = np.zeros_like(t)
    np.add.at(t, ids, treated)
    np.add.at(c, ids, reference)
    t /= count[:, None]
    c /= count[:, None]
    effect = np.log2((t + pseudocount) / (c + pseudocount))
    return effect, t, c


def cohort_masks(atoms, measurements, minimum=20):
    """Eligibility uses metadata support, never a response or prediction score."""
    result = atoms.copy()
    minima = measurements.groupby('atomic_id').n_treated_cells.min()
    result['min_measurement_cells'] = result.atomic_id.map(minima)
    if result.min_measurement_cells.isna().any():
        raise ValueError('Atom missing measurements')
    group = result.groupby(['source_entity_key', 'dose_value'], sort=False)
    if not group.cell_line.nunique().eq(3).all() or not group.size().eq(3).all():
        raise ValueError('Expected full three-line drug-dose grid before support gates')
    result['main_eligible'] = group.n_treated_cells.transform('min') >= minimum
    result['sensitivity_eligible'] = result.main_eligible & (
        group.min_measurement_cells.transform('min') >= minimum)
    return result


def read_metadata(metadata):
    atoms = pd.read_csv(metadata / 'atomic_conditions.tsv', sep='\t').sort_values('atomic_id').reset_index(drop=True)
    measurements = pd.read_csv(metadata / 'measurements.tsv', sep='\t').sort_values('measurement_id').reset_index(drop=True)
    treated = pd.read_csv(metadata / 'treated_cell_membership.tsv.gz', sep='\t').sort_values('source_row').reset_index(drop=True)
    controls = pd.read_csv(metadata / 'control_cell_membership.tsv.gz', sep='\t').sort_values('source_row').reset_index(drop=True)
    for frame in (treated, controls):
        if frame.source_row.duplicated().any() or frame.source_cell_id.duplicated().any():
            raise ValueError('Duplicate source cell membership')
        if not frame.time.eq(24).all():
            raise ValueError('Non-24h member admitted')
    if np.intersect1d(treated.source_row, controls.source_row).size:
        raise ValueError('Treated/control membership overlap')
    well_fields = ['cell_line', 'time', 'replicate', 'plate', 'well']
    wells = controls.groupby(well_fields, sort=True).size().reset_index(name='n_cells')
    wells = wells.sort_values(['plate', 'well']).reset_index(drop=True)
    if wells.duplicated(['plate', 'well']).any() or not wells.groupby('plate').size().eq(2).all():
        raise ValueError('Each plate must have exactly two distinct vehicle wells')
    wells['control_role'] = np.where(wells.groupby('plate').cumcount().eq(0), 'A', 'B')
    wells['well_id'] = wells.plate + '|' + wells.well
    wells['control_group'] = np.arange(len(wells))
    controls = controls.merge(wells, on=well_fields, how='left', validate='many_to_one', sort=False)
    controls = controls.sort_values('source_row').reset_index(drop=True)
    if controls.control_group.isna().any():
        raise ValueError('Unmatched control well')
    measurements['measurement_group'] = np.arange(len(measurements))
    group_map = measurements.set_index('measurement_id').measurement_group
    treated['measurement_group'] = treated.measurement_id.map(group_map)
    if treated.measurement_group.isna().any():
        raise ValueError('Unmatched treatment measurement')
    observed_counts = treated.groupby('measurement_id').size().reindex(measurements.measurement_id).to_numpy()
    if not np.array_equal(observed_counts, measurements.n_treated_cells):
        raise ValueError('Treatment measurement count mismatch')
    for role in ('A', 'B'):
        match = wells[wells.control_role.eq(role)].set_index('plate')
        measurements[f'control_{role}_group'] = measurements.plate.map(match.control_group)
        for field in ('cell_line', 'time', 'replicate'):
            if not measurements[field].eq(measurements.plate.map(match[field])).all():
                raise ValueError(f'Nonexact control match: {role}/{field}')
    if not measurements.groupby('atomic_id').replicate.nunique().eq(2).all():
        raise ValueError('Expected two distinct replicate labels per atom')
    atom_counts = measurements.groupby('atomic_id').n_treated_cells.sum().reindex(atoms.atomic_id).to_numpy()
    if not np.array_equal(atom_counts, atoms.n_treated_cells):
        raise ValueError('Measurement/atom support count mismatch')
    atoms = cohort_masks(atoms, measurements)
    measurements['atom_group'] = measurements.atomic_id.map(pd.Series(atoms.index, index=atoms.atomic_id))
    return atoms, measurements, treated, controls, wells


def verify_input(source, treated, controls, feature_axis):
    x = source['X']
    shape = tuple(int(v) for v in x.attrs['shape'])
    if x.attrs['encoding-type'] != 'csr_matrix':
        raise ValueError('Raw expression is not CSR')
    if shape[1] != len(feature_axis) or not np.array_equal(feature_axis.source_feature_row, np.arange(shape[1])):
        raise ValueError('Original feature order mismatch')
    if feature_axis.original_ensembl_id.duplicated().any():
        raise ValueError('Original source feature IDs are not unique')
    old_ids = read_h5_column(source['var'], 'ensembl_id')
    if not np.array_equal(old_ids, feature_axis.legacy_truncated_ensembl_id):
        raise ValueError('Verified gene axis no longer matches H5AD')
    obs = source['obs']
    for key in (obs.attrs['_index'], 'plate', 'well', 'cell_line', 'replicate', 'time', 'perturbation'):
        # Some unselected raw cells lack hash/plate assignments. Validate codes
        # only for admitted members; never silently reinterpret missing labels.
        node = obs[key]
        if isinstance(node, h5py.Group):
            codes = node['codes'][:]
            categories = node['categories']
            categories = categories.asstr()[:] if categories.dtype.kind in 'OSU' else categories[:]
            values = None
        else:
            values = node.asstr()[:] if node.dtype.kind in 'OSU' else node[:]
        for role, members in [('treated', treated), ('control', controls)]:
            selected_rows = members.source_row.to_numpy(dtype=int)
            if values is None:
                selected_codes = codes[selected_rows]
                if np.any(selected_codes < 0) or np.any(selected_codes >= len(categories)):
                    raise ValueError(f'Missing selected categorical value: {role}/{key}')
                actual = categories[selected_codes]
            else:
                actual = values[selected_rows]
            if key == obs.attrs['_index']:
                expected = members.source_cell_id.to_numpy()
            elif key == 'perturbation':
                if not np.all((actual == 'control') == (role == 'control')):
                    raise ValueError('Raw treatment/control role mismatch')
                continue
            else:
                expected = members[key].to_numpy()
            if not np.array_equal(actual, expected):
                raise ValueError(f'Raw/member mismatch: {role}/{key}')
    return shape


def aggregate(source, members, groups, human_rows, config, *, subset_human=None, logarithms=False):
    """Read bounded source-row blocks; validate and aggregate selected cells only."""
    x = source['X']
    n_source, _ = map(int, x.attrs['shape'])
    pointers = x['indptr'][:]
    if len(pointers) != n_source + 1 or pointers[0] != 0 or np.any(np.diff(pointers) < 0):
        raise ValueError('Invalid source CSR pointers')
    if pointers[-1] != len(x['data']) or len(x['data']) != len(x['indices']):
        raise ValueError('CSR entries do not match pointers')
    rows = members.source_row.to_numpy(dtype=np.int64)
    if not len(rows) or rows[0] < 0 or rows[-1] >= n_source or np.any(np.diff(rows) <= 0):
        raise ValueError('Source rows must be unique, strictly ascending and in range')
    selected_groups = members[groups].to_numpy(dtype=np.int64)
    n_groups = int(selected_groups.max()) + 1
    n_features = len(human_rows) if subset_human is None else len(subset_human)
    sums = np.zeros((n_groups, n_features), dtype=np.float64)
    sum_log = np.zeros_like(sums) if logarithms else None
    sum_square = np.zeros_like(sums) if logarithms else None
    counts = np.zeros(n_groups, dtype=np.int64)
    libraries = np.empty(len(rows), dtype=np.float64)
    audit = {'n_selected_cells': len(rows), 'n_validated_cells': 0,
             'n_stored_entries_read': 0, 'n_selected_entries_validated': 0,
             'source_row_chunk_size': config['chunk_source_rows'],
             'normalization_uses_all_human_features': True}
    started = time.perf_counter()
    for chunk_number, start in enumerate(range(0, n_source, config['chunk_source_rows'])):
        stop = min(n_source, start + config['chunk_source_rows'])
        left, right = (int(value) for value in np.searchsorted(rows, [start, stop]))
        if left == right:
            continue
        entry_start, entry_stop = int(pointers[start]), int(pointers[stop])
        data = x['data'][entry_start:entry_stop]
        indices = x['indices'][entry_start:entry_stop]
        block = sparse.csr_matrix((data, indices, pointers[start:stop + 1] - entry_start),
                                  shape=(stop - start, int(x.attrs['shape'][1])))
        selected = block[rows[left:right] - start]
        normalized, library = normalize_human(selected, human_rows, config['normalization_total'])
        libraries[left:right] = library
        group_ids = selected_groups[left:right]
        counts += np.bincount(group_ids, minlength=n_groups)
        if subset_human is not None:
            normalized = normalized[:, subset_human].tocsr()
        present, values = reduce_groups(normalized, group_ids, n_groups)
        sums[present] += values
        if logarithms:
            normalized.data = np.log1p(normalized.data)
            present, values = reduce_groups(normalized, group_ids, n_groups)
            sum_log[present] += values
            normalized.data **= 2
            present, values = reduce_groups(normalized, group_ids, n_groups)
            sum_square[present] += values
        audit['n_validated_cells'] += right - left
        audit['n_stored_entries_read'] += len(data)
        audit['n_selected_entries_validated'] += selected.nnz
        if chunk_number % 10 == 0:
            print(json.dumps({'stage': 'control' if logarithms else 'treated',
                              'source_rows_read_through': stop,
                              'selected_cells_done': audit['n_validated_cells'],
                              'seconds': round(time.perf_counter() - started, 1)}), flush=True)
    if audit['n_validated_cells'] != len(rows) or np.any(counts <= 0):
        raise ValueError('Incomplete selected-cell aggregation')
    audit['seconds'] = time.perf_counter() - started
    audit['human_library_quantiles'] = dict(zip(['min', 'q25', 'median', 'q75', 'max'],
                                              np.quantile(libraries, [0, .25, .5, .75, 1]).tolist()))
    return sums / counts[:, None], sum_log, sum_square, counts, libraries, audit


def run(config_path):
    started = time.perf_counter()
    config = json.loads(config_path.read_text())
    output = ROOT / config['output_root']
    if output.exists():
        raise FileExistsError(f'Refusing existing or partial output: {output}')
    metadata, source_root = ROOT / config['metadata_root'], ROOT / config['source_root']
    raw = Path(config['raw_h5ad'])
    before = file_stat(raw)
    small_inputs = [config_path, Path(__file__), ROOT / 'code/inspect_sciplex_expression_sample.py',
                    source_root / 'restored_feature_axis.tsv', source_root / 'source_verification.json',
                    ROOT / 'qa/sciplex_provenance_review.md']
    small_inputs += [metadata / name for name in ['atomic_conditions.tsv', 'measurements.tsv',
                    'treated_cell_membership.tsv.gz', 'control_cell_membership.tsv.gz']]
    hashes = {str(path): digest(path) for path in small_inputs}
    atoms, measurements, treated, controls, wells = read_metadata(metadata)
    sizes = {'atomic_conditions': len(atoms), 'treated_cells': len(treated),
             'control_cells': len(controls), 'measurements': len(measurements),
             'control_wells': len(wells), 'main_atoms': int(atoms.main_eligible.sum()),
             'sensitivity_atoms': int(atoms.sensitivity_eligible.sum())}
    for key, value in sizes.items():
        if value != config['expected_' + key]:
            raise ValueError(f'Predeclared cohort size mismatch {key}: {value}')
    feature = pd.read_csv(source_root / 'restored_feature_axis.tsv', sep='\t')
    human_rows = np.flatnonzero(feature.is_human_identifier.to_numpy(dtype=bool))
    if len(human_rows) != config['expected_human_features']:
        raise ValueError('Human feature count changed')
    candidate = ~feature.iloc[human_rows].is_PAR_Y_feature.to_numpy(dtype=bool)
    output.mkdir(parents=True)
    write_json(output / 'RUNNING.json', {'started_utc': datetime.now(timezone.utc).isoformat(),
               'status': 'INCOMPLETE_UNTIL_AUDIT_PASS', 'input_sha256': hashes,
               'raw_file_stat_before': before, 'config': config})
    write_tsv(output / 'atomic_index.tsv', atoms)
    write_tsv(output / 'measurement_index.tsv', measurements)
    write_tsv(output / 'control_partition.tsv', wells)
    with h5py.File(raw, 'r') as source:
        shape = verify_input(source, treated, controls, feature)
        means, logs, squares, counts, libraries, control_audit = aggregate(
            source, controls, 'control_group', human_rows, config, logarithms=True)
        if not np.array_equal(counts, wells.n_cells):
            raise ValueError('Control cell count changed during read')
        np.savez_compressed(output / 'control_well_statistics.npz',
                            well_id=wells.well_id.to_numpy(dtype=str), source_feature_row=human_rows,
                            normalized_mean=means, log_sum=logs, log_square_sum=squares, n_cells=counts)
        fold_rows, fold_panels = [], {}
        for heldout in config['cell_lines']:
            selector = wells.control_role.eq('A').to_numpy() & ~wells.cell_line.eq(heldout).to_numpy()
            chosen, variance = select_features(logs[selector].sum(axis=0), squares[selector].sum(axis=0),
                int(counts[selector].sum()), candidate, human_rows, config['genes_per_fold'])
            fold_panels[heldout] = human_rows[chosen]
            for rank, idx in enumerate(chosen, 1):
                source_row = int(human_rows[idx])
                fold_rows.append({'heldout_cell_line': heldout, 'rank': rank,
                    'source_feature_row': source_row,
                    'original_ensembl_id': feature.iloc[source_row].original_ensembl_id,
                    'gene_symbol': feature.iloc[source_row].original_gene_symbol,
                    'source_control_variance': float(variance[idx])})
        union = np.unique(np.concatenate(list(fold_panels.values())))
        union_lookup = {int(row): i for i, row in enumerate(union)}
        for row in fold_rows:
            row['union_column'] = union_lookup[row['source_feature_row']]
        write_tsv(output / 'fold_gene_panels.tsv', pd.DataFrame(fold_rows))
        write_json(output / 'feature_selection_sealed.json', {
            'stage': 'COMPLETED_BEFORE_TREATED_EXPRESSION_READ',
            'fold_gene_panels_sha256': digest(output / 'fold_gene_panels.tsv'),
            'source_A_well_ids_per_fold': {heldout: wells.loc[wells.control_role.eq('A') &
                ~wells.cell_line.eq(heldout), 'well_id'].tolist() for heldout in config['cell_lines']},
            'feature_selection_uses_target_control_cells': False,
            'feature_selection_uses_any_treated_expression': False})
        union_human = np.searchsorted(human_rows, union)
        treatment_means, _, _, treatment_counts, treated_libraries, treatment_audit = aggregate(
            source, treated, 'measurement_group', human_rows, config, subset_human=union_human)
    if not np.array_equal(treatment_counts, measurements.n_treated_cells):
        raise ValueError('Treatment cell count changed during read')
    reference = means[measurements.control_B_group.to_numpy(dtype=int)][:, union_human]
    effect, atom_treatment, atom_reference = atomic_effects(treatment_means, reference,
        measurements.atom_group.to_numpy(dtype=int), len(atoms), config['effect_pseudocount'])
    state_A, state_B = [], []
    for line in config['cell_lines']:
        for role, collection in [('A', state_A), ('B', state_B)]:
            select = wells.cell_line.eq(line) & wells.control_role.eq(role)
            collection.append(np.log1p(means[select].mean(axis=0)[union_human]))
    np.savez_compressed(output / 'arrays.npz', atomic_id=atoms.atomic_id.to_numpy(dtype=str),
        source_feature_row=union, effect_log2fc=effect.astype(np.float32),
        treatment_mean=atom_treatment.astype(np.float32), reference_B_mean=atom_reference.astype(np.float32),
        cell_line=np.asarray(config['cell_lines']), control_state_A=np.asarray(state_A, dtype=np.float32),
        control_state_B=np.asarray(state_B, dtype=np.float32))
    np.savez_compressed(output / 'measurement_means.npz',
        measurement_id=measurements.measurement_id.to_numpy(dtype=str), source_feature_row=union,
        normalized_mean=treatment_means.astype(np.float32),
        control_B_mean=reference.astype(np.float32),
        control_A_mean=means[measurements.control_A_group.to_numpy(dtype=int)][:, union_human].astype(np.float32))
    for name, frame, library in [('control', controls, libraries), ('treated', treated, treated_libraries)]:
        qc = frame[['source_row', 'source_cell_id', 'plate', 'well']].copy()
        qc['human_library_counts'] = library.astype(np.int64)
        qc.to_csv(output / f'{name}_selected_cell_qc.tsv.gz', sep='\t', index=False, compression='gzip')
    after = file_stat(raw)
    if not content_stat_equal(before, after):
        raise RuntimeError('Raw input content-related stat changed during run')
    if any(digest(path) != old for path, old in hashes.items()):
        raise RuntimeError('Source/config/code changed during run')
    finite = np.isfinite(effect).all() and np.isfinite(state_A).all() and np.isfinite(state_B).all()
    if not finite:
        raise ValueError('Final effect/state contains nonfinite values')
    output_hashes = {p.name: digest(p) for p in output.iterdir() if p.is_file()}
    audit = {'status': 'PASS', 'stage': 'ATOMIC_EFFECTS_RECONSTRUCTED_NOT_PREDICTION_RESULT',
             'completed_utc': datetime.now(timezone.utc).isoformat(), 'sizes': sizes,
             'raw_shape': shape, 'n_union_gene_features': len(union),
             'genes_per_fold': config['genes_per_fold'], 'control': control_audit, 'treated': treatment_audit,
             'all_selected_cells_count_validated': True, 'raw_content_stat_unchanged': True,
             'raw_stat_after': after, 'input_sha256': hashes, 'output_sha256': output_hashes,
             'effect_min': float(effect.min()), 'effect_max': float(effect.max()),
             'float32_effect_export_max_abs_error': float(np.max(np.abs(effect - effect.astype(np.float32)))),
             'elapsed_seconds': time.perf_counter() - started,
             'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
             'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                          'pandas': pd.__version__, 'h5py': h5py.__version__},
             'limits': ['Well separation does not establish batch independence.',
                       'Main inclusion permits some small individual replicate wells; restricted sensitivity is predefined.',
                       'Three LOCO folds belong to one study and are not independent replication.',
                       'Human PAR_Y contributes to library size but is not a prediction target.',
                       'Raw file content stats were checked, not a complete large-file cryptographic rehash.']}
    write_json(output / 'audit.json', audit)
    print(json.dumps({'status': 'PASS', 'output': str(output), 'sizes': sizes,
                      'n_union_gene_features': len(union), 'seconds': audit['elapsed_seconds'],
                      'peak_rss_mib': audit['peak_rss_mib']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/effect_prediction_v1.json')
    args = parser.parse_args()
    run(args.config.resolve())
