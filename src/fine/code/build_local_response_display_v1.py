#!/usr/bin/env python3
"""Export a source-selected, explicitly illustrative JQ1 response display.

No fitting, target-based gene selection, or changes to frozen predictions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def js(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def table(path):
    return pd.read_csv(path, sep='\t', keep_default_na=False,
                       na_values=['NA'], float_precision='round_trip')


def write_table(path, frame):
    frame.to_csv(path, sep='\t', index=False, na_rep='NA', float_format='%.17g', mode='x')


def select_genes(source_effects, source_feature_rows, count):
    """Only source responses enter ranking; deterministic ties use raw feature row."""
    x = np.asarray(source_effects, dtype=np.float64)
    rows = np.asarray(source_feature_rows)
    if x.ndim != 2 or x.shape[1] != len(rows) or not np.isfinite(x).all():
        raise ValueError('invalid source effects or feature axis')
    if len(set(rows.tolist())) != len(rows) or not 0 < count <= len(rows):
        raise ValueError('duplicate feature rows or invalid count')
    score = np.mean(np.abs(x), axis=0)
    order = np.lexsort((rows, -score))
    return order[:count], order, score


def run(config_path):
    start = time.perf_counter()
    cfg = json.loads(config_path.read_text())
    out = ROOT / cfg['output_root']
    out.mkdir(parents=True, exist_ok=False)
    js(out / 'frozen_contract.json', cfg)
    prediction, effects = ROOT / cfg['prediction_root'], ROOT / cfg['effects_root']
    qa_path = ROOT / cfg['prediction_independent_qa']
    qa = json.loads(qa_path.read_text())
    prod = json.loads((prediction / 'audit.json').read_text())
    effect_audit = json.loads((effects / 'audit.json').read_text())
    assert qa['status'] == prod['status'] == effect_audit['status'] == 'PASS'
    assert sha(prediction / 'audit.json') == qa['production_audit_sha256']
    inputs = [config_path, Path(__file__), qa_path, prediction / 'audit.json', effects / 'audit.json']
    for name in ['arrays.npz', 'atomic_index.tsv', 'measurement_index.tsv', 'measurement_means.npz']:
        p = effects / name
        assert sha(p) == effect_audit['output_sha256'][name]
        inputs.append(p)
    for fold in cfg['folds']:
        for name in ['frozen_predictions.npz', 'evaluation_truth.npz', 'source_atoms.tsv',
                     'gene_panel.tsv', 'primary_condition_metrics.tsv']:
            inputs.append(prediction / fold / name)
        p = prediction / fold / 'frozen_predictions.npz'
        assert sha(p) == qa['frozen_prediction_sha256_unchanged'][str(p)]
    hashes = {str(p): sha(p) for p in inputs}
    js(out / 'input_manifest.json', hashes)
    meta = table(effects / 'atomic_index.tsv')
    assert len(meta) == 2256 and meta.atomic_id.is_unique
    eligible = meta.main_eligible.astype(str).str.lower().eq('true')
    entity = meta.source_entity_key.eq(cfg['entity_key_exact'])
    assert meta.loc[entity, 'source_entity_name'].eq(cfg['entity_name_exact']).all()
    assert entity.sum() == 12 and eligible[entity].all()
    assert meta.loc[entity, 'dose_unit'].eq('nM').all()
    assert meta.loc[entity, 'time'].eq(cfg['time_hours']).all()
    assert sorted(meta.loc[entity, 'dose_value'].unique().tolist()) == cfg['dose_nM']
    for fold in cfg['folds']:
        assert sorted(meta.loc[entity & meta.cell_line.eq(fold), 'dose_value'].tolist()) == cfg['dose_nM']
    with np.load(effects / 'arrays.npz', allow_pickle=False) as data:
        np.testing.assert_array_equal(data['atomic_id'], meta.atomic_id)
        response = data['effect_log2fc']
        feature_rows = data['source_feature_row']

    # Select and seal every fold's genes before opening any prediction/target display values.
    selections = {}
    score_frames = []
    source_membership = []
    for fold in cfg['folds']:
        panel = table(prediction / fold / 'gene_panel.tsv')
        assert len(panel) == 3000 and panel.source_feature_row.is_unique
        col = pd.Index(feature_rows).get_indexer(panel.source_feature_row)
        assert (col >= 0).all()
        train = np.flatnonzero((eligible & meta.cell_line.ne(fold)).to_numpy())
        np.testing.assert_array_equal(table(prediction / fold / 'source_atoms.tsv').atomic_id,
                                      meta.iloc[train].atomic_id)
        source_rows = np.flatnonzero((eligible & entity & meta.cell_line.ne(fold)).to_numpy())
        assert len(source_rows) == 8 and set(source_rows).issubset(set(train))
        source_values = response[np.ix_(source_rows, col)]
        selected, order, score = select_genes(source_values, panel.source_feature_row, cfg['genes_per_fold'])
        ranking = panel.iloc[order].copy()
        ranking['selection_rank'] = np.arange(1, 3001)
        ranking['source_mean_absolute_JQ1_effect'] = score[order]
        ranking['selected_for_display'] = ranking.selection_rank.le(cfg['genes_per_fold'])
        ranking['n_source_JQ1_atoms'] = 8
        ranking['n_source_contexts'] = 2
        score_frames.append(ranking)
        src = meta.iloc[source_rows].copy()
        src.insert(0, 'heldout_cell_line', fold)
        source_membership.append(src)
        selections[fold] = (panel, col, selected, source_rows)
    write_table(out / 'gene_selection_scores.tsv', pd.concat(score_frames, ignore_index=True))
    write_table(out / 'gene_selection_source_atoms.tsv', pd.concat(source_membership, ignore_index=True))
    js(out / 'selection_sealed.json', {
        'utc': datetime.now(timezone.utc).isoformat(),
        'config_sha256': sha(config_path),
        'ranking_sha256': sha(out / 'gene_selection_scores.tsv'),
        'source_membership_sha256': sha(out / 'gene_selection_source_atoms.tsv'),
        'scope': 'source-only logical selection; arrays.npz physically contains all response rows',
        'target_or_prediction_values_used_for_selection': False,
        'retrospective_illustration': True
    })

    measurements = table(effects / 'measurement_index.tsv')
    with np.load(effects / 'measurement_means.npz', allow_pickle=False) as data:
        np.testing.assert_array_equal(data['measurement_id'], measurements.measurement_id)
        np.testing.assert_array_equal(data['source_feature_row'], feature_rows)
        cache_dtype = {k: str(data[k].dtype) for k in ['normalized_mean', 'control_B_mean']}
        treated, controls = data['normalized_mean'], data['control_B_mean']
    values, replicates, source_values, whole_scores = [], [], [], []
    methods = prod['method_order']
    assert len(methods) == 29 and len(set(methods)) == 29
    for fold, (panel, col, selected, source_rows) in selections.items():
        query_rows = np.flatnonzero((eligible & entity & meta.cell_line.eq(fold)).to_numpy())
        query_rows = query_rows[np.argsort(meta.iloc[query_rows].dose_value.to_numpy())]
        query = meta.iloc[query_rows]
        selected_panel = panel.iloc[selected]
        selected_col = col[selected]
        with np.load(prediction / fold / 'frozen_predictions.npz', allow_pickle=False) as p:
            assert p['method'].tolist() == methods
            np.testing.assert_array_equal(p['source_feature_row'], panel.source_feature_row)
            p_rows = pd.Index(p['atomic_id']).get_indexer(query.atomic_id)
            assert (p_rows >= 0).all()
            pred = p['predictions'][:, p_rows][:, :, selected]
        truth = response[np.ix_(query_rows, selected_col)]
        with np.load(prediction / fold / 'evaluation_truth.npz', allow_pickle=False) as t:
            t_rows = pd.Index(t['atomic_id']).get_indexer(query.atomic_id)
            assert (t_rows >= 0).all()
            np.testing.assert_array_equal(t['source_feature_row'], panel.source_feature_row)
            np.testing.assert_array_equal(t['truth'][t_rows][:, selected], truth)
        metric = table(prediction / fold / 'primary_condition_metrics.tsv')
        metric = metric[metric.atomic_id.isin(query.atomic_id)].copy()
        assert len(metric) == 4 * 29
        metric.insert(0, 'display_heldout_cell_line', fold)
        metric['score_gene_scope'] = 'all_original_3000_genes_not_display_subset'
        whole_scores.append(metric)
        for qi, atom in enumerate(query.itertuples()):
            for gi, gene in enumerate(selected_panel.itertuples()):
                common = dict(heldout_cell_line=fold, atomic_id=atom.atomic_id,
                              source_entity_name=atom.source_entity_name, dose_nM=int(atom.dose_value),
                              time_hours=int(atom.time), selection_rank=gi + 1,
                              source_feature_row=int(gene.source_feature_row),
                              original_ensembl_id=gene.original_ensembl_id, gene_symbol=gene.gene_symbol,
                              n_treated_cells=int(atom.n_treated_cells))
                for mi, method in enumerate(methods):
                    values.append(dict(common, method=method, observed_log2_effect=float(truth[qi, gi]),
                                       predicted_log2_effect=float(pred[mi, qi, gi])))
                mrows = np.flatnonzero(measurements.atomic_id.eq(atom.atomic_id).to_numpy())
                assert len(mrows) == 2 and set(measurements.iloc[mrows].replicate) == {'rep1', 'rep2'}
                for mr in mrows:
                    measurement = measurements.iloc[mr]
                    fc = np.log2((float(treated[mr, selected_col[gi]]) + 1) /
                                 (float(controls[mr, selected_col[gi]]) + 1))
                    replicates.append(dict(common, measurement_id=measurement.measurement_id,
                                           replicate=measurement.replicate, replicate_log2_effect=float(fc)))
        for sr in source_rows:
            atom = meta.iloc[sr]
            for gi, gene in enumerate(selected_panel.itertuples()):
                source_values.append(dict(heldout_cell_line=fold, source_cell_line=atom.cell_line,
                    atomic_id=atom.atomic_id, dose_nM=int(atom.dose_value), selection_rank=gi + 1,
                    source_feature_row=int(gene.source_feature_row), gene_symbol=gene.gene_symbol,
                    source_log2_effect=float(response[sr, selected_col[gi]])))
    frames = {'target_predictions.tsv': pd.DataFrame(values), 'target_replicates.tsv': pd.DataFrame(replicates),
              'source_selected_gene_responses.tsv': pd.DataFrame(source_values),
              'original_whole_gene_condition_scores.tsv': pd.concat(whole_scores, ignore_index=True)}
    assert [len(x) for x in frames.values()] == [4176, 288, 288, 348]
    for name, frame in frames.items():
        write_table(out / name, frame)
    assert all(sha(Path(p)) == h for p, h in hashes.items())
    js(out / 'audit.json', {
        'status': 'PASS', 'scope': 'illustrative source-data export; independent QA pending',
        'completed_utc': datetime.now(timezone.utc).isoformat(), 'seconds': time.perf_counter() - start,
        'n_independent_studies': 1, 'n_target_contexts': 3, 'n_doses': 4, 'n_unique_target_atoms': 12,
        'n_methods': 29, 'n_display_genes_per_fold': 12, 'n_gene_selection_rows': 9000,
        'n_target_prediction_rows': 4176, 'n_target_replicate_gene_rows': 288,
        'n_source_response_rows': 288, 'n_original_whole_gene_scores': 348,
        'replicate_cache_dtypes': cache_dtype, 'new_fits': 0, 'target_based_selection': False,
        'inputs_unchanged': True,
        'output_sha256': {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}
    })
    print(json.dumps({'status': 'PASS', 'output': str(out), 'seconds': time.perf_counter() - start}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/local_response_display_v1.json')
    run(parser.parse_args().config.resolve())
