"""Synthetic E08 producer tests; no response archive/model inference is accessed."""
import argparse
import ast
import hashlib
import inspect
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import run_e08_prediction as p


def toy_metadata(n_entities=3):
    rows = []
    for e in range(n_entities):
        for dose in [10, 100, 1000, 10000]:
            for line in ['A549', 'K562', 'MCF7']:
                rows.append(dict(atomic_id=f'a{e:03}_{dose:05}_{line}', source_entity_key=f'e{e}',
                                 dose_value=dose, time=24, cell_line=line))
    return pd.DataFrame(rows).sort_values('atomic_id').reset_index(drop=True)


class TestProducer(unittest.TestCase):
    def config(self):
        return json.loads((p.PACKAGE/'experiments/e08_prediction_v1/config.json').read_text())

    def test_registry_complete_33_17_98(self):
        c = self.config(); m = p.method_registry(c); q = p.contrast_registry(c, m)
        self.assertEqual(m.groupby('cohort').size().to_dict(), {'mechanism57': 33, 'target56': 17})
        self.assertEqual(q.groupby('cohort').size().to_dict(), {'mechanism57': 64, 'target56': 34})
        self.assertEqual(int(m.kind.ne('reference').sum())*3, 126)
        self.assertEqual(q[q.family == 'added_knowledge'].shape[0], 7)

    def test_source_landmarks_retain_entities_and_order(self):
        m = toy_metadata(); rows = np.flatnonzero(m.cell_line.ne('A549'))
        selected = p.landmarks(m, rows, 10, 20260914)
        self.assertEqual(len(set(selected)), 10)
        self.assertTrue(set(selected) <= set(rows)); self.assertEqual(m.iloc[selected].source_entity_key.nunique(), 3)
        np.testing.assert_array_equal(selected, p.landmarks(m, rows[::-1], 10, 20260914))
        with self.assertRaises(ValueError):
            p.landmarks(m, rows, 2, 20260914)

    def test_source_landmark_hash_rule_hand_reconstruction(self):
        m = toy_metadata(); rows = np.flatnonzero(m.cell_line.ne('K562'))
        ordered = sorted(rows, key=lambda i: (hashlib.sha256(f'20260914|{m.iloc[i].atomic_id}'.encode()).hexdigest(), m.iloc[i].atomic_id))
        first, seen = [], set()
        for i in ordered:
            if m.iloc[i].source_entity_key not in seen:
                first.append(i); seen.add(m.iloc[i].source_entity_key)
        expected = first + [i for i in ordered if i not in first][:10-len(first)]
        self.assertEqual(p.landmarks(m, rows, 10).tolist(), expected)

    def test_cosine_exact_duplicate_broadcast(self):
        rng = np.random.default_rng(4); x = rng.normal(size=(12, 9)).astype('float32'); x[5] = x[2]; x[9] = x[2]
        kernel = p.unique_cosine(x, [1, 2, 4, 7])
        self.assertTrue(np.array_equal(kernel[2].view('uint64'), kernel[9].view('uint64')))
        np.testing.assert_allclose(kernel, p.cosine_similarity(x, x[[1, 2, 4, 7]]), atol=2e-15)

    def test_native_full_binary_int64_and_dose(self):
        x = np.array([[1]*300, [1]*200+[0]*100, [0]*100+[1]*200], np.uint8)
        expected = np.array([[1, 2/3], [2/3, 1], [2/3, 1/3]])
        np.testing.assert_allclose(p.binary_jaccard(x, [0, 1]), expected)
        doses = [10, 100, 10]
        np.testing.assert_allclose(p.native_kernel(x, doses, [0, 1]), (expected+np.array([[1,0],[0,1],[1,0]]))/2)
        with self.assertRaises(ValueError):
            p.binary_jaccard(np.zeros((2, 3)), [0])

    def test_fusion_literal_raw_kernel_no_rank(self):
        c = np.array([[-1., .25, 1.], [0, -.5, .9]]); native = np.array([[0, .5, 1.], [.2, .4, .8]])
        expected = (c+1)/4 + native/2
        np.testing.assert_array_equal(p.fusion_kernel(c, native), expected)
        self.assertFalse(np.array_equal(p.fusion_kernel(c, native), p.fusion_kernel(c/2, native)))

    def test_ridge_source_scaling_threshold_and_duplicate_prediction(self):
        f = np.array([[0., 1., 1e-9], [1,1,2e-9], [2,1,3e-9], [3,1,4e-9]])
        y = np.array([[1.,2], [4,3], [3,5], [2,6]])
        query = np.array([[2.,1,3e-9], [0,1,1e-9], [2,1,3e-9]])
        prediction, pars = p.fit_predict(f, y, query)
        self.assertEqual(pars['feature_scale'][1], 1); self.assertEqual(pars['feature_scale'][2], 1)
        np.testing.assert_array_equal(prediction[0].view('uint64'), prediction[2].view('uint64'))
        z = (f-pars['feature_mean'])/pars['feature_scale']
        coef = np.linalg.solve(z.T@z+10*np.eye(3), z.T@(y-y.mean(0)))
        np.testing.assert_allclose(pars['coef'], coef)
        np.testing.assert_allclose(prediction, (query-pars['feature_mean'])/pars['feature_scale']@coef+y.mean(0))

    def test_source_observations_not_deduplicated(self):
        f = np.array([[0.], [0.], [1.], [2.]])
        y = np.array([[0.], [10.], [4.], [5.]])
        got, pars = p.fit_predict(f, y, f)
        self.assertEqual(pars['target_mean'][0], 4.75)
        wrong, _ = p.fit_predict(f[[0,2,3]], y[[0,2,3]], f)
        self.assertFalse(np.allclose(got, wrong))

    def test_affine_boundary_not_assumed_equal(self):
        scale = np.sqrt(1.25)
        x = np.arange(4.)[:,None]*1.5e-8/scale
        y = np.arange(4.)[:,None]
        _, raw = p.fit_predict(x, y, x)
        _, affine = p.fit_predict((x+1)/2, y, (x+1)/2)
        self.assertLess(raw['feature_scale'][0], 1)
        self.assertEqual(affine['feature_scale'][0], 1)

    def test_structured_source_vocabulary_no_context(self):
        m = toy_metadata(); source = np.flatnonzero(m.cell_line.ne('A549'))
        kernel, audit = p.structured_kernel(m, source, source[:8])
        a, b = np.flatnonzero(m.source_entity_key.eq('e0') & m.dose_value.eq(10))[:2]
        np.testing.assert_array_equal(kernel[a], kernel[b])
        changed = m.copy(); changed.loc[a,'source_entity_key'] = 'unseen'
        if a not in source:
            with self.assertRaises(ValueError):
                p.structured_kernel(changed, source, source[:8])
        self.assertNotIn('cell_line', audit)

    def test_tfidf_source_unique_text_only(self):
        texts = np.array(['alpha beta', 'alpha beta', 'gamma alpha', 'queryonly omega'])
        kernel, audit = p.tfidf_kernel(texts, [0,1,2], [0,2], self.config())
        self.assertNotIn('queryonly', audit['vocabulary']); self.assertEqual(audit['n_unique_source_texts'], 2)
        np.testing.assert_array_equal(kernel[0], kernel[1]); np.testing.assert_array_equal(kernel[3], [0,0])

    def test_references_current_cohort_and_exact_two_sources(self):
        m = toy_metadata(2); source = np.flatnonzero(m.cell_line.ne('A549')); query = np.flatnonzero(m.cell_line.eq('A549'))
        y = np.arange(len(source)*2, dtype=float).reshape(-1,2)
        values, pairs = p.references(m, source, query, y)
        np.testing.assert_array_equal(values['source_mean'], np.broadcast_to(y.mean(0), (len(query),2)))
        self.assertEqual(len(pairs), len(query))
        for i, r in enumerate(m.iloc[query].itertuples()):
            rows = np.flatnonzero(m.iloc[source].source_entity_key.eq(r.source_entity_key)&m.iloc[source].dose_value.eq(r.dose_value))
            np.testing.assert_array_equal(values['same_drug_source_mean'][i], y[rows].mean(0))
        with self.assertRaises(ValueError):
            p.references(m, source[1:], query, y[1:])

    def test_no_query_truth_fit_signature_or_state_input(self):
        self.assertEqual(list(inspect.signature(p.fit_predict).parameters), ['source_features','source_y','query_features','alpha'])
        syntax = ast.parse(Path(p.__file__).read_text())
        self.assertFalse(any(isinstance(n,ast.Subscript) and isinstance(n.slice,ast.Constant) and n.slice.value in ['control_state_A','control_state_B'] for n in ast.walk(syntax)))

    def test_na_shared_pair_and_equal_children(self):
        x = p.paired_gain([1,np.nan,3], [2,5,np.nan], True)
        np.testing.assert_allclose(x, [1,np.nan,np.nan], equal_nan=True)
        a = p.summary_row({'g':'a'}, 'rho', [1,np.nan]); b = p.summary_row({'g':'a'}, 'rho', [0,0,0,0])
        result = p.aggregate(pd.DataFrame([a,b]), ['g','metric'], 2).iloc[0]
        self.assertEqual(result.value, .5); self.assertEqual(result.n_units, 6); self.assertEqual(result.n_valid_units, 5)
        self.assertTrue(np.isnan(p.summary_row({},'rho',[np.nan])['value']))

    def test_complete_scoring_synthetic_na_ties_pairs(self):
        m = toy_metadata(20); query = m.loc[m.cell_line.eq('A549')].reset_index(drop=True)
        rng = np.random.default_rng(77); truth = rng.normal(size=(len(query),4)).astype('float32')
        methods = ['zero','same_drug_source_mean','test__C']; pred = np.stack([np.zeros_like(truth), truth*.8, truth])
        genes = pd.DataFrame({'source_feature_row':range(4),'original_ensembl_id':['g'+str(i) for i in range(4)],'gene_symbol':['G'+str(i) for i in range(4)]})
        contrasts = pd.DataFrame([dict(cohort='toy',contrast='C_vs_zero',left='test__C',right='zero',family='toy')])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dose, pair, counts = p.score_fold(root,'toy','A549',methods,contrasts,pred,truth,query,genes,20)
            self.assertEqual(counts, dict(condition_rows=240,gene_rows=48,paired_condition_rows=80,paired_gene_rows=16))
            zero = dose.loc[dose.method.eq('zero')]
            self.assertTrue(zero.loc[zero.metric.eq('gene_spearman'),'value'].isna().all())
            self.assertTrue(zero.loc[zero.metric.eq('gene_order'),'value'].eq(.5).all())
            self.assertTrue(pair.loc[pair.metric.eq('gene_spearman'),'value'].isna().all())
            self.assertTrue(pair.loc[pair.metric.eq('gene_order'),'value'].eq(.5).all())
            self.assertEqual(len(pd.read_csv(root/'gene_metrics.tsv.gz',sep='\t')),48)

    def test_existing_file_refused_and_no_unapproved_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); p.save_npz(root/'x.npz',x=np.arange(3))
            with self.assertRaises(FileExistsError): p.save_npz(root/'x.npz',x=np.arange(4))
            p.write_json(root/'x.json', {'a':1})
            with self.assertRaises(FileExistsError): p.write_json(root/'x.json', {'a':2})
            with self.assertRaisesRegex(ValueError,'explicit --execute'):
                p.run(root/'absent.json', False, None)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists(): raise FileExistsError(args.report)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestProducer)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    p.write_json(args.report, {'status':'PASS' if result.wasSuccessful() else 'FAIL', 'tests_run':result.testsRun,
        'errors':len(result.errors),'failures':len(result.failures),'script_sha256':p.sha(p.__file__),
        'tests_sha256':p.sha(__file__),'response_values_read':False,'created_utc':p.now()})
    if not result.wasSuccessful(): raise SystemExit(1)


if __name__ == '__main__':
    with p.threadpool_limits(limits=1): main()
