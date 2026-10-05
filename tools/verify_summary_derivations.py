#!/usr/bin/env python3
"""Check existing summary tables, not raw data; never overwrite frozen results.

The E08 aggregate is extracted from a hash-verified native producer. The OP3
baseline task/macro tables are checked against their published aggregation and
the recovered producer-provenance record. This does not claim a surviving
historical execution log.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
NATIVE_SHA='9237155f6b85dbbb5fc96ea026e05444f7bb6dea32afc6a6850c568d451553cb'

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--figshare-root',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    a=p.parse_args()
    if a.report.exists(): raise SystemExit('Refusing to overwrite report: '+str(a.report))
    report={'status':'NOT_RUN','scope':'existing summary tables only; no expression/models/predictions opened'}
    try:
        root=a.figshare_root
        src=ROOT/'provenance/native_snapshots/e08/run_e08_prediction.py'
        if sha(src)!=NATIVE_SHA:raise ValueError('Native E08 producer SHA mismatch')
        seal=json.loads((ROOT/'provenance/e08/execution_seal_public.json').read_text())
        if seal['script_sha256']!=NATIVE_SHA:raise ValueError('Execution seal producer mismatch')
        om=json.loads((ROOT/'provenance/e08/output_manifest.json').read_text())
        s=root/'sources/deep/legacy/supp/result3'
        for name in ['paired_dose_summary','paired_fold_summary']:
            path=s/('e08_'+name+'.tsv')
            rec=next(x for x in om['files'] if x['path']==name+'.tsv')
            if sha(path)!=rec['sha256'] or path.stat().st_size!=rec['bytes']:
                raise ValueError('Output manifest mismatch: '+str(path))
        mod=ast.parse(src.read_text());ns={'pd':pd,'np':np}
        fns=[f for f in mod.body if isinstance(f,ast.FunctionDef) and f.name in {'require','aggregate'}]
        if len(fns)!=2:raise ValueError('Expected two isolated native functions')
        exec(compile(ast.Module(body=fns,type_ignores=[]),str(src),'exec'),ns)
        dose=pd.read_csv(s/'e08_paired_dose_summary.tsv',sep='\t',float_precision='round_trip')
        saved=pd.read_csv(s/'e08_paired_fold_summary.tsv',sep='\t',float_precision='round_trip')
        keys=['cohort','heldout_cell_line','contrast','left','right','family','metric']
        got=ns['aggregate'](dose,keys,4)
        if list(saved.columns)!=list(got.columns):raise ValueError('E08 columns mismatch')
        for c in saved.columns:
            if c=='value':np.testing.assert_allclose(got[c],saved[c],rtol=0,atol=2e-15,equal_nan=True)
            else:pd.testing.assert_series_equal(got[c],saved[c],check_names=True)
        report['e08']={'rows':len(got),'columns':len(got.columns),'max_abs_error':float(np.nanmax(np.abs(got.value-saved.value))),'status':'HASH_LINKED_NATIVE_PRODUCER_AND_AGGREGATION_MATCH','source_sha256':sha(s/'e08_paired_fold_summary.tsv')}
        task=pd.read_csv(root/'sources/deep/added/baseline_l2_task_metrics.tsv',sep='\t',float_precision='round_trip')
        saved=pd.read_csv(root/'sources/deep/added/baseline_l2_macro_summary.tsv',sep='\t',float_precision='round_trip').set_index('method')
        if task.duplicated(['method','cell_type','donor_id']).any():raise ValueError('Duplicate baseline task')
        if not task.groupby(['method','cell_type']).size().eq(3).all():raise ValueError('Expected three donors per type')
        if not saved.n_types_nominal.eq(2).all() or not saved.n_studies.eq(1).all():raise ValueError('Unexpected baseline scope')
        if not saved.aggregation.str.strip().eq('equal_donor_within_type_then_equal_type;descriptive_only').all():raise ValueError('Unsupported aggregation definition')
        metrics=['rsa','ndcg_mean','random_ndcg_mean','excess_ndcg_mean']
        typed=task.groupby(['method','cell_type'],sort=False)[metrics].mean()
        macro=typed.groupby('method',sort=False).mean().loc[saved.index]
        count=typed.groupby('method',sort=False).count().loc[saved.index]
        np.testing.assert_allclose(macro[metrics],saved[metrics],rtol=0,atol=2e-15,equal_nan=True)
        for c in metrics:np.testing.assert_array_equal(count[c],saved[c+'__valid_types'])
        pa=ROOT/'provenance/baseline_geometry/BASELINE_PRODUCER_RECOVERY_AUDIT.json'
        prov=json.loads(pa.read_text())
        if prov.get('historical_producer_sha256')!='42d07e6e1e21ad56c9c6b8fd0cac77b8ba2244694ddd0e220e683efd5f41d289':
            raise ValueError('Recovered baseline producer provenance mismatch')
        if prov.get('released_outputs',{}).get('baseline_l2_task_metrics.tsv')!=sha(root/'sources/deep/added/baseline_l2_task_metrics.tsv'):
            raise ValueError('Baseline task output provenance mismatch')
        if prov.get('released_outputs',{}).get('baseline_l2_macro_summary.tsv')!=sha(root/'sources/deep/added/baseline_l2_macro_summary.tsv'):
            raise ValueError('Baseline macro output provenance mismatch')
        pubprod=ROOT/'src/paper/kaggle_baseline_geometry/compute_baseline_geometry.py'
        txt=pubprod.read_text()
        for target in ['baseline_l2_task_metrics.tsv','baseline_l2_macro_summary.tsv']:
            if target not in txt: raise ValueError('Recovered public producer missing write target: '+target)
        report['baseline_macro']={'task_rows':len(task),'macro_rows':len(macro),'max_abs_error':float(np.nanmax(np.abs(macro[metrics].to_numpy()-saved[metrics].to_numpy()))),'status':'AGGREGATION_MATCH_HISTORICAL_PRODUCER_SOURCE_RECOVERED_OUTPUT_HASH_MATCH_EXECUTION_LOG_NOT_RECOVERED','historical_producer_sha256':prov['historical_producer_sha256']}
        report['status']='PASS_SUMMARY_DERIVATIONS_ONLY'
    except Exception as e:
        report.update(status='FAIL',error=type(e).__name__+': '+str(e));raise
    finally:
        a.report.parent.mkdir(parents=True,exist_ok=True)
        with a.report.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
        print(json.dumps(report,indent=2))
if __name__=='__main__':main()
