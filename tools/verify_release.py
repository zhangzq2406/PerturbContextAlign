#!/usr/bin/env python3
"""Verify published frozen table claims without recomputing models or responses."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

def verify(root):
    checks=[]
    def read(p):return pd.read_csv(root/p,sep='\t')
    def check(name,value,expected):
        ok=bool(value==expected)
        checks.append({'check':name,'observed':str(value),'expected':str(expected),'status':'PASS' if ok else 'FAIL'})
        if not ok:raise AssertionError(f'{name}: observed {value}, expected {expected}')
    r=read('r1/R1_DATASET_AUTHORITY.tsv');check('R1_dataset_n',len(r),9);check('R1_semantic_instances',int(r.semantic_instance_n.sum()),186319)
    a=read('r2/R2_ELIGIBLE_RESPONSE_ATOM_INDEX_v20.tsv.gz');check('response_atom_n',len(a),9054);check('unique_response_atom_n',a.response_atom_id.nunique(),9054)
    check('dataset_condition_n',len(a[['dataset_id','condition_id']].drop_duplicates()),5882);check('global_condition_string_n',a.condition_id.nunique(),3942)
    rsa=read('r2/R2_FIG1D_RSA_MATRIX_v2.tsv');cols=['BGE','Qwn','Sap','Bio','Art','Qry'];check('Article_RSA_top_datasets',int(rsa[cols].idxmax(axis=1).eq('Art').sum()),5)
    n=read('r2/R2_FIG1D_NDCG_GAIN_MATCHED_RANDOM_MATRIX_v2.tsv');check('Article_local_top_datasets',int(n[cols].idxmax(axis=1).eq('Art').sum()),6)
    for ds in ['replogle_k562_essential','replogle_rpe1','tian_activation','tian_inhibition']:check('onehot_NA_'+ds,bool(rsa.loc[rsa.dataset_id.eq(ds),'OH'].isna().all()),True)
    k=read('r2/R2_CORRECTED_P5A_DELTA_SUMMARY_v1_2.tsv');check('knowledge_positive_encoders',int(k.delta_rsa_macro_mean.gt(0).sum()),4);check('knowledge_negative_encoders',int(k.delta_rsa_macro_mean.lt(0).sum()),2)
    check('knowledge_positive_CI_encoders',','.join(k.loc[k.delta_rsa_bootstrap_ci_low.gt(0),'model_key']),'qwen3_0_6b')
    f=read('r2/R2_CORRECTED_NATIVE_FUSION_BY_COMPARISON_v1.tsv');check('fusion_comparisons',len(f),33);check('fusion_gains',int(f.fusion_minus_best_single_rsa.gt(0).sum()),10);check('fusion_losses',int(f.fusion_minus_best_single_rsa.lt(0).sum()),23)
    e=read('deep/R1/r1_paired_changes.tsv');e=e[e.contrast_id.isin(['E1','E2'])];check('matched_exposure_records',len(e),36)
    for col in ['delta_dose_accuracy','delta_dose_macro_f1','delta_excess_ndcg_mean']:check('exposure_positive_'+col,int(e[col].gt(0).sum()),36)
    check('exposure_RSA_positive',int(e.delta_rsa.gt(0).sum()),15);check('exposure_RSA_negative',int(e.delta_rsa.lt(0).sum()),21)
    b=read('deep/R3/baseline_paired_differences.tsv');refs=['same_drug_source_mean','identity_exposure','tfidf_entity_exposure','morgan_entity_exposure']
    for ref in refs:
        q=b[b.baseline.eq(ref)];check('Kaggle_'+ref+'_paired_n',len(q),36);check('Kaggle_'+ref+'_MAE_down_order_down',int((q.mae_gain.gt(0)&q.gene_spearman_gain.lt(0)).sum()),36)
    c=read('r2/R2_CROSS_CONTEXT_IDENTITY_CONDITIONING_GAIN_v1.tsv')
    for v,ra,lo in [('P2',6,11),('P4',5,7)]:
        q=c[c.identity_conditioned_view.eq(v)];check('crosscontext_'+v+'_comparisons',len(q),18);check('crosscontext_'+v+'_RSA_direction_positive',int(q.delta_rsa_identity_conditioning_mean.gt(0).sum()),ra);check('crosscontext_'+v+'_local_direction_positive',int(q.delta_hit_gain_identity_conditioning_mean.gt(0).sum()),lo)
    return checks

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--report',type=Path);a=p.parse_args();result=verify(a.data_root)
    report={'status':'PASS','scope':'Frozen source-table checks, not scientific recomputation','check_n':len(result),'checks':result}
    if a.report:a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
