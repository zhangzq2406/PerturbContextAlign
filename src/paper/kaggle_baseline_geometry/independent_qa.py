"""Independent no-refit QA of the three frozen-input baseline geometries.

No producer or frozen_geometry imports.  Scores are independently reconstructed
using scipy Spearman and a tie-block rank-weight implementation of expected DCG.
"""
from pathlib import Path
import hashlib
import json
import traceback
import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ConstantInputWarning

OUT = Path(os.environ.get('PERTURBCONTEXTALIGN_BASELINE_GEOMETRY_OUT', str(Path(__file__).resolve().parents[1]))).resolve()
R3 = Path(os.environ['PERTURBCONTEXTALIGN_R3_ROOT']).resolve()
PRED = Path(os.environ['PERTURBCONTEXTALIGN_FINE_ANALYSIS_ROOT']).resolve() / 'predictions/kaggle_prediction_v1'
METHODS = ['identity_exposure', 'tfidf_entity_exposure', 'morgan_entity_exposure']
LLMS = {'bge_entity_exposure':'bge_m3', 'qwen3_entity_exposure':'qwen3_0_6b',
        'sapbert_entity_exposure':'sapbert', 'biomedbert_entity_exposure':'biomedbert',
        'medcpt_article_entity_exposure':'medcpt_article', 'medcpt_query_entity_exposure':'medcpt_query'}
CHECKS = []
MAXDIFF = {}

def read(p):
    return pd.read_csv(p, sep='\t', float_precision='round_trip')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''):h.update(block)
    return h.hexdigest()

def check(condition, description):
    if not bool(condition):raise AssertionError(description)
    CHECKS.append(description)

def same(a,b,label,tol=1e-12):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    check(a.shape==b.shape,label+': shape')
    check(np.array_equal(np.isnan(a),np.isnan(b)),label+': NA mask')
    mask=np.isfinite(a)&np.isfinite(b)
    diff=float(np.max(np.abs(a[mask]-b[mask]))) if mask.any() else 0.
    MAXDIFF[label]=diff
    check(np.allclose(a,b,atol=tol,rtol=0,equal_nan=True),label+': values')

def tie_rank_weights(scores,k=10):
    # Per-item expected discounted rank contribution; a boundary tie extends
    # past k and its whole block, not just the chosen top-k slice, is averaged.
    unique,inv,count=np.unique(scores,return_inverse=True,return_counts=True)
    greater=np.cumsum(count[::-1])[::-1]-count
    discounts=1./np.log2(np.arange(2,k+2))
    weights=np.zeros(len(unique))
    for j,(n_greater,n_tied) in enumerate(zip(greater,count)):
        end=min(k,int(n_greater+n_tied))
        if n_greater<k:
            weights[j]=discounts[int(n_greater):end].sum()/n_tied
    return weights[inv]

def independent_neighbors(sim,truth,k=10):
    n=len(truth); rel=np.zeros_like(truth); idcg=np.empty(n); rnd=np.empty(n); ndcg=np.empty(n)
    discounts=1./np.log2(np.arange(2,k+2))
    for i in range(n):
        mask=np.arange(n)!=i; ts=truth[i,mask]
        check(len(np.unique(ts))>1,'informative truth '+str(i))
        threshold=np.sort(ts)[-k]
        r=(ts>threshold).astype(float)
        equal=ts==threshold
        r[equal]=(k-r.sum())/equal.sum()
        rel[i,mask]=r
        idcg[i]=np.dot(np.sort(r)[-k:][::-1],discounts)
        rnd[i]=r.mean()*discounts.sum()/idcg[i]
        ndcg[i]=np.dot(r,tie_rank_weights(sim[i,mask],k))/idcg[i]
    return ndcg,rnd,rel,idcg

def main():
    consumed=read(OUT/'provenance/input_manifest.tsv')
    check(consumed.path.is_unique,'source manifest paths unique')
    pred_pins={r['path']:r for r in json.loads((PRED/'output_manifest.json').read_text())['files']}
    r3_pins=read(R3/'output_manifest.tsv').set_index('path')
    for row in consumed.itertuples():
        p=Path(row.path)
        check(p.stat().st_size==row.size_bytes and sha(p)==row.sha256,'source SHA256 '+str(p))
        if p.is_relative_to(PRED) and str(p.relative_to(PRED)) in pred_pins:
            check(row.sha256==pred_pins[str(p.relative_to(PRED))]['sha256'],'original prediction manifest '+p.name)
        if p.is_relative_to(R3) and str(p.relative_to(R3)) in r3_pins.index:
            check(row.sha256==r3_pins.loc[str(p.relative_to(R3)),'sha256'],'accepted R3 output manifest '+p.name)
    for row in read(OUT/'provenance/geometry_manifest.tsv').itertuples():
        check(sha(OUT/row.path)==row.sha256,'new geometry SHA256 '+row.path)
    tasks=read(R3/'results/task_cohort_manifest.tsv')
    members=read(R3/'results/cohort_membership.tsv')
    genes=read(R3/'results/gene_axis_manifest.tsv')
    got_t=read(OUT/'results/baseline_l2_task_metrics.tsv')
    got_q=read(OUT/'results/baseline_l2_query_metrics.tsv')
    binding=read(OUT/'results/query_landmark_binding.tsv')
    got_llm=read(OUT/'results/LLM_kernel_restriction_checks.tsv')
    check(len(tasks)==6 and len(got_t)==18 and len(got_q)==2430 and len(binding)==810,'six-task complete row counts')
    check(not got_t.duplicated(['task_id','method']).any(),'unique task-method records')
    check(not got_q.duplicated(['task_id','method','atomic_id']).any(),'unique query-method records')
    check(not binding.duplicated(['task_id','atomic_id']).any(),'unique query-landmark records')
    check(set(got_t.method)==set(METHODS) and set(got_t.task_id)==set(tasks.task_id),'method/task scope preserved')
    check(len(got_llm)==36 and set(got_llm.status)=={'PASS'},'36 reported LLM checks PASS')
    computed=[]; restriction_diffs=[]
    for t in tasks.itertuples():
        d=PRED/t.task_id
        query=read(d/'query_atoms.tsv'); landmarks=read(d/'landmarks.tsv'); source=read(d/'source_atoms.tsv')
        primary=query[query.dose_uM.eq(1)].reset_index(drop=True)
        check(len(primary)==t.n_query_primary and len(query)-len(primary)==1,'primary count '+t.task_id)
        check(set(primary.timepoint_hr)=={24} and primary.sm_lincs_id.is_unique,'fixed exposure unique drugs '+t.task_id)
        mem=members[(members.task_id==t.task_id)&members.included_primary].sort_values('primary_position')
        check(primary.atomic_id.tolist()==mem.atomic_id.tolist(),'unchanged condition axis '+t.task_id)
        check(set(mem.cohort_id)=={t.cohort_id},'accepted cohort ID '+t.task_id)
        with np.load(d/'landmark_kernels.npz',allow_pickle=False) as z, np.load(R3/'geometry'/f'{t.task_id}.npz',allow_pickle=False) as old, np.load(OUT/'geometry'/f'{t.task_id}.npz',allow_pickle=False) as new:
            ids=z['all_atomic_id'].tolist(); qrows=np.array([ids.index(i) for i in primary.atomic_id])
            check(z['landmark_atomic_id'].tolist()==landmarks.atomic_id.tolist(),'landmark axis '+t.task_id)
            check(set(landmarks.atomic_id)<=set(source.atomic_id),'source-only landmarks '+t.task_id)
            check(np.array_equal(new['atomic_id'],primary.atomic_id.to_numpy(str)) and np.array_equal(old['atomic_id'],new['atomic_id']),'old/new atomic axes '+t.task_id)
            gm=genes[genes.task_id==t.task_id].sort_values('rank')
            for key in ['source_gene_id','source_feature_row']:
                check(np.array_equal(new[key],old[key]) and np.array_equal(new[key],gm[key]),'gene axis '+key+' '+t.task_id)
            for key in ['truth_cosine','truth_relevance','idcg','random_ndcg','truth_status']:
                check(np.array_equal(new[key],old[key]),'unchanged truth '+key+' '+t.task_id)
            cols=[];matches=[]
            for q in primary.itertuples():
                hit=np.flatnonzero(landmarks.sm_lincs_id.eq(q.sm_lincs_id)&landmarks.dose_uM.eq(q.dose_uM)&landmarks.timepoint_hr.eq(q.timepoint_hr))
                check(len(hit)>0,'exact exposure landmark exists '+q.atomic_id)
                cols.append(hit[0]);matches.append(hit)
            bt=binding[binding.task_id==t.task_id].sort_values('query_position')
            check(bt.atomic_id.tolist()==primary.atomic_id.tolist(),'binding condition order '+t.task_id)
            check(np.array_equal(bt.source_landmark_position,cols),'binding selected landmark '+t.task_id)
            check(np.array_equal(bt.matching_landmarks,[len(i) for i in matches]),'binding duplicate counts '+t.task_id)
            check(bt.source_landmark_atomic_id.tolist()==landmarks.iloc[cols].atomic_id.tolist(),'binding landmark IDs '+t.task_id)
            for method in METHODS+list(LLMS):
                kernel=z['base__'+method]
                sim=kernel[np.ix_(qrows,cols)]
                check(np.isfinite(sim).all(),'finite geometry '+t.task_id+method)
                same(sim,sim.T,'symmetric '+t.task_id+method,1e-10)
                same(np.diag(sim),np.ones(len(sim)),'unit diagonal '+t.task_id+method,1e-10)
                for hit in matches:
                    same(kernel[np.ix_(qrows,hit)],np.repeat(kernel[qrows,hit[0],None],len(hit),axis=1),'duplicate landmark '+t.task_id+method+str(hit[0]),1e-10)
                # Corresponding source atoms and target atoms have the same
                # metadata-only native inputs. Verify their full kernel rows.
                same(kernel[qrows],kernel[z['landmark_rows'][cols]],'donor-invariant native input '+t.task_id+method,1e-10)
                if method in LLMS:
                    diff=float(np.max(np.abs(sim-old['repr__'+LLMS[method]])))
                    check(diff<=1e-10,'independent LLM restriction '+t.task_id+method)
                    r=got_llm[(got_llm.task_id==t.task_id)&(got_llm.method==method)]
                    check(len(r)==1,'one LLM check record '+t.task_id+method)
                    same(diff,r.max_abs_difference_from_accepted_LLM.iloc[0],'reported LLM difference '+t.task_id+method)
                    restriction_diffs.append(diff)
                    continue
                check(np.array_equal(sim,new['repr__'+method]),'new native similarity identical '+t.task_id+method)
                tri=np.triu_indices(len(sim),1)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',ConstantInputWarning)
                    rsa=float(spearmanr(sim[tri],old['truth_cosine'][tri]).statistic)
                ndcg,rnd,rel,idcg=independent_neighbors(sim,old['truth_cosine'])
                same(rel,old['truth_relevance'],'independent fractional relevance '+t.task_id+method)
                same(idcg,old['idcg'],'independent IDCG '+t.task_id+method)
                same(rnd,old['random_ndcg'],'independent random NDCG '+t.task_id+method)
                qq=got_q[(got_q.task_id==t.task_id)&(got_q.method==method)].sort_values('query_position')
                check(len(qq)==len(primary) and qq.atomic_id.tolist()==primary.atomic_id.tolist(),'all query rows '+t.task_id+method)
                check(set(qq.l2_status)=={'VALID'} and set(qq.n_candidates)=={len(primary)-1},'query statuses '+t.task_id+method)
                check(set(qq.cohort_id)=={t.cohort_id} and set(qq.gene_axis_id)=={t.gene_axis_id},'query cohort and gene IDs '+t.task_id+method)
                same(ndcg,qq.ndcg,'independent NDCG '+t.task_id+method)
                same(rnd,qq.random_ndcg,'query random NDCG '+t.task_id+method)
                same(ndcg-rnd,qq.excess_ndcg,'independent excess NDCG '+t.task_id+method)
                row=got_t[(got_t.task_id==t.task_id)&(got_t.method==method)].iloc[0]
                check(row.cohort_id==t.cohort_id and row.gene_axis_id==t.gene_axis_id,'task IDs '+t.task_id+method)
                check(row.n_query_atoms==len(primary) and row.n_genes==3000 and row.n_l2_valid==len(primary) and row.n_l2_na==0,'task denominators '+t.task_id+method)
                vals=dict(rsa=rsa,ndcg_mean=float(ndcg.mean()),random_ndcg_mean=float(rnd.mean()),excess_ndcg_mean=float((ndcg-rnd).mean()))
                for key,value in vals.items():same(value,row[key],'task '+key+' '+t.task_id+method)
                expected_status='CONSTANT_PAIR_GEOMETRY' if np.isnan(rsa) else 'VALID'
                check(row.rsa_status==expected_status,'RSA status '+t.task_id+method)
                if method=='identity_exposure':
                    check(np.isnan(rsa) and len(np.unique(sim[tri]))==1,'Identity constant-pair RSA NA '+t.task_id)
                    same(ndcg-rnd,np.zeros(len(primary)),'Identity tied neighbor chance '+t.task_id)
                computed.append(dict(task_id=t.task_id,cell_type=t.cell_type,method=method,**vals))
    ct=pd.DataFrame(computed); type_expected=ct.groupby(['cell_type','method'])[['rsa','ndcg_mean','random_ndcg_mean','excess_ndcg_mean']].mean()
    type_got=read(OUT/'results/baseline_l2_type_summary.tsv').set_index(['cell_type','method'])
    check(len(type_got)==6,'six type-method summaries')
    same(type_expected.to_numpy(),type_got.loc[type_expected.index,type_expected.columns].to_numpy(),'equal-three-donor type summaries')
    macro_expected=type_expected.groupby('method').mean()
    macro_got=read(OUT/'results/baseline_l2_macro_summary.tsv').set_index('method')
    check(len(macro_got)==3 and set(macro_got.n_studies)=={1},'three descriptive macro rows one study')
    same(macro_expected.to_numpy(),macro_got.loc[macro_expected.index,macro_expected.columns].to_numpy(),'equal-two-type macro summaries')
    app=read(OUT/'results/baseline_applicability.tsv')
    check(len(app)==14 and app.method.nunique()==7,'all-seven baseline applicability records')
    for method in ['zero','source_mean','source_median','same_drug_source_mean']:
        sub=app[app.method==method]
        check(len(sub)==2 and set(sub.status)=={'NOT_APPLICABLE'} and set(sub.n_tasks_evaluated)=={0},'prediction-only L2 not substituted '+method)
    return dict(status='PASS',scope='independent frozen native-kernel reconstruction and numerical scoring; no training, encoding, predictions, significance tests, or old-output writes',
                source_files_verified=len(consumed),tasks=6,query_bindings=810,baseline_task_rows=18,baseline_query_rows=2430,baseline_macro_rows=3,
                independent_LLM_checks=36,max_LLM_restriction_error=max(restriction_diffs),
                independent_scoring='scipy.stats.spearmanr with average ties; full equal-score-block rank weights for expectedDCG; independently reconstructed fractional top10 truth relevance',
                old_cohort_and_gene_axes_unchanged=True,constant_RSA_preserved_as_NA=True,identity_excess_NDCG_zero=True,
                check_count=len(CHECKS),maximum_numeric_difference=max(MAXDIFF.values()),differences=MAXDIFF)

if __name__=='__main__':
    try:report=main()
    except Exception as error:
        report=dict(status='FAIL',error=str(error),traceback=traceback.format_exc(),checks_before_failure=len(CHECKS))
    report['independent_qa_script_sha256']=sha(Path(__file__))
    (OUT/'qa/independent_QA.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='differences'},indent=2))
    raise SystemExit(0 if report['status']=='PASS' else 1)
