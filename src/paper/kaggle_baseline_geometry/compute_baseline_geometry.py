"""Recover frozen input kernels and score missing R3-matched baselines; no fit."""
from pathlib import Path
import hashlib,json,sys,os
import numpy as np
import pandas as pd
import frozen_geometry as fg

OUT=Path(os.environ.get('PERTURBCONTEXTALIGN_BASELINE_GEOMETRY_OUT', str(Path(__file__).resolve().parents[1]))).resolve()
SRC=Path(os.environ['PERTURBCONTEXTALIGN_FINE_ANALYSIS_ROOT']).resolve()
PRED=SRC/'predictions/kaggle_prediction_v1'
R3=Path(os.environ['PERTURBCONTEXTALIGN_R3_ROOT']).resolve()
MAP={'bge_m3':'bge_entity_exposure','qwen3_0_6b':'qwen3_entity_exposure','sapbert':'sapbert_entity_exposure','biomedbert':'biomedbert_entity_exposure','medcpt_article':'medcpt_article_entity_exposure','medcpt_query':'medcpt_query_entity_exposure'}
BASE=['identity_exposure','tfidf_entity_exposure','morgan_entity_exposure']
ONLY=['zero','source_mean','source_median','same_drug_source_mean']
NATIVE={'identity_exposure':'source_drug_dose_onehot_cosine','tfidf_entity_exposure':'source_fitted_exact_prompt_tfidf_cosine','morgan_entity_exposure':'half_tanimoto_plus_half_dose_equality'}
MAN=[]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def pin(p,expected=None,binding='existing accepted input'):
    p=Path(p);s=sha(p)
    if expected is not None:assert s==expected,(p,'hash mismatch')
    MAN.append(dict(path=str(p),sha256=s,size_bytes=p.stat().st_size,binding=binding))
    return p
def table(p):return pd.read_csv(p,sep='\t',float_precision='round_trip')
def write(name,frame):frame.to_csv(OUT/'results'/name,sep='\t',index=False,na_rep='NA',float_format='%.17g',mode='x')
def dump(name,obj):
    with (OUT/name).open('x') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')
def axis_digest(vals):return hashlib.sha256(json.dumps([str(v) for v in vals],ensure_ascii=False,separators=(',',':')).encode()).hexdigest()

def run():
    oldin=table(pin(R3/'input_manifest.tsv'));oldout=table(pin(R3/'output_manifest.tsv'))
    oldmap=oldout.set_index('path').sha256.to_dict()
    def accepted(rel):return pin(R3/rel,oldmap[rel],'accepted R3 output manifest')
    predmanifest=PRED/'output_manifest.json'
    expected=oldin.set_index('path').loc[str(predmanifest),'sha256']
    pf=json.loads(pin(predmanifest,expected,'R3 pinned prediction manifest').read_text())
    pmap={x['path']:x['sha256'] for x in pf['files']}
    def frozen(rel):return pin(PRED/rel,pmap[rel],'original frozen prediction output manifest')
    same=pin(R3/'code/r3app/frozen_geometry.py',oldmap['code/r3app/frozen_geometry.py'],'accepted metric helper')
    assert sha(same)==sha(Path(__file__).with_name('frozen_geometry.py'))
    membership=table(accepted('results/cohort_membership.tsv'))
    cohorts=table(accepted('results/task_cohort_manifest.tsv'))
    genes=table(accepted('results/gene_axis_manifest.tsv'))
    oldtask=table(accepted('results/l2_task_metrics.tsv'))
    assert len(cohorts)==6 and len(membership)==816 and membership.atomic_id.is_unique
    tasks=[];queries=[];bindings=[];checks=[];outputs=[]
    for task in cohorts.itertuples():
        taskid=task.task_id
        q=table(frozen(f'{taskid}/query_atoms.tsv'));s=table(frozen(f'{taskid}/source_atoms.tsv'))
        lm=table(frozen(f'{taskid}/landmarks.tsv'))
        fitted=json.loads(frozen(f'{taskid}/feature_fit_metadata.json').read_text())
        fit=json.loads(frozen(f'{taskid}/fit_audit.json').read_text())
        kernelpath=frozen(f'{taskid}/landmark_kernels.npz')
        truthpath=frozen(f'{taskid}/evaluation_truth.npz')
        cachepath=accepted(f'geometry/{taskid}.npz')
        cm=membership[membership.task_id==taskid].sort_values('original_query_position')
        assert q.atomic_id.tolist()==cm.atomic_id.tolist()
        keep=cm.included_primary.astype(str).str.lower().eq('true').to_numpy()
        qm=q.loc[keep].reset_index(drop=True);n=len(qm)
        assert n==task.n_query_primary and n in (134,136) and len(q)-n==1
        assert qm.sm_lincs_id.is_unique and np.all(qm.dose_uM.to_numpy(float)==1) and np.all(qm.timepoint_hr.to_numpy(float)==24)
        assert axis_digest(qm.atomic_id)==task.cohort_id
        assert set(fitted['identity']['fit_atomic_id'])==set(s.atomic_id)
        assert fitted['tfidf']['fit_scope']=='source unique exact prompts' and fitted['tfidf']['svd_used'] is False
        assert fit['target_effects_passed_to_fit'] is False and fit['target_B_passed_to_features'] is False
        exposure=lambda row:(str(row.sm_lincs_id),float(row.dose_uM),float(row.timepoint_hr))
        keys={}
        for j,row in enumerate(lm.itertuples()):keys.setdefault(exposure(row),[]).append(j)
        colgroups=[keys[exposure(row)] for row in qm.itertuples()]
        cols=np.array([v[0] for v in colgroups],int)
        assert set(lm.atomic_id)<=set(s.atomic_id)
        assert len(fitted['tfidf']['feature_names'])==len(fitted['tfidf']['idf'])
        with np.load(kernelpath,allow_pickle=False) as z,np.load(cachepath,allow_pickle=False) as g,np.load(truthpath,allow_pickle=False) as yt:
            assert np.array_equal(z['query_atomic_id'],q.atomic_id.to_numpy(str))
            assert np.array_equal(z['source_atomic_id'],s.atomic_id.to_numpy(str))
            assert np.array_equal(z['landmark_atomic_id'],lm.atomic_id.to_numpy(str))
            assert np.array_equal(z['all_atomic_id'][z['query_rows']],z['query_atomic_id'])
            assert np.array_equal(z['all_atomic_id'][z['landmark_rows']],z['landmark_atomic_id'])
            assert np.array_equal(g['atomic_id'],qm.atomic_id.to_numpy(str))
            assert np.array_equal(g['sm_lincs_id'],qm.sm_lincs_id.to_numpy(str))
            ga=genes[genes.task_id==taskid].sort_values('rank')
            assert len(ga)==3000 and np.array_equal(g['source_feature_row'],ga.source_feature_row.to_numpy(int)) and np.array_equal(g['source_gene_id'],ga.source_gene_id.to_numpy(str))
            assert np.array_equal(yt['atomic_id'],q.atomic_id.to_numpy(str))
            assert np.array_equal(yt['source_feature_row'],g['source_feature_row']) and np.array_equal(yt['source_gene_id'],g['source_gene_id'])
            y=yt['truth'][keep].astype(float);assert np.isfinite(y).all() and (np.linalg.norm(y,axis=1)>0).all()
            assert np.allclose(fg.cosine_matrix(y),g['truth_cosine'],rtol=0,atol=1e-12)
            triu=np.triu_indices(n,1);rows=z['query_rows'][keep];lmrows=z['landmark_rows'][cols]
            gs={k:g[k].copy() for k in ['atomic_id','sm_lincs_id','source_feature_row','source_gene_id','truth_cosine','truth_relevance','idcg','random_ndcg','truth_status']}
            for queryi,(qrow,candidates) in enumerate(zip(qm.itertuples(),colgroups)):
                bindings.append(dict(task_id=taskid,atomic_id=qrow.atomic_id,query_position=queryi,source_landmark_position=int(candidates[0]),source_landmark_atomic_id=lm.iloc[candidates[0]].atomic_id,matching_landmarks=len(candidates),sm_lincs_id=qrow.sm_lincs_id,dose_uM=float(qrow.dose_uM),timepoint_hr=float(qrow.timepoint_hr),cohort_id=task.cohort_id))
            for method in list(MAP.values())+BASE:
                raw=z['base__'+method];assert raw.shape==(len(q)+len(s),len(lm)) and np.isfinite(raw).all()
                # Exact duplicate source exposures must give identical kernel columns.
                for cg in colgroups:
                    for other in cg[1:]:assert np.array_equal(raw[:,cg[0]],raw[:,other]),(taskid,method,'duplicate kernel columns differ')
                assert np.array_equal(raw[rows],raw[lmrows]),(taskid,method,'same inputs differ across donors')
                sim=raw[np.ix_(rows,cols)].copy()
                assert np.allclose(sim,sim.T,rtol=0,atol=1e-12) and np.allclose(np.diag(sim),1,rtol=0,atol=1e-12)
                assert sim.min()>=-1-1e-12 and sim.max()<=1+1e-12
                if method in MAP.values():
                    model=next(k for k,v in MAP.items() if v==method)
                    diff=float(np.max(np.abs(sim-g['repr__'+model])))
                    assert diff<=1e-10,(taskid,method,'accepted LLM geometry mismatch',diff)
                    checks.append(dict(task_id=taskid,method=method,max_abs_difference_from_accepted_LLM=diff,status='PASS'))
                    continue
                gs['repr__'+method]=sim
                rsa=fg.rank_correlation(sim[triu],g['truth_cosine'][triu])
                ndcg=fg.score_neighbors(sim,g['truth_relevance'],g['idcg'],k=10)
                random=g['random_ndcg'];excess=ndcg-random
                assert np.isfinite(ndcg).all() and np.isfinite(random).all()
                rs='VALID' if np.isfinite(rsa) else 'CONSTANT_PAIR_GEOMETRY'
                if method=='identity_exposure':
                    assert len(np.unique(sim[triu]))==1 and not np.isfinite(rsa)
                    assert np.allclose(ndcg,random,rtol=0,atol=1e-14)
                context=dict(task_id=taskid,cell_type=task.cell_type,donor_id=task.donor_id,cohort_id=task.cohort_id,gene_axis_id=task.gene_axis_id,method=method)
                tasks.append(dict(**context,n_query_atoms=n,n_genes=3000,native_similarity=NATIVE[method],rsa=rsa,rsa_status=rs,ndcg_mean=float(ndcg.mean()),random_ndcg_mean=float(random.mean()),excess_ndcg_mean=float(excess.mean()),n_l2_valid=int(np.isfinite(ndcg).sum()),n_l2_na=0))
                for i,qr in enumerate(qm.itertuples()):
                    queries.append(dict(**context,atomic_id=qr.atomic_id,sm_lincs_id=qr.sm_lincs_id,query_position=i,ndcg=float(ndcg[i]),random_ndcg=float(random[i]),excess_ndcg=float(excess[i]),l2_status=str(g['truth_status'][i]),n_candidates=n-1))
            outpath=OUT/'geometry'/f'{taskid}.npz'
            with outpath.open('xb') as h:np.savez_compressed(h,**gs)
            outputs.append(dict(path=str(outpath.relative_to(OUT)),sha256=sha(outpath),size_bytes=outpath.stat().st_size))
        print('COMPLETED',taskid,n,'primary conditions',flush=True)
    t=pd.DataFrame(tasks);q=pd.DataFrame(queries)
    assert len(t)==18 and len(q)==2430 and len(bindings)==810 and len(checks)==36
    assert not t.duplicated(['task_id','method']).any() and not q.duplicated(['task_id','method','atomic_id']).any()
    metrics=['rsa','ndcg_mean','random_ndcg_mean','excess_ndcg_mean']
    typed=[];macro=[]
    def fm(x):return float(np.mean(x[np.isfinite(x)])) if np.isfinite(x).any() else np.nan
    for (ct,m),part in t.groupby(['cell_type','method'],sort=False):
        row=dict(cell_type=ct,method=m,n_tasks_nominal=len(part))
        for field in metrics:
            v=part[field].to_numpy(float);row[field]=fm(v);row[field+'__valid_tasks']=int(np.isfinite(v).sum())
        typed.append(row)
    typed=pd.DataFrame(typed)
    for m,part in typed.groupby('method',sort=False):
        row=dict(method=m,n_types_nominal=len(part),n_studies=1,aggregation='equal_donor_within_type_then_equal_type;descriptive_only')
        for field in metrics:
            v=part[field].to_numpy(float);row[field]=fm(v);row[field+'__valid_types']=int(np.isfinite(v).sum())
        macro.append(row)
    statuses=[]
    for m in BASE+ONLY:
        for metric in ['rsa','excess_ndcg_mean']:
            reason='PREDICTION_ONLY_NO_INPUT_REPRESENTATION' if m in ONLY else 'CONSTANT_PAIR_GEOMETRY' if m==BASE[0] and metric=='rsa' else 'CALCULATED'
            statuses.append(dict(method=m,metric=metric,status='NOT_APPLICABLE' if m in ONLY else 'UNDEFINED' if reason=='CONSTANT_PAIR_GEOMETRY' else 'VALID',reason=reason,n_tasks_evaluated=0 if m in ONLY else 6))
    write('baseline_l2_task_metrics.tsv',t);write('baseline_l2_query_metrics.tsv',q)
    write('baseline_l2_type_summary.tsv',typed);write('baseline_l2_macro_summary.tsv',pd.DataFrame(macro))
    write('baseline_applicability.tsv',pd.DataFrame(statuses));write('query_landmark_binding.tsv',pd.DataFrame(bindings));write('LLM_kernel_restriction_checks.tsv',pd.DataFrame(checks))
    manifest=pd.DataFrame(MAN).drop_duplicates('path')
    for row in manifest.itertuples():assert sha(row.path)==row.sha256,'Input changed during run'
    manifest.to_csv(OUT/'provenance'/'input_manifest.tsv',sep='\t',index=False)
    pd.DataFrame(outputs).to_csv(OUT/'provenance'/'geometry_manifest.tsv',sep='\t',index=False)
    dump('qa/producer_QA.json',dict(status='PASS_PENDING_INDEPENDENT_QA',baseline_task_rows=18,baseline_query_rows=2430,condition_bindings=810,LLM_restriction_checks=36,cohort_exclusions_changed=False,new_training=False,new_encoding=False,new_inference_tests=False,source_file_count=len(manifest)))
    print(pd.DataFrame(macro)[['method']+metrics].to_string(index=False))

if __name__=='__main__':run()
