"""R3 cache-only scoring using the two exact, vendored frozen metric helpers."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from . import frozen_geometry as fg
from . import frozen_prediction as fp
from .common import (MODEL_METHOD,MODEL_OF,BASELINES,METHODS,SELECTED,GENE_MIN,K,
                     FrozenDataset,metadata_fields,digest_axis,write_table,write_json,
                     save_npz,require,finite_mean,EMB)


def gene_status(pred, truth, scores):
    ct=np.all(truth==truth[0:1],axis=0); cp=np.all(pred==pred[0:1],axis=0)
    status=np.full(truth.shape[1],'VALID',dtype='U40')
    status[ct]='CONSTANT_TRUTH'; status[cp]='CONSTANT_PREDICTION'
    status[ct&cp]='CONSTANT_TRUTH_AND_PREDICTION'
    if len(truth)<GENE_MIN:status[:]='INSUFFICIENT_COMPOUNDS'
    require(np.array_equal(np.isfinite(scores),status=='VALID'), 'Unexplained gene Spearman NA')
    return status,ct,cp


def tag(frame,t,method=None,model=None):
    frame=frame.copy()
    for name in ['task_id','cell_type','donor_id','cohort_id','gene_axis_id']:frame[name]=t[name]
    if method is not None:frame['method']=method
    if model is not None:frame['model_key']=model
    return frame


def task_context(t):
    return {k:t[k] for k in ['task_id','cell_type','donor_id','cohort_id','gene_axis_id']}


def paired_rows(t,query_mae,gene_score):
    rows=[]
    for model,method in MODEL_METHOD.items():
        a=gene_score[method]
        for baseline in BASELINES:
            b=gene_score[baseline]; av=np.isfinite(a); bv=np.isfinite(b); common=av&bv
            same=np.array_equal(av,bv)
            ga=finite_mean(a[common]); gb=finite_mean(b[common])
            rows.append(dict(**task_context(t),model_key=model,method=method,baseline=baseline,
                n_query_atoms=len(t['qm']),n_genes_total=3000,
                model_mae=float(query_mae[method].mean()),baseline_mae=float(query_mae[baseline].mean()),
                mae_gain=float((query_mae[baseline]-query_mae[method]).mean()),
                model_gene_spearman_original_mean=finite_mean(a),
                baseline_gene_spearman_original_mean=finite_mean(b),
                model_n_genes_valid=int(av.sum()),baseline_n_genes_valid=int(bv.sum()),
                model_n_genes_na=int((~av).sum()),baseline_n_genes_na=int((~bv).sum()),
                n_common_valid_genes=int(common.sum()),same_valid_gene_set=bool(same),
                model_common_gene_spearman_mean=ga,baseline_common_gene_spearman_mean=gb,
                gene_spearman_gain=finite_mean((a-b)[common]),
                rank_gain_scope=('ORIGINAL_VALID_SET' if same else 'COMMON_VALID_GENE_SUBSET') if common.any() else 'NO_COMMON_VALID_GENES',
                rank_gain_status='VALID' if common.any() else 'NO_COMMON_VALID_GENES'))
    return rows


def type_and_macro(df, group_col, metrics):
    """Legacy hierarchy: equal donors within type, then equal types; report denominators."""
    rows=[]
    for (ct,method),sub in df.groupby(['cell_type',group_col],sort=False):
        r={'cell_type':ct,group_col:method,'n_tasks_nominal':len(sub)}
        for m in metrics:
            vals=sub[m].to_numpy(float); r[m]=finite_mean(vals); r[m+'__valid_tasks']=int(np.isfinite(vals).sum())
        rows.append(r)
    typed=pd.DataFrame(rows); rows=[]
    for method,sub in typed.groupby(group_col,sort=False):
        r={group_col:method,'n_types_nominal':len(sub),'n_studies':1,
           'aggregation':'equal_donor_within_type_then_equal_type;descriptive_only'}
        for m in metrics:
            vals=sub[m].to_numpy(float); r[m]=finite_mean(vals);r[m+'__valid_types']=int(np.isfinite(vals).sum())
        rows.append(r)
    return typed,pd.DataFrame(rows)


def run(root,out,spec):
    out=Path(out); res=out/'results'; geo=out/'geometry'
    ds=FrozenDataset(root,spec)
    role=[]
    for i,m in enumerate(METHODS):
        role.append(dict(method_axis=i,method=m,model_key=MODEL_OF.get(m,''),
                         role='DIRECT_ENCODER' if m in MODEL_OF else ('BASELINE' if m in BASELINES else 'STATE_NOT_SCORED'),
                         l2_included=m in MODEL_OF,l3_included=m in SELECTED))
    write_table(res/'method_role_map.tsv',pd.DataFrame(role))
    l2q=[];l2t=[];l3q=[];l3t=[];l3g=[];paired=[]
    cohorts=[];members=[];axes=[];bindings=[];na=[];preflight=[];similarity_checks=[]
    geometry_by_type={}
    for task_id in ds.tasks.task_id:
        print('R3_PRODUCER_TASK',task_id,flush=True)
        t=ds.task(task_id); qm=t['qm']; n=len(qm)
        y=t['truth'].astype(np.float64); truth=fg.cosine_matrix(y)
        rel,idcg,rnd,status,unique=fg.prepare_truth(truth,k=K,minimum_group_n=12)
        context=task_context(t)
        preflight.append(dict(task_id=task_id,n_query_original=len(t['q']),n_query_primary=n,
            n_genes=3000,truth_dtype=str(t['truth'].dtype),prediction_dtype=str(t['predictions'].dtype),
            truth_finite=True,prediction_finite=True,source_targets_finite=True,
            truth_min=float(y.min()),truth_max=float(y.max()),
            minimum_truth_norm=float(np.linalg.norm(y,axis=1).min()),zero_norm_rows=0,status='PASS'))
        cohorts.append(dict(**context,n_original_query=len(t['q']),n_source_atoms=len(t['s']),
            n_query_primary=n,n_unique_compounds=qm.sm_lincs_id.nunique(),n_excluded=1,
            dose_uM='1',timepoint_hr='24',n_genes=3000,n_candidate_per_query=n-1,
            training_set_changed=False,truth_origin='evaluation_truth.npz:truth;original_float32_promoted_float64'))
        cm=metadata_fields(t['q']);cm['task_id']=task_id
        cm['original_query_position']=np.arange(len(cm));cm['included_primary']=t['keep']
        cm['primary_position']=np.where(t['keep'],np.cumsum(t['keep'])-1,-1)
        cm['reason']=np.where(t['keep'],'PRIMARY_1uM_24h','BELINOSTAT_0_1uM_NOT_PRIMARY')
        cm['cohort_id']=t['cohort_id'];members.append(cm)
        gax=t['panel'][['rank','source_feature_row','source_gene_id','gene_symbol','union_column']].copy()
        axes.append(tag(gax,t))
        for a in t['excluded'].itertuples():
            na.append(dict(task_id=task_id,level='CONDITION',method='',unit_id=a.atomic_id,
                           reason='BELINOSTAT_0_1uM_NOT_PRIMARY',n_units=1))
        cache={};query_mae={};gene_score={}
        triu=np.triu_indices(n,1)
        gs={'atomic_id':qm.atomic_id.to_numpy(str),'sm_lincs_id':qm.sm_lincs_id.to_numpy(str),
            'source_feature_row':t['features'],'source_gene_id':t['genes'],
            'truth_cosine':truth,'truth_relevance':rel,'idcg':idcg,'random_ndcg':rnd,'truth_status':status}
        for model,method in MODEL_METHOD.items():
            x=t['embeddings'][model].astype(np.float64); sim=fg.cosine_matrix(x)
            gs['repr__'+model]=sim
            rsa=fg.rank_correlation(sim[triu],truth[triu])
            rsa_status='VALID' if np.isfinite(rsa) else 'CONSTANT_PAIR_GEOMETRY'
            ndcg=fg.score_neighbors(sim,rel,idcg,k=K); excess=ndcg-rnd
            fq=tag(metadata_fields(qm),t,method,model)
            fq['query_position']=np.arange(n);fq['ndcg']=ndcg;fq['random_ndcg']=rnd;fq['excess_ndcg']=excess
            fq['l2_status']=status;fq['n_candidates']=n-1;fq['truth_unique_scores']=unique
            l2q.append(fq)
            row=dict(**context,model_key=model,method=method,n_query_atoms=n,n_genes=3000,
                     rsa=rsa,rsa_status=rsa_status,ndcg_mean=finite_mean(ndcg),
                     random_ndcg_mean=finite_mean(rnd),excess_ndcg_mean=finite_mean(excess),
                     n_l2_valid=int(np.isfinite(ndcg).sum()),n_l2_na=int((~np.isfinite(ndcg)).sum()))
            l2t.append(row)
            reg=ds.reg_lookup.loc[qm.atomic_id].reset_index(drop=True)
            b=tag(reg[['atomic_id','text_id','prompt_sha256','text_row','view','variant']],t,method,model)
            b['embedding_file']=str(ds.root/EMB/(model+'__source_name.npz'));b['embedding_dimension']=x.shape[1]
            b['embedding_sha256']=ds.enc.set_index('model_key').loc[model,'sha256'];bindings.append(b)
            order=np.argsort(qm.sm_lincs_id.to_numpy(str),kind='stable')
            key=(t['cell_type'],model); canonical=sim[np.ix_(order,order)]
            canonical_x=x[order];canonical_drugs=qm.sm_lincs_id.to_numpy(str)[order]
            if key in geometry_by_type:
                prev=geometry_by_type[key]
                require(np.array_equal(prev[0],canonical_drugs) and np.array_equal(prev[1],canonical_x),
                        'Same-drug encoder values differ across donors within type')
                require(np.allclose(prev[2],canonical,rtol=0,atol=1e-10),
                        'Identical encoder features have nonmatching geometry')
                similarity_checks.append(dict(task_id=task_id,model_key=model,cell_type=t['cell_type'],
                    canonical_drug_id_match=True,embedding_bits_match=True,
                    max_geometry_abs_difference=float(np.max(np.abs(prev[2]-canonical))),status='PASS'))
            else:
                geometry_by_type[key]=(canonical_drugs,canonical_x,canonical)
            for reason in np.unique(status[status!='VALID']):
                na.append(dict(task_id=task_id,level='L2_QUERY',method=method,unit_id='ALL',
                    reason=str(reason),n_units=int(np.sum(status==reason))))
            if rsa_status!='VALID':
                na.append(dict(task_id=task_id,level='RSA',method=method,unit_id='TASK',reason=rsa_status,n_units=1))
        save_npz(geo/(task_id+'.npz'),**gs)
        for method in SELECTED:
            pred=t['predictions'][METHODS.index(method)].astype(np.float64)
            # Exactly the legacy MAE and average-rank Spearman helpers. No refit.
            cmets=fp.condition_metrics(pred,y)
            gmets=fp.gene_metrics(pred,y)
            score=gmets['spearman'];gst,ct,cp=gene_status(pred,y,score)
            if method in ['zero','source_mean','source_median']:
                require(cp.all(),'Expected constant reference varies across drugs: '+method)
            qframe=tag(metadata_fields(qm),t,method,MODEL_OF.get(method,''))
            qframe['query_position']=np.arange(n);qframe['mae']=cmets['mae'];qframe['n_genes']=3000
            l3q.append(qframe)
            gf=tag(gax[['rank','source_feature_row','source_gene_id','gene_symbol']],t,method,MODEL_OF.get(method,''))
            gf['gene_drug_spearman']=score;gf['n_conditions']=n;gf['spearman_valid']=np.isfinite(score)
            gf['spearman_n']=gmets['spearman_n'];gf['minimum_required_compounds']=GENE_MIN
            gf['constant_truth']=ct;gf['constant_prediction']=cp;gf['status']=gst;l3g.append(gf)
            tr=dict(**context,method=method,model_key=MODEL_OF.get(method,''),
                role='DIRECT_ENCODER' if method in MODEL_OF else 'BASELINE',
                n_query_atoms=n,n_genes_total=3000,n_genes_valid=int(np.isfinite(score).sum()),
                n_genes_na=int((~np.isfinite(score)).sum()),mae_mean=float(cmets['mae'].mean()),
                gene_spearman_mean=finite_mean(score),n_genes_constant_truth=int(ct.sum()),
                n_genes_constant_prediction=int(cp.sum()),n_genes_both_constant=int((ct&cp).sum()))
            l3t.append(tr);query_mae[method]=cmets['mae'];gene_score[method]=score
            for reason in np.unique(gst[gst!='VALID']):
                na.append(dict(task_id=task_id,level='L3_GENE',method=method,unit_id='ALL',
                               reason=str(reason),n_units=int(np.sum(gst==reason))))
        paired.extend(paired_rows(t,query_mae,gene_score))
        print('R3_PRODUCER_TASK_DONE',task_id,'primary_queries='+str(n),flush=True)
    frames={
        'task_cohort_manifest.tsv':pd.DataFrame(cohorts),'cohort_membership.tsv':pd.concat(members,ignore_index=True),
        'gene_axis_manifest.tsv':pd.concat(axes,ignore_index=True),'representation_binding.tsv':pd.concat(bindings,ignore_index=True),
        'l2_query_metrics.tsv':pd.concat(l2q,ignore_index=True),'l2_task_metrics.tsv':pd.DataFrame(l2t),
        'l3_query_metrics.tsv':pd.concat(l3q,ignore_index=True),'l3_task_metrics.tsv':pd.DataFrame(l3t),
        'l3_gene_drug_spearman.tsv.gz':pd.concat(l3g,ignore_index=True),
        'baseline_paired_differences.tsv':pd.DataFrame(paired),'na_and_exclusions.tsv':pd.DataFrame(na),
        'representation_cross_donor_checks.tsv':pd.DataFrame(similarity_checks)}
    f2q,f2t,f3q,f3t=[frames[n] for n in ['l2_query_metrics.tsv','l2_task_metrics.tsv','l3_query_metrics.tsv','l3_task_metrics.tsv']]
    require(len(f2q)==4860 and len(f2t)==36 and len(f3q)==10530 and len(f3t)==78,'Unexpected result rows')
    require(len(frames['l3_gene_drug_spearman.tsv.gz'])==234000,'Incomplete per-gene records')
    keys=['task_id','method','model_key','cohort_id','gene_axis_id']
    qjoin=f2q.merge(f3q[keys+['atomic_id','mae','n_genes']],on=keys+['atomic_id'],validate='one_to_one')
    tjoin=f2t.merge(f3t[keys+['mae_mean','gene_spearman_mean','n_genes_valid','n_genes_na']],on=keys,validate='one_to_one')
    require(len(qjoin)==4860 and len(tjoin)==36,'Join lost or duplicated rows')
    frames['l2_l3_query_join.tsv']=qjoin;frames['l2_l3_task_join.tsv']=tjoin
    frames['support_summary.tsv']=f3t[['task_id','method','cell_type','donor_id','n_query_atoms','n_genes_total',
                                     'n_genes_valid','n_genes_na','n_genes_constant_truth','n_genes_constant_prediction','n_genes_both_constant']]
    for prefix,data,gcol,metrics in [('l2',f2t,'model_key',['rsa','ndcg_mean','random_ndcg_mean','excess_ndcg_mean']),
                                   ('l3',f3t,'method',['mae_mean','gene_spearman_mean'])]:
        typed,macro=type_and_macro(data,gcol,metrics)
        frames[prefix+'_type_summary.tsv']=typed;frames[prefix+'_macro_summary.tsv']=macro
    for name,frame in frames.items():write_table(res/name,frame)
    write_table(out/'qa/numeric_preflight.tsv',pd.DataFrame(preflight))
    audit=dict(status='PRODUCER_COMPLETE_INDEPENDENT_QA_PENDING',n_tasks=6,n_primary_atoms=810,
               n_main_task_records=36,n_main_query_records=4860,n_baseline_task_records=42,
               n_total_l3_task_records=78,n_total_gene_records=234000,n_paired_baseline_records=252,
               new_encoding=False,new_training=False,raw_X_read=False,state_predictions_scored=False,
               effect_union_numeric_array_read=False,new_significance_tests=False,
               scoring_precision='original_float32_promoted_to_float64;frozen_helpers',
               zero_effect_policy='legacy caller fails affected task;no silent zero cosine',
               units_not_independent=True)
    write_json(out/'qa/producer_audit.json',audit)
    return audit
