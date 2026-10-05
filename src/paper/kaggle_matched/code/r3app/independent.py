"""Independent R3 score checker.
Reads original frozen inputs again. Does NOT import producer or either frozen
metric helper. IO/identity declarations are shared, numerical implementations
are separate: explicit average ranks, tie blocks, direct absolute-error means.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from .common import (MODEL_METHOD,MODEL_OF,BASELINES,METHODS,SELECTED,ATOL,GENE_MIN,
                     FrozenDataset,load_table,write_json,require)


def ranks(a):
    a=np.asarray(a,dtype=np.float64)
    flat=a.ndim==1
    if flat:a=a[:,None]
    n=a.shape[0]
    order=np.argsort(a,axis=0,kind='stable')
    values=np.take_along_axis(a,order,axis=0)
    pos=np.broadcast_to(np.arange(n)[:,None],a.shape)
    starts=np.ones(a.shape,bool);ends=np.ones(a.shape,bool)
    starts[1:]=values[1:]!=values[:-1];ends[:-1]=values[:-1]!=values[1:]
    lower=np.maximum.accumulate(np.where(starts,pos,0),axis=0)
    upper=np.minimum.accumulate(np.where(ends,pos,n-1)[::-1],axis=0)[::-1]
    out=np.empty(a.shape,float)
    np.put_along_axis(out,order,(lower+upper)*0.5+1,axis=0)
    return out[:,0] if flat else out


def correlations(a,b):
    ra=ranks(a);rb=ranks(b)
    ra=ra-np.mean(ra,axis=0);rb=rb-np.mean(rb,axis=0)
    num=np.einsum('ij,ij->j',ra,rb) if ra.ndim==2 else np.dot(ra,rb)
    den=np.sqrt(np.sum(ra*ra,axis=0)*np.sum(rb*rb,axis=0))
    out=np.full(np.shape(num),np.nan,dtype=float)
    np.divide(num,den,out=out,where=den>0)
    return np.clip(out,-1,1)


def cosine(a):
    a=np.asarray(a,np.float64)
    require(a.ndim==2 and np.isfinite(a).all(),'QA nonfinite/invalid cosine input')
    norm=np.sqrt(np.add.reduce(a*a,axis=1))
    require(np.all(norm>0),'QA zero cosine norm; no silent zero fill')
    unit=a/norm[:,None]
    return np.clip(np.matmul(unit,unit.T),-1,1)


def neighborhood(sim,truth,k=10):
    n=len(truth);nd=np.full(n,np.nan);rnd=np.full(n,np.nan);idcg=np.full(n,np.nan)
    st=np.full(n,'INSUFFICIENT_CANDIDATES',dtype='U32');uniq=np.zeros(n,int)
    relevance=np.zeros((n,n),float)
    if n<12 or n-1<=k:return nd,rnd,idcg,st,uniq,relevance
    discounts=1/np.log2(np.arange(2,k+2))
    weights=np.zeros(n-1);weights[:k]=discounts
    for i in range(n):
        keep=np.arange(n)!=i;actual=truth[i,keep];pred=sim[i,keep]
        uniq[i]=np.unique(actual).size
        cutoff=np.sort(actual)[-k]
        rel=np.zeros(n-1);rel[actual>cutoff]=1
        eq=actual==cutoff;rel[eq]=(k-np.count_nonzero(actual>cutoff))/np.count_nonzero(eq)
        relevance[i,keep]=rel
        if uniq[i]==1:
            st[i]='UNINFORMATIVE_TRUTH';continue
        best=np.sort(rel)[::-1][:k]
        idcg[i]=np.sum(best*discounts)
        rnd[i]=(np.sum(rel)/(n-1))*np.sum(discounts)/idcg[i]
        # Groups in ascending score order; high-score block starts at count
        # of strictly larger scores. Whole ties contribute, even past rank k.
        _,inverse,counts=np.unique(pred,return_inverse=True,return_counts=True)
        sums=np.bincount(inverse,weights=rel,minlength=len(counts))
        greater=(n-1)-np.cumsum(counts)
        cumulative=np.r_[0.0,np.cumsum(weights)]
        group_weight=cumulative[greater+counts]-cumulative[greater]
        nd[i]=np.dot(sums/counts,group_weight)/idcg[i]
        st[i]='VALID'
    return nd,rnd,idcg,st,uniq,relevance


def l3(pred,y):
    p=np.asarray(pred,np.float64);y=np.asarray(y,np.float64)
    mae=np.mean(np.abs(p-y),axis=1)
    ct=np.all(y==y[0],axis=0);cp=np.all(p==p[0],axis=0)
    r=correlations(p,y)
    st=np.full(y.shape[1],'VALID',dtype='U40')
    st[ct]='CONSTANT_TRUTH';st[cp]='CONSTANT_PREDICTION';st[ct&cp]='CONSTANT_TRUTH_AND_PREDICTION'
    if len(y)<20:r[:]=np.nan;st[:]='INSUFFICIENT_COMPOUNDS'
    require(np.array_equal(np.isfinite(r),st=='VALID'),'QA unexplained gene NA')
    return mae,r,st,ct,cp


def avg(x):
    x=np.asarray(x,float); good=np.isfinite(x)
    return float(np.sum(x[good])/np.count_nonzero(good)) if good.any() else np.nan


def numbers(x):
    if isinstance(x,pd.Series):return pd.to_numeric(x.where(x!='NA',np.nan),errors='raise').to_numpy(float)
    if isinstance(x,str):return np.nan if x=='NA' else float(x)
    return np.asarray(x,float)


class Verify:
    def __init__(self):self.count=0;self.maximum={}
    def num(self,name,observed,expected):
        a=np.asarray(numbers(observed),float);b=np.asarray(expected,float)
        require(a.shape==b.shape,name+': shape differs')
        require(np.array_equal(np.isnan(a),np.isnan(b)),name+': NA masks differ')
        require(np.allclose(a,b,atol=ATOL,rtol=0,equal_nan=True),name+': numeric mismatch')
        good=np.isfinite(a)&np.isfinite(b)
        diff=float(np.max(np.abs(a[good]-b[good]))) if good.any() else 0.0
        self.maximum[name]=max(self.maximum.get(name,0.0),diff);self.count+=1
    def exact(self,name,a,b):
        require(np.array_equal(np.asarray(a).astype(str),np.asarray(b).astype(str)),name+': exact values differ')
        self.count+=1


def run(root,out,spec):
    out=Path(out);res=out/'results';v=Verify()
    # Fresh reads of the frozen input, not the producer's geometry/metrics.
    ds=FrozenDataset(root,spec)
    names=['l2_query_metrics.tsv','l2_task_metrics.tsv','l3_query_metrics.tsv','l3_task_metrics.tsv',
           'l3_gene_drug_spearman.tsv.gz','l2_l3_query_join.tsv','l2_l3_task_join.tsv',
           'baseline_paired_differences.tsv','support_summary.tsv','method_role_map.tsv',
           'task_cohort_manifest.tsv','cohort_membership.tsv','gene_axis_manifest.tsv','representation_binding.tsv','na_and_exclusions.tsv']
    f={n:load_table(res/n) for n in names}
    expected_rows={'l2_query_metrics.tsv':4860,'l2_task_metrics.tsv':36,'l3_query_metrics.tsv':10530,
        'l3_task_metrics.tsv':78,'l3_gene_drug_spearman.tsv.gz':234000,'l2_l3_query_join.tsv':4860,
        'l2_l3_task_join.tsv':36,'baseline_paired_differences.tsv':252,'support_summary.tsv':78,
        'method_role_map.tsv':22,'task_cohort_manifest.tsv':6,'cohort_membership.tsv':816,
        'gene_axis_manifest.tsv':18000,'representation_binding.tsv':4860}
    for name,count in expected_rows.items():require(len(f[name])==count,'QA row count: '+name)
    for name in ['l2_query_metrics.tsv','l3_query_metrics.tsv','l2_l3_query_join.tsv']:
        require(not f[name].duplicated(['task_id','method','atomic_id']).any(),'QA duplicate query join')
    for name in ['l2_task_metrics.tsv','l3_task_metrics.tsv','l2_l3_task_join.tsv']:
        require(not f[name].duplicated(['task_id','method']).any(),'QA duplicate task join')
    require(not f['l3_gene_drug_spearman.tsv.gz'].duplicated(['task_id','method','source_feature_row']).any(),
            'QA duplicate gene rows')
    role=f['method_role_map.tsv']
    v.exact('method_axis',role.method,METHODS)
    v.exact('selected_methods',role.loc[role.l3_included=='True','method'],SELECTED)
    v.exact('direct_methods',role.loc[role.l2_included=='True','method'],list(MODEL_METHOD.values()))
    counts={'tasks':0,'main_query':0,'l3_query':0,'gene_scores':0,'main_task':0,'baseline_task':0}
    typed_inputs={'l2':[],'l3':[]}
    expected_na=[]
    for task_id in ds.tasks.task_id:
        print('R3_INDEPENDENT_QA_TASK',task_id,flush=True)
        t=ds.task(task_id);qm=t['qm'];n=len(qm);y=t['truth'].astype(float)
        ids=qm.atomic_id.to_numpy(str);truth=cosine(y);up=np.triu_indices(n,1)
        for ex in t['excluded'].itertuples():
            expected_na.append((task_id,'CONDITION','',ex.atomic_id,'BELINOSTAT_0_1uM_NOT_PRIMARY',1))
        sf={name:df[df.task_id==task_id].reset_index(drop=True) for name,df in f.items() if 'task_id' in df.columns}
        v.exact('cohort_ids',sf['cohort_membership.tsv'].atomic_id,t['q'].atomic_id)
        v.exact('primary_membership',sf['cohort_membership.tsv'].included_primary,t['keep'])
        v.exact('gene_axis_features',sf['gene_axis_manifest.tsv'].source_feature_row,t['features'])
        v.exact('gene_axis_ids',sf['gene_axis_manifest.tsv'].source_gene_id,t['genes'])
        cohort=sf['task_cohort_manifest.tsv'].iloc[0]
        v.num('cohort_size',cohort.n_query_primary,n);v.exact('cohort_hash',cohort.cohort_id,t['cohort_id'])
        for name,frame in sf.items():
            if 'gene_axis_id' in frame:
                v.exact('output_gene_axis_binding',frame.gene_axis_id.to_numpy(),np.repeat(t['gene_axis_id'],len(frame)))
            if 'cohort_id' in frame:
                v.exact('output_cohort_binding',frame.cohort_id.to_numpy(),np.repeat(t['cohort_id'],len(frame)))
        # A source-derived baseline consistency check is evaluation QA only;
        # it does not refit a model or replace its saved predictions.
        sy=t['source_y'].astype(float)
        for method in ['zero','source_mean','source_median','same_drug_source_mean']:
            p=t['predictions'][METHODS.index(method)]
            if method=='zero':expected=np.zeros_like(p)
            elif method=='source_mean':expected=np.broadcast_to(sy.mean(axis=0).astype(np.float32),p.shape)
            elif method=='source_median':expected=np.broadcast_to(np.median(sy,axis=0).astype(np.float32),p.shape)
            else:
                expected=[]
                for r in qm.itertuples():
                    match=(t['s'].sm_lincs_id==r.sm_lincs_id)&(t['s'].dose_key==r.dose_key)&(t['s'].time_key==r.time_key)
                    require(match.sum()==2,'QA same-drug source cardinality')
                    expected.append(sy[match.to_numpy()].mean(axis=0).astype(np.float32))
                expected=np.asarray(expected)
            v.num('frozen_reference_source_consistency',p,expected)
        maes={};genescores={}
        for method in SELECTED:
            p=t['predictions'][METHODS.index(method)].astype(float)
            mae,r,st,ct,cp=l3(p,y);maes[method]=mae;genescores[method]=r
            for reason in np.unique(st[st!='VALID']):
                expected_na.append((task_id,'L3_GENE',method,'ALL',str(reason),int(np.sum(st==reason))))
            oq=sf['l3_query_metrics.tsv'];oq=oq[oq.method==method].reset_index(drop=True)
            og=sf['l3_gene_drug_spearman.tsv.gz'];og=og[og.method==method].reset_index(drop=True)
            ot=sf['l3_task_metrics.tsv'];ot=ot[ot.method==method].iloc[0]
            v.exact('l3_query_ids',oq.atomic_id,ids);v.num('query_MAE',oq.mae,mae)
            v.exact('l3_gene_order',og.source_feature_row,t['features']);v.exact('l3_gene_ids',og.source_gene_id,t['genes'])
            v.num('all_gene_Spearman',og.gene_drug_spearman,r);v.exact('gene_NA_reason',og.status,st)
            v.exact('gene_constant_truth',og.constant_truth,ct);v.exact('gene_constant_prediction',og.constant_prediction,cp)
            v.exact('gene_validity',og.spearman_valid,np.isfinite(r))
            v.num('gene_effective_n',og.spearman_n,np.where(np.isfinite(r),n,0))
            v.num('gene_nominal_n',og.n_conditions,np.repeat(n,3000))
            v.num('task_MAE',ot.mae_mean,avg(mae));v.num('task_gene_Spearman',ot.gene_spearman_mean,avg(r))
            v.num('valid_gene_count',ot.n_genes_valid,np.isfinite(r).sum())
            v.num('NA_gene_count',ot.n_genes_na,(~np.isfinite(r)).sum())
            v.num('constant_truth_count',ot.n_genes_constant_truth,ct.sum())
            v.num('constant_prediction_count',ot.n_genes_constant_prediction,cp.sum())
            v.num('both_constant_count',ot.n_genes_both_constant,(ct&cp).sum())
            os=sf['support_summary.tsv'];os=os[os.method==method].iloc[0]
            for col in ['n_query_atoms','n_genes_total','n_genes_valid','n_genes_na','n_genes_constant_truth','n_genes_constant_prediction','n_genes_both_constant']:
                v.exact('support_'+col,os[col],ot[col])
            # Condition and gene permutations preserve the corresponding scores.
            maep,rp,_,_,_=l3(p[::-1,::-1],y[::-1,::-1])
            v.num('MAE_reorder_invariance',maep,mae[::-1]);v.num('Spearman_reorder_invariance',rp,r[::-1])
            typed_inputs['l3'].append(dict(cell_type=t['cell_type'],method=method,mae_mean=avg(mae),gene_spearman_mean=avg(r)))
            counts['l3_query']+=n;counts['gene_scores']+=3000
            counts['baseline_task']+=int(method in BASELINES)
        with np.load(out/'geometry'/(task_id+'.npz'),allow_pickle=False) as z:
            v.exact('geometry_atom_axis',z['atomic_id'],ids)
            v.num('truth_cosine_from_original',z['truth_cosine'],truth)
            for model,method in MODEL_METHOD.items():
                sim=cosine(t['embeddings'][model]);rsa=correlations(sim[up],truth[up]).item()
                nd,rnd,idcg,status,unique,rel=neighborhood(sim,truth)
                excess=nd-rnd
                for reason in np.unique(status[status!='VALID']):
                    expected_na.append((task_id,'L2_QUERY',method,'ALL',str(reason),int(np.sum(status==reason))))
                if not np.isfinite(rsa):
                    expected_na.append((task_id,'RSA',method,'TASK','CONSTANT_PAIR_GEOMETRY',1))
                oq=sf['l2_query_metrics.tsv'];oq=oq[oq.method==method].reset_index(drop=True)
                ot=sf['l2_task_metrics.tsv'];ot=ot[ot.method==method].iloc[0]
                jq=sf['l2_l3_query_join.tsv'];jq=jq[jq.method==method].reset_index(drop=True)
                jt=sf['l2_l3_task_join.tsv'];jt=jt[jt.method==method].iloc[0]
                v.exact('L2_atom_axis',oq.atomic_id,ids);v.exact('query_join_axis',jq.atomic_id,ids)
                for col,expect in [('ndcg',nd),('random_ndcg',rnd),('excess_ndcg',excess)]:
                    v.num('query_'+col,oq[col],expect);v.num('joined_'+col,jq[col],expect)
                v.num('joined_MAE',jq.mae,maes[method]);v.exact('L2_status',oq.l2_status,status)
                v.num('candidate_count',oq.n_candidates,np.repeat(n-1,n));v.num('truth_unique_scores',oq.truth_unique_scores,unique)
                values={'rsa':rsa,'ndcg_mean':avg(nd),'random_ndcg_mean':avg(rnd),'excess_ndcg_mean':avg(excess)}
                for col,val in values.items():v.num('task_'+col,ot[col],val);v.num('joined_task_'+col,jt[col],val)
                v.num('joined_task_mae',jt.mae_mean,avg(maes[method]));v.num('joined_task_gene',jt.gene_spearman_mean,avg(genescores[method]))
                v.num('L2_valid_count',ot.n_l2_valid,np.isfinite(nd).sum())
                v.num('L2_NA_count',ot.n_l2_na,(~np.isfinite(nd)).sum())
                v.num('geometry_repr_from_original',z['repr__'+model],sim)
                v.num('truth_relevance',z['truth_relevance'],rel);v.num('truth_IDCG',z['idcg'],idcg)
                v.num('truth_random',z['random_ndcg'],rnd);v.exact('truth_status',z['truth_status'],status)
                v.num('cosine_row_permutation',cosine(t['embeddings'][model][::-1]),sim[::-1,::-1])
                pn,pr,_,ps,_,_=neighborhood(sim[::-1,::-1],truth[::-1,::-1])
                v.num('NDCG_reindex_invariance',pn,nd[::-1]);v.num('random_reindex_invariance',pr,rnd[::-1])
                rb=sf['representation_binding.tsv'];rb=rb[rb.model_key==model].reset_index(drop=True)
                reg=ds.reg_lookup.loc[ids]
                v.exact('representation_atom_axis',rb.atomic_id,ids)
                for col in ['text_id','prompt_sha256','text_row']:
                    v.exact('representation_'+col,rb[col],reg[col].to_numpy())
                counts['main_query']+=n;counts['main_task']+=1
                typed_inputs['l2'].append(dict(cell_type=t['cell_type'],model_key=model,**values))
        diff=sf['baseline_paired_differences.tsv']
        for model,method in MODEL_METHOD.items():
            a=genescores[method];av=np.isfinite(a)
            for baseline in BASELINES:
                b=genescores[baseline];bv=np.isfinite(b);common=av&bv
                row=diff[(diff.method==method)&(diff.baseline==baseline)].iloc[0]
                expected={'model_mae':avg(maes[method]),'baseline_mae':avg(maes[baseline]),
                    'mae_gain':avg(maes[baseline]-maes[method]),
                    'model_gene_spearman_original_mean':avg(a),'baseline_gene_spearman_original_mean':avg(b),
                    'model_n_genes_valid':av.sum(),'baseline_n_genes_valid':bv.sum(),
                    'model_n_genes_na':(~av).sum(),'baseline_n_genes_na':(~bv).sum(),
                    'n_common_valid_genes':common.sum(),'model_common_gene_spearman_mean':avg(a[common]),
                    'baseline_common_gene_spearman_mean':avg(b[common]),'gene_spearman_gain':avg((a-b)[common])}
                for col,val in expected.items():v.num('paired_'+col,row[col],val)
                same=np.array_equal(av,bv)
                v.exact('paired_same_support',row.same_valid_gene_set,same)
                scope=('ORIGINAL_VALID_SET' if same else 'COMMON_VALID_GENE_SUBSET') if common.any() else 'NO_COMMON_VALID_GENES'
                v.exact('paired_rank_scope',row.rank_gain_scope,scope)
        counts['tasks']+=1
    # NA/exclusion counts are reconstructed from original input, not accepted on trust.
    na_records=[]
    for row in f['na_and_exclusions.tsv'].itertuples():
        na_records.append((row.task_id,row.level,row.method,row.unit_id,row.reason,int(row.n_units)))
    require(sorted(na_records)==sorted(expected_na),'NA/exclusion ledger differs from original inputs')
    # Independently reconstruct the specified hierarchical descriptive summaries.
    for prefix,groupcol,metrics in [('l2','model_key',['rsa','ndcg_mean','random_ndcg_mean','excess_ndcg_mean']),
                                     ('l3','method',['mae_mean','gene_spearman_mean'])]:
        dat=pd.DataFrame(typed_inputs[prefix]);typed=load_table(res/(prefix+'_type_summary.tsv'));macro=load_table(res/(prefix+'_macro_summary.tsv'))
        for method in dat[groupcol].unique():
            types={}
            for ct in ['NK cells','T cells CD4+']:
                sub=dat[(dat[groupcol]==method)&(dat.cell_type==ct)]
                rt=typed[(typed[groupcol]==method)&(typed.cell_type==ct)].iloc[0]
                types[ct]={}
                for metric in metrics:
                    val=avg(sub[metric]);types[ct][metric]=val
                    v.num('type_'+metric,rt[metric],val)
                    v.num('type_valid_'+metric,rt[metric+'__valid_tasks'],np.isfinite(sub[metric]).sum())
            rm=macro[macro[groupcol]==method].iloc[0]
            for metric in metrics:
                vals=[types[ct][metric] for ct in types]
                v.num('macro_'+metric,rm[metric],avg(vals))
                v.num('macro_valid_'+metric,rm[metric+'__valid_types'],np.isfinite(vals).sum())
    expected={'tasks':6,'main_query':4860,'l3_query':10530,'gene_scores':234000,'main_task':36,'baseline_task':42}
    require(counts==expected,'Independent coverage incomplete')
    report=dict(status='PASS',scope='fresh_original_inputs_to_all_selected_R3_scores;independent_numeric_implementation',
        counts=counts,numeric_and_identity_checks=v.count,atol=ATOL,rtol=0,maximum_absolute_differences=v.maximum,
        shared_components='declarative method mapping and frozen-input IO validators only',
        not_imported=['producer.py','frozen_geometry.py','frozen_prediction.py'],
        upstream_raw_cells_reaggregated=False,old_predictors_retrained=False,
        limitations=['Existing upstream QA is reused, not a new full validation of all biological assumptions.',
                     'Nine state prediction slices exist in the compressed file but are not scored.',
                     'No p-values, bootstrap, causal explanation, noise ceiling, or cross-layer total score.'])
    write_json(out/'qa/independent_r3_audit.json',report)
    return report
