#!/usr/bin/env python3
"""Independent all-group R2 verification; never imports producer/scoring helpers."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
for _thread_variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_thread_variable] = '1'
import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
QA=ROOT/'qa'
import os
A=Path(os.environ['PCA_DEEP_ROOT'])
E=A/'effects/atomic_effects_v1'
ALIGN=A/'metrics/effect_alignment_v1'
LINES=('A549','K562','MCF7')
DOSES=((10,188),(100,187),(1000,187),(10000,188))
REPS=('rep1','rep2')
MODELS=('bge_m3','sapbert','qwen3_0_6b','biomedbert','medcpt_article','medcpt_query')
BASELINES=('tfidf','structured_onehot','random_field512')
ATOL=1e-10
READS={}
COUNTS=Counter()
MAX_DRIFT=defaultdict(float)


def ensure(value,message):
    if not value:raise AssertionError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def track(path):
    path=Path(path)
    # No helper/producer source files are read by this numerical checker.
    ensure(path.suffix!='.py' or path==Path(__file__),'Forbidden external Python source read')
    READS.setdefault(str(path),sha(path))
    return path


def table(path):
    with track(path).open(newline='') as stream:return list(csv.DictReader(stream,delimiter='\t'))


def payload(path):return json.loads(track(path).read_text())


def scalar(value):
    if value is None or str(value) in ('','NA','NaN','nan','None'):return None
    result=float(value)
    ensure(math.isfinite(result),'Unexpected infinite/nonfinite scalar')
    return result


def close(actual,expected,label):
    a,b=scalar(actual),scalar(expected)
    if a is None or b is None:ensure(a is b,f'{label}: NA mismatch')
    else:
        delta=abs(a-b)
        ensure(delta<=ATOL,f'{label}: drift {delta:.17g}, actual={a:.17g}, expected={b:.17g}')
        MAX_DRIFT[label.split('/')[0]]=max(MAX_DRIFT[label.split('/')[0]],delta)
    COUNTS['scalar_checks']+=1


def arrays_close(actual,expected,label):
    actual=np.asarray(actual);expected=np.asarray(expected)
    ensure(actual.shape==expected.shape,f'{label}: shape mismatch')
    ensure(np.isfinite(actual).all() and np.isfinite(expected).all(),f'{label}: nonfinite array')
    delta=float(np.max(np.abs(actual-expected))) if actual.size else 0.0
    ensure(delta<=ATOL,f'{label}: array drift {delta:.17g}')
    MAX_DRIFT[label.split('/')[0]]=max(MAX_DRIFT[label.split('/')[0]],delta)
    COUNTS['array_values_compared']+=actual.size


def sequence_hash(values):return hashlib.sha256(('\n'.join(map(str,values))+'\n').encode()).hexdigest()


def unique(rows,keys,label):
    out={}
    for row in rows:
        key=tuple(row[k] for k in keys)
        ensure(key not in out,f'{label}: duplicate key {key}')
        out[key]=row
    return out


def cosine_independent(values):
    values=np.asarray(values,dtype=np.float64)
    ensure(values.ndim==2 and np.isfinite(values).all(),'Invalid effect matrix')
    squared=np.einsum('ij,ij->i',values,values,optimize=False)
    ensure((squared>0).all(),'ZERO_NORM_EFFECT: stop group, no substitution/exclusion')
    norm=np.sqrt(squared)
    dots=np.einsum('ik,jk->ij',values,values,optimize=False)
    result=np.clip(dots/(norm[:,None]*norm[None,:]),-1.0,1.0)
    ensure(np.isfinite(result).all(),'Nonfinite independent cosine')
    arrays_close(result,result.T,'cosine_symmetry')
    return result


def average_ranks(values):
    vals=list(map(float,values))
    ensure(all(map(math.isfinite,vals)),'Nonfinite rank input')
    order=sorted(range(len(vals)),key=lambda i:vals[i])
    ranks=[0.0]*len(vals);start=0
    while start<len(order):
        end=start+1
        while end<len(order) and vals[order[end]]==vals[order[start]]:end+=1
        rank=(start+end-1)/2+1
        for i in order[start:end]:ranks[i]=rank
        start=end
    return ranks


def rank_correlation_independent(first,second):
    ensure(len(first)==len(second) and len(first)>=2,'Invalid pair arrays')
    a,b=average_ranks(first),average_ranks(second)
    center=(len(a)+1)/2
    a=[v-center for v in a];b=[v-center for v in b]
    va=math.fsum(v*v for v in a);vb=math.fsum(v*v for v in b)
    if va==0 or vb==0:return None
    return max(-1.0,min(1.0,math.fsum(x*y for x,y in zip(a,b))/math.sqrt(va*vb)))


def score_blocks(scores):
    scores=list(map(float,scores))
    ensure(all(map(math.isfinite,scores)),'Nonfinite candidate score')
    order=sorted(range(len(scores)),key=lambda i:-scores[i])
    blocks=[];start=0
    while start<len(order):
        end=start+1
        while end<len(order) and scores[order[end]]==scores[order[start]]:end+=1
        blocks.append((start,end,order[start:end]))
        start=end
    return blocks


def query_independent(ranking,truth,k=10):
    ranking=list(map(float,ranking));truth=list(map(float,truth))
    ensure(len(ranking)==len(truth),'Candidate axes differ')
    n=len(truth)
    result=dict(n_candidates=n,status='INSUFFICIENT_CANDIDATES',ndcg=None,random_ndcg=None,excess_ndcg=None,
                idcg=None,relevance_sum=0.0,n_unique_truth_scores=0,truth_boundary_tie_size=0,
                truth_boundary_relevance=None,prediction_boundary_tie_size=0)
    if n+1<12 or n<=k:return result,np.zeros(n)
    truth_blocks=score_blocks(truth);ranking_blocks=score_blocks(ranking)
    relevance=[0.0]*n
    for start,end,members in truth_blocks:
        if start>=k:break
        mass=min(k,end)-start
        value=mass/len(members)
        for i in members:relevance[i]=value
        if start<k<=end:
            result['truth_boundary_tie_size']=len(members)
            result['truth_boundary_relevance']=value
    result['relevance_sum']=math.fsum(relevance)
    ensure(abs(result['relevance_sum']-k)<=1e-12,'Top-k relevance mass')
    result['n_unique_truth_scores']=len(truth_blocks)
    for start,end,members in ranking_blocks:
        if start<k<=end:result['prediction_boundary_tie_size']=len(members)
    if len(truth_blocks)==1:
        result['status']='UNINFORMATIVE_TRUTH'
        return result,np.asarray(relevance)
    discount=[1/math.log2(rank+2) for rank in range(k)]
    ideal=math.fsum(v*w for v,w in zip(sorted(relevance,reverse=True)[:k],discount))
    dcg_terms=[]
    for start,end,members in ranking_blocks:
        if start>=k:break
        gain=math.fsum(relevance[i] for i in members)/len(members)
        dcg_terms.append(gain*math.fsum(discount[start:min(end,k)]))
    ndcg=math.fsum(dcg_terms)/ideal
    random=(math.fsum(relevance)/n)*math.fsum(discount)/ideal
    result.update(status='VALID',idcg=ideal,ndcg=ndcg,random_ndcg=random,excess_ndcg=ndcg-random)
    return result,np.asarray(relevance)


def neighbor_matrix_independent(ranking,truth):
    ranking=np.asarray(ranking);truth=np.asarray(truth)
    n=len(truth)
    ensure(ranking.shape==truth.shape==(n,n),'Neighbor matrix shape')
    rows=[]
    for query in range(n):
        others=[i for i in range(n) if i!=query]
        row,_=query_independent(ranking[query,others],truth[query,others])
        rows.append(row)
    return rows


def fixtures():
    reports=[]
    scores=list(range(12,0,-1))
    result,_=query_independent(scores,scores)
    close(result['ndcg'],1,'fixture_correct_ranking')
    reports.append(dict(fixture='no_ties_correct_ranking',status='PASS'))
    result,_=query_independent([1]*12,scores)
    close(result['ndcg'],result['random_ndcg'],'fixture_prediction_all_tied')
    close(result['excess_ndcg'],0,'fixture_prediction_all_tied_excess')
    reports.append(dict(fixture='all_predicted_tied_random',status='PASS'))
    result,_=query_independent(scores,[1]*12)
    ensure(result['status']=='UNINFORMATIVE_TRUTH' and all(result[k] is None for k in ('ndcg','random_ndcg','excess_ndcg')),'All-truth-tied NA fixture')
    reports.append(dict(fixture='all_truth_tied_NA',status='PASS'))
    boundary=list(range(9,0,-1))+[0,0,0]
    result,relevance=query_independent(boundary,boundary)
    arrays_close(relevance,np.asarray([1]*9+[1/3]*3),'fixture_boundary_relevance')
    close(result['ndcg'],1,'fixture_boundary_correct_ranking')
    predicted=[2]*9+[1]*3
    result,_=query_independent(predicted,scores)
    weights=[1/math.log2(i+2) for i in range(10)]
    close(result['ndcg'],(math.fsum(weights[:9])+weights[9]/3)/math.fsum(weights),'fixture_prediction_boundary_block')
    reports.append(dict(fixture='boundary_tie_fraction',status='PASS'))
    result,_=query_independent(list(range(10)),list(range(10)))
    ensure(result['status']=='INSUFFICIENT_CANDIDATES' and result['ndcg'] is None,'Insufficient-candidate fixture')
    reports.append(dict(fixture='insufficient_candidates',status='PASS'))
    rng=np.random.default_rng(20260925)
    first=rng.normal(size=(14,9));second=rng.normal(size=(14,9))
    order=np.asarray([3,0,12,5,1,8,13,6,11,4,2,10,7,9]);gene=np.arange(8,-1,-1)
    a,b=cosine_independent(first),cosine_independent(second)
    c,d=cosine_independent(first[np.ix_(order,gene)]),cosine_independent(second[np.ix_(order,gene)])
    arrays_close(c,a[np.ix_(order,order)],'fixture_joint_permutation_cosine_1')
    arrays_close(d,b[np.ix_(order,order)],'fixture_joint_permutation_cosine_2')
    tri=np.triu_indices(14,1)
    close(rank_correlation_independent(a[tri],b[tri]),rank_correlation_independent(c[tri],d[tri]),'fixture_joint_permutation_rsa')
    before=neighbor_matrix_independent(a,b);after=neighbor_matrix_independent(c,d)
    for i,old in enumerate(order):
        ensure(before[old]['status']==after[i]['status'],'Permutation query status')
        for key in ('ndcg','random_ndcg','excess_ndcg'):close(before[old][key],after[i][key],'fixture_joint_permutation_'+key)
    reports.append(dict(fixture='joint_condition_and_gene_permutation',status='PASS'))
    return reports


def load_sources():
    contract=payload(ROOT/'execution_contract.json')
    ensure(contract['qa_atol']==ATOL and contract['neighbor_k']==10 and contract['minimum_group_n']==12,'Contract metric parameters changed')
    audit=payload(E/'audit.json')
    paths=['atomic_index.tsv','measurement_index.tsv','fold_gene_panels.tsv','control_partition.tsv','measurement_means.npz']
    for name in paths:
        path=track(E/name)
        ensure(READS[str(path)]==audit['output_sha256'][name],f'Frozen source checksum: {name}')
    atoms=table(E/'atomic_index.tsv');measurements=table(E/'measurement_index.tsv')
    controls=table(E/'control_partition.tsv');panels=table(E/'fold_gene_panels.tsv')
    atom_map=unique(atoms,('atomic_id',),'Atomic source');measurement_map=unique(measurements,('measurement_id',),'Measurements')
    pair_map=unique(measurements,('atomic_id','replicate'),'Atom-replicate source')
    ensure(len(atom_map)==2256 and len(measurement_map)==4512,'Source dimensions')
    ensure(set(pair_map)=={(r['atomic_id'],rep) for r in atoms for rep in REPS},'Exactly two source replicates')
    primary=[r for r in atoms if r['main_eligible']=='True']
    old_primary=table(ALIGN/'atomic_index.tsv')
    ensure([r['atomic_id'] for r in primary]==[r['atomic_id'] for r in old_primary] and len(primary)==2250,'Primary ID order')
    excluded=[r for r in atoms if r['main_eligible']!='True']
    ensure(len(excluded)==6 and {r['source_entity_key'] for r in excluded}=={'localentity:1ee163940bc4f2734d95d8de'},'Six original YM155 exclusions')
    ensure({(r['cell_line'],int(r['dose_value'])) for r in excluded}=={(line,dose) for line in LINES for dose in (100,1000)},'YM155 excluded cells/doses')
    control_map={r['control_group']:r for r in controls}
    ensure(len(control_map)==96,'96 control wells')
    B_by_rep=defaultdict(set)
    for measurement in measurements:
        control=control_map[measurement['control_B_group']]
        ensure(control['control_role']=='B' and all(control[k]==measurement[k] for k in ('cell_line','time','replicate','plate')),'Matched B control fields')
        B_by_rep[measurement['replicate']].add(measurement['control_B_group'])
    ensure(len(B_by_rep['rep1'])==len(B_by_rep['rep2'])==24 and not B_by_rep['rep1']&B_by_rep['rep2'],'Disjoint replicate B pools')
    cache=E/'measurement_means.npz'
    with np.load(cache,allow_pickle=False) as source:
        ids=source['measurement_id'];features=source['source_feature_row']
        treated=source['normalized_mean'];reference=source['control_B_mean']
    ensure(ids.tolist()==[r['measurement_id'] for r in measurements],'Measurement axis order')
    ensure(treated.dtype==reference.dtype==np.dtype('float32') and treated.shape==reference.shape==(4512,3730),'Mean cache dtype/shape')
    ensure(np.isfinite(treated).all() and np.isfinite(reference).all() and (treated>=0).all() and (reference>=0).all(),'Invalid source means; stop dependent analysis')
    ensure(len(set(features.tolist()))==3730,'Unique source gene axis')
    feature_map={int(value):i for i,value in enumerate(features)}
    row_map={(r['atomic_id'],r['replicate']):i for i,r in enumerate(measurements)}
    return dict(atoms=atoms,primary=primary,excluded=excluded,measurements=measurements,controls=controls,panels=panels,
                treated=treated,reference=reference,feature_map=feature_map,row_map=row_map,contract=contract)


def source_groups(source):
    for line in LINES:
        panel=sorted([r for r in source['panels'] if r['heldout_cell_line']==line],key=lambda r:int(r['rank']))
        genes=[int(r['source_feature_row']) for r in panel]
        ensure(len(genes)==len(set(genes))==3000,f'{line}: 3000 unique genes')
        columns=[source['feature_map'][gene] for gene in genes]
        for dose,n in DOSES:
            gid=f'{line}__dose_{dose}'
            positions=[i for i,r in enumerate(source['primary']) if r['cell_line']==line and int(r['dose_value'])==dose]
            atoms=[source['primary'][i] for i in positions];ids=[r['atomic_id'] for r in atoms]
            ensure(len(ids)==n,f'{gid}: original group size')
            old=track(ALIGN/'groups'/gid/'truth_geometry.npz')
            with np.load(old,allow_pickle=False) as saved:
                ensure(saved['atomic_id'].tolist()==ids and saved['source_feature_row'].tolist()==genes and saved['primary_rows'].tolist()==positions,f'{gid}: old ID/gene/row axis')
            effects={};similarities={};rows={}
            for rep in REPS:
                selected=[source['row_map'][aid,rep] for aid in ids]
                treated=source['treated'][np.ix_(selected,columns)].astype(np.float64)
                reference=source['reference'][np.ix_(selected,columns)].astype(np.float64)
                effects[rep]=np.log2((treated+1.0)/(reference+1.0))
                ensure(np.isfinite(effects[rep]).all(),f'{gid}/{rep}: finite effect')
                similarities[rep]=cosine_independent(effects[rep])
                rows[rep]=selected
            tri=np.triu_indices(n,1)
            rho=rank_correlation_independent(similarities['rep1'][tri],similarities['rep2'][tri])
            queries=[]
            for ranking,truth in (('rep1','rep2'),('rep2','rep1')):
                direction=ranking+'_to_'+truth
                scores=neighbor_matrix_independent(similarities[ranking],similarities[truth])
                for i,score in enumerate(scores):
                    queries.append(dict(group_id=gid,cell_line=line,dose_value=dose,direction=direction,ranking_replicate=ranking,
                                        truth_replicate=truth,atomic_id=ids[i],source_entity_key=atoms[i]['source_entity_key'],**score))
            yield dict(group_id=gid,cell_line=line,dose_value=dose,ids=ids,genes=genes,primary_rows=positions,
                       effects=effects,similarities=similarities,measurement_rows=rows,geometry_spearman=rho,queries=queries,
                       cohort_sha256=sequence_hash(ids),gene_axis_sha256=sequence_hash(genes))


def write_rows(path,rows):
    ensure(path.parent==QA and not path.exists(),'QA output must be new and local')
    with path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),delimiter='\t')
        writer.writeheader()
        for row in rows:writer.writerow({k:'NA' if v is None else v for k,v in row.items()})


def verify_production(source):
    result=ROOT/'results'
    geom_rows=table(result/'replicate_geometry_consistency.tsv')
    query_rows=table(result/'replicate_neighbor_query_metrics.tsv')
    summary_rows=table(result/'replicate_neighbor_group_summary.tsv')
    manifest_rows=table(result/'group_axis_manifest.tsv')
    geometries=unique(geom_rows,('group_id',),'Geometry results')
    queries=unique(query_rows,('group_id','direction','atomic_id'),'Query results')
    summaries=unique(summary_rows,('group_id','direction'),'Directional summaries')
    group_manifest=unique(manifest_rows,('group_id',),'Group manifest')
    expected_gids={f'{line}__dose_{dose}' for line in LINES for dose,_ in DOSES}
    ensure(set(geometries)==set(group_manifest)=={(g,) for g in expected_gids},'All twelve groups retained')
    ensure(len(queries)==4500 and len(summaries)==24,'Full query/direction counts')
    expected_geometry=[];expected_queries=[];expected_summaries=[];matrix_checks=[];identities={}
    for group in source_groups(source):
        gid=group['group_id'];n=len(group['ids']);pairs=n*(n-1)//2
        common=dict(group_id=gid,cell_line=group['cell_line'],dose_value=group['dose_value'],n_atoms=n,n_genes=3000,
                    cohort_sha256=group['cohort_sha256'],gene_axis_sha256=group['gene_axis_sha256'])
        identities[gid]=common
        for row in (geometries[gid,],group_manifest[gid,]):
            for key,value in common.items():ensure(str(row[key])==str(value),f'{gid}: common identity {key}')
            ensure(row['effect_definition_id']==source['contract']['effect_definition_id'] and int(row['n_independent_studies'])==1,f'{gid}: effect/study identity')
        old_path=ALIGN/'groups'/gid/'truth_geometry.npz'
        manifest=group_manifest[gid,]
        ensure(manifest['source_truth_geometry_path']==str(old_path) and manifest['source_truth_geometry_sha256']==READS[str(old_path)],f'{gid}: frozen truth reference identity')
        ensure(manifest['measurement_cache_path']==str(E/'measurement_means.npz') and manifest['measurement_cache_sha256']==READS[str(E/'measurement_means.npz')],f'{gid}: measurement cache identity')
        ensure(manifest['source_dtype']=='float32' and manifest['compute_dtype']=='float64',f'{gid}: precision declaration')
        saved_path=track(ROOT/'geometry'/(gid+'.npz'))
        with np.load(saved_path,allow_pickle=False) as saved:
            ensure(saved['atomic_id'].tolist()==group['ids'] and saved['source_feature_row'].tolist()==group['genes'] and saved['primary_rows'].tolist()==group['primary_rows'],f'{gid}: saved geometry axes')
            for rep in REPS:
                original=group['similarities'][rep]
                actual=saved[rep+'_cosine']
                arrays_close(actual,original,'full_cosine/'+gid+'/'+rep)
                ensure(np.all(actual>=-1) and np.all(actual<=1),f'{gid}/{rep}: cosine range')
                arrays_close(actual,actual.T,'saved_cosine_symmetry/'+gid+'/'+rep)
                independent_norm=np.sqrt(np.einsum('ij,ij->i',group['effects'][rep],group['effects'][rep],optimize=False))
                arrays_close(saved[rep+'_effect_norm'],independent_norm,'effect_norm/'+gid+'/'+rep)
                ensure(saved[rep+'_measurement_rows'].tolist()==group['measurement_rows'][rep],f'{gid}/{rep}: measurement rows')
                checksum=hashlib.sha256(np.ascontiguousarray(group['effects'][rep]).tobytes()).hexdigest()
                saved_hash=str(saved[rep+'_effect_sha256'].item())
                hash_equal=checksum==saved_hash
                COUNTS['effect_byte_hash_matches']+=int(hash_equal)
                matrix_checks.append(dict(group_id=gid,replicate=rep,n_atoms=n,n_genes=3000,
                                          n_cosine_values=n*n,max_abs_cosine_difference=float(np.max(np.abs(actual-original))),
                                          effect_norm_max_abs_difference=float(np.max(np.abs(saved[rep+'_effect_norm']-independent_norm))),
                                          independent_effect_sha256=checksum,saved_effect_sha256=saved_hash,effect_hash_equal=hash_equal))
        observed=geometries[gid,]
        close(observed['replicate_geometry_spearman'],group['geometry_spearman'],'geometry_spearman/'+gid)
        for field,value in [('n_atoms_nominal',n),('n_atoms_finite',n),('n_pairs_nominal',pairs),('n_pairs_finite',pairs),('n_zero_norm_rep1',0),('n_zero_norm_rep2',0)]:
            ensure(int(observed[field])==value,f'{gid}: support {field}')
        ensure((observed['rsa_status']=='VALID')==(group['geometry_spearman'] is not None),f'{gid}: RSA NA status')
        expected_geometry.append(dict(**common,replicate_geometry_spearman=group['geometry_spearman'],n_pairs=pairs))
        for expected in group['queries']:
            direction=expected['direction'];aid=expected['atomic_id']
            actual=queries[gid,direction,aid]
            for key in ('group_id','cell_line','dose_value','direction','ranking_replicate','truth_replicate','atomic_id','source_entity_key','status','n_candidates'):
                ensure(str(actual[key])==str(expected[key]),f'Query identity/status {gid}/{direction}/{aid}/{key}')
            for rep_field,measurement_field in [('ranking_replicate','ranking_measurement_id'),('truth_replicate','truth_measurement_id')]:
                source_row=source['row_map'][aid,expected[rep_field]]
                ensure(actual[measurement_field]==source['measurements'][source_row]['measurement_id'],f'{gid}/{direction}/{aid}: measurement direction')
            ensure(actual['cohort_sha256']==group['cohort_sha256'] and actual['gene_axis_sha256']==group['gene_axis_sha256'],'Query cohort/gene identity')
            for key in ('ndcg','random_ndcg','excess_ndcg','idcg','relevance_sum'):
                close(actual[key],expected[key],'query_'+key+'/'+gid+'/'+direction+'/'+aid)
            ensure(int(actual['n_unique_truth_scores'])==expected['n_unique_truth_scores'] and int(actual['truth_boundary_tie_n'])==expected['truth_boundary_tie_size'],f'{gid}/{direction}/{aid}: exact truth ties')
            ensure((actual['na_reason']=='')==(expected['status']=='VALID'),f'{gid}/{direction}/{aid}: explicit NA reason')
            expected_queries.append(expected)
        for ranking,truth in (('rep1','rep2'),('rep2','rep1')):
            direction=ranking+'_to_'+truth
            expected=[r for r in group['queries'] if r['direction']==direction]
            valid=[r for r in expected if r['status']=='VALID']
            actual=summaries[gid,direction]
            for key,value in common.items():ensure(str(actual[key])==str(value),f'{gid}/{direction}: summary identity {key}')
            ensure(actual['ranking_replicate']==ranking and actual['truth_replicate']==truth,'Summary direction reversed')
            for key,value in [('n_queries_nominal',n),('n_queries_valid',len(valid)),('n_queries_na',n-len(valid)),('n_candidates',n-1)]:
                ensure(int(actual[key])==value,f'{gid}/{direction}: summary denominator {key}')
            independent=dict(**common,direction=direction,n_queries_nominal=n,n_queries_valid=len(valid),n_queries_na=n-len(valid))
            for key in ('ndcg','random_ndcg','excess_ndcg'):
                value=math.fsum(r[key] for r in valid)/len(valid) if valid else None
                close(actual[key+'_mean'],value,'summary_'+key+'/'+gid+'/'+direction)
                independent[key+'_mean']=value
            counts=Counter('numeric_zero' if abs(r['excess_ndcg'])<=1e-12 else 'positive' if r['excess_ndcg']>0 else 'negative' for r in valid)
            for label in ('positive','negative','numeric_zero'):
                ensure(int(actual['n_'+label+'_excess'])==counts[label],f'{gid}/{direction}: descriptive query directions')
            ensure((actual['status']=='VALID')==(len(valid)==n),f'{gid}/{direction}: summary NA status')
            expected_summaries.append(independent)
        print('INDEPENDENT_R2_GROUP_PASS',gid,flush=True)
        COUNTS['complete_groups']+=1
    ensure(set(queries)=={(r['group_id'],r['direction'],r['atomic_id']) for r in expected_queries},'No omitted or additional query keys')
    check_metadata_and_controls(source)
    check_references(identities)
    check_input_identities()
    COUNTS.update(geometry_rows=12,directional_rows=24,query_rows=4500)
    return dict(geometry_reference=expected_geometry,query_reference=expected_queries,
                direction_reference=expected_summaries,matrix_checks=matrix_checks)


def check_metadata_and_controls(source):
    result=ROOT/'results'
    atoms={r['atomic_id']:r for r in source['atoms']};primary={r['atomic_id'] for r in source['primary']}
    members=unique(table(result/'condition_membership.tsv'),('atomic_id',),'Condition membership')
    ensure(set(members)=={(aid,) for aid in atoms},'All original conditions accounted')
    for (aid,),row in members.items():
        atom=atoms[aid]
        for key in ('source_entity_key','source_entity_name','cell_line','dose_value','dose_unit','time'):
            ensure(row[key]==atom[key],f'{aid}: membership metadata {key}')
        ensure(row['group_id']==f"{atom['cell_line']}__dose_{atom['dose_value']}" and row['included_primary']==str(aid in primary),f'{aid}: membership assignment')
        ensure((row['exclusion_reason']=='')==(aid in primary),f'{aid}: exclusion reason')
        for rep in REPS:
            ensure(row[rep+'_measurement_id']==source['measurements'][source['row_map'][aid,rep]]['measurement_id'],f'{aid}: replicate measurement identity')
    links=unique(table(result/'measurement_control_links.tsv'),('measurement_id',),'Measurement links')
    ensure(len(links)==4512 and set(links)=={(r['measurement_id'],) for r in source['measurements']},'All original measurement links')
    by_B=defaultdict(list);by_group=defaultdict(list)
    for i,measurement in enumerate(source['measurements']):
        row=links[measurement['measurement_id'],];atom=atoms[measurement['atomic_id']]
        for key in ('measurement_id','atomic_id','cell_line','replicate','plate','time','control_B_group','control_A_group','n_treated_cells'):
            ensure(row[key]==measurement[key],f'Measurement link {measurement["measurement_id"]}/{key}')
        ensure(int(row['measurement_row'])==i and row['source_entity_key']==atom['source_entity_key'] and row['dose_value']==atom['dose_value'],'Measurement row/entity/dose')
        ensure(row['included_primary']==str(measurement['atomic_id'] in primary),'Measurement primary membership')
        group_id=f"{atom['cell_line']}__dose_{atom['dose_value']}"
        ensure(row['group_id']==group_id,'Measurement group assignment')
        by_B[measurement['control_B_group']].append(i)
        if measurement['atomic_id'] in primary:
            by_group[group_id,measurement['replicate'],measurement['control_B_group']].append(measurement['atomic_id'])
    controls={r['control_group']:r for r in source['controls']}
    sharing=unique(table(result/'control_sharing_summary.tsv'),('control_B_group',),'Control sharing')
    ensure(set(sharing)=={(group,) for group in by_B} and len(sharing)==48,'All B-control sharing groups')
    for (key,),row in sharing.items():
        indices=by_B[key];measurements=[source['measurements'][i] for i in indices]
        selected=[m for m in measurements if m['atomic_id'] in primary]
        control=controls[key]
        for field in ('replicate','cell_line','time','plate'):ensure(row[field]==control[field],'Control metadata')
        ensure(row['control_well']==control['well'] and int(row['control_cells'])==int(control['n_cells']),'Control well/cell count')
        for field,value in [('n_measurements_all',len(measurements)),('n_atoms_all',len({m['atomic_id'] for m in measurements})),
                            ('n_measurements_primary',len(selected)),('n_atoms_primary',len({m['atomic_id'] for m in selected})),
                            ('n_dose_groups_primary',len({atoms[m['atomic_id']]['dose_value'] for m in selected}))]:
            ensure(int(row[field])==value,'Control reuse count '+field)
        ensure(np.array_equal(source['reference'][indices],np.broadcast_to(source['reference'][indices[0]],(len(indices),3730))),'Shared B vectors differ')
        ensure(row['replicate_B_sets_disjoint']=='True' and row['same_cached_B_vector']=='True' and row['independent_across_drugs']=='False','Shared-control interpretation flags')
    grouped=unique(table(result/'control_sharing_by_group.tsv'),('group_id','replicate','control_B_group'),'Grouped control sharing')
    ensure(set(grouped)==set(by_group),'Group/rep/control sharing keys')
    for key,row in grouped.items():
        ensure(int(row['n_measurements'])==len(by_group[key]) and int(row['n_atoms'])==len(set(by_group[key])),'Grouped control reuse counts')
    exclusions=table(result/'na_and_exclusions.tsv')
    excluded=[r for r in exclusions if r['scope']=='original_exclusion']
    ensure(len(excluded)==6 and {r['atomic_id'] for r in excluded}=={r['atomic_id'] for r in source['excluded']},'Six exact original exclusions')
    for row in excluded:ensure(row['reason']=='ORIGINAL_MAIN_ELIGIBLE_FALSE' and int(row['n_atoms_nominal'])==1,'Original exclusion reason/denominator')
    COUNTS.update(condition_membership_rows=2256,measurement_link_rows=4512,shared_B_groups=48,control_group_rows=len(grouped))


def check_references(identities):
    source_path=ALIGN/'group_summary.tsv'
    old=table(source_path)
    config=payload(ALIGN/'config.json')
    representation=unique(payload(ALIGN/'representation_manifest.json'),('method','view'),'Old representation manifest')
    for filename,methods in [('representation_alignment_reference.tsv',MODELS),('baseline_alignment_reference.tsv',BASELINES)]:
        rows=table(ROOT/'results'/filename)
        mapping=unique(rows,('method','group_id'),'Fixed-dose reference')
        ensure(set(mapping)=={(method,gid) for method in methods for gid in identities},'Fixed reference methods/groups')
        for (method,gid),row in mapping.items():
            source=old[int(row['source_row_0based'])]
            ensure(source['method']==method and source['group_id']==gid and source['group_kind']=='within_cell_line_dose' and source['view']=='complete_metadata','Reference original source-row key')
            for key,value in source.items():ensure(row[key]==value,f'Reference exact source string {method}/{gid}/{key}')
            ensure(row['source_path']==str(source_path) and row['source_sha256']==READS[str(source_path)],'Reference source file identity')
            ensure(row['variant']=='source_name' and row['source_version']==config['version'],'Reference variant/version')
            for key in ('cohort_sha256','gene_axis_sha256'):ensure(row[key]==identities[gid][key],'Reference member/gene binding')
            old_path=ALIGN/'groups'/gid/'truth_geometry.npz'
            ensure(row['source_truth_geometry_path']==str(old_path) and row['source_truth_geometry_sha256']==READS[str(old_path)],'Reference truth binding')
            rep=representation[method,'complete_metadata']
            ensure(row['representation_path']==rep['path'] and row['representation_sha256_recorded']==rep['sha256'],'Reference representation binding')
            ensure(row['representation_payload_rehashed_this_run']=='False','Reference payload rehash declaration')
            ensure(row['truth_definition']=='original_pooled_means_then_log2ratio_not_repeat_logFC' and row['paired_measurement_endpoint']=='False' and row['ceiling_or_correction_permitted']=='False','Reference endpoint difference/correction restriction')
        COUNTS['reference_rows_copied']+=len(rows)


def check_input_identities():
    rows=table(ROOT/'input_manifest.tsv');mapping=unique(rows,('path',),'Input manifest')
    for path,checksum in READS.items():
        if path.startswith(str(A)+'/'):
            ensure((path,) in mapping and mapping[path,]['sha256']==checksum,'Production input hash binding: '+path)
            ensure(mapping[path,]['permission']=='READ_ONLY','Input permission marker')
            ensure(int(mapping[path,]['size_bytes'])==Path(path).stat().st_size and int(mapping[path,]['mtime_ns'])==Path(path).stat().st_mtime_ns,'Input stat drift: '+path)
    contract=QA/'INDEPENDENT_QA_CONTRACT.md'
    ensure(mapping[str(contract),]['sha256']==sha(contract)=='c85e4433e536de4e4202c5c6214d2651f2ca47a0334ad75d0c0e1a4ed8acc28f','Pre-result independent QA contract changed')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--production-complete',action='store_true',required=True)
    parser.add_argument('--artifact-prefix',default='independent_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    args=parser.parse_args()
    prefix=args.artifact_prefix
    ensure(prefix.replace('_','').replace('-','').isalnum(),'Unsafe QA output prefix')
    audit_path=QA/(prefix+'_audit.json')
    ensure(not audit_path.exists(),'Refuse to overwrite a prior independent audit')
    audit=dict(status='FAIL',producer_or_scoring_helper_read_or_imported=False,qa_atol=ATOL,
               source_precision='float32 caches; float64 computation',checker_sha256=sha(Path(__file__)),
               contract_sha256=sha(QA/'INDEPENDENT_QA_CONTRACT.md'),artifact_prefix=prefix)
    try:
        with threadpool_limits(limits=1):
            synthetic=fixtures()
            source=load_sources()
            references=verify_production(source)
        ensure(all(sha(path)==value for path,value in READS.items()),'Input or production output changed during QA')
        write_rows(QA/(prefix+'_fixtures.tsv'),synthetic)
        for name,rows in references.items():write_rows(QA/(prefix+'_'+name+'.tsv'),rows)
        audit.update(status='PASS',counts=dict(COUNTS),max_abs_drift=dict(MAX_DRIFT),input_output_sha256=READS,
                     n_independent_studies=1,all_reads_unchanged=True,
                     limits='Cache-level consistency reference; no raw-cell reconstruction, inference, ceiling, fitting or representation recomputation')
    except Exception as error:
        audit.update(error_type=type(error).__name__,error=str(error),counts=dict(COUNTS),max_abs_drift=dict(MAX_DRIFT))
        raise
    finally:
        with audit_path.open('x') as stream:json.dump(audit,stream,indent=2,allow_nan=False);stream.write('\n')
        print(json.dumps({k:v for k,v in audit.items() if k!='input_output_sha256'},allow_nan=False))


if __name__=='__main__':main()
