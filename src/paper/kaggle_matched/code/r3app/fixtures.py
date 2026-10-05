"""Small synthetic formula tests. These are NOT scientific R3 results."""
from __future__ import annotations
import itertools
import numpy as np
from . import frozen_geometry as fg
from . import frozen_prediction as fp
from . import independent as iq
from .common import require,write_json,ATOL


def run(output=None):
    checks=[]
    def close(name,a,b):
        require(np.allclose(a,b,atol=ATOL,rtol=0,equal_nan=True),'Fixture failed: '+name)
        checks.append(name)
    rng=np.random.default_rng(87319)
    raw=rng.integers(-4,5,size=(28,17)).astype(float)
    close('independent_average_ranks',iq.ranks(raw),__import__('scipy').stats.rankdata(raw,axis=0))
    y=rng.normal(size=(24,41)).astype(np.float32);p=(0.4*y+rng.normal(size=y.shape)).astype(np.float32)
    p[:,0]=0;y[:,1]=1;p[:,2]=3;y[:,2]=4
    y[:,3]=np.round(y[:,3]);p[:,3]=np.round(p[:,3])
    mae,r,st,ct,cp=iq.l3(p,y)
    close('MAE_matches_frozen_float64_rule',mae,fp.condition_metrics(p,y)['mae'])
    close('gene_Spearman_matches_frozen_rule',r,fp.gene_metrics(p,y)['spearman'])
    require(st[0]=='CONSTANT_PREDICTION' and st[1]=='CONSTANT_TRUTH' and
            st[2]=='CONSTANT_TRUTH_AND_PREDICTION','Fixture NA reason mismatch')
    checks.append('constant_prediction_truth_and_both_remain_NA')
    close('exact_zero_error_not_NA',iq.l3(y,y)[0],np.zeros(len(y)))
    require(np.isnan(iq.l3(p[:19],y[:19])[1]).all(),'Minimum support failed')
    checks.append('twenty_compound_minimum')
    truth=fg.cosine_matrix(y);sim=fg.cosine_matrix(p)
    close('cosine_independent',iq.cosine(y),truth)
    rr,idd,rand,status,unique=fg.prepare_truth(truth)
    nd=fg.score_neighbors(sim,rr,idd)
    a,b,c,d,e,f=iq.neighborhood(sim,truth)
    for name,u,v in [('NDCG',nd,a),('random',rand,b),('IDCG',idd,c),('relevance',rr,f)]:close(name,u,v)
    require(np.array_equal(status,d) and np.array_equal(unique,e),'Fixture truth status mismatch')
    close('perfect_ranking',fg.score_neighbors(truth,rr,idd),np.ones(len(y)))
    close('all_prediction_ties_equal_random',fg.score_neighbors(np.ones_like(sim),rr,idd),rand)
    close('independent_all_prediction_ties',iq.neighborhood(np.ones_like(sim),truth)[0],rand)
    un=np.ones((14,14));np.fill_diagonal(un,0)
    ur,ui,ub,us,_=fg.prepare_truth(un)
    require(np.isnan(ui).all() and np.all(us=='UNINFORMATIVE_TRUTH'),'Truth all ties not NA')
    require(np.isnan(iq.neighborhood(un,un)[0]).all(),'Independent truth ties not NA')
    checks.append('all_truth_ties_NA')
    small=fg.cosine_matrix(y[:11]);sr,si,sb,ss,_=fg.prepare_truth(small)
    require(np.all(ss=='INSUFFICIENT_CANDIDATES') and np.isnan(si).all(),'Candidate guard failed')
    checks.append('insufficient_candidates_NA')
    # At cutoff 10: eight strict winners, five tied candidates share two places.
    vals=np.r_[np.arange(20,12,-1),np.repeat(5.,5)]
    close('fractional_truth_cutoff_ties',fg.fractional_topk(vals,10),np.r_[np.ones(8),np.repeat(0.4,5)])
    # Entire predicted tie block includes candidates below the requested cutoff.
    scores=np.array([4.,3.,3.,3.,1.]);rel=np.array([0.2,1.,0.4,0.,0.])
    vals=[]
    for order in itertools.permutations([1,2,3]):
        vals.append(rel[0]/np.log2(2)+rel[order[0]]/np.log2(3))
    close('prediction_tie_crosses_cutoff_bruteforce',fg.expected_dcg(scores,rel,2),np.mean(vals))
    truth2=np.tile(np.r_[np.arange(20,12,-1),np.repeat(5.,6)],(14,1)).astype(float)
    sim2=np.ones_like(truth2)
    rel2,ic2,rd2,_,_=fg.prepare_truth(truth2)
    close('independent_fractional_and_prediction_ties',iq.neighborhood(sim2,truth2)[0],fg.score_neighbors(sim2,rel2,ic2))
    altered=truth.copy();np.fill_diagonal(altered,1234.)
    close('self_exclusion',iq.neighborhood(sim,altered)[0],a)
    close('condition_reindex',iq.neighborhood(sim[::-1,::-1],truth[::-1,::-1])[0],a[::-1])
    close('gene_permutation',iq.l3(p[:,::-1],y[:,::-1])[1],r[::-1])
    close('rank_correlation',iq.correlations(sim[np.triu_indices(24,1)],truth[np.triu_indices(24,1)]),
          fg.rank_correlation(sim[np.triu_indices(24,1)],truth[np.triu_indices(24,1)]))
    require(np.isnan(iq.correlations(np.ones(20),np.arange(20))), 'Constant RSA should be NA')
    checks.append('constant_pair_geometry_NA')
    # Paired common support has to differ from separate marginal means.
    aa=np.array([0.1,np.nan,0.7]);bb=np.array([0.4,0.8,np.nan]);common=np.isfinite(aa)&np.isfinite(bb)
    close('common_valid_gene_difference',iq.avg((aa-bb)[common]),-0.3)
    require(common.sum()==1,'Common valid support failed');checks.append('common_support_count')
    # Largest frozen rank minimum and unexpected numeric input rules remain explicit.
    try:iq.cosine(np.zeros((12,5)))
    except ValueError:checks.append('zero_effect_rejected_without_substitution')
    else:raise ValueError('Zero norm fixture did not stop')
    for method in ['nan','inf']:
        bad=y.copy();bad[0,0]=float(method)
        try:fp.gene_metrics(p,bad)
        except ValueError:checks.append('unexpected_'+method+'_rejected')
        else:raise ValueError('Nonfinite fixture did not stop')
    result={'status':'PASS','scope':'SYNTHETIC_FORMULA_FIXTURES_NOT_REAL_DATA',
            'n_checks':len(checks),'checks':checks,'atol':ATOL,'rtol':0}
    if output is not None:write_json(output,result)
    return result
