#!/usr/bin/env python3
"""R2 only: frozen repeat means -> within fixed-dose geometry, no fits or raw reads."""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, platform, resource, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import scipy

MODELS = ["bge_m3","sapbert","qwen3_0_6b","biomedbert","medcpt_article","medcpt_query"]
BASELINES = ["tfidf","structured_onehot","random_field512"]

def require(ok, message):
    if not bool(ok):
        raise RuntimeError(message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(4*1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def axis_sha(values):
    return hashlib.sha256(("\n".join(map(str,values))+"\n").encode()).hexdigest()

def read_tsv(path):
    return pd.read_csv(path,sep="\t",dtype=str,keep_default_na=False)

def read_json(path):
    return json.loads(Path(path).read_text())

def write_json(path, value):
    with Path(path).open("x") as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write("\n")

def write_tsv(path, rows):
    with Path(path).open("x") as f:
        pd.DataFrame(rows).to_csv(f,sep="\t",index=False,na_rep="NA",float_format="%.17g")

def finite_mean(x):
    y=np.asarray(x,float)
    return float(y[np.isfinite(y)].mean()) if np.isfinite(y).any() else np.nan

class Tee:
    def __init__(self,*streams): self.streams=streams
    def write(self,s):
        for f in self.streams: f.write(s); f.flush()
    def flush(self):
        for f in self.streams: f.flush()

def fixture_checks(m):
    # Formula checks, not scientific experiments. Fixed before real scores.
    n=14; k=10; v=np.arange(n,dtype=float)
    s=-(v[:,None]-v[None,:])**2 + (v[:,None]+v[None,:])*0.001
    r,idcg,random,status,_=m.prepare_truth(s,k)
    require((status=="VALID").all(),"fixture: valid no-tie truth")
    np.testing.assert_allclose(m.score_neighbors(s,r,idcg,k),1,atol=1e-12,rtol=0)
    np.testing.assert_allclose(m.score_neighbors(np.ones((n,n)),r,idcg,k)-random,0,atol=1e-12,rtol=0)
    _,i0,r0,st,_=m.prepare_truth(np.ones((n,n)),k)
    require((st=="UNINFORMATIVE_TRUTH").all() and np.isnan(i0).all() and np.isnan(r0).all(),"fixture: all truth tied")
    boundary=np.array(list(range(20,12,-1))+[10,10,10,10,0],float)
    rel=m.fractional_topk(boundary,10)
    np.testing.assert_array_equal(rel,np.array([1]*8+[0.5]*4+[0],float))
    _,small,_,st,_=m.prepare_truth(np.eye(11),k)
    require((st=="INSUFFICIENT_CANDIDATES").all() and np.isnan(small).all(),"fixture: insufficient")
    x=np.arange(1,1+14*17,dtype=float).reshape(14,17)**0.7
    perm=np.array([7,0,13,2,6,4,1,12,5,8,3,10,9,11])
    gperm=np.arange(17)[::-1]
    before=m.cosine_matrix(x)
    after=m.cosine_matrix(x[perm][:,gperm])
    np.testing.assert_allclose(after,before[np.ix_(perm,perm)],atol=1e-12,rtol=0)
    # Permute an exactly known score matrix as well, avoiding accidental near-ties.
    rp,ip,rp0,sp,_=m.prepare_truth(s[np.ix_(perm,perm)],k)
    np.testing.assert_allclose(rp,r[np.ix_(perm,perm)],atol=0,rtol=0)
    np.testing.assert_allclose(ip,idcg[perm],atol=0,rtol=0)
    return dict(status="PASS",fixtures=6,scientific_experiment=False)

def run(root):
    start=time.perf_counter()
    cfg=read_json(root/"execution_contract.json")
    A=Path(cfg["source_root"]); E=A/"effects/atomic_effects_v1"; G=A/"metrics/effect_alignment_v1"
    R0=Path(cfg["r0_root"]); R1=Path(cfg["r1_accepted_root"])
    require(not any((root/"results").iterdir()) and not any((root/"geometry").iterdir()),"Refuse existing results")
    groups=[(f"{line}__dose_{dose}",line,dose,n) for line in cfg["cell_lines"] for dose,n in cfg["dose_supports"].items()]
    inputs=[root/"execution_contract.json",root/"EXECUTION_CONTRACT.md",root/"qa/INDEPENDENT_QA_CONTRACT.md",Path(__file__),Path(cfg["prompt_file"]),
            R0/"asset_manifest.tsv",R0/"analysis_contracts.md",R0/"readiness.tsv",
            R1/"R1_SUMMARY.md",R1/"R1_RESULTS.md",R1/"qa/independent_audit_final.json"]
    inputs += [E/name for name in ["measurement_means.npz","measurement_index.tsv","atomic_index.tsv","fold_gene_panels.tsv","control_partition.tsv","audit.json"]]
    inputs += [G/name for name in ["atomic_index.tsv","group_manifest.json","config.json","group_summary.tsv","representation_manifest.json","input_manifest.json","audit.json"]]
    inputs += [G/"groups"/gid/"truth_geometry.npz" for gid,_,_,_ in groups]
    inputs += [A/"code"/name for name in ["effect_alignment_metrics.py","run_effect_alignment_v1.py","run_response_diagnostics_v1.py"]]
    inputs += [A/"qa"/name/"audit.json" for name in ["effect_alignment_independent_v1","response_diagnostics_independent_v2"]]
    require(len(set(map(str,inputs)))==len(inputs),"Duplicate inputs")
    r0=read_tsv(R0/"asset_manifest.tsv")
    expected={}
    for row in r0.to_dict("records"):
        hs={row[k] for k in ["sha256_live","sha256_recorded"] if row[k]}
        require(len(hs)<=1,"R0 conflicting hashes: "+row["path"])
        if hs: expected[row["path"]]=hs.pop()
    effect_audit=read_json(E/"audit.json")
    for name,h in effect_audit["output_sha256"].items():
        if str(E/name) in expected: require(expected[str(E/name)]==h,"Parent hash conflict "+name)
        expected[str(E/name)]=h
    manifest=[]
    for path in inputs:
        st=path.stat(); h=sha(path); old=expected.get(str(path),"")
        require(not old or old==h,"INPUT_HASH_CONFLICT: "+str(path))
        manifest.append(dict(path=str(path),size_bytes=st.st_size,mtime_ns=str(st.st_mtime_ns),sha256=h,
                             prior_sha256=old,verification="MATCHED_PRIOR" if old else "NEW_RUN_IDENTITY",permission="READ_ONLY"))
    before={r["path"]:r["sha256"] for r in manifest}
    write_tsv(root/"input_manifest.tsv",manifest)
    require(effect_audit["status"]=="PASS" and read_json(R1/"qa/independent_audit_final.json")["status"]=="PASS","Prior acceptance missing")
    for d in ["effect_alignment_independent_v1","response_diagnostics_independent_v2"]:
        require(read_json(A/"qa"/d/"audit.json")["status"]=="PASS","Old QA not PASS")
    oldcfg=read_json(G/"config.json")
    require(oldcfg["neighbor_k"]==10 and oldcfg["minimum_group_n_for_neighbors"]==12 and oldcfg["variant"]=="source_name","Metric scope drift")
    require(oldcfg["zero_effect_policy"].startswith("fail closed"),"Zero norm policy drift")
    spec=importlib.util.spec_from_file_location("r2_frozen_metric_helper",A/"code/effect_alignment_metrics.py")
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    write_json(root/"qa/producer_formula_fixtures.json",fixture_checks(m))
    atoms=read_tsv(E/"atomic_index.tsv"); primary=read_tsv(G/"atomic_index.tsv")
    measurements=read_tsv(E/"measurement_index.tsv"); controls=read_tsv(E/"control_partition.tsv")
    panel=read_tsv(E/"fold_gene_panels.tsv")
    require(len(atoms)==2256 and atoms.atomic_id.is_unique and len(primary)==2250 and primary.atomic_id.is_unique,"Atomic IDs")
    require(set(atoms.main_eligible)=={"True","False"},"Inclusion flags")
    selected=atoms.loc[atoms.main_eligible.eq("True")].reset_index(drop=True)
    require(selected.equals(primary),"Primary source metadata/order mismatch")
    excluded=atoms.loc[atoms.main_eligible.eq("False")]
    require(len(excluded)==6 and set(excluded.source_entity_key)=={"localentity:1ee163940bc4f2734d95d8de"} and set(excluded.dose_value)=={"100","1000"},"Frozen exclusions drift")
    require(len(measurements)==4512 and measurements.measurement_id.is_unique and not measurements.duplicated(["atomic_id","replicate"]).any(),"Measurement IDs")
    require(set(measurements.atomic_id)==set(atoms.atomic_id) and set(measurements.replicate)=={"rep1","rep2"},"Measurement membership")
    require(measurements.groupby("atomic_id").size().eq(2).all(),"Each atom needs two repetitions")
    require(len(controls)==96 and controls.control_group.is_unique and set(controls.control_role)=={"A","B"},"Control group definition")
    control_lookup=controls.set_index("control_group")
    atom_lookup=atoms.set_index("atomic_id")
    measure_lookup={(r.atomic_id,r.replicate):i for i,r in enumerate(measurements.itertuples())}
    measurement_links=[]
    for i,r in enumerate(measurements.itertuples()):
        a=atom_lookup.loc[r.atomic_id]
        require(r.cell_line==a.cell_line and r.time==a.time=="24","Measurement/atom match")
        require(int(r.measurement_group)==i and atoms.iloc[int(r.atom_group)].atomic_id==r.atomic_id,"Stored positional mapping")
        for role in ["A","B"]:
            cid=getattr(r,"control_"+role+"_group"); c=control_lookup.loc[cid]
            require(c.control_role==role,"Control role")
            for field in ["cell_line","time","replicate","plate"]:
                require(getattr(r,field)==c[field],"Control match failed "+r.measurement_id+"/"+field)
        measurement_links.append(dict(measurement_id=r.measurement_id,measurement_row=i,atomic_id=r.atomic_id,
          source_entity_key=a.source_entity_key,group_id=f"{a.cell_line}__dose_{a.dose_value}",cell_line=r.cell_line,
          dose_value=a.dose_value,replicate=r.replicate,plate=r.plate,time=r.time,control_B_group=r.control_B_group,
          control_A_group=r.control_A_group,included_primary=a.main_eligible=="True",n_treated_cells=int(r.n_treated_cells)))
    bsets={rep:set(measurements.loc[measurements.replicate.eq(rep),"control_B_group"]) for rep in ["rep1","rep2"]}
    require(all(len(v)==24 for v in bsets.values()) and not bsets["rep1"]&bsets["rep2"],"B control overlap/count")
    with np.load(E/"measurement_means.npz",allow_pickle=False) as z:
        np.testing.assert_array_equal(z["measurement_id"],measurements.measurement_id.to_numpy(str))
        feature=z["source_feature_row"].copy()
        treated=z["normalized_mean"]; reference=z["control_B_mean"]
    require(len(feature)==3730 and len(set(feature.tolist()))==3730,"Unique feature axis")
    numeric=[]
    for name,x in [("normalized_mean",treated),("control_B_mean",reference)]:
        require(x.dtype==np.float32 and x.shape==(4512,3730),"Mean shape/dtype "+name)
        require(np.isfinite(x).all() and (x>=0).all(),"INVALID_MEAN_VALUES "+name)
        numeric.append(dict(array=name,source_dtype=str(x.dtype),shape=list(x.shape),minimum=float(x.min()),maximum=float(x.max()),all_finite=True,all_nonnegative=True))
    # Verify same shared control has the same cached B vector; no control averaging here.
    for cid,rows in measurements.groupby("control_B_group",sort=False).groups.items():
        ix=np.asarray(list(rows))
        require(np.array_equal(reference[ix],np.broadcast_to(reference[ix[0]],(len(ix),len(feature)))),"Same B group different cached vector "+cid)
    feature_map={int(g):i for i,g in enumerate(feature)}
    group_defs=[]; memberships=[]; norms=[]; blocked=[]
    old_groups={r["group_id"]:r for r in read_json(G/"group_manifest.json")}
    for gid,line,dose,n in groups:
        rows=np.flatnonzero(primary.cell_line.eq(line)&primary.dose_value.eq(dose))
        require(len(rows)==n,"Group size "+gid)
        ids=primary.iloc[rows].atomic_id.to_numpy(str)
        localpanel=panel.loc[panel.heldout_cell_line.eq(line)].assign(_rank=lambda d:d["rank"].astype(int)).sort_values("_rank")
        genes=localpanel.source_feature_row.to_numpy(dtype=np.int64)
        require(len(genes)==3000 and len(set(genes))==3000,"Panel "+gid)
        columns=np.array([feature_map[int(g)] for g in genes])
        np.testing.assert_array_equal(columns,localpanel.union_column.to_numpy(dtype=int))
        truth_path=G/"groups"/gid/"truth_geometry.npz"
        with np.load(truth_path,allow_pickle=False) as z:
            np.testing.assert_array_equal(z["atomic_id"],ids)
            np.testing.assert_array_equal(z["primary_rows"],rows)
            np.testing.assert_array_equal(z["source_feature_row"],genes)
        old=old_groups[gid]
        require(old["n_atoms"]==n and old["n_genes"]==3000 and old["group_kind"]=="within_cell_line_dose","Old group mismatch")
        rep_rows={rep:np.array([measure_lookup[(key,rep)] for key in ids]) for rep in ["rep1","rep2"]}
        for rep,ix in rep_rows.items():
            effect=np.log2((treated[np.ix_(ix,columns)].astype(np.float64)+1)/(reference[np.ix_(ix,columns)].astype(np.float64)+1))
            require(np.isfinite(effect).all(),"Nonfinite logFC "+gid+"/"+rep)
            norm=np.linalg.norm(effect,axis=1)
            if (norm==0).any():
                blocked.append(dict(group_id=gid,replicate=rep,reason="ZERO_EFFECT_NORM_FAIL_CLOSED",atomic_ids=ids[norm==0].tolist()))
            norms.append(dict(group_id=gid,replicate=rep,n_atoms=n,n_genes=3000,n_zero_norm=int((norm==0).sum()),min_norm=float(norm.min()),max_norm=float(norm.max())))
        group_defs.append(dict(group_id=gid,cell_line=line,dose_value=dose,n_atoms=n,n_genes=3000,rows=rows,ids=ids,genes=genes,columns=columns,rep_rows=rep_rows,
          cohort_sha256=axis_sha(ids),gene_axis_sha256=axis_sha(genes),source_truth_geometry_path=str(truth_path),source_truth_geometry_sha256=before[str(truth_path)]))
    preflight=dict(status="PASS" if not blocked else "BLOCKED_GROUPS",mean_checks=numeric,group_norm_checks=norms,blocked_groups=blocked,
          n_primary=2250,n_measurements=4512,n_controls=96,n_B_groups=48,rep_B_groups_disjoint=True,
          same_B_group_cache_exact=True,source_dtype="float32",compute_dtype="float64",arrays_npz_read=False)
    write_json(root/"qa/numeric_preflight.json",preflight)
    print("R2_NUMERIC_PREFLIGHT",preflight["status"],flush=True)
    blocked_ids={x["group_id"] for x in blocked}
    geometry_rows=[]; query_rows=[]; summary_rows=[]; manifest_rows=[]; norm_rows=[]; na=[]
    for a in atoms.itertuples():
        if a.main_eligible=="False":
            na.append(dict(scope="original_exclusion",group_id=f"{a.cell_line}__dose_{a.dose_value}",direction="NA",atomic_id=a.atomic_id,
                 reason="ORIGINAL_MAIN_ELIGIBLE_FALSE",n_atoms_nominal=1,n_queries_valid=0))
        memberships.append(dict(atomic_id=a.atomic_id,source_entity_key=a.source_entity_key,source_entity_name=a.source_entity_name,
           group_id=f"{a.cell_line}__dose_{a.dose_value}",cell_line=a.cell_line,dose_value=a.dose_value,dose_unit=a.dose_unit,time=a.time,
           included_primary=a.main_eligible=="True",exclusion_reason="" if a.main_eligible=="True" else "ORIGINAL_MAIN_ELIGIBLE_FALSE",
           rep1_measurement_id=measurements.iloc[measure_lookup[(a.atomic_id,"rep1")]].measurement_id,
           rep2_measurement_id=measurements.iloc[measure_lookup[(a.atomic_id,"rep2")]].measurement_id))
    for g in group_defs:
        gid=g["group_id"]; n=g["n_atoms"]
        common={k:g[k] for k in ["group_id","cell_line","dose_value","n_atoms","n_genes","cohort_sha256","gene_axis_sha256"]}
        common.update(effect_definition_id=cfg["effect_definition_id"],n_independent_studies=1)
        manifest_rows.append(dict(**common,n_candidates=n-1,n_pairs_nominal=n*(n-1)//2,
          source_truth_geometry_path=g["source_truth_geometry_path"],source_truth_geometry_sha256=g["source_truth_geometry_sha256"],
          measurement_cache_path=str(E/"measurement_means.npz"),measurement_cache_sha256=before[str(E/"measurement_means.npz")],
          source_dtype="float32",compute_dtype="float64",status="BLOCKED_ZERO_NORM" if gid in blocked_ids else "READY"))
        if gid in blocked_ids:
            na.append(dict(scope="blocked_group",group_id=gid,direction="both",atomic_id="ALL",reason="ZERO_EFFECT_NORM_FAIL_CLOSED",n_atoms_nominal=n,n_queries_valid=0))
            continue
        sims={}; data={"atomic_id":g["ids"],"source_feature_row":g["genes"],"primary_rows":g["rows"]}
        for rep,ix in g["rep_rows"].items():
            x=np.log2((treated[np.ix_(ix,g["columns"])].astype(np.float64)+1)/(reference[np.ix_(ix,g["columns"])].astype(np.float64)+1))
            norms_x=np.linalg.norm(x,axis=1)
            sims[rep]=m.cosine_matrix(x)
            require(np.isfinite(sims[rep]).all() and (norms_x>0).all(),"Cosine preconditions changed")
            np.testing.assert_allclose(sims[rep],sims[rep].T,atol=1e-12,rtol=0)
            np.testing.assert_allclose(np.diag(sims[rep]),1,atol=1e-12,rtol=0)
            data[rep+"_cosine"]=sims[rep]; data[rep+"_effect_norm"]=norms_x; data[rep+"_measurement_rows"]=ix
            data[rep+"_effect_sha256"]=np.array(hashlib.sha256(x.tobytes(order="C")).hexdigest())
        upper=np.triu_indices(n,1)
        rho=m.rank_correlation(sims["rep1"][upper],sims["rep2"][upper])
        geometry_rows.append(dict(**common,n_atoms_nominal=n,n_atoms_finite=n,n_pairs_nominal=len(upper[0]),n_pairs_finite=len(upper[0]),
                                  replicate_geometry_spearman=rho,rsa_status="VALID" if np.isfinite(rho) else "CONSTANT_GEOMETRY",
                                  n_zero_norm_rep1=0,n_zero_norm_rep2=0))
        if not np.isfinite(rho):
            na.append(dict(scope="geometry",group_id=gid,direction="symmetric",atomic_id="ALL",reason="CONSTANT_GEOMETRY",n_atoms_nominal=n,n_queries_valid=0))
        for ranking,truth in cfg["directions"]:
            direction=ranking+"_to_"+truth
            relevance,idcg,random,status,unique=m.prepare_truth(sims[truth],cfg["neighbor_k"],cfg["minimum_group_n"])
            score=m.score_neighbors(sims[ranking],relevance,idcg,cfg["neighbor_k"]); excess=score-random
            valid=np.isfinite(score)
            require(np.array_equal(valid,status=="VALID"),"Query status mismatch")
            require(np.isfinite(random[valid]).all() and np.isfinite(excess[valid]).all(),"Invalid valid-query metric")
            require(((score[valid]>=-1e-12)&(score[valid]<=1+1e-12)).all(),"NDCG range")
            for i,key in enumerate(g["ids"]):
                truth_scores=sims[truth][i,np.arange(n)!=i]
                threshold=np.partition(truth_scores,n-1-cfg["neighbor_k"])[n-1-cfg["neighbor_k"]]
                query_rows.append(dict(group_id=gid,cell_line=g["cell_line"],dose_value=g["dose_value"],direction=direction,
                  ranking_replicate=ranking,truth_replicate=truth,atomic_id=key,source_entity_key=atom_lookup.loc[key].source_entity_key,
                  ranking_measurement_id=measurements.iloc[g["rep_rows"][ranking][i]].measurement_id,
                  truth_measurement_id=measurements.iloc[g["rep_rows"][truth][i]].measurement_id,n_candidates=n-1,
                  status=str(status[i]),na_reason="" if valid[i] else str(status[i]),ndcg=float(score[i]),
                  random_ndcg=float(random[i]),excess_ndcg=float(excess[i]),idcg=float(idcg[i]),
                  n_unique_truth_scores=int(unique[i]),truth_boundary_tie_n=int((truth_scores==threshold).sum()),
                  relevance_sum=float(relevance[i].sum()),cohort_sha256=g["cohort_sha256"],gene_axis_sha256=g["gene_axis_sha256"]))
                if not valid[i]:
                    na.append(dict(scope="query",group_id=gid,direction=direction,atomic_id=key,reason=str(status[i]),n_atoms_nominal=n,n_queries_valid=0))
            summary_rows.append(dict(**common,direction=direction,ranking_replicate=ranking,truth_replicate=truth,
              n_queries_nominal=n,n_queries_valid=int(valid.sum()),n_queries_na=int((~valid).sum()),n_candidates=n-1,
              ndcg_mean=finite_mean(score),random_ndcg_mean=finite_mean(random),excess_ndcg_mean=finite_mean(excess),
              n_positive_excess=int((excess[valid]>1e-12).sum()),n_negative_excess=int((excess[valid]<-1e-12).sum()),
              n_numeric_zero_excess=int((np.abs(excess[valid])<=1e-12).sum()),status="VALID" if valid.any() else "NO_VALID_QUERIES"))
        with (root/"geometry"/(gid+".npz")).open("xb") as f: np.savez_compressed(f,**data)
        print("R2_GROUP_COMPLETE",gid,n,flush=True)
    links=pd.DataFrame(measurement_links); sharing=[]; by_group=[]
    for cid,frame in links.groupby("control_B_group",sort=False):
        c=control_lookup.loc[cid]; selected_links=frame.loc[frame.included_primary]
        sharing.append(dict(control_B_group=cid,replicate=c.replicate,cell_line=c.cell_line,time=c.time,plate=c.plate,
           control_well=c.well,control_cells=int(c.n_cells),n_measurements_all=len(frame),n_atoms_all=frame.atomic_id.nunique(),
           n_measurements_primary=len(selected_links),n_atoms_primary=selected_links.atomic_id.nunique(),
           n_dose_groups_primary=selected_links.group_id.nunique(),replicate_B_sets_disjoint=True,
           same_cached_B_vector=True,independent_across_drugs=False))
    for (gid,rep,cid),frame in links.loc[links.included_primary].groupby(["group_id","replicate","control_B_group"],sort=False):
        by_group.append(dict(group_id=gid,replicate=rep,control_B_group=cid,n_atoms=frame.atomic_id.nunique(),n_measurements=len(frame)))
    outputs={"group_axis_manifest.tsv":manifest_rows,"replicate_geometry_consistency.tsv":geometry_rows,
      "replicate_neighbor_query_metrics.tsv":query_rows,"replicate_neighbor_group_summary.tsv":summary_rows,
      "control_sharing_summary.tsv":sharing,"control_sharing_by_group.tsv":by_group,"measurement_control_links.tsv":measurement_links,
      "condition_membership.tsv":memberships,"na_and_exclusions.tsv":na,"repeat_norm_checks.tsv":norms}
    for name,rows in outputs.items(): write_tsv(root/"results"/name,rows)
    # Frozen, predeclared descriptive references; no representation computation.
    oldsummary=read_tsv(G/"group_summary.tsv")
    rep_manifest=read_json(G/"representation_manifest.json")
    refs=[]; baserefs=[]
    ref_errors=[]
    try:
        for g in group_defs:
            gid=g["group_id"]; sub=oldsummary.loc[oldsummary.group_id.eq(gid)&oldsummary.view.eq("complete_metadata")]
            require(len(sub)==9 and set(sub.method)==set(MODELS+BASELINES),"Reference completeness "+gid)
            for source_row,row in sub.iterrows():
                require(int(row.n_atoms)==g["n_atoms"] and int(row.n_genes)==3000 and row.group_kind=="within_cell_line_dose","Reference unit mismatch")
                spec=[s for s in rep_manifest if s["method"]==row.method and s["view"]=="complete_metadata"]
                require(len(spec)==1,"Reference manifest missing")
                record=row.to_dict()
                record.update(source_row_0based=int(source_row),source_path=str(G/"group_summary.tsv"),
                  source_sha256=before[str(G/"group_summary.tsv")],variant="source_name",source_version=oldcfg["version"],
                  cohort_sha256=g["cohort_sha256"],gene_axis_sha256=g["gene_axis_sha256"],
                  source_truth_geometry_path=g["source_truth_geometry_path"],source_truth_geometry_sha256=g["source_truth_geometry_sha256"],
                  representation_path=spec[0]["path"],representation_sha256_recorded=spec[0]["sha256"],fit_scope=spec[0]["fit_scope"],
                  representation_payload_rehashed_this_run=False,truth_definition="original_pooled_means_then_log2ratio_not_repeat_logFC",
                  paired_measurement_endpoint=False,ceiling_or_correction_permitted=False)
                (refs if row.method in MODELS else baserefs).append(record)
        require(len(refs)==72 and len(baserefs)==36,"Reference counts")
        write_tsv(root/"results/representation_alignment_reference.tsv",refs)
        write_tsv(root/"results/baseline_alignment_reference.tsv",baserefs)
    except Exception as error:
        ref_errors.append(str(error))
        write_json(root/"qa/reference_binding_failure.json",dict(status="BLOCKED",error=str(error),no_fabricated_reference=True))
    require(all(sha(path)==digest for path,digest in before.items()),"Inputs changed during R2")
    if not blocked_ids:
        require(len(geometry_rows)==12 and len(summary_rows)==24 and len(query_rows)==4500,"Full output counts")
    outfiles=[p for folder in ["results","geometry"] for p in sorted((root/folder).glob("*")) if p.is_file()]
    audit=dict(status="PASS" if not blocked_ids else "PARTIAL_BLOCKED",reference_status="PASS" if not ref_errors else "BLOCKED",
      n_groups=len(geometry_rows),n_direction_groups=len(summary_rows),n_query_rows=len(query_rows),n_primary_atoms=2250,
      n_na_query_rows=sum(r["status"]!="VALID" for r in query_rows),n_na_geometry=sum(r["rsa_status"]!="VALID" for r in geometry_rows),
      n_old_exclusions=6,n_B_groups=len(sharing),n_reference_rows=len(refs),n_baseline_reference_rows=len(baserefs),
      all_input_hashes_unchanged=True,n_inputs=len(inputs),r1_rerun=False,new_fitting=False,new_encoding=False,
      raw_h5ad_read=False,arrays_npz_read=False,old_geometry_recomputed=False,new_repeat_geometry_computed=True,
      statistical_tests=False,reliability_filter=False,noise_ceiling_claim=False,other_branches_run=False,
      independent_qa="PENDING",python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,
      elapsed_seconds=time.perf_counter()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
      contract_sha256=before[str(root/"execution_contract.json")],producer_sha256=before[str(Path(__file__))],
      helper_sha256=before[str(A/"code/effect_alignment_metrics.py")],outputs=[dict(path=str(p),sha256=sha(p)) for p in outfiles])
    write_json(root/"qa/producer_audit.json",audit)
    print("R2_PRODUCER",audit["status"],audit["n_groups"],audit["n_query_rows"],"seconds",audit["elapsed_seconds"],flush=True)
    return 0 if audit["status"]=="PASS" else 2

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--run-dir",required=True,type=Path)
    args=ap.parse_args();root=args.run_dir.resolve()
    log=root/"logs"/"producer.log"
    with log.open("x") as f:
        stdout,stderr=sys.stdout,sys.stderr;sys.stdout=Tee(stdout,f);sys.stderr=Tee(stderr,f)
        code=1
        try: code=run(root)
        except Exception as e:
            traceback.print_exc()
            tag=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            write_json(root/"qa"/("producer_failure_"+tag+".json"),dict(status="FAIL",error_type=type(e).__name__,error=str(e),old_inputs_modified=False))
        finally:
            write_json(root/"logs/exit_status.json",dict(exit_code=code,completed_utc=datetime.now(timezone.utc).isoformat()))
            sys.stdout,sys.stderr=stdout,stderr
    raise SystemExit(code)
