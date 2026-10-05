#!/usr/bin/env python3
"""Frozen, original-name sci-Plex descriptive effect geometry; no training."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, normalize
from effect_alignment_metrics import cosine_matrix, rank_correlation, prepare_truth, score_neighbors

ROOT=Path(__file__).resolve().parents[1]
FIELDS={"entity":["source_entity_key"],
        "entity_exposure":["source_entity_key","dose_value","time","dose_unit"],
        "entity_context":["source_entity_key","cell_line"],
        "complete_metadata":["source_entity_key","dose_value","time","dose_unit","cell_line"]}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1<<20),b""):h.update(block)
    return h.hexdigest()


def write_json(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n")


def save_npz(path,**kwargs):
    with Path(path).open("wb") as stream:np.savez_compressed(stream,**kwargs)


def finite_mean(values):
    values=np.asarray(values,dtype=float)
    return float(values[np.isfinite(values)].mean()) if np.isfinite(values).any() else np.nan


def baseline_features(method,view,metadata,texts,config):
    params={}
    if method=="tfidf":
        tf={k:v for k,v in config["tfidf"].items() if k!="svd"}
        tf["ngram_range"]=tuple(tf["ngram_range"])
        fitted=TfidfVectorizer(**tf).fit(sorted(set(texts)))
        matrix=fitted.transform(texts)
        params=dict(feature_name=fitted.get_feature_names_out().astype(str),idf=fitted.idf_)
        return matrix,params
    fields=FIELDS[view]
    values=metadata[fields].astype(str)
    if method=="structured_onehot":
        fitted=OneHotEncoder(handle_unknown="ignore",sparse_output=True,dtype=np.float64).fit(values)
        matrix=normalize(fitted.transform(values),norm="l2",axis=1)
        params={"categories_"+field:cat.astype(str) for field,cat in zip(fields,fitted.categories_)}
        return matrix,params
    if method!="random_field512":raise ValueError("Unexpected baseline")
    dimension=config["random_field"]["dimensions"]
    seed=config["random_field"]["seed"]
    vectors={}
    for field in fields:
        for val in sorted(values[field].unique()):
            token=f"{seed}|{field}|{val}"
            rng=np.random.default_rng(int.from_bytes(hashlib.sha256(token.encode()).digest()[:8],"little"))
            vector=rng.standard_normal(dimension)
            vectors[(field,val)]=vector/np.linalg.norm(vector)
    matrix=np.stack([sum((vectors[(field,str(row[field]))] for field in fields),np.zeros(dimension))
                     for _,row in values.iterrows()])
    matrix=normalize(matrix,norm="l2",axis=1)
    keys=sorted(vectors)
    params=dict(field=np.array([key[0] for key in keys]),value=np.array([key[1] for key in keys]),
                vector=np.stack([vectors[key] for key in keys]))
    return matrix,params


def run(config_path):
    start=time.perf_counter()
    config=json.loads(Path(config_path).read_text())
    output=Path(config["output_root"])
    assert not output.exists(),"Refuse existing or partial output"
    assert config["variant"]=="source_name" and config["views"]==list(FIELDS)
    assert config["neighbor_k"]==10 and config["minimum_group_n_for_neighbors"]==12
    rep_config=json.loads(Path(config["representation_config"]).read_text())
    effect=Path(rep_config["effects_root"]);clean=Path(rep_config["clean_views_root"])
    previous=Path(rep_config["output_root"])
    previous_audit=json.loads((previous/"audit.json").read_text())
    effect_audit=json.loads((effect/"audit.json").read_text())
    assert previous_audit["status"]==effect_audit["status"]=="PASS"
    rep_manifest=json.loads((previous/"representation_manifest.json").read_text())
    assert len(rep_manifest)==6
    inputs=[Path(config_path),Path(config["representation_config"]),Path(__file__),
            ROOT/"code/effect_alignment_metrics.py",effect/"audit.json",effect/"arrays.npz",
            effect/"atomic_index.tsv",effect/"fold_gene_panels.tsv",clean/"config.json",
            clean/"unique_texts.tsv",clean/"row_to_text_registry.tsv",
            previous/"audit.json",previous/"representation_manifest.json"]
    for item in rep_manifest:inputs.extend([Path(item["embedding_path"]),Path(item["audit_path"])])
    inputs=list(dict.fromkeys(inputs))
    before={str(path):sha(path) for path in inputs}
    for name in ["arrays.npz","atomic_index.tsv","fold_gene_panels.tsv"]:
        assert before[str(effect/name)]==effect_audit["output_sha256"][name]
    metadata_all=pd.read_csv(effect/"atomic_index.tsv",sep="\t")
    assert metadata_all.atomic_id.is_unique and len(metadata_all)==2256
    select=metadata_all.main_eligible.eq(True).to_numpy()
    metadata=metadata_all.loc[select].reset_index(drop=True)
    assert len(metadata)==2250 and metadata.source_entity_key.nunique()==188
    with np.load(effect/"arrays.npz",allow_pickle=False) as source:
        assert np.array_equal(source["atomic_id"],metadata_all.atomic_id.to_numpy(dtype=str))
        response=source["effect_log2fc"][select].astype(np.float64)
        source_genes=source["source_feature_row"]
    assert np.isfinite(response).all()
    panels=pd.read_csv(effect/"fold_gene_panels.tsv",sep="\t")
    common=sorted(set.intersection(*[set(g.source_feature_row) for _,g in panels.groupby("heldout_cell_line")]))
    assert len(common)==config["expected_global_genes"]==2099
    gene_lookup=pd.Index(source_genes)
    definitions=[("global",None,None,np.arange(len(metadata)),np.asarray(common),"global_common2099")]
    for line in ("A549","K562","MCF7"):
        genes=panels[panels.heldout_cell_line.eq(line)].sort_values("rank").source_feature_row.to_numpy()
        assert len(genes)==3000
        line_rows=np.flatnonzero(metadata.cell_line.eq(line))
        assert len(line_rows)==750
        definitions.append((line,line,None,line_rows,genes,"within_cell_line"))
        for dose,expected in [(10,188),(100,187),(1000,187),(10000,188)]:
            rows=np.flatnonzero(metadata.cell_line.eq(line)&metadata.dose_value.eq(dose))
            assert len(rows)==expected
            definitions.append((f"{line}__dose_{dose}",line,dose,rows,genes,"within_cell_line_dose"))
    registry=pd.read_csv(clean/"row_to_text_registry.tsv",sep="\t")
    registry=registry[registry.variant.eq("source_name")]
    text_table=pd.read_csv(clean/"unique_texts.tsv",sep="\t").set_index("text_id")
    assert text_table.index.is_unique
    mapped={}
    for view in FIELDS:
        rows=registry[registry.view.eq(view)].set_index("atomic_id").loc[metadata.atomic_id]
        assert rows.index.is_unique
        assert np.array_equal(rows.prompt_sha256.to_numpy(),text_table.loc[rows.text_id].prompt_sha256.to_numpy())
        mapped[view]=rows
    output.mkdir(parents=True)
    (output/"groups").mkdir();(output/"representations").mkdir()
    write_json(output/"input_manifest.json",[{"path":path,"sha256":digest} for path,digest in before.items()])
    write_json(output/"config.json",config)
    metadata.to_csv(output/"atomic_index.tsv",sep="\t",index=False)
    groups=[];group_manifest=[]
    for gid,line,dose,rows,genes,kind in definitions:
        columns=gene_lookup.get_indexer(genes);assert (columns>=0).all()
        y=response[np.ix_(rows,columns)]
        assert (np.linalg.norm(y,axis=1)>0).all(),"Zero effect vector: refuse silent cosine or cohort substitution"
        similarity=cosine_matrix(y)
        relevance,idcg,random,status,unique=prepare_truth(similarity)
        upper=similarity[np.triu_indices(len(rows),1)]
        path=output/"groups"/gid;path.mkdir()
        save_npz(path/"truth_geometry.npz",atomic_id=metadata.iloc[rows].atomic_id.to_numpy(dtype=str),
                 primary_rows=rows,source_feature_row=genes,similarity_upper=upper,
                 relevance=relevance,idcg=idcg,random_ndcg=random,status=status,
                 n_unique_effect_scores=unique)
        record=dict(group_id=gid,group_kind=kind,cell_line=line or "ALL",dose_value=dose or "ALL",
                    n_atoms=len(rows),n_genes=len(genes),n_pairs=len(upper),
                    n_zero_norm_effect_rows=int((np.linalg.norm(y,axis=1)==0).sum()),
                    valid_query_n=int(np.isfinite(idcg).sum()))
        groups.append((record,rows,upper,relevance,idcg,random,status,unique))
        group_manifest.append(record)
    write_json(output/"group_manifest.json",group_manifest)
    print("TRUTH_GROUPS_FROZEN",len(groups),flush=True)
    specs=[(item["model_key"],item) for item in rep_manifest]+[(name,None) for name in config["baselines"]]
    summaries=[];all_query=[];representations=[]
    for method,spec in specs:
        if spec:
            assert before[spec["embedding_path"]]==spec["embedding_sha256"]
            with np.load(spec["embedding_path"],allow_pickle=False) as vectors:
                ids=vectors["text_id"];x=vectors["X"];digests=vectors["prompt_sha256"]
            assert len(set(ids))==len(ids) and x.dtype==np.float32 and np.isfinite(x).all()
            assert np.array_equal(ids,np.array(["text:"+str(s) for s in digests]))
            lookup=pd.Index(ids)
        for view in FIELDS:
            mapping=mapped[view]
            texts=text_table.loc[mapping.text_id].prompt_text.to_numpy(dtype=str)
            if spec:
                indexes=lookup.get_indexer(mapping.text_id);assert (indexes>=0).all()
                assert np.array_equal(digests[indexes],mapping.prompt_sha256.to_numpy())
                features=x[indexes];params={}
                kernel=cosine_matrix(features)
            else:
                features,params=baseline_features(method,view,metadata,texts,config)
                kernel=np.asarray((features@features.T).toarray()) if hasattr(features,"tocsr") else cosine_matrix(features)
                kernel=np.clip(kernel,-1,1)
            assert kernel.shape==(2250,2250) and np.isfinite(kernel).all()
            key=method+"__"+view
            path=output/"representations"/(key+".npz")
            save_npz(path,atomic_id=metadata.atomic_id.to_numpy(dtype=str),
                     similarity_upper=kernel[np.triu_indices(2250,1)],**params)
            representations.append(dict(method=method,view=view,path=str(path),sha256=sha(path),
                                        feature_dimension=features.shape[1],
                                        fit_scope="frozen_pretrained" if spec else config["baseline_fit_scope"]))
            for record,rows,truth_upper,relevance,idcg,random,status,unique in groups:
                small=kernel[np.ix_(rows,rows)]
                rep_upper=small[np.triu_indices(len(rows),1)]
                rsa=rank_correlation(truth_upper,rep_upper)
                ndcg=score_neighbors(small,relevance,idcg)
                excess=ndcg-random
                summaries.append(dict(method=method,view=view,**record,rsa=rsa,
                    rsa_status="VALID" if np.isfinite(rsa) else "CONSTANT_PAIR_GEOMETRY",
                    ndcg_mean=finite_mean(ndcg),random_ndcg_mean=finite_mean(random),
                    excess_ndcg_mean=finite_mean(excess),valid_ndcg_n=int(np.isfinite(ndcg).sum()),
                    n_independent_studies=1))
                all_query.append(pd.DataFrame(dict(method=method,view=view,group_id=record["group_id"],
                    group_kind=record["group_kind"],atomic_id=metadata.iloc[rows].atomic_id.to_numpy(),
                    source_entity_key=metadata.iloc[rows].source_entity_key.to_numpy(),
                    n_candidates=len(rows)-1,ndcg=ndcg,random_ndcg=random,excess_ndcg=excess,
                    status=status,n_unique_effect_scores=unique)))
            print("ALIGNMENT_PASS",method,view,flush=True)
    summary=pd.DataFrame(summaries)
    assert len(summary)==576
    summary.to_csv(output/"group_summary.tsv",sep="\t",index=False,na_rep="NA")
    query=pd.concat(all_query,ignore_index=True)
    assert len(query)==243000
    query.to_csv(output/"query_metrics.tsv.gz",sep="\t",index=False,na_rep="NA",compression="gzip")
    write_json(output/"representation_manifest.json",representations)
    metrics=["rsa","ndcg_mean","random_ndcg_mean","excess_ndcg_mean"]
    local=summary[summary.group_kind.eq("within_cell_line_dose")]
    byline=local.groupby(["method","view","cell_line"],sort=False)[metrics].mean().reset_index()
    counts=local.groupby(["method","view","cell_line"],sort=False)[metrics].count().add_suffix("__valid_dose_groups").reset_index()
    byline=byline.merge(counts,on=["method","view","cell_line"])
    byline.to_csv(output/"equal_dose_by_cell_summary.tsv",sep="\t",index=False,na_rep="NA")
    macro=byline.groupby(["method","view"],sort=False)[metrics].mean().reset_index()
    counts=byline.groupby(["method","view"],sort=False)[metrics].count().add_suffix("__valid_cell_groups").reset_index()
    macro=macro.merge(counts,on=["method","view"]);macro["n_independent_studies"]=1
    macro.to_csv(output/"descriptive_equal_dose_cell_summary.tsv",sep="\t",index=False,na_rep="NA")
    assert before=={str(path):sha(path) for path in inputs},"Inputs changed during analysis"
    write_json(output/"audit.json",dict(status="PASS",scope=config["scope"],n_models=9,n_language_encoders=6,
        n_views=4,n_groups=16,n_group_results=len(summary),n_query_rows=len(query),n_atoms=2250,
        global_genes=2099,local_genes=3000,n_independent_studies=1,
        n_rsa_valid=int(np.isfinite(summary.rsa).sum()),n_ndcg_valid=int(np.isfinite(query.ndcg).sum()),
        input_hashes_unchanged=True,no_raw_read=True,no_prediction_changes=True,
        seconds=time.perf_counter()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        created_utc=datetime.now(timezone.utc).isoformat(),
        source_scripts_sha256={str(Path(__file__)):sha(Path(__file__)),str(ROOT/"code/effect_alignment_metrics.py"):sha(ROOT/"code/effect_alignment_metrics.py")}))
    print("E02_ORIGINAL_NAME_CORE_PASS",time.perf_counter()-start,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--config",required=True)
    run(parser.parse_args().config)
