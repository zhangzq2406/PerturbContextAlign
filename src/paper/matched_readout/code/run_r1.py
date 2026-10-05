#!/usr/bin/env python3
"""R1 only: rescore frozen OOF on the common cohort and join accepted geometry.
No encoding, fitting, raw expression, geometry recomputation, inference tests or plots.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import sklearn

def require(ok, message):
    if not bool(ok):
        raise RuntimeError(message)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def axis_digest(values):
    return hashlib.sha256(("\n".join(map(str, values)) + "\n").encode()).hexdigest()

def table(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)

def load_json(path):
    return json.loads(Path(path).read_text())

def write_json(path, data):
    with Path(path).open("x") as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

def write_tsv(path, records, columns=None):
    frame = records if isinstance(records, pd.DataFrame) else pd.DataFrame(records, columns=columns)
    with Path(path).open("x") as f:
        frame.to_csv(f, sep="\t", index=False, na_rep="NA", float_format="%.17g")

def run(run_dir):
    started = time.perf_counter()
    cfg_path = run_dir / "execution_contract.json"
    cfg = load_json(cfg_path)
    A = Path(cfg["source_root"])
    E1, E2 = A / "metrics/information_fidelity_v1", A / "metrics/effect_alignment_v1"
    effect = A / "effects/atomic_effects_v1"
    clean = A / "representations/clean_views_v1"
    r0 = Path(cfg["r0_root"])
    outputs = run_dir / "results"
    require(not any(outputs.iterdir()), "Refuse to overwrite existing/partial results")
    models, baselines = cfg["models"], cfg["baselines"]
    views, cells, doses = cfg["views"], cfg["cell_lines"], cfg["dose_labels"]
    methods = models + baselines
    require(len(models) == 6 and len(views) == 4 and len(cells) == 3, "Contract matrix changed")
    require([c["contrast_family"] for c in cfg["contrasts"]] ==
            ["exposure_addition", "exposure_addition", "context_addition_sensitivity", "context_addition_sensitivity"],
            "Contrast grouping changed")
    old_assets = table(r0 / "asset_manifest.tsv").set_index("path")
    rep_records = load_json(E1 / "representation_manifest.json")
    specs = {r["model_key"]: r for r in rep_records}
    require(set(specs) == set(models), "Six frozen encoders do not match contract")
    e1_inputs = {str(Path(x["path"]) if Path(x["path"]).is_absolute() else A / x["path"]): x["sha256"]
                 for x in load_json(E1 / "input_manifest.json")}
    e2_inputs = {str(Path(x["path"]) if Path(x["path"]).is_absolute() else A / x["path"]): x["sha256"]
                 for x in load_json(E2 / "input_manifest.json")}
    source_paths = [
        cfg_path, Path(__file__), r0 / "asset_manifest.tsv", r0 / "analysis_contracts.md",
        E1 / "atomic_index.tsv", E1 / "folds.tsv", E1 / "input_manifest.json",
        E1 / "representation_manifest.json", E1 / "audit.json",
        E2 / "atomic_index.tsv", E2 / "group_summary.tsv", E2 / "group_manifest.json",
        E2 / "representation_manifest.json", E2 / "input_manifest.json", E2 / "config.json",
        E2 / "audit.json", effect / "fold_gene_panels.tsv",
        clean / "row_to_text_registry.tsv", clean / "config.json",
        A / "qa/information_fidelity_independent_v1_format_compatible/audit.json",
        A / "qa/effect_alignment_independent_v1/audit.json",
    ]
    source_paths += [E1 / "readout" / m / v / "oof_predictions.tsv" for m in methods for v in views]
    source_paths += [E2 / "groups" / cell / "truth_geometry.npz" for cell in cells]
    source_paths += [Path(specs[m]["path"]) for m in models]
    source_paths = list(dict.fromkeys(source_paths))
    manifest = []
    for path in source_paths:
        require(path.is_file(), f"Missing input: {path}")
        st = path.stat()
        sha = digest(path)
        r0_sha = ""
        if str(path) in old_assets.index:
            old = old_assets.loc[str(path)]
            require(st.st_size == int(old.size_bytes) and st.st_mtime_ns == int(old.mtime_ns),
                    f"Input stat changed since R0: {path}")
            r0_sha = old.sha256_live or old.sha256_recorded
            if r0_sha:
                require(sha in r0_sha.split(";"), f"Input hash changed since R0: {path}")
        manifest.append(dict(path=str(path), size_bytes=st.st_size, mtime_ns=str(st.st_mtime_ns),
                             sha256=sha, r0_expected_sha256=r0_sha,
                             verification="MATCH_R0" if r0_sha else "LIVE_IDENTITY_NEW_HASH",
                             permission="read_only_input"))
    before = {r["path"]: r for r in manifest}
    for m in models:
        p = specs[m]["path"]
        require(e1_inputs[p] == e2_inputs[p] == specs[m]["sha256"] == before[p]["sha256"],
                f"E01/E02 encoder version mismatch: {m}")
    registry_path = str(clean / "row_to_text_registry.tsv")
    require(e1_inputs[registry_path] == e2_inputs[registry_path] == before[registry_path]["sha256"],
            "Text registry version mismatch")
    for qa in source_paths:
        if qa.name == "audit.json":
            require(load_json(qa)["status"] == "PASS", f"Parent QA not PASS: {qa}")
    metadata = table(E1 / "atomic_index.tsv")
    geom_meta = table(E2 / "atomic_index.tsv")
    folds = table(E1 / "folds.tsv")
    require(len(metadata) == 2256 and metadata.atomic_id.is_unique, "E01 atom axis")
    require(len(geom_meta) == 2250 and geom_meta.atomic_id.is_unique, "E02 atom axis")
    require(folds.atomic_id.is_unique and set(folds.atomic_id) == set(metadata.atomic_id), "Fold axis")
    require(folds.groupby("source_entity_key").fold.nunique().eq(1).all(), "Entity crosses OOF folds")
    require(folds.groupby("fold").size().to_dict() == {"0":456,"1":456,"2":456,"3":444,"4":444}, "Original fold support")
    require(len(set(metadata.source_entity_key)) == 188, "Entity count")
    meta_i = metadata.set_index("atomic_id")
    common = meta_i.loc[geom_meta.atomic_id].reset_index()
    for key in ["source_entity_key", "cell_line", "dose_value", "dose_unit", "time", "main_eligible"]:
        require(common[key].tolist() == geom_meta[key].tolist(), f"Metadata disagreement: {key}")
    require(set(common.atomic_id) == set(metadata.loc[metadata.main_eligible.eq("True"), "atomic_id"]), "Primary membership")
    excluded = metadata.loc[~metadata.atomic_id.isin(common.atomic_id)].copy()
    require(len(excluded) == 6 and
            excluded.source_entity_key.eq("localentity:1ee163940bc4f2734d95d8de").all() and
            excluded.source_entity_name.eq("YM155 (Sepantronium Bromide)").all() and
            excluded.main_eligible.eq("False").all(), "Unexpected excluded atoms")
    require(set(excluded.dose_value) == {"100","1000"} and set(excluded.cell_line) == set(cells), "YM155 exclusions")
    common = common.merge(folds[["atomic_id","fold"]], on="atomic_id", validate="one_to_one")
    fold_hash = before[str(E1 / "folds.tsv")]["sha256"]
    global_cohort_hash = axis_digest(common.atomic_id)
    groups = load_json(E2 / "group_manifest.json")
    panels = table(effect / "fold_gene_panels.tsv")
    registry = table(clean / "row_to_text_registry.tsv")
    registry = registry.loc[registry.variant.eq(cfg["variant"])]
    geom = table(E2 / "group_summary.tsv")
    geom["source_row_0based"] = np.arange(len(geom))
    numeric_fields = ["rsa","ndcg_mean","random_ndcg_mean","excess_ndcg_mean"]
    for col in numeric_fields:
        geom[col] = pd.to_numeric(geom[col].replace("NA", np.nan), errors="raise")
    geom = geom.loc[geom.group_kind.eq("within_cell_line") & geom.method.isin(methods) & geom.view.isin(views)].copy()
    require(len(geom) == 108 and not geom.duplicated(["method","view","cell_line"]).any(), "Geometry selection cardinality")
    axis_records, support, na_rows, binding_records = [], [], [], []
    group_info = {}
    for cell in cells:
        sub = common.loc[common.cell_line.eq(cell)]
        require(len(sub) == 750 and sub.source_entity_key.nunique() == 188, f"Common cohort: {cell}")
        dose_counts = sub.dose_value.astype(int).value_counts().to_dict()
        require(dose_counts == {10:188,100:187,1000:187,10000:188}, f"Dose support: {cell}")
        for dose in doses:
            support.append(dict(cell_line=cell,field="dose_value",label=dose,n_conditions=dose_counts[dose],
                                n_classes=4,evaluated=True,scope="within_cell_line_cross_dose"))
        for field in ["cell_line","time","dose_unit"]:
            require(sub[field].nunique() == 1, f"Expected constant label {cell}/{field}")
            support.append(dict(cell_line=cell,field=field,label=sub[field].iloc[0],n_conditions=750,
                                n_classes=1,evaluated=False,scope="within_cell_line_cross_dose"))
            na_rows.append(dict(kind="UNEVALUATED_CONSTANT_LABEL",model="ALL",view="ALL",cell_line=cell,
                                atomic_id="",field=field,reason="CONSTANT_IN_EVALUATION_COHORT",n_affected=750))
        for dose in doses:
            for field in ["dose_value","cell_line"]:
                na_rows.append(dict(kind="BLOCKED_EVALUATION_UNIT",model="ALL",view="ALL",cell_line=cell,
                                    atomic_id="",field=field,reason=f"CONSTANT_WITHIN_CELL_LINE_DOSE_{dose}",n_affected=dose_counts[dose]))
        truth_path = E2 / "groups" / cell / "truth_geometry.npz"
        with np.load(truth_path, allow_pickle=False) as z:
            ids = z["atomic_id"].astype(str)
            genes = z["source_feature_row"].astype(int)
            primary_rows = z["primary_rows"]
            upper = z["similarity_upper"]
            idcg, random, truth_status = z["idcg"], z["random_ndcg"], z["status"].astype(str)
        require(ids.tolist() == sub.atomic_id.tolist(), f"Truth member/order mismatch {cell}")
        require(np.array_equal(primary_rows,np.flatnonzero(common.cell_line.eq(cell))), "Primary row pointer")
        expected_genes = panels.loc[panels.heldout_cell_line.eq(cell)].sort_values("rank", key=lambda s:s.astype(int)).source_feature_row.astype(int).to_numpy()
        require(np.array_equal(genes,expected_genes) and len(set(genes)) == 3000, f"Gene axis: {cell}")
        require(upper.shape == (280875,) and np.isfinite(upper).all(), f"Truth geometry nonfinite {cell}")
        require(((upper >= -1-1e-12)&(upper <= 1+1e-12)).all(), "Truth similarity bounds")
        require(np.isfinite(idcg).all() and (idcg>0).all() and np.isfinite(random).all(), "Unexpected truth NA: stop before selection")
        require(set(truth_status) == {"VALID"}, "Unexpected truth status")
        group_id = "sciplex:"+cell+":cross_dose:"+axis_digest(ids)[:16]
        gene_id = "source_feature_row:"+axis_digest(genes)
        group_info[cell] = dict(cohort_id=group_id,cohort_sha256=axis_digest(ids),gene_axis_id=gene_id,
                                n_conditions=750,n_entities=188,n_dose_classes=4,n_genes=3000,n_pairs=280875,
                                truth_cache_sha256=before[str(truth_path)]["sha256"])
        axis_records.append(dict(cell_line=cell,**group_info[cell],dose_support_json=json.dumps(dose_counts,sort_keys=True),
                                 probe_fold_hash=fold_hash,global_primary_cohort_sha256=global_cohort_hash))
        g = geom.loc[geom.cell_line.eq(cell)]
        for col,want in [("n_atoms",750),("n_genes",3000),("n_pairs",280875),("valid_ndcg_n",750),("valid_query_n",750)]:
            require(g[col].astype(int).eq(want).all(), f"Geometry denominator {cell}/{col}")
        require(np.allclose(g.random_ndcg_mean.to_numpy(float),random.mean(),rtol=0,atol=1e-14), "Random-reference mismatch")
        for view in views:
            rr = registry.loc[registry.view.eq(view)].set_index("atomic_id")
            require(rr.index.is_unique, f"Duplicate registry {view}")
            rr = rr.loc[sub.atomic_id]
            require(rr.source_entity_key.tolist() == sub.source_entity_key.tolist(), "Text identity mapping")
            require((rr.text_id == "text:"+rr.prompt_sha256).all(), "Prompt/text ID binding")
            for m in models:
                binding_records.append(dict(model=m,view=view,cell_line=cell,cohort_id=group_id,
                                            representation_sha256=specs[m]["sha256"],
                                            representation_path=specs[m]["path"],registry_sha256=before[registry_path]["sha256"],
                                            ordered_view_text_sha256=axis_digest(rr.text_id),gene_axis_id=gene_id,
                                            representation_fit_scope="frozen_pretrained",
                                            probe_fit_scope="original_five_entity_folds_pooled_three_cell_lines",
                                            geometry_fit_scope="descriptive_no_learning_fold"))
    print("PREFLIGHT_PASS: hashes, cohort, folds, labels, text versions and three geometry axes",flush=True)
    write_tsv(run_dir / "input_manifest.tsv", manifest)
    write_json(run_dir / "qa/producer_preflight.json", dict(
        status="PASS",inputs=len(manifest),common_atoms=2250,per_cell_line=750,n_models=6,n_views=4,
        prior_qa_reused=True,geometry_recomputed=False,probe_refitted=False,
        main_interpretation="same-evaluation-cohort descriptive pairing, not unified end-to-end generalization"))
    base, baseline_read, baseline_geom, oof_checks, confusion = [], [], [], [], []
    geom_lookup = geom.set_index(["method","view","cell_line"])
    bindings = pd.DataFrame(binding_records).set_index(["model","view","cell_line"])
    for model in methods:
        for view in views:
            op = E1 / "readout" / model / view / "oof_predictions.tsv"
            data = table(op)
            require(len(data)==2256 and data.atomic_id.is_unique and set(data.atomic_id)==set(metadata.atomic_id),
                    f"OOF membership {model}/{view}")
            indexed = data.set_index("atomic_id").loc[metadata.atomic_id]
            for key in ["source_entity_key","cell_line","dose_value"]:
                require(indexed[key].tolist()==metadata[key].tolist(),f"OOF metadata {model}/{view}/{key}")
            require(indexed.fold.tolist()==folds.set_index("atomic_id").loc[metadata.atomic_id].fold.tolist(),"OOF fold drift")
            for col in ["dose_value_prediction","dose_value_training_majority"]:
                require(set(data[col].astype(int)).issubset(set(doses)), f"Invalid dose output {model}/{view}/{col}")
            require(data.dose_value_correct.eq("True").to_numpy().tolist()==
                    data.dose_value.eq(data.dose_value_prediction).to_numpy().tolist(),"Saved correct flag inconsistent")
            oof_checks.append(dict(model=model,view=view,source_n=2256,common_n=2250,excluded_n=6,
                                   unique_id=True,metadata_match=True,fold_match=True,
                                   sha256=before[str(op)]["sha256"],is_primary=model in models))
            di = data.set_index("atomic_id")
            for cell in cells:
                sub = common.loc[common.cell_line.eq(cell)]
                picked = di.loc[sub.atomic_id]
                truth = picked.dose_value.astype(int).to_numpy()
                pred = picked.dose_value_prediction.astype(int).to_numpy()
                maj = picked.dose_value_training_majority.astype(int).to_numpy()
                metric = dict(dose_accuracy=float(accuracy_score(truth,pred)),
                              dose_macro_f1=float(f1_score(truth,pred,labels=doses,average="macro",zero_division=0)),
                              training_majority_accuracy=float(accuracy_score(truth,maj)),
                              training_majority_macro_f1=float(f1_score(truth,maj,labels=doses,average="macro",zero_division=0)))
                cm = confusion_matrix(truth,pred,labels=doses)
                for i,t in enumerate(doses):
                    for j,p in enumerate(doses):
                        confusion.append(dict(model=model,view=view,cell_line=cell,true_dose=t,predicted_dose=p,
                                              n=int(cm[i,j]),is_primary=model in models))
                gr = geom_lookup.loc[(model,view,cell)]
                require(gr.rsa_status in ["VALID","CONSTANT_PAIR_GEOMETRY"], "Unknown RSA NA rule")
                if gr.rsa_status=="VALID":
                    require(np.isfinite(gr.rsa) and -1-1e-12<=gr.rsa<=1+1e-12, "Invalid RSA value")
                else:
                    require(np.isnan(gr.rsa), "Missing RSA expected by source status")
                    na_rows.append(dict(kind="SOURCE_METRIC_NA",model=model,view=view,cell_line=cell,
                                        atomic_id="",field="rsa",reason=gr.rsa_status,n_affected=750))
                require(np.isfinite(gr[["ndcg_mean","random_ndcg_mean","excess_ndcg_mean"]].to_numpy(float)).all(),"Nonfinite neighborhood summary")
                require(abs(gr.ndcg_mean-gr.random_ndcg_mean-gr.excess_ndcg_mean)<=1e-12,"NDCG source arithmetic")
                geometry = {key:float(gr[key]) for key in numeric_fields}
                geometry.update(rsa_status=gr.rsa_status,valid_ndcg_n=int(gr.valid_ndcg_n),
                                geometry_source=str(E2/"group_summary.tsv"),geometry_source_row_0based=int(gr.source_row_0based))
                if model in models:
                    binding = bindings.loc[(model,view,cell)]
                    base.append(dict(model=model,view=view,cell_line=cell,**group_info[cell],**metric,**geometry,
                                     representation_sha256=binding.representation_sha256,
                                     ordered_view_text_sha256=binding.ordered_view_text_sha256,
                                     probe_fold_hash=fold_hash,readout_oof_source=str(op),
                                     readout_source_n=2256,readout_original_scope="three_cell_lines_entity_OOF",
                                     geometry_scope=cfg["geometry_scope"],effect_definition_id=cfg["effect_definition_id"],
                                     readout_status="VALID",n_independent_studies=1))
                else:
                    scope = "source_fold_fitted" if model in ["tfidf","structured_onehot"] else "fixed_random_rule_not_cross_layer_bitwise_bound"
                    baseline_read.append(dict(model=model,view=view,cell_line=cell,**group_info[cell],**metric,
                                              readout_fit_scope=scope,paired_across_layers=False,readout_oof_source=str(op)))
                    baseline_geom.append(dict(model=model,view=view,cell_line=cell,**group_info[cell],**geometry,
                                              geometry_fit_scope="full_primary_fitted" if model in ["tfidf","structured_onehot"] else "fixed_random_rule_not_cross_layer_bitwise_bound",
                                              paired_across_layers=False))
    bframe = pd.DataFrame(base)
    require(len(bframe)==72 and not bframe.duplicated(["model","view","cell_line"]).any(),"Base cardinality")
    lookup = bframe.set_index(["model","view","cell_line"])
    pairs = []
    for contrast in cfg["contrasts"]:
        for model in models:
            for cell in cells:
                src = lookup.loc[(model,contrast["view_from"],cell)]
                dst = lookup.loc[(model,contrast["view_to"],cell)]
                for key in ["cohort_id","gene_axis_id","representation_sha256","probe_fold_hash","effect_definition_id"]:
                    require(src[key]==dst[key],f"Unmatched contrast {key}")
                record = dict(model=model,cell_line=cell,**contrast,**group_info[cell],
                              representation_sha256=src.representation_sha256,probe_fold_hash=fold_hash,
                              effect_definition_id=cfg["effect_definition_id"],
                              view_text_hash_from=src.ordered_view_text_sha256,view_text_hash_to=dst.ordered_view_text_sha256,
                              scope=cfg["geometry_scope"],n_independent_studies=1)
                for metric in ["dose_accuracy","dose_macro_f1","rsa","excess_ndcg_mean"]:
                    record[metric+"_from"]=float(src[metric])
                    record[metric+"_to"]=float(dst[metric])
                    record["delta_"+metric]=float(dst[metric]-src[metric])
                record["status"]="VALID" if all(np.isfinite(record["delta_"+m]) for m in ["dose_accuracy","dose_macro_f1","rsa","excess_ndcg_mean"]) else "RETAINED_SOURCE_NA"
                pairs.append(record)
    pframe = pd.DataFrame(pairs)
    require(len(pframe)==72 and pframe.groupby("contrast_family").size().to_dict()==
            {"exposure_addition":36,"context_addition_sensitivity":36},"Paired cardinality")
    require(len(baseline_read)==len(baseline_geom)==36, "Baseline reference cardinality")
    atom_join = metadata.merge(folds[["atomic_id","fold"]],on="atomic_id",validate="one_to_one")
    row_e2 = {key:i for i,key in enumerate(geom_meta.atomic_id)}
    atom_join["e01_row_0based"]=np.arange(len(atom_join))
    atom_join["e02_row_0based"]=atom_join.atomic_id.map(row_e2).astype("Int64")
    atom_join["included_R1"]=atom_join.atomic_id.isin(common.atomic_id)
    atom_join["exclusion_reason"]=np.where(atom_join.included_R1,"","ORIGINAL_MAIN_INELIGIBLE_YM155")
    for _,r in excluded.iterrows():
        na_rows.append(dict(kind="ORIGINAL_EXCLUDED_ATOM",model="ALL",view="ALL",cell_line=r.cell_line,
                            atomic_id=r.atomic_id,field="",reason="ORIGINAL_MAIN_INELIGIBLE_YM155",n_affected=1))
    drift = [str(p) for p in source_paths if digest(p)!=before[str(p)]["sha256"] or
             str(p.stat().st_mtime_ns)!=before[str(p)]["mtime_ns"]]
    require(not drift, "Source changed during run: "+str(drift))
    products = {
        "r1_base_metrics.tsv":bframe,"r1_paired_changes.tsv":pframe,
        "baseline_readout_reference.tsv":baseline_read,"baseline_geometry_reference.tsv":baseline_geom,
        "atom_join.tsv":atom_join,"label_support.tsv":support,"na_and_exclusions.tsv":na_rows,
        "cohort_manifest.tsv":axis_records,"representation_bindings.tsv":binding_records,
        "oof_integrity_checks.tsv":oof_checks,"dose_confusion_counts.tsv":confusion,
    }
    for name,content in products.items():
        write_tsv(outputs/name,content)
    elapsed=time.perf_counter()-started
    audit=dict(status="PASS",n_base=72,n_pairs=72,n_readout_baseline=36,n_geometry_baseline=36,
               n_common_atoms=2250,n_original_excluded=6,n_oof_files=36,
               n_base_metric_na=int(bframe[["dose_accuracy","dose_macro_f1","rsa","excess_ndcg_mean"]].isna().sum().sum()),
               sources_unchanged=True,probe_refitted=False,new_encoding=False,geometry_recomputed=False,
               other_branches_run=False,statistical_tests=False,figures=False,manuscript_edit=False,
               seconds=elapsed,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,
               created_utc=datetime.now(timezone.utc).isoformat(),contract_sha256=digest(cfg_path),
               producer_sha256=digest(Path(__file__)),
               outputs=[dict(path=str(outputs/name),sha256=digest(outputs/name)) for name in products])
    write_json(run_dir/"qa/producer_audit.json",audit)
    print(json.dumps(audit,ensure_ascii=False),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--run-dir",type=Path,required=True)
    args=parser.parse_args()
    try:
        run(args.run_dir.resolve())
    except Exception as exc:
        failure=args.run_dir/"qa"/("producer_failure_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+".json")
        write_json(failure,dict(status="STOPPED",error_type=type(exc).__name__,message=str(exc),
                                no_automatic_source_repair=True))
        raise
