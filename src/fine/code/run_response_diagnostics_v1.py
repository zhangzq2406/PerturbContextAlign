#!/usr/bin/env python3
"""Source-reference and paired-measurement diagnostics on frozen predictions.

No refitting, sample selection, low-rank projection or new gene selection.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with path.open("x") as handle:
        json.dump(value,handle,indent=2,allow_nan=False)
        handle.write("\n")


def tsv(path):
    return pd.read_csv(path,sep="\t",keep_default_na=False,na_values=["NA"],float_precision="round_trip")


def flag(values):
    values = pd.Series(values).astype(str).str.lower()
    assert values.isin(["true","false"]).all()
    return values.eq("true").to_numpy()


def row_rho(first, second):
    a,b = rankdata(first,axis=1,method="average"),rankdata(second,axis=1,method="average")
    a -= a.mean(axis=1,keepdims=True)
    b -= b.mean(axis=1,keepdims=True)
    denominator = np.sqrt((a*a).sum(axis=1)*(b*b).sum(axis=1))
    return np.divide((a*b).sum(axis=1),denominator,out=np.full(len(a),np.nan),where=denominator>0)


def scalar_rho(a,b):
    if np.ptp(a)==0 or np.ptp(b)==0:
        return np.nan
    return float(spearmanr(a,b).statistic)


def run(config_path):
    started = time.perf_counter()
    config = json.loads(config_path.read_text())
    output = ROOT/config["output_root"]
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    write_json(output/"frozen_contract.json",config)
    original, effects = ROOT/config["prediction_root"], ROOT/config["effects_root"]
    qa = json.loads((ROOT/config["prediction_independent_qa"]).read_text())
    production = json.loads((original/"audit.json").read_text())
    assert qa["status"] == production["status"] == "PASS"
    assert digest(original/"audit.json") == qa["production_audit_sha256"]
    inputs = [config_path,Path(__file__),ROOT/config["prediction_independent_qa"],original/"audit.json",effects/"audit.json"]
    effect_audit = json.loads((effects/"audit.json").read_text())
    for name in ["arrays.npz","atomic_index.tsv","fold_gene_panels.tsv","measurement_index.tsv","measurement_means.npz"]:
        path = effects/name
        assert digest(path) == effect_audit["output_sha256"][name]
        inputs.append(path)
    for line in config["folds"]:
        for name in ["frozen_predictions.npz","evaluation_truth.npz","source_atoms.tsv","gene_panel.tsv",
                     "primary_condition_metrics.tsv","restricted_evaluation_condition_metrics.tsv"]:
            inputs.append(original/line/name)
        path = original/line/"frozen_predictions.npz"
        assert digest(path) == qa["frozen_prediction_sha256_unchanged"][str(path)]
    hashes = {str(p):digest(p) for p in inputs}
    write_json(output/"input_manifest.json",hashes)
    metadata, measurement = tsv(effects/"atomic_index.tsv"),tsv(effects/"measurement_index.tsv")
    assert len(metadata)==2256 and metadata.atomic_id.is_unique and len(measurement)==4512
    assert measurement.groupby("atomic_id").replicate.nunique().eq(2).all()
    primary, restricted = flag(metadata.main_eligible),flag(metadata.sensitivity_eligible)
    with np.load(effects/"arrays.npz",allow_pickle=False) as data:
        np.testing.assert_array_equal(data["atomic_id"],metadata.atomic_id)
        all_effects,feature_axis=data["effect_log2fc"],data["source_feature_row"]
    with np.load(effects/"measurement_means.npz",allow_pickle=False) as data:
        np.testing.assert_array_equal(data["measurement_id"],measurement.measurement_id)
        np.testing.assert_array_equal(data["source_feature_row"],feature_axis)
        cache_dtypes = {key:str(data[key].dtype) for key in ["normalized_mean","control_B_mean"]}
        treatment,reference = data["normalized_mean"].astype(float),data["control_B_mean"].astype(float)
    measurement_rows = {(row.atomic_id,row.replicate):i for i,row in enumerate(measurement.itertuples())}
    condition_tables,repeat_tables,checks = [],[],[]
    for line in config["folds"]:
        print("RESPONSE_DIAGNOSTICS_START",line,flush=True)
        fold = output/line
        fold.mkdir()
        train = np.flatnonzero(primary & metadata.cell_line.ne(line))
        test = np.flatnonzero(primary & metadata.cell_line.eq(line))
        assert len(train)==1500 and len(test)==750 and restricted[test].sum()==734
        np.testing.assert_array_equal(tsv(original/line/"source_atoms.tsv").atomic_id,metadata.iloc[train].atomic_id)
        panel = tsv(original/line/"gene_panel.tsv")
        columns = pd.Index(feature_axis).get_indexer(panel.source_feature_row)
        assert len(columns)==3000 and (columns>=0).all()
        truth = all_effects[np.ix_(test,columns)].astype(float)
        mean = all_effects[np.ix_(train,columns)].astype(float).mean(axis=0)
        with np.load(original/line/"frozen_predictions.npz",allow_pickle=False) as data:
            predictions,methods = data["predictions"],data["method"].tolist()
            assert methods==production["method_order"] and len(methods)==29
            np.testing.assert_array_equal(data["atomic_id"],metadata.iloc[test].atomic_id)
            np.testing.assert_array_equal(data["source_feature_row"],panel.source_feature_row)
        references = {"source_training_mean":np.broadcast_to(mean,truth.shape),
            "same_drug_source_reference":predictions[methods.index("same_drug_dose_mean")].astype(float)}
        np.savez_compressed(fold/"source_references.npz",source_training_mean=mean,
            same_drug_source_reference=references["same_drug_source_reference"],atomic_id=metadata.iloc[test].atomic_id.to_numpy(str),
            source_atomic_id=metadata.iloc[train].atomic_id.to_numpy(str),source_feature_row=panel.source_feature_row.to_numpy())
        saved_raw = tsv(original/line/"primary_condition_metrics.tsv")
        true_rank = rankdata(truth,axis=0)
        np.testing.assert_array_equal(true_rank,rankdata(truth-mean,axis=0))
        for i,method in enumerate(methods):
            predicted = predictions[i].astype(float)
            original_rho = row_rho(predicted,truth)
            raw = saved_raw[saved_raw.method.eq(method)].set_index("atomic_id").loc[metadata.iloc[test].atomic_id]
            np.testing.assert_allclose(original_rho,raw.spearman,atol=1e-10,rtol=0,equal_nan=True)
            raw_error = predicted-truth
            raw_mae,raw_rmse=np.abs(raw_error).mean(axis=1),np.sqrt((raw_error*raw_error).mean(axis=1))
            np.testing.assert_allclose(raw_mae,raw.mae,atol=1e-10,rtol=0)
            np.testing.assert_allclose(raw_rmse,raw.rmse,atol=1e-10,rtol=0)
            np.testing.assert_array_equal(rankdata(predicted,axis=0),rankdata(predicted-mean,axis=0))
            for kind,r in references.items():
                observed_delta,predicted_delta=truth-r,predicted-r
                changed_rho = row_rho(predicted_delta,observed_delta)
                shifted_error=predicted_delta-observed_delta
                error_drift=float(np.max(np.abs(shifted_error-raw_error)))
                assert error_drift<=config["numerical_atol"]
                for j in range(3):
                    np.testing.assert_allclose(changed_rho[j],scalar_rho(predicted_delta[j],observed_delta[j]),
                                               atol=1e-10,rtol=0,equal_nan=True)
                if method=="same_drug_dose_mean" and kind=="same_drug_source_reference":
                    assert np.array_equal(predicted_delta,np.zeros_like(predicted_delta)) and np.isnan(changed_rho).all()
                frame=metadata.iloc[test][["atomic_id","source_entity_key","cell_line","dose_value","time"]].reset_index(drop=True).copy()
                frame["method"],frame["transform"]=method,kind
                frame["raw_spearman"],frame["transformed_spearman"]=original_rho,changed_rho
                frame["paired_valid"]=np.isfinite(original_rho)&np.isfinite(changed_rho)
                frame["spearman_change"]=changed_rho-original_rho
                frame["original_mae"],frame["original_rmse"]=raw_mae,raw_rmse
                frame["sensitivity_eligible"]=restricted[test]
                frame["n_genes"]=3000
                condition_tables.append(frame)
                checks.append(dict(line=line,method=method,transform=kind,max_error_identity_drift=error_drift,
                    n_values=truth.size,source_mean_gene_ranks_exact=True,status="PASS"))
        rep1=np.array([measurement_rows[(key,"rep1")] for key in metadata.iloc[test].atomic_id])
        rep2=np.array([measurement_rows[(key,"rep2")] for key in metadata.iloc[test].atomic_id])
        r1=np.log2((treatment[np.ix_(rep1,columns)]+1)/(reference[np.ix_(rep1,columns)]+1))
        r2=np.log2((treatment[np.ix_(rep2,columns)]+1)/(reference[np.ix_(rep2,columns)]+1))
        assert np.isfinite(r1).all() and np.isfinite(r2).all()
        repeat=metadata.iloc[test][["atomic_id","source_entity_key","cell_line","dose_value","time"]].reset_index(drop=True).copy()
        repeat["replicate_spearman"]=row_rho(r1,r2)
        repeat["replicate_mean_absolute_difference"]=np.abs(r1-r2).mean(axis=1)
        repeat["rep1_mean_absolute_effect"],repeat["rep2_mean_absolute_effect"]=np.abs(r1).mean(axis=1),np.abs(r2).mean(axis=1)
        repeat["rep1_treated_cells"],repeat["rep2_treated_cells"]=measurement.iloc[rep1].n_treated_cells.to_numpy(),measurement.iloc[rep2].n_treated_cells.to_numpy()
        repeat["rep1_measurement_id"],repeat["rep2_measurement_id"]=measurement.iloc[rep1].measurement_id.to_numpy(),measurement.iloc[rep2].measurement_id.to_numpy()
        repeat["sensitivity_eligible"],repeat["n_genes"]=restricted[test],3000
        for j in range(3):
            np.testing.assert_allclose(repeat.replicate_spearman.iloc[j],scalar_rho(r1[j],r2[j]),atol=1e-10,rtol=0,equal_nan=True)
        repeat_tables.append(repeat)
        repeat.to_csv(fold/"replicate_agreement.tsv",sep="\t",index=False,na_rep="NA")
        print("RESPONSE_DIAGNOSTICS_FOLD_PASS",line,flush=True)
    condition=pd.concat(condition_tables,ignore_index=True)
    repeat=pd.concat(repeat_tables,ignore_index=True)
    condition.to_csv(output/"condition_response_diagnostics.tsv.gz",sep="\t",index=False,na_rep="NA")
    summaries,repeat_summary=[],[]
    for cohort in config["cohorts"]:
        c=condition if cohort=="primary" else condition[condition.sensitivity_eligible]
        for (line,method,transform),g in c.groupby(["cell_line","method","transform"],sort=False):
            summaries.append(dict(cohort=cohort,cell_line=line,method=method,transform=transform,n_queries=len(g),
                raw_spearman_mean=float(g.raw_spearman.mean()),raw_valid_n=int(g.raw_spearman.notna().sum()),
                transformed_spearman_mean=float(g.transformed_spearman.mean()),transformed_valid_n=int(g.transformed_spearman.notna().sum()),
                paired_spearman_change_mean=float(g.spearman_change.mean()),paired_valid_n=int(g.paired_valid.sum()),
                n_independent_studies=1))
        r=repeat if cohort=="primary" else repeat[repeat.sensitivity_eligible]
        for (line,dose),g in r.groupby(["cell_line","dose_value"],sort=False):
            repeat_summary.append(dict(cohort=cohort,cell_line=line,dose_value=dose,n_atoms=len(g),n_valid=int(g.replicate_spearman.notna().sum()),
                replicate_spearman_median=float(g.replicate_spearman.median()),replicate_spearman_q25=float(g.replicate_spearman.quantile(.25)),
                replicate_spearman_q75=float(g.replicate_spearman.quantile(.75)),
                mean_absolute_replicate_difference=float(g.replicate_mean_absolute_difference.mean()),n_independent_studies=1))
    summary=pd.DataFrame(summaries)
    summary.to_csv(output/"fold_cohort_summary.tsv",sep="\t",index=False,na_rep="NA")
    pd.DataFrame(repeat_summary).to_csv(output/"replicate_dose_cohort_summary.tsv",sep="\t",index=False,na_rep="NA")
    assert len(condition)==130500 and len(summary)==348 and len(repeat)==2250
    for path,checksum in hashes.items():
        assert digest(path)==checksum
    write_json(output/"arithmetic_invariance_checks.json",checks)
    write_json(output/"audit.json",dict(status="PASS",independent_acceptance="PENDING_SEPARATE_CHECKER",config=config,
        n_methods=29,n_folds=3,n_independent_studies=1,n_condition_diagnostic_rows=len(condition),n_fold_cohort_summaries=len(summary),
        n_paired_measurement_atoms=len(repeat),cache_dtypes=cache_dtypes,full_raw_reaggregation=False,
        error_identity_max_drift=max(v["max_error_identity_drift"] for v in checks),source_mean_gene_ranks_exact=True,
        no_new_fits=True,no_reliability_based_exclusion=True,all_input_hashes_unchanged=True,
        output_sha256={str(p.relative_to(output)):digest(p) for p in output.rglob("*") if p.is_file()},seconds=time.perf_counter()-started))
    print("RESPONSE_DIAGNOSTICS_PASS",time.perf_counter()-started,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"configs/response_diagnostics_v1.json")
    args=parser.parse_args()
    with threadpool_limits(limits=4):
        run(args.config)
