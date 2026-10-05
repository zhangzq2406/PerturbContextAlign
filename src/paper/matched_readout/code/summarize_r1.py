#!/usr/bin/env python3
"""Descriptive summaries of all prespecified R1 contrasts; no inference."""
import argparse,csv,hashlib,json,math,statistics
from pathlib import Path

def rows(p):
    with p.open() as f:return list(csv.DictReader(f,delimiter="\t"))
def num(x):
    return float("nan") if x=="NA" else float(x)
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def sign(x,tol):
    return "NA" if not math.isfinite(x) else "positive" if x>tol else "negative" if x < -tol else "numeric_zero"
def save(p,data):
    with p.open("x") as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]),delimiter="\t",lineterminator="\n")
        w.writeheader();w.writerows(data)
def main(root):
    cfg=json.loads((root/"execution_contract.json").read_text())
    src=root/"results/r1_paired_changes.tsv"
    before=sha(src); pairs=rows(src); assert len(pairs)==72
    tol=cfg["numeric_zero_tolerance"]
    metrics=["dose_accuracy","dose_macro_f1","rsa","excess_ndcg_mean"]
    summary=[]; joint=[]
    for c in cfg["contrasts"]:
        rs=[r for r in pairs if r["contrast_id"]==c["contrast_id"]]
        assert len(rs)==18
        for metric in metrics:
            xs=[num(r["delta_"+metric]) for r in rs]
            valid=[x for x in xs if math.isfinite(x)]
            row={**c,"metric":metric,"n_records":len(rs),"n_independent_studies":1,
                 "n_finite":len(valid),"n_positive":sum(sign(x,tol)=="positive" for x in xs),
                 "n_negative":sum(sign(x,tol)=="negative" for x in xs),
                 "n_numeric_zero":sum(sign(x,tol)=="numeric_zero" for x in xs),
                 "n_na":sum(not math.isfinite(x) for x in xs),
                 "mean_delta":statistics.fmean(valid) if valid else "NA",
                 "median_delta":statistics.median(valid) if valid else "NA",
                 "min_delta":min(valid) if valid else "NA","max_delta":max(valid) if valid else "NA",
                 "direction_rule":"abs(delta)<=1e-12 is numerical zero; not significance",
                 "aggregation":"descriptive_equal_weight_model_cell_line_records_not_independent_replicates"}
            summary.append(row)
        for readout in ["dose_accuracy","dose_macro_f1"]:
            for alignment in ["rsa","excess_ndcg_mean"]:
                row={**c,"readout_metric":readout,"alignment_metric":alignment,"n_records":18,"n_independent_studies":1}
                for a in ["positive","negative","numeric_zero","NA"]:
                    for b in ["positive","negative","numeric_zero","NA"]:
                        row[a+"__"+b]=sum(sign(num(r["delta_"+readout]),tol)==a and sign(num(r["delta_"+alignment]),tol)==b for r in rs)
                joint.append(row)
    assert before==sha(src)
    save(root/"results/contrast_descriptive_summary.tsv",summary)
    save(root/"results/joint_direction_counts.tsv",joint)
    audit={"status":"PASS","source_sha256":before,"script_sha256":sha(Path(__file__)),
           "contract_sha256":sha(root/"execution_contract.json"),"n_contrasts":4,
           "n_metric_summary_rows":16,"n_joint_summary_rows":16,"statistical_tests":False,
           "all_contrasts_retained":True,"numeric_zero_tolerance":tol,
           "outputs":{name:sha(root/"results"/name) for name in ["contrast_descriptive_summary.tsv","joint_direction_counts.tsv"]}}
    with (root/"qa/descriptive_summary_audit.json").open("x") as f:json.dump(audit,f,indent=2);f.write("\n")
    print(json.dumps(summary,ensure_ascii=False))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--run-dir",type=Path,required=True)
    main(p.parse_args().run_dir)
