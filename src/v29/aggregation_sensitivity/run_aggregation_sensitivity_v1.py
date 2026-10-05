#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

LANGS = ["bge_m3","qwen3_0_6b","sapbert","biomedbert","medcpt_article","medcpt_query"]
METRICS = ["spearman_rsa", "ndcg_gain_vs_matched_random"]

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--outdir', type=Path, required=True)
    args=ap.parse_args()
    out=args.outdir
    out.mkdir(parents=True, exist_ok=False)
    df=pd.read_csv(args.input, sep='\t')
    df=df[df.model_key.isin(LANGS)].copy()
    if set(df.model_key)!=set(LANGS): raise SystemExit('six encoder set mismatch')
    if df.dataset_id.nunique()!=9: raise SystemExit('expected 9 datasets')
    if df.duplicated(['dataset_id','model_key']).any(): raise SystemExit('duplicate dataset/model')
    if not np.isfinite(df[METRICS].to_numpy(float)).all(): raise SystemExit('nonfinite primary metric')
    for metric in METRICS:
        df[metric+'_rank']=df.groupby('dataset_id')[metric].rank(ascending=False, method='average')
    datasets=sorted(df.dataset_id.unique())
    # Primary equal-dataset summaries.
    primary=[]
    weighted=[]
    for (key,display),g in df.groupby(['model_key','model_display'], sort=False):
        for metric in METRICS:
            primary.append(dict(model_key=key, model_display=display, metric=metric,
                                aggregation='equal_dataset', n_datasets=len(g),
                                mean_score=float(g[metric].mean()), mean_rank=float(g[metric+'_rank'].mean())))
            weighted.append(dict(model_key=key, model_display=display, metric=metric,
                                 aggregation='response_atom_weighted', n_datasets=len(g),
                                 weight_sum=int(g.response_atom_n.sum()),
                                 weighted_mean_score=float(np.average(g[metric], weights=g.response_atom_n)),
                                 weighted_mean_rank=float(np.average(g[metric+'_rank'], weights=g.response_atom_n))))
    primary=pd.DataFrame(primary).sort_values(['metric','mean_rank','model_key'])
    weighted=pd.DataFrame(weighted).sort_values(['metric','weighted_mean_rank','model_key'])
    # LODO summaries use the same within-dataset ranks as the primary analysis, dropping one dataset at a time.
    lodo=[]
    for omit in datasets:
        keep=df[df.dataset_id.ne(omit)]
        for (key,display),g in keep.groupby(['model_key','model_display'], sort=False):
            for metric in METRICS:
                lodo.append(dict(omitted_dataset=omit, model_key=key, model_display=display,
                                 metric=metric, n_datasets=8,
                                 mean_score=float(g[metric].mean()), mean_rank=float(g[metric+'_rank'].mean())))
    lodo=pd.DataFrame(lodo)
    lodo['rank_within_omission']=lodo.groupby(['omitted_dataset','metric']).mean_rank.rank(method='average')
    # Top model and rank ranges across LODO.
    robust=[]
    for (key,display,metric),g in lodo.groupby(['model_key','model_display','metric'], sort=False):
        robust.append(dict(model_key=key, model_display=display, metric=metric,
                           lodo_mean_rank_mean=float(g.mean_rank.mean()),
                           lodo_mean_rank_min=float(g.mean_rank.min()),
                           lodo_mean_rank_max=float(g.mean_rank.max()),
                           lodo_top1_n=int((g.rank_within_omission==1).sum()),
                           lodo_n=len(g)))
    robust=pd.DataFrame(robust).sort_values(['metric','lodo_mean_rank_mean'])
    # Compact top-model summary.
    top=[]
    for metric in METRICS:
        p=primary[primary.metric.eq(metric)].sort_values('mean_rank').iloc[0]
        w=weighted[weighted.metric.eq(metric)].sort_values('weighted_mean_rank').iloc[0]
        lg=lodo[lodo.metric.eq(metric)]
        winners=(lg.loc[lg.rank_within_omission.eq(1),'model_key'].value_counts()).to_dict()
        top.append(dict(metric=metric, equal_dataset_top=p.model_key, equal_dataset_top_mean_rank=float(p.mean_rank),
                        response_atom_weighted_top=w.model_key, response_atom_weighted_top_mean_rank=float(w.weighted_mean_rank),
                        lodo_top_counts=json.dumps(winners, sort_keys=True)))
    top=pd.DataFrame(top)
    df.to_csv(out/'dataset_model_metrics_with_ranks.tsv', sep='\t', index=False)
    primary.to_csv(out/'equal_dataset_summary.tsv', sep='\t', index=False)
    weighted.to_csv(out/'response_atom_weighted_summary.tsv', sep='\t', index=False)
    lodo.to_csv(out/'leave_one_dataset_out_summary.tsv', sep='\t', index=False)
    robust.to_csv(out/'leave_one_dataset_out_robustness.tsv', sep='\t', index=False)
    top.to_csv(out/'aggregation_top_model_summary.tsv', sep='\t', index=False)
    manifest={
      'status':'PASS', 'version':'v1', 'input':str(args.input), 'input_sha256':sha256(args.input),
      'six_encoders':LANGS, 'metrics':METRICS,
      'primary_estimand':'equal dataset weight; within-dataset encoder ranks, then mean rank',
      'lodo':'drop one dataset, retain equal weight over remaining eight',
      'response_atom_weighted':'weights dataset-level scores and within-dataset ranks by response_atom_n; sensitivity only',
      'no_new_scientific_data':True,
    }
    (out/'audit.json').write_text(json.dumps(manifest, indent=2)+'\n')
    # Human-readable result, deliberately descriptive rather than inferential.
    lines=['AGGREGATION SENSITIVITY v1','='*80]
    for row in top.itertuples(index=False):
        lines += [f"metric={row.metric}",
                  f"  equal_dataset_top={row.equal_dataset_top} mean_rank={row.equal_dataset_top_mean_rank:.6f}",
                  f"  response_atom_weighted_top={row.response_atom_weighted_top} weighted_mean_rank={row.response_atom_weighted_top_mean_rank:.6f}",
                  f"  LODO_top_counts={row.lodo_top_counts}"]
    lines += ['', 'INTERPRETATION_BOUNDARY:',
              'LODO tests single-dataset dependence of the prespecified equal-dataset ranking.',
              'Response-atom weighting changes the estimand by giving large screens more influence; it is a sensitivity analysis, not a replacement primary analysis.',
              'No p-values or independent-replicate claim are introduced.']
    (out/'COMPACT.txt').write_text('\n'.join(lines)+'\n')
if __name__=='__main__': main()
