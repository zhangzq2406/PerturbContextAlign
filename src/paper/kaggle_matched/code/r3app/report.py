"""Neutral descriptive reports; no significance tests or selected examples."""
from pathlib import Path
import pandas as pd
import numpy as np
from .common import load_table


def md_table(df,columns,labels=None):
    labels=labels or columns
    lines=['| '+' | '.join(labels)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for _,row in df.iterrows():
        vals=[]
        for col in columns:
            x=row[col]
            if isinstance(x,(float,np.floating)):s='NA' if not np.isfinite(x) else f'{x:.8g}'
            else:s=str(x)
            vals.append(s.replace('|','\\|'))
        lines.append('| '+' | '.join(vals)+' |')
    return '\n'.join(lines)


def run(out):
    out=Path(out);r=out/'results'
    join=pd.read_csv(r/'l2_l3_task_join.tsv',sep='\t',keep_default_na=False,na_values=['NA'])
    l3=pd.read_csv(r/'l3_task_metrics.tsv',sep='\t',keep_default_na=False,na_values=['NA'])
    macro=pd.read_csv(r/'l3_macro_summary.tsv',sep='\t',keep_default_na=False,na_values=['NA'])
    pair=pd.read_csv(r/'baseline_paired_differences.tsv',sep='\t',keep_default_na=False,na_values=['NA'])
    lines=['# R3 results — frozen Kaggle L2–L3 analysis','',
           '**Scientific scoring and independent numerical QA: PASS.**',
           'This is new scoring of fixed outputs, not a new model fit or a blinded confirmation study.','',
           '## Scope',
           'Six existing within-type donor-transfer tasks; 810 primary queries at 1 µM and 24 h. '
           'NK retains 136 drugs per task; CD4 retains 134. Original training/source sets are unchanged. '
           'Six 0.1 µM Belinostat conditions are omitted from the primary evaluation only. '
           'The six direct text representations contain drug, dose and duration, not donor or cell type.','',
           'Both layers use each task’s original evaluation_truth.npz, promoted from float32 to float64. '
           'Raw provided-scale matched-control expression differences are not sci-Plex log2FC.','',
           '## All 36 matched task/model results','',
           md_table(join,['cell_type','donor_id','model_key','rsa','excess_ndcg_mean','mae_mean',
                          'gene_spearman_mean','n_genes_valid']), '',
           'RSA describes all off-diagonal pairs; excess NDCG describes local neighbor recovery. '
           'MAE describes error over the fixed gene axis; gene Spearman describes drug ordering separately per gene. '
           'They are not subtracted or combined into a gap score.','',
           '## All selected L3 methods: equal-donor/equal-type descriptive means','',
           md_table(macro,['method','mae_mean','gene_spearman_mean','mae_mean__valid_types','gene_spearman_mean__valid_types']), '',
           'See results/l3_task_metrics.tsv for every baseline in every task. The hierarchy is equal donors '
           'within type, followed by equal types; counts of defined scores accompany all averages.','',
           '## Paired differences against each baseline','',
           'Positive MAE gain = baseline minus model. Positive ranking gain = model minus baseline '
           'on the common set of defined gene correlations. Different valid-gene sets are explicitly labeled. '
           'Counts below are dependent descriptive records, not tests or independent experiments.','']
    counts=[]
    for baseline,sub in pair.groupby('baseline',sort=False):
        for metric in ['mae_gain','gene_spearman_gain']:
            vals=sub[metric].to_numpy(float)
            counts.append({'baseline':baseline,'endpoint':metric,'positive':int(np.sum(vals>0)),
                           'negative':int(np.sum(vals<0)),'zero':int(np.sum(vals==0)),
                           'NA':int(np.sum(~np.isfinite(vals)))})
    lines += [md_table(pd.DataFrame(counts),['baseline','endpoint','positive','negative','zero','NA']), '',
              'No pooled task/model correlations, p-values, regression, bootstrap, or post-hoc high-alignment subsets were computed.','',
              '## Missingness and interpretation','',
              f"Across 78 method/task records the number of defined gene correlations ranges from {int(l3.n_genes_valid.min())} to {int(l3.n_genes_valid.max())} of 3000. "
              'Constant truth and constant prediction are distinguished; undefined correlations are never zero-filled. '
              'A low MAE for a constant predictor is not evidence of drug discrimination.','',
              'This joins L2 and L3 in one cohort; it does not measure donor/type information retention, '
              'unseen-drug performance, unseen-type performance, or cross-study training transfer. '
              'Donors, plates, libraries, shared controls, models and folds are not independent studies. '
              'The results do not determine whether any limitation comes from information content, the chosen geometry, '
              'the frozen decoder, or measurement structure.','',
              'R1’s exposure-addition improvements and R2’s repeatability evidence are not revised, recalibrated, '
              'or used as a normalization denominator by R3.','',
              '## Verification boundary','',
              'The independent checker rereads frozen embeddings, truth and predictions and scores all selected results '
              'without importing producer or frozen scoring functions. Input identity hashes are verified before and after. '
              'Shared code is limited to declared paths/method names and IO/axis validators. '
              'Raw-cell aggregation, encoder inference and prior model training are not redone. '
              'The review archive contains new tables, small geometries, code and QA—not the input prediction or expression matrices.','']
    (out/'R3_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    summary='\n'.join(['# R3 summary','',
        'SCIENTIFIC_STATUS=PASS','INDEPENDENT_NUMERIC_QA=PASS',
        'TASKS=6','PRIMARY_QUERY_ATOMS=810','DIRECT_TASK_RECORDS=36','DIRECT_QUERY_JOINS=4860',
        'BASELINE_TASK_RECORDS=42','TOTAL_L3_TASK_RECORDS=78','GENE_ROWS=234000',
        'BASELINE_PAIRED_ROWS=252','NEW_TRAINING=FALSE','NEW_ENCODING=FALSE','RAW_H5AD_READ=FALSE',
        'STATE_METHODS_SCORED=FALSE','NEW_INFERENCE_TESTS=FALSE','',
        'All numeric results and boundaries are in R3_RESULTS.md. Completion does not authorize additional analyses.',''])
    (out/'R3_SUMMARY.md').write_text(summary,encoding='utf-8')
    (out/'COMPACT_RESULTS.txt').write_text(summary+'\n'+macro.to_csv(sep='\t',index=False,na_rep='NA'),encoding='utf-8')
