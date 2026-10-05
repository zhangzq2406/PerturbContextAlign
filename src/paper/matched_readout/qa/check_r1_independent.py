#!/usr/bin/env python3
"""Independent R1 QA: counters, original OOF rows, and frozen geometry copies.

Never imports or reads the producer. Only qa/ output files are writable.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
import os
A = Path(os.environ['PCA_DEEP_ROOT'])
ROOT = Path(__file__).resolve().parents[1]
QA = Path(__file__).resolve().parent
R0 = ROOT.parent/'R0_execution_gate_20260924T170450534123665Z'
MODELS = ('bge_m3', 'sapbert', 'qwen3_0_6b', 'biomedbert', 'medcpt_article', 'medcpt_query')
BASELINES = ('tfidf', 'structured_onehot', 'random_field512')
VIEWS = ('entity', 'entity_exposure', 'entity_context', 'complete_metadata')
LINES = ('A549', 'K562', 'MCF7')
DOSES = ('10', '100', '1000', '10000')
CONTRASTS = {
    ('entity', 'entity_exposure'): 'exposure',
    ('entity_context', 'complete_metadata'): 'exposure',
    ('entity', 'entity_context'): 'context',
    ('entity_exposure', 'complete_metadata'): 'context',
}
GEOMETRY = ('rsa', 'ndcg_mean', 'random_ndcg_mean', 'excess_ndcg_mean')
DELTA = {'delta_dose_accuracy': 'dose_accuracy', 'delta_dose_macro_f1': 'dose_macro_f1',
         'delta_rsa': 'rsa', 'delta_excess_ndcg_mean': 'excess_ndcg_mean'}
READS: dict[str, str] = {}
COUNTS: Counter = Counter()


def ensure(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def table(path):
    path = Path(path)
    READS.setdefault(str(path), sha(path))
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream, delimiter='\t'))


def payload(path):
    path = Path(path)
    READS.setdefault(str(path), sha(path))
    return json.loads(path.read_text())


def numeric(value):
    if value is None or str(value).strip() in ('', 'NA', 'NaN', 'nan', 'None'):
        return None
    result = float(value)
    ensure(math.isfinite(result), f'Unexpected nonfinite value: {value}')
    return result


def canonical_dose(value):
    value = float(value)
    ensure(value.is_integer(), f'Noninteger dose: {value}')
    return str(int(value))


def close(actual, expected, message, exact=False):
    a, b = numeric(actual), numeric(expected)
    if a is None or b is None:
        ensure(a is b, f'{message}: NA mismatch {actual!r} / {expected!r}')
    else:
        ensure(a == b if exact else abs(a-b) <= 1e-12,
               f'{message}: {actual!r} / expected {expected!r}')
    COUNTS['numeric_checks'] += 1


def unique(rows, keys, message):
    out = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        ensure(key not in out, f'{message}: duplicate {key}')
        out[key] = row
    return out


def manual_scores(rows, prediction_column):
    confusion = Counter()
    for row in rows:
        true = canonical_dose(row['dose_value'])
        prediction = canonical_dose(row[prediction_column])
        ensure(true in DOSES and prediction in DOSES, 'Unexpected dose class')
        confusion[true, prediction] += 1
    n = sum(confusion.values())
    f1s = []
    for label in DOSES:
        tp = confusion[label, label]
        fp = sum(v for (t, p), v in confusion.items() if p == label and t != label)
        fn = sum(v for (t, p), v in confusion.items() if t == label and p != label)
        denominator = 2*tp + fp + fn
        f1s.append(2*tp/denominator if denominator else 0.0)
    return sum(confusion[d, d] for d in DOSES)/n, sum(f1s)/4, confusion


def required_result(name):
    path = ROOT/'results'/name
    ensure(path.is_file(), f'Missing completed production output: {path}')
    return table(path)


def prepare_reference():
    info = A/'metrics/information_fidelity_v1'
    alignment = A/'metrics/effect_alignment_v1'
    atoms = table(info/'atomic_index.tsv')
    atom_map = unique(atoms, ('atomic_id',), 'Original atomic metadata')
    ensure(len(atoms) == 2256, 'Original cohort must have 2256 atoms')
    primary = table(alignment/'atomic_index.tsv')
    primary_map = unique(primary, ('atomic_id',), 'Primary cohort')
    ensure(len(primary) == 2250, 'Primary cohort must have 2250 atoms')
    ensure(set(primary_map) == {k for k,r in atom_map.items() if r['main_eligible'] == 'True'},
           'E01 main_eligible differs from E02 primary cohort')
    for key,row in primary_map.items():
        original = atom_map[key]
        ensure(all(row[k] == original[k] for k in ('atomic_id','source_entity_key','cell_line','dose_value','time','dose_unit')),
               f'Original/primary semantic mismatch: {key}')
    excluded = [r for k,r in atom_map.items() if k not in primary_map]
    ensure(len(excluded) == 6 and {r['source_entity_name'] for r in excluded} == {'YM155 (Sepantronium Bromide)'}
           and {r['source_entity_key'] for r in excluded} == {'localentity:1ee163940bc4f2734d95d8de'},
           'Unexpected excluded entities')
    ensure({canonical_dose(r['dose_value']) for r in excluded} == {'100','1000'}, 'Unexpected excluded doses')
    ensure({(r['cell_line'],canonical_dose(r['dose_value'])) for r in excluded} ==
           {(line,dose) for line in LINES for dose in ('100','1000')}, 'Excluded line/dose memberships')
    folds = table(info/'folds.tsv')
    fold_map = unique(folds, ('atomic_id',), 'Original folds')
    ensure(set(fold_map) == set(atom_map), 'Fold/atom membership mismatch')
    ensure(Counter(int(r['fold']) for r in folds) == {0:456,1:456,2:456,3:444,4:444}, 'Original fold supports changed')
    entity_folds = defaultdict(set)
    for row in folds:
        entity_folds[row['source_entity_key']].add(row['fold'])
    ensure(len(entity_folds) == 188 and all(len(v)==1 for v in entity_folds.values()), 'Entity leakage between folds')
    groups = {}
    for line in LINES:
        part = [r for r in primary if r['cell_line']==line]
        groups[line] = part
        ensure(len(part)==750 and len({r['source_entity_key'] for r in part})==188, f'{line}: cohort support')
        ensure(Counter(canonical_dose(r['dose_value']) for r in part)==dict(zip(DOSES,(188,187,187,188))), f'{line}: dose support')
        ensure({r['time'] for r in part}=={'24'} and {r['dose_unit'] for r in part}=={'nM'}, 'Time/unit drift')
    all_geometry = table(alignment/'group_summary.tsv')
    old_geometry = [dict(r,source_row_0based=i) for i,r in enumerate(all_geometry) if r['group_kind']=='within_cell_line']
    geometry = unique(old_geometry, ('method','view','cell_line'), 'Frozen within-cell-line geometry')
    expected = {}
    confusion_rows = []
    for model in MODELS+BASELINES:
        for view in VIEWS:
            data = table(info/'readout'/model/view/'oof_predictions.tsv')
            mapped = unique(data, ('atomic_id',), f'OOF {model}/{view}')
            ensure(set(mapped)==set(atom_map), f'OOF atom set {model}/{view}')
            for key,row in mapped.items():
                original = atom_map[key]
                ensure(all(row[k]==original[k] for k in ('source_entity_key','cell_line','dose_value')), f'OOF labels {model}/{view}/{key}')
                ensure(row['fold']==fold_map[key]['fold'], f'OOF fold {model}/{view}/{key}')
            for line in LINES:
                selected = [mapped[(r['atomic_id'],)] for r in groups[line]]
                ensure({r['fold'] for r in selected}=={'0','1','2','3','4'}, f'{model}/{view}/{line}: five folds lost')
                accuracy, macro_f1, confusion = manual_scores(selected, 'dose_value_prediction')
                majority_accuracy, majority_f1, _ = manual_scores(selected, 'dose_value_training_majority')
                ref = geometry[(model,view,line)]
                ensure(ref['group_id']==line and int(ref['n_atoms'])==750 and int(ref['n_genes'])==3000,
                       f'{model}/{view}/{line}: wrong frozen geometry cohort')
                entry = dict(model=model,view=view,cell_line=line,n_conditions=750,n_entities=188,
                             dose_accuracy=accuracy,dose_macro_f1=macro_f1,
                             training_majority_accuracy=majority_accuracy,training_majority_macro_f1=majority_f1)
                entry.update({key:numeric(ref[key]) for key in GEOMETRY})
                entry.update(geometry_source_row_0based=ref['source_row_0based'],rsa_status=ref['rsa_status'],
                             valid_ndcg_n=int(ref['valid_ndcg_n']))
                expected[model,view,line] = entry
                for true in DOSES:
                    for prediction in DOSES:
                        confusion_rows.append(dict(model=model,view=view,cell_line=line,true_dose=true,
                                                   predicted_dose=prediction,n=confusion[true,prediction]))
            COUNTS['original_oof_files'] += 1
    e01 = payload(info/'input_manifest.json')
    e02 = payload(alignment/'input_manifest.json')
    e01 = {r['path']:r['sha256'] for r in e01}
    e02 = {r['path']:r['sha256'] for r in e02}
    overlap = set(e01)&set(e02)
    ensure(len(overlap)==13 and all(e01[k]==e02[k] for k in overlap), 'Frozen source identity mismatch')
    for rel in ('qa/information_fidelity_independent_v1_format_compatible/audit.json',
                'qa/effect_alignment_independent_v1/audit.json'):
        ensure(payload(A/rel)['status']=='PASS', f'Existing QA not PASS: {rel}')
    return expected, primary, atom_map, fold_map, excluded, confusion_rows


def check_outputs(expected):
    base = required_result('r1_base_metrics.tsv')
    readout = required_result('baseline_readout_reference.tsv')
    geometry = required_result('baseline_geometry_reference.tsv')
    key_fields = ('model','view','cell_line')
    for rows, models, label, metrics in (
        (base, MODELS, 'neural base', ('dose_accuracy','dose_macro_f1')+GEOMETRY),
        (readout, BASELINES, 'baseline readout', ('dose_accuracy','dose_macro_f1')),
        (geometry, BASELINES, 'baseline geometry', GEOMETRY),
    ):
        mapping = unique(rows, key_fields, label)
        keys = {(m,v,l) for m in models for v in VIEWS for l in LINES}
        ensure(set(mapping)==keys, f'{label}: incomplete/extra keys')
        for key,row in mapping.items():
            ref = expected[key]
            if 'n_conditions' in row: ensure(int(row['n_conditions'])==750, f'{label}/{key}: n_conditions')
            if 'n_entities' in row: ensure(int(row['n_entities'])==188, f'{label}/{key}: n_entities')
            for metric in metrics: close(row[metric],ref[metric], f'{label}/{key}/{metric}')
            if 'geometry_source_row_0based' in row:
                ensure(int(row['geometry_source_row_0based'])==ref['geometry_source_row_0based'], f'{label}/{key}: source row')
                ensure(row['geometry_source']==str(A/'metrics/effect_alignment_v1/group_summary.tsv'),f'{label}/{key}: geometry source')
                ensure(row['rsa_status']==ref['rsa_status'] and int(row['valid_ndcg_n'])==ref['valid_ndcg_n'],f'{label}/{key}: NA/status denominator')
            if 'readout_oof_source' in row:
                ensure(row['readout_oof_source']==str(A/'metrics/information_fidelity_v1/readout'/key[0]/key[1]/'oof_predictions.tsv'),f'{label}/{key}: OOF source')
            if 'paired_across_layers' in row:
                ensure(row['paired_across_layers']=='False',f'{label}/{key}: baseline falsely paired')
            for metric in ('training_majority_accuracy','training_majority_macro_f1'):
                if metric in row: close(row[metric],ref[metric],f'{label}/{key}/{metric}')
            for forbidden in ('cell_line_accuracy','cell_line_macro_f1','time_accuracy','dose_unit_accuracy'):
                ensure(forbidden not in row or numeric(row[forbidden]) is None, f'Constant label scored: {forbidden}')
        COUNTS[label.replace(' ','_')+'_rows'] = len(rows)
    paired = required_result('r1_paired_changes.tsv')
    pair_map = unique(paired, ('model','cell_line','view_from','view_to'), 'Paired changes')
    keys = {(m,l,a,b) for m in MODELS for l in LINES for a,b in CONTRASTS}
    ensure(set(pair_map)==keys and len(paired)==72, 'Paired contrast coverage')
    family_map = defaultdict(set); contrast_map = defaultdict(set)
    for key,row in pair_map.items():
        m,line,before,after = key
        family_map[CONTRASTS[before,after]].add(row['contrast_family'])
        contrast_map[before,after].add(row['contrast_id'])
        for delta,metric in DELTA.items():
            a,b = expected[m,before,line][metric],expected[m,after,line][metric]
            value = None if a is None or b is None else b-a
            close(row[metric+'_from'],a,f'Delta endpoint {key}/{metric}_from')
            close(row[metric+'_to'],b,f'Delta endpoint {key}/{metric}_to')
            close(row[delta],value,f'Delta {key}/{delta}')
    ensure(all(len(v)==1 for v in family_map.values()) and len(set.union(*family_map.values()))==2, 'Contrast family mapping')
    ensure(sorted(Counter(r['contrast_family'] for r in paired).values())==[36,36], 'Two contrast families of 36')
    ensure(all(len(v)==1 for v in contrast_map.values()) and len(set.union(*contrast_map.values()))==4, 'Four distinct contrast IDs')
    COUNTS['paired_rows']=72
    return base, readout, geometry, paired


def check_auxiliary(primary, atom_map, fold_map, excluded):
    # Identity checks here are independent of production scoring implementation.
    join = required_result('atom_join.tsv')
    ensure('atomic_id' in join[0], 'atom_join lacks atomic_id')
    ensure(len(unique(join,('atomic_id',),'atom_join'))==2256,'atom_join size/uniqueness')
    primary_ids={r['atomic_id'] for r in primary}
    e01_rows={r['atomic_id']:i for i,r in enumerate(atom_map.values())}
    e02_rows={r['atomic_id']:i for i,r in enumerate(primary)}
    for row in join:
        ensure((row['atomic_id'],) in atom_map, 'atom_join unknown atom')
        original = atom_map[row['atomic_id'],]
        for key in ('source_entity_key','cell_line','dose_value'):
            if key in row: ensure(row[key]==original[key], f'atom_join {key}')
        if 'fold' in row: ensure(row['fold']==fold_map[row['atomic_id'],]['fold'], 'atom_join fold')
        ensure(int(row['e01_row_0based'])==e01_rows[row['atomic_id']],'atom_join E01 row')
        ensure((row['included_R1']=='True')==(row['atomic_id'] in primary_ids),'atom_join inclusion')
        if row['atomic_id'] in primary_ids:
            ensure(int(row['e02_row_0based'])==e02_rows[row['atomic_id']],'atom_join E02 row')
            ensure(row['exclusion_reason']=='','Included atom marked excluded')
        else:
            ensure(numeric(row['e02_row_0based']) is None and row['exclusion_reason'],'Excluded atom lacks accounting')
    ensure(primary_ids == {r['atomic_id'] for r in join if r['included_R1']=='True'}, 'atom_join primary set')
    support = required_result('label_support.tsv')
    expected_support={}
    for line in LINES:
        for dose,n in zip(DOSES,(188,187,187,188)):
            expected_support[line,'dose_value',dose]=(n,4,'True')
        for field,label in [('cell_line',line),('time','24'),('dose_unit','nM')]:
            expected_support[line,field,label]=(750,1,'False')
    support_map=unique(support,('cell_line','field','label'),'label support')
    ensure(set(support_map)==set(expected_support),'Label-support keys')
    for key,row in support_map.items():
        ensure((int(row['n_conditions']),int(row['n_classes']),row['evaluated'])==expected_support[key],f'Label support {key}')
    exclusions = required_result('na_and_exclusions.tsv')
    ensure(bool(exclusions), 'Empty exclusion/NA accounting')
    old_exclusions=[r for r in exclusions if r['kind']=='ORIGINAL_EXCLUDED_ATOM']
    ensure(len(old_exclusions)==6 and {r['atomic_id'] for r in old_exclusions}=={r['atomic_id'] for r in excluded},'Six exact original exclusions')
    constants=[r for r in exclusions if r['kind']=='UNEVALUATED_CONSTANT_LABEL']
    ensure(len(constants)==9 and {(r['cell_line'],r['field']) for r in constants}==
           {(line,field) for line in LINES for field in ('cell_line','time','dose_unit')},'Constant-label exclusion accounting')
    blocked=[r for r in exclusions if r['kind']=='BLOCKED_EVALUATION_UNIT']
    ensure(len(blocked)==24 and {(r['cell_line'],r['field'],r['reason']) for r in blocked}==
           {(line,field,'CONSTANT_WITHIN_CELL_LINE_DOSE_'+dose) for line in LINES for field in ('cell_line','dose_value') for dose in DOSES},'Blocked fixed-dose accounting')
    for row in constants:ensure(int(row['n_affected'])==750,'Constant-label denominator')
    for row in blocked:
        dose=row['reason'].split('_')[-1]
        ensure(int(row['n_affected'])==dict(zip(DOSES,(188,187,187,188)))[dose],'Blocked-unit denominator')
    manifest = table(ROOT/'input_manifest.tsv')
    ensure(bool(manifest), 'Empty input manifest')
    manifest_map=unique(manifest,('path',),'input manifest')
    for path,checksum in READS.items():
        if path.startswith(str(A)+'/'):
            ensure((path,) in manifest_map, f'Consumed old source missing from production manifest: {path}')
            row=manifest_map[path,]
            ensure(row['sha256']==checksum, f'Input-manifest hash mismatch: {path}')
            ensure(int(row['size_bytes'])==Path(path).stat().st_size and int(row['mtime_ns'])==Path(path).stat().st_mtime_ns,f'Input-manifest stat mismatch: {path}')
            ensure(row['permission']=='read_only_input', f'Input permission marker: {path}')
    COUNTS['atom_join_rows']=len(join)
    COUNTS['label_support_rows']=len(support)
    COUNTS['na_exclusion_rows']=len(exclusions)
    COUNTS['input_manifest_rows']=len(manifest)


def check_r0_source_identities():
    rows=table(R0/'asset_manifest.tsv')
    recorded=defaultdict(set)
    for row in rows:
        if row.get('sha256_live'):
            recorded[row['path']].add(row['sha256_live'])
    checked=0
    for path,actual in READS.items():
        if path.startswith(str(A)+'/') and path in recorded:
            ensure(recorded[path]=={actual},f'Original source differs from R0 identity: {path}')
            checked+=1
    for model in MODELS+BASELINES:
        for view in VIEWS:
            path=str(A/'metrics/information_fidelity_v1/readout'/model/view/'oof_predictions.tsv')
            ensure(path in recorded and recorded[path]=={READS[path]},f'OOF R0 identity missing or changed: {path}')
    COUNTS['original_source_hashes_matched_to_r0']=checked


def sequence_hash(values):
    return hashlib.sha256(('\n'.join(map(str,values))+'\n').encode('utf-8')).hexdigest()


def check_identities(primary, base, readout, geometry, paired):
    # NumPy is used only for existing ID/gene axes, never metric computation.
    import numpy as np
    cohorts=unique(required_result('cohort_manifest.tsv'),('cell_line',),'cohort manifest')
    ensure(set(cohorts)=={(line,) for line in LINES},'Three cohort manifests')
    panels=table(A/'effects/atomic_effects_v1/fold_gene_panels.tsv')
    registry_path=A/'representations/clean_views_v1/row_to_text_registry.tsv'
    registry=unique([r for r in table(registry_path) if r['variant']=='source_name'],('view','atomic_id'),'Source text registry')
    rep_manifest={r['model_key']:r for r in payload(A/'metrics/information_fidelity_v1/representation_manifest.json')}
    input_manifest=unique(table(ROOT/'input_manifest.tsv'),('path',),'input manifest identity')
    folds_hash=sha(A/'metrics/information_fidelity_v1/folds.tsv')
    expected_identity={}; text_hashes={}
    for line in LINES:
        ids=[r['atomic_id'] for r in primary if r['cell_line']==line]
        cohort_hash=sequence_hash(ids)
        panel=[int(r['source_feature_row']) for r in sorted([r for r in panels if r['heldout_cell_line']==line],key=lambda r:int(r['rank']))]
        path=A/'metrics/effect_alignment_v1/groups'/line/'truth_geometry.npz'
        READS[str(path)]=sha(path)
        with np.load(path,allow_pickle=False) as source:
            ensure(source['atomic_id'].tolist()==ids,f'{line}: exact truth atom order')
            ensure(source['source_feature_row'].tolist()==panel,f'{line}: exact truth gene axis')
        ensure(input_manifest[str(path),]['sha256']==READS[str(path)],f'{line}: truth identity')
        identity=dict(cohort_id=f'sciplex:{line}:cross_dose:{cohort_hash[:16]}',cohort_sha256=cohort_hash,
                      gene_axis_id='source_feature_row:'+sequence_hash(panel),truth_cache_sha256=READS[str(path)],
                      probe_fold_hash=folds_hash)
        expected_identity[line]=identity
        row=cohorts[line,]
        for key,value in identity.items():ensure(row[key]==value,f'{line}: cohort identity {key}')
        ensure(row['global_primary_cohort_sha256']==sequence_hash(r['atomic_id'] for r in primary),'Global primary identity')
        ensure(json.loads(row['dose_support_json'])==dict(zip(DOSES,(188,187,187,188))),'Cohort dose JSON')
        for view in VIEWS:
            text_hashes[line,view]=sequence_hash(registry[view,aid]['text_id'] for aid in ids)
    binding=unique(required_result('representation_bindings.tsv'),('model','view','cell_line'),'Representation binding')
    ensure(set(binding)=={(m,v,l) for m in MODELS for v in VIEWS for l in LINES},'Binding coverage')
    for row in base+readout+geometry+paired:
        line=row['cell_line']
        for field,value in expected_identity[line].items():
            if field in row:ensure(row[field]==value,f'{line}: output identity {field}')
        for field,value in [('n_conditions',750),('n_entities',188),('n_dose_classes',4),('n_genes',3000),('n_pairs',280875)]:
            ensure(int(row[field])==value,f'{line}: output support {field}')
        if 'effect_definition_id' in row:
            ensure(row['effect_definition_id']=='sciplex_atomic_pooled_normalized_means_log2ratio_pc1_cosine_v1','Effect definition changed')
        if row['model'] in MODELS:
            rep=rep_manifest[row['model']]
            ensure(row['representation_sha256']==rep['sha256'],'Neural representation identity')
            ensure(input_manifest[rep['path'],]['sha256']==rep['sha256'],'Representation input hash binding')
        if 'view' in row and row['model'] in MODELS:
            key=row['model'],row['view'],line
            ensure(row['ordered_view_text_sha256']==text_hashes[line,row['view']],'Ordered text identity')
            bound=binding[key]
            ensure(bound['representation_path']==rep_manifest[row['model']]['path'],'Bound original embedding path')
            ensure(bound['representation_sha256']==row['representation_sha256'],'Binding representation hash')
            ensure(bound['registry_sha256']==READS[str(registry_path)],'Registry identity')
            ensure(bound['ordered_view_text_sha256']==text_hashes[line,row['view']],'Binding ordered text identity')
            ensure(bound['gene_axis_id']==expected_identity[line]['gene_axis_id'],'Binding gene axis')
        if 'view_from' in row:
            ensure(row['view_text_hash_from']==text_hashes[line,row['view_from']] and
                   row['view_text_hash_to']==text_hashes[line,row['view_to']],'Paired text identity')
    COUNTS['cohort_identity_groups']=3
    COUNTS['representation_binding_rows']=len(binding)


def direction(value):
    value=numeric(value)
    return 'NA' if value is None else 'numeric_zero' if abs(value)<=1e-12 else 'positive' if value>0 else 'negative'


def check_descriptive_tables(paired):
    summary=required_result('contrast_descriptive_summary.tsv')
    joint=required_result('joint_direction_counts.tsv')
    grouped=defaultdict(list)
    for row in paired:grouped[row['contrast_id']].append(row)
    metrics=('dose_accuracy','dose_macro_f1','rsa','excess_ndcg_mean')
    smap=unique(summary,('contrast_id','metric'),'Descriptive summary')
    ensure(set(smap)=={(cid,m) for cid in grouped for m in metrics},'All four metrics and contrasts in descriptive summary')
    for (cid,metric),row in smap.items():
        values=[numeric(r['delta_'+metric]) for r in grouped[cid]]
        finite=[v for v in values if v is not None]
        counts=Counter(direction(v) for v in values)
        ensure(int(row['n_records'])==18 and int(row['n_independent_studies'])==1,'Descriptive record/study n')
        for col,n in [('n_finite',len(finite)),('n_positive',counts['positive']),('n_negative',counts['negative']),('n_numeric_zero',counts['numeric_zero']),('n_na',counts['NA'])]:
            ensure(int(row[col])==n,f'Descriptive {cid}/{metric}/{col}')
        for col,value in [('mean_delta',sum(finite)/len(finite) if finite else None),('median_delta',statistics.median(finite) if finite else None),('min_delta',min(finite) if finite else None),('max_delta',max(finite) if finite else None)]:
            close(row[col],value,f'Descriptive {cid}/{metric}/{col}')
    jmap=unique(joint,('contrast_id','readout_metric','alignment_metric'),'Joint direction counts')
    ensure(set(jmap)=={(cid,r,a) for cid in grouped for r in ('dose_accuracy','dose_macro_f1') for a in ('rsa','excess_ndcg_mean')},'Joint direction endpoint coverage')
    for (cid,read,align),row in jmap.items():
        count=Counter((direction(r['delta_'+read]),direction(r['delta_'+align])) for r in grouped[cid])
        ensure(int(row['n_records'])==18 and int(row['n_independent_studies'])==1,'Joint direction n')
        for first in ('positive','negative','numeric_zero','NA'):
            for second in ('positive','negative','numeric_zero','NA'):
                ensure(int(row[first+'__'+second])==count[first,second],f'Joint directions {cid}/{read}/{align}/{first}/{second}')
    COUNTS['descriptive_summary_rows']=len(summary)
    COUNTS['joint_direction_rows']=len(joint)


def check_confusion_counts(expected):
    rows=required_result('dose_confusion_counts.tsv')
    actual=unique(rows,('model','view','cell_line','true_dose','predicted_dose'),'Confusion counts')
    reference=unique(expected,('model','view','cell_line','true_dose','predicted_dose'),'Independent confusions')
    ensure(set(actual)==set(reference),'Confusion matrix coverage')
    for key,row in actual.items():
        ensure(int(row['n'])==reference[key]['n'],f'Confusion count {key}')
        ensure(row['is_primary']==str(row['model'] in MODELS),f'Confusion main-method flag {key}')
    COUNTS['confusion_matrix_cells']=len(actual)


def write_table(name,rows):
    path=QA/name
    ensure(not path.exists(),f'Refuse to overwrite existing QA artifact: {path}')
    with path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),delimiter='\t')
        writer.writeheader()
        for row in rows: writer.writerow({k:'NA' if v is None else v for k,v in row.items()})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--production-complete',action='store_true',required=True)
    parser.add_argument('--audit-name',default='independent_audit.json')
    parser.add_argument('--artifact-prefix',default='independent')
    args=parser.parse_args()
    ensure(Path(args.audit_name).name==args.audit_name and args.audit_name.endswith('.json'),'QA audit must be a local JSON filename')
    ensure(args.artifact_prefix.replace('_','').replace('-','').isalnum(),'QA artifact prefix must be a simple filename prefix')
    ensure(not (QA/args.audit_name).exists(),'Refuse to overwrite an existing independent audit')
    ensure(QA==ROOT/'qa','Write directory is not qa/')
    status={'status':'FAIL','scope':'independent original-OOF re-scoring, frozen geometry copying, joins and differences',
            'producer_read_or_imported':False,'old_geometry_recomputed':False,'model_fitting_performed':False,
            'checker_sha256':sha(Path(__file__)),'qa_contract_sha256':sha(QA/'independent_qa_contract.md'),
            'audit_name':args.audit_name,'artifact_prefix':args.artifact_prefix}
    try:
        expected,primary,atoms,folds,excluded,confusions=prepare_reference()
        base,readout,geometry,paired=check_outputs(expected)
        check_auxiliary(primary,atoms,folds,excluded)
        check_identities(primary,base,readout,geometry,paired)
        check_descriptive_tables(paired)
        check_confusion_counts(confusions)
        check_r0_source_identities()
        ensure(all(sha(Path(path))==checksum for path,checksum in READS.items()),'Input/output file changed during independent QA')
        write_table(args.artifact_prefix+'_expected_base.tsv',list(expected.values()))
        write_table(args.artifact_prefix+'_dose_confusion_matrices.tsv',confusions)
        status.update(status='PASS',counts=dict(COUNTS),source_and_production_hashes=READS,
                      n_primary_atoms=2250,n_independent_studies=1,
                      limitation='Old geometry QA reused; no independent encoding, fitting, effect reconstruction, geometry recomputation or biological replication')
    except Exception as error:
        status.update(error_type=type(error).__name__,error=str(error),counts=dict(COUNTS))
        raise
    finally:
        path=QA/args.audit_name
        ensure(not path.exists(),f'Refuse to overwrite existing QA artifact: {path}')
        with path.open('x') as stream:json.dump(status,stream,indent=2,allow_nan=False);stream.write('\n')
        print(json.dumps({k:v for k,v in status.items() if k!='source_and_production_hashes'},allow_nan=False))


if __name__=='__main__':main()
