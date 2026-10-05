#!/usr/bin/env python3
"""Matched negative control for sci-Plex 3 mechanism-text augmentation.

This extension keeps the frozen 57-drug cohort, 3 cell lines, 4 doses, six
encoder snapshots, response effects, source splits, landmarks, alpha and gene
panels.  It replaces each drug's admitted mechanism text by another admitted
mechanism text using 20 deterministic text-level derangements.

Stages:
  preflight  - verify frozen inputs; build derangements/prompts; check token lengths
  encode     - smoke-check frozen encoder reproduction; encode shuffled full prompts
  geometry   - compare text geometry with measured perturbation-response geometry
  predict    - source-only fitting; freeze shuffled predictions before query truth access
  score      - score frozen predictions on held-out responses
  summarize  - create paper-facing compact tables and paired drug bootstrap for decomposable endpoints
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from scipy.stats import rankdata

MODELS = ["bge_m3","sapbert","qwen3_0_6b","biomedbert","medcpt_article","medcpt_query"]
PERM_SEEDS = [2026100501+i for i in range(20)]
N_PERM = len(PERM_SEEDS)
BOOTSTRAP_SEED = 2026100529
BOOTSTRAP_N = 2000
TOL_EMBED = 1e-4
TOL_REPRO = 5e-8


def now(): return datetime.now(timezone.utc).isoformat()
def require(x, msg):
    if not x: raise RuntimeError(msg)
def sha(path: Path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()
def json_write(path: Path, obj):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f"refuse overwrite: {path}")
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
def tsv(path: Path, df: pd.DataFrame):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f"refuse overwrite: {path}")
    df.to_csv(path, sep='\t', index=False, na_rep='NA')
def load_module(name: str, path: Path):
    spec=importlib.util.spec_from_file_location(name, path)
    require(spec and spec.loader, f"cannot load {path}")
    mod=importlib.util.module_from_spec(spec); sys.modules[name]=mod; spec.loader.exec_module(mod); return mod

def read_tsv(path): return pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False)
def bools(s):
    x=s.astype(str).str.lower(); require(x.isin(['true','false']).all(),'nonboolean'); return x.eq('true')

def modules(deep: Path):
    code=deep/'code'
    sys.path.insert(0, str(code))
    enc=load_module('pca_kenc', code/'encode_knowledge_views_v1.py')
    kg=load_module('pca_kg', code/'run_knowledge_alignment_v1.py')
    e08=load_module('pca_e08', deep/'extensions/e08_prediction/run_e08_prediction.py')
    return enc,kg,e08

def paths(deep: Path):
    return {
      'knowledge_cfg': deep/'configs/knowledge_encoding_v1.json',
      'e08_cfg': deep/'experiments/e08_prediction_v1/config.json',
      'coverage': deep/'metadata/knowledge_inputs_v1/entity_coverage.tsv',
      'krowmap': deep/'metadata/knowledge_inputs_v1/rowmap.tsv',
      'clean_rowmap': deep/'representations/clean_views_v1/row_to_text_registry.tsv',
      'clean_texts': deep/'representations/clean_views_v1/unique_texts.tsv',
      'knowledge_emb': deep/'representations/knowledge_embeddings_v1/full',
      'effects_index': deep/'effects/atomic_effects_v1/atomic_index.tsv',
      'effects_arrays': deep/'effects/atomic_effects_v1/arrays.npz',
      'fold_panels': deep/'effects/atomic_effects_v1/fold_gene_panels.tsv',
      'knowledge_metrics': deep/'metrics/knowledge_alignment_v1',
      'e08_results': deep/'experiments/e08_prediction_v1/results',
    }

def validate_inputs(deep: Path):
    p=paths(deep)
    for name,path in p.items():
        require(path.exists(), f"missing required {name}: {path}")
    return p

def load_mechanism_design(deep: Path, e08):
    p=validate_inputs(deep)
    cfg=json.loads(p['e08_cfg'].read_text())
    require(cfg['models']==MODELS, 'six-model order changed')
    require(cfg['alpha']==10 and cfg['n_landmarks']==256 and cfg['seed']==20260914, 'prediction policy changed')
    atoms, cohorts, panels=e08.load_metadata(cfg)
    meta=cohorts['mechanism57'].copy().reset_index(drop=True)
    require(len(meta)==684 and meta.source_entity_key.nunique()==57, 'mechanism57 cohort changed')
    coverage=read_tsv(p['coverage'])
    coverage=coverage.loc[bools(coverage.admitted_mechanism)].copy()
    require(len(coverage)==57 and coverage.source_entity_key.is_unique, 'mechanism entity universe changed')
    require(coverage.knowledge_text.ne('').all(), 'empty admitted mechanism text')
    mechanism=coverage.set_index('source_entity_key').knowledge_text.to_dict()
    require(set(mechanism)==set(meta.source_entity_key), 'mechanism/meta entity mismatch')
    # Base complete prompts.
    cr=read_tsv(p['clean_rowmap'])
    cr=cr.loc[cr.variant.eq('source_name') & cr.view.eq('complete_metadata')]
    require(cr.atomic_id.is_unique, 'base complete rowmap duplicate')
    ct=read_tsv(p['clean_texts']).set_index('text_id')
    base=cr.set_index('atomic_id').loc[meta.atomic_id, ['text_id','prompt_sha256']].copy()
    base['prompt_text']=ct.loc[base.text_id,'prompt_text'].to_numpy(str)
    # Correct CK prompt exact-construction check.
    kr=read_tsv(p['krowmap'])
    kr=kr.loc[kr.view.eq('complete_metadata+knowledge')]
    require(kr.atomic_id.is_unique, 'correct knowledge rowmap duplicate')
    kr=kr.set_index('atomic_id').loc[meta.atomic_id]
    for i,aid in enumerate(meta.atomic_id):
        expected=base.iloc[i].prompt_text+' '+mechanism[meta.iloc[i].source_entity_key]
        require(expected==kr.iloc[i].prompt_text, f'correct CK prompt construction differs at {aid}')
        digest=hashlib.sha256(expected.encode()).hexdigest()
        require(digest==kr.iloc[i].prompt_sha256 and kr.iloc[i].text_id=='text:'+digest, 'correct CK prompt digest mismatch')
    return cfg,atoms,meta,panels,mechanism,base.reset_index(drop=True),kr.reset_index()

def deterministic_derangements(entities, mechanism):
    entities=list(sorted(entities)); original=[mechanism[e] for e in entities]
    out=[]
    for pi,seed in enumerate(PERM_SEEDS):
        rng=np.random.default_rng(seed); accepted=None
        for attempt in range(1,200001):
            order=rng.permutation(len(entities))
            assigned=[original[j] for j in order]
            if all(order[i]!=i and assigned[i]!=original[i] for i in range(len(entities))):
                accepted=(order,assigned,attempt); break
        require(accepted is not None, f'no text-level derangement found seed={seed}')
        order,assigned,attempt=accepted
        require(sorted(hashlib.sha256(x.encode()).hexdigest() for x in assigned)==sorted(hashlib.sha256(x.encode()).hexdigest() for x in original), 'text multiset not preserved')
        for i,e in enumerate(entities):
            src=entities[int(order[i])]
            out.append(dict(perm_index=pi, perm_id=f'shuffle_{pi:02d}', seed=seed, attempts=attempt,
                            target_entity=e, shuffled_from_entity=src,
                            original_text_sha256=hashlib.sha256(original[i].encode()).hexdigest(),
                            shuffled_text_sha256=hashlib.sha256(assigned[i].encode()).hexdigest(),
                            original_n_chars=len(original[i]), shuffled_n_chars=len(assigned[i]),
                            shuffled_mechanism_text=assigned[i]))
    frame=pd.DataFrame(out)
    require(len(frame)==N_PERM*len(entities), 'permutation row count')
    return frame

def build_prompt_registry(meta,base,perms):
    base_by_atom=dict(zip(meta.atomic_id, base.prompt_text))
    ent_by_atom=dict(zip(meta.atomic_id, meta.source_entity_key))
    lookup=perms.set_index(['perm_index','target_entity'])
    rows=[]
    for pi in range(N_PERM):
        for aid in meta.atomic_id:
            ent=ent_by_atom[aid]; text=lookup.loc[(pi,ent),'shuffled_mechanism_text']
            prompt=base_by_atom[aid]+' '+text
            digest=hashlib.sha256(prompt.encode()).hexdigest()
            rows.append(dict(perm_index=pi, perm_id=f'shuffle_{pi:02d}', atomic_id=aid,
                             source_entity_key=ent, text_id='text:'+digest,
                             prompt_sha256=digest, prompt_text=prompt))
    frame=pd.DataFrame(rows)
    frame.insert(0,'assignment_row',np.arange(len(frame),dtype=int))
    require(len(frame)==N_PERM*len(meta), 'shuffled prompt assignment cardinality')
    require(not frame.duplicated(['perm_index','atomic_id']).any(), 'duplicate perm/atom')
    # The same target atom may receive the same shuffled mechanism in different
    # deterministic permutations. Such rows intentionally have identical prompt
    # text/text_id and are valid reuse, not an assignment collision.  A text_id
    # must still map to exactly one prompt/hash.
    by_id=frame.groupby('text_id',sort=False).agg(
        n_prompt_sha=('prompt_sha256','nunique'), n_prompt_text=('prompt_text','nunique'))
    require((by_id.n_prompt_sha.eq(1)&by_id.n_prompt_text.eq(1)).all(), 'text_id collision with nonidentical prompt')
    return frame

def stage_preflight(deep: Path, out: Path):
    require(not out.exists(), f'preflight output exists: {out}')
    out.mkdir(parents=True)
    enc,kg,e08=modules(deep)
    cfg,atoms,meta,panels,mechanism,base,correct=load_mechanism_design(deep,e08)
    perms=deterministic_derangements(sorted(mechanism),mechanism)
    prompts=build_prompt_registry(meta,base,perms)
    tsv(out/'permutation_manifest.tsv',perms)
    tsv(out/'shuffled_prompt_registry.tsv.gz',prompts)
    # Authenticate exact local snapshots and tokenizers; no expression/effect data enters encoding.
    kcfg=json.loads((deep/'configs/knowledge_encoding_v1.json').read_text())
    models,caches,sources,versions=enc.authenticate_six_models(kcfg)
    from transformers import AutoTokenizer
    # Tokenize each distinct prompt once. Assignment-level duplicates are
    # preserved in shuffled_prompt_registry.tsv.gz and mapped back by text_id.
    unique_prompts=(prompts[['text_id','prompt_sha256','prompt_text']]
                    .drop_duplicates('text_id',keep='first').reset_index(drop=True))
    unique_prompts.insert(0,'text_row',np.arange(len(unique_prompts),dtype=int))
    require(unique_prompts.text_id.is_unique, 'unique prompt registry text_id')
    tsv(out/'shuffled_unique_texts.tsv.gz',unique_prompts)
    token_rows=[]
    text_list=unique_prompts.prompt_text.tolist()
    for model in models:
        tok=AutoTokenizer.from_pretrained(model['snapshot_path'],local_files_only=True,trust_remote_code=False)
        lengths=[len(ids) for ids in tok(text_list,padding=False,truncation=False,add_special_tokens=True)['input_ids']]
        require(max(lengths)<=512 and min(lengths)>0, f'token length requires truncation for {model["model_key"]}')
        token_rows += [dict(model_key=model['model_key'], text_id=t, n_tokens=n) for t,n in zip(unique_prompts.text_id,lengths)]
    tsv(out/'token_lengths.tsv.gz',pd.DataFrame(token_rows))
    input_files=[deep/'configs/knowledge_encoding_v1.json', deep/'experiments/e08_prediction_v1/config.json',
                 deep/'metadata/knowledge_inputs_v1/entity_coverage.tsv', deep/'metadata/knowledge_inputs_v1/rowmap.tsv',
                 deep/'representations/clean_views_v1/row_to_text_registry.tsv', deep/'representations/clean_views_v1/unique_texts.tsv',
                 deep/'effects/atomic_effects_v1/atomic_index.tsv', deep/'effects/atomic_effects_v1/arrays.npz', deep/'effects/atomic_effects_v1/fold_gene_panels.tsv']
    json_write(out/'audit.json',dict(status='PASS_PREFLIGHT',created_utc=now(),deep_root=str(deep),
        n_entities=57,n_atoms=684,n_permutations=N_PERM,n_prompt_assignments=len(prompts),
        n_unique_prompts=len(unique_prompts),n_reused_prompt_assignments=len(prompts)-len(unique_prompts),seeds=PERM_SEEDS,
        permutation_policy='deterministic RNG permutations rejected until target entity differs and assigned mechanism text differs exactly; global mechanism-text multiset preserved',
        model_records=models,package_versions=versions,input_hashes={str(p):sha(p) for p in input_files},
        expression_or_effect_values_read=False,network_access=False))
    print('PASS_PREFLIGHT',f'assignments={len(prompts)}',f'unique_prompts={len(unique_prompts)}',
          f'reused={len(prompts)-len(unique_prompts)}',flush=True)

def load_preflight(run: Path):
    pre=run/'preflight'; require((pre/'audit.json').is_file(),'missing preflight')
    a=json.loads((pre/'audit.json').read_text()); require(a['status']=='PASS_PREFLIGHT','preflight not PASS')
    prompts=pd.read_csv(pre/'shuffled_prompt_registry.tsv.gz',sep='\t',dtype=str,keep_default_na=False)
    prompts['assignment_row']=pd.to_numeric(prompts.assignment_row).astype(int); prompts['perm_index']=pd.to_numeric(prompts.perm_index).astype(int)
    return a,prompts

def stage_encode(deep: Path, run: Path, out: Path):
    require(not out.exists(), f'encode output exists: {out}'); out.mkdir(parents=True)
    pre,prompts=load_preflight(run)
    enc,kg,e08=modules(deep)
    cfg,atoms,meta,panels,mechanism,base,correct=load_mechanism_design(deep,e08)
    kcfg=json.loads((deep/'configs/knowledge_encoding_v1.json').read_text())
    models,caches,sources,versions=enc.authenticate_six_models(kcfg)
    # Existing correct CK vectors supply a smoke reference only.
    _,_,correct_vectors=e08.texts_and_vectors(cfg,meta,True)
    import torch
    require(torch.cuda.is_available(),'CUDA unavailable; no silent CPU fallback')
    torch.set_num_threads(4); torch.manual_seed(kcfg['seed']); np.random.seed(kcfg['seed'])
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    smoke_idx=np.array([0,len(meta)//2,len(meta)-1],dtype=int)
    smoke_frame=correct.iloc[smoke_idx][['text_id','prompt_sha256','prompt_text']].copy().reset_index(drop=True)
    smoke_frame.insert(0,'text_row',np.arange(len(smoke_frame)))
    outputs=[]; smoke=[]
    for model in models:
        key=model['model_key']; encoder=enc.encoder_for_model(key)
        xsm,runtime=encoder(smoke_frame,model,kcfg); enc.v1.validate_vectors(xsm,smoke_frame,model['dimension'])
        ref=correct_vectors[(key,'CK')][smoke_idx]
        diff=float(np.max(np.abs(xsm-ref)))
        require(diff<=TOL_EMBED,f'correct-CK smoke mismatch {key}: {diff}')
        smoke.append(dict(model_key=key,n=len(smoke_idx),max_abs_error=diff))
        frame=(prompts[['text_id','prompt_sha256','prompt_text']]
               .drop_duplicates('text_id',keep='first').reset_index(drop=True))
        frame.insert(0,'text_row',np.arange(len(frame),dtype=int))
        require(frame.text_id.is_unique, 'encoding frame text_id uniqueness')
        print('ENCODE_START',key,len(frame),f'from_assignments={len(prompts)}',flush=True)
        x,runtime=encoder(frame,model,kcfg); enc.v1.validate_vectors(x,frame,model['dimension'])
        path=out/f'{key}__mechanism_shuffled20.npz'
        with path.open('xb') as f:
            np.savez_compressed(f,X=x,text_id=frame.text_id.to_numpy(str),prompt_sha256=frame.prompt_sha256.to_numpy(str))
        outputs.append(dict(model_key=key,model_id=model['model_id'],revision=model['revision'],dimension=model['dimension'],
                            path=str(path),sha256=sha(path),bytes=path.stat().st_size,runtime=runtime))
        print('ENCODE_PASS',key,flush=True)
    tsv(out/'smoke_reproduction.tsv',pd.DataFrame(smoke))
    json_write(out/'audit.json',dict(status='PASS_ENCODING',created_utc=now(),preflight_audit_sha256=sha(run/'preflight/audit.json'),
        prompt_registry_sha256=sha(run/'preflight/shuffled_prompt_registry.tsv.gz'),n_prompts=len(prompts),n_permutations=N_PERM,
        smoke_max_abs_error=max(x['max_abs_error'] for x in smoke),outputs=outputs,package_versions=versions,
        expression_or_effect_data_read=False,network_access=False,tf32=False))

def shuffled_vectors_for_model(run: Path, model: str, meta: pd.DataFrame):
    prompts=pd.read_csv(run/'preflight/shuffled_prompt_registry.tsv.gz',sep='\t',dtype=str,keep_default_na=False)
    prompts['perm_index']=pd.to_numeric(prompts.perm_index).astype(int)
    with np.load(run/f'encode/{model}__mechanism_shuffled20.npz',allow_pickle=False) as z:
        X=z['X']; ids=z['text_id'].astype(str); hashes=z['prompt_sha256'].astype(str)
    require(len(X)==len(ids)==len(hashes) and len(set(ids.tolist()))==len(ids),'shuffled unique embedding axis')
    unique=(prompts[['text_id','prompt_sha256','prompt_text']].drop_duplicates('text_id',keep='first'))
    expect=unique.set_index('text_id').loc[ids]
    require(np.array_equal(hashes,expect.prompt_sha256.to_numpy(str)),'shuffled unique embedding hashes')
    id_to_row={x:i for i,x in enumerate(ids.tolist())}
    result=[]
    for pi in range(N_PERM):
        sub=prompts[prompts.perm_index.eq(pi)].set_index('atomic_id').loc[meta.atomic_id]
        rows=np.fromiter((id_to_row[x] for x in sub.text_id),dtype=np.int64,count=len(sub))
        result.append(X[rows])
    return result

def aggregate_geometry_groups(frame: pd.DataFrame):
    # equal dose within cell line, then equal cell line
    line=(frame.groupby(['model_key','variant','perm_index','cell_line'],dropna=False,sort=False)
          .agg(rsa=('rsa','mean'),excess_ndcg=('excess_ndcg','mean')).reset_index())
    macro=(line.groupby(['model_key','variant','perm_index'],dropna=False,sort=False)
           .agg(rsa=('rsa','mean'),excess_ndcg=('excess_ndcg','mean')).reset_index())
    return line,macro

def stage_geometry(deep: Path, run: Path, out: Path):
    require(not out.exists(),f'geometry output exists: {out}'); out.mkdir(parents=True)
    enc,kg,e08=modules(deep); cfg,atoms,meta,panels,mechanism,base,correct=load_mechanism_design(deep,e08)
    _,_,vectors=e08.texts_and_vectors(cfg,meta,True)
    shuffled={m:shuffled_vectors_for_model(run,m,meta) for m in MODELS}
    records=[]; qrecords=[]
    for line in cfg['folds']:
        panel=panels.loc[panels.heldout_cell_line.eq(line)].sort_values('rank').reset_index(drop=True)
        for dose in sorted(meta.dose_value.unique()):
            rows=np.flatnonzero(meta.cell_line.eq(line).to_numpy() & meta.dose_value.eq(dose).to_numpy())
            require(len(rows)==57,'geometry group size')
            truth=kg.truth_geometry(e08.response_slice(cfg,atoms,meta.iloc[rows].effect_row,panel),10,12)
            ix=np.triu_indices(len(rows),1)
            for model in MODELS:
                variants=[('base',-1,vectors[(model,'C')]),('correct',-1,vectors[(model,'CK')])]
                variants += [('shuffled',pi,shuffled[model][pi]) for pi in range(N_PERM)]
                for variant,pi,xall in variants:
                    sim,_=kg.cosine(xall[rows],unique_rows=True); score=kg.score_geometry(sim[ix],truth,10,12)
                    records.append(dict(model_key=model,variant=variant,perm_index=pi,cell_line=line,dose_value=float(dose),n_drugs=57,
                                        rsa=score['rsa'],excess_ndcg=float(np.nanmean(score['excess_ndcg']))))
                    for local_i,r in enumerate(rows):
                        qrecords.append(dict(model_key=model,variant=variant,perm_index=pi,cell_line=line,dose_value=float(dose),
                                             atomic_id=meta.iloc[r].atomic_id,source_entity_key=meta.iloc[r].source_entity_key,
                                             excess_ndcg=score['excess_ndcg'][local_i]))
    groups=pd.DataFrame(records); q=pd.DataFrame(qrecords); line,macro=aggregate_geometry_groups(groups)
    # Validate frozen base->correct deltas against archived E08 geometry summary.
    arch=pd.read_csv(deep/'metrics/knowledge_alignment_v1/paired_descriptive_equal_dose_cell_summary.tsv',sep='\t')
    checks=[]
    for model in MODELS:
        row=arch[(arch.cohort=='mechanism_text') & (arch.contrast_id==model+'__augmentation_minus_complete')].iloc[0]
        b=macro[(macro.model_key==model)&(macro.variant=='base')].iloc[0]
        c=macro[(macro.model_key==model)&(macro.variant=='correct')].iloc[0]
        dr=float(c.rsa-b.rsa); dn=float(c.excess_ndcg-b.excess_ndcg)
        checks.append(dict(model_key=model,rsa_recomputed=dr,rsa_archived=row.rsa_change,rsa_abs_error=abs(dr-row.rsa_change),
                           excess_ndcg_recomputed=dn,excess_ndcg_archived=row.excess_ndcg_change_paired_mean,
                           excess_ndcg_abs_error=abs(dn-row.excess_ndcg_change_paired_mean)))
    checks=pd.DataFrame(checks)
    require(checks[['rsa_abs_error','excess_ndcg_abs_error']].to_numpy().max()<=TOL_REPRO,'frozen geometry reproduction failed')
    tsv(out/'group_metrics.tsv',groups); tsv(out/'query_metrics.tsv.gz',q); tsv(out/'line_summary.tsv',line); tsv(out/'macro_summary.tsv',macro); tsv(out/'frozen_reproduction.tsv',checks)
    json_write(out/'audit.json',dict(status='PASS_GEOMETRY',created_utc=now(),n_group_rows=len(groups),n_query_rows=len(q),
        frozen_reproduction_max_abs_error=float(checks[['rsa_abs_error','excess_ndcg_abs_error']].to_numpy().max()),
        effect_definition='frozen sci-Plex 3 atomic effects and heldout-line source-selected 3000-gene panels; no new effect construction'))

def prediction_axes(deep: Path,e08,cfg,atoms,meta,panels,heldout):
    source=np.flatnonzero(meta.cell_line.ne(heldout)); query=np.flatnonzero(meta.cell_line.eq(heldout))
    lm=e08.landmarks(meta,source,cfg['n_landmarks'],cfg['seed'])
    panel=panels.loc[panels.heldout_cell_line.eq(heldout)].sort_values('rank').reset_index(drop=True)
    source_y=e08.response_slice(cfg,atoms,meta.iloc[source].effect_row,panel).astype(np.float64)
    return source,query,lm,panel,source_y

def stage_predict(deep: Path, run: Path, out: Path):
    require(not out.exists(),f'predict output exists: {out}'); out.mkdir(parents=True)
    enc,kg,e08=modules(deep); cfg,atoms,meta,panels,mechanism,base,correct=load_mechanism_design(deep,e08)
    _,_,vectors=e08.texts_and_vectors(cfg,meta,True)
    shuffled={m:shuffled_vectors_for_model(run,m,meta) for m in MODELS}
    checks=[]; manifests=[]
    for heldout in cfg['folds']:
        folder=out/heldout; folder.mkdir()
        source,query,lm,panel,source_y=prediction_axes(deep,e08,cfg,atoms,meta,panels,heldout)
        # Existing E08 predictions provide exact regression for the frozen C/CK path.
        with np.load(deep/f'experiments/e08_prediction_v1/results/mechanism57/{heldout}/predictions.npz',allow_pickle=False) as z:
            old_pred=z['predictions']; old_methods=z['method'].tolist(); old_ids=z['atomic_id']
        require(np.array_equal(old_ids,meta.iloc[query].atomic_id.to_numpy(str)),'archived query axis')
        for model in MODELS:
            for view in ['C','CK']:
                x=vectors[(model,view)]; kernel=e08.unique_cosine(x,lm)
                pred,_=e08.fit_predict(kernel[source],source_y,kernel[query],cfg['alpha']); pred=pred.astype(np.float32)
                old=old_pred[old_methods.index(model+'__'+view)]
                err=float(np.max(np.abs(pred.astype(np.float64)-old.astype(np.float64))))
                bitwise=bool(np.array_equal(pred.view(np.uint32),old.view(np.uint32)))
                require(bitwise or err<=1e-7,f'prediction frozen regression failed {heldout} {model} {view}: {err}')
                checks.append(dict(heldout_cell_line=heldout,model_key=model,view=view,max_abs_error=err,bitwise_equal=bitwise))
            arr=[]
            for pi in range(N_PERM):
                x=shuffled[model][pi]; kernel=e08.unique_cosine(x,lm)
                pred,_=e08.fit_predict(kernel[source],source_y,kernel[query],cfg['alpha']); arr.append(pred.astype(np.float32))
            arr=np.stack(arr)
            require(arr.shape==(N_PERM,len(query),3000) and np.isfinite(arr).all(),'shuffled prediction shape')
            path=folder/f'{model}__shuffled20_predictions.npz'
            with path.open('xb') as f:
                np.savez_compressed(f,predictions=arr,perm_index=np.arange(N_PERM,dtype=np.int16),
                                    atomic_id=meta.iloc[query].atomic_id.to_numpy(str),
                                    source_feature_row=panel.source_feature_row.to_numpy(int),
                                    original_ensembl_id=panel.original_ensembl_id.to_numpy(str))
            manifests.append(dict(heldout_cell_line=heldout,model_key=model,path=str(path.relative_to(out)),sha256=sha(path),bytes=path.stat().st_size))
            print('PREDICT_PASS',heldout,model,flush=True)
    checks=pd.DataFrame(checks); tsv(out/'frozen_prediction_regression.tsv',checks)
    seal=dict(status='ALL_SHUFFLED_PREDICTIONS_FROZEN_BEFORE_QUERY_TRUTH_SCORING',created_utc=now(),n_permutations=N_PERM,
              n_models=6,n_folds=3,files=manifests,regression_max_abs_error=float(checks.max_abs_error.max()),
              regression_all_bitwise=bool(checks.bitwise_equal.all()),target_truth_read=False)
    json_write(out/'predictions_sealed.json',seal)
    print('PREDICTIONS_SEALED',sha(out/'predictions_sealed.json'),flush=True)

def dose_prediction_metrics(e08,pred,truth,query,panel):
    rows=[]; cond=[]
    for dose in sorted(query.dose_value.unique()):
        idx=np.flatnonzero(query.dose_value.to_numpy()==dose)
        c=e08.condition_metrics(pred[idx],truth[idx])
        g=e08.gene_metrics(pred[idx],truth[idx])
        mae=float(np.nanmean(c['mae'])); rho=float(np.nanmean(g['spearman']))
        rows.append(dict(dose_value=float(dose),mae=mae,gene_spearman=rho))
        for local,j in enumerate(idx): cond.append(dict(atomic_id=query.iloc[j].atomic_id,source_entity_key=query.iloc[j].source_entity_key,dose_value=float(dose),mae=float(c['mae'][local])))
    return pd.DataFrame(rows),pd.DataFrame(cond)

def stage_score(deep: Path, run: Path, out: Path):
    require(not out.exists(),f'score output exists: {out}'); out.mkdir(parents=True)
    seal=json.loads((run/'predict/predictions_sealed.json').read_text()); require(seal['status'].startswith('ALL_SHUFFLED'),'prediction seal missing')
    for rec in seal['files']: require(sha(run/'predict'/rec['path'])==rec['sha256'],'frozen shuffled prediction changed')
    enc,kg,e08=modules(deep); cfg,atoms,meta,panels,mechanism,base,correct=load_mechanism_design(deep,e08)
    # First reproduce archived C/CK score summaries using already frozen E08 predictions.
    archived_macro=pd.read_csv(deep/'experiments/e08_prediction_v1/results/macro_summary.tsv',sep='\t')
    dose_rows=[]; cond_rows=[]; checks=[]
    for heldout in cfg['folds']:
        source,query,lm,panel,source_y=prediction_axes(deep,e08,cfg,atoms,meta,panels,heldout)
        truth=e08.response_slice(cfg,atoms,meta.iloc[query].effect_row,panel)
        qmeta=meta.iloc[query].reset_index(drop=True)
        with np.load(deep/f'experiments/e08_prediction_v1/results/mechanism57/{heldout}/predictions.npz',allow_pickle=False) as z:
            old_pred=z['predictions']; old_methods=z['method'].tolist()
        for model in MODELS:
            for view,variant in [('C','base'),('CK','correct')]:
                pred=old_pred[old_methods.index(model+'__'+view)]
                d,c=dose_prediction_metrics(e08,pred,truth,qmeta,panel); d['heldout_cell_line']=heldout; d['model_key']=model; d['variant']=variant; d['perm_index']=-1
                c['heldout_cell_line']=heldout; c['model_key']=model; c['variant']=variant; c['perm_index']=-1
                dose_rows.append(d); cond_rows.append(c)
        for model in MODELS:
            path=run/f'predict/{heldout}/{model}__shuffled20_predictions.npz'
            with np.load(path,allow_pickle=False) as z:
                preds=z['predictions']; require(np.array_equal(z['atomic_id'],qmeta.atomic_id.to_numpy(str)),'shuffled score query axis')
            for pi in range(N_PERM):
                d,c=dose_prediction_metrics(e08,preds[pi],truth,qmeta,panel); d['heldout_cell_line']=heldout; d['model_key']=model; d['variant']='shuffled'; d['perm_index']=pi
                c['heldout_cell_line']=heldout; c['model_key']=model; c['variant']='shuffled'; c['perm_index']=pi
                dose_rows.append(d); cond_rows.append(c)
    dose=pd.concat(dose_rows,ignore_index=True); cond=pd.concat(cond_rows,ignore_index=True)
    fold=(dose.groupby(['model_key','variant','perm_index','heldout_cell_line'],sort=False)[['mae','gene_spearman']].mean().reset_index())
    macro=(fold.groupby(['model_key','variant','perm_index'],sort=False)[['mae','gene_spearman']].mean().reset_index())
    # Reproduce archived mechanism57 C and CK macros.
    for model in MODELS:
        for view,variant in [('C','base'),('CK','correct')]:
            for metric,arch_metric in [('mae','mae'),('gene_spearman','gene_spearman')]:
                got=float(macro[(macro.model_key==model)&(macro.variant==variant)][metric].iloc[0])
                method=model+'__'+view
                old=archived_macro[(archived_macro.cohort=='mechanism57')&(archived_macro.method==method)&(archived_macro.metric==arch_metric)]
                require(len(old)==1,f'archived macro missing {method} {metric}')
                exp=float(old.value.iloc[0]); err=abs(got-exp); require(err<=2e-10,f'archived score reproduction {method} {metric}: {err}')
                checks.append(dict(model_key=model,view=view,metric=metric,recomputed=got,archived=exp,abs_error=err))
    checks=pd.DataFrame(checks)
    tsv(out/'dose_summary.tsv',dose); tsv(out/'condition_mae.tsv.gz',cond); tsv(out/'fold_summary.tsv',fold); tsv(out/'macro_summary.tsv',macro); tsv(out/'frozen_score_reproduction.tsv',checks)
    json_write(out/'audit.json',dict(status='PASS_SCORE',created_utc=now(),prediction_seal_sha256=sha(run/'predict/predictions_sealed.json'),
        frozen_score_reproduction_max_abs_error=float(checks.abs_error.max()),n_macro_rows=len(macro),query_truth_read_only_after_prediction_seal=True))

def percentile_interval(x,lo=.025,hi=.975):
    x=np.asarray(x,float); return float(np.quantile(x,lo)),float(np.quantile(x,hi))

def drug_bootstrap(diff_by_drug, n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    x=np.asarray(diff_by_drug,float); require(len(x)==57 and np.isfinite(x).all(),'drug bootstrap vector')
    rng=np.random.default_rng(seed); vals=np.empty(n,float)
    for i in range(n): vals[i]=x[rng.integers(0,len(x),len(x))].mean()
    lo,hi=percentile_interval(vals); return float(x.mean()),lo,hi

def stage_summarize(deep: Path, run: Path, out: Path):
    require(not out.exists(),f'summary output exists: {out}'); out.mkdir(parents=True)
    geom=pd.read_csv(run/'geometry/macro_summary.tsv',sep='\t')
    gq=pd.read_csv(run/'geometry/query_metrics.tsv.gz',sep='\t')
    pred=pd.read_csv(run/'score/macro_summary.tsv',sep='\t')
    pc=pd.read_csv(run/'score/condition_mae.tsv.gz',sep='\t')
    rows=[]
    for model in MODELS:
        for metric,direction in [('rsa','higher'),('excess_ndcg','higher')]:
            b=float(geom[(geom.model_key==model)&(geom.variant=='base')][metric].iloc[0]); c=float(geom[(geom.model_key==model)&(geom.variant=='correct')][metric].iloc[0])
            s=geom[(geom.model_key==model)&(geom.variant=='shuffled')].sort_values('perm_index')[metric].to_numpy(float)
            lo,hi=percentile_interval(s)
            rows.append(dict(layer='alignment',model_key=model,metric=metric,better=direction,base=b,correct=c,correct_gain_vs_base=c-b,
                             shuffled_mean=float(s.mean()),shuffled_median=float(np.median(s)),shuffled_q025=lo,shuffled_q975=hi,
                             correct_minus_shuffled_mean=c-float(s.mean()),n_shuffled_better=int((s>=c).sum()),n_permutations=N_PERM))
        for metric,direction in [('mae','lower'),('gene_spearman','higher')]:
            b=float(pred[(pred.model_key==model)&(pred.variant=='base')][metric].iloc[0]); c=float(pred[(pred.model_key==model)&(pred.variant=='correct')][metric].iloc[0])
            s=pred[(pred.model_key==model)&(pred.variant=='shuffled')].sort_values('perm_index')[metric].to_numpy(float); lo,hi=percentile_interval(s)
            gain=(b-c) if metric=='mae' else (c-b); cm=(float(s.mean())-c) if metric=='mae' else (c-float(s.mean()))
            nb=int((s<=c).sum()) if metric=='mae' else int((s>=c).sum())
            rows.append(dict(layer='prediction',model_key=model,metric=metric,better=direction,base=b,correct=c,correct_gain_vs_base=gain,
                             shuffled_mean=float(s.mean()),shuffled_median=float(np.median(s)),shuffled_q025=lo,shuffled_q975=hi,
                             correct_minus_shuffled_mean=cm,n_shuffled_better=nb,n_permutations=N_PERM))
    summary=pd.DataFrame(rows)
    # Drug-level paired bootstrap for decomposable local-neighbour and MAE endpoints.
    boot=[]
    for model in MODELS:
        # NDCG: aggregate the 12 query records per drug. Compare correct against mean across shuffled permutations.
        q=gq[gq.model_key.eq(model)].copy()
        correct=(q[q.variant.eq('correct')].groupby('source_entity_key').excess_ndcg.mean())
        sh=(q[q.variant.eq('shuffled')].groupby(['perm_index','source_entity_key']).excess_ndcg.mean().unstack(0).mean(axis=1))
        ids=sorted(set(correct.index)&set(sh.index)); require(len(ids)==57,'NDCG drug bootstrap IDs')
        obs,lo,hi=drug_bootstrap((correct.loc[ids]-sh.loc[ids]).to_numpy(),seed=BOOTSTRAP_SEED)
        boot.append(dict(model_key=model,metric='excess_ndcg_correct_minus_mean_shuffled',n_drugs=57,n_bootstrap=BOOTSTRAP_N,estimate=obs,ci_low=lo,ci_high=hi))
        c=pc[(pc.model_key==model)&(pc.variant=='correct')].groupby('source_entity_key').mae.mean()
        s=pc[(pc.model_key==model)&(pc.variant=='shuffled')].groupby(['perm_index','source_entity_key']).mae.mean().unstack(0).mean(axis=1)
        ids=sorted(set(c.index)&set(s.index)); require(len(ids)==57,'MAE drug bootstrap IDs')
        obs,lo,hi=drug_bootstrap((s.loc[ids]-c.loc[ids]).to_numpy(),seed=BOOTSTRAP_SEED+1)
        boot.append(dict(model_key=model,metric='mae_mean_shuffled_minus_correct',n_drugs=57,n_bootstrap=BOOTSTRAP_N,estimate=obs,ci_low=lo,ci_high=hi))
    boot=pd.DataFrame(boot)
    tsv(out/'mechanism_control_summary.tsv',summary); tsv(out/'drug_level_bootstrap.tsv',boot)
    # Compact report intentionally descriptive.
    lines=['MECHANISM SHUFFLED CONTROL v1','='*90]
    for model in MODELS:
        lines.append(model)
        for row in summary[summary.model_key.eq(model)].itertuples(index=False):
            lines.append(f"  {row.metric}: base={row.base:.6g} correct={row.correct:.6g} shuffled_mean={row.shuffled_mean:.6g} correct_vs_shuffled={row.correct_minus_shuffled_mean:+.6g} shuffled_as_good_or_better={row.n_shuffled_better}/{row.n_permutations}")
    lines += ['', 'BOUNDARY: 20 shuffled mappings are deterministic negative controls, not independent biological replicates.',
              'Drug bootstrap is reported only for decomposable excess-NDCG query means and MAE; RSA and gene-Spearman use the fixed 20-permutation control distribution.',
              'No hyperparameter, encoder, split, gene panel, threshold or permutation was selected from outcomes.']
    (out/'COMPACT.txt').write_text('\n'.join(lines)+'\n')
    json_write(out/'audit.json',dict(status='PASS_SUMMARY',created_utc=now(),n_permutations=N_PERM,seeds=PERM_SEEDS,
        bootstrap_n=BOOTSTRAP_N,bootstrap_seed=BOOTSTRAP_SEED,no_outcome_based_selection=True,
        inference_boundary='shuffled permutations are controls, not independent biological replicates; bootstrap unit for reported intervals is drug identity'))
    print((out/'COMPACT.txt').read_text(),flush=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=['preflight','encode','geometry','predict','score','summarize'])
    ap.add_argument('--deep-root',type=Path,required=True)
    ap.add_argument('--run-root',type=Path,required=True)
    args=ap.parse_args(); deep=args.deep_root.resolve(); run=args.run_root.resolve(); run.mkdir(parents=True,exist_ok=True)
    out=run/args.stage
    {'preflight':stage_preflight,'encode':stage_encode,'geometry':stage_geometry,'predict':stage_predict,'score':stage_score,'summarize':stage_summarize}[args.stage](deep,run,out) if args.stage!='preflight' else stage_preflight(deep,out)
if __name__=='__main__': main()
