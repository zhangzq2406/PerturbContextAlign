"""R3 frozen-file IO, identity validation, and declarative method mapping.
No model fitting, scientific scores, or source-project imports live here.
"""
from __future__ import annotations
import csv
from decimal import Decimal
import hashlib
import importlib.resources as resources
import json
from pathlib import Path
import numpy as np
import pandas as pd

import os
ROOT = os.environ.get('PCA_DEEP_ROOT', str(Path.cwd()))
OUTBASE = os.environ.get('PCA_PAPER_ROOT', str(Path.cwd() / 'paper_outputs'))
MODEL_METHOD = {
    'bge_m3': 'bge_entity_exposure', 'sapbert': 'sapbert_entity_exposure',
    'qwen3_0_6b': 'qwen3_entity_exposure', 'biomedbert': 'biomedbert_entity_exposure',
    'medcpt_article': 'medcpt_article_entity_exposure', 'medcpt_query': 'medcpt_query_entity_exposure'}
BASELINES = ['zero', 'source_mean', 'source_median', 'same_drug_source_mean',
             'identity_exposure', 'tfidf_entity_exposure', 'morgan_entity_exposure']
BASE4 = BASELINES[:4]
KERNELS = BASELINES[4:] + list(MODEL_METHOD.values())
METHODS = BASE4 + [k+s for k in KERNELS for s in ('', '__control_state')]
SELECTED = [m for m in METHODS if m in BASELINES or m in MODEL_METHOD.values()]
META = 'metadata/kaggle_atomic_preflight_v1'
VIEWS = 'representations/kaggle_clean_views_v1'
EMB = 'representations/kaggle_clean_embeddings_v1/full'
EFFECT = 'effects/kaggle_atomic_effects_v1'
PRED = 'predictions/kaggle_prediction_v1'
ATOL = 1e-10
K = 10
MIN_GROUP = 12
GENE_MIN = 20
MODEL_OF = {v:k for k,v in MODEL_METHOD.items()}


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def pins():
    return json.loads(resources.files('r3app').joinpath('pins.json').read_text('utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def digest_axis(values):
    return hashlib.sha256(json.dumps([str(v) for v in values], ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def decimal(value):
    d = Decimal(str(value))
    require(d.is_finite(), 'Nonfinite decimal metadata')
    return format(d.normalize(), 'f')


def load_table(path):
    return pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False)


def write_table(path, frame):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    compression = {'method':'gzip', 'mtime':0} if path.name.endswith('.gz') else None
    frame.to_csv(path, sep='\t', index=False, na_rep='NA', float_format='%.17g',
                 lineterminator='\n', mode='x', compression=compression)


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def save_npz(path, **kw):
    require(all(np.asarray(v).dtype.kind != 'O' for v in kw.values()), 'Object NPZ forbidden')
    with Path(path).open('xb') as f:
        np.savez_compressed(f, **kw)


def finite_mean(x):
    a = np.asarray(x, dtype=np.float64); a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else np.nan


def metadata_fields(df):
    return df[['atomic_id','sm_lincs_id','sm_name','donor_id','cell_type',
               'dose_uM','timepoint_hr','context_id']].reset_index(drop=True).copy()


def verify_inputs(root, spec):
    records=[]
    root=Path(root).resolve()
    for item in spec['files']:
        p=root/item['path']
        require(p.resolve().is_relative_to(root), 'Input resolves outside source root: '+str(p))
        require(p.is_file(), 'Missing pinned input: '+str(p))
        st=p.stat()
        require(st.st_size==int(item['size_bytes']), 'Pinned file size changed: '+str(p))
        actual=sha(p)
        require(actual==item['sha256'], 'Pinned SHA256 changed: '+str(p))
        records.append(dict(path=str(p),relative_path=item['path'],sha256=actual,
                            size_bytes=st.st_size,mtime_ns=str(st.st_mtime_ns),binding=item['binding']))
    return pd.DataFrame(records)


def verify_unchanged(manifest):
    for row in manifest.itertuples():
        p=Path(row.path); st=p.stat()
        require(st.st_size==int(row.size_bytes) and str(st.st_mtime_ns)==str(row.mtime_ns),
                'Input stat changed during R3: '+str(p))
        require(sha(p)==row.sha256, 'Input bytes changed during R3: '+str(p))


class FrozenDataset:
    """Validation against actual ID/text/gene axes; never loads raw expression."""
    def __init__(self, root, spec):
        self.root=Path(root); self.spec=spec
        self.atoms=load_table(self.root/META/'atomic_index.tsv')
        self.tasks=load_table(self.root/META/'task_index.tsv')
        self.members=load_table(self.root/META/'fold_atomic_membership.tsv')
        self.registry=load_table(self.root/VIEWS/'row_to_text_registry.tsv')
        self.texts=load_table(self.root/VIEWS/'unique_texts.tsv')
        self.panels=load_table(self.root/EFFECT/'fold_gene_panels.tsv')
        self.enc=load_table(self.root/EMB/'encoding_manifest.tsv')
        conf=json.loads((self.root/'configs/kaggle_prediction_v1.json').read_text())
        require(conf['method_order']==METHODS, 'Unexpected 22-method configuration')
        require(conf['gene_min_n']==GENE_MIN and conf['expected_genes_per_fold']==3000,
                'Frozen scoring thresholds differ')
        require(self.tasks.to_dict('records')==spec['tasks'], 'Task registry no longer matches capture')
        require(len(self.atoms)==816 and self.atoms.atomic_id.is_unique,'816 unique atoms required')
        require(self.atoms.atomic_row.astype(int).tolist()==list(range(816)), 'Atomic row axis differs')
        require(set(self.atoms.cell_type)=={'NK cells','T cells CD4+'}, 'Unexpected cell types')
        require(set(self.atoms.donor_id)=={'donor_0','donor_1','donor_2'}, 'Unexpected donors')
        require(set(self.atoms.timepoint_hr.map(decimal))=={'24'}, 'Unexpected time')
        require(set(self.members.role)=={'source','query'},'Unexpected member roles')
        require(len(self.members)==2448 and not self.members.duplicated(['task_id','atomic_id']).any(),
                'Membership size/keys differ')
        q=self.members[self.members.role=='query']
        require(len(q)==816 and q.atomic_id.is_unique and set(q.atomic_id)==set(self.atoms.atomic_id),
                'Every original atom must be a query exactly once')
        require(self.registry.atomic_id.tolist()==self.atoms.atomic_id.tolist(),'Registry/atom order mismatch')
        require(set(self.registry.view)=={'entity_exposure'} and set(self.registry.variant)=={'source_name'},
                'Only entity_exposure/source_name is admitted')
        require(len(self.texts)==137 and self.texts.text_id.is_unique,'137 unique exact prompts required')
        require(self.texts.text_row.astype(int).tolist()==list(range(137)), 'Text row axis differs')
        for col in ['sm_lincs_id','sm_name','dose_uM','timepoint_hr']:
            require(self.registry[col].equals(self.atoms[col]), 'Registry metadata differs: '+col)
        for t in self.texts.itertuples():
            prompt=f'{t.sm_name}; dose: {decimal(Decimal(t.dose_uM)*1000)} nM; duration: 24 h.'
            dig=hashlib.sha256(prompt.encode()).hexdigest()
            require(t.prompt_text==prompt and t.prompt_sha256==dig and t.text_id=='text:'+dig,
                    'Exact prompt reconstruction mismatch')
        ti=self.texts.set_index('text_id')
        for col in ['text_row','prompt_sha256','prompt_text']:
            require(np.array_equal(self.registry[col],ti.loc[self.registry.text_id,col].to_numpy()),
                    'Registry-to-text binding differs: '+col)
        self.text_rows=self.registry.text_row.to_numpy(int)
        self.atom_lookup=self.atoms.set_index('atomic_id',drop=False)
        self.reg_lookup=self.registry.set_index('atomic_id',drop=False)
        require(len(self.panels)==18000,'Six complete 3000-gene panels required')
        with np.load(self.root/EFFECT/'arrays.npz',allow_pickle=False) as z:
            # Deliberately DO NOT access effect_provided_scale/state_A/state_B.
            require(np.array_equal(z['atomic_id'],self.atoms.atomic_id.to_numpy(str)), 'Union atom axis differs')
            self.union_features=z['source_feature_row'].copy()
        require(len(self.union_features)==3586 and len(np.unique(self.union_features))==3586,
                '3586 unique union feature rows required')
        self.x={}
        require(set(self.enc.model_key)==set(MODEL_METHOD) and self.enc.model_key.is_unique, 'Encoder manifest incomplete')
        for model in MODEL_METHOD:
            e=self.enc[self.enc.model_key==model].iloc[0]
            path=self.root/EMB/(model+'__source_name.npz')
            with np.load(path,allow_pickle=False) as z:
                require(set(z.files)=={'X','text_id','prompt_sha256'},'Encoder NPZ keys differ')
                require(np.array_equal(z['text_id'],self.texts.text_id.to_numpy(str)) and
                        np.array_equal(z['prompt_sha256'],self.texts.prompt_sha256.to_numpy(str)),
                        'Encoder exact text axes differ')
                x=z['X'].copy()
            require(x.dtype==np.float32 and x.shape==(137,int(e.dimension)) and np.isfinite(x).all(),
                    'Invalid encoder matrix: '+model)
            require((np.linalg.norm(x.astype(np.float64),axis=1)>0).all(),'Zero embedding norm: '+model)
            require(e.sha256==sha(path),'Encoder manifest hash differs: '+model)
            self.x[model]=x
        # These are pre-existing gates, not this run's independent score validation.
        qa_paths=['qa/kaggle_clean_embeddings_independent_v1.json',
                  'qa/kaggle_effects_independent_v1/audit.json','qa/kaggle_prediction_independent_v2/audit.json']
        for rel in qa_paths:
            gate=json.loads((self.root/rel).read_text())
            require(str(gate['status']).startswith('PASS'),'Historical gate not PASS: '+rel)
        self.config=conf

    def task(self, task_id):
        task=self.tasks.set_index('task_id').loc[task_id]
        d=self.root/PRED/task_id
        q=load_table(d/'query_atoms.tsv'); s=load_table(d/'source_atoms.tsv')
        n=137 if task.cell_type=='NK cells' else 135
        require(len(q)==n and len(s)==2*n,'Source/query counts differ: '+task_id)
        require(q.atomic_id.is_unique and s.atomic_id.is_unique and not set(q.atomic_id)&set(s.atomic_id),
                'Duplicate/overlapping atom IDs')
        require(set(q.cell_type)==set(s.cell_type)=={task.cell_type}, 'Task type differs')
        require(set(q.donor_id)=={task.heldout_donor} and set(s.donor_id)==set(task.source_donors.split('|')),
                'Task donor roles differ')
        member=self.members[self.members.task_id==task_id]
        for name,part in [('query',q),('source',s)]:
            require(set(part.atomic_id)==set(member[member.role==name].atomic_id), 'Membership differs: '+name)
            expected=self.atom_lookup.loc[part.atomic_id].reset_index(drop=True)
            require(part[self.atoms.columns].reset_index(drop=True).equals(expected[self.atoms.columns]),
                    'Task metadata not identical to original atoms: '+name)
            require(not part.duplicated(['sm_lincs_id','dose_key','time_key','donor_id']).any(), 'Duplicate exposure')
            require(np.array_equal(part.dose_key,part.dose_uM.map(decimal)) and
                    np.array_equal(part.time_key,part.timepoint_hr.map(decimal)),'Decimal exposure mismatch')
        groupcols=['sm_lincs_id','dose_key','time_key']
        sk=s.groupby(groupcols).donor_id.agg(set)
        require(sk.map(lambda v:v==set(s.donor_id)).all(),'Incomplete two-source support')
        require(set(sk.index)==set(map(tuple,q[groupcols].to_numpy())), 'Source/query exposure support differs')
        panel=self.panels[self.panels.task_id==task_id].copy()
        panel['rank']=panel['rank'].astype(int); panel=panel.sort_values('rank').reset_index(drop=True)
        require(panel['rank'].tolist()==list(range(1,3001)) and panel.source_feature_row.is_unique,
                'Invalid panel rank or feature IDs')
        features=panel.source_feature_row.to_numpy(np.int64); genes=panel.source_gene_id.to_numpy(str)
        require(panel.source_gene_id.is_unique,'Gene IDs not unique')
        require(np.array_equal(self.union_features[panel.union_column.to_numpy(int)],features),'Panel union axis differs')
        require(set(panel.cell_type)=={task.cell_type} and set(panel.heldout_donor)=={task.heldout_donor},'Panel task identity differs')
        arrays={}
        for filename,key,ids in [('frozen_predictions.npz','predictions',q.atomic_id),
                                  ('evaluation_truth.npz','truth',q.atomic_id),
                                  ('source_targets.npz','source_y',s.atomic_id)]:
            with np.load(d/filename,allow_pickle=False) as z:
                keys={key,'atomic_id','source_feature_row','source_gene_id'} | ({'method'} if key=='predictions' else set())
                require(set(z.files)==keys,'NPZ keys differ: '+filename)
                require(np.array_equal(z['atomic_id'],ids.to_numpy(str)),'NPZ atom order differs: '+filename)
                require(np.array_equal(z['source_feature_row'],features) and np.array_equal(z['source_gene_id'],genes),
                        'NPZ ordered gene identity differs: '+filename)
                if key=='predictions':require(z['method'].tolist()==METHODS,'NPZ method axis differs')
                arr=z[key].copy()
            exp=(22,n,3000) if key=='predictions' else ((n,3000) if key=='truth' else (2*n,3000))
            require(arr.dtype==np.float32 and arr.shape==exp and np.isfinite(arr).all(),
                    'Unexpected dtype/shape/nonfinite data: '+filename)
            arrays[key]=arr
        keep=q.dose_uM.map(decimal).eq('1').to_numpy()
        qm=q.loc[keep].reset_index(drop=True)
        excluded=q.loc[~keep].reset_index(drop=True)
        require(len(qm)==n-1 and qm.sm_lincs_id.is_unique, 'Primary cohort count/unique drugs differ')
        require(len(excluded)==1 and excluded.dose_uM.map(decimal).iloc[0]=='0.1' and
                excluded.sm_name.iloc[0]=='Belinostat','Unexpected primary exclusion')
        require(set(qm.timepoint_hr.map(decimal))=={'24'},'Primary time not fixed')
        truth=arrays['truth'][keep].copy()
        # Matches the actual legacy caller, which refuses zero effect vectors.
        require((np.linalg.norm(truth.astype(np.float64),axis=1)>0).all(),
                'ZERO_EFFECT_NORM: L2 undefined; stop this task, no zero-fill or cohort substitution')
        embrows=self.reg_lookup.loc[qm.atomic_id].text_row.to_numpy(int)
        co=digest_axis(qm.atomic_id)
        ga=digest_axis([f'{a}|{b}' for a,b in zip(features,genes)])
        return dict(task_id=task_id,cell_type=task.cell_type,donor_id=task.heldout_donor,
                    q=q,s=s,qm=qm,keep=keep,excluded=excluded,panel=panel,
                    features=features,genes=genes,cohort_id=co,gene_axis_id=ga,
                    truth=truth,predictions=arrays['predictions'][:,keep,:].copy(),
                    source_y=arrays['source_y'],embrows=embrows,
                    embeddings={m:x[embrows].copy() for m,x in self.x.items()})
