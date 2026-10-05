#!/usr/bin/env python3
"""New matched-cohort E08 runner; upstream/archive files are immutable inputs."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ARCHIVE = HERE.parents[1]
PACKAGE = HERE.parents[1]  # public fine-workspace configuration root
sys.path.insert(0, str(ARCHIVE / 'code'))
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits
from landmark_decoder import LandmarkRidge, cosine_similarity
from prediction_metrics import condition_metrics, gene_metrics, paired_order_accuracy_by_gene

MODELS = ['bge_m3', 'sapbert', 'qwen3_0_6b', 'biomedbert', 'medcpt_article', 'medcpt_query']
REFS = ['zero', 'source_mean', 'source_median', 'same_drug_source_mean']
METRICS = ['mae', 'rmse', 'condition_spearman', 'gene_spearman', 'gene_order']


def require(test, message):
    if not test:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def table(path, frame):
    path = Path(path)
    require(not path.exists(), 'refuse overwrite: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep='\t', index=False, na_rep='NA', compression='infer')


def save_npz(destination, **arrays):
    destination = Path(destination); destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as stream:
        np.savez_compressed(stream, **arrays)


def read_tsv(path):
    return pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False)


def bools(values):
    values = values.astype(str).str.lower()
    require(values.isin(['true', 'false']).all(), 'nonboolean flag')
    return values.eq('true')


def method_registry(config):
    rows = []
    for cohort in config['cohorts']:
        c = cohort['cohort']; native = cohort['native']
        for ref in REFS:
            rows.append(dict(cohort=c, method=ref, kind='reference', model='', view='', native=''))
        for model in MODELS:
            for view in (['C', 'CK', 'K'] if c == 'mechanism57' else ['C']):
                rows.append(dict(cohort=c, method=model + '__' + view, kind='llm', model=model, view=view, native=''))
        if c == 'mechanism57':
            for view in ['C', 'CK', 'K']:
                rows.append(dict(cohort=c, method='tfidf__' + view, kind='tfidf', model='tfidf', view=view, native=''))
            rows.append(dict(cohort=c, method='structured_identity_exposure', kind='structured', model='', view='', native=''))
        rows.append(dict(cohort=c, method='native_' + native + '_exposure', kind='native', model='', view='', native=native))
        for model in MODELS:
            rows.append(dict(cohort=c, method=model + '__fusion_' + native, kind='fusion', model=model, view='C', native=native))
    frame = pd.DataFrame(rows)
    require(not frame.duplicated(['cohort', 'method']).any(), 'duplicate method')
    for c in config['cohorts']:
        selected = frame[frame.cohort == c['cohort']]
        require(len(selected) == c['methods'] and selected.kind.ne('reference').sum() == c['ridge_methods'], 'method counts')
    return frame


def contrast_registry(config, methods):
    rows = []
    for c in config['cohorts']:
        cohort = c['cohort']; native = 'native_' + c['native'] + '_exposure'
        def add(left, right, family):
            rows.append(dict(cohort=cohort, contrast=left + '__vs__' + right, left=left, right=right, family=family))
        for method in methods.loc[methods.cohort == cohort, 'method']:
            if method != 'same_drug_source_mean':
                add(method, 'same_drug_source_mean', 'vs_same_drug')
        if cohort == 'mechanism57':
            for model in MODELS + ['tfidf']:
                add(model + '__CK', model + '__C', 'added_knowledge')
                add(model + '__K', model + '__C', 'knowledge_only')
        for model in MODELS:
            add(model + '__fusion_' + c['native'], model + '__C', 'fusion_vs_C')
            add(model + '__fusion_' + c['native'], native, 'fusion_vs_native')
            add(native, model + '__C', 'native_vs_C')
    frame = pd.DataFrame(rows)
    require(not frame.duplicated(['cohort', 'left', 'right']).any(), 'duplicate contrast')
    require(len(frame) == config['expected_contrasts'], 'contrast count')
    return frame


def landmarks(metadata, train_rows, count=256, seed=20260914):
    train_rows = np.asarray(train_rows, dtype=int)
    require(len(set(train_rows)) == len(train_rows), 'duplicate source rows')
    source = metadata.iloc[train_rows].copy()
    source['_row'] = train_rows
    source['_hash'] = source.atomic_id.map(lambda x: hashlib.sha256(f'{seed}|{x}'.encode()).hexdigest())
    ordered = source.sort_values(['_hash', 'atomic_id'], kind='stable')
    first = ordered.drop_duplicates('source_entity_key', keep='first')
    require(len(first) <= count <= len(source), 'invalid landmark count')
    rest = ordered.loc[~ordered._row.isin(first._row)]
    return pd.concat([first, rest.head(count-len(first))])._row.to_numpy(int)


def unique_cosine(x, landmark_rows):
    x = np.asarray(x)
    require(x.ndim == 2 and np.isfinite(x).all(), 'invalid vectors')
    unique, inverse = np.unique(x, axis=0, return_inverse=True)
    return cosine_similarity(unique, x[np.asarray(landmark_rows, int)])[inverse]


def binary_jaccard(x, landmark_rows):
    x = np.asarray(x)
    require(x.ndim == 2 and np.isin(x, [0, 1]).all() and (x.sum(axis=1) > 0).all(), 'empty/nonbinary native prior')
    x = x.astype(np.int64)
    ref = x[np.asarray(landmark_rows, int)]
    overlap = x @ ref.T
    union = x.sum(1)[:, None] + ref.sum(1)[None, :] - overlap
    require((union > 0).all(), 'empty native set')
    return overlap / union


def native_kernel(x, doses, landmark_rows):
    doses = np.asarray(doses, dtype=float)
    return (binary_jaccard(x, landmark_rows) + (doses[:, None] == doses[np.asarray(landmark_rows)][None, :])) / 2


def fusion_kernel(cosine, native):
    cosine, native = np.asarray(cosine), np.asarray(native)
    require(cosine.shape == native.shape and np.isfinite(cosine).all() and np.isfinite(native).all(), 'fusion dimensions/finite')
    return 0.5 * ((cosine + 1) / 2) + 0.5 * native


def structured_kernel(metadata, source_rows, landmark_rows):
    source = metadata.iloc[source_rows]
    entities = sorted(source.source_entity_key.unique()); doses = sorted(source.dose_value.unique())
    entity_rows = pd.Index(entities).get_indexer(metadata.source_entity_key)
    dose_rows = pd.Index(doses).get_indexer(metadata.dose_value)
    require((entity_rows >= 0).all() and (dose_rows >= 0).all(), 'unknown structured source vocabulary')
    x = np.zeros((len(metadata), len(entities)+len(doses)))
    x[np.arange(len(x)), entity_rows] = 1
    x[np.arange(len(x)), len(entities)+dose_rows] = 1
    return unique_cosine(x, landmark_rows), {'source_entity_vocabulary': entities, 'source_dose_vocabulary': [int(x) for x in doses]}


def tfidf_kernel(texts, source_rows, landmark_rows, config):
    texts = np.asarray(texts, str)
    source_texts = sorted(set(texts[np.asarray(source_rows)]))
    spec = config['tfidf']
    vectorizer = TfidfVectorizer(lowercase=spec['lowercase'], ngram_range=tuple(spec['ngram_range']),
        min_df=spec['min_df'], max_features=spec['max_features'], sublinear_tf=spec['sublinear_tf'])
    vectorizer.fit(source_texts)
    unique, inverse = np.unique(texts, return_inverse=True)
    x = normalize(vectorizer.transform(unique), norm='l2', copy=True)
    kernel = (x @ x[inverse[np.asarray(landmark_rows)]].T).toarray()[inverse]
    return kernel, {'source_unique_prompt_sha256': [hashlib.sha256(t.encode()).hexdigest() for t in source_texts],
                    'vocabulary': vectorizer.get_feature_names_out().tolist(), 'idf': vectorizer.idf_.tolist(),
                    'n_unique_source_texts': len(source_texts), 'zero_vector_rows': int((x.getnnz(1)[inverse] == 0).sum())}


def fit_predict(source_features, source_y, query_features, alpha=10):
    """No query truth argument; source observations are deliberately not deduplicated."""
    model = LandmarkRidge(alpha).fit(source_features, source_y)
    query_features = np.asarray(query_features)
    unique, inverse = np.unique(query_features, axis=0, return_inverse=True)
    predicted = model.predict(unique)[inverse]
    return predicted, {name: getattr(model, name + '_') for name in ['feature_mean', 'feature_scale', 'target_mean', 'coef']}


def references(metadata, source_rows, query_rows, source_y):
    source_y = np.asarray(source_y, np.float64)
    n_query = len(query_rows)
    values = {'zero': np.zeros((n_query, source_y.shape[1])),
              'source_mean': np.broadcast_to(source_y.mean(0), (n_query, source_y.shape[1])).copy(),
              'source_median': np.broadcast_to(np.median(source_y, axis=0), (n_query, source_y.shape[1])).copy()}
    source = metadata.iloc[source_rows]
    matches = {}
    for i, row in enumerate(source.itertuples()):
        matches.setdefault((row.source_entity_key, row.dose_value, row.time), []).append(i)
    pair_rows, prediction = [], []
    for query in metadata.iloc[query_rows].itertuples():
        rows = matches.get((query.source_entity_key, query.dose_value, query.time), [])
        require(len(rows) == 2 and source.iloc[rows].cell_line.nunique() == 2, 'same-drug exact two-source support')
        prediction.append(source_y[rows].mean(0))
        pair_rows.append(dict(atomic_id=query.atomic_id, source_atomic_id_1=source.iloc[rows[0]].atomic_id,
                             source_atomic_id_2=source.iloc[rows[1]].atomic_id))
    values['same_drug_source_mean'] = np.asarray(prediction)
    return values, pd.DataFrame(pair_rows)


def load_config(path):
    config = json.loads(Path(path).read_text())
    require(config['models'] == MODELS and config['references'] == REFS and config['metrics'] == METRICS, 'registry contract')
    require((config['alpha'], config['seed'], config['n_landmarks'], config['gene_min_n'], config['n_genes']) == (10, 20260914, 256, 20, 3000), 'fixed parameters')
    require(sha(PACKAGE/config['review_path']) == config['review_sha256'], 'design review changed')
    require(sha(ARCHIVE/'code/landmark_decoder.py') == config['decoder_sha256'], 'archived decoder changed')
    require(sha(ARCHIVE/'code/prediction_metrics.py') == config['scorer_sha256'], 'archived scorer changed')
    return config


def verified_inputs(config):
    root = Path(config['source_root']); paths = set()
    def add(rel, expected=None):
        path = root/rel; digest = sha(path)
        if expected is not None:
            require(digest == expected, 'upstream changed: ' + str(path))
        paths.add(path)
        return path
    def audit(rel):
        path = add(rel)
        value = json.loads(path.read_text())
        require(value['status'] == 'PASS', 'producer audit not PASS ' + rel)
        return value
    for gate in config['gates']:
        path = add(gate['path'], gate['sha256'])
        value = json.loads(path.read_text())
        require(value['status'] == gate['status'], 'independent gate status')
        bound = add(gate['binding_path'])
        require(value[gate['binding_field']] == sha(bound), 'independent binding')
    eff = audit('effects/atomic_effects_v1/audit.json')
    for name in ['arrays.npz', 'atomic_index.tsv', 'fold_gene_panels.tsv']:
        add('effects/atomic_effects_v1/' + name, eff['output_sha256'][name])
    add('effects/atomic_effects_v1/arrays.npz', config['effect_array_sha256'])
    add('effects/atomic_effects_v1/feature_selection_sealed.json')
    # Existing independent effect check is explicitly a bounded CSR subset, not full reconstruction.
    audit('qa/atomic_effect_subset_check_v1/audit.json')
    audit('qa/prediction_six_encoder_independent_v1/audit.json')
    for folder in ['metadata/knowledge_inputs_v1', 'representations/knowledge_embeddings_v1/full']:
        manifest = json.loads(add(folder + '/output_manifest.json').read_text())
        for item in manifest['files']:
            add(folder + '/' + item['path'], item['sha256'])
    for folder in ['representations/clean_embeddings_v1/full', 'representations/clean_embeddings_v2_additional_four/full']:
        encoded = audit(folder + '/audit.json')
        for item in encoded['sources']:
            p = Path(item['path']); require(p.is_relative_to(root), 'unexpected encoder source outside R')
            add(str(p.relative_to(root)), item['sha256'])
        for item in encoded['outputs']:
            p = Path(item['path']); require(p.is_relative_to(root), 'unexpected encoder output')
            add(str(p.relative_to(root)), item['sha256'])
        add(folder + '/encoding_manifest.tsv')
    for name in ['config.json', 'unique_texts.tsv', 'row_to_text_registry.tsv']:
        add('representations/clean_views_v1/' + name)
    morgan = audit('representations/morgan_v1/audit.json')
    for name in ['arrays.npz', 'entity_fingerprint_manifest.tsv']:
        add('representations/morgan_v1/' + name, morgan['output_sha256'][name])
    return sorted(paths)


def load_metadata(config):
    root = Path(config['source_root'])
    atoms = read_tsv(root/'effects/atomic_effects_v1/atomic_index.tsv')
    require(len(atoms) == 2256 and atoms.atomic_id.is_unique and atoms.atomic_id.tolist() == sorted(atoms.atomic_id), 'original atom axis')
    atoms['effect_row'] = np.arange(len(atoms))
    atoms['dose_value'] = pd.to_numeric(atoms.dose_value)
    atoms['time'] = pd.to_numeric(atoms.time)
    coverage = read_tsv(root/'metadata/knowledge_inputs_v1/entity_coverage.tsv')
    require(len(coverage) == 188 and coverage.source_entity_key.is_unique, 'coverage identity')
    panels = read_tsv(root/'effects/atomic_effects_v1/fold_gene_panels.tsv')
    for column in ['rank', 'source_feature_row', 'union_column']:
        panels[column] = pd.to_numeric(panels[column])
    cohorts = {}
    for c in config['cohorts']:
        entities = coverage.loc[bools(coverage[c['flag']]), 'source_entity_key']
        meta = atoms.loc[bools(atoms.main_eligible) & atoms.source_entity_key.isin(entities)].reset_index(drop=True)
        require(len(entities) == c['entities'] and len(meta) == c['atoms'], 'cohort admission count')
        require(meta.dose_unit.eq('nM').all() and meta.time.eq(24).all(), 'exposure mismatch')
        require(not meta.duplicated(['source_entity_key', 'cell_line', 'dose_value']).any(), 'duplicate condition')
        counts = meta.groupby(['cell_line', 'dose_value']).size()
        require(len(counts) == 12 and counts.eq(c['entities']).all(), 'balanced12groups')
        for line in config['folds']:
            require(meta.cell_line.eq(line).sum() == c['query_atoms'], 'query count')
            panel = panels.loc[panels.heldout_cell_line == line].sort_values('rank')
            require(len(panel) == 3000 and panel.source_feature_row.is_unique and panel.union_column.is_unique, '3000gene panel')
            require(panel['rank'].tolist() == list(range(1, 3001)), 'panel rank')
        cohorts[c['cohort']] = meta
    return atoms, cohorts, panels


def texts_and_vectors(config, metadata, include_knowledge):
    root = Path(config['source_root'])
    registry = read_tsv(root/'representations/clean_views_v1/row_to_text_registry.tsv')
    registry = registry.loc[registry.variant.eq('source_name') & registry.view.eq('complete_metadata')]
    require(registry.atomic_id.is_unique, 'C registry duplicate')
    registry = registry.set_index('atomic_id').loc[metadata.atomic_id]
    text = read_tsv(root/'representations/clean_views_v1/unique_texts.tsv').set_index('text_id')
    mapping = registry.reset_index()[['atomic_id', 'text_id', 'prompt_sha256']].copy()
    mapping['prompt_text'] = text.loc[mapping.text_id, 'prompt_text'].to_numpy()
    mapping['view'] = 'C'
    rows = [mapping]
    if include_knowledge:
        knowledge = read_tsv(root/'metadata/knowledge_inputs_v1/rowmap.tsv')
        for view, raw_view in [('CK', 'complete_metadata+knowledge'), ('K', 'knowledge_only')]:
            selected = knowledge.loc[knowledge.view.eq(raw_view)]
            require(selected.atomic_id.is_unique, 'knowledge duplicate atom/view')
            selected = selected.set_index('atomic_id').loc[metadata.atomic_id].reset_index()
            require(np.array_equal(selected.original_complete_text_id, mapping.text_id), 'knowledge-to-C identity')
            selected['view'] = view
            rows.append(selected[['atomic_id', 'text_id', 'prompt_sha256', 'prompt_text', 'view']])
    mappings = pd.concat(rows, ignore_index=True)
    require((mappings.prompt_text.map(lambda x: hashlib.sha256(x.encode()).hexdigest()) == mappings.prompt_sha256).all(), 'prompt checksum')
    require((mappings.text_id == 'text:' + mappings.prompt_sha256).all(), 'text ID checksum')
    result, texts = {}, {}
    for view in mappings.view.unique():
        texts[view] = mappings.loc[mappings.view.eq(view), 'prompt_text'].to_numpy(str)
    for model in MODELS:
        old = 'clean_embeddings_v1' if model in MODELS[:2] else 'clean_embeddings_v2_additional_four'
        for views, path in [(['C'], root/f'representations/{old}/full/{model}__source_name.npz'),
                            (['CK', 'K'] if include_knowledge else [], root/f'representations/knowledge_embeddings_v1/full/{model}__verified_chembl37_snapshot.npz')]:
            if not views:
                continue
            with np.load(path, allow_pickle=False) as data:
                x, ids, hashes = data['X'], data['text_id'], data['prompt_sha256']
            require(x.dtype == np.float32 and len(ids) == len(set(ids)) and len(ids) == len(x) == len(hashes) and np.isfinite(x).all(), 'embedding values/axes')
            for view in views:
                selected = mappings.loc[mappings.view.eq(view)]
                indices = pd.Index(ids).get_indexer(selected.text_id)
                require((indices >= 0).all() and np.array_equal(hashes[indices], selected.prompt_sha256.to_numpy(str)), 'embedding text mapping')
                result[(model, view)] = x[indices]
    return mappings, texts, result


def native_inputs(config, metadata, native):
    root = Path(config['source_root'])
    if native == 'morgan':
        with np.load(root/'representations/morgan_v1/arrays.npz', allow_pickle=False) as data:
            ids, x = data['source_entity_key'], data['fingerprints']
        require(x.shape == (188, 2048) and x.dtype == np.uint8 and len(set(ids)) == 188, 'Morgan axis')
        indices = pd.Index(ids).get_indexer(metadata.source_entity_key)
        require((indices >= 0).all(), 'missing Morgan')
        matrix = x[indices]
    else:
        table_ = read_tsv(root/'metadata/knowledge_inputs_v1/human_target_membership.tsv')
        require(not table_.duplicated(['source_entity_key', 'target_chembl_id']).any(), 'duplicate target tuples')
        require(table_.organism.eq('Homo sapiens').all() and table_.tax_id.eq('9606').all(), 'nonhuman target')
        targets = sorted(table_.target_chembl_id.unique())
        members = table_.groupby('source_entity_key').target_chembl_id.agg(set).to_dict()
        require(set(metadata.source_entity_key) <= set(members), 'missing target set')
        matrix = np.array([[int(t in members[e]) for t in targets] for e in metadata.source_entity_key], np.uint8)
    require(np.isin(matrix, [0, 1]).all() and (matrix.sum(1) > 0).all(), 'native empty/nonbinary')
    return matrix


def make_kernels(config, metadata, source_rows, query_rows, landmark_rows, registry, texts, embeddings, native):
    kernels, feature_audit = {}, {}
    native_value = native_kernel(native, metadata.dose_value.to_numpy(), landmark_rows)
    for row in registry.itertuples():
        if row.kind == 'reference':
            continue
        if row.kind == 'llm':
            features = unique_cosine(embeddings[(row.model, row.view)], landmark_rows)
            audit = {'n_unique_input_source': len(np.unique(embeddings[(row.model, row.view)][source_rows], axis=0)),
                     'n_unique_input_query': len(np.unique(embeddings[(row.model, row.view)][query_rows], axis=0))}
        elif row.kind == 'native':
            features, audit = native_value, {'native': row.native}
        elif row.kind == 'fusion':
            features = fusion_kernel(kernels[row.model + '__C'], native_value)
            audit = {'fusion': config['fusion']}
        elif row.kind == 'structured':
            features, audit = structured_kernel(metadata, source_rows, landmark_rows)
        elif row.kind == 'tfidf':
            features, audit = tfidf_kernel(texts[row.view], source_rows, landmark_rows, config)
        else:
            raise ValueError(row.kind)
        require(features.shape == (len(metadata), len(landmark_rows)) and np.isfinite(features).all(), 'kernel shape/finite')
        kernels[row.method] = features
        feature_audit[row.method] = {**audit, 'n_unique_source_feature_rows': len(np.unique(features[source_rows], axis=0)),
            'n_unique_query_feature_rows': len(np.unique(features[query_rows], axis=0)),
            'n_source_std_lt_1e8': int((features[source_rows].std(0) < 1e-8).sum())}
    return kernels, feature_audit


def prepare(config_path):
    """Metadata/vectors/hash-only preflight. Never opens the response-valued NPZ member."""
    config_path = Path(config_path).resolve(); experiment = config_path.parent
    require(not (experiment/'preflight').exists() and not (experiment/'execution_sealed.json').exists(), 'refuse preflight overwrite')
    config = load_config(config_path)
    paths = verified_inputs(config)
    paths += [config_path, Path(__file__), HERE/'test_e08_prediction.py', HERE/'OUTPUT_SCHEMA.md',
              PACKAGE/config['review_path'], ARCHIVE/'code/landmark_decoder.py', ARCHIVE/'code/prediction_metrics.py']
    paths = sorted(set(paths))
    manifest = [{'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size} for path in paths]
    atoms, cohorts, panels = load_metadata(config)
    methods = method_registry(config); contrasts = contrast_registry(config, methods)
    output = experiment/'preflight'; output.mkdir()
    table(output/'method_registry.tsv', methods); table(output/'contrast_registry.tsv', contrasts)
    table(output/'gene_panels.tsv', panels)
    counts = []
    for c in config['cohorts']:
        meta = cohorts[c['cohort']]
        mappings, texts, vectors = texts_and_vectors(config, meta, c['cohort'] == 'mechanism57')
        native_inputs(config, meta, c['native'])
        table(output/(c['cohort'] + '__atoms.tsv'), meta)
        table(output/(c['cohort'] + '__text_mapping.tsv'), mappings)
        for fold in config['folds']:
            source = np.flatnonzero(meta.cell_line.ne(fold)); query = np.flatnonzero(meta.cell_line.eq(fold))
            selected = landmarks(meta, source, config['n_landmarks'], config['seed'])
            require(len(source) == c['source_atoms'] and len(query) == c['query_atoms'] and meta.iloc[selected].source_entity_key.nunique() == c['entities'], 'fold support')
            table(output/(c['cohort'] + '__' + fold + '__landmarks.tsv'), meta.iloc[selected])
            counts.append(dict(cohort=c['cohort'], heldout_cell_line=fold, n_source=len(source), n_query=len(query),
                               n_landmarks=len(selected), n_entities=c['entities'], n_genes=3000))
    for item in manifest:
        require(sha(item['path']) == item['sha256'], 'input changed during preflight')
    table(output/'fold_support.tsv', pd.DataFrame(counts))
    write_json(experiment/'input_manifest.json', {'files': manifest, 'created_utc': now(), 'response_values_read': False})
    write_json(output/'audit.json', {'status': 'PASS_METADATA_VECTORS_HASH_ONLY', 'response_values_read': False,
               'n_method_fold_outputs': 150, 'n_ridge_fits': 126, 'n_contrasts': len(contrasts),
               'input_manifest_sha256': sha(experiment/'input_manifest.json')})
    print(json.dumps({'status': 'PASS_METADATA_VECTORS_HASH_ONLY', 'files': len(manifest), 'config_sha256': sha(config_path)}))


def seal(config_path, test_report):
    experiment = Path(config_path).resolve().parent
    test = json.loads(Path(test_report).read_text())
    require(test['status'] == 'PASS' and test['script_sha256'] == sha(__file__) and test['tests_sha256'] == sha(HERE/'test_e08_prediction.py'), 'tests not current PASS')
    audit = json.loads((experiment/'preflight/audit.json').read_text())
    require(audit['status'] == 'PASS_METADATA_VECTORS_HASH_ONLY' and not audit['response_values_read'], 'preflight not accepted')
    manifest = json.loads((experiment/'input_manifest.json').read_text())
    for row in manifest['files']:
        require(sha(row['path']) == row['sha256'], 'changed input before seal')
    outputs = [{'path': str(p.relative_to(experiment)), 'sha256': sha(p)} for p in sorted((experiment/'preflight').rglob('*')) if p.is_file()]
    write_json(experiment/'execution_sealed.json', {'status': 'SEALED_BEFORE_NEW_RESPONSE_FIT', 'created_utc': now(),
        'config_sha256': sha(config_path), 'input_manifest_sha256': sha(experiment/'input_manifest.json'),
        'script_sha256': sha(__file__), 'test_report_path': str(Path(test_report).resolve()), 'test_report_sha256': sha(test_report),
        'preflight_outputs': outputs, 'production_requires_explicit_execute_and_exact_seal_sha': True})
    print(json.dumps({'status': 'SEALED_BEFORE_NEW_RESPONSE_FIT', 'seal_sha256': sha(experiment/'execution_sealed.json')}))


def summary_row(keys, metric, values):
    values = np.asarray(values, np.float64)
    valid = np.isfinite(values)
    require(not np.isinf(values).any(), 'infinite metric')
    return {**keys, 'metric': metric, 'value': float(values[valid].mean()) if valid.any() else np.nan,
            'n_units': len(values), 'n_valid_units': int(valid.sum()), 'n_na_units': int((~valid).sum()),
            'n_children': len(values), 'n_valid_children': int(valid.sum())}


def aggregate(table_, keys, expected_children):
    rows = []
    for identity, group in table_.groupby(keys, sort=False, dropna=False):
        if not isinstance(identity, tuple):
            identity = (identity,)
        values = group.value.to_numpy(float)
        valid = np.isfinite(values)
        require(len(group) == expected_children, 'missing aggregation children')
        rows.append({**dict(zip(keys, identity)), 'value': float(values[valid].mean()) if valid.any() else np.nan,
                     'n_units': int(group.n_units.sum()), 'n_valid_units': int(group.n_valid_units.sum()),
                     'n_na_units': int(group.n_na_units.sum()), 'n_children': len(group), 'n_valid_children': int(valid.sum())})
    return pd.DataFrame(rows)


def paired_gain(left, right, error=False):
    left, right = np.asarray(left, np.float64), np.asarray(right, np.float64)
    require(left.shape == right.shape and not np.isinf(left).any() and not np.isinf(right).any(), 'paired invalid')
    valid = np.isfinite(left) & np.isfinite(right)
    value = np.full(left.shape, np.nan)
    value[valid] = right[valid] - left[valid] if error else left[valid] - right[valid]
    return value


def score_fold(output, cohort, fold, methods, contrasts, predictions, truth, query, panel, gene_min_n):
    """All predictions are frozen before this scoring-only function is called."""
    require(predictions.shape == (len(methods), len(query), len(panel)) and truth.shape == predictions.shape[1:], 'score shape')
    condition_frames, gene_frames, dose_summary = [], [], []
    condition_cache, gene_cache = {}, {}
    doses = sorted(query.dose_value.unique())
    require(len(doses) == 4 and query.cell_line.nunique() == query.time.nunique() == 1, 'score context/exposure')
    for i, method in enumerate(methods):
        scores = condition_metrics(predictions[i], truth)
        conditions = query[['atomic_id', 'source_entity_key', 'dose_value', 'time']].reset_index(drop=True).copy()
        conditions['cohort'] = cohort; conditions['heldout_cell_line'] = fold; conditions['method'] = method
        for key, value in scores.items():
            conditions[key] = value
        condition_frames.append(conditions); condition_cache[method] = conditions
        for dose in doses:
            selected = np.flatnonzero(query.dose_value.to_numpy() == dose)
            require(len(selected) >= gene_min_n and query.iloc[selected].source_entity_key.is_unique, 'gene endpoint support')
            gene = gene_metrics(predictions[i, selected], truth[selected])
            order = paired_order_accuracy_by_gene(predictions[i, selected], truth[selected])
            frame = panel[['source_feature_row', 'original_ensembl_id', 'gene_symbol']].reset_index(drop=True).copy()
            frame['cohort'] = cohort; frame['heldout_cell_line'] = fold; frame['method'] = method; frame['dose_value'] = dose
            for key, value in gene.items():
                frame[key] = value
            frame['pair_order_accuracy'] = order['accuracy']; frame['n_pairs'] = order['n_pairs']
            gene_frames.append(frame); gene_cache[(method, dose)] = frame
            values = {'mae': scores['mae'][selected], 'rmse': scores['rmse'][selected],
                      'condition_spearman': scores['spearman'][selected], 'gene_spearman': gene['spearman'], 'gene_order': order['accuracy']}
            keys = dict(cohort=cohort, heldout_cell_line=fold, method=method, dose_value=dose)
            dose_summary += [summary_row(keys, metric, values[metric]) for metric in METRICS]
    table(output/'condition_metrics.tsv.gz', pd.concat(condition_frames, ignore_index=True))
    table(output/'gene_metrics.tsv.gz', pd.concat(gene_frames, ignore_index=True))
    paired_conditions, paired_genes, paired_dose = [], [], []
    for contrast in contrasts.itertuples():
        left, right = condition_cache[contrast.left], condition_cache[contrast.right]
        require(np.array_equal(left.atomic_id, right.atomic_id), 'paired condition axis')
        common = dict(cohort=cohort, heldout_cell_line=fold, contrast=contrast.contrast, left=contrast.left, right=contrast.right)
        pair = left[['atomic_id', 'dose_value']].copy()
        for key, value in common.items():
            pair[key] = value
        for metric in ['mae', 'rmse', 'spearman']:
            pair[metric + '_gain'] = paired_gain(left[metric], right[metric], metric in ['mae', 'rmse'])
        paired_conditions.append(pair)
        for dose in doses:
            leftg, rightg = gene_cache[(contrast.left, dose)], gene_cache[(contrast.right, dose)]
            require(np.array_equal(leftg.source_feature_row, rightg.source_feature_row), 'paired gene axis')
            pairg = leftg[['source_feature_row']].copy()
            for key, value in common.items():
                pairg[key] = value
            pairg['dose_value'] = dose
            pairg['spearman_gain'] = paired_gain(leftg.spearman, rightg.spearman)
            pairg['order_gain'] = paired_gain(leftg.pair_order_accuracy, rightg.pair_order_accuracy)
            paired_genes.append(pairg)
            selected = pair.dose_value.eq(dose)
            values = {'mae': pair.loc[selected, 'mae_gain'], 'rmse': pair.loc[selected, 'rmse_gain'],
                      'condition_spearman': pair.loc[selected, 'spearman_gain'],
                      'gene_spearman': pairg.spearman_gain, 'gene_order': pairg.order_gain}
            keys = {**common, 'family': contrast.family, 'dose_value': dose}
            paired_dose += [summary_row(keys, metric, values[metric]) for metric in METRICS]
    table(output/'paired_condition_metrics.tsv.gz', pd.concat(paired_conditions, ignore_index=True))
    table(output/'paired_gene_metrics.tsv.gz', pd.concat(paired_genes, ignore_index=True))
    return pd.DataFrame(dose_summary), pd.DataFrame(paired_dose), {
        'condition_rows': sum(len(x) for x in condition_frames), 'gene_rows': sum(len(x) for x in gene_frames),
        'paired_condition_rows': sum(len(x) for x in paired_conditions), 'paired_gene_rows': sum(len(x) for x in paired_genes)}


def response_slice(config, original_atoms, rows, panel):
    """Physical NPZ decompression is separate from the fold-scoped logical slice."""
    path = Path(config['source_root'])/'effects/atomic_effects_v1/arrays.npz'
    with np.load(path, allow_pickle=False) as data:
        require(np.array_equal(data['atomic_id'], original_atoms.atomic_id.to_numpy(str)), 'effect atom axis')
        columns = panel.union_column.to_numpy(int)
        require(np.array_equal(data['source_feature_row'][columns], panel.source_feature_row.to_numpy(int)), 'effect feature axis')
        raw = data['effect_log2fc']
        require(raw.dtype == np.float32 and raw.shape == (2256, 3730), 'effect dtype/shape')
        selected = raw[np.ix_(np.asarray(rows, int), columns)].copy()
    require(np.isfinite(selected).all(), 'selected nonfinite effects')
    return selected


def copy_exact(source, target):
    with Path(target).open('xb') as stream:
        stream.write(Path(source).read_bytes())


def run(config_path, execute, expected_seal_sha256):
    require(execute, 'explicit --execute required')
    config_path = Path(config_path).resolve(); experiment = config_path.parent
    seal_path = experiment/'execution_sealed.json'
    require(expected_seal_sha256 and sha(seal_path) == expected_seal_sha256, 'explicit exact execution seal hash required')
    config = load_config(config_path)
    sealed = json.loads(seal_path.read_text())
    require(sealed['status'] == 'SEALED_BEFORE_NEW_RESPONSE_FIT' and sealed['script_sha256'] == sha(__file__), 'script/seal')
    require(sealed['config_sha256'] == sha(config_path) and sealed['input_manifest_sha256'] == sha(experiment/'input_manifest.json'), 'config/input seal')
    require(sha(sealed['test_report_path']) == sealed['test_report_sha256'], 'test report seal')
    for row in sealed['preflight_outputs']:
        require(sha(experiment/row['path']) == row['sha256'], 'preflight changed')
    manifest = json.loads((experiment/'input_manifest.json').read_text())
    for row in manifest['files']:
        require(sha(row['path']) == row['sha256'], 'input changed before execution: ' + row['path'])
    output = experiment/'results'; output.mkdir(exist_ok=False)
    start = time.perf_counter(); counts = []; folds = []; n_fits = 0
    try:
        copy_exact(config_path, output/'frozen_config.json')
        copy_exact(experiment/'input_manifest.json', output/'input_manifest.json')
        copy_exact(seal_path, output/'execution_seal.json')
        write_json(output/'run_started.json', {'started_utc': now(), 'execution_seal_sha256': expected_seal_sha256})
        atoms, cohorts, panels = load_metadata(config)
        methods = method_registry(config); contrasts = contrast_registry(config, methods)
        table(output/'method_registry.tsv', methods); table(output/'contrast_registry.tsv', contrasts)
        for c in config['cohorts']:
            cohort = c['cohort']; meta = cohorts[cohort]
            registry = methods.loc[methods.cohort.eq(cohort)]
            mappings, texts, vectors = texts_and_vectors(config, meta, cohort == 'mechanism57')
            native = native_inputs(config, meta, c['native'])
            for heldout in config['folds']:
                print('FIT_START', cohort, heldout, flush=True)
                folder = output/cohort/heldout; folder.mkdir(parents=True)
                source = np.flatnonzero(meta.cell_line.ne(heldout)); query = np.flatnonzero(meta.cell_line.eq(heldout))
                selected = landmarks(meta, source, config['n_landmarks'], config['seed'])
                panel = panels.loc[panels.heldout_cell_line.eq(heldout)].sort_values('rank').reset_index(drop=True)
                table(folder/'source_atoms.tsv', meta.iloc[source]); table(folder/'query_atoms.tsv', meta.iloc[query])
                table(folder/'landmarks.tsv', meta.iloc[selected]); table(folder/'gene_panel.tsv', panel)
                table(folder/'text_mapping.tsv', mappings)
                features, feature_audit = make_kernels(config, meta, source, query, selected, registry, texts, vectors, native)
                # Only source logical rows are selected here. No target truth enters any fit function.
                source_y = response_slice(config, atoms, meta.iloc[source].effect_row, panel).astype(np.float64)
                axes = dict(source_feature_row=panel.source_feature_row.to_numpy(int), original_ensembl_id=panel.original_ensembl_id.to_numpy(str))
                save_npz(folder/'source_response.npz', source_y=source_y, atomic_id=meta.iloc[source].atomic_id.to_numpy(str), **axes)
                predictions, pairs = references(meta, source, query, source_y)
                table(folder/'same_drug_source_pairs.tsv', pairs)
                for method in registry.loc[registry.kind.ne('reference'), 'method']:
                    kernel = features[method]
                    save_npz(folder/'kernels'/f'{method}.npz', source_features=kernel[source], query_features=kernel[query],
                        source_atomic_id=meta.iloc[source].atomic_id.to_numpy(str), query_atomic_id=meta.iloc[query].atomic_id.to_numpy(str),
                        landmark_atomic_id=meta.iloc[selected].atomic_id.to_numpy(str))
                    prediction, parameters = fit_predict(kernel[source], source_y, kernel[query], config['alpha'])
                    predictions[method] = prediction
                    save_npz(folder/'parameters'/f'{method}.npz', **parameters)
                    n_fits += 1
                values = np.stack([predictions[m] for m in registry.method]).astype(np.float32)
                require(values.shape == (c['methods'], c['query_atoms'], 3000) and np.isfinite(values).all(), 'all method prediction shape')
                save_npz(folder/'predictions.npz', predictions=values, method=registry.method.to_numpy(str),
                         atomic_id=meta.iloc[query].atomic_id.to_numpy(str), **axes)
                write_json(folder/'feature_audit.json', feature_audit)
                record = {'cohort': cohort, 'heldout_cell_line': heldout, 'path': str((folder/'predictions.npz').relative_to(output)),
                          'sha256': sha(folder/'predictions.npz'), 'n_methods': len(registry), 'n_queries': len(query), 'n_genes': 3000}
                folds.append(record)
                print('FIT_FROZEN', cohort, heldout, flush=True)
        require(n_fits == config['expected_ridge_fits'] and sum(x['n_methods'] for x in folds) == config['expected_method_fold_outputs'], 'full fit count')
        write_json(output/'predictions_sealed.json', {'status': 'ALL_150_OUTPUTS_FROZEN_BEFORE_SCORING', 'created_utc': now(),
            'n_ridge_fits': n_fits, 'n_method_fold_outputs': 150, 'folds': folds})
        write_json(output/'scoring_started.json', {'started_utc': now(), 'predictions_sealed_sha256': sha(output/'predictions_sealed.json')})
        summaries, paired = [], []
        for record in folds:
            cohort, heldout = record['cohort'], record['heldout_cell_line']
            print('SCORING', cohort, heldout, flush=True)
            meta = cohorts[cohort]; query_rows = np.flatnonzero(meta.cell_line.eq(heldout))
            query = meta.iloc[query_rows].reset_index(drop=True)
            folder = output/cohort/heldout
            panel = panels.loc[panels.heldout_cell_line.eq(heldout)].sort_values('rank').reset_index(drop=True)
            require(sha(folder/'predictions.npz') == record['sha256'], 'prediction changed before score')
            with np.load(folder/'predictions.npz', allow_pickle=False) as data:
                prediction, method_names = data['predictions'], data['method'].tolist()
                require(np.array_equal(data['atomic_id'], query.atomic_id.to_numpy(str)), 'prediction/query axis')
            truth = response_slice(config, atoms, query.effect_row, panel)
            save_npz(folder/'truth.npz', truth=truth, atomic_id=query.atomic_id.to_numpy(str),
                     source_feature_row=panel.source_feature_row.to_numpy(int), original_ensembl_id=panel.original_ensembl_id.to_numpy(str))
            summary, pair, count = score_fold(folder, cohort, heldout, method_names, contrasts.loc[contrasts.cohort.eq(cohort)],
                                              prediction, truth, query, panel, config['gene_min_n'])
            summaries.append(summary); paired.append(pair); counts.append(count)
            require(sha(folder/'predictions.npz') == record['sha256'], 'scoring modified predictions')
            print('SCORED', cohort, heldout, flush=True)
        dose = pd.concat(summaries, ignore_index=True); pair_dose = pd.concat(paired, ignore_index=True)
        fold_keys = ['cohort', 'heldout_cell_line', 'method', 'metric']; macro_keys = ['cohort', 'method', 'metric']
        pair_fold_keys = ['cohort', 'heldout_cell_line', 'contrast', 'left', 'right', 'family', 'metric']
        pair_macro_keys = [key for key in pair_fold_keys if key != 'heldout_cell_line']
        fold_table = aggregate(dose, fold_keys, 4); macro = aggregate(fold_table, macro_keys, 3)
        pair_fold = aggregate(pair_dose, pair_fold_keys, 4); pair_macro = aggregate(pair_fold, pair_macro_keys, 3)
        for name, frame in [('dose_summary', dose), ('fold_summary', fold_table), ('macro_summary', macro),
                            ('paired_dose_summary', pair_dose), ('paired_fold_summary', pair_fold), ('paired_macro_summary', pair_macro)]:
            table(output/(name + '.tsv'), frame)
        actual_counts = {key: sum(row[key] for row in counts) for key in counts[0]}
        require(actual_counts == {'condition_rows': 33996, 'gene_rows': 1800000, 'paired_condition_rows': 66624, 'paired_gene_rows': 3528000}, 'complete metric row counts')
        for row in manifest['files']:
            require(sha(row['path']) == row['sha256'], 'input mutated during execution: ' + row['path'])
        audit = {'status': 'PASS', 'scope': config['scope'], 'created_utc': now(), 'n_ridge_fits': n_fits,
            'n_method_fold_outputs': 150, 'n_contrasts': len(contrasts), 'metric_rows': actual_counts,
            'config_sha256': sha(config_path), 'script_sha256': sha(__file__), 'input_manifest_sha256': sha(output/'input_manifest.json'),
            'execution_seal_sha256': sha(output/'execution_seal.json'), 'predictions_sealed_sha256': sha(output/'predictions_sealed.json'),
            'scoring_started_sha256': sha(output/'scoring_started.json'), 'input_hashes_unchanged': True,
            'target_truth_passed_to_fit': False, 'independent_numeric_validation': False,
            'seconds': time.perf_counter()-start, 'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'package_versions': {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'threadpoolctl']}}
        write_json(output/'audit.json', audit)
        files = [{'path': str(path.relative_to(output)), 'bytes': path.stat().st_size, 'sha256': sha(path)}
                 for path in sorted(output.rglob('*')) if path.is_file()]
        write_json(output/'output_manifest.json', {'files': files})
        print(json.dumps({'status': 'PASS', 'seconds': audit['seconds'], 'metric_rows': actual_counts,
                          'audit_sha256': sha(output/'audit.json'), 'output_manifest_sha256': sha(output/'output_manifest.json')}), flush=True)
    except BaseException as error:
        if not (output/'failure.json').exists():
            write_json(output/'failure.json', {'status': 'FAIL', 'created_utc': now(), 'error_type': type(error).__name__,
                       'message': str(error), 'script_sha256': sha(__file__), 'input_manifest_sha256': sha(experiment/'input_manifest.json')})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'seal', 'run'])
    parser.add_argument('--config', type=Path, default=PACKAGE/'experiments/e08_prediction_v1/config.json')
    parser.add_argument('--test-report', type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--expected-seal-sha256')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.command == 'prepare':
            prepare(args.config)
        elif args.command == 'seal':
            require(args.test_report is not None, '--test-report required')
            seal(args.config, args.test_report)
        else:
            run(args.config, args.execute, args.expected_seal_sha256)


if __name__ == '__main__':
    main()
