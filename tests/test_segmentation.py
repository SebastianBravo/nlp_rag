"""Ejecución completa de la segmentación sobre un corpus sintético, con tokenizador y separador de oraciones falsos."""
import hashlib
import json
import re
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pipeline import segmentation
from pipeline.profiling import hash_lines, normalize_lines

FOOTER = 'Suscríbete al boletín del medio para recibir noticias'
CONFIG = {
    'batch_size': 7, 'min_fragment_tokens': 3, 'features': None, 'example_text': None,
    'documents': {'exclude_template_groups': {'collection': 'url_crawls', 'min_copies': 4}},
    'boilerplate': {'exclude': True},
    'sentences': {'languages': {'es': 'spanish'}, 'default': 'english'},
    'strategies': {'A': {'name': 'ventana de tokens', 'version': 1, 'chunk_tokens': 12, 'overlap_tokens': 3},
                   'B': {'name': 'estructura', 'version': 1, 'chunk_tokens': 12, 'overlap_sentences': 1, 'paragraph_min_fill': 0.5}},
}

def word_offsets(texts):
    return [[m.span() for m in re.finditer(r'\w+|[^\w\s]', t)] for t in texts]

def corpus():
    rows = []
    for i in range(30):
        post = i % 3 == 0
        if post: text = f'Post zqx{i} corto sobre humedales.'
        elif i in (2, 4, 5, 7): text = 'Portada del medio con secciones y menú de navegación.'      # plantilla repetida
        else: text = (f'Noticia zqx{i} sobre el río y sus aves migratorias. Los vecinos {i} piden un plan de manejo serio.\n'
                      f'{FOOTER}\nEl concejo {i} votará el presupuesto pronto.')
        if i == 29: text = rows[1]['text']                                               # duplicado exacto
        rows.append({'document_id': f'doc{i:03d}', 'text': text, 'collection_name': 'bluesky_posts' if post else 'url_crawls',
                     'source': 'bluesky' if post else 'media_cloud', 'url': f'https://x/{i}', 'title': None if post else f'T{i}',
                     'language_base': 'es' if i % 5 else None, 'created_at': '2025-01-01', 'url_domain': 'bsky.app' if post else 'medio.com',
                     'text_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest()})
    return pd.DataFrame(rows)

@pytest.fixture(scope='module')
def result(tmp_path_factory):
    root = tmp_path_factory.mktemp('seg')
    df = corpus()
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), root / 'corpus.parquet')
    (root / 'sample_30_ids.txt').write_text('\n'.join(df['document_id']) + '\n', encoding='utf-8')
    features = pd.DataFrame({'document_id': df['document_id'], 'language_detected': 'es', 'has_mojibake': df.index == 3})
    lines = pd.DataFrame({'url_domain': ['medio.com'], 'line_hash': hash_lines(normalize_lines(pd.Series([FOOTER])))})
    kwargs = dict(token_offsets=word_offsets, count_tokens=lambda texts: [len(o) + 4 for o in word_offsets(texts)], max_tokens=20,
                  features=features, boilerplate_lines=lines, splitter_for=lambda language: (lambda t: [m.span() for m in re.finditer(r'[^.!?]+[.!?]*', t)]),
                  log=lambda m: None)
    infos = {s: segmentation.run(root / 'corpus.parquet', root / 'sample_30_ids.txt', s, CONFIG, root / 'chunks', root / 'reports', **kwargs) for s in 'AB'}
    again = segmentation.run(root / 'corpus.parquet', root / 'sample_30_ids.txt', 'A', CONFIG, root / 'chunks_again', root / 'reports_again', **kwargs)
    return root, df, infos, again

def test_chunks_are_slices_of_the_original_text(result):
    root, df, _, _ = result
    texts = df.set_index('document_id')['text']
    for strategy in 'AB':
        chunks = pd.read_parquet(root / 'chunks' / strategy / 'sample_30.parquet')
        assert all(texts[d][s:e] == t for d, s, e, t in zip(chunks['document_id'], chunks['char_start'], chunks['char_end'], chunks['text']))
        assert not chunks['text'].str.contains('Suscríbete').any()
        assert (chunks['n_tokens'] <= 20).all() and set(chunks['strategy']) == {strategy}

def test_templates_excluded_and_duplicates_traced(result):
    root, _, infos, _ = result
    assert infos['A']['excluded_template_documents'] == 4
    chunks = pd.read_parquet(root / 'chunks' / 'A' / 'sample_30.parquet')
    assert not chunks['document_id'].isin(['doc002', 'doc004', 'doc005', 'doc007']).any()
    assert chunks['chunk_sha256'].is_unique
    traced = chunks[chunks['provenance_ids'].map(len) > 1]
    assert len(traced) and all(list(p) == ['doc001', 'doc029'] for p in traced['provenance_ids'])
    assert chunks.set_index('document_id').loc['doc003', 'has_mojibake'].all()

def test_ids_are_deterministic_between_runs(result):
    root, _, _, _ = result
    first = pd.read_parquet(root / 'chunks' / 'A' / 'sample_30.parquet')['chunk_id'].tolist()
    second = pd.read_parquet(root / 'chunks_again' / 'A' / 'sample_30.parquet')['chunk_id'].tolist()
    assert first == second and len(set(first)) == len(first)

def test_comparison_and_reports_without_corpus_text(result):
    root, _, infos, _ = result
    table = pd.read_csv(root / 'reports' / 'tables' / 'comparacion_sample_30.csv')
    assert set(table['strategy']) == {'A', 'B'} and 'total' in set(table['collection_name'])
    total = table[table['collection_name'] == 'total'].set_index('strategy')
    assert total.loc['B', 'mid_sentence_share'] == 0 and total.loc['A', 'share_over_max_tokens'] == 0
    for name in ('tokens_sample_30.png', 'chunks_por_documento_sample_30.png', 'metricas_sample_30.png'):
        assert (root / 'reports' / 'figures' / name).stat().st_size > 0
    for path in (root / 'reports').rglob('*'):
        if path.suffix in ('.csv', '.json'):
            assert 'zqx' not in path.read_text(encoding='utf-8'), path.name
    run = json.loads((root / 'reports' / 'runs' / 'B_sample_30.json').read_text())
    assert run['params']['overlap_sentences'] == 1 and len(run['parquet']['sha256']) == 64
