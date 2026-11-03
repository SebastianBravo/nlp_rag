"""Ejecución completa del perfilado sobre un corpus sintético, con tokenizador y detector de idioma falsos."""
import hashlib
import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pipeline import eda

CONFIG = {
    'seed': 3, 'batch_size': 50, 'version_date': '2026-10-19',
    'samples': {'sizes': [20, 100], 'strata': ['collection_name', 'language_base'], 'min_share': 0.05},
    'tokens': {'thresholds': [8, 16]},
    'dates': {'plausible_start': {'bluesky_posts': '2023-01-01'}},
    'language': {'max_chars': 200, 'min_share': 0.05, 'length_bins': [0, 100, 1000, 100000]},
    'near_duplicates': {'sample': 100, 'shingle_words': 3, 'num_perm': 64, 'threshold': 0.8, 'examples_per_type': 2},
    'noise': {'short_words': 5, 'boilerplate_collections': ['url_crawls'], 'boilerplate_min_chars': 10, 'boilerplate_min_docs': 5},
    'urls': {'tracking_params': ['utm_*'], 'top_domains': 5},
    'acquisition_fields': ['strategy', 'keyword_family'],
}

def synthetic_corpus(n=240):
    rows = []
    for i in range(n):
        posts = i % 3 == 0
        text = (f'post zqx{i} sobre el tema número {i % 7}' if posts else
                f'Titular zqx{i}\nCuerpo de la noticia {i} con detalles propios zqx{i}\nSuscríbete al boletín del medio')
        if i % 40 == 1: text = rows[0]['text']                       # duplicado exacto entre colecciones
        rows.append({'document_id': f'doc{i:04d}', 'text': text,
                     'collection_name': 'bluesky_posts' if posts else 'url_crawls',
                     'source': 'bluesky' if posts else 'media_cloud', 'language_base': ['es', 'en', None][i % 3 if i % 5 else 0],
                     'url_domain': 'bsky.app' if posts else f'medio{i % 4}.com',
                     'url': f'https://medio{i % 4}.com/n/{i}?utm_source=x' if not posts else f'https://bsky.app/p/{i}',
                     'final_url': None, 'uri': f'u{i}', 'mongo_id': f'm{i}',
                     'created_at': '2025-01-0%d' % (1 + i % 9) if i % 2 else f'2025-02-03T10:00:{i % 60:02d}Z',
                     'synced_at': '2026-10-06 10:00:00+00:00', 'strategy': None if posts else 'whitelist',
                     'keyword_family': 'A', 'collection_query': 'q', 'keyword_matches': ['k'] if i % 2 else [],
                     'word_count': len(text.split()), 'char_count': len(text),
                     'text_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest()})
    return pd.DataFrame(rows)

@pytest.fixture(scope='module')
def result(tmp_path_factory):
    root = tmp_path_factory.mktemp('eda')
    df = synthetic_corpus()
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), root / 'corpus.parquet')
    profile = eda.run(root / 'corpus.parquet', CONFIG, root / 'reports', root / 'samples', root / 'work',
                      count_tokens=lambda texts: [len(t.split()) + 3 for t in texts], max_tokens=16,
                      detect_languages=lambda texts, max_chars: ['es' if 'noticia' in t else 'en' for t in texts], log=lambda m: None)
    return root, df, profile

def test_integrity_and_counts(result):
    _, df, profile = result
    integrity = profile['integrity']
    assert integrity['schema_ok'] and integrity['document_id_unique']
    assert integrity['recount_mismatches'] == {'char_count': 0, 'word_count': 0, 'text_sha256': 0}
    assert profile['distributions']['documents'] == len(df)
    assert profile['duplicates']['cross_collection_groups'] == 1

def test_outputs_exist(result):
    root, _, profile = result
    for name in profile['outputs']['tables']:
        assert (root / 'reports' / 'tables' / name).is_file()
    for name in profile['outputs']['figures']:
        assert (root / 'reports' / 'figures' / name).stat().st_size > 0
    assert (root / 'reports' / 'profile.json').is_file()
    assert (root / 'work' / 'document_features.parquet').is_file()
    lines = pd.read_parquet(root / 'work' / 'boilerplate_lines.parquet')
    assert set(lines['url_domain']) == {f'medio{i}.com' for i in range(4)} and (lines['documents'] >= 5).all()

def test_samples_are_nested_and_written(result):
    root, _, _ = result
    small = (root / 'samples' / 'sample_20_ids.txt').read_text().split()
    large = (root / 'samples' / 'sample_100_ids.txt').read_text().split()
    assert len(small) == 20 and len(large) == 100 and set(small) <= set(large)
    meta = json.loads((root / 'samples' / 'sample_20_meta.json').read_text())
    assert meta['seed'] == 3 and meta['nested_in'] == [100] and len(meta['parquet_sha256']) == 64
    assert json.loads((root / 'samples' / 'sample_full_meta.json').read_text())['size'] == 240

def test_reports_contain_no_corpus_text(result):
    root, _, _ = result
    published = [p for p in (root / 'reports').rglob('*') if p.suffix in ('.csv', '.json')]
    published += list((root / 'samples').glob('*'))
    for path in published:
        content = path.read_text(encoding='utf-8')
        assert 'zqx' not in content and 'Suscríbete' not in content and 'bsky.app/p/' not in content, path.name
