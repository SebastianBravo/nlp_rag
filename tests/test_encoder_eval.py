"""Análisis del encoder de punta a punta con un encoder falso (bolsa de palabras) sobre un corpus sintético."""
import hashlib
import json
import re
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pipeline import embedding, encoder_eval

DIM = 64

def bow(texts):
    out = np.zeros((len(texts), DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in re.findall(r'\w+', t.lower()):
            out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1
    out += 1e-3
    return out / np.linalg.norm(out, axis=1, keepdims=True)

DOCS = {
    'd1': ('zqxa humedal restaurado con aves migratorias en la cuenca', '2025-02-26'),
    'd2': ('zqxb plenaria suspendida por falta de quorum en cali', '2024-11-16'),
    'd3': ('zqxc paneles solares frente a biocombustibles en tierra agricola', '2025-05-01'),
    'd4': ('zqxd la plenaria se reanudo en roma con foco en financiacion', '2025-02-26'),
}

@pytest.fixture(scope='module')
def result(tmp_path_factory):
    root = tmp_path_factory.mktemp('enc')
    corpus = pd.DataFrame([{'document_id': d, 'text': t, 'created_at': f, 'collection_name': 'url_crawls'} for d, (t, f) in DOCS.items()])
    pq.write_table(pa.Table.from_pandas(corpus, preserve_index=False), root / 'corpus.parquet')
    chunks = pd.DataFrame([{'chunk_id': f'c{d}', 'document_id': d, 'provenance_ids': [d], 'char_start': 0, 'text': t,
                            'collection_name': 'url_crawls', 'language_detected': 'es', 'created_at': f}
                           for d, (t, f) in DOCS.items()])
    chunks.to_parquet(root / 'chunks.parquet', index=False)
    pq.write_table(embedding.vectors_table(chunks['chunk_id'], bow(chunks['text'].tolist()), [10] * 4), root / 'vectors.parquet')
    gold = [
        {'id': 'Q1', 'split': 'dev', 'tipo': 'directa', 'idioma_pregunta': 'es', 'idioma_evidencia': 'es', 'colecciones': ['url_crawls'],
         'ventana_fecha': None, 'pregunta': 'paneles solares y biocombustibles', 'debe_abstenerse': False,
         'relevantes': [{'document_id': 'd3', 'char_start': 6, 'char_end': 21, 'grado': 2}]},
        {'id': 'Q2', 'split': 'dev', 'tipo': 'directa', 'idioma_pregunta': 'es', 'idioma_evidencia': 'es', 'colecciones': ['url_crawls'],
         'ventana_fecha': {'desde': '2025-02-25', 'hasta': '2025-02-28'}, 'pregunta': 'donde se reanudo la plenaria',
         'debe_abstenerse': False, 'relevantes': [{'document_id': 'd4', 'char_start': None, 'char_end': None, 'grado': 2}]},
        {'id': 'Q3', 'split': 'test', 'tipo': 'directa', 'idioma_pregunta': 'es', 'idioma_evidencia': 'es', 'colecciones': ['url_crawls'],
         'ventana_fecha': None, 'pregunta': 'no se debe usar', 'debe_abstenerse': False,
         'relevantes': [{'document_id': 'd1', 'char_start': None, 'char_end': None, 'grado': 2}]},
    ]
    (root / 'gold.jsonl').write_text(''.join(json.dumps(g, ensure_ascii=False) + '\n' for g in gold), encoding='utf-8')
    paraphrases = [{'id': 'Q1', 'parafrasis': {'idioma': 'es', 'texto': 'biocombustibles frente a paneles'},
                    'traduccion': {'idioma': 'en', 'texto': 'solar panels and biofuels'}},
                   {'id': 'Q2', 'parafrasis': {'idioma': 'es', 'texto': 'lugar donde se retomo la plenaria'},
                    'traduccion': {'idioma': 'en', 'texto': 'where did the plenary resume'}}]
    (root / 'para.jsonl').write_text(''.join(json.dumps(p, ensure_ascii=False) + '\n' for p in paraphrases), encoding='utf-8')
    cfg = {'query_variants': {'prefijo_e5': 'query: '}, 'dimensions': [64, 32], 'neighbors': 2}
    profile = {'model': 'falso', 'query_prefix': '', 'passage_prefix': '', 'normalize': True}
    info, summary = encoder_eval.run(root / 'chunks.parquet', root / 'vectors.parquet', root / 'gold.jsonl', root / 'para.jsonl',
                                     root / 'corpus.parquet', profile, lambda texts, prefix: bow([prefix + t for t in texts]),
                                     root / 'out', cfg, {'d1': 'es'}, log=lambda m: None)
    return root, info, summary

def test_only_dev_and_all_variants(result):
    _, info, summary = result
    assert info['gold']['questions'] == 2
    assert set(summary['variant']) == {'perfil', 'prefijo_e5', 'sin_filtro_fecha', 'parafrasis', 'traduccion'}
    assert summary[['recall@10', 'ndcg@10', 'mrr']].apply(lambda s: s.between(0, 1)).all().all()

def test_date_window_restricts_candidates(result):
    root, _, _ = result
    per_question = pd.read_csv(root / 'out' / 'tables' / 'metricas_por_pregunta.csv')
    row = per_question[(per_question['id'] == 'Q2') & (per_question['variant'] == 'perfil')].iloc[0]
    assert row['recall@1'] == 1
    neighbors = pd.read_csv(root / 'out' / 'tables' / 'vecinos_ejemplo.csv')
    assert set(neighbors.loc[neighbors['id'] == 'Q2', 'document_id']) <= {'d1', 'd4'}

def test_matryoshka_and_outputs_without_corpus_text(result):
    root, info, _ = result
    mrl = pd.read_csv(root / 'out' / 'tables' / 'matryoshka.csv')
    assert sorted(zip(mrl['dimension'], mrl['renormalized'])) == [(32, False), (32, True), (64, True)]
    for name in info['outputs']['figures']:
        assert (root / 'out' / 'figures' / name).stat().st_size > 0
    for path in (root / 'out').rglob('*'):
        if path.suffix in ('.csv', '.json'):
            assert 'zqx' not in path.read_text(encoding='utf-8'), path.name
