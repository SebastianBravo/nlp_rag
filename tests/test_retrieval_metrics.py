import math
import numpy as np
import pandas as pd
import pytest
from pipeline import embedding, metrics, retrieval

def question(relevantes, **kw):
    return {'id': kw.get('id', 'Q1'), 'tipo': 'directa', 'idioma_pregunta': 'es', 'idioma_evidencia': 'es',
            'colecciones': ['url_crawls'], 'debe_abstenerse': kw.get('abstenerse', False), 'relevantes': relevantes,
            'ventana_fecha': kw.get('ventana')}

def test_metrics_by_hand():
    q = question([{'document_id': 'a', 'grado': 2}, {'document_id': 'b', 'grado': 1}, {'document_id': 'z', 'grado': 0}])
    ranked = ['x', 'b', 'y', 'a']
    row = metrics.question_metrics(ranked, q, ks=(1, 2, 4))
    assert row['recall@1'] == 0 and row['recall@2'] == 0.5 and row['recall@4'] == 1
    assert row['hit@1'] == 0 and row['hit@2'] == 1
    assert row['mrr'] == 0.5
    dcg = 1 / math.log2(3) + 2 / math.log2(5)
    idcg = 2 / math.log2(2) + 1 / math.log2(3)
    assert row['ndcg@10'] == pytest.approx(dcg / idcg)

def test_only_answerable_questions_are_evaluated():
    questions = [question([{'document_id': 'a', 'grado': 2}], id='Q1'), question([], id='Q2'),
                 question([{'document_id': 'a', 'grado': 2}], id='Q3', abstenerse=True)]
    table = metrics.evaluate({'Q1': ['a'], 'Q2': ['a'], 'Q3': ['a']}, questions, ks=(1,))
    assert table['id'].tolist() == ['Q1'] and table['recall@1'].iloc[0] == 1
    summary = metrics.summarize(table.assign(g='x'), 'g')
    assert summary['group'].tolist() == ['x', 'total'] and summary['n'].tolist() == [1, 1]

def test_top_chunks_order_and_mask():
    matrix = np.array([[1, 0], [0.6, 0.8], [0, 1]], dtype=np.float32)
    idx, scores = retrieval.top_chunks(np.array([[1, 0]], dtype=np.float32), matrix, 2)
    assert idx[0].tolist() == [0, 1] and scores[0][0] == pytest.approx(1)
    idx, _ = retrieval.top_chunks(np.array([[1, 0]], dtype=np.float32), matrix, 2, mask=np.array([[False, True, True]]))
    assert idx[0].tolist() == [1, 2]

def test_rank_documents_uses_provenance_and_best_rank():
    provenance = [['a'], ['b', 'c'], ['a'], ['d']]
    assert retrieval.rank_documents([0, 1, 2, 3], provenance) == ['a', 'b', 'c', 'd']
    assert retrieval.rank_documents([0, 1, 2, 3], provenance, allowed={'c', 'd'}) == ['c', 'd']
    assert retrieval.rank_documents([0, 1, 2, 3], provenance, limit=2) == ['a', 'b']

def test_documents_in_window_includes_both_ends():
    dates = pd.Series({'a': '2025-02-25', 'b': '2025-02-28T23:59:00Z', 'c': '2025-03-01', 'd': None})
    assert retrieval.documents_in_window(dates, {'desde': '2025-02-25', 'hasta': '2025-02-28'}) == {'a', 'b'}

def test_evidence_hit_with_offsets_and_provenance_text():
    chunks = pd.DataFrame({'document_id': ['a', 'x'], 'char_start': [10, 0], 'text': ['diez a veinte', 'otro texto'],
                           'provenance_ids': [['a', 'b'], ['x']]})
    texts = {'b': 'prefijo largo diez a veinte'}
    assert retrieval.evidence_hit([0], chunks, {'a': [(15, 30)]}, texts, 1) == 1.0
    assert retrieval.evidence_hit([0], chunks, {'a': [(40, 50)]}, texts, 1) == 0.0
    assert retrieval.evidence_hit([0], chunks, {'b': [(14, 18)]}, texts, 1) == 1.0
    assert retrieval.evidence_hit([1, 0], chunks, {'a': [(15, 30)]}, texts, 1) == 0.0

def test_embed_in_parts_resumes_and_keeps_order(tmp_path):
    calls = []
    def fake_embed(texts):
        calls.append(list(texts))
        return np.array([[float(len(t)), 1.0] for t in texts])
    ids, texts = [f'c{i}' for i in range(5)], ['a', 'bb', 'ccc', 'dddd', 'eeeee']
    out = tmp_path / 'v.parquet'
    parts = out.with_name('v.parquet.parts')
    parts.mkdir()
    import pyarrow.parquet as pq
    pq.write_table(embedding.vectors_table(ids[:2], np.array([[1.0, 1.0], [2.0, 1.0]]), [1, 1]), parts / 'part-00000.parquet')
    computed = embedding.embed_in_parts(ids, texts, fake_embed, lambda t: [1] * len(t), out, part_size=2, log=lambda m: None)
    assert computed == 2 and calls == [['ccc', 'dddd'], ['eeeee']]
    read_ids, matrix, n_tokens = embedding.read_vectors(out)
    assert read_ids == ids and matrix.dtype == np.float32 and matrix[:, 0].tolist() == [1, 2, 3, 4, 5]
    assert not parts.exists() and n_tokens.tolist() == [1] * 5

def test_strict_variant_uses_only_grade_two():
    q = question([{'document_id': 'a', 'grado': 2}, {'document_id': 'b', 'grado': 1}])
    row = metrics.question_metrics(['x', 'b', 'y', 'a'], q, ks=(2, 4))
    assert row['recall@2_g2'] == 0 and row['recall@4_g2'] == 1 and row['hit@2_g2'] == 0 and row['mrr_g2'] == 0.25
    only1 = question([{'document_id': 'b', 'grado': 1}])
    row = metrics.question_metrics(['b'], only1, ks=(1,))
    assert row['hit@1'] == 1 and math.isnan(row['recall@1_g2']) and math.isnan(row['hit@1_g2']) and math.isnan(row['mrr_g2'])
