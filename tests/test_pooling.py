import json, math
import pytest
from pipeline import pooling

def hit(chunk_id, doc, provenance=None, start=0, text='texto'):
    return (chunk_id, 0.9, {'document_id': doc, 'provenance_ids': provenance or [doc], 'char_start': start, 'text': text})

def question(qid, tipo='directa', split='test', relevantes=None, anotador_2='S', **kw):
    return {'id': qid, 'split': split, 'tipo': tipo, 'pregunta': f'pregunta {qid}', 'ventana_fecha': kw.get('ventana'),
            'relevantes': relevantes or [], 'debe_abstenerse': kw.get('abstenerse', False), 'anotador_2': anotador_2,
            'pooling_pendiente': kw.get('pendiente', False), 'respuesta_referencia': 'ref', 'criterio': None,
            'idioma_evidencia': 'es', 'colecciones': ['url_crawls'], 'estado': 'borrador'}

# ---------------------------------------------------------------- recuperación y unión

def test_rank_with_passages_uses_provenance_window_and_limit():
    hits = [hit('c1', 'a', ['a', 'b']), hit('c2', 'c'), hit('c3', 'b'), hit('c4', 'd')]
    rows = pooling.rank_with_passages(hits)
    assert [r['document_id'] for r in rows] == ['a', 'b', 'c', 'd']          # b entra por la procedencia de c1
    assert rows[1]['chunk_id'] == 'c1' and rows[1]['canonical'] == 'a'
    assert [r['document_id'] for r in pooling.rank_with_passages(hits, allowed={'b', 'd'})] == ['b', 'd']
    assert len(pooling.rank_with_passages(hits, limit=2)) == 2

def test_build_pool_merges_collections_and_keeps_best_passage():
    a = {'Q1': pooling.rank_with_passages([hit('a1', 'x', text='de A'), hit('a2', 'y')])}
    b = {'Q1': pooling.rank_with_passages([hit('b1', 'y', text='de B'), hit('b2', 'x')]), 'Q2': pooling.rank_with_passages([hit('b3', 'z')])}
    pool = pooling.build_pool({'A': a, 'B': b})
    q1 = {c['document_id']: c for c in pool['Q1']}
    assert set(q1) == {'x', 'y'} and q1['x']['ranks'] == {'A': 1, 'B': 2} and q1['y']['ranks'] == {'A': 2, 'B': 1}
    assert q1['x']['best_rank'] == 1 and q1['x']['passage']['text'] == 'de A' and q1['y']['passage']['text'] == 'de B'
    assert [c['document_id'] for c in pool['Q2']] == ['z'] and pool['Q2'][0]['ranks'] == {'B': 1}

def test_pooling_records_have_ids_and_ranks_but_no_text():
    pool = pooling.build_pool({'A': {'Q1': pooling.rank_with_passages([hit('a1', 'x', text='TEXTO SECRETO DEL CORPUS'), hit('a2', 'y')])}})
    q = question('Q1', relevantes=[{'document_id': 'x', 'grado': 2}, {'document_id': 'w', 'grado': 1}, {'document_id': 'v', 'grado': 0}])
    rec = pooling.pooling_records(pool, {'A': 'col_a'}, 20, 'abc', [q])[0]
    assert 'SECRETO' not in json.dumps(rec) and rec['known_relevant'] == 2 and rec['known_relevant_in_pool'] == 1
    assert rec['candidates'][0] == {'document_id': 'x', 'ranks': {'A': 1}} and rec['gold_sha256'] == 'abc'

# ---------------------------------------------------------------- muestras y orden

def test_shuffled_is_deterministic_and_independent_of_other_candidates():
    cands = [{'document_id': d} for d in 'abcdefgh']
    first = [c['document_id'] for c in pooling.shuffled(cands, 42, 'Q1')]
    assert first == [c['document_id'] for c in pooling.shuffled(list(reversed(cands)), 42, 'Q1')]
    assert sorted(first) == list('abcdefgh')
    # quitar un candidato no cambia el orden relativo de los demás
    rest = [c['document_id'] for c in pooling.shuffled([c for c in cands if c['document_id'] != first[0]], 42, 'Q1')]
    assert rest == first[1:]

def test_second_annotator_sample_is_stratified_and_stable():
    qs = []
    for tipo, n in (('directa', 11), ('comparativa', 6), ('sin_respuesta', 5), ('adversarial', 4), ('ambigua', 4)):
        qs += [question(f'{tipo[:3]}{i}', tipo) for i in range(n)]
    qs += [question('dev1', split='dev'), question('Q50', anotador_2=None)]
    chosen = pooling.second_annotator_sample(qs, 16, 42)
    by_type = {q['id']: q['tipo'] for q in qs}
    counts = {t: sum(by_type[i] == t for i in chosen) for t in set(by_type.values())}
    assert len(chosen) == 16 and counts == {'directa': 6, 'comparativa': 3, 'sin_respuesta': 3, 'adversarial': 2, 'ambigua': 2}
    assert 'dev1' not in chosen and 'Q50' not in chosen
    assert chosen == pooling.second_annotator_sample(list(reversed(qs)), 16, 42)
    assert len(pooling.second_annotator_sample(qs, 100, 42)) == 30

# ---------------------------------------------------------------- hojas

def test_passage_with_context_marks_the_chunk():
    doc = 'abcdefghij' * 10
    chunk = doc[40:50]
    out = pooling.passage_with_context(doc, chunk, 40, margin=5)
    assert out == '…' + doc[35:40] + f'\n>>> {chunk} <<<\n' + doc[50:55] + '…'
    other = 'xx' + chunk + 'yy'      # documento de procedencia: el offset del canónico no sirve, se busca el texto
    assert '>>> ' + chunk + ' <<<' in pooling.passage_with_context(other, chunk, 40, margin=5) and 'xx' in pooling.passage_with_context(other, chunk, 40, margin=5)
    assert pooling.passage_with_context('sin relación', chunk, 40) == f'>>> {chunk} <<<'

def test_annotation_rows_are_blind_and_prefilled_only_when_asked():
    cands = [{'document_id': d, 'ranks': {'A': i + 1}, 'best_rank': i + 1} for i, d in enumerate('abcd')]
    q = question('Q1', ventana={'desde': '2025-02-01', 'hasta': '2025-02-28'})
    docs = {d: {'collection_name': 'url_crawls', 'title': f'título {d}', 'created_at': '2025-02-10T00:00:00'} for d in 'abcd'}
    rows = pooling.annotation_rows('Q1', q, cands, docs, {d: f'pasaje {d}' for d in 'abcd'}, {'b': 2}, 42)
    assert sorted(r['document_id'] for r in rows) == list('abcd') and [r['n'] for r in rows] == [1, 2, 3, 4]
    assert {r['document_id']: r['grado'] for r in rows} == {'a': None, 'b': 2, 'c': None, 'd': None}
    assert rows[0]['ventana_fecha'] == '2025-02-01 a 2025-02-28' and rows[0]['fecha'] == '2025-02-10'
    assert all('ranks' not in r and 'rank' not in r for r in rows)
    assert len(pooling.annotation_rows('Q1', q, cands, docs, {}, None, 42, depth=2)) == 2

def test_sheet_roundtrip_keeps_formula_like_text_and_validates_grades(tmp_path):
    pytest.importorskip('openpyxl')
    from openpyxl import load_workbook
    rows = [{'id': 'Q1', 'document_id': 'a', 'pasaje': '=SUM(1,2)', 'grado': None}, {'id': 'Q1', 'document_id': 'b', 'pasaje': 'normal', 'grado': 2}]
    path = tmp_path / 'h.xlsx'
    pooling.write_sheets(path, {'anotacion': (['id', 'document_id', 'pasaje', 'grado'], rows)}, {'anotacion': {'grado': pooling.GRADES}})
    back = pooling.read_sheet(path, 'anotacion')
    assert back[0]['pasaje'] == '=SUM(1,2)' and back[1]['grado'] == 2 and back[0]['grado'] is None
    ws = load_workbook(path)['anotacion']
    assert ws.data_validations.dataValidation[0].formula1 == '"0,1,2"'

def test_parse_grade_and_grades_from_rows():
    assert [pooling.parse_grade(v) for v in (None, '', ' ', 0, '1', 2.0, '2,0')] == [None, None, None, 0, 1, 2, 2]
    for bad in ('3', 'x', 1.5, -1):
        with pytest.raises(ValueError): pooling.parse_grade(bad)
    rows = [{'id': 'Q1', 'document_id': 'a', 'grado': 2}, {'id': 'Q1', 'document_id': 'b', 'grado': 0},
            {'id': 'Q2', 'document_id': 'a', 'grado': None}, {'id': 'Q3', 'document_id': 'a', 'grado': 1},
            {'id': 'Q3', 'document_id': 'b', 'grado': None}, {'id': 'Q4', 'document_id': 'a', 'grado': 'x'}]
    grades, blank, partial, errors = pooling.grades_from_rows(rows)
    assert grades == {('Q1', 'a'): 2, ('Q1', 'b'): 0, ('Q3', 'a'): 1} and blank == ['Q2'] and partial == ['Q3'] and len(errors) == 1

# ---------------------------------------------------------------- acuerdo

def test_kappa_known_values():
    assert pooling.cohen_kappa([0, 1, 2, 1], [0, 1, 2, 1]) == pytest.approx(1)
    assert pooling.cohen_kappa([0, 1, 0, 1], [1, 0, 1, 0], labels=(0, 1), weights='none') == pytest.approx(-1)
    assert pooling.cohen_kappa([0, 2], [2, 0]) == pytest.approx(-1)
    assert math.isnan(pooling.cohen_kappa([0, 0, 0], [0, 0, 0])) and math.isnan(pooling.cohen_kappa([], []))
    # 2x2 clásico: 20 acuerdos en 1, 15 en 0, 5 y 10 desacuerdos -> kappa = 0,4 (po=0,7, pe=0,5)
    a = [1] * 25 + [0] * 25
    b = [1] * 20 + [0] * 5 + [1] * 10 + [0] * 15
    assert pooling.cohen_kappa(a, b, labels=(0, 1), weights='none') == pytest.approx(0.4)
    assert pooling.cohen_kappa([0, 1, 2, 2], [0, 2, 2, 1], weights='quadratic') < pooling.cohen_kappa([0, 1, 2, 2], [0, 1, 2, 1], weights='quadratic')

def test_agreement_uses_only_common_pairs():
    g1 = {('Q1', 'a'): 2, ('Q1', 'b'): 0, ('Q1', 'c'): 1, ('Q2', 'a'): 0}
    g2 = {('Q1', 'a'): 2, ('Q1', 'b'): 1, ('Q1', 'd'): 2, ('Q2', 'a'): 0}
    summary, table, disagreements = pooling.agreement(g1, g2, {'Q1': 'directa', 'Q2': 'sin_respuesta'})
    assert summary['pairs'] == 3 and summary['questions'] == 2 and summary['exact_agreement'] == pytest.approx(2 / 3)
    assert disagreements == [{'id': 'Q1', 'document_id': 'b', 'grado_1': 0, 'grado_2': 1}]
    assert table[0]['id'] == 'Q1' and table[0]['pairs'] == 2 and table[1]['tipo'] == 'sin_respuesta'

# ---------------------------------------------------------------- congelamiento

def test_merge_relevantes_keeps_offsets_and_adds_documents():
    q = question('Q1', relevantes=[{'document_id': 'a', 'char_start': 1, 'char_end': 9, 'grado': 2},
                                   {'document_id': 'a', 'char_start': 20, 'char_end': 30, 'grado': 2},
                                   {'document_id': 'b', 'char_start': 5, 'char_end': 6, 'grado': 1},
                                   {'document_id': 'z', 'char_start': None, 'char_end': None, 'grado': 2}])
    merged = pooling.merge_relevantes(q, {'a': 1, 'b': 0, 'n': 2, 'm': 0})
    assert [(r['document_id'], r['grado']) for r in merged] == [('a', 1), ('a', 1), ('z', 2), ('n', 2)]
    assert merged[0]['char_start'] == 1 and merged[-1] == {'document_id': 'n', 'char_start': None, 'char_end': None, 'grado': 2}
    assert q['relevantes'][0]['grado'] == 2       # no modifica la entrada

def run(module, monkeypatch, *argv):
    monkeypatch.setattr('sys.argv', [module.__name__, *argv])
    module.main()

def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows) + '\n', encoding='utf-8')

def test_annotation_to_freeze_flow(tmp_path, monkeypatch, capsys):
    pytest.importorskip('openpyxl')
    from scripts import apply_annotations, freeze_gold
    monkeypatch.chdir(tmp_path)
    q1 = question('Q1', relevantes=[{'document_id': 'a', 'char_start': 1, 'char_end': 9, 'grado': 2}])
    q2 = question('Q2', 'sin_respuesta', abstenerse=True, pendiente=True)
    q3 = question('Q3', 'ambigua', pendiente=True)
    write_jsonl(tmp_path / 'data/gold/gold.jsonl', [q1, q2, q3])
    head = ['id', 'tipo', 'pregunta', 'ventana_fecha', 'n', 'document_id', 'fuente', 'fecha', 'titulo', 'pasaje', 'grado', 'comentario']
    def row(qid, doc, grade): return {'id': qid, 'tipo': 'x', 'pregunta': f'p {qid}', 'document_id': doc, 'pasaje': f'pasaje {doc}', 'grado': grade}
    sheet1 = [row('Q1', 'a', 2), row('Q1', 'b', 0), row('Q2', 'a', 0), row('Q2', 'b', 0), row('Q3', 'a', 2), row('Q3', 'c', 1)]
    sheet2 = [row('Q1', 'a', 2), row('Q1', 'b', 1), row('Q2', 'a', 0), row('Q2', 'b', 0)]
    ann = tmp_path / 'data/annotation'
    ann.mkdir(parents=True)
    pooling.write_sheets(ann / 'hoja_anotador1.xlsx', {'anotacion': (head, sheet1)})
    pooling.write_sheets(ann / 'hoja_anotador2.xlsx', {'anotacion': (head, sheet2), 'respuestas': (pooling.ANSWER_COLUMNS,
                         [{'id': 'Q1', 'respuesta_propia': 'algo'}, {'id': 'Q2', 'respuesta_propia': 'No consta.'}])})
    pooling.write_sheets(ann / 'hoja_anotador2_fase2.xlsx', {'validez': (pooling.PHASE2_COLUMNS, [{'id': 'Q1', 'validez': 'parcial'}])})
    run(apply_annotations, monkeypatch)
    summary = json.loads(capsys.readouterr().out)
    assert summary['pairs'] == 4 and summary['disagreements'] == 1 and summary['exact_agreement'] == 0.75
    ann2 = [json.loads(l) for l in (tmp_path / 'data/gold/anotacion_2.jsonl').read_text().splitlines()]
    assert {'id': 'Q2', 'tipo': 'respuesta_propia', 'no_consta': True} in ann2 and {'id': 'Q1', 'tipo': 'validez', 'validez': 'parcial'} in ann2
    assert 'pasaje' not in (tmp_path / 'data/gold/anotacion_1.jsonl').read_text()       # solo ids y grados
    # freeze falla mientras haya desacuerdos sin resolver
    with pytest.raises(SystemExit): run(freeze_gold, monkeypatch)
    capsys.readouterr()
    disagreements = pooling.read_sheet(ann / 'desacuerdos.xlsx', 'desacuerdos')
    assert [(d['id'], d['document_id'], d['grado_1'], d['grado_2']) for d in disagreements] == [('Q1', 'b', 0, 1)]
    pooling.write_sheets(ann / 'desacuerdos.xlsx', {'desacuerdos': (list(disagreements[0]), [{**disagreements[0], 'grado_final': 1}])})
    run(freeze_gold, monkeypatch)                                                            # ensayo: no escribe
    dry = json.loads(capsys.readouterr().out)
    assert dry['written'] is False and '"estado": "congelado"' not in (tmp_path / 'data/gold/gold.jsonl').read_text()
    run(freeze_gold, monkeypatch, '--write')
    out = json.loads(capsys.readouterr().out)
    frozen = {q['id']: q for q in map(json.loads, (tmp_path / 'data/gold/gold.jsonl').read_text().splitlines())}
    assert out['written'] and out['gold_sha256_after'] != out['gold_sha256_before']
    assert [(r['document_id'], r['grado']) for r in frozen['Q1']['relevantes']] == [('a', 2), ('b', 1)]
    assert frozen['Q2']['relevantes'] == [] and frozen['Q3']['relevantes'][0]['document_id'] == 'a'
    assert all(q['estado'] == 'congelado' and not q['pooling_pendiente'] for q in frozen.values())
    assert frozen['Q1']['doble_anotacion'] is True and 'doble_anotacion' not in frozen['Q3']

def test_freeze_blocks_abstention_with_relevant_documents(tmp_path, monkeypatch, capsys):
    from scripts import freeze_gold
    monkeypatch.chdir(tmp_path)
    write_jsonl(tmp_path / 'data/gold/gold.jsonl', [question('Q2', 'sin_respuesta', abstenerse=True, pendiente=True)])
    write_jsonl(tmp_path / 'data/gold/anotacion_1.jsonl', [{'id': 'Q2', 'document_id': 'a', 'grado': 2}])
    with pytest.raises(SystemExit): run(freeze_gold, monkeypatch)
    assert 'debe abstenerse' in capsys.readouterr().out

def test_freeze_blocks_unreviewed_pending_questions(tmp_path, monkeypatch, capsys):
    from scripts import freeze_gold
    monkeypatch.chdir(tmp_path)
    write_jsonl(tmp_path / 'data/gold/gold.jsonl', [question('Q9', 'ambigua', pendiente=True)])
    write_jsonl(tmp_path / 'data/gold/anotacion_1.jsonl', [])
    with pytest.raises(SystemExit): run(freeze_gold, monkeypatch)
    assert 'Q9' in capsys.readouterr().out


def test_pool_gold_script_end_to_end_with_fake_milvus(tmp_path, monkeypatch, capsys):
    """Humo del flujo completo con un Milvus simulado y un corpus de juguete: hojas, pooling.jsonl y cobertura."""
    pytest.importorskip('openpyxl')
    import numpy as np
    import pandas as pd
    from scripts import pool_gold
    monkeypatch.chdir(tmp_path)
    corpus = pd.DataFrame([{'document_id': d, 'text': f'texto largo del documento {d} ' * 5, 'collection_name': 'url_crawls',
                            'title': f'título {d}', 'created_at': '2025-02-10'} for d in ('a', 'b', 'c', 'd')])
    corpus.to_parquet(tmp_path / 'corpus.parquet')
    chunk = corpus.set_index('document_id')['text']['a'][:30]
    qs = [question('Q1', relevantes=[{'document_id': 'a', 'char_start': 0, 'char_end': 10, 'grado': 2}], pendiente=True),
          question('Q2', 'sin_respuesta', abstenerse=True, pendiente=True, ventana={'desde': '2025-02-01', 'hasta': '2025-02-28'})]
    write_jsonl(tmp_path / 'data/gold/gold.jsonl', qs)
    (tmp_path / 'enc.yaml').write_text('profiles:\n  granite-97m: {}\n')
    (tmp_path / 'milvus.yaml').write_text('encoders_config: enc.yaml\nencoder: granite-97m\nsearch: {ef: 64}\n'
        'collections:\n  sent256_s10k: {name: col_b, strategy: B}\n  win256_s10k: {name: col_a, strategy: A}\n')

    class FakeClient:
        def __init__(self, uri=None): pass
        def has_collection(self, name): return True
    def fake_search(client, name, vectors, k, ef, expr='', fields=None):
        ent = lambda d, prov=None: {'document_id': d, 'provenance_ids': prov or [d], 'char_start': 0, 'text': chunk}
        hits = [('c1', .9, ent('a', ['a', 'b'])), ('c2', .8, ent('c'))] if name == 'col_a' else [('c3', .9, ent('d')), ('c4', .8, ent('a'))]
        return [hits]
    monkeypatch.setattr(pool_gold, 'MilvusClient', FakeClient)
    monkeypatch.setattr(pool_gold, 'embed_array', lambda texts, kind, profile: np.zeros((len(texts), 4), dtype='float32'))
    monkeypatch.setattr(pool_gold.indexing, 'search', fake_search)
    run(pool_gold, monkeypatch, '--config', 'milvus.yaml', '--input', str(tmp_path / 'corpus.parquet'), '--sample-size', '1')
    out = json.loads(capsys.readouterr().out)
    assert out['hoja1_preguntas'] == 2 and out['relevantes_conocidos'] == 1 and out['relevantes_conocidos_en_pool'] == 1
    records = {r['id']: r for r in map(json.loads, (tmp_path / 'data/gold/pooling.jsonl').read_text().splitlines())}
    assert {c['document_id'] for c in records['Q1']['candidates']} == {'a', 'b', 'c', 'd'}
    assert 'texto largo' not in (tmp_path / 'data/gold/pooling.jsonl').read_text()
    sheet = pooling.read_sheet(tmp_path / 'data/annotation/hoja_anotador1.xlsx', 'anotacion')
    assert {r['id'] for r in sheet} == {'Q1', 'Q2'} and any('>>>' in (r['pasaje'] or '') for r in sheet)
    assert {r['grado'] for r in sheet if r['id'] == 'Q1' and r['document_id'] == 'a'} == {2}      # juicio previo precargado
    sheet2 = pooling.read_sheet(tmp_path / 'data/annotation/hoja_anotador2.xlsx', 'anotacion')
    assert {r['grado'] for r in sheet2} == {None} and set(sheet2[0]) == set(pooling.SHEET_COLUMNS)  # ciega: sin juicios previos
    assert len(pooling.read_sheet(tmp_path / 'data/annotation/hoja_anotador2.xlsx', 'respuestas')) == 1
    assert (tmp_path / 'reports/gold/tables/pooling_cobertura.csv').is_file()
    run(pool_gold, monkeypatch, '--config', 'milvus.yaml', '--fase2', '--sample-size', '1')
    assert json.loads(capsys.readouterr().out)['questions'] == out['muestra_doble_anotacion']
