import csv, json
from collections import Counter
from scripts import judge_prepare as jp

TIPOS = ['adversarial'] * 3 + ['sin_respuesta'] * 3 + ['ambigua'] * 3 + ['comparativa'] * 3 + ['directa'] * 6
MODELS = ['m1', 'm2', 'm3', 'm4']
GOLD = {f'Q{i:02d}': {'id': f'Q{i:02d}', 'tipo': t, 'debe_abstenerse': t == 'sin_respuesta', 'pregunta': f'¿pregunta {i}?',
                      'respuesta_referencia': None if t == 'sin_respuesta' else f'ref {i}', 'criterio': None}
        for i, t in enumerate(TIPOS, 1)}

def rec(qid, model, condition='principal', run=1, split='test', error=None, answer=None):
    return {'key': f'{condition}|{model}|{run}|{qid}', 'condition': condition, 'model': model, 'run': run, 'qid': qid, 'split': split,
            'error': error, 'answer': answer or f'respuesta de {model} a {qid} [1]', 'k': 5, 'count': None, 'filters': {},
            'collection': 'B', 'prompt_id': 'rag_v1', 'retrieved_doc_ids': ['d0', 'd1'], 'context': {'n_context': 2}}

RAW = [rec(q, m) for q in GOLD for m in MODELS]

def test_select_filters_and_keeps_last_valid_line():
    raw = [rec('Q01', 'm1', error='X'), rec('Q01', 'm1'), rec('Q01', 'm1', run=2), rec('Q01', 'm1', condition='k3'),
           rec('Q02', 'm1', split='dev'), rec('Q03', 'm1', error='X')]
    assert [r['key'] for r in jp.select(raw, 'principal', 1, ['test'])] == ['principal|m1|1|Q01']

def test_anonymize_is_reproducible_and_opaque():
    a, b = jp.anonymize(RAW, 7), jp.anonymize(RAW, 7)
    assert a == b and a != jp.anonymize(RAW, 8) and len(set(a.values())) == len(RAW)
    assert all('m' not in rid for rid in a.values())

def test_items_and_render_hide_the_model():
    rids = jp.anonymize(RAW)
    items = jp.build_items(RAW, GOLD, {q: ['<doc n="1">uno</doc>', '<doc n="2">dos</doc>'] for q in GOLD}, rids)
    assert len(items) == len(GOLD) and all(len(i['respuestas']) == 4 for i in items)
    text = jp.render(items[0])
    assert 'tipo: adversarial' in text and '<doc n="1">uno</doc>' in text and text.count('**R0') == 4
    assert 'model' not in json.dumps(items[0])
    one = items[0]['respuestas'][0]['rid']
    assert jp.render(items[0], only={one}).count('**R0') == 1
    assert 'debe abstenerse: sí' in jp.render(next(i for i in items if i['tipo'] == 'sin_respuesta'))

def test_audit_is_stratified_by_type_and_balanced_by_model():
    rids = jp.anonymize(RAW)
    audit = jp.pick_audit(RAW, GOLD, rids)
    assert len(audit) == 12 and len({a['qid'] for a in audit}) == 12
    assert Counter(GOLD[a['qid']]['tipo'] for a in audit) == dict(jp.AUDIT_QUOTA)
    by_rid = {rid: key for key, rid in rids.items()}
    assert set(Counter(by_rid[a['rid']].split('|')[1] for a in audit).values()) == {3}
    assert audit == jp.pick_audit(RAW, GOLD, rids)

def test_recover_contexts_checks_ids_and_respects_count():
    class Rag:
        @staticmethod
        def format_document(n, chunk, prompt): return f'[{n}] {chunk["text"]}'
    class Deps:
        rag = Rag()
        def retrieve(self, query, k, filters, collection):
            return {'chunks': [{'document_id': f'd{i}', 'text': f't{i}'} for i in range(2)]}
        def load_prompt(self, pid): return {'id': pid}
    rank = lambda chunks: [c['document_id'] for c in chunks]
    ctx = jp.recover_contexts([rec('Q01', 'm1'), rec('Q01', 'm2')], GOLD, Deps(), rank)
    assert ctx == {'Q01': ['[1] t0', '[2] t1']}
    bad = {**rec('Q02', 'm1'), 'retrieved_doc_ids': ['otro']}
    try: jp.recover_contexts([bad], GOLD, Deps(), rank)
    except SystemExit as e: assert 'Q02' in str(e)
    else: raise AssertionError('debía fallar')

def test_write_outputs(tmp_path):
    rids = jp.anonymize(RAW)
    items = jp.build_items(RAW, GOLD, {q: ['[1] pasaje'] for q in GOLD}, rids)
    audit = jp.pick_audit(RAW, GOLD, rids)
    n = jp.write_outputs(tmp_path, items, rids, RAW, audit, per_part=8)
    assert n == 3 and len(list(tmp_path.glob('lote_parte_*.md'))) == 3
    part = (tmp_path / 'lote_parte_01.md').read_text(encoding='utf-8')
    assert part.startswith('# Instrucciones del juez') and part.count('## Pregunta') == 8 and 'm1' not in part.split('---', 1)[1].replace('respuesta de m', '')
    key = [json.loads(l) for l in (tmp_path / 'clave.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(key) == len(RAW) and {'rid', 'key', 'model', 'qid'} == set(key[0])
    rows = list(csv.DictReader((tmp_path / 'auditoria_humana.csv').open(encoding='utf-8')))
    assert len(rows) == 12 and rows[0]['correccion'] == ''
    assert (tmp_path / 'auditoria_humana.md').read_text(encoding='utf-8').count('**R0') == 12
