import pandas as pd
import pytest
from pipeline import rag_check

ADV = {'name': 'rag_adv', 'questions': ['Q45']}

def question(qid='Q01', tipo='directa', abstain=False, relevant=('d1',)):
    return {'id': qid, 'tipo': tipo, 'debe_abstenerse': abstain, 'ventana_fecha': None,
            'relevantes': [{'document_id': d, 'grado': 2} for d in relevant]}

def result(abstained=False, cited=(1,), evidence=('d9', 'd1'), cold=False):
    ev = [{'document_id': d, 'provenance_ids': [d], 'rank': i + 1, 'in_context': True, 'cited': i + 1 in cited}
          for i, d in enumerate(evidence)]
    return {'evidence': ev, 'abstained': abstained, 'citations': [{'n': n} for n in cited], 'invalid_citations': [],
            'warnings': [] if cited or abstained else ['sin_citas'], 'retrieval': {'collection': 'rag_x', 'k': 2},
            'context': {'n_context': 2, 'omitted': 0, 'estimated_tokens': 100}, 'cold_start': cold, 'prompt': {'id': 'p'},
            'tokens': {'prompt': 90, 'completion': 10, 'per_second': 5.0},
            'timings_ms': {'embed': 5, 'search': 1, 'llm': 1000, 'total': 1010}}

def test_only_dev_and_adversarial_collection(tmp_path):
    with pytest.raises(ValueError): rag_check.load_questions(tmp_path / 'gold.jsonl', 'test')
    assert rag_check.plan({'id': 'Q45', 'ventana_fecha': None}, 'rag_main', ADV) == ('rag_adv', {})
    window = {'desde': '2025-02-25', 'hasta': '2025-02-28'}
    assert rag_check.plan({'id': 'Q51', 'ventana_fecha': window}, 'rag_main', ADV) == ('rag_main', window)

def test_token_factor_takes_worst_model_plus_margin():
    table = pd.DataFrame({'model': ['a', 'b'], 'ratio_max': [1.12, 1.31]})
    assert rag_check.token_factor(table, 0.05) == 1.40
    assert rag_check.token_factor(table.assign(ratio_max=[1.0, 1.0]), 0.0) == 1.0

def test_question_row_and_summary():
    rows = [rag_check.question_row(question(), result(cited=(2,)), 'm'),            # cita el relevante (rank 2)
            rag_check.question_row(question('Q02'), result(cited=(1,)), 'm'),       # cita uno no relevante
            rag_check.question_row(question('Q08', 'sin_respuesta', True, ()), result(True, ()), 'm'),
            rag_check.question_row(question('Q03'), result(cited=(), cold=True), 'm')]
    assert [r['cita_relevante'] for r in rows] == [True, False, None, False]
    assert [r['abstencion_correcta'] for r in rows] == [True, True, True, True]
    table = rag_check.summary(pd.DataFrame(rows)).set_index('tipo')
    assert table.loc['total', 'n'] == 4 and table.loc['total', 'cita_relevante'] == pytest.approx(1 / 3)
    assert table.loc['total', 'sin_citas'] == 0.25 and table.loc['total', 'arranques_en_frio'] == 1
    assert table.loc['sin_respuesta', 'abstencion'] == 1.0
