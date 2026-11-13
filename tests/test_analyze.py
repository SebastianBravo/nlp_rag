import json
import numpy as np
import pandas as pd
import pytest
from scripts import analyze

def gold_q(qid, relevantes, abst=False, split='test'):
    return {'id': qid, 'split': split, 'tipo': 'directa', 'debe_abstenerse': abst, 'relevantes': relevantes}

def rec(key, qid, model='m1', run=1, abstained=False, cited=('d0',), ctx=('d0', 'd1'), warnings=(), error=None, cold=False, split='test'):
    return {'key': key, 'condition': 'principal', 'model': model, 'run': run, 'qid': qid, 'split': split, 'tipo': 'directa',
            'debe_abstenerse': False, 'k': 5, 'prompt_id': 'rag_v1', 'count': None, 'error': error, 'abstained': abstained,
            'citations': [{'n': i + 1, 'document_id': d} for i, d in enumerate(cited)], 'invalid_citations': [],
            'context_doc_ids': list(ctx), 'warnings': list(warnings), 'cold_start': cold,
            'tokens': {'prompt': 100, 'completion': 20, 'per_second': 50.0},
            'timings_ms': {'embed': 10, 'search': 5, 'context': 1, 'llm': 400, 'llm_generation': 300},
            'resources': {'cpu_pct_mean': 20.0, 'system_used_mb_peak': 9000}, 'model_memory': {'vram_mb': 1448}}

REL = {'Q1': {'g1': {'d0', 'd1'}, 'g2': {'d0'}, 'abst': False}, 'Q2': {'g1': {'d9'}, 'g2': set(), 'abst': True}}

def test_mean_ci_basic_and_nan():
    mean, lo, hi, n = analyze.mean_ci([1, 1, 0, 0, np.nan], B=2000)
    assert mean == 0.5 and n == 4 and lo < 0.5 < hi
    assert analyze.mean_ci([np.nan])[3] == 0
    assert analyze.mean_ci([1, 1, 1], B=500) == (1.0, 1.0, 1.0, 3)

def test_paired_diff_uses_complete_pairs_and_detects_zero():
    mean, lo, hi, n = analyze.paired_diff_ci([1, 0, 1, np.nan, 0], [0, 0, 1, 1, 0], B=2000)
    assert n == 4 and mean == 0.25
    mean, lo, hi, n = analyze.paired_diff_ci([1] * 30, [0] * 30, B=500)
    assert lo > 0

def test_cell_row_quality_metrics():
    ok = analyze.cell_row(rec('a', 'Q1'), REL)
    assert ok['abstencion_correcta'] == 1 and ok['cita_relevante'] == 1 and ok['precision_citas'] == 1
    assert ok['contexto_con_relevante'] == 1 and ok['contexto_con_relevante_g2'] == 1 and ok['total_ms'] == 416
    assert np.isnan(ok['respondio_sin_evidencia'])                       # Q1 es respondible
    bad = analyze.cell_row(rec('b', 'Q1', abstained=True, cited=(), warnings=['sin_citas']), REL)
    assert bad['abstencion_indebida'] == 1 and bad['abstencion_correcta'] == 0 and np.isnan(bad['cita_relevante'])
    hallu = analyze.cell_row(rec('c', 'Q2', abstained=False, cited=('d9',)), REL)
    assert hallu['respondio_sin_evidencia'] == 1 and hallu['abstencion_correcta'] == 0 and np.isnan(hallu['cita_relevante'])
    mixed = analyze.cell_row(rec('d', 'Q1', cited=('d0', 'x')), REL)
    assert mixed['precision_citas'] == 0.5
    err = analyze.cell_row(rec('e', 'Q1', error='ConnectionError: x'), REL)
    assert err['error'] is True and 'abstencion_correcta' not in err

def test_build_cells_keeps_last_record_per_key():
    raw = [rec('k1', 'Q1', error='Timeout'), rec('k1', 'Q1'), rec('k2', 'Q1', model='m2')]
    cells = analyze.build_cells(raw, REL)
    assert len(cells) == 2 and not cells['error'].any()

def test_generation_and_latency_summaries():
    raw = [rec(f'{m}{r}{q}', q, model=m, run=r, cold=(r == 1 and q == 'Q1'))
           for m in ('m1', 'm2') for r in (1, 2, 3) for q in ('Q1', 'Q2')]
    raw.append(rec('boom', 'Q1', model='m1', error='X'))
    cells = analyze.build_cells(raw, REL)
    summary = analyze.generation_summary(cells, B=300)
    assert set(summary['split']) == {'test', 'todas'} and {'m1', 'm2'} == set(summary['model'])
    row = summary[(summary['model'] == 'm1') & (summary['metric'] == 'abstencion_correcta') & (summary['split'] == 'todas')].iloc[0]
    assert row['n'] == 2 and row['mean'] == 0.5          # Q1 bien, Q2 (debía abstenerse) respondida
    lat = analyze.latency_summary(cells)
    assert set(lat['arranque']) == {'frio', 'caliente'} and (lat['total_ms_p50'] == 416).all()
    assert lat[lat['model'] == 'm1']['errores'].iloc[0] == 1 and lat[lat['model'] == 'm2']['tasa_errores'].iloc[0] == 0

def test_retrieval_summary_pairs_questions_across_indexes():
    rows = []
    for qid, a, b in (('Q1', 1.0, 0.0), ('Q2', 1.0, 1.0), ('Q3', 0.5, 0.5), ('Q4', 1.0, 0.0)):
        for indice, v in (('A', a), ('B', b)):
            rows.append({'id': qid, 'indice': indice, 'split': 'test', 'tipo': 'directa', 'recall@5': v, 'mrr': v, 'mrr_g2': v})
    summary, by_type, diff = analyze.retrieval_summary(pd.DataFrame(rows), B=500)
    d = diff[diff['metric'] == 'mrr'].iloc[0]
    assert d['diff_A_menos_B'] == pytest.approx(0.5) and d['n'] == 4
    assert set(summary['split']) == {'test', 'todas'} and set(by_type['indice']) == {'A', 'B'}

def test_main_writes_tables(tmp_path):
    gold = tmp_path / 'gold.jsonl'
    gold.write_text('\n'.join(json.dumps(gold_q(q, [{'document_id': d, 'grado': g} for d, g in rel.items()], abst))
                              for q, rel, abst in (('Q1', {'d0': 2, 'd1': 1}, False), ('Q2', {'d9': 1}, True))), encoding='utf-8')
    raw = tmp_path / 'g.jsonl'
    raw.write_text('\n'.join(json.dumps(rec(f'{q}{r}', q, run=r)) for q in ('Q1', 'Q2') for r in (1, 2)), encoding='utf-8')
    analyze.main(['--gold', str(gold), '--raw', str(raw), '--reports', str(tmp_path / 'rep'), '--bootstrap', '200'])
    names = {p.name for p in (tmp_path / 'rep' / 'tables').iterdir()}
    assert {'generacion_celdas.csv', 'generacion_resumen.csv', 'generacion_latencia.csv'} <= names
    cells = pd.read_csv(tmp_path / 'rep' / 'tables' / 'generacion_celdas.csv')
    assert 'answer' not in cells.columns and len(cells) == 4
