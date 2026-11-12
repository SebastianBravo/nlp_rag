import json
import numpy as np
import pandas as pd
from scripts import analyze_plataforma as ap

GOLD = [
    {'id': 'Q1', 'colecciones': ['url_crawls']},
    {'id': 'Q2', 'colecciones': ['bluesky_posts']},
    {'id': 'Q3', 'colecciones': ['bluesky_posts', 'url_crawls']},
    {'id': 'Q4', 'colecciones': ['synthetic', 'url_crawls']},
    {'id': 'Q5', 'colecciones': ['url_crawls']}]
GROUPS = {'Q1': 'prensa', 'Q2': 'bluesky', 'Q3': 'mixta', 'Q4': 'sintetica_prensa', 'Q5': 'prensa'}

def test_group_of():
    assert [ap.group_of(q['colecciones']) for q in GOLD] == ['prensa', 'bluesky', 'mixta', 'sintetica_prensa', 'prensa']
    assert ap.group_of(['otra']) == 'otro'

def test_load_groups(tmp_path):
    path = tmp_path / 'gold.jsonl'
    path.write_text('\n'.join(json.dumps(q) for q in GOLD) + '\n', encoding='utf-8')
    assert ap.load_groups(path) == GROUPS

def test_retrieval_by_group():
    rows = [{'id': q, 'indice': i, 'split': 'test', 'recall@5': r, 'hit@5': r, 'mrr': r, 'ndcg@10': r}
            for i in 'AB' for q, r in (('Q1', 1.0), ('Q2', 0.0), ('Q3', 0.5), ('Q4', np.nan), ('Q5', 0.0))]
    out = ap.retrieval_by_group(pd.DataFrame(rows), GROUPS, B=50)
    row = out[(out['indice'] == 'A') & (out['grupo'] == 'prensa') & (out['split'] == 'test') & (out['metric'] == 'mrr')].iloc[0]
    assert row['mean'] == 0.5 and row['n'] == 2
    assert set(out['split']) == {'test', 'todas'}
    nan_row = out[(out['grupo'] == 'sintetica_prensa') & (out['metric'] == 'mrr')].iloc[0]
    assert nan_row['n'] == 0

def cells():
    rows = []
    for model, offset in (('mA', 0.0), ('mB', 1.0)):
        for run in (1, 2):
            for q in GROUPS:
                rows.append({'condition': 'principal', 'model': model, 'run': run, 'qid': q, 'split': 'test', 'error': False,
                             'abstencion_correcta': 1.0 - offset * (q == 'Q2'), 'cita_relevante': np.nan})
    rows.append({'condition': 'k3', 'model': 'mA', 'run': 1, 'qid': 'Q1', 'split': 'test', 'error': False,
                 'abstencion_correcta': 0.0, 'cita_relevante': np.nan})
    rows.append({'condition': 'principal', 'model': 'mA', 'run': 3, 'qid': 'Q1', 'split': 'test', 'error': True})
    return pd.DataFrame(rows)

def test_generation_by_group_pools_models_and_ignores_errors_and_other_conditions():
    out = ap.generation_by_group(cells(), GROUPS, B=50)
    get = lambda model, grupo: out[(out['model'] == model) & (out['grupo'] == grupo) & (out['split'] == 'test') &
                                   (out['metric'] == 'abstencion_correcta')].iloc[0]
    assert get('mA', 'prensa')['mean'] == 1.0
    assert get('mB', 'bluesky')['mean'] == 0.0
    assert get('todos', 'bluesky')['mean'] == 0.5
    assert get('mA', 'prensa')['n'] == 2

def test_judge_by_group():
    rows = [{'modelo': m, 'qid': q, 'split': 'test', 'prom_correccion': c, 'prom_corr_ge1': 1.0, 'prom_fidelidad': np.nan, 'prom_citas': 1.0}
            for m, c in (('mA', 2.0), ('mB', 0.0)) for q in GROUPS]
    out = ap.judge_by_group(pd.DataFrame(rows), GROUPS, B=50)
    todos = out[(out['model'] == 'todos') & (out['grupo'] == 'mixta') & (out['metric'] == 'prom_correccion')].iloc[0]
    assert todos['mean'] == 1.0 and todos['n'] == 1
    assert out[(out['metric'] == 'prom_fidelidad')]['n'].eq(0).all()

def test_main_writes_tables_and_skips_missing(tmp_path, capsys):
    gold = tmp_path / 'gold.jsonl'
    gold.write_text('\n'.join(json.dumps(q) for q in GOLD) + '\n', encoding='utf-8')
    tables = tmp_path / 'rep' / 'tables'
    tables.mkdir(parents=True)
    pd.DataFrame([{'id': q, 'indice': 'A', 'split': 'test', 'recall@5': 1.0, 'hit@5': 1.0, 'mrr': 1.0, 'ndcg@10': 1.0}
                  for q in GROUPS]).to_csv(tables / 'recuperacion_por_pregunta.csv', index=False)
    cells().assign(split='test').to_csv(tables / 'generacion_celdas.csv', index=False)
    ap.main(['--gold', str(gold), '--reports', str(tmp_path / 'rep'), '--bootstrap', '50'])
    assert (tables / 'plataforma_recuperacion.csv').is_file()
    assert (tables / 'plataforma_generacion.csv').is_file()
    assert not (tables / 'plataforma_juez.csv').exists()
    assert 'Falta' in capsys.readouterr().out
