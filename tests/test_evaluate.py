import json
import pandas as pd
import pytest
from pipeline import metrics
from scripts import evaluate

def question(qid, split='test', tipo='directa', abstenerse=False, relevantes=None):
    return {'id': qid, 'split': split, 'tipo': tipo, 'pregunta': f'pregunta {qid}', 'debe_abstenerse': abstenerse,
            'relevantes': relevantes if relevantes is not None else [{'document_id': 'd0', 'grado': 2}, {'document_id': 'd1', 'grado': 1}],
            'idioma_pregunta': 'es', 'idioma_evidencia': 'es', 'colecciones': ['url_crawls'], 'ventana_fecha': None}

class FakeDeps:
    """Misma interfaz que RealDeps, sin Milvus ni Ollama."""
    def __init__(self, fail_once=None):
        self.retrieve_calls, self.fail_once = [], fail_once
    def retrieve(self, query, k, filters, collection):
        self.retrieve_calls.append((query, k, collection))
        chunks = [{'chunk_id': f'c{i}', 'document_id': f'd{i}', 'provenance_ids': [f'd{i}', f'dup{i}'], 'collection_name': 'url_crawls'}
                  for i in range(k)]
        chunks.append({'chunk_id': 'syn', 'document_id': 'syn1', 'collection_name': 'synthetic'})
        return {'collection': collection, 'filter': '', 'ef': 64, 'chunks': chunks, 'timings_ms': {'embed': 1.0, 'search': 2.0}}
    def plan(self, q, default_collection):
        return ('adv' if q['id'] == 'Q45' else default_collection), {}
    def generate(self, question, chunks, model, prompt, budget, temperature, seed, count):
        if self.fail_once == (model, question):
            self.fail_once = None
            raise RuntimeError('Ollama no responde')
        assert count is None, 'en rag.generate, count es una función contadora de tokens, no un número de fragmentos'
        n = len(chunks)
        return {'answer': 'respuesta [1]', 'model': model, 'abstained': False, 'warnings': [], 'citations': [],
                'invalid_citations': [], 'context': {'n_context': n}, 'tokens': {'prompt': 10, 'completion': 5, 'per_second': 50.0},
                'timings_ms': {'llm': 100.0, 'llm_generation': 80.0}, 'cold_start': False,
                'prompt': {'id': prompt['id']}, 'options': {'seed': seed}}
    def load_prompt(self, prompt_id): return {'id': prompt_id}
    def budget(self): return {}
    def answerable(self, q): return metrics.answerable(q)
    def evaluate(self, rankings, questions, ks, extra): return metrics.evaluate(rankings, questions, ks, extra)
    def digests(self): return {'m1': 'abc', 'm2': 'def'}
    def loaded(self, model): return {'size_mb': 1000, 'vram_mb': 900}
    def versions(self): return {'ollama': '0.0'}

CFG = {'seed': 42, 'base': {'collection': 'B', 'k': 5, 'prompt': 'rag_v1', 'count': None},
       'conditions': {'principal': {}, 'k3': {'k': 3}, 'top3': {'count': 3}},
       'retrieval': {'k': 20, 'collections': {'B': 'B', 'A': 'A'}}}

class NoSampler:
    def __enter__(self): return self
    def __exit__(self, *exc): return False
    def summary(self): return {'samples': 0}

def test_rank_docs_uses_provenance_and_skips_synthetic():
    chunks = [{'document_id': 'a', 'provenance_ids': ['a', 'b']}, {'document_id': 'c', 'collection_name': 'synthetic'},
              {'document_id': 'b'}, {'document_id': 'd'}]
    assert evaluate.rank_docs(chunks) == ['a', 'b', 'c', 'd']
    assert evaluate.rank_docs(chunks, skip_synthetic=True) == ['a', 'b', 'd']

def test_generation_writes_one_line_per_cell_and_shares_retrieval(tmp_path):
    deps, out = FakeDeps(), tmp_path / 'g.jsonl'
    qs = [question('Q1'), question('Q45'), question('Q3', abstenerse=True, relevantes=[])]
    written = evaluate.run_generation(CFG, deps, qs, ['principal', 'k3', 'top3'], ['m1', 'm2'], 2, out, NoSampler, log=lambda *_: None)
    rows = evaluate.read_jsonl(out)
    assert written == len(rows) == 2 * 3 * 2 * 3 and len({r['key'] for r in rows}) == len(rows)
    # se recupera una vez por (colección, k, pregunta): principal y top3 comparten k=5; k3 es distinto
    assert len(deps.retrieve_calls) == 2 * 3
    q45 = [r for r in rows if r['qid'] == 'Q45']
    assert {r['collection'] for r in q45} == {'adv'}
    top3 = next(r for r in rows if r['condition'] == 'top3')
    assert len(top3['context_doc_ids']) == 3 + 3 and top3['count'] == 3   # 3 chunks con sus procedencias duplicadas
    assert all(r['error'] is None and r['model_digest'] in ('abc', 'def') for r in rows)
    assert all(r['model_memory'] == {'size_mb': 1000, 'vram_mb': 900} for r in rows)
    assert any(r['retrieval_cached'] for r in rows) and not rows[0]['retrieval_cached']

def test_generation_resumes_and_retries_errors(tmp_path):
    out = tmp_path / 'g.jsonl'
    qs = [question('Q1'), question('Q2')]
    args = (CFG, FakeDeps(fail_once=('m1', 'pregunta Q2')), qs, ['principal'], ['m1'], 1, out, NoSampler)
    evaluate.run_generation(*args, log=lambda *_: None)
    rows = evaluate.read_jsonl(out)
    assert [r['error'] is None for r in rows] == [True, False] and 'Ollama no responde' in rows[1]['error']
    again = evaluate.run_generation(CFG, FakeDeps(), qs, ['principal'], ['m1'], 1, out, NoSampler, log=lambda *_: None)
    rows = evaluate.read_jsonl(out)
    assert again == 1 and len(rows) == 3 and rows[2]['error'] is None and rows[2]['qid'] == 'Q2'
    assert evaluate.run_generation(CFG, FakeDeps(), qs, ['principal'], ['m1'], 1, out, NoSampler, log=lambda *_: None) == 0

def test_retrieval_writes_metrics_latency_and_id_only_rankings(tmp_path):
    qs = [question('Q1'), question('Q2', split='dev'), question('Q3', abstenerse=True, relevantes=[])]
    table = evaluate.run_retrieval(CFG, FakeDeps(), qs, 2, tmp_path, log=lambda *_: None)
    assert sorted(table['indice'].unique()) == ['A', 'B'] and len(table) == 4          # Q3 no es respondible
    assert table['recall@5'].eq(1.0).all()                                             # d0 (g2) y d1 (g1) entran en los 5 primeros
    assert table['recall@1'].eq(0.5).all() and table['recall@1_g2'].eq(1.0).all()      # en el 1.º solo d0: laxo 1/2, estricto 1/1
    latency = pd.read_csv(tmp_path / 'tables' / 'recuperacion_latencia.csv')
    assert len(latency) == 2 * 2 * 2 and {'embed_ms', 'search_ms'} <= set(latency.columns)
    ranks = evaluate.read_jsonl(tmp_path / 'runs' / 'recuperacion_rankings.jsonl')
    assert len(ranks) == 4 and all('syn1' not in r['ranking'] for r in ranks)          # sintéticos fuera de las métricas
    assert all(set(r) == {'indice', 'id', 'ranking'} for r in ranks)

def test_sampler_summary_has_keys_when_psutil_is_available():
    sampler = evaluate.ResourceSampler(interval=0.05)
    with sampler:
        sum(range(100000))
    summary = sampler.summary()
    assert summary is None or {'cpu_pct_mean', 'ollama_rss_mb_peak', 'system_used_mb_peak'} <= set(summary)

def test_load_questions_filters_split_and_limit(tmp_path):
    path = tmp_path / 'gold.jsonl'
    path.write_text('\n'.join(json.dumps(question(f'Q{i}', split='test' if i % 2 else 'dev')) for i in range(6)), encoding='utf-8')
    assert [q['id'] for q in evaluate.load_questions(path, ['test'])] == ['Q1', 'Q3', 'Q5']
    assert len(evaluate.load_questions(path, None, 2)) == 2

def test_generation_stops_after_consecutive_errors_and_resumes(tmp_path):
    class Down(FakeDeps):
        def generate(self, *args, **kwargs): raise ConnectionError('Ollama caído')
    out = tmp_path / 'g.jsonl'
    qs = [question(f'Q{i}') for i in range(1, 8)]
    with pytest.raises(RuntimeError, match='3 errores seguidos'):
        evaluate.run_generation(CFG, Down(), qs, ['principal'], ['m1'], 1, out, NoSampler, log=lambda *_: None, max_errors=3)
    assert len(evaluate.read_jsonl(out)) == 3
    evaluate.run_generation(CFG, FakeDeps(), qs, ['principal'], ['m1'], 1, out, NoSampler, log=lambda *_: None)
    ok = {r['qid'] for r in evaluate.read_jsonl(out) if not r['error']}
    assert ok == {q['id'] for q in qs}
