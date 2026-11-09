from datetime import date
import pytest
from fastapi.testclient import TestClient
from pymilvus import MilvusException
from app import api, core, llm, rag

CHUNKS = [{'chunk_id': f'c{i}', 'document_id': f'd{i}', 'chunk_index': 0, 'char_start': 0, 'char_end': 9, 'text': f'palabra {i}',
           'title': '', 'url': '', 'collection_name': 'bluesky_posts', 'source': 'bluesky', 'language': 'es',
           'created_at': None, 'provenance_ids': [f'd{i}'], 'score': 0.9 - i / 10} for i in (1, 2)]

@pytest.fixture
def client(monkeypatch, tmp_path):
    calls = {}
    def retrieve(query, k=5, filters=None, collection=None):
        calls['retrieve'] = {'query': query, 'k': k, 'filters': filters, 'collection': collection}
        if collection == 'rag_no_existe': raise core.CollectionNotFound('La colección rag_no_existe no existe en Milvus')
        return {'collection': collection or 'rag_x', 'filter': '', 'ef': 64, 'chunks': CHUNKS[:k],
                'timings_ms': {'embed': 1.0, 'search': 1.0}}
    def chat(model, system, user, options):
        if model == 'no-descargado': raise llm.ModelNotFound(f'El modelo {model} no está descargado en Ollama. '
                                                            f'Descárgalo con: docker compose exec ollama ollama pull {model}')
        return {'text': 'Respuesta [1].', 'done_reason': 'stop', 'prompt_tokens': 100, 'completion_tokens': 5,
                'load_ms': 1.0, 'prompt_ms': 1.0, 'generation_ms': 100.0, 'llm_total_ms': 102.0}
    monkeypatch.setattr(core, 'retrieve', retrieve)
    monkeypatch.setattr(llm, 'chat', chat)
    monkeypatch.setattr(llm, 'models', lambda: [{'name': 'qwen2.5:1.5b', 'digest': 'abc', 'size_mb': 986,
                                                 'parameter_size': '1.5B', 'quantization': 'Q4_K_M'}])
    monkeypatch.setattr(rag, 'count_tokens', lambda texts, kind: [t.count('palabra') for t in texts])
    monkeypatch.setenv('LOG_DIR', str(tmp_path))
    c = TestClient(api.app)
    c.calls, c.log_dir = calls, tmp_path
    return c

def test_ask_returns_answer_evidence_and_log(client):
    r = client.post('/ask', json={'query': '¿Qué pasó?', 'model': 'qwen2.5:1.5b', 'k': 2,
                                  'filters': {'language': 'es', 'desde': '2025-01-01', 'hasta': '2025-01-31'}})
    assert r.status_code == 200
    out = r.json()
    assert out['answer'] == 'Respuesta [1].' and out['logged'] and out['citations'][0]['chunk_id'] == 'c1'
    assert [(e['rank'], e['cited'], e['in_context']) for e in out['evidence']] == [(1, True, True), (2, False, True)]
    assert client.calls['retrieve']['filters'] == {'language': 'es', 'desde': date(2025, 1, 1), 'hasta': date(2025, 1, 31)}
    assert {'embed', 'search', 'context', 'llm', 'total'} <= set(out['timings_ms']) and out['options']['seed'] == 42
    logs = list((client.log_dir / 'ask').glob('*.jsonl'))
    assert len(logs) == 1 and '"model_digest": "abc"' in logs[0].read_text(encoding='utf-8')

def test_old_request_body_still_works(client):
    r = client.post('/ask', json={'query': '¿Qué pasó?', 'model': 'qwen2.5:1.5b', 'k': 1, 'temperature': 0.0})
    assert r.status_code == 200 and len(r.json()['evidence']) == 1

@pytest.mark.parametrize('body, status, text', [
    ({'model': 'no-descargado'}, 404, 'ollama pull no-descargado'),
    ({'collection': 'rag_no_existe'}, 404, 'no existe'),
    ({'filters': {'language': 'español'}}, 422, 'language'),
    ({'filters': {'desde': '2025-02-01', 'hasta': '2025-01-01'}}, 422, 'posterior'),
    ({'k': 40}, 422, 'k'),
])
def test_errors_have_specific_status(client, body, status, text):
    r = client.post('/ask', json={'query': '¿Qué pasó?', **body})
    assert r.status_code == status and text in r.text

def test_failed_requests_are_logged(client):
    client.post('/ask', json={'query': '¿Qué pasó?', 'model': 'no-descargado'})
    assert 'ModelNotFound' in next((client.log_dir / 'ask').glob('*.jsonl')).read_text(encoding='utf-8')

def test_only_collections_of_the_api_encoder(monkeypatch):
    class FakeMilvus:
        def list_collections(self): return ['rag_e5small_char1000_v1', 'rag_granite97m_win256_s10k_v1', 'otra']
        def get_collection_stats(self, name): return {'row_count': 7}
        def has_collection(self, name): return True
    monkeypatch.setenv('MILVUS_COLLECTION', 'rag_granite97m_sent256_s10k_v1')
    assert core.collections(FakeMilvus()) == [{'name': 'rag_granite97m_win256_s10k_v1', 'entities': 7}]
    with pytest.raises(core.CollectionNotFound, match='no es del encoder'):
        core.retrieve('¿Qué pasó?', collection='rag_e5small_char1000_v1', c=FakeMilvus())

def test_models_and_degraded_health(client, monkeypatch):
    assert client.get('/models').json()['models'][0]['quantization'] == 'Q4_K_M'
    def down(*_): raise MilvusException(message='sin conexión')
    def ollama_down(): raise llm.LLMError('Ollama no responde')
    monkeypatch.setattr(core, 'client', down)
    monkeypatch.setattr(llm, 'version', ollama_down)
    r = client.get('/health')
    assert r.status_code == 503 and r.json()['status'] == 'degradado'
    assert not r.json()['milvus']['ok'] and not r.json()['ollama']['ok']
