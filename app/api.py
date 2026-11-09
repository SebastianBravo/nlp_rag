"""API del asistente: recuperación en Milvus y generación con Ollama (Fases E y F).

Cada consulta a `/ask` queda registrada en `data/logs/ask/<fecha>.jsonl` (ver `app.rag.run_record`).
"""
import os, uuid
from contextlib import asynccontextmanager
from datetime import date
from typing import Literal
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator
from pymilvus import MilvusException
from app import core, llm, rag
from app.embeddings import _load, encoder, settings

@asynccontextmanager
async def lifespan(_):
    # Carga el encoder al arrancar para que la primera consulta no pague esa espera.
    try: encoder()
    except Exception as e: print(f'No se pudo cargar el encoder al iniciar: {e}', flush=True)
    yield

app = FastAPI(title='Asistente RAG · Milvus + LLM locales', lifespan=lifespan)

class Filters(BaseModel):
    language: str | None = Field(None, pattern=r'^[a-z]{2,3}$', description='Idioma declarado o, si falta, detectado')
    collection_name: Literal['bluesky_posts', 'url_crawls'] | None = Field(None, description='Bluesky o prensa digital')
    desde: date | None = Field(None, description='Primer día incluido (UTC)')
    hasta: date | None = Field(None, description='Último día incluido (UTC)')

    @model_validator(mode='after')
    def _range(self):
        if self.desde and self.hasta and self.desde > self.hasta: raise ValueError('`desde` es posterior a `hasta`')
        return self

class RetrieveRequest(BaseModel):
    query: str = Field(min_length=3, max_length=2000)
    k: int = Field(5, ge=1, le=15)
    collection: str | None = Field(None, pattern=r'^rag_[A-Za-z0-9_]+$', description='Por defecto, MILVUS_COLLECTION')
    filters: Filters = Field(default_factory=Filters)

class AskRequest(RetrieveRequest):
    model: str = Field(default_factory=lambda: os.getenv('LLM_MODEL', 'qwen2.5:1.5b'), min_length=1, max_length=100)
    temperature: float = Field(0.0, ge=0, le=1)
    seed: int | None = Field(None, description='Por defecto, LLM_SEED')

def _http_error(error):
    if isinstance(error, core.CollectionNotFound): return HTTPException(404, str(error))
    if isinstance(error, llm.LLMError): return HTTPException(error.status, str(error))
    if isinstance(error, MilvusException): return HTTPException(503, f'Milvus no responde: {error}')
    return None

def _digest(model):
    try: return next((m['digest'] for m in llm.models() if m['name'] == model), None)
    except llm.LLMError: return None

@app.get('/health')
def health():
    """Estado de Milvus (colección por defecto), Ollama y el encoder. Responde 503 si falta alguno."""
    body = {'encoder': {**settings(), 'loaded': _load.cache_info().currsize > 0}}
    try:
        c, name = core.client(), core.default_collection()
        entities = next((x['entities'] for x in core.collections(c) if x['name'] == name), None)
        body['milvus'] = {'ok': bool(entities), 'collection': name, 'entities': entities}
    except Exception as e:
        body['milvus'] = {'ok': False, 'error': str(e)}
    try:
        body['ollama'] = {'ok': True, 'version': llm.version(), 'models': len(llm.models())}
    except llm.LLMError as e:
        body['ollama'] = {'ok': False, 'error': str(e)}
    ok = body['milvus']['ok'] and body['ollama']['ok']
    return JSONResponse({'status': 'ok' if ok else 'degradado', **body}, status_code=200 if ok else 503)

@app.get('/models')
def models():
    """Modelos descargados en Ollama, para el selector de la interfaz."""
    try: return {'models': llm.models(), 'default': os.getenv('LLM_MODEL', 'qwen2.5:1.5b')}
    except llm.LLMError as e: raise HTTPException(e.status, str(e))

@app.get('/collections')
def collections():
    """Colecciones del proyecto en Milvus con su número de entidades."""
    try: return {'collections': core.collections(), 'default': core.default_collection()}
    except MilvusException as e: raise HTTPException(503, f'Milvus no responde: {e}')

@app.post('/retrieve')
def retrieve(req: RetrieveRequest):
    try: return core.retrieve(req.query, req.k, req.filters.model_dump(exclude_none=True), req.collection)
    except Exception as e:
        raise _http_error(e) or e

@app.post('/ask')
def ask(req: AskRequest):
    run_id, request = uuid.uuid4().hex, req.model_dump(mode='json')
    try:
        result = rag.answer(req.query, req.model, req.k, req.filters.model_dump(exclude_none=True), req.collection,
                            req.temperature, req.seed)
    except Exception as e:
        rag.write_log(rag.run_record(run_id, request, error=f'{type(e).__name__}: {e}'))
        raise _http_error(e) or e
    log = rag.write_log(rag.run_record(run_id, request, result, model_digest=_digest(req.model)))
    return {'run_id': run_id, 'logged': log is not None, **result,
            'notice': 'La respuesta requiere contrastar las fuentes originales.'}
