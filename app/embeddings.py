"""Encoder compartido por `api` y `pipeline`: único punto donde se cargan y aplican los embeddings.

Modelo, prefijos y normalización salen de variables de entorno (.env) para que consultas y pasajes
se codifiquen siempre igual. Los valores por defecto corresponden a E5 (`query: ` / `passage: `).
Cambiar cualquiera de ellos cambia los vectores: exige una colección Milvus nueva.
"""
import os
from functools import lru_cache
from sentence_transformers import SentenceTransformer

def _flag(name, default):
    value = os.getenv(name)
    if value is None: return default
    value = value.strip().lower()
    if value in ('1', 'true', 'yes'): return True
    if value in ('0', 'false', 'no'): return False
    raise ValueError(f'{name} debe ser true o false, no {value!r}')

def settings():
    """Parámetros efectivos del encoder; sirven también para registrar cada ejecución."""
    return {'model': os.getenv('EMBEDDING_MODEL', 'intfloat/multilingual-e5-small'),
            'query_prefix': os.getenv('EMBEDDING_QUERY_PREFIX', 'query: '),
            'passage_prefix': os.getenv('EMBEDDING_PASSAGE_PREFIX', 'passage: '),
            'normalize': _flag('EMBEDDING_NORMALIZE', True)}

@lru_cache(maxsize=1)
def encoder():
    return SentenceTransformer(settings()['model'])

def embed(texts, kind='passage'):
    cfg = settings()
    prefix = cfg['query_prefix'] if kind == 'query' else cfg['passage_prefix']
    return encoder().encode([prefix + t for t in texts], normalize_embeddings=cfg['normalize']).tolist()
