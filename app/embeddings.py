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

def _with_prefix(texts, kind):
    cfg = settings()
    prefix = cfg['query_prefix'] if kind == 'query' else cfg['passage_prefix']
    return [prefix + t for t in texts], cfg

def embed(texts, kind='passage'):
    prefixed, cfg = _with_prefix(texts, kind)
    return encoder().encode(prefixed, normalize_embeddings=cfg['normalize']).tolist()

def count_tokens(texts, kind='passage'):
    """Tokens que procesa el encoder por texto: prefijo, texto y tokens especiales, sin truncar.

    Es la misma medida para el perfilado del corpus y para los chunks: un valor mayor que
    `max_tokens()` significa que el encoder truncaría el texto en silencio.
    """
    prefixed, _ = _with_prefix(texts, kind)
    encoded = encoder().tokenizer(prefixed, add_special_tokens=True, truncation=False, verbose=False)
    return [len(ids) for ids in encoded['input_ids']]

def max_tokens():
    """Tope de tokens del encoder; lo que lo supera se trunca al codificar."""
    return encoder().max_seq_length
