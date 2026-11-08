"""Encoder compartido por `api` y `pipeline`: único punto donde se cargan y aplican los embeddings.

Por defecto los parámetros salen de variables de entorno (.env), que es lo que usa la API. El pipeline
puede pasar un perfil explícito (un dict de configs/encoders.yaml): así la segmentación mide siempre con
el mismo tokenizador y la vectorización usa el encoder elegido. Las consultas y los pasajes de una
colección se codifican con el mismo perfil; cambiar cualquier parámetro exige una colección Milvus nueva.
"""
import os
from functools import lru_cache
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

def _flag(name, default):
    value = os.getenv(name)
    if value is None: return default
    value = value.strip().lower()
    if value in ('1', 'true', 'yes'): return True
    if value in ('0', 'false', 'no'): return False
    raise ValueError(f'{name} debe ser true o false, no {value!r}')

def settings(profile=None):
    """Parámetros efectivos del encoder: los del perfil dado o, si no hay, los de .env."""
    if profile is not None:
        return {'model': profile['model'], 'revision': profile.get('revision'),
                'query_prefix': profile.get('query_prefix') or '', 'passage_prefix': profile.get('passage_prefix') or '',
                'normalize': bool(profile.get('normalize', True)), 'dimension': profile.get('dimension')}
    dimension = os.getenv('EMBEDDING_DIMENSION')
    return {'model': os.getenv('EMBEDDING_MODEL', 'intfloat/multilingual-e5-small'),
            'revision': os.getenv('EMBEDDING_REVISION') or None,
            'query_prefix': os.getenv('EMBEDDING_QUERY_PREFIX', 'query: '),
            'passage_prefix': os.getenv('EMBEDDING_PASSAGE_PREFIX', 'passage: '),
            'normalize': _flag('EMBEDDING_NORMALIZE', True),
            'dimension': int(dimension) if dimension else None}

@lru_cache(maxsize=4)
def _load(model, revision):
    # float32 explícito: con pesos bfloat16 (p. ej. Qwen3) la CPU del contenedor fue 16 veces más lenta.
    return SentenceTransformer(model, revision=revision, model_kwargs={'dtype': torch.float32})

def encoder(profile=None):
    cfg = settings(profile)
    return _load(cfg['model'], cfg['revision'])

def _with_prefix(texts, kind, profile=None):
    cfg = settings(profile)
    prefix = cfg['query_prefix'] if kind == 'query' else cfg['passage_prefix']
    return [prefix + t for t in texts], cfg

def embed_array(texts, kind='passage', profile=None, batch_size=32):
    """Vectores float32 de `texts`. Si el perfil fija `dimension`, se recortan (Matryoshka) y se renormalizan."""
    prefixed, cfg = _with_prefix(texts, kind, profile)
    vectors = encoder(profile).encode(prefixed, batch_size=batch_size, normalize_embeddings=cfg['normalize'],
                                      convert_to_numpy=True)
    vectors = np.asarray(vectors, dtype=np.float32)
    if cfg['dimension']:
        vectors = vectors[:, :cfg['dimension']]
        if cfg['normalize']: vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors

def embed(texts, kind='passage', profile=None):
    return embed_array(texts, kind, profile).tolist()

def count_tokens(texts, kind='passage', profile=None):
    """Tokens que procesa el encoder por texto: prefijo, texto y tokens especiales, sin truncar.

    Es la misma medida para el perfilado del corpus y para los chunks: un valor mayor que
    `max_tokens()` significa que el encoder truncaría el texto en silencio.
    """
    prefixed, _ = _with_prefix(texts, kind, profile)
    encoded = encoder(profile).tokenizer(prefixed, add_special_tokens=True, truncation=False, verbose=False)
    return [len(ids) for ids in encoded['input_ids']]

def token_offsets(texts, profile=None):
    """Por texto, los (inicio, fin) en caracteres de cada token de contenido, sin prefijo ni tokens especiales.

    Sirve para cortar chunks en fronteras de token sobre el texto original; su tamaño final se mide
    con `count_tokens`, que sí incluye prefijo y tokens especiales.
    """
    encoded = encoder(profile).tokenizer(list(texts), add_special_tokens=False, truncation=False, verbose=False,
                                         return_offsets_mapping=True)
    return [[(start, end) for start, end in offsets if end > start] for offsets in encoded['offset_mapping']]

def max_tokens(profile=None):
    """Tope de tokens del encoder; lo que lo supera se trunca al codificar."""
    return encoder(profile).max_seq_length
