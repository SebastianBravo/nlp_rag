"""Recuperación en Milvus para la API. Las colecciones las crea y carga `scripts.ingest` (pipeline)."""
import os, time
from pymilvus import MilvusClient
from app.embeddings import embed
from app.filters import build_filter

SEARCH_EF = int(os.getenv('MILVUS_SEARCH_EF', '64'))
OUTPUT_FIELDS = ['document_id', 'chunk_index', 'char_start', 'char_end', 'text', 'title', 'url', 'collection_name',
                 'source', 'language', 'created_at', 'provenance_ids']
class CollectionNotFound(Exception):
    pass

def default_collection():
    return os.getenv('MILVUS_COLLECTION', 'rag_granite97m_sent256_s10k_v1')

def encoder_prefix():
    """`rag_<encoder>_` de la colección por defecto: solo se busca en colecciones del mismo encoder que la API."""
    return '_'.join(default_collection().split('_')[:2]) + '_'

def client():
    return MilvusClient(uri=os.getenv('MILVUS_URI', 'http://milvus:19530'), timeout=10)

def collections(c=None):
    """Colecciones del encoder de la API con su número de entidades."""
    c = c or client()
    return [{'name': n, 'entities': int(c.get_collection_stats(n).get('row_count', 0))}
            for n in sorted(c.list_collections()) if n.startswith(encoder_prefix())]

def retrieve(query, k=5, filters=None, collection=None, c=None):
    """Los `k` chunks más cercanos a la consulta, con sus metadatos y los tiempos de cada paso (ms)."""
    c = c or client()
    name = collection or default_collection()
    if not name.startswith(encoder_prefix()):
        raise CollectionNotFound(f'La colección {name} no es del encoder de la API ({encoder_prefix()}*): '
                                 'sus vectores no son comparables con los de la consulta')
    if not c.has_collection(name): raise CollectionNotFound(f'La colección {name} no existe en Milvus')
    expr = build_filter(**(filters or {}))
    t0 = time.perf_counter()
    vector = embed([query], kind='query')
    t1 = time.perf_counter()
    rows = c.search(collection_name=name, data=vector, limit=k, filter=expr, output_fields=OUTPUT_FIELDS,
                    search_params={'metric_type': 'COSINE', 'params': {'ef': max(SEARCH_EF, k)}})[0]
    t2 = time.perf_counter()
    chunks = [{'chunk_id': x['chunk_id'], 'score': round(float(x['distance']), 4),
               **{key: (list(value) if key == 'provenance_ids' else value) for key, value in x['entity'].items()}}
              for x in rows]
    return {'collection': name, 'filter': expr, 'ef': max(SEARCH_EF, k), 'chunks': chunks,
            'timings_ms': {'embed': round((t1 - t0) * 1e3, 1), 'search': round((t2 - t1) * 1e3, 1)}}
