"""Recuperación en Milvus para la API. Las colecciones las crea y carga `scripts.ingest` (pipeline)."""
import os
from pymilvus import MilvusClient
from app.embeddings import embed

COLLECTION = os.getenv('MILVUS_COLLECTION', 'rag_granite97m_sent256_s10k_v1')
SEARCH_EF = int(os.getenv('MILVUS_SEARCH_EF', '64'))
OUTPUT_FIELDS = ['document_id', 'chunk_index', 'char_start', 'char_end', 'text', 'title', 'url', 'collection_name',
                 'source', 'language', 'created_at', 'provenance_ids']

def client():
    return MilvusClient(uri=os.getenv('MILVUS_URI', 'http://milvus:19530'))

def retrieve(query, k=5):
    c = client()
    if not c.has_collection(COLLECTION):
        return []
    rows = c.search(collection_name=COLLECTION, data=embed([query], kind='query'), limit=k, output_fields=OUTPUT_FIELDS,
                    search_params={'metric_type': 'COSINE', 'params': {'ef': max(SEARCH_EF, k)}})[0]
    return [{'chunk_id': x['chunk_id'], 'score': x['distance'],
             **{key: (list(value) if key == 'provenance_ids' else value) for key, value in x['entity'].items()}} for x in rows]
