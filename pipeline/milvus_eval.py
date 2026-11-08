"""Verificación de las colecciones Milvus (Fase D): calibración de ef, filtros, métricas en dev y persistencia.

Todo se compara contra la búsqueda exacta (numpy) sobre los mismos vectores, para separar lo que pierde el
índice aproximado (HNSW) de lo que depende del encoder o de la segmentación.
"""
import time
import numpy as np
import pandas as pd
from pipeline import indexing, metrics, retrieval

def exact_top(query_vectors, matrix, k):
    idx, _ = retrieval.top_chunks(query_vectors, matrix, k)
    return idx

def calibrate_ef(client, collection, query_vectors, matrix, chunk_ids, ef_candidates, k=10):
    """Recall@k del HNSW frente a la búsqueda exacta y latencia por consulta, para cada ef."""
    exact = [{chunk_ids[i] for i in row} for row in exact_top(query_vectors, matrix, k)]
    rows = []
    for ef in ef_candidates:
        recalls, latencies = [], []
        for vector, truth in zip(query_vectors, exact):
            start = time.perf_counter()
            hits = indexing.search(client, collection, [vector], k, ef)[0]
            latencies.append((time.perf_counter() - start) * 1000)
            recalls.append(len({h[0] for h in hits} & truth) / k)
        rows.append({'ef': ef, 'queries': len(recalls), f'recall@{k}_mean': float(np.mean(recalls)),
                     f'recall@{k}_min': float(np.min(recalls)), 'latency_ms_p50': float(np.percentile(latencies, 50)),
                     'latency_ms_p95': float(np.percentile(latencies, 95))})
    return pd.DataFrame(rows)

def choose_ef(calibration, target, k=10):
    """El menor ef cuyo recall medio alcanza `target`; si ninguno lo alcanza, el de mayor recall."""
    ok = calibration[calibration[f'recall@{k}_mean'] >= target]
    return int(ok['ef'].min()) if len(ok) else int(calibration.loc[calibration[f'recall@{k}_mean'].idxmax(), 'ef'])

def milvus_rankings(client, collection, questions, query_vectors, ef, doc_dates, spans, texts, k_chunks=300):
    """Rankings de documentos y acierto de evidencia usando Milvus (con filtro de fecha si la pregunta lo pide)."""
    rankings, extra = {}, {}
    for q, vector in zip(questions, query_vectors):
        window = q.get('ventana_fecha')
        expr = indexing.window_filter(window) if window else ''
        hits = indexing.search(client, collection, [vector], k_chunks, max(ef, k_chunks), expr,
                               ['document_id', 'char_start', 'text', 'provenance_ids'])[0]
        frame = pd.DataFrame([{**h[2], 'chunk_id': h[0]} for h in hits])
        provenance = frame['provenance_ids'].map(list).tolist() if len(frame) else []
        allowed = retrieval.documents_in_window(doc_dates, window) if window else None
        rankings[q['id']] = retrieval.rank_documents(range(len(frame)), provenance, allowed, limit=20)
        frame = frame.assign(provenance_ids=provenance)
        extra[q['id']] = {f'evidence@{k}': retrieval.evidence_hit(list(range(len(frame))), frame, spans[q['id']], texts, k) for k in (5, 10)}
    return rankings, extra

def filter_checks(client, collection, vector, checks, ef):
    """Comprueba que cada filtro devuelve solo entidades que lo cumplen. `checks`: [(nombre, expresión, función)]."""
    rows = []
    for name, expr, predicate in checks:
        hits = indexing.search(client, collection, [vector], 20, ef, expr, ['language', 'collection_name', 'created_at', 'provenance_ids'])[0]
        ok = all(predicate(h[2]) for h in hits)
        rows.append({'check': name, 'filter': expr, 'results': len(hits), 'all_match': bool(ok and len(hits) > 0)})
    return pd.DataFrame(rows)

def snapshot(client, collections, query_vectors, ef, k=10):
    """Conteos y top-k de unas consultas fijas por colección: sirve para comprobar la persistencia."""
    out = {}
    for key, name in collections.items():
        if not client.has_collection(name): continue
        client.load_collection(name)
        out[key] = {'name': name, 'entities': indexing.count(client, name),
                    'top': [[h[0] for h in hits] for hits in indexing.search(client, name, query_vectors, k, ef)]}
    return out

def compare_snapshots(before, after):
    rows = []
    for key, b in before.items():
        a = after.get(key, {})
        same_top = a.get('top') == b['top'] if a else False
        rows.append({'collection': b['name'], 'entities_before': b['entities'], 'entities_after': a.get('entities'),
                     'same_count': a.get('entities') == b['entities'], 'same_top_results': same_top})
    return pd.DataFrame(rows)
