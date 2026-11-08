"""Búsqueda exacta por similitud coseno sobre vectores normalizados (numpy), con resultados a nivel documento.

Sirve para evaluar encoders y estrategias antes de Milvus y con los mismos criterios del gold: cada chunk
cuenta para su documento canónico y para los de `provenance_ids`, y cada documento se queda con su mejor rank.
"""
import numpy as np
import pandas as pd

def top_chunks(query_vectors, matrix, k, mask=None):
    """Índices y puntajes de los `k` chunks más similares por consulta. `mask` excluye chunks (False)."""
    scores = np.asarray(query_vectors, dtype=np.float32) @ matrix.T
    if mask is not None: scores = np.where(mask, scores, -np.inf)
    k = min(k, matrix.shape[0])
    top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    order = np.take_along_axis(scores, top, axis=1).argsort(axis=1)[:, ::-1]
    top = np.take_along_axis(top, order, axis=1)
    return top, np.take_along_axis(scores, top, axis=1)

def rank_documents(chunk_indices, provenance, allowed=None, limit=20):
    """Documentos en orden de aparición (mejor rank) a partir de chunks ordenados; `allowed` filtra documentos."""
    ranked, seen = [], set()
    for i in chunk_indices:
        for doc in provenance[i]:
            if doc in seen or (allowed is not None and doc not in allowed): continue
            seen.add(doc)
            ranked.append(doc)
            if len(ranked) == limit: return ranked
    return ranked

def documents_in_window(created_at, window):
    """IDs cuyo `created_at` cae en la ventana {desde, hasta} (ambos días incluidos)."""
    dates = pd.to_datetime(created_at, utc=True, errors='coerce', format='mixed')
    start = pd.Timestamp(window['desde'], tz='UTC')
    end = pd.Timestamp(window['hasta'], tz='UTC') + pd.DateOffset(days=1)
    return set(dates[(dates >= start) & (dates < end)].index)

def evidence_hit(chunk_indices, chunks, spans, texts, k):
    """1 si alguno de los primeros `k` chunks solapa un intervalo anotado de un documento relevante.

    `spans`: {document_id: [(inicio, fin), ...]}. Para un documento de `provenance_ids` distinto del canónico,
    la posición del chunk se busca en su texto (`texts`), porque los offsets del chunk son del canónico.
    """
    for i in chunk_indices[:k]:
        row = chunks.iloc[i]
        for doc in row['provenance_ids']:
            if doc not in spans: continue
            if doc == row['document_id']: start = row['char_start']
            else:
                start = texts.get(doc, '').find(row['text'])
                if start < 0: continue
            end = start + len(row['text'])
            if any(start < e and s < end for s, e in spans[doc]): return 1.0
    return 0.0
