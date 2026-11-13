"""Indexación en Milvus (Fase D): esquema, conversión de filas, carga por lotes idempotente y búsqueda.

La carga lee los Parquet de chunks (Fase B) y de vectores (Fase C), los une por `chunk_id` y hace
`upsert` por lotes: repetirla no duplica. Un archivo de progreso permite retomar una carga interrumpida,
y cada lote se reintenta con espera exponencial.
"""
import hashlib, json, os, time
from pathlib import Path
import numpy as np
import pandas as pd
from pymilvus import DataType, MilvusClient
from app.filters import window_bounds, window_filter  # una sola implementación para la API y el pipeline
from pipeline.profiling import DATE_ONLY, parse_dates

OUTPUT_FIELDS = ['document_id', 'chunk_index', 'char_start', 'char_end', 'text', 'title', 'url', 'collection_name',
                 'source', 'language', 'created_at', 'provenance_ids']

def connect(uri=None, timeout=10):
    """Abre el cliente de Milvus y comprueba que responde; si no, termina con un mensaje accionable."""
    uri = uri or os.getenv('MILVUS_URI', 'http://milvus:19530')
    try:
        client = MilvusClient(uri=uri, timeout=timeout)
        client.list_collections()
    except Exception as e:
        host = uri.split('://', 1)[-1]
        raise SystemExit(f'Milvus no responde en {host}; ejecuta `docker compose up -d` y espera a que esté healthy '
                         f'(`docker compose ps`). Detalle: {type(e).__name__}') from e
    return client

def build_schema(dim, fields):
    s = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    s.add_field('chunk_id', DataType.VARCHAR, is_primary=True, max_length=64)
    s.add_field('vector', DataType.FLOAT_VECTOR, dim=dim)
    s.add_field('document_id', DataType.VARCHAR, max_length=256)
    s.add_field('chunk_index', DataType.INT32)
    s.add_field('char_start', DataType.INT64)
    s.add_field('char_end', DataType.INT64)
    s.add_field('text', DataType.VARCHAR, max_length=fields['text_max'])
    s.add_field('chunk_sha256', DataType.VARCHAR, max_length=64)
    s.add_field('n_tokens', DataType.INT32)
    s.add_field('strategy', DataType.VARCHAR, max_length=8)
    s.add_field('strategy_version', DataType.INT32)
    s.add_field('collection_name', DataType.VARCHAR, max_length=64)
    s.add_field('source', DataType.VARCHAR, max_length=64)
    s.add_field('url', DataType.VARCHAR, max_length=fields['url_max'])
    s.add_field('title', DataType.VARCHAR, max_length=fields['title_max'])
    s.add_field('language', DataType.VARCHAR, max_length=16)
    s.add_field('language_source', DataType.VARCHAR, max_length=16)
    s.add_field('created_at', DataType.INT64, nullable=True)
    s.add_field('created_at_precision', DataType.VARCHAR, max_length=16)
    s.add_field('created_at_plausible', DataType.BOOL)
    s.add_field('has_mojibake', DataType.BOOL)
    s.add_field('provenance_ids', DataType.ARRAY, element_type=DataType.VARCHAR,
                max_capacity=fields['provenance_capacity'], max_length=256)
    s.add_field('n_provenance', DataType.INT32)
    return s

def index_params(client, cfg):
    params = MilvusClient.prepare_index_params()
    params.add_index(field_name='vector', index_type=cfg['type'], metric_type=cfg['metric'],
                     params={'M': cfg['M'], 'efConstruction': cfg['efConstruction']})
    for field, kind in (cfg.get('scalar') or {}).items():
        params.add_index(field_name=field, index_type=kind)
    return params

def to_epoch(values):
    """Segundos UTC desde 1970 (None si falta o no se interpreta)."""
    return [None if pd.isna(p) else int(p.timestamp()) for p in parse_dates(values)]

def date_precision(values):
    is_date = values.str.fullmatch(DATE_ONLY, na=False)
    return np.where(values.isna(), 'nula', np.where(is_date, 'fecha', 'fecha_hora')).tolist()

def to_rows(chunks, vectors, plausible_start=None):
    """Filas para Milvus a partir del Parquet de chunks y la matriz de vectores alineada."""
    created = to_epoch(chunks['created_at'])
    starts = {k: int(pd.Timestamp(v, tz='UTC').timestamp()) for k, v in (plausible_start or {}).items()}
    declared = chunks['language']
    detected = chunks['language_detected'] if 'language_detected' in chunks else pd.Series([None] * len(chunks))
    language = declared.where(declared.notna(), detected).fillna('und')
    source = np.where(declared.notna(), 'declarado', np.where(detected.notna(), 'detectado', 'ninguno'))
    precision = date_precision(chunks['created_at'])
    rows = []
    for i, r in enumerate(chunks.itertuples(index=False)):
        start = starts.get(r.collection_name)
        rows.append({
            'chunk_id': r.chunk_id, 'vector': vectors[i].tolist(), 'document_id': r.document_id,
            'chunk_index': int(r.chunk_index), 'char_start': int(r.char_start), 'char_end': int(r.char_end),
            'text': r.text, 'chunk_sha256': r.chunk_sha256, 'n_tokens': int(r.n_tokens), 'strategy': r.strategy,
            'strategy_version': int(r.strategy_version), 'collection_name': r.collection_name or '', 'source': r.source or '',
            'url': r.url or '', 'title': r.title or '', 'language': language.iat[i], 'language_source': str(source[i]),
            'created_at': created[i], 'created_at_precision': precision[i],
            'created_at_plausible': bool(created[i] is not None and (start is None or created[i] >= start)),
            'has_mojibake': bool(r.has_mojibake), 'provenance_ids': list(r.provenance_ids),
            'n_provenance': len(r.provenance_ids)})
    return rows

def fingerprint(*paths):
    """Huella de las entradas de una carga (ruta, tamaño y fecha de modificación): si cambian, el progreso se reinicia."""
    digest = hashlib.sha256()
    for p in map(Path, paths): digest.update(f'{p}:{p.stat().st_size}:{p.stat().st_mtime_ns}'.encode())
    return digest.hexdigest()[:16]

def upsert_batches(client, collection, rows, batch_size, checkpoint, retries=5, backoff=2.0, log=print, sleep=time.sleep):
    """Carga `rows` por lotes con reintentos; `checkpoint` (dict con 'path' y 'key') guarda los lotes hechos."""
    path = Path(checkpoint['path'])
    state = json.loads(path.read_text()) if path.is_file() else {}
    if state.get('key') != checkpoint['key']: state = {'key': checkpoint['key'], 'done': []}
    done = set(state['done'])
    total = -(-len(rows) // batch_size)
    retried = 0
    for b in range(total):
        if b in done: continue
        batch = rows[b * batch_size:(b + 1) * batch_size]
        for attempt in range(retries):
            try:
                client.upsert(collection_name=collection, data=batch)
                break
            except Exception as error:
                if attempt == retries - 1: raise
                retried += 1
                log(f'lote {b + 1}: reintento {attempt + 1} tras {type(error).__name__}')
                sleep(backoff * 2 ** attempt)
        done.add(b)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'key': checkpoint['key'], 'done': sorted(done)}))
        if (b + 1) % 20 == 0 or b + 1 == total: log(f'lote {b + 1}/{total}')
    return {'batches': total, 'retries': retried}

def count(client, collection):
    return int(client.query(collection_name=collection, filter='', output_fields=['count(*)'])[0]['count(*)'])

def search(client, collection, vectors, k, ef, filter_expr='', output_fields=None):
    """Búsqueda HNSW; devuelve por consulta la lista de (chunk_id, puntaje, entidad)."""
    result = client.search(collection_name=collection, data=[np.asarray(v, dtype=np.float32).tolist() for v in vectors],
                           limit=k, filter=filter_expr, output_fields=output_fields or [],
                           search_params={'metric_type': 'COSINE', 'params': {'ef': max(ef, k)}})
    return [[(h['chunk_id'], float(h['distance']), dict(h['entity'])) for h in hits] for hits in result]

def ingest(client, name, chunks, vectors, cfg, checkpoint_path, inputs_key, fresh=False, log=print):
    """Crea (si falta) y carga una colección; verifica que el conteo coincida con los chunks."""
    if fresh:
        if client.has_collection(name): client.drop_collection(name)
        Path(checkpoint_path).unlink(missing_ok=True)
    created = not client.has_collection(name)
    if created:
        client.create_collection(collection_name=name, schema=build_schema(vectors.shape[1], cfg['fields']),
                                 index_params=index_params(client, cfg['index']), consistency_level='Strong')
    started = time.perf_counter()
    rows = to_rows(chunks, vectors, cfg['dates']['plausible_start'])
    ing = cfg['ingest']
    stats = upsert_batches(client, name, rows, ing['batch_size'], {'path': checkpoint_path, 'key': f'{name}:{inputs_key}'},
                           ing['retries'], ing['backoff_seconds'], log)
    client.flush(name)
    client.load_collection(name)
    total = count(client, name)
    if total != len(chunks):
        raise ValueError(f'{name}: Milvus tiene {total} entidades y el Parquet {len(chunks)} chunks.')
    return {'collection': name, 'created': created, 'entities': total, 'chunks': len(chunks), 'dimension': int(vectors.shape[1]),
            'seconds': round(time.perf_counter() - started, 1), **stats}

def adversarial_chunks(questions, synthetic_docs, base_chunks, base_vectors, chunk_fn, embed_fn):
    """Chunks y vectores de la colección adversarial: documentos reales de las preguntas y documentos sintéticos.

    `chunk_fn(doc)` devuelve el DataFrame de chunks de un documento sintético (mismo esquema que la Fase B) y
    `embed_fn(textos)` sus vectores. Los documentos reales se toman tal cual de la muestra base.
    """
    real = {r['document_id'] for q in questions for r in q['relevantes'] + q['documentos_contexto']
            if not r['document_id'].startswith('synthetic:')}
    keep = base_chunks['provenance_ids'].map(lambda ids: bool(set(ids) & real)).to_numpy()
    synthetic = pd.concat([chunk_fn(doc) for doc in synthetic_docs], ignore_index=True)
    chunks = pd.concat([base_chunks[keep], synthetic], ignore_index=True)
    vectors = np.vstack([base_vectors[keep], np.asarray(embed_fn(synthetic['text'].tolist()), dtype=np.float32)])
    missing = real - {d for ids in chunks['provenance_ids'] for d in ids}
    return chunks, vectors, sorted(missing)

def chunk_frame(doc, chunks, strategy, version, count_tokens):
    """Chunks de un documento suelto (p. ej. sintético) con el esquema del Parquet de la Fase B."""
    from pipeline.chunking import chunk_id
    from pipeline.dedup import text_sha256
    text, doc_id = doc['text'], doc['document_id']
    pieces = [text[c['char_start']:c['char_end']] for c in chunks]
    return pd.DataFrame([{
        'chunk_id': chunk_id(doc_id, strategy, version, c['char_start'], c['char_end']), 'document_id': doc_id,
        'strategy': strategy, 'strategy_version': version, 'chunk_index': i, 'char_start': c['char_start'],
        'char_end': c['char_end'], 'n_tokens': n, 'text': piece, 'chunk_sha256': sha,
        'collection_name': doc.get('collection_name') or 'synthetic', 'source': 'synthetic', 'url': '',
        'title': doc.get('title') or '', 'language': doc.get('language_base'), 'language_detected': None,
        'created_at': None, 'has_mojibake': False, 'provenance_ids': [doc_id]}
        for i, (c, piece, sha, n) in enumerate(zip(chunks, pieces, text_sha256(pieces), count_tokens(pieces)))])
