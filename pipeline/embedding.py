"""Vectorización de chunks por partes, con reanudación, y lectura y escritura de vectores en Parquet.

Los vectores se guardan como lista de tamaño fijo de float32 junto a `chunk_id`, en el mismo orden que el
Parquet de chunks. Si la ejecución se interrumpe, las partes ya escritas se reutilizan al relanzarla.
"""
import shutil
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

def vectors_table(chunk_ids, vectors, n_tokens):
    vectors = np.asarray(vectors, dtype=np.float32)
    flat = pa.array(vectors.reshape(-1), type=pa.float32())
    return pa.table({'chunk_id': pa.array(list(chunk_ids), type=pa.string()),
                     'vector': pa.FixedSizeListArray.from_arrays(flat, vectors.shape[1]),
                     'n_tokens': pa.array(list(n_tokens), type=pa.int32())})

def read_vectors(path):
    """(chunk_ids, matriz float32, n_tokens) de un Parquet de vectores."""
    table = pq.read_table(path)
    column = table.column('vector').combine_chunks()
    matrix = column.values.to_numpy(zero_copy_only=False).reshape(len(table), column.type.list_size)
    return table.column('chunk_id').to_pylist(), matrix.astype(np.float32, copy=False), table.column('n_tokens').to_numpy()

def embed_in_parts(chunk_ids, texts, embed_fn, count_fn, output, part_size=2048, log=print):
    """Vectoriza `texts` en partes de `part_size`; escribe `output` y devuelve las partes calculadas en esta ejecución."""
    output = Path(output)
    parts_dir = output.with_name(output.name + '.parts')
    parts_dir.mkdir(parents=True, exist_ok=True)
    computed = 0
    starts = range(0, len(texts), part_size)
    for number, start in enumerate(starts):
        part = parts_dir / f'part-{number:05d}.parquet'
        if part.exists(): continue
        chunk = texts[start:start + part_size]
        pq.write_table(vectors_table(chunk_ids[start:start + part_size], embed_fn(chunk), count_fn(chunk)), part)
        computed += 1
        log(f'parte {number + 1}/{len(starts)}')
    parts = sorted(parts_dir.glob('part-*.parquet'))
    table = pa.concat_tables([pq.read_table(p) for p in parts])
    if table.column('chunk_id').to_pylist() != list(chunk_ids):
        raise ValueError('Las partes no coinciden con los chunks: borra la carpeta .parts y vuelve a ejecutar.')
    pq.write_table(table, output)
    shutil.rmtree(parts_dir)
    return computed

def measure_cost(texts, embed_fn, warmup=8):
    """Chunks por segundo y RAM pico del proceso al vectorizar `texts` (tras calentar con `warmup` textos).

    Conviene medir cada encoder en un proceso propio: la RAM pico es del proceso completo.
    """
    import resource, time
    embed_fn(texts[:warmup])
    start = time.perf_counter()
    vectors = embed_fn(texts)
    seconds = time.perf_counter() - start
    return {'chunks': len(texts), 'seconds': round(seconds, 2), 'chunks_per_second': round(len(texts) / seconds, 2),
            'dimension': int(np.asarray(vectors).shape[1]),
            'peak_rss_mb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)}

def reuse_vectors(chunk_sha256, source_chunks_path, source_vectors_path):
    """Para cada texto (`chunk_sha256`), el vector ya calculado en otra muestra o None si no existe.

    Con el mismo encoder y revisión, el mismo texto produce el mismo vector: la muestra de 2.000 reutiliza
    los de la de 10.000, que la contiene.
    """
    import pandas as pd
    source = pd.read_parquet(source_chunks_path, columns=['chunk_id', 'chunk_sha256'])
    ids, matrix, n_tokens = read_vectors(source_vectors_path)
    if ids != source['chunk_id'].tolist(): raise ValueError('Los vectores de origen no están alineados con sus chunks.')
    position = {sha: i for i, sha in enumerate(source['chunk_sha256'])}
    found = [position.get(sha) for sha in chunk_sha256]
    vectors = [matrix[i] if i is not None else None for i in found]
    tokens = [int(n_tokens[i]) if i is not None else None for i in found]
    return vectors, tokens
