"""Lectura del corpus: Parquet por columnas y por lotes, esquema, manifiesto y huellas."""
import hashlib, json
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

def sha256_file(path, block_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(block_size), b''):
            digest.update(block)
    return digest.hexdigest()

def manifest_path(parquet_path):
    """Ruta del manifiesto junto al Parquet: data/documentos_rag.parquet -> data/documentos_rag.manifest.json."""
    return Path(parquet_path).with_suffix('.manifest.json')

def load_manifest(path):
    path = Path(path)
    if not path.is_file(): return None
    with open(path, encoding='utf-8') as source:
        return json.load(source)

def parquet_schema(path):
    """Columnas del archivo en orden, con su tipo Arrow."""
    schema = pq.read_schema(path)
    return pd.DataFrame({'position': range(len(schema.names)), 'column': schema.names,
                         'arrow_type': [str(field.type) for field in schema]})

def parquet_metadata(path):
    parquet = pq.ParquetFile(path)
    key_value = {k.decode(): v.decode() for k, v in (parquet.schema_arrow.metadata or {}).items()}
    return {'rows': parquet.metadata.num_rows, 'row_groups': parquet.metadata.num_row_groups,
            'created_by': parquet.metadata.created_by, 'key_value': key_value}

def read_columns(path, columns):
    return pq.read_table(path, columns=list(columns)).to_pandas()

def iter_batches(path, columns, batch_size):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=list(columns)):
        yield batch.to_pandas()

def read_rows(path, columns, ids, id_column='document_id'):
    """Filas cuyo `id_column` está en `ids` (p. ej. el texto de una muestra) sin cargar el resto."""
    return pq.read_table(path, columns=list(columns), filters=[(id_column, 'in', list(ids))]).to_pandas()
