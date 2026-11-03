import hashlib
import json
import pyarrow as pa
import pyarrow.parquet as pq
from pipeline import io

def write_parquet(path):
    pq.write_table(pa.table({'document_id': [f'd{i}' for i in range(10)], 'text': [f't{i}' for i in range(10)],
                             'n': list(range(10))}), path)

def test_sha256_file(tmp_path):
    path = tmp_path / 'f.bin'
    path.write_bytes(b'abc' * 1000)
    assert io.sha256_file(path, block_size=7) == hashlib.sha256(b'abc' * 1000).hexdigest()

def test_schema_and_metadata(tmp_path):
    path = tmp_path / 'c.parquet'
    write_parquet(path)
    schema = io.parquet_schema(path)
    assert list(schema['column']) == ['document_id', 'text', 'n']
    assert list(schema['position']) == [0, 1, 2]
    assert io.parquet_metadata(path)['rows'] == 10

def test_batches_cover_all_rows_with_selected_columns(tmp_path):
    path = tmp_path / 'c.parquet'
    write_parquet(path)
    batches = list(io.iter_batches(path, ['document_id', 'n'], batch_size=4))
    assert [len(b) for b in batches] == [4, 4, 2]
    assert all(list(b.columns) == ['document_id', 'n'] for b in batches)

def test_read_rows_filters_by_id(tmp_path):
    path = tmp_path / 'c.parquet'
    write_parquet(path)
    rows = io.read_rows(path, ['document_id', 'text'], ['d2', 'd7'])
    assert sorted(rows['text']) == ['t2', 't7']

def test_manifest_next_to_parquet(tmp_path):
    parquet = tmp_path / 'documentos_rag.parquet'
    assert io.manifest_path(parquet) == tmp_path / 'documentos_rag.manifest.json'
    assert io.load_manifest(io.manifest_path(parquet)) is None
    io.manifest_path(parquet).write_text(json.dumps({'rows': 10}), encoding='utf-8')
    assert io.load_manifest(io.manifest_path(parquet)) == {'rows': 10}
