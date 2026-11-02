import hashlib
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pipeline import corpus

def write_parquet(path, rows):
    pq.write_table(pa.table({'document_id': [str(i) for i in range(rows)], 'text': ['x'] * rows}), path)

@pytest.fixture
def source(tmp_path, monkeypatch):
    """Simula open_corpus_co_es sin red: catálogo con el hash del archivo publicado y caché local."""
    published = tmp_path / 'published.parquet'
    write_parquet(published, 3)
    cache = tmp_path / 'cache'
    calls = []
    def download(name, force=False):
        calls.append(force)
        (cache / name).mkdir(parents=True, exist_ok=True)
        (cache / name / f'{name}.parquet').write_bytes(published.read_bytes())
    entry = {'extension': 'parquet', 'sha256': hashlib.sha256(published.read_bytes()).hexdigest(),
             'filas': 3, 'fecha_version': '2026-10-19'}
    monkeypatch.setattr(corpus, 'download_corpus', download)
    monkeypatch.setattr(corpus, 'get_corpus_path', lambda name: str(cache / name))
    monkeypatch.setattr(corpus, 'list_corpus', lambda: {'demo': entry})
    return calls, entry

def test_downloads_and_verifies(tmp_path, source):
    output = tmp_path / 'data' / 'documentos.parquet'
    summary = corpus.fetch_corpus('demo', output)
    assert summary['status'] == 'downloaded' and summary['sha256_matches_catalog']
    assert summary['rows'] == summary['rows_catalog'] == 3
    assert list(output.parent.iterdir()) == [output]

def test_existing_output_is_not_overwritten_without_force(tmp_path, source):
    calls, _ = source
    output = tmp_path / 'documentos.parquet'
    write_parquet(output, 1)
    before = output.read_bytes()
    summary = corpus.fetch_corpus('demo', output)
    assert summary['status'] == 'skipped' and not summary['sha256_matches_catalog']
    assert output.read_bytes() == before and calls == []

def test_force_replaces_existing_output(tmp_path, source):
    calls, _ = source
    output = tmp_path / 'documentos.parquet'
    write_parquet(output, 1)
    summary = corpus.fetch_corpus('demo', output, force=True)
    assert summary['status'] == 'downloaded' and summary['sha256_matches_catalog']
    assert summary['rows'] == 3 and calls == [True]

def test_hash_mismatch_leaves_no_output(tmp_path, source):
    _, entry = source
    entry['sha256'] = '0' * 64
    output = tmp_path / 'data' / 'documentos.parquet'
    with pytest.raises(ValueError):
        corpus.fetch_corpus('demo', output)
    assert list(output.parent.iterdir()) == []

def test_unknown_corpus_fails(tmp_path, source):
    with pytest.raises(ValueError):
        corpus.fetch_corpus('no_existe', tmp_path / 'x.parquet')

def test_sha256_file(tmp_path):
    path = tmp_path / 'f.bin'
    path.write_bytes(b'abc' * 1000)
    assert corpus.sha256_file(path, block_size=7) == hashlib.sha256(b'abc' * 1000).hexdigest()
