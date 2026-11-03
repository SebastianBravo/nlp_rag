"""Descarga del corpus con open_corpus_co_es y copia verificada a data/."""
import json, os, re, shutil
from pathlib import Path
import gdown
import pyarrow.parquet as pq
from open_corpus_co_es import download_corpus, list_corpus
from open_corpus_co_es.downloader import get_corpus_path
from pipeline.io import sha256_file

def catalog_entry(name):
    catalog = list_corpus()
    if name not in catalog: raise ValueError(f'Corpus {name!r} no está en el catálogo de open_corpus_co_es.')
    return catalog[name]

def fetch_corpus(name, output, force=False):
    """Descarga `name` con download_corpus (valida tamaño y sha256) y lo copia byte a byte a `output`.

    No usa load_corpus: devuelve una lista de dicts sin las columnas nulas de cada fila, y volver a
    escribirla cambiaría tipos, orden de columnas y hash. Sin `force`, un `output` existente no se toca;
    con `force` se vuelve a descargar y se reemplaza.
    """
    meta = catalog_entry(name)
    output = Path(output)
    if output.exists() and not force:
        status, digest = 'skipped', sha256_file(output)
    else:
        download_corpus(name, force=force)
        source = Path(get_corpus_path(name)) / f"{name}.{meta['extension']}"
        output.parent.mkdir(parents=True, exist_ok=True)
        tmp = output.with_name(output.name + '.tmp')
        try:
            shutil.copyfile(source, tmp)
            digest = sha256_file(tmp)
            if digest != meta['sha256']:
                raise ValueError(f'sha256 de la copia ({digest}) distinto al del catálogo ({meta["sha256"]}).')
            os.replace(tmp, output)
        finally:
            if tmp.exists(): tmp.unlink()
        status = 'downloaded'
    summary = {'corpus': name, 'status': status, 'output': str(output), 'bytes': output.stat().st_size,
               'sha256': digest, 'sha256_matches_catalog': digest == meta['sha256'],
               'catalog_version': meta.get('fecha_version')}
    if meta['extension'] == 'parquet':
        summary.update(rows=pq.ParquetFile(output).metadata.num_rows, rows_catalog=meta.get('filas'))
    return summary

def fetch_manifest(name, output, force=False):
    """Descarga el manifiesto de procedencia que el catálogo publica junto al corpus (JSON en Drive).

    El manifiesto trae el sha256 del archivo, los conteos de valores no vacíos por columna y las
    distribuciones que verifica el perfilado. Sin `force`, un manifiesto existente no se toca.
    """
    output = Path(output)
    url = catalog_entry(name).get('manifiesto_url')
    if not url: return {'status': 'unavailable', 'output': None}
    if output.exists() and not force: return {'status': 'skipped', 'output': str(output)}
    match = re.search(r'/d/([\w-]+)', url)
    if not match: raise ValueError(f'No se reconoce el id de Drive del manifiesto: {url}')
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + '.tmp')
    try:
        if gdown.download(id=match.group(1), output=str(tmp), quiet=True, use_cookies=False) is None:
            raise IOError('Google Drive no devolvió el manifiesto.')
        with open(tmp, encoding='utf-8') as source: json.load(source)
        os.replace(tmp, output)
    finally:
        if tmp.exists(): tmp.unlink()
    return {'status': 'downloaded', 'output': str(output)}
