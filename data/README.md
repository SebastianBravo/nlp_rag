Aquí va `documentos_rag.parquet`, que NO se incluye en el repositorio. Se descarga y verifica (sha256) con:
`docker compose --profile tools run --rm pipeline python -m scripts.download_corpus`
El archivo Markdown entregado por el docente describe el Parquet pero no contiene los datos.
`samples/` guarda solo listas de IDs y metadatos de las muestras; es lo único de `data/` que se versiona.
