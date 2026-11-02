# Taller de RAG con Milvus: repositorio demostrativo

Este repositorio forma parte del taller de generación aumentada por recuperación (RAG) de la Pontificia Universidad Javeriana. Es el punto de partida para las actividades prácticas de ingesta, recuperación, generación y evaluación del taller.

**Proyecto base académico (scaffold), no solución terminada ni rendimiento certificado.** Únicamente Milvus; frontend Streamlit; FastAPI; Ollama local; Python y Docker Compose. Se deben descargar imágenes y pesos de modelos gratuitos, sin API de pago. Por compatibilidad, verificar imágenes disponibles y licencias de cada modelo antes de la clase.

## Requisitos

Docker Engine/Compose v2 (2.20 o superior, por `include`), al menos 16 GB RAM recomendados para la pila completa (depende de hardware y modelos), espacio para embeddings, datos y pesos. Puede ser mayor a escala completa; empezar por una muestra.

`docker-compose.yml` solo incluye los archivos de `docker/`, uno por conjunto de servicios, cada uno con su Dockerfile:

| Archivo | Servicios |
|---|---|
| `docker/milvus.yml` | Milvus, etcd, MinIO y Attu (http://localhost:3000) |
| `docker/ollama.yml` | Ollama |
| `docker/api.yml` | API FastAPI |
| `docker/frontend.yml` | Frontend Streamlit |
| `docker/pipeline.yml` | Trabajos batch (perfil `tools`; no arranca con `up`) |

## Inicio

1. `cp .env.example .env` (Windows: `copy .env.example .env`).
2. `docker compose up -d --build`.
3. Descargar el corpus (verifica el sha256 y no sobrescribe sin `--force`): `docker compose --profile tools run --rm pipeline python -m scripts.download_corpus`
4. `docker compose exec ollama ollama pull qwen2.5:1.5b`
5. `docker compose --profile tools run --rm pipeline python -m scripts.ingest --input data/documentos_rag.parquet --limit 500`
6. Abrir http://localhost:8501, documentación API http://localhost:8000/docs.
7. Comparadores opcionales: `docker compose exec ollama ollama pull llama3.2:1b`, `llama3.2:3b`, `gemma2:2b`.

Pruebas: `docker compose --profile tools run --rm pipeline pytest -q`. Entorno local de exploración (Python 3.12): `python3.12 -m venv .venv && .venv/bin/pip install -r requirements/dev.txt`.

**Importante:** los ejemplos son candidatos sujetos a licencias/recursos disponibles. El encoder se configura en `.env` (`EMBEDDING_MODEL`, `EMBEDDING_QUERY_PREFIX`, `EMBEDDING_PASSAGE_PREFIX`, `EMBEDDING_NORMALIZE`); los prefijos `query:` / `passage:` corresponden a E5, el modelo predeterminado. Al cambiar el encoder, sus prefijos o la normalización se debe cambiar el nombre de la colección, rehacer embeddings y validar la dimensión.

## Endpoints

- `GET /health`: disponibilidad de API (no prueba dependencia).
- `POST /retrieve`: chunks recuperados.
- `POST /ask`: respuesta, modelo, evidencia y latencia. JSON `{"query":"¿Qué señalan los documentos?","model":"qwen2.5:1.5b","k":5}`.

## Diseño y limitaciones deliberadas

- Esqueleto funcional en condiciones compatibles, **no ejecutado end-to-end en este entorno**.
- Chunking inicial por caracteres con solapamiento; estudiantes deben investigar tokenizer, límites, distribución y compararlo con chunking semántico.
- Indexación piloto idempotente por `chunk_id` con `upsert`; falta checkpoint robusto de procesamiento, reintentos y control de lotes grandes.
- Modelo de embedding fijo para comparar generadores sin sesgo de recuperación. Para ablations de embedding, colecciones separadas.
- Metadatos base y offsets; estudiantes ampliarán filtros, control de calidad, deduplicación, reranking, instrumentación y seguridad.
- Se guarda texto en Milvus por simplicidad educativa; evaluar limites y almacenamiento alterno para producción.
- **No publicar documentos ni datos personales** en GitHub: publicar código, instrucciones, muestras autorizadas y artefactos no sensibles.
- El corpus incluye publicaciones sociales y noticias: distinguir alegaciones, atribuciones y evidencia; no equiparar cobertura con veracidad.

## Propuesta de ramas y entregas

`main` infraestructura; `feature/ingesta`, `feature/retrieval`, `feature/ui`, `feature/evaluacion`; abrir PR con pruebas y resultados. Entregar commits, instrucciones desde cero, matriz de experimentos, benchmark, informe, demo y registro de equipo/hardware.

## Reproducibilidad

Fijar versiones y digests de imágenes finales, registrar sha256 del Parquet, semilla, nombres/versiones/hashes de los modelos, hardware, cuantización, parámetros de inferencia, tamaños de chunk, k, tiempos por etapa, subconjunto y selección aleatoria estratificada. La descarga del Parquet es un paso explícito (`scripts.download_corpus`), y el repositorio no provee gold estándar.

## Contacto

- **Luis Gabriel Moreno Sandoval**
- Pontificia Universidad Javeriana
- Correo institucional: [morenoluis@javeriana.edu.co](mailto:morenoluis@javeriana.edu.co)
