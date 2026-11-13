# Asistente RAG con Milvus y LLM locales

Asistente de preguntas y respuestas con generación aumentada por recuperación (RAG) sobre un corpus de prensa digital y publicaciones de Bluesky. Los fragmentos relevantes se recuperan en Milvus y la respuesta la genera un LLM local servido por Ollama, al que se le pide citar la evidencia que usa. Todo corre en Docker Compose, sin APIs de pago.

Proyecto académico de la Maestría en Inteligencia Artificial, Pontificia Universidad Javeriana.

## El taller

El proyecto responde al *Taller de investigación aplicada: Sistemas RAG con Milvus y LLM locales* (octubre de 2026). El reto es diseñar, implementar, desplegar y evaluar de forma reproducible un asistente RAG sobre un corpus real, sin servicios de pago ni dependencias obligatorias de APIs comerciales.

**Pregunta de investigación:** ¿qué LLM local pequeño ofrece el mejor equilibrio entre exactitud factual, fidelidad a las fuentes, abstención apropiada, latencia y recursos en este corpus?

El experimento principal cambia solo el LLM. Todo lo demás queda fijo: corpus, muestra, chunks, encoder, colección de Milvus, índice, `k`, contexto recuperado y prompt. Las ablaciones varían una sola dimensión a la vez (`k`, segmentación, encoder o estrategia de contexto).

**Requisitos del enunciado:**

- un único `docker-compose.yml` en la raíz, con Dockerfiles, `.env.example`, volúmenes y README;
- Milvus standalone como único motor de recuperación vectorial (etcd y MinIO son su infraestructura, no motores alternativos);
- un encoder abierto y local, y FastAPI con `/health`, `/retrieve` y `/ask`;
- una interfaz web para elegir el modelo, preguntar, ajustar `k` y examinar las fuentes;
- al menos tres LLM locales pequeños servidos con Ollama;
- un gold estándar de al menos 40 preguntas, con conjuntos de desarrollo y de prueba.

### Fases y estado

| Fase | Qué pide el enunciado | Estado | Resultados |
|---|---|---|---|
| 0. Fundaciones | Infraestructura en Docker y descarga verificada del corpus | Terminada | este README |
| A. Exploración y calidad | Fuentes, longitudes, idiomas, duplicados, fechas, sesgos y muestras reproducibles | Terminada | [`reports/eda/resumen.md`](reports/eda/resumen.md) |
| B. Segmentación | Dos estrategias de chunking con tokens, solapamiento, offsets, IDs estables y deduplicación trazable | Terminada | [`reports/chunking/resumen.md`](reports/chunking/resumen.md) |
| C. Embeddings | Encoder multilingüe justificado, vecinos semánticos y sensibilidad | Terminada | [`reports/embeddings/resumen.md`](reports/embeddings/resumen.md) |
| D. Milvus | Colección, campos, índice, filtros, reindexación idempotente, conteos y persistencia | Terminada | [`reports/milvus/resumen.md`](reports/milvus/resumen.md) |
| E. Orquestación RAG | Recuperar y generar por separado, prompts versionados, presupuesto de contexto, citas y abstención | Terminada | [`reports/rag/resumen.md`](reports/rag/resumen.md) |
| F. API e interfaz | Respuesta, tiempos, evidencias y enlaces; errores, estados de carga y modelo elegido en vivo | Terminada | [`app/`](app/) y [Uso](#uso) |
| G. Operación | Instrucciones desde cero, modelos, pruebas, persistencia, logs y límites de recursos | Parcial | este README |
| Gold estándar | 40 preguntas o más: ambas colecciones, varios idiomas y fechas, `dev` y `test` | Congelado (`gold-v1`, 52 preguntas) | [`data/gold/README.md`](data/gold/README.md) |
| Evaluación | 3 LLM o más, ablaciones y métricas de recuperación, generación, rendimiento y robustez | Ejecutada, con resultados | [`docs/EVALUACION.md`](docs/EVALUACION.md) y `reports/eval/tables/` |

Las decisiones de diseño de todas las fases, con sus alternativas y evidencia, están en [`docs/decisiones.md`](docs/decisiones.md).

## Arquitectura

```mermaid
flowchart LR
  C[(documentos_rag.parquet)] --> P[pipeline<br/>perfilado, chunking,<br/>embeddings, ingesta]
  P --> M[(Milvus)]
  U((Usuario)) --> F[frontend<br/>Streamlit]
  F --> A[api<br/>FastAPI]
  A -- recuperación --> M
  A -- generación --> O[Ollama]
```

| Servicio | Rol | Puerto local |
|---|---|---|
| `milvus` | Base de datos vectorial (standalone). WebUI en `/webui` | 19530, 9091 |
| `etcd`, `minio` | Metadatos y almacenamiento de objetos de Milvus. Consola de MinIO en 9001 | 9001 |
| `attu` | Interfaz para inspeccionar colecciones de Milvus | 3000 |
| `ollama` | LLM locales | 11434 |
| `api` | FastAPI: recuperación y generación | 8000 |
| `frontend` | Interfaz Streamlit | 8501 |
| `pipeline` | Trabajos batch (descarga, perfilado, segmentación, vectores, ingesta, reportes y pruebas). Perfil `tools`: no arranca con `up` | — |

`docker-compose.yml` solo incluye los archivos de `docker/`, uno por conjunto de servicios: `milvus.yml` (milvus, etcd, minio, attu), `ollama.yml`, `api.yml`, `frontend.yml` y `pipeline.yml`. Cada servicio propio tiene su Dockerfile en `docker/` y su lista de dependencias en `requirements/`.

El encoder se carga desde un único módulo, `app/embeddings.py`, que usan tanto `api` como `pipeline`. Ambos servicios comparten el volumen `hf_cache` con los pesos del modelo, así que las consultas y los pasajes se codifican exactamente igual.

- **Encoder:** `ibm-granite/granite-embedding-97m-multilingual-r2` (384 dimensiones, sin prefijos, coseno). La API lo toma de `.env` y el pipeline de `configs/encoders.yaml`, con la misma revisión fijada.
- **Segmentación:** mide siempre el tamaño de los chunks con el tokenizador de e5-small, para que los chunks no cambien aunque cambie el encoder.
- **Índice:** HNSW (M = 16, efConstruction = 200, COSINE), con `ef = 64` calibrado frente a la búsqueda exacta. Las colecciones las crea solo la ingesta; la API solo busca.
- **Orquestación:** recuperar (`app/core.py`) y generar (`app/rag.py`, `app/llm.py`) son pasos separados. El prompt está versionado en `prompts/` y el contexto se ajusta a un presupuesto en tokens, igual para todos los LLM. La respuesta debe citar los documentos y abstenerse si no hay evidencia.

## Requisitos

- Docker Engine con Compose 2.20 o superior (por `include`).
- RAM: se recomiendan 16 GB para la pila completa. Depende del LLM elegido. Las cifras de este repositorio se midieron con 10 CPU y 8 GB asignados a Docker.
- Disco: conviene tener unos 30 GB libres. En macOS con Apple Silicon y los cuatro LLM se midieron 17,5 GB de imágenes y 9,3 GB de volúmenes (modelos y Milvus), más la caché de build.

## Puesta en marcha

```bash
cp .env.example .env
docker compose up -d --build

# Corpus: descarga, verifica el sha256 y no sobrescribe sin --force
docker compose --profile tools run --rm pipeline python -m scripts.download_corpus

# Perfilado, muestras, chunks y vectores de la muestra de 10.000 (estrategia B; la vectorización tarda unos 35 min)
docker compose --profile tools run --rm pipeline python -m scripts.profile_corpus
docker compose --profile tools run --rm pipeline python -m scripts.build_chunks --strategy B --sample data/samples/sample_10000_ids.txt
docker compose --profile tools run --rm pipeline python -m scripts.embed_chunks --encoder granite-97m --strategy B --sample sample_10000

# Carga en Milvus de la colección que usa la API
docker compose --profile tools run --rm pipeline python -m scripts.ingest --collection sent256_s10k

# LLM por defecto
docker compose exec ollama ollama pull qwen2.5:1.5b
```

Las demás colecciones (piloto de 2.000, estrategia A y la adversarial) se cargan igual con su clave. El orden completo está en [Reproducir los resultados desde cero](#reproducir-los-resultados-desde-cero).

> **Orden:** `docker compose up -d` debe ir antes de cualquier comando que use Milvus (`ingest`, `milvus_report`, `check_rag`) y Milvus tiene que estar `healthy` (`docker compose ps`). Si no lo está, `ingest` termina con «Milvus no responde en milvus:19530; ejecuta `docker compose up -d`».

### Prueba rápida con la muestra piloto

Para comprobar la instalación sin esperar los 35 minutos de la muestra de 10.000, se puede usar la de 2.000 documentos (en la verificación con 12 CPU: 128 s el perfilado, 243 s la vectorización y 1,3 s la carga, sin contar las descargas):

```bash
docker compose --profile tools run --rm pipeline python -m scripts.download_corpus
docker compose --profile tools run --rm pipeline python -m scripts.profile_corpus
docker compose --profile tools run --rm pipeline python -m scripts.build_chunks --strategy B --sample data/samples/sample_2000_ids.txt
docker compose --profile tools run --rm pipeline python -m scripts.embed_chunks --encoder granite-97m --strategy B --sample sample_2000
docker compose --profile tools run --rm pipeline python -m scripts.ingest --collection sent256_s2k
docker compose exec ollama ollama pull qwen2.5:1.5b
```

Luego se cambia en `.env` la línea `MILVUS_COLLECTION=rag_granite97m_sent256_s2k_v1` y se recrea la API (`docker compose up -d --force-recreate api`). Comprobación: `curl -s localhost:8000/health` debe mostrar `"entities": 7517`.

Después:

- Interfaz: http://localhost:8501
- Documentación de la API: http://localhost:8000/docs
- Attu: http://localhost:3000. Conectar a `milvus:19530`, sin usuario ni contraseña.
- WebUI de Milvus: http://localhost:9091/webui/

## Uso

### Interfaz

1. Abrir http://localhost:8501. La barra lateral muestra el estado de Milvus y de Ollama.
2. Elegir el LLM. La lista sale de los modelos descargados en Ollama (ver [Modelos LLM](#modelos-llm)); si no hay ninguno, la interfaz indica cómo descargarlo.
3. Ajustar `k`, el número de fragmentos que se recuperan (de 1 a 15).
4. Opcional, en **Opciones avanzadas**: la colección, el idioma, la fuente (prensa o Bluesky), un rango de fechas, la temperatura y la semilla.
5. Escribir la pregunta y pulsar **Consultar**.

La interfaz muestra:

- la respuesta, con un aviso si el modelo se abstuvo, si no citó ningún documento o si citó documentos que no recibió;
- el tiempo total, el de recuperación y el de generación, los tokens por segundo y cuántos fragmentos entraron al contexto;
- el modelo, la versión del prompt, la colección, la temperatura y la semilla;
- cada evidencia con su número `[n]`, título, fuente, fecha, score, offsets, `document_id`, procedencias, enlace y texto. Las citadas aparecen abiertas, y se marcan las que quedaron fuera del contexto por el presupuesto.

Cada error tiene su mensaje: modelo sin descargar (con el comando para descargarlo), colección inexistente, Milvus u Ollama caídos y tiempo agotado. Las respuestas siempre deben contrastarse con las fuentes.

### API

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/health` | Estado de Milvus (colección por defecto y entidades), Ollama y el encoder. Responde 503 si falta alguno |
| `GET` | `/models` | Modelos descargados en Ollama, con digest, tamaño y cuantización |
| `GET` | `/collections` | Colecciones `rag_*` de Milvus con su número de entidades |
| `POST` | `/retrieve` | Fragmentos recuperados con su score, metadatos y tiempos |
| `POST` | `/ask` | Respuesta, citas, abstención, avisos, evidencias, tokens, tiempos por etapa, prompt y `run_id` |

```bash
curl -s localhost:8000/health
curl -s localhost:8000/models
curl -s -X POST localhost:8000/retrieve -H 'Content-Type: application/json' \
  -d '{"query": "¿Qué se acordó en la plenaria de la COP16?", "k": 5}'
curl -s -X POST localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"query": "¿Qué se acordó en la plenaria de la COP16?", "model": "qwen2.5:1.5b", "k": 5,
       "filters": {"language": "es", "desde": "2024-10-01", "hasta": "2024-11-30"}}'
```

Cuerpo de `/retrieve` y `/ask`:

| Campo | Valores |
|---|---|
| `query` | De 3 a 2.000 caracteres |
| `k` | De 1 a 15 (5 por defecto) |
| `collection` | Una colección `rag_*`; por defecto, `MILVUS_COLLECTION` |
| `filters.language` | Código ISO de 2 o 3 letras (idioma declarado o, si falta, detectado) |
| `filters.collection_name` | `url_crawls` (prensa) o `bluesky_posts` |
| `filters.desde`, `filters.hasta` | Días `AAAA-MM-DD` en UTC, ambos incluidos |
| `model` | Solo `/ask`. Un modelo de Ollama; por defecto, `LLM_MODEL` |
| `temperature`, `seed` | Solo `/ask`. Temperatura de 0 a 1 (0 por defecto) y semilla (por defecto, `LLM_SEED`) |

La respuesta de `/ask` incluye:

- `answer`, `abstained` y `citations` (cada `[n]` citado, con su `chunk_id`, `document_id` y `url`);
- `invalid_citations` y `warnings`: `sin_citas`, `citas_invalidas`, `citas_en_texto` (citó con palabras, "según el documento n"), `abstencion_parcial`, `solo_citas` (solo números entre corchetes), `contexto_recortado`, `respuesta_cortada`, `posible_truncamiento`, `sin_evidencia` o `filtra_instrucciones`;
- `context`: fragmentos que entraron, omitidos y tokens estimados;
- `tokens`: del prompt y de la respuesta, según Ollama, y tokens por segundo;
- `timings_ms`: consulta, búsqueda, contexto, carga del modelo, lectura del prompt, generación y total;
- `cold_start`, `prompt` (id, versión y sha256), `options`, `retrieval` y `run_id`;
- `evidence`, una entrada por fragmento recuperado con `rank`, `in_context` y `cited`.

Cada consulta queda registrada en `data/logs/ask/<fecha>.jsonl` (no se versiona).

Cada fragmento de la evidencia trae:

- `chunk_id` y `score` (similitud coseno);
- `document_id`, `chunk_index`, y `char_start` y `char_end` (offsets en caracteres sobre el texto original);
- `text`, `title` y `url`;
- `collection_name`, `source` y `language`;
- `created_at` (segundos UTC, o `null`);
- `provenance_ids`: todos los documentos que contienen ese mismo texto.

### Colecciones

La API busca en la colección de `MILVUS_COLLECTION`. Las cinco colecciones se definen en `configs/milvus.yaml` y se cargan con `scripts.ingest --collection <clave>`:

| Clave | Colección | Contenido |
|---|---|---|
| `sent256_s10k` | `rag_granite97m_sent256_s10k_v1` | Estrategia B, muestra de 10.000 (37.919 chunks). La usa la API por defecto |
| `win256_s10k` | `rag_granite97m_win256_s10k_v1` | Estrategia A, muestra de 10.000 (33.431 chunks) |
| `sent256_s2k` | `rag_granite97m_sent256_s2k_v1` | Estrategia B, piloto de 2.000 (7.517 chunks) |
| `win256_s2k` | `rag_granite97m_win256_s2k_v1` | Estrategia A, piloto de 2.000 (6.659 chunks) |
| `sent256_adv` | `rag_granite97m_sent256_adv_v1` | Prueba adversarial (44 chunks con gold-v1): documentos de Q45-Q48 y 4 documentos sintéticos con inyección de prompt |

Para que la API use otra colección, se cambia `MILVUS_COLLECTION` en `.env` y se recrea el servicio:

```bash
docker compose up -d --force-recreate api
```

La colección adversarial contiene instrucciones maliciosas a propósito, así que no se debe usar como colección por defecto.

## Configuración

Las variables se leen de `.env` (plantilla en `.env.example`):

| Variable | Uso |
|---|---|
| `MILVUS_URI` | Dirección de Milvus |
| `MILVUS_COLLECTION` | Colección en la que busca la API. El nombre identifica encoder, segmentación, muestra y versión, p. ej. `rag_granite97m_sent256_s10k_v1` |
| `MILVUS_SEARCH_EF` | Amplitud de búsqueda del índice HNSW (64, calibrada en `reports/milvus/`) |
| `EMBEDDING_MODEL` | Modelo de Sentence Transformers |
| `EMBEDDING_REVISION` | Revisión (commit) fija del modelo en Hugging Face |
| `EMBEDDING_QUERY_PREFIX`, `EMBEDDING_PASSAGE_PREFIX` | Prefijos de consulta y pasaje. Vacíos para granite; con E5 serían `"query: "` y `"passage: "`. Van entre comillas para conservar el espacio final |
| `EMBEDDING_NORMALIZE` | Normalización L2 de los vectores (`true` o `false`) |
| `OLLAMA_URL` | Dirección de Ollama |
| `LLM_MODEL` | Modelo por defecto de la API y la interfaz; cada petición puede elegir otro |
| `LLM_NUM_CTX`, `LLM_NUM_PREDICT` | Ventana de contexto fija para todos los modelos (4096) y máximo de tokens de respuesta (512) |
| `LLM_TOKEN_FACTOR` | Tokens del LLM por cada token del encoder, calibrado en `reports/rag/`; ajusta el presupuesto de contexto |
| `LLM_SEED`, `LLM_TIMEOUT` | Semilla por defecto y tiempo máximo de espera de Ollama en segundos |
| `LLM_NUM_THREAD` | Hilos de Ollama por petición (8 de 10), para dejar CPU libre a etcd y Milvus |
| `PROMPT_ID` | Prompt versionado de `prompts/<id>.yaml` |
| `LOG_DIR` | Carpeta del registro JSONL de consultas (`data/logs`) |

Si cambia el encoder, sus prefijos, la normalización o la estrategia de chunking, los vectores dejan de ser comparables. En ese caso hay que usar otra colección y regenerar los embeddings; nunca se mezclan vectores de dos configuraciones en una colección.

## Pipeline

Los trabajos batch corren en el contenedor `pipeline`, que monta `./data`, `./reports` y `./configs`:

```bash
docker compose --profile tools run --rm pipeline python -m scripts.<tarea>
```

| Tarea | Qué hace |
|---|---|
| `download_corpus` | Descarga `dragons_rag` con `open_corpus_co_es`, verifica tamaño y sha256 contra el catálogo y lo copia sin modificar a `data/documentos_rag.parquet`, junto con su manifiesto (`data/documentos_rag.manifest.json`). Opciones: `--output`, `--force` |
| `profile_corpus` | Perfilado del corpus (integridad, longitudes y tokens, idioma, fechas, duplicados, URLs, ruido y sesgos) y muestras estratificadas. Escribe `reports/eda/`, `data/samples/` y artefactos intermedios en `data/eda/` (no versionados). Parámetros en `configs/eda.yaml`; tarda unos 3 minutos |
| `build_chunks` | Segmenta una muestra con la estrategia A (ventana de 256 tokens con 32 de solapamiento) o B (párrafos y oraciones hasta 256 tokens, con una oración de solapamiento), deduplica conservando procedencias y actualiza la comparación en `reports/chunking/`. Los chunks quedan en `data/chunks/<estrategia>/<muestra>.parquet` (no versionados). Opciones: `--strategy A\|B`, `--sample` (lista de IDs o `full`). Requiere los artefactos de `profile_corpus` |
| `benchmark_encoders` | Mide en CPU, dentro del contenedor, los chunks por segundo y la RAM de cada perfil de `configs/encoders.yaml` |
| `embed_chunks` | Vectoriza los chunks de una estrategia y muestra con un perfil de encoder y escribe `data/embeddings/<perfil>/<estrategia>/<muestra>.parquet` (no versionado). Si se interrumpe, se puede reanudar. Con `--reuse <muestra>` toma los vectores ya calculados en otra muestra por el hash del texto y solo calcula los que faltan. Opciones: `--encoder`, `--strategy`, `--sample`, `--reuse` |
| `encoder_report` | Analiza el encoder con las preguntas `dev` del gold: variantes de consulta, paráfrasis y traducción, idiomas, dimensión y vecinos de ejemplo. Escribe `reports/embeddings/` |
| `ingest` | Carga en Milvus una colección de `configs/milvus.yaml` desde los Parquet de chunks y de vectores. Hace `upsert` por lotes (repetirla no duplica), reintenta con espera creciente, retoma desde el último lote si se interrumpe y verifica el conteo. Opciones: `--collection`, `--fresh` (borra la colección y su progreso antes de cargar) |
| `milvus_report` | Verifica las colecciones: conteos, calibración de `ef` frente a la búsqueda exacta, filtros y métricas en dev. Con `--snapshot` y `--compare` comprueba la persistencia tras reiniciar Milvus. Escribe `reports/milvus/` |
| `pool_gold` | Pooling del gold: consulta las colecciones A y B de Milvus con cada pregunta, une los primeros 20 documentos de cada una y escribe `data/gold/pooling.jsonl` (solo ids y rangos), la cobertura de los relevantes ya anotados y las hojas de anotación `.xlsx` (locales, con texto del corpus; en `data/annotation/`). Con `--fase2`, la hoja de validación de las respuestas de referencia |
| `apply_annotations` | Lee las hojas llenas, escribe `data/gold/anotacion_1.jsonl` y `anotacion_2.jsonl` (ids y grados), calcula el acuerdo entre las dos preanotaciones en `reports/gold/` y prepara `desacuerdos.xlsx` |
| `freeze_gold` | Aplica los grados finales al gold y lo congela (`estado = congelado`). Sin `--write` es un ensayo; bloquea si queda pooling pendiente, desacuerdos sin resolver o una pregunta sin respuesta con documentos relevantes |
| `evaluate` | Evaluación del taller con `configs/experimentos.yaml`. `retrieval` mide A y B con el gold; `generation` ejecuta las condiciones, los modelos y las corridas, y reanuda si se interrumpe. Escribe `data/logs/eval/generacion.jsonl` (con respuestas completas, no se versiona) y `reports/eval/` |
| `analyze` | Métricas por celda, intervalos bootstrap del 95 % y latencias (frío y caliente) en `reports/eval/tables/`, sin texto del corpus. Con otro `--raw`, usar un `--reports` distinto para no sobrescribir las tablas principales |
| `judge_prepare` | Prepara el lote anónimo para los jueces LLM y la hoja de auditoría humana en `data/logs/eval/juez/` (contiene texto del corpus, no se versiona) |
| `judge_analyze` | Lee las calificaciones de los jueces y la auditoría humana, calcula el acuerdo y las tablas `juez_*.csv` en `reports/eval/tables/` |
| `check_rag` | Revisa el RAG solo con las preguntas `dev`. Con `--calibrar` mide en cada LLM los tokens reales del prompt frente a los del encoder y propone `LLM_TOKEN_FACTOR`; sin esa opción responde las preguntas con cada prompt y modelo dados y resume abstención, citas, tokens y latencias. Escribe `reports/rag/`; las respuestas completas van a `data/logs/check_rag/`. Opciones: `--prompts`, `--models`, `--k`, `--calibrar` |

La ingesta necesita Milvus en marcha (`docker compose up -d`). El código va dentro de la imagen: después de cambiarlo hay que reconstruir con `docker compose --profile tools build pipeline`.

### Reproducir los resultados desde cero

`data/` no se versiona, y los volúmenes de Milvus son locales, así que en una máquina nueva los artefactos se regeneran en este orden. Los tiempos aproximados:

- perfilado y segmentación: unos 5 minutos;
- vectorización de la muestra de 10.000 con granite-97m: unos 35 minutos por estrategia;
- carga en Milvus: menos de un minuto.

A eso se suman las descargas del corpus (245 MB) y de los modelos (solo la primera vez).

```bash
docker compose --profile tools run --rm pipeline python -m scripts.download_corpus
docker compose --profile tools run --rm pipeline python -m scripts.profile_corpus
for n in 2000 10000; do for s in A B; do
  docker compose --profile tools run --rm pipeline python -m scripts.build_chunks --strategy $s --sample data/samples/sample_${n}_ids.txt
done; done
for s in A B; do
  docker compose --profile tools run --rm pipeline python -m scripts.embed_chunks --encoder granite-97m --strategy $s --sample sample_10000
  # la muestra de 2.000 está dentro de la de 10.000: reutiliza sus vectores
  docker compose --profile tools run --rm pipeline python -m scripts.embed_chunks --encoder granite-97m --strategy $s --sample sample_2000 --reuse sample_10000
done
docker compose --profile tools run --rm pipeline python -m scripts.encoder_report --encoder granite-97m --strategy B --sample sample_10000
docker compose up -d
for c in sent256_s2k win256_s2k sent256_s10k win256_s10k sent256_adv; do
  docker compose --profile tools run --rm pipeline python -m scripts.ingest --collection $c
done
docker compose --profile tools run --rm pipeline python -m scripts.milvus_report
# revisión del RAG en dev (necesita los modelos de Ollama, ver Modelos LLM)
docker compose --profile tools run --rm pipeline python -m scripts.check_rag --calibrar --models qwen2.5:1.5b llama3.2:1b llama3.2:3b gemma2:2b
docker compose --profile tools run --rm pipeline python -m scripts.check_rag --prompts rag_v1 rag_v2 rag_v3 rag_v4
```

Si el entorno reproduce los resultados, `git status` solo muestra cambios en fechas, duraciones y tiempos:

- `reports/eda/profile.json`;
- `data/samples/*_meta.json`;
- `reports/chunking/runs/*.json`, `reports/embeddings/runs/*.json`, `reports/milvus/runs/*.json`, `reports/rag/runs/*.json` y `reports/milvus/verificacion.json`. Además de fechas y tiempos, cambian los campos que dependen del recorrido: en `reports/embeddings/runs` si los vectores se reutilizaron o se calcularon (`reused_from`, `parts`, `peak_rss_mb`), en `reports/milvus/runs` si la colección se creó (`created`), y las versiones de las librerías si no coinciden con las del registro;
- las columnas de tiempo de `reports/chunking/tables/comparacion_*.csv` y las figuras de `reports/chunking/figures/`, cuyos bytes pueden cambiar aunque los datos coincidan;
- las latencias de `reports/milvus/tables/calibracion_ef.csv` y su figura, y las de `reports/rag/tables/dev_*.csv`.

Las demás tablas y las listas de IDs salen idénticas, y los conteos de chunks y sus estadísticas también (en la verificación desde un clon limpio, 7.581 → 7.517 chunks en la muestra de 2.000 con la estrategia B). El grafo HNSW puede construirse algo distinto en otra máquina, así que el Recall@10 de la calibración puede variar en milésimas. Las respuestas de los LLM usan semilla y temperatura 0, pero pueden cambiar entre versiones de Ollama o entre máquinas.

### Reproducir la evaluación

La evaluación se ejecutó en el Mac local, con la venv y con Ollama nativo (en Apple Silicon usa la GPU), no en el contenedor `pipeline`. Milvus sigue en Docker:

```bash
export MILVUS_URI=http://localhost:19530
export OLLAMA_URL=http://localhost:11434
python -m scripts.evaluate retrieval
python -m scripts.evaluate generation
python -m scripts.analyze
python -m scripts.judge_prepare
```

Los lotes de `data/logs/eval/juez/lote_parte_NN.md` se entregan a cada juez en un chat nuevo. Sus salidas se guardan como `<juez>.jsonl` junto con `auditoria_humana.csv` (la hoja de la auditoría del equipo), y entonces:

```bash
python -m scripts.judge_analyze
```

La generación completa (4 modelos, 6 condiciones, 3 corridas, 52 preguntas) son 3.744 celdas y tarda varias horas. Los parámetros están en `configs/experimentos.yaml`.

## Modelos LLM

```bash
docker compose exec ollama ollama pull qwen2.5:1.5b   # también llama3.2:1b, llama3.2:3b y gemma2:2b
docker compose exec ollama ollama list
docker compose exec ollama ollama rm <modelo>
```

Los cuatro modelos del experimento ocupan unos 5,9 GB. El selector de la interfaz muestra los modelos descargados (`GET /models`). Si se pide un modelo que no está descargado, la API responde 404 con el comando para descargarlo.

## Datos

El corpus es `dragons_rag` (versión 2026-10-19) del paquete [`open_corpus_co_es`](https://puj-nlp.github.io/open_corpus_co_es/). Según su catálogo tiene 169.248 documentos y 33 columnas: 64.670 publicaciones de Bluesky, 103.380 noticias de Media Cloud y 1.198 de GDELT. El script de descarga verifica el sha256 (`f4a669f1…b181728`) e informa el número de filas.

Reglas de datos:

- El corpus y sus derivados no se versionan. De `data/` solo se versionan `README.md`, las listas de IDs de `data/samples/` y el gold estándar de `data/gold/`.
- `reports/` contiene solo agregados (conteos, percentiles, figuras): ni texto del corpus, ni handles de Bluesky, ni nombres de personas.
- El Parquet original nunca se modifica. Las limpiezas producen artefactos nuevos.
- Los campos de adquisición (palabras clave, `source`) describen cómo se recolectó el documento; no son etiquetas semánticas.

El perfilado completo (integridad, idioma, fechas, duplicados, ruido, sesgos) y las decisiones que implica están en [`reports/eda/resumen.md`](reports/eda/resumen.md), con su narrativa en [`notebooks/01_exploracion_calidad.ipynb`](notebooks/01_exploracion_calidad.ipynb). Las muestras reproducibles de 2.000 y 10.000 documentos (solo IDs) están en `data/samples/`.

### Gold estándar

`data/gold/` tiene 52 preguntas con evidencia anotada por documento y rango de caracteres, sin texto del corpus:

- **Partición:** 20 de `dev` y 32 de `test`.
- **Tipos:** directas, comparativas, ambiguas, sin respuesta y adversariales.
- **Cobertura:** las dos colecciones, preguntas en español e inglés, evidencia en español, inglés y húngaro, y dos preguntas limitadas a una ventana de fechas.
- **Documentos de evidencia:** los 53 están en la muestra de 10.000.
- **Uso:** `dev` sirve para ajustar y `test` solo para reportar.

Está congelado (`estado = congelado`, tag `gold-v1`): el pooling con las colecciones de Milvus (`pool_gold`), la doble anotación (preanotaciones de dos LLM de apoyo, validadas por el equipo) y su acuerdo (`apply_annotations`) y el congelamiento (`freeze_gold`) ya se ejecutaron. El esquema, las métricas y el procedimiento están en [`data/gold/README.md`](data/gold/README.md).

## Estructura

```
app/            servicios en línea: api.py, frontend.py, core.py (Milvus), rag.py (contexto, citas, registro),
                llm.py (Ollama), filters.py, embeddings.py (encoder)
prompts/        prompts versionados (rag_v1.yaml, el de la API, y las variantes rag_v2 a rag_v4 probadas en dev)
pipeline/       lógica batch reutilizable y testeable
scripts/        CLI que se ejecutan en el contenedor pipeline (python -m scripts.<tarea>)
configs/        parámetros YAML de las etapas (eda.yaml, chunking.yaml, encoders.yaml, milvus.yaml)
notebooks/      exploración y narrativa; leen los artefactos de reports/, no recalculan
docker/         compose por conjunto de servicios y un Dockerfile por servicio
requirements/   dependencias por imagen (api, frontend, pipeline, torch-cpu) y dev.txt para local
tests/          pruebas con pytest
data/           corpus y derivados (no se versiona, salvo samples/ y gold/)
reports/        resultados agregados (eda/, chunking/, embeddings/, milvus/, rag/)
docs/           protocolo de evaluación (EVALUACION.md) y registro de decisiones (decisiones.md)
```

## Desarrollo local y pruebas

```bash
# Pruebas en Docker (referencia)
docker compose --profile tools run --rm pipeline pytest -q

# Entorno local (las imágenes usan Python 3.12; la evaluación se ejecutó con 3.13.7)
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements/dev.txt
python -m nltk.downloader punkt_tab   # separador de oraciones de la segmentación (ya incluido en la imagen)
pytest -q
```

Algunas pruebas se omiten si falta lo que necesitan: las del tokenizador real, si el encoder no está en caché; la de integración con Milvus, si no está en marcha; y la de punta a punta con el LLM, si falta `qwen2.5:1.5b` en Ollama (se puede saltar con `RUN_LLM_TESTS=0`). El entorno local sirve para explorar. Toda cifra que se reporte sale de lo que corre en Docker.

## Limitaciones actuales

- El contexto se arma con los fragmentos en orden de rank; agruparlos por documento queda como ablación de la evaluación.
- El presupuesto de tokens se mide con el tokenizador del encoder y un factor calibrado, porque Ollama no expone el conteo antes de generar. La API compara después con el conteo real y avisa si se pasó.
- La abstención depende de que el modelo use la frase fija del prompt. Una abstención con otras palabras no se detecta.
- El prompt por defecto (`rag_v1`) prioriza abstenerse bien: en `dev`, con `qwen2.5:1.5b`, el 45 % de sus respuestas no cita documentos. `rag_v4` cita más, pero se abstiene peor (ver [`reports/rag/resumen.md`](reports/rag/resumen.md)).
- Hay dos entornos de medida. En Docker, en CPU, con `qwen2.5:1.5b` y `k = 5` una respuesta tarda 7,8 s de mediana, y la primera tras cambiar de modelo suma unos 6,6 s de carga (revisión de desarrollo). La evaluación final usó Ollama 0.40.16 nativo con la GPU de Apple, donde la mediana es de unos 1,24 s con el mismo modelo. Una medida adicional en Docker, en CPU (Ollama 0.40.16, VM con 12 CPU y 7,7 GB; solo test y una corrida, en `reports/eval/docker_cpu`), da 4,0 s con `qwen2.5:1.5b` y 8,4 s con `llama3.2:3b`, unas 3 veces más que en nativo, sobre todo por el procesamiento del prompt de unos 1.700 tokens. La diferencia con los 7,8 s de la revisión de desarrollo (10 núcleos, Ollama 0.40.13) no se puede atribuir a una causa con estos datos. Las latencias de un entorno no son comparables con las del otro. Los modelos más grandes son más lentos.
- Se indexaron las muestras de 2.000 y 10.000 documentos. Vectorizar el corpus completo en CPU tomaría unas 10 horas.
- La imagen de Ollama usa la etiqueta `latest`. Las demás imágenes de terceros tienen versión fija, pero aún sin digest.

## Uso responsable

- El corpus mezcla publicaciones en redes sociales y noticias. Hay que distinguir alegaciones, atribuciones y evidencia, y no equiparar cobertura con veracidad.
- Las respuestas del asistente deben contrastarse con las fuentes originales que cita.
- No se publican documentos del corpus ni datos personales en el repositorio.

## Reproducibilidad

Cada experimento debe registrar:

- el sha256 del Parquet y del gold;
- la semilla;
- la versión y el digest de las imágenes;
- el nombre, la versión, el hash y la cuantización de cada modelo;
- los parámetros de inferencia, la colección (encoder y segmentación) y `k`;
- los tiempos por etapa;
- el hardware.

Las muestras se obtienen con muestreo estratificado y semilla fija. Las tareas del pipeline registran sus parámetros, versiones y hashes de entrada en JSON: `reports/eda/profile.json` y `reports/<etapa>/runs/`.

## Solución de problemas

| Síntoma | Causa y solución |
|---|---|
| `/ask` responde 404 con "no está descargado en Ollama" | Descargar el modelo con el comando que trae el mensaje: `docker compose exec ollama ollama pull <modelo>` |
| `/ask` responde 404 con "La colección … no existe" | La colección no está cargada. Hay que cargarla (`scripts.ingest --collection sent256_s10k`) o revisar `MILVUS_COLLECTION` en `.env` |
| `/ask` responde 504 | El modelo tardó más de `LLM_TIMEOUT` segundos. Probar con menos fragmentos (`k`) o un modelo más pequeño |
| `/health` responde 503 | Milvus u Ollama no responden, o la colección por defecto está vacía; el cuerpo dice cuál |
| Milvus se reinicia mientras Ollama genera | Con la CPU saturada, etcd se atrasa y Milvus pierde su sesión; se levanta solo (`restart: unless-stopped`). `LLM_NUM_THREAD` deja núcleos libres; si pasa seguido, bajarlo |
| `/ask` responde 503 con "model runner has unexpectedly stopped" | El modelo no cupo en la RAM de Docker. `docker/ollama.yml` ya deja un solo modelo cargado y desactiva la caché de prompts en RAM (`LLAMA_ARG_CACHE_RAM=0`); si aun así falla, cerrar otros procesos o usar un modelo más pequeño |
| La respuesta trae el aviso `sin_evidencia` | Los filtros no dejaron ningún fragmento. Ampliar el rango de fechas o quitar el filtro de idioma |
| `ingest` dice «Milvus no responde en milvus:19530» | La pila no está en marcha o Milvus aún no está `healthy`. Ejecutar `docker compose up -d` y esperar a que `docker compose ps` lo muestre sano |
| `ingest` falla porque no encuentra `data/embeddings/...` | Falta vectorizar esa estrategia y muestra con `embed_chunks`. Para el piloto basta `--reuse sample_10000` |
| `ingest` dice que los vectores no están alineados con los chunks | Los chunks se regeneraron después de vectorizar. Hay que repetir `embed_chunks` para esa estrategia y muestra |
| El build falla con `No space left on device` | Falta espacio en la VM de Docker. Liberar la caché con `docker builder prune` |
| Un cambio en `pipeline/` o `scripts/` no se refleja | El código va dentro de la imagen: `docker compose --profile tools build pipeline` |
| Un cambio en `.env` no se refleja en la API | Hay que recrear el servicio: `docker compose up -d --force-recreate api` |

## Flujo de trabajo

- **`main`:** siempre funciona. Solo entra lo que construye, levanta y pasa las pruebas.
- **Ramas:** una rama corta por etapa (`feature/fase-0-fundaciones`, `feature/fase-a-perfilado`, …, `feature/fase-d-milvus`, `feature/fase-e-f-orquestacion`). El gold estándar avanza en paralelo en `feature/gold-estandar`.
- **Experimentos:** no van en ramas. Se parametrizan en `configs/` y sus resultados agregados van a `reports/`.
- **Integración:** cada rama entra por PR con merge commit (`--no-ff`). Cada hito lleva un tag (`fase-0`, `fase-a`, …, `entrega-final`).
- **Commits:** en formato Conventional Commits con descripción en español (`feat: …`, `fix: …`, `docs: …`, `test: …`).
- **Retomar en otra máquina:**
  1. clonar el repositorio y copiar `.env.example` a `.env`;
  2. regenerar `data/` y las colecciones con [Reproducir los resultados desde cero](#reproducir-los-resultados-desde-cero) y comprobar que `git status` solo muestra los cambios esperados;
  3. abrir la rama de la fase siguiente desde `main` actualizado.

  Los Parquet de vectores (`data/embeddings/`) solo tienen `chunk_id`, vector y número de tokens, sin texto del corpus. Si el equipo los comparte por un canal privado (nunca por git), se ahorra la vectorización, y la ingesta comprueba que coincidan con los chunks regenerados.
