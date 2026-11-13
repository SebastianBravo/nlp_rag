# Indexación en Milvus (Fase D)

Este resumen describe las colecciones Milvus del proyecto, cómo se cargan y cómo se verificaron. Las cifras salen de `runs/*.json`, `tables/*.csv` y `verificacion.json`, que regeneran:

```bash
for c in sent256_s2k win256_s2k sent256_s10k win256_s10k sent256_adv; do
  docker compose --profile tools run --rm pipeline python -m scripts.ingest --collection $c
done
docker compose --profile tools run --rm pipeline python -m scripts.milvus_report
docker compose --profile tools run --rm pipeline python -m scripts.milvus_report --snapshot antes
docker compose restart milvus
docker compose --profile tools run --rm pipeline python -m scripts.milvus_report --snapshot despues
docker compose --profile tools run --rm pipeline python -m scripts.milvus_report --compare antes despues
```

Los parámetros están en `configs/milvus.yaml`. Milvus 2.6.6 standalone y pymilvus 2.6.

## Colecciones

El nombre de cada colección identifica encoder, segmentación, alcance y versión, así que nunca se mezclan vectores de dos configuraciones.

| Colección | Estrategia | Muestra | Chunks = entidades |
|---|---|---|---|
| `rag_granite97m_sent256_s2k_v1` | B (estructura) | piloto de 2.000 | 7.517 |
| `rag_granite97m_win256_s2k_v1` | A (ventana) | piloto de 2.000 | 6.659 |
| `rag_granite97m_sent256_s10k_v1` | B | 10.000 | 37.919 |
| `rag_granite97m_win256_s10k_v1` | A | 10.000 | 33.431 |
| `rag_granite97m_sent256_adv_v1` | B | documentos de Q45-Q48 y 4 sintéticos | 44 (36 con el gold en borrador) |

- **La API usa `sent256_s10k`**: la estrategia B es la candidata por defecto, y la muestra de 10.000 contiene los 53 documentos de evidencia del gold.
- **El piloto de 2.000** sirve para iterar rápido sobre la orquestación. Solo contiene 7 de esos 53 documentos, así que no sirve para medir con el gold.
- **La colección adversarial** une los 32 chunks reales de las preguntas Q45-Q48 con los 4 chunks de los documentos sintéticos con inyección de prompt. Está aparte para no contaminar las colecciones generales.

## Esquema

| Grupo | Campos |
|---|---|
| Identidad y vector | `chunk_id` (clave primaria, determinista), `vector` (384 dimensiones, granite-97m), `chunk_sha256` |
| Trazabilidad | `document_id`, `chunk_index`, `char_start`, `char_end`, `provenance_ids` (lista de todos los documentos con ese texto) y `n_provenance` |
| Contenido | `text`, `n_tokens`, `strategy`, `strategy_version` |
| Filtros | `language`, `language_source`, `collection_name`, `source`, `created_at`, `created_at_precision`, `created_at_plausible`, `has_mojibake` |
| Procedencia | `url`, `title` |

- **`language`:** es el idioma declarado y, si falta, el detectado en la Fase A. `language_source` indica cuál de los dos es (`declarado`, `detectado` o `ninguno`). Así los 763 chunks de Bluesky sin idioma declarado se pueden filtrar igual.
- **`created_at`:**
  - se guarda en segundos UTC y admite nulos;
  - `created_at_precision` distingue las fechas sin hora (el 95 % de la prensa) de las que tienen hora;
  - `created_at_plausible` marca como no plausibles los posts de Bluesky anteriores a 2023.
- **Títulos completos:** se guardan sin cortar (el más largo tiene 7.998 caracteres). El piloto anterior los cortaba en 2.048.

## Índices

- **Vector: HNSW con métrica COSINE, M = 16 y efConstruction = 200.**
  - Reemplaza el AUTOINDEX anterior, que no dejaba documentar ni calibrar los parámetros.
  - M = 16 y efConstruction = 200 son valores estándar para colecciones de decenas de miles de vectores.
  - Los vectores de granite vienen normalizados, así que el coseno es el producto interno.
- **Campos escalares:** INVERTED en `language`, `collection_name` y `source`, y STL_SORT en `created_at`, para los filtros por igualdad y por rango de fechas.

### Calibración de `ef`

`ef` es la amplitud de búsqueda en el grafo. La regla, fijada antes de medir, fue elegir el menor `ef` cuyo Recall@10 medio frente a la búsqueda exacta llegue a 0,99.

Se midió con 545 consultas sobre `sent256_s10k`: las 45 variantes de las preguntas de dev (original, paráfrasis y traducción) y 500 vectores de chunks elegidos al azar con semilla fija.

| ef | Recall@10 medio | Recall@10 mínimo | Latencia p50 | Latencia p95 |
|---|---|---|---|---|
| 16 | 0,970 | 0,20 | 0,70 ms | 0,89 ms |
| 32 | 0,989 | 0,30 | 0,76 ms | 1,14 ms |
| **64** | **0,996** | 0,40 | 0,81 ms | 0,88 ms |
| 128 | 0,999 | 0,90 | 0,97 ms | 1,06 ms |
| 256 | 1,000 | 0,90 | 1,25 ms | 1,58 ms |

- **Elegido: `ef = 64`** (`MILVUS_SEARCH_EF` en `.env`).
- **El peor caso con `ef = 64` es una consulta concreta.** Un diagnóstico aparte mostró que el mínimo de 0,40 corresponde a una paráfrasis de dev cuyos vecinos están muy apretados: el décimo resultado exacto tiene similitud 0,8205 y el décimo del índice, 0,8102. Con `ef = 128` el mínimo sube a 0,90 a cambio de unos 0,15 ms por consulta.
- **Ese costo no importa en la práctica:** el LLM tarda segundos, así que se puede subir `ef` en la fase de orquestación si hiciera falta.

Figura: `figures/calibracion_ef.png`.

## Carga

- **Cómo se carga:** `scripts.ingest` lee los Parquet de chunks (Fase B) y de vectores (Fase C), comprueba que estén alineados por `chunk_id` y hace `upsert` por lotes de 500. La colección se crea con consistencia `Strong`.
- **Robustez:**
  - cada lote se reintenta hasta 5 veces, con esperas de 2, 4, 8… segundos;
  - un archivo de progreso en `data/milvus/` permite retomar una carga interrumpida;
  - ese progreso se invalida si cambian los Parquet de entrada.
- **Verificación en cada carga:** al terminar se comparan las entidades de Milvus con los chunks del Parquet. Si no coinciden, la carga falla.
- **Tiempos:** de 0,6 s (adversarial) a 11,8 s (`sent256_s10k`), sin reintentos.
- **Repetir la carga no duplica.** Se borró el progreso de `sent256_s2k` y se cargó de nuevo: los 7.517 chunks se volvieron a escribir y la colección siguió con 7.517 entidades.
- **Vectores del piloto reutilizados.** Los vectores de la muestra de 2.000 no se recalcularon: `embed_chunks --reuse sample_10000` los tomó de la de 10.000 por el hash del texto de cada chunk. El 100 % se reutilizó (7.517 de B y 6.659 de A), porque la muestra de 2.000 está dentro de la de 10.000 y la segmentación es determinista.
- **Vectorización de la estrategia A:** 33.431 chunks en 36,5 minutos (15,3 chunks/s, 2,1 GB de RAM pico). Sus chunks son algo más largos que los de B: mediana de 228 tokens de granite, frente a 198.

## Verificación

1. **Conteos** (`tables/conteos.csv`): las 5 colecciones tienen exactamente tantas entidades como chunks sus Parquet.
2. **Persistencia** (`tables/persistencia.csv`): tras `docker compose restart milvus`, las 5 colecciones conservan sus conteos y devuelven los mismos 10 primeros resultados para 5 consultas fijas.
3. **Filtros** (`tables/filtros.csv`). Los 6 filtros probados devuelven solo entidades que los cumplen:
   - idioma húngaro;
   - solo Bluesky;
   - prensa en español;
   - `array_contains` sobre `provenance_ids`;
   - las ventanas de fechas de Q51 y Q52.
4. **Métricas en dev, Milvus frente a búsqueda exacta** (`tables/metricas_dev.csv`, 15 preguntas, `ef = 64`). Son iguales en las dos estrategias: el índice no pierde nada a nivel documento en estas preguntas.

| Colección | Recall@10 | MRR | nDCG@10 | Evidencia@10 |
|---|---|---|---|---|
| `sent256_s10k` (B) | 0,933 | 0,734 | 0,764 | 0,867 |
| `win256_s10k` (A) | 0,933 | 0,734 | 0,780 | 0,933 |

5. **API:** responde `/ask` con granite y `sent256_s10k`, con las fuentes, su idioma, su fecha y su procedencia.

**Primera comparación extrínseca de las estrategias de segmentación.** A y B empatan en Recall@10 y MRR en dev. A queda algo arriba en nDCG@10 y en acierto de evidencia, pero en 15 preguntas una sola equivale a 6,7 puntos, así que la diferencia no es concluyente. La comparación que cuenta es la del split `test` en la evaluación, con las dos colecciones ya listas.

## Decisiones que implica

| # | Decisión | Motivo |
|---|---|---|
| 1 | Cinco colecciones: piloto de 2.000 y muestra de 10.000 con A y B, más la adversarial | El alcance acordado; la ablación A/B queda lista para la evaluación |
| 2 | HNSW (M = 16, efConstruction = 200, COSINE) con `ef = 64` calibrado | Recall@10 de 0,996 frente a la búsqueda exacta, a menos de 1 ms por consulta |
| 3 | Campos filtrables con idioma completado, fechas en época UTC con su precisión y procedencia como lista | Los filtros de idioma, colección y fecha que piden las fases siguientes y el gold (Q51, Q52) |
| 4 | Ingesta idempotente por lotes, con reintentos, progreso para retomar y verificación de conteos | Cargas repetibles y sin duplicados |
| 5 | Vectores del piloto reutilizados por hash de texto | Evita recalcular lo que ya existe; mismo texto y mismo encoder producen el mismo vector |
| 6 | La API usa granite, `sent256_s10k` y `ef` desde `.env`; la colección la crea la ingesta, no la API | La API solo busca; el esquema y los índices quedan en un solo lugar |
| 7 | Se elimina la ingesta piloto por caracteres | La reemplazan las estrategias A y B por tokens |

## Limitaciones

- **Sin corpus completo:** se indexaron las muestras de 2.000 y 10.000, que es el alcance acordado. Indexar el corpus completo con B costaría unas 10 horas de vectorización.
- **Métricas indicativas:** las de dev salen de 15 preguntas de un gold en borrador.
- **Fecha de los duplicados:** el filtro de fecha usa la fecha del documento canónico de cada chunk. Si un texto duplicado aparece en documentos de fechas distintas, el resto de procedencias no se filtra por su propia fecha. Afecta a pocos chunks: en `sent256_s10k`, 387 tienen más de una procedencia y 176 de ellos (el 0,5 % de la colección) tienen procedencias de días distintos.
- **Latencias de esta máquina:** las de la calibración son de Apple Silicon con 10 núcleos asignados a Docker.
