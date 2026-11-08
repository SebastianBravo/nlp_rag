# Gold estándar (borrador)

Conjunto de 48 preguntas para evaluar los pipelines RAG (Solr vs Milvus y comparación de LLM).
**Estado: `borrador`.** Falta el pooling con Solr y Milvus, la segunda anotación y el congelamiento (`gold-v1`).

## Archivos

- `gold.jsonl`: una pregunta por línea.
- `synthetic/synthetic_docs.jsonl`: 4 documentos sintéticos (SYN-01 a SYN-04) para las preguntas Q45–Q48 (inyección de prompt). Se indexan en una colección aparte y nunca en el índice de recuperación general.

## Esquema de `gold.jsonl`

| Campo | Significado |
|---|---|
| `id`, `split` | `Q01`…`Q48`; `dev` (18) o `test` (30) |
| `tipo` | `directa`, `comparativa`, `ambigua`, `sin_respuesta`, `adversarial` |
| `señal` | `lexica` / `mixta` / `null` (solo anotada donde se evaluó) |
| `idioma_pregunta`, `idioma_evidencia` | idioma real del texto, no la etiqueta del corpus (que tiene errores); `null` = sin anotar |
| `colecciones` | colecciones de los documentos asociados |
| `relevantes[]` | `document_id`, `char_start`, `char_end` (offsets sobre la columna `text`), `grado` (2 = responde, 1 = parcial/contexto, 0 = no relevante) |
| `documentos_contexto[]` | documentos relacionados que no cuentan como relevantes para Recall/MRR/nDCG (por ejemplo, los de preguntas sin respuesta o los sintéticos) |
| `respuesta_referencia`, `debe_abstenerse`, `criterio`, `notas` | ver preguntas |
| `pooling_pendiente` | `true` si los relevantes o las ausencias se deben confirmar con todo el índice |
| `anotador_1`, `anotador_2`, `estado` | trazabilidad de la anotación |

## Reglas

- Las métricas de recuperación (Recall@k, MRR, nDCG@10, hit@k) se calculan **solo** sobre preguntas respondibles (`debe_abstenerse = false` y con `relevantes` no vacío).
- Los hiperparámetros y prompts se ajustan **solo con `dev`**. El `test` no se toca después de verlo.
- Las ausencias (`debe_abstenerse = true`) se verificaron solo dentro de la muestra de 10.000 documentos; se reconfirman en el pooling.
- El corpus (parquet) no se publica en el repositorio.

## Integridad

Al congelar: `shasum -a 256 data/gold/gold.jsonl` y registrar el hash en el informe y en cada corrida de evaluación.
