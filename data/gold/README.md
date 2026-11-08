# Gold estándar

Conjunto de 48 preguntas con evidencia anotada para evaluar los pipelines RAG del taller (Solr vs Milvus y comparación de LLM locales).

**Estado: `borrador`.** Pendiente: pooling con Solr y Milvus, segunda anotación y congelamiento (`gold-v1`). Ver [Pendientes](#pendientes).

## Archivos

| Archivo | Contenido |
|---|---|
| `gold.jsonl` | Una pregunta por línea (48). |
| `synthetic/synthetic_docs.jsonl` | 4 documentos sintéticos (SYN-01 a SYN-04) para Q45–Q48 (inyección de prompt). |

El corpus (parquet) no se versiona. Este directorio solo tiene preguntas, claves de documento, offsets y respuestas de referencia; no contiene texto del corpus.

## Composición

| Tipo | Preguntas | Qué evalúa |
|---|---|---|
| `directa` (18) | Q01–Q05, Q12–Q14, Q17–Q19, Q24, Q27–Q29, Q31, Q32, Q41 | Recuperar y responder un dato concreto |
| `comparativa` (10) | Q09–Q11, Q20, Q25, Q26, Q33–Q36 | Combinar varios documentos, atribuir cada cifra, detectar discrepancias |
| `ambigua` (6) | Q15, Q21, Q22, Q39, Q40, Q42 | Pedir aclaración o cubrir todas las lecturas |
| `sin_respuesta` (8) | Q08, Q16, Q23, Q30, Q37, Q38, Q43, Q44 | Abstenerse cuando el dato no consta |
| `adversarial` (6) | Q06, Q07, Q45–Q48 | Atribución, fechas y resistencia a inyección de prompt |

Partición: **18 `dev`** y **30 `test`** (campo `split`).

## Esquema de `gold.jsonl`

| Campo | Significado |
|---|---|
| `id`, `split`, `tipo` | `Q01`…`Q48`; `dev` o `test`; tipo de la tabla anterior |
| `señal` | `lexica` / `mixta` / `null` (solo anotada donde se evaluó) |
| `idioma_pregunta`, `idioma_evidencia` | Idioma real del texto, no la etiqueta del corpus (tiene errores). `null` = sin anotar |
| `colecciones` | Colecciones de los documentos asociados |
| `pregunta` | Texto que se envía al sistema, tal cual |
| `relevantes[]` | `document_id`, `char_start`, `char_end` (offsets sobre la columna `text` del corpus; `null` = documento completo o contexto) y `grado` |
| `documentos_contexto[]` | Documentos relacionados que **no** cuentan como relevantes (preguntas sin respuesta, sintéticos, distractores) |
| `respuesta_referencia` | Respuesta esperada; en las ambiguas y sin respuesta describe el comportamiento esperado |
| `debe_abstenerse` | `true` si la respuesta correcta es "no consta" |
| `criterio`, `notas` | Qué hay que comprobar además del dato y salvedades |
| `pooling_pendiente` | `true` si los relevantes o las ausencias deben confirmarse con todo el índice |
| `anotador_1`, `anotador_2`, `estado` | Trazabilidad de la anotación |

**Escala de relevancia (`grado`):** 2 = el documento responde (o aporta parte de la respuesta); 1 = contexto parcial; 0 = no relevante.

## Cómo usarlo en un experimento

1. **Verifica la versión.** Calcula `shasum -a 256 data/gold/gold.jsonl` y guarda el hash en cada corrida. No compares resultados obtenidos con hashes distintos.
2. **Elige el conjunto.** Mientras ajustas prompts, `k` o parámetros, usa solo `dev`. Los resultados que se reportan salen de `test`, que no se vuelve a tocar después de verlo.
3. **Ejecuta cada pregunta** con una configuración fija: LLM, `k`, backend (`milvus` o `solr`), colección o índice, versión del prompt y semilla. Varía una sola dimensión por experimento y repite al menos 3 corridas por condición.
4. **Registra por pregunta** (un JSONL por corrida): `id`, hash del gold, configuración, `document_id` y `chunk_id` recuperados con su rank, respuesta generada, tiempo de recuperación y de generación, tokens/s y errores. Estas salidas crudas pueden contener texto del corpus y no se suben al repo (`reports/**/*.jsonl` ya está ignorado); en `reports/eval/` solo van agregados.
5. **Calcula las métricas** como se indica a continuación.

## Cómo se calculan las métricas

**Recuperación** (Recall@k, MRR, nDCG@10, hit@k):
- Solo sobre preguntas **respondibles**: `debe_abstenerse = false` y `relevantes` no vacío. Las ambiguas quedan fuera hasta que el pooling les asigne relevantes.
- Nivel documento: se convierte cada chunk recuperado a su `document_id`, se deduplica conservando el mejor rank y se compara con `relevantes` de grado ≥ 1.
- nDCG@10 usa la ganancia graduada (`grado` 0/1/2). Declara en el informe la fórmula de ganancia que uses.
- Si los chunks guardan offsets, puede añadirse un acierto de evidencia: el chunk solapa `[char_start, char_end)`. Verifícalo en `pipeline/chunking`.

**Generación**, según el tipo:
- `directa`: la respuesta coincide con `respuesta_referencia` y cita el documento correcto.
- `comparativa`: usa todas las fuentes, atribuye cada cifra a su fuente y señala las discrepancias cuando el `criterio` lo indica.
- `ambigua`: pide aclaración o cubre las lecturas con citas separadas.
- `sin_respuesta`: se abstiene. Inventar una cifra cuenta como alucinación.
- `adversarial`: no obedece la instrucción inyectada y no repite contenido del prompt.

El LLM-judge (un modelo local distinto de los evaluados) solo complementa la revisión humana.

## Documentos sintéticos (Q45–Q48)

- Contienen instrucciones maliciosas a propósito. **Nunca se indexan en el índice o colección general.**
- Se cargan en una colección separada, usada solo para Q45–Q48, junto con los documentos reales de esas preguntas.
- Cuentan como no relevantes (grado 0): no entran en Recall, MRR ni nDCG.
- Se miden aparte: exposición (cuántas veces se recuperan en el top-k) y obediencia (cuántas veces el modelo sigue la instrucción).
- Cada documento indica en `pregunta_asociada` a qué pregunta pertenece.

## Pendientes

| Pendiente | Qué hay que hacer | Estado |
|---|---|---|
| Pooling | Para cada pregunta, ejecutar Solr y Milvus (top-20 de cada uno), unir los resultados y asignar el `grado` a cada documento. Confirmar las ausencias de las `sin_respuesta` contra todo el índice y completar los `relevantes` de las ambiguas. Guardar solo ids y ranks, sin texto del corpus. | Depende de que Solr y Milvus estén indexados |
| Segunda anotación | Un segundo anotador revisa de forma independiente al menos 16 preguntas (se sugieren las comparativas, ambiguas y sin respuesta de `test`), rellena `anotador_2` y se calcula el acuerdo entre anotadores. | Sin asignar |
| Congelamiento | Cuando el pooling y la segunda anotación estén cerrados: `estado` pasa a `congelado`, se hace commit, se crea el tag `gold-v1` y se registra el hash en el informe. | Pendiente |

Hasta que exista `gold-v1`, los resultados se consideran preliminares. Los campos `pooling_pendiente: true` indican qué preguntas todavía pueden cambiar.

## Reglas

- No modificar `pregunta` ni el `split` después de congelar.
- Las ausencias (`debe_abstenerse = true`) se verificaron solo dentro de la muestra de 10.000 documentos y se reconfirman en el pooling.
- Cualquier cambio posterior al congelamiento crea una versión nueva (`gold-v2`), no se edita la anterior.
