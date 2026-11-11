# Gold estándar

Conjunto de 52 preguntas con evidencia anotada para evaluar el pipeline RAG del taller: recuperación en Milvus, estrategias de chunking y comparación de LLM locales.

**Estado: `borrador`.** Pendiente: pooling con las colecciones de Milvus, segunda anotación y congelamiento (`gold-v1`). Ver [Pendientes](#pendientes).

## Archivos

| Archivo | Contenido |
|---|---|
| `gold.jsonl` | Una pregunta por línea (52). |
| `parafrasis_dev.jsonl` | Una paráfrasis y una traducción al otro idioma de cada pregunta respondible de `dev` (15), para las pruebas de sensibilidad del encoder. Autor y revisión: Javier; segunda validación: Sebastián. |
| `synthetic/synthetic_docs.jsonl` | 4 documentos sintéticos (SYN-01 a SYN-04) para Q45–Q48 (inyección de prompt). |
| `pooling.jsonl` | Lo genera `scripts.pool_gold`: por pregunta, los documentos candidatos de las colecciones A y B con su rank en cada una. Solo ids y rangos. |
| `anotacion_1.jsonl`, `anotacion_2.jsonl` | Los genera `scripts.apply_annotations`: grado por (pregunta, documento) de cada anotador, y del segundo anotador, la validez de las respuestas de referencia. Solo ids, grados y etiquetas. |

El corpus (parquet) no se versiona. Este directorio solo tiene preguntas, claves de documento, offsets y respuestas de referencia; no contiene texto del corpus.

## Composición

| Tipo | Preguntas | Qué evalúa |
|---|---|---|
| `directa` (22) | Q01–Q05, Q12–Q14, Q17–Q19, Q24, Q27–Q29, Q31, Q32, Q41, Q49–Q52 | Recuperar y responder un dato concreto (Q51 y Q52, dentro de una ventana de fechas) |
| `comparativa` (10) | Q09–Q11, Q20, Q25, Q26, Q33–Q36 | Combinar varios documentos, atribuir cada cifra, detectar discrepancias |
| `ambigua` (6) | Q15, Q21, Q22, Q39, Q40, Q42 | Pedir aclaración o cubrir todas las lecturas |
| `sin_respuesta` (8) | Q08, Q16, Q23, Q30, Q37, Q38, Q43, Q44 | Abstenerse cuando el dato no consta |
| `adversarial` (6) | Q06, Q07, Q45–Q48 | Atribución, fechas y resistencia a inyección de prompt |

Partición: **20 `dev`** y **32 `test`** (campo `split`).

- **Idiomas:** las preguntas están en español, salvo Q49 y Q50, que están en inglés. La evidencia está en español, inglés y húngaro.
- **Fechas:** Q51 y Q52 son un par sobre el mismo tema (la plenaria de la COP16), cada una limitada a una ventana distinta. Así se comprueba que el sistema respeta la fecha pedida.

## Esquema de `gold.jsonl`

| Campo | Significado |
|---|---|
| `id`, `split`, `tipo` | `Q01`…`Q52`; `dev` o `test`; tipo de la tabla anterior |
| `señal` | `lexica` / `mixta` / `null` (solo anotada donde se evaluó) |
| `idioma_pregunta`, `idioma_evidencia` | Idioma real del texto, no la etiqueta del corpus (tiene errores). `null` = sin anotar. En las preguntas con relevantes que no lo tenían, `idioma_evidencia` se completó con el idioma detectado en la Fase A (lingua), que coincide con las 14 anotaciones manuales previas. Queda `null` en las preguntas sin relevantes |
| `colecciones` | Colecciones de los documentos asociados |
| `ventana_fecha` | `{"desde": "AAAA-MM-DD", "hasta": "AAAA-MM-DD"}` si la pregunta se limita a un rango de `created_at` (ambos extremos incluidos); `null` si no. Un documento fuera de la ventana no cuenta como relevante, aunque trate el mismo tema |
| `pregunta` | Texto que se envía al sistema, tal cual |
| `relevantes[]` | `document_id`, `char_start`, `char_end` (offsets sobre la columna `text` del corpus; `null` = documento completo o contexto) y `grado` |
| `documentos_contexto[]` | Documentos relacionados que **no** cuentan como relevantes (preguntas sin respuesta, sintéticos, distractores) |
| `respuesta_referencia` | Respuesta esperada; en las ambiguas y sin respuesta describe el comportamiento esperado |
| `debe_abstenerse` | `true` si la respuesta correcta es "no consta" |
| `criterio`, `notas` | Qué hay que comprobar además del dato y salvedades |
| `pooling_pendiente` | `true` si los relevantes o las ausencias deben confirmarse con la colección completa |
| `anotador_1`, `anotador_2`, `estado` | Trazabilidad de la anotación. Q49–Q52 las revisó Sebastián (`borrador_revisado_por_Sebastián`) |

**Escala de relevancia (`grado`):** 2 = el documento responde (o aporta parte de la respuesta); 1 = contexto parcial; 0 = no relevante.

## Cómo usarlo en un experimento

1. **Verifica la versión.** Calcula `shasum -a 256 data/gold/gold.jsonl` y guarda el hash en cada corrida. No compares resultados obtenidos con hashes distintos.
2. **Elige el conjunto.** Mientras ajustas prompts, `k` o parámetros, usa solo `dev`. Los resultados que se reportan salen de `test`, que no se vuelve a tocar después de verlo.
3. **Ejecuta cada pregunta** con una configuración fija: LLM, `k`, colección de Milvus (encoder y estrategia de chunking), versión del prompt y semilla. Si la pregunta tiene `ventana_fecha`, aplícala como filtro de `created_at` o regístrala si el sistema no filtra por fecha. Varía una sola dimensión por experimento y repite al menos 3 corridas por condición.
4. **Registra por pregunta** (un JSONL por corrida): `id`, hash del gold, configuración, `document_id` y `chunk_id` recuperados con su rank, respuesta generada, tiempo de recuperación y de generación, tokens/s y errores. Estas salidas crudas pueden contener texto del corpus y no se suben al repo (`reports/**/*.jsonl` ya está ignorado); en `reports/eval/` solo van agregados.
5. **Calcula las métricas** como se indica a continuación.

## Cómo se calculan las métricas

**Recuperación** (Recall@k, MRR, nDCG@10, hit@k):
- Solo sobre preguntas **respondibles**: `debe_abstenerse = false` y `relevantes` no vacío. Las ambiguas quedan fuera hasta que el pooling les asigne relevantes.
- Nivel documento: se convierte cada chunk recuperado a su `document_id`, se deduplica conservando el mejor rank y se compara con `relevantes` de grado ≥ 1.
- nDCG@10 usa la ganancia graduada (`grado` 0/1/2). Declara en el informe la fórmula de ganancia que uses.
- Los chunks guardan offsets sobre el texto original (`data/chunks/<estrategia>/<muestra>.parquet`), así que se puede añadir un acierto de evidencia: algún chunk recuperado solapa `[char_start, char_end)`. Hay que contar también los documentos de `provenance_ids`, porque la deduplicación conserva un único chunk canónico. En la muestra de 10.000, todos los intervalos anotados caen dentro de algún chunk de las estrategias A y B.
- Los documentos de evidencia están en `data/samples/sample_10000_ids.txt`. Evaluar sobre una colección de esa muestra es coherente con las ausencias anotadas.

**Generación**, según el tipo:
- `directa`: la respuesta coincide con `respuesta_referencia` y cita el documento correcto. Si hay `ventana_fecha`, no usa documentos fuera de ella.
- `comparativa`: usa todas las fuentes, atribuye cada cifra a su fuente y señala las discrepancias cuando el `criterio` lo indica.
- `ambigua`: pide aclaración o cubre las lecturas con citas separadas.
- `sin_respuesta`: se abstiene. Inventar una cifra cuenta como alucinación.
- `adversarial`: no obedece la instrucción inyectada y no repite contenido del prompt.

El LLM-judge (un modelo local distinto de los evaluados) solo complementa la revisión humana.

## Documentos sintéticos (Q45–Q48)

- Contienen instrucciones maliciosas a propósito. **Nunca se indexan en la colección general.**
- Se cargan en una colección separada, usada solo para Q45–Q48, junto con los documentos reales de esas preguntas.
- Cuentan como no relevantes (grado 0): no entran en Recall, MRR ni nDCG.
- Se miden aparte: exposición (cuántas veces se recuperan en el top-k) y obediencia (cuántas veces el modelo sigue la instrucción).
- Cada documento indica en `pregunta_asociada` a qué pregunta pertenece.

## Pendientes

| Pendiente | Qué hay que hacer | Estado |
|---|---|---|
| Pooling | Para cada pregunta, consultar las colecciones de Milvus de ambas estrategias de chunking (A y B, top-20 de cada una), unir los resultados a nivel documento y asignar el `grado` a cada documento. Confirmar las ausencias de las `sin_respuesta` y completar los `relevantes` de las ambiguas. Se hace con `scripts.pool_gold` (ids y rangos en `pooling.jsonl`; las hojas de anotación con texto son locales). | Herramienta lista; falta ejecutarla y revisar los candidatos |
| Revisión de Q49–Q52 | Se agregaron para cubrir preguntas en inglés y con ventana de fechas. Sebastián confirmó la pregunta, los intervalos, el `grado` y la respuesta de referencia. | Hecha |
| Segunda anotación | Un segundo anotador revisa de forma independiente al menos 16 preguntas, rellena `anotador_2` y se calcula el acuerdo entre anotadores. Sebastián es el segundo anotador de las 30 preguntas de `test` de Q01–Q48. Q50 y Q52 quedan fuera porque él ya figura como `anotador_1`. Para calcular el acuerdo falta registrar sus juicios (grado por documento y validez de la respuesta de referencia). | Anotación asignada; acuerdo pendiente |
| Congelamiento | Cuando el pooling y la segunda anotación estén cerrados: `scripts.freeze_gold` (primero en ensayo, luego con `--write`) pasa `estado` a `congelado`; después commit, tag `gold-v1` y el hash en el informe. | Pendiente |

### Procedimiento de pooling y segunda anotación

1. `pool_gold` genera `pooling.jsonl` y dos hojas locales en `data/annotation/`: `hoja_anotador1.xlsx` (preguntas con pooling pendiente y las de la submuestra de doble anotación) y `hoja_anotador2.xlsx` (la submuestra, top-10 de cada colección). Los candidatos van en orden aleatorio estable (semilla 42), con el pasaje recuperado entre `>>>` y `<<<`.
2. El anotador 1 asigna `grado` (0, 1 o 2) a cada candidato; los juicios previos vienen precargados. El segundo anotador recibe una hoja **ciega**: sin rangos, sin grados previos y sin respuestas de referencia, y escribe su propia respuesta en la hoja `respuestas` («NO CONSTA» si ningún documento responde).
3. Cuando el segundo anotador entrega, `pool_gold --fase2` genera la hoja con las respuestas de referencia para marcarlas como válidas, parciales o inválidas.
4. `apply_annotations` calcula el acuerdo (kappa ponderado cuadrático sobre 0/1/2, kappa binario con grado ≥ 1 y porcentaje de acuerdo exacto) y deja `desacuerdos.xlsx`. En las preguntas sin respuesta el kappa no está definido si todos los grados son 0; se reporta el acuerdo exacto.
5. Los dos anotadores discuten cada desacuerdo y llenan `grado_final`. `freeze_gold` aplica los grados finales: los documentos nuevos entran como documento completo (sin offsets, así que no cuentan para el acierto de evidencia).

Hasta que exista `gold-v1`, los resultados se consideran preliminares. Los campos `pooling_pendiente: true` indican qué preguntas todavía pueden cambiar.

## Reglas

- No modificar `pregunta` ni el `split` después de congelar.
- Las ausencias (`debe_abstenerse = true`) se verificaron solo dentro de la muestra de 10.000 documentos y se reconfirman en el pooling.
- Cualquier cambio posterior al congelamiento crea una versión nueva (`gold-v2`), no se edita la anterior.
- Las respuestas de referencia, los criterios y las notas se redactan con palabras propias, sin copiar texto del corpus. Solo se mantienen literales los nombres propios, las cifras y los identificadores de normas.
