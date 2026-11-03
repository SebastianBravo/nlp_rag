# Segmentación del corpus (Fase B)

Este resumen compara las dos estrategias de segmentación antes de indexar en Milvus. Todas las cifras salen de `runs/*.json` y de `tables/comparacion_<muestra>.csv`, que regeneran:

```bash
docker compose --profile tools run --rm pipeline python -m scripts.build_chunks --strategy A --sample data/samples/sample_10000_ids.txt
docker compose --profile tools run --rm pipeline python -m scripts.build_chunks --strategy B --sample data/samples/sample_10000_ids.txt
```

**Entradas:**
- el Parquet original, que no se modifica;
- las muestras de `data/samples/`;
- dos artefactos de la Fase A: `data/eda/document_features.parquet` (idioma detectado y mojibake) y `data/eda/boilerplate_lines.parquet`;
- los parámetros de `configs/chunking.yaml`.

Los chunks, con su texto, quedan en `data/chunks/<estrategia>/<muestra>.parquet`, que no se versiona. Los reportes solo contienen métricas.

## Reglas comunes

- **Offsets sobre el original:** cada chunk es un intervalo continuo del texto original, y siempre se cumple `text[char_start:char_end] == chunk`. Se verificó en todos los chunks de ambas muestras.
- **Tokens:** se miden con el tokenizador del encoder (`app.embeddings`). El presupuesto es de 256 tokens de contenido; el prefijo `passage: ` y los tokens especiales se suman al medir `n_tokens`. Si algún chunk superara 512, la ejecución falla.
- **Chunk único:** un documento o tramo que cabe en el presupuesto queda como un solo chunk.
- **Boilerplate:** las líneas marcadas en la Fase A se saltan como intervalos. El texto no se reescribe, solo se deja fuera de los chunks.
- **Fragmentos:** en documentos con varios chunks se descartan los de menos de 8 tokens.
- **Páginas plantilla:** se excluyen los textos de prensa repetidos 100 veces o más.
- **Mojibake:** se conserva y se marca con `has_mojibake`.
- **Identificador:** `chunk_id = sha256(document_id|estrategia|versión|char_start|char_end)[:32]`. Repetir una ejecución produce exactamente los mismos chunks e IDs.
- **Deduplicación:** es exacta por `chunk_sha256`. Se conserva el chunk del menor `(document_id, char_start)` y `provenance_ids` guarda todos los documentos que lo contienen.

## Estrategias

- **A · ventana de tokens:** ventanas de 256 tokens con 32 de solapamiento. Corta en cualquier frontera de token.
- **B · estructura:**
  - toma párrafos (líneas) y oraciones, separadas con nltk punkt según el idioma del documento;
  - las agrupa hasta 256 tokens y repite una oración entre chunks consecutivos;
  - si el grupo se cortaría a mitad de un párrafo, prefiere cerrar el chunk en un cambio de párrafo, siempre que conserve al menos la mitad del presupuesto;
  - solo corta a mitad de oración cuando una oración sola supera el presupuesto.

## Resultados (muestra de 10.000 documentos)

| Métrica | A · ventana | B · estructura |
|---|---|---|
| Chunks antes de deduplicar | 34.025 | 38.692 (+13,7 %) |
| Chunks guardados tras deduplicar | 33.431 | 37.919 |
| Chunks por noticia (media / p50 / p95) | 4,9 / 4 / 11 | 5,7 / 5 / 13 |
| Tokens por chunk de prensa (p5 / p50 / p95 / máx.) | 75 / 260 / 261 / 261 | 107 / 222 / 257 / 261 |
| Chunks que superan 512 tokens | 0 | 0 |
| Chunks que cortan una oración (prensa) | 93,7 % | 0,14 % |
| Texto repetido por solapamiento (prensa) | 12,0 % | 16,5 % |
| Chunks eliminados por duplicados (total / prensa) | 594 (1,7 %) / 289 (1,0 %) | 773 (2,0 %) / 468 (1,3 %) |
| Tiempo de segmentación | 9,7 s | 9,4 s |

**Común a ambas estrategias:**
- **Bluesky:** los 3.821 posts quedan como un chunk cada uno (máximo 144 tokens). Se eliminan 305 duplicados (8,0 %).
- **Prensa:**
  - se excluyen 63 páginas plantilla;
  - el boilerplate saca el 7,8 % de los caracteres;
  - los chunks cubren el 92,1 % del texto; el resto es boilerplate, fragmentos descartados (unos 1.600) y espacios;
  - 3 documentos no producen chunks porque todas sus líneas son boilerplate.
- **Piloto de 2.000 documentos:** da proporciones casi iguales. A corta oraciones en el 83,0 % de los chunks y B en el 0,03 %, y B produce un 12,8 % más chunks.

Figuras: `tokens_sample_10000.png`, `chunks_por_documento_sample_10000.png` y `metricas_sample_10000.png`, más las equivalentes del piloto.

## Lectura

- **Ninguna estrategia trunca.** El chunk más largo tiene 261 tokens con prefijo, la mitad del tope de 512 del encoder.
- **A produce chunks de tamaño casi constante**, la mayoría en 260 tokens. A cambio, el 94 % de los chunks de prensa empieza o termina a mitad de oración: una cita textual o un dato quedan partidos entre dos chunks.
- **B mantiene las oraciones enteras**, a costa de un 13,7 % más de chunks, tamaños más variables (p5 de 107 tokens) y más texto repetido por el solapamiento de oraciones completas.
- **B elimina más duplicados en prensa (468 frente a 289).** Una explicación plausible es que dos noticias con párrafos iguales, por ejemplo sindicadas, producen chunks idénticos cuando el corte sigue la estructura. Con ventanas de tokens, una diferencia al inicio desplaza todos los cortes. Es una hipótesis que no se verificó.
- **Las métricas intrínsecas no dicen cuál recupera mejor.** Esa comparación es extrínseca (Recall@k con el gold estándar) y se hace en la fase de evaluación, con cada estrategia indexada en su propia colección de Milvus.

## Ejemplo ilustrativo

`ejemplos/ejemplo_A.json` y `ejemplos/ejemplo_B.json` segmentan un texto **sintético** (`configs/ejemplo_segmentacion.txt`, que no es del corpus). Con A, los 3 chunks cortan oraciones; con B, ninguno. El notebook `02_segmentacion` muestra las fronteras resaltadas.

## Decisiones que implica

| # | Decisión | Motivo |
|---|---|---|
| 1 | Bluesky va como chunk único en ambas estrategias | Ningún post supera el presupuesto |
| 2 | Cada estrategia va a su propia colección de Milvus: `rag_e5small_win256_v1` (A) y `rag_e5small_sent256_v1` (B) | Nunca se mezclan vectores de dos segmentaciones |
| 3 | B es la estrategia candidata por defecto para el corpus completo, sujeta a la comparación extrínseca | Conserva oraciones enteras, útil para citar evidencia, con un costo moderado (+13,7 % chunks) |
| 4 | La ingesta (Fase D) lee estos Parquet de chunks, no el corpus crudo | Los chunks, sus IDs y sus procedencias quedan fijados antes de indexar |

## Limitaciones

- **Húngaro:** punkt no tiene modelo de húngaro (3,9 % de la prensa) y se usa el de inglés, así que sus fronteras de oración son menos fiables.
- **Métrica de corte de oraciones:** usa punkt como referencia, que es el mismo separador que usa B. B queda favorecida por construcción, y la cifra de A se lee frente a ese mismo criterio.
- **Boilerplate:** la heurística (línea repetida en el dominio) viene de la Fase A. En 4 chunks de B aparece una oración igual a una línea boilerplate de otros documentos, pero dentro de una línea más larga que no lo es.
- **Alcance:** solo se segmentaron las muestras. El corpus completo se segmenta en la Fase D con la estrategia elegida; al ritmo medido se estiman unos 4 minutos.
