# Encoder del corpus (Fase C)

Este resumen justifica el encoder que vectoriza el corpus y describe cómo se comporta sobre los datos del taller. Las cifras salen de `runs/*.json`, `tables/*.csv` y `analisis_encoder.json`, que regeneran:

```bash
docker compose --profile tools run --rm pipeline python -m scripts.benchmark_encoders
docker compose --profile tools run --rm pipeline python -m scripts.embed_chunks --encoder granite-97m --strategy B --sample sample_10000
docker compose --profile tools run --rm pipeline python -m scripts.encoder_report --encoder granite-97m --strategy B --sample sample_10000
```

Los perfiles de encoder, con la revisión de Hugging Face fijada, están en `configs/encoders.yaml`. Todo se ejecutó en el contenedor `pipeline`: solo CPU, 10 núcleos y 8 GB de RAM compartidos con Milvus y Ollama.

## Decisión

**Encoder elegido:** `ibm-granite/granite-embedding-97m-multilingual-r2` (revisión `835ad14`):

- 97M parámetros y vectores de 384 dimensiones;
- pooling CLS y normalización L2 incluidos en el propio modelo;
- similitud coseno;
- sin prefijos ni instrucciones en consultas ni pasajes.

**Criterio de selección.** En este proyecto la restricción dura es el costo en CPU y RAM. Por eso se buscó el encoder con mejor recuperación multilingüe publicada entre los que permiten vectorizar el corpus completo en una noche dentro del contenedor y dejan memoria libre para Milvus y el LLM.

| | multilingual-e5-small | **granite-97m-r2** | Qwen3-Embedding-0.6B |
|---|---|---|---|
| Parámetros | 117,7M | 97,4M | 595,8M |
| Dimensión / memoria por 100.000 vectores | 384 / 154 MB | 384 / 154 MB | 1024 / 410 MB |
| Velocidad medida (chunks/s) | 32,7 | 19,5 | 0,78 |
| Horas por 100.000 chunks | 0,85 | 1,43 | 35,6 |
| RAM pico del proceso (lote de 32) | 1,5 GB | 1,9 GB | 7,0 GB |
| Contexto máximo (tokens) | 512 | 32.768 | 32.768 |
| Recuperación MTEB multilingüe, publicada | 50,9 | 60,3 | 64,64 |
| Licencia | MIT | Apache-2.0 | Apache-2.0 |

Velocidad y RAM vienen de `tables/costo_encoders.csv`: los mismos 256 chunks, el mismo lote y un proceso por modelo. Los puntajes de recuperación son los publicados:

- e5-small y granite: tabla de 18 tareas de MTEB(Multilingual, v2) del anuncio de Granite R2 (IBM);
- Qwen3: su ficha en Hugging Face.

No son una medición propia.

**Por qué granite y no los otros:**

- **Frente a e5-small, el encoder por defecto anterior:**
  - misma dimensión, así que Milvus ocupa la misma memoria;
  - 9,4 puntos más de recuperación multilingüe publicada;
  - entrenamiento explícito de recuperación y entre idiomas en 52 idiomas, que incluyen español, inglés y húngaro;
  - contexto de 32K tokens, sin riesgo de truncamiento;
  - no depende de prefijos.

  Su costo es 1,7 veces el de e5-small, que sigue siendo asumible.
- **Qwen3-Embedding-0.6B tiene el mejor puntaje publicado, pero no es viable aquí:**
  - vectorizar la muestra de 10.000 tomaría unas 13 horas y el corpus completo, cientos;
  - con lote de 32 su proceso llegó a 7 GB, casi toda la RAM de Docker.
- **Otros candidatos revisados, sin medir:**
  - multilingual-e5-base, gte-multilingual-base y EmbeddingGemma-300m son más pesados, con 86M a 113M parámetros sin contar la tabla de embeddings, frente a 28M de granite. Además gte y e5-base tienen menor recuperación publicada, y EmbeddingGemma pide aceptar términos;
  - BGE-M3 tiene 568M parámetros;
  - jina-embeddings-v5-text-nano tiene licencia no comercial.

**Hallazgo técnico.** Con transformers 5, los modelos se cargan por defecto en el formato de sus pesos. En una prueba de diagnóstico, Qwen3 en bfloat16 fue unas 16 veces más lento que en float32 en la CPU del contenedor. `app/embeddings.py` carga ahora todos los modelos en float32.

## Vectorización de la muestra de 10.000 (estrategia B)

- **Volumen y costo:**
  - 37.919 chunks en 34 minutos (18,6 chunks/s);
  - RAM pico de 2,1 GB;
  - 384 dimensiones, todas con norma 1.
- **Sin truncamiento:** con el tokenizador de granite, los chunks tienen una mediana de 198 tokens, un p95 de 250 y un máximo de 511, muy lejos del tope de 32.768.
- **Chunks fijos:** los chunks son los de la Fase B. Su tamaño se midió con el tokenizador de e5-small, que queda fijado para la segmentación: cambiar de encoder no cambia los chunks.
- **Salida:** `data/embeddings/granite-97m/B/sample_10000.parquet`, que no se versiona. Se calcula por partes y se puede reanudar.
- **Corpus completo (estimación):** a este ritmo, unos 650.000 chunks tardarían unas 10 horas (1,43 h por cada 100.000), y los vectores ocuparían 1 GB.

## Comportamiento en `dev`

Se usaron las 15 preguntas respondibles del split `dev` del gold, que todavía es borrador; `test` no se tocó. La búsqueda es exacta por coseno sobre los 37.919 chunks, y la evaluación es a nivel documento: cada chunk cuenta para su documento y los de su procedencia, con su mejor rank. Con 15 preguntas, una pregunta equivale a 6,7 puntos, así que estas cifras son indicativas.

| Variante de consulta | Recall@5 | Recall@10 | MRR | nDCG@10 | Evidencia@10 |
|---|---|---|---|---|---|
| Pregunta tal cual (formato del perfil) | 0,800 | 0,933 | 0,734 | 0,764 | 0,867 |
| Con prefijo `query: ` | 0,800 | 0,933 | 0,746 | 0,765 | 0,867 |
| Sin filtro de fecha | 0,800 | 0,933 | 0,734 | 0,764 | 0,867 |
| Paráfrasis | 0,867 | 0,867 | 0,687 | 0,728 | 0,867 |
| Traducción al otro idioma | 0,650 | 0,833 | 0,581 | 0,622 | 0,867 |

"Evidencia@10" es la proporción de preguntas en las que uno de los 10 primeros chunks contiene el intervalo anotado.

- **Formato de la consulta:** agregar el prefijo de E5 casi no cambia el resultado. granite no depende del formato, y se mantiene sin prefijo, como indica su ficha.
- **Fechas:** el filtro no cambió nada en dev, porque el encoder ya pone en primer lugar el documento dentro de la ventana de Q51. El filtro se mantiene para la Fase D: es la garantía cuando el tema se repite en otras fechas.
- **Paráfrasis y traducción:**
  - el vector de la consulta cambia poco: coseno medio de 0,969 con la paráfrasis y de 0,944 con la traducción;
  - el ranking cambia más: el Jaccard medio del top-10 de documentos es 0,62 y 0,49;
  - la traducción es la que más afecta el orden (MRR de 0,734 a 0,581), aunque el Recall@20 sigue en 1,0.
- **Entre idiomas** (pregunta original; `tables/metricas_por_idioma.csv`):
  - las 4 preguntas con evidencia en inglés llegan a Recall@10 = 1,0, y tres de ellas son preguntas en español;
  - las 10 con evidencia en español, a 0,9;
  - la única con evidencia en húngaro (Q04) encuentra su documento en el puesto 3.
- **Errores:**
  - Q01 pregunta por unas "reflexiones finales sobre la COP16", y el chunk con la evidencia queda en el puesto 51, porque la muestra tiene muchos documentos sobre la COP16;
  - en Q04, el documento correcto aparece en el puesto 3, pero por otro de sus chunks: el chunk con la evidencia queda en el puesto 11.

  Son casos para revisar con la recuperación híbrida o el reranking de la Fase E.
- **Similitudes comprimidas:** los puntajes de coseno de granite se concentran en una franja estrecha. En los 5 primeros documentos de cada pregunta van de 0,79 a 0,97, con mediana 0,86 (`tables/vecinos_ejemplo.csv`). Por eso no conviene fijar umbrales absolutos de similitud; lo informativo es el orden.

## Dimensión y normalización

| Dimensiones | Renormalizado | Recall@10 | nDCG@10 | Memoria por 100.000 vectores |
|---|---|---|---|---|
| 384 | sí | 0,933 | 0,764 | 154 MB |
| 256 | sí / no | 0,867 / 0,867 | 0,738 / 0,751 | 102 MB |
| 128 | sí / no | 0,733 / 0,800 | 0,680 / 0,667 | 51 MB |

- **Se guardan las 384 dimensiones.** granite-97m no se entrenó con Matryoshka, y recortar ya cuesta en dev con 256 dimensiones. La memoria no es un problema: 1 GB para el corpus completo.
- **Normalización:** el modelo normaliza sus vectores, así que coseno y producto interno dan el mismo orden. En Milvus se usará la métrica COSINE. Sin renormalizar después de recortar, los vectores dejan de tener norma 1 y el orden cambia.

## Decisiones que implica

| # | Decisión | Motivo |
|---|---|---|
| 1 | Vectorizar el corpus con granite-97m-r2, 384 dimensiones, coseno, sin prefijos | Mejor recuperación multilingüe publicada entre los encoders viables en CPU, con el mismo tamaño de vector que e5-small |
| 2 | Descartar Qwen3-Embedding-0.6B | 35,6 h por cada 100.000 chunks y 7 GB de RAM en el contenedor |
| 3 | Cargar los modelos en float32 | En bfloat16 la CPU del contenedor fue unas 16 veces más lenta |
| 4 | Medir los chunks con el tokenizador de e5-small, fijado | Los chunks no dependen del encoder de vectorización |
| 5 | Guardar los vectores en Parquet por partes reanudables antes de Milvus | Separa la vectorización, que es la etapa más costosa, de la indexación |
| 6 | La API pasa a granite en la Fase D, con una colección nueva por estrategia (`rag_granite97m_sent256_v1` para B) | Nunca se mezclan vectores de dos encoders |
| 7 | No se compara la calidad de varios encoders con el gold; la elección se apoya en el costo medido y en benchmarks publicados | Por tiempo, por decisión del equipo |

## Limitaciones

- **Pocas preguntas:** las métricas salen de 15 preguntas de `dev` en un gold en borrador, así que son indicativas.
- **Puntajes no medidos aquí:** los de recuperación de MTEB son publicados por los autores (IBM, Qwen) y no se reprodujeron en este proyecto.
- **Sin ablación de encoder:** no hay comparación con un segundo encoder en este corpus.
- **Paráfrasis:** las paráfrasis y traducciones de `dev` (`data/gold/parafrasis_dev.jsonl`) las redactó y revisó Javier, con segunda validación de Sebastián.
- **Velocidades del contenedor:** dependen de la máquina (Apple Silicon, 10 núcleos asignados a Docker).
