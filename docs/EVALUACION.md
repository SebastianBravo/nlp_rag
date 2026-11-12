# Protocolo de evaluación requerido

1. Crear al menos 40 preguntas originales con diversidad de fuentes (Bluesky/prensa), idiomas, preguntas factuales y comparativas, preguntas sin evidencia y adversariales; dividir desarrollo y test, sin usar test para iterar prompts.
2. Anotar manualmente IDs de documento/chunk relevante y respuesta de referencia cuando sea posible; dos anotadores para una submuestra, resolver desacuerdos; no usar `keyword_family` como verdad semántica.
3. Fijar el mismo subconjunto, índice, embeddings, chunking, k, prompt y temperatura al comparar LLM. Registrar variantes y no confundir cambio de recuperación con cambio del LLM.
4. Métricas de recuperación: Recall@k, MRR, nDCG@k y hit@k. Generación: groundedness/faithfulness con rúbrica humana, corrección, abstención, citas válidas, relevancia, costo de RAM/CPU/GPU, tokens/s y p50/p95 de latencia.
5. Repetir al menos 3 veces cada condición, registrar arranque frío/caliente y variaciones, no afirmar significancia sin intervalos o análisis apropiado.
6. Evaluar contaminación por texto malicioso en documentos (prompt injection), referencias engañosas, duplicados, fechas y sesgos por plataforma.
7. Separar tablas de recuperación de tablas de generación. Adjuntar JSONL con pregunta, respuesta, evidencia, modelo, prompt versionado, configuración, tiempo y anotaciones.
8. Concluir con compromisos verificables: calidad, recursos, velocidad, trazabilidad, reproducibilidad y riesgos.

---

# Protocolo aplicado en este repositorio

Esta parte describe cómo se aplicó el protocolo anterior. Las decisiones de detalle están en `docs/decisiones.md` (desde la #79) y los parámetros en `configs/experimentos.yaml`.

## Diseño

- Gold congelado (`gold-v1`): 52 preguntas, 20 de desarrollo y 32 de test. Test es la partición principal; desarrollo es complementaria y es donde se eligió el prompt `rag_v1`.
- Experimento principal: cambia solo el LLM (`qwen2.5:1.5b`, `llama3.2:1b`, `llama3.2:3b`, `gemma2:2b`). La recuperación se hace una vez por pregunta y el contexto es idéntico para todos los modelos. Fijos: colección `sent256_s10k` (chunking B), k = 5, prompt `rag_v1`, temperatura 0, semilla 42.
- Ablaciones, una dimensión cada vez: `k3`, `k10`, `chunking_A`, `prompt_v4`, `contexto_top3`. `k3` y `contexto_top3` dan resultados idénticos (el contexto de k = 3 es el prefijo de k = 5), por lo que `contexto_top3` es redundante.
- Tres corridas por condición, con arranque frío y caliente registrados. Con temperatura 0 y semilla fija las corridas no son idénticas entre sí; sirven para medir variabilidad, no como réplicas independientes.
- Intervalos: bootstrap percentil del 95 % (B = 10.000, semilla 7) remuestreando preguntas. Las diferencias entre condiciones se calculan por pares de preguntas.

## Hipótesis

Redactadas a partir del diseño durante la elaboración del informe. No se registraron antes de lanzar la generación: lo que sí quedó fijado antes (condiciones, corridas, métricas y particiones) está en `configs/experimentos.yaml`, y el prompt se eligió en desarrollo sin usar test. Los resultados preliminares ya se conocían al redactarlas, así que se leen como hipótesis contrastadas, no como predicciones.

| # | Hipótesis | Resultado |
|---|---|---|
| H1 | Los modelos de 3B superan en corrección a los de 2B o menos | No confirmada en general: solo `llama3.2:1b` es claramente peor; `llama3.2:3b`, `gemma2:2b` y `qwen2.5:1.5b` no se distinguen con los intervalos disponibles |
| H2 | La latencia crece con el tamaño del modelo | Confirmada (`llama3.2:1b` < `qwen2.5:1.5b` < `gemma2:2b` < `llama3.2:3b`), ver `generacion_latencia.csv` |
| H3 | Un prompt que prioriza abstenerse reduce respuestas sin evidencia, a costa de abstenciones indebidas (`rag_v1` frente a `prompt_v4`) | Confirmada en las métricas automáticas; la calificación con jueces de `prompt_v4` queda como tentativa |
| H4 | El chunking A y el B no difieren de forma detectable en recuperación | Sin diferencia concluyente: los intervalos de la diferencia pareada A − B incluyen el cero |

## Calificación de las respuestas

- Métricas automáticas (abstención, citas válidas y relevantes, filtración de instrucciones, tiempos, memoria) sobre todas las celdas.
- Rúbrica v1 (corrección 0/1/2, fidelidad, citas, abstención, inyección) aplicada a las 128 respuestas de la condición principal, test, corrida 1, por dos jueces LLM de familias distintas, sin ver qué modelo produjo cada respuesta (ids opacos, orden barajado con semilla 7).
- Auditoría humana de 12 respuestas estratificadas por tipo. No hubo segundo evaluador humano, así que no hay acuerdo entre humanos: es una desviación del protocolo, que pide dos anotadores humanos.
- Acuerdo entre jueces y con la auditoría: acuerdo exacto, kappa ponderado cuadrático (ordinales) y kappa de Cohen (categóricas), en `juez_acuerdo.csv`. La fidelidad es la dimensión con menor acuerdo.

## Cumplimiento por punto

| Punto | Estado | Evidencia o desviación |
|---|---|---|
| 1 | Cumplido | 52 preguntas (≥ 40) con tipos directa, comparativa, sin respuesta, ambigua y adversarial; desarrollo y test separados |
| 2 | Parcial | Doble anotación de 16 preguntas de test con kappa y desacuerdos resueltos; el pooling y parte de la anotación se apoyaron en LLM |
| 3 | Cumplido | Contexto idéntico entre LLM por caché; una dimensión por ablación |
| 4 | Parcial | Recuperación completa; en generación, rúbrica por jueces LLM más auditoría humana de n = 12 en lugar de rúbrica humana completa; RAM/CPU/GPU registrados |
| 5 | Cumplido | 3 corridas, frío y caliente, intervalos bootstrap; sin afirmaciones de significancia sin intervalo |
| 6 | Parcial | Inyección evaluada solo en Q45–Q48 (3 en test); fecha en Q07; duplicados y fechas como parte del corpus; sesgo por plataforma (Bluesky frente a prensa) no desglosado |
| 7 | Cumplido | Tablas de recuperación y de generación separadas; JSONL con configuración, prompt versionado, tiempos y calificaciones. El JSONL con respuestas completas contiene texto del corpus y no se versiona |
| 8 | En el informe | Conclusiones con intervalos y amenazas a la validez declaradas |

## Amenazas a la validez

- Ollama nativo (GPU de Apple, memoria unificada) en la evaluación frente a Docker en CPU en las medidas previas de desarrollo; versiones 0.40.16 y 0.40.13. Las latencias no son comparables entre ambos entornos.
- Medida adicional en Docker, en CPU (`reports/eval/docker_cpu`): Ollama 0.40.16, VM de Docker Desktop con 12 CPU y 7,7 GB, solo `qwen2.5:1.5b` y `llama3.2:3b`, test y una corrida. La mediana de latencia total es 4,03 s y 8,44 s, frente a 1,24 s y 2,72 s en nativo (unas 3 veces más). La mayor parte está en procesar el prompt (unos 1.700 tokens): 3,51 s frente a 0,86 s con `qwen2.5:1.5b` y 7,30 s frente a 1,83 s con `llama3.2:3b`; generar la respuesta cambia mucho menos. En calidad, las diferencias (por ejemplo, cita relevante de `qwen2.5:1.5b` 0,214 frente a 0,296) caben en el ruido de 32 preguntas y 1 corrida frente a 3. La revisión de desarrollo, también en Docker, dio 7,8 s con `qwen2.5:1.5b` (10 núcleos, Ollama 0.40.13); con estos datos no se puede separar el efecto de los núcleos del de la versión.
- Python 3.13 en el entorno local y 3.12 en las imágenes.
- Los vectores de consulta salen de un entorno de torch distinto (MPS) del de la ingesta.
- Los anotadores juzgaron pasajes, no documentos completos; el pooling recupera 58 de 61 relevantes y no incluye los de Q01, Q26 y Q41.
- Un solo humano en la auditoría; los dos jueces pueden compartir sesgos; el juez vio cuatro respuestas por pregunta.
- Test tiene 3 preguntas de inyección y 4 sin respuesta: los resultados por tipo son anecdóticos.
- Hipótesis no preregistradas. Solo se calificó con jueces la condición principal en test; las ablaciones tienen únicamente métricas automáticas.
- En `chunking_A`, Q45–Q48 siguen usando la colección adversarial, construida con chunking B.
