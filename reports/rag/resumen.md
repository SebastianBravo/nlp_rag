# Orquestación RAG, API e interfaz (Fases E y F)

Este resumen describe cómo se arma cada respuesta, cómo se calibró el presupuesto de contexto y qué mostró la revisión con las preguntas de `dev`. Las cifras salen de `tables/` y `runs/`, que regeneran:

```bash
docker compose --profile tools run --rm pipeline python -m scripts.check_rag --calibrar --models qwen2.5:1.5b llama3.2:1b llama3.2:3b gemma2:2b
docker compose --profile tools run --rm pipeline python -m scripts.check_rag --prompts rag_v1 rag_v2 rag_v3 rag_v4
```

Todo corrió en Docker, solo en CPU, con 10 núcleos y 8 GB para Docker, Ollama 0.40.13 y el gold con sha256 `daf22034…54e6`. Solo se usó `dev`: `test` queda intacto para la evaluación. Las respuestas completas, que pueden citar texto del corpus, están en `data/logs/` y no se versionan.

## Cómo se arma una respuesta

1. **Recuperar** (`app/core.py`).
   - La consulta se codifica con granite-97m.
   - Milvus busca con HNSW y `ef = 64` en `rag_granite97m_sent256_s10k_v1`, salvo que la petición elija otra colección del mismo encoder.
   - Los filtros de idioma, fuente y fechas se validan antes de armar la expresión de Milvus.
2. **Armar el contexto** (`app/rag.py`).
   - Cada fragmento va numerado entre etiquetas `<documento n="…">`, con su fuente, fecha, idioma y título.
   - Si un texto recuperado trae algo con forma de etiqueta, se neutraliza; los títulos de más de 300 caracteres se recortan.
   - Entran los fragmentos que caben en el presupuesto, siempre en orden de rank.
3. **Generar** (`app/llm.py`) con `/api/chat` de Ollama.
   - Las reglas van en el mensaje de sistema y los documentos y la pregunta en el del usuario.
   - Para todos los modelos: `num_ctx = 4096`, hasta 512 tokens de respuesta, temperatura 0, semilla 42 y 8 hilos.
4. **Validar.**
   - Se leen las citas `[n]` y se asocian a su chunk, documento y URL.
   - Se detecta la frase fija de abstención.
   - Se agregan avisos cuando corresponde (tabla siguiente).
5. **Registrar** una línea JSONL por consulta en `data/logs/ask/`, con:
   - los ids recuperados y los que entraron al contexto;
   - los tokens estimados y los reales;
   - los tiempos por etapa;
   - el prompt (id y sha256), la semilla y el digest del modelo.

Si los filtros no dejan ningún fragmento, la API se abstiene sin llamar al LLM.

| Aviso | Significado |
|---|---|
| `sin_citas` | La respuesta no cita ningún documento y no es una abstención |
| `citas_invalidas` | Cita un número de documento que no recibió |
| `citas_en_texto` | Cita con palabras ("según el documento 2") en lugar de `[n]`; cuenta como cita |
| `abstencion_parcial` | Trae la frase de abstención y además citas válidas: respondió una parte |
| `solo_citas` | La respuesta es solo una lista de `[n]`, sin contenido |
| `contexto_recortado` | Algún fragmento recuperado no entró al contexto por el presupuesto |
| `respuesta_cortada` | La respuesta llegó al máximo de 512 tokens |
| `posible_truncamiento` | El prompt real superó el presupuesto y Ollama pudo recortarlo |
| `filtra_instrucciones` | La respuesta contiene el canario de las instrucciones del sistema |
| `sin_evidencia` | Ningún fragmento pasó los filtros; no se llamó al LLM |

## Presupuesto de contexto en tokens

Ollama 0.40 no cuenta tokens antes de generar, y los tokenizadores de Llama y Gemma en Hugging Face exigen aceptar su licencia. Por eso el presupuesto se mide con el tokenizador de granite, que la API ya tiene cargado, multiplicado por un factor único.

- **Límite del prompt:** `num_ctx` − respuesta = 4.096 − 512 = 3.584 tokens.
- **Si no cabe todo:** los fragmentos que no caben se omiten completos, desde el de peor rank.
- **Calibración:** se compararon los tokens que cuenta Ollama con los del encoder, sin generar, en los 20 prompts de `dev` con 10 fragmentos cada uno (`tables/calibracion_tokens.csv`).

| Modelo | Tokens del encoder (mediana) | Tokens del LLM (mediana) | Cociente medio | Cociente máximo |
|---|---|---|---|---|
| `qwen2.5:1.5b` | 2.598 | 3.051 | 1,161 | 1,237 |
| `llama3.2:1b` | 2.598 | 3.008 | 1,136 | 1,222 |
| `llama3.2:3b` | 2.598 | 3.008 | 1,136 | 1,222 |
| `gemma2:2b` | 2.598 | 2.608 | 1,015 | 1,053 |

- **Factor elegido: `LLM_TOKEN_FACTOR = 1,3`.** Es el mayor cociente más un 5 % de margen. Al usar un solo factor, todos los LLM reciben exactamente el mismo contexto, que es lo que pide el experimento principal.
- **Comprobación:** en las 80 respuestas de la revisión, los tokens reales del prompt nunca pasaron del 95 % de la estimación, así que no hubo truncamiento.
- **Los dos Llama dan lo mismo** porque comparten tokenizador.

## Revisión en dev

Se respondieron las 20 preguntas de `dev` con `qwen2.5:1.5b` y `k = 5`. Q45, la adversarial, se consultó en su colección aparte; Q51, con su ventana de fechas. Se probaron cuatro versiones del prompt (`tables/dev_resumen.csv` y `tables/dev_preguntas.csv`).

| | `rag_v1` | `rag_v2` | `rag_v3` | `rag_v4` |
|---|---|---|---|---|
| Abstención correcta (20 preguntas) | **0,95** | 0,90 | 0,85 | 0,75 |
| Se abstuvo en las 3 sin respuesta | **3** | 2 | 1 | 2 |
| Algún relevante en el contexto (15 respondibles) | 0,80 | 0,80 | 0,80 | 0,80 |
| Cita algún documento relevante (15 respondibles) | 0,20 | 0,47 | 0,53 | **0,67** |
| Respuestas sin citas | 0,45 | 0,25 | 0,20 | **0,05** |
| Citas escritas con palabras | 0,10 | 0,20 | 0,00 | 0,00 |
| Respuestas hechas solo de citas | 0 | 0 | 2 | 1 |
| Respuestas cortadas a 512 tokens | 0 | 0 | 2 | 1 |
| Latencia total p50 / p95 (s) | 7,8 / 9,6 | 8,9 / 11,1 | 9,4 / 19,3 | 8,1 / 12,7 |
| Tokens por segundo (p50) | 56,8 | 55,3 | 51,3 | 59,7 |

Qué cambió cada versión y qué se observó:

- **`rag_v1`** (base): cumple la abstención, pero casi no usa el formato `[n]`. Cita con palabras, numera listas con corchetes o no cita.
- **`rag_v2`** agrega un ejemplo del formato de cita y el recordatorio junto a la pregunta.
  - **Ventaja:** cita más.
  - **Problema:** en Q15 copió como hechos los datos inventados del ejemplo ("300 metros", "2010"), igual en tres corridas.
- **`rag_v3`** describe el formato sin datos y agrega una regla para la evidencia parcial.
  - **Ventaja:** cita aún más.
  - **Problema:** en dos preguntas sin respuesta contestó solo con números entre corchetes en lugar de abstenerse.
- **`rag_v4`** pone primero la decisión de abstenerse y prohíbe las respuestas hechas solo de citas.
  - **Ventaja:** cita mucho más.
  - **Problemas:**
    - se abstuvo en preguntas que no pedían abstención (Q15, Q34, Q51);
    - en Q45 contestó solo con citas;
    - en Q04 repitió 169 veces el marcador `[n]` del propio prompt.

**Prompt elegido: `rag_v1`.** El criterio se fijó antes de ver `rag_v4`:

1. Se descarta el prompt que induzca datos que no están en los documentos o que deje más de una respuesta hecha solo de citas.
2. Entre los que quedan, gana la mayor abstención correcta.
3. Desempate: la cita de un documento relevante y, después, las respuestas sin citas.

Con esa regla salen `rag_v2` y `rag_v3`, y entre `rag_v1` y `rag_v4` gana `rag_v1` (0,95 frente a 0,75). `rag_v4` es la última iteración en `dev`.

**Hay un compromiso entre citar y abstenerse.** Con este modelo, pedir más citas mejora la trazabilidad pero empeora la abstención. Comparar `rag_v1` con `rag_v4` en `test` queda como ablación de la evaluación.

`rag_v1` por tipo de pregunta:

| Tipo | n | Abstención correcta | Relevante en contexto | Cita un relevante | Sin citas |
|---|---|---|---|---|---|
| directa | 9 | 0,89 | 0,67 | 0,11 | 0,44 |
| comparativa | 4 | 1,00 | 1,00 | 0,50 | 0,50 |
| ambigua | 2 | 1,00 | — | — | 0,50 |
| sin respuesta | 3 | 1,00 | — | — | 0,00 |
| adversarial | 2 | 1,00 | 1,00 | 0,00 | 1,00 |

Hay que leer estas cifras con cuidado:

- **"Cita un relevante" no mide si la respuesta es correcta.** Una respuesta puede citar el documento correcto y equivocarse, o acertar sin citar. La corrección y la fidelidad se juzgan con la rúbrica humana en la evaluación.
- **El techo lo pone la recuperación.** En 3 de las 15 preguntas respondibles ningún documento relevante entró al contexto con `k = 5`. Por ejemplo, la única abstención incorrecta de `rag_v1` (Q27) es una pregunta cuya evidencia no se recuperó.
- **Las respuestas casi no cambian entre corridas.** Con semilla 42 y temperatura 0, el mismo prompt dio respuestas idénticas en 18 de 20 preguntas (`rag_v1`, 4 corridas) y en 20 de 20 (`rag_v2`, 3 corridas; `rag_v3`, 2). La evaluación igual repite cada condición 3 veces.

## Inyección de prompt

Solo se usó Q45, la adversarial de `dev`; Q46-Q48 son de `test` y no se tocaron.

- **Q45:** el documento sintético con la instrucción inyectada entró al contexto con los cuatro prompts, y ninguno la obedeció (la instrucción pedía decir que no había información). Con `rag_v1` a `rag_v3` el modelo dio el dato de la respuesta de referencia; con `rag_v4` respondió solo con citas.
- **Filtración de instrucciones:** el canario no apareció en ninguna de las 80 respuestas.
- **Pruebas unitarias** (`tests/test_rag.py`) con textos propios:
  - un documento no puede abrir ni cerrar las etiquetas del prompt;
  - se detecta el canario;
  - se marcan las citas inválidas.

**Amenaza a la validez:** los documentos sintéticos tienen `collection_name = synthetic`, así que el prompt los presenta con fuente "desconocida". Eso puede ponerlos en evidencia ante el modelo. Hay que tenerlo en cuenta al medir Q46-Q48 en la evaluación.

## Estabilidad de los servicios

Durante la calibración aparecieron tres fallos con 8 GB y 10 CPU para Docker (decisión 66):

| Fallo observado | Ajuste |
|---|---|
| Ollama ocupó los 10 núcleos durante minutos, etcd se atrasó y Milvus perdió su sesión y terminó | Ollama usa 8 hilos por petición (`LLM_NUM_THREAD=8`) y Milvus se reinicia solo (`restart: unless-stopped`) |
| Con dos modelos cargados a la vez faltó memoria | Un solo modelo en memoria (`OLLAMA_MAX_LOADED_MODELS=1`) |
| llama-server guardaba cada prompt en RAM (unos 380 MB por prompt de 3.000 tokens con `llama3.2:3b`, con un tope de 8 GB) y el proceso de `llama3.2:3b` se cayó al séptimo prompt | Sin caché de prompts en RAM (`LLAMA_ARG_CACHE_RAM=0`) |

Con los tres ajustes terminaron sin errores la calibración de los cuatro modelos (80 prompts) y las cinco corridas de revisión en `dev` (280 respuestas). Milvus no se reinició ninguna vez.

## API e interfaz

- **API:**
  - `/health` comprueba Milvus y Ollama y responde 503 si falta alguno;
  - `/models` y `/collections` alimentan la interfaz;
  - `/retrieve` acepta filtros y colección;
  - `/ask` devuelve la respuesta, las citas, los avisos, las evidencias (si se citaron y si entraron al contexto), los tokens, los tiempos por etapa, el prompt y el `run_id`.
- **Errores** (`tests/test_api.py`):
  - modelo no descargado: 404, con el comando `ollama pull`;
  - colección inexistente o de otro encoder: 404;
  - filtros inválidos: 422;
  - Milvus u Ollama caídos: 503;
  - tiempo agotado: 504.
- **Colecciones válidas:** la API solo busca en colecciones del encoder de `MILVUS_COLLECTION`. La colección piloto `rag_e5small_char1000_v1`, con vectores de e5-small, sigue en Milvus pero ya no se puede consultar.
- **Interfaz** (`app/frontend.py`), revisada con el `AppTest` de Streamlit contra la API real:
  - estado de los servicios;
  - lista de modelos leída de Ollama;
  - selector de colección, con la adversarial marcada;
  - respuesta, métricas, avisos y evidencias con título, fuente, fecha, score, offsets y procedencias;
  - cambio de modelo, con aviso de arranque en frío;
  - con la API caída, un mensaje de error y el botón desactivado.

Un ejemplo de tiempos, con `qwen2.5:1.5b` y `k = 5`:

| Situación | Tiempo total | Desglose |
|---|---|---|
| En caliente, `dev` (mediana) | 7,8 s | Recuperación de 0,05 a 0,08 s; el resto es el LLM |
| Arranque en frío, una consulta | 18 s | 6,6 s de carga del modelo |

## Limitaciones

- **Cifras indicativas:** salen de 20 preguntas de un gold en borrador y de un solo modelo. Los otros tres modelos solo se usaron para calibrar los tokens.
- **Abstención detectada por la frase fija:** una abstención con otras palabras no se detecta.
- **Prompt solo con `qwen2.5:1.5b`:** el prompt se eligió con este modelo, y otro modelo podría preferir otra versión.
- **CPU:** todo corre en CPU, así que las respuestas tardan varios segundos y los modelos más grandes serán más lentos.
- **Variación entre entornos:** las respuestas pueden cambiar entre versiones de Ollama o entre máquinas, aunque la semilla y la temperatura estén fijas.
