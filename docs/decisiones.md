# Registro de decisiones

Este documento reúne las decisiones de diseño del proyecto, con las alternativas consideradas y la evidencia que las respalda. Es la fuente para la sección de arquitectura y decisiones del informe. Las cifras salen de los reportes versionados de cada fase:

- `reports/eda/`: perfilado del corpus;
- `reports/chunking/`: segmentación;
- `reports/embeddings/`: encoder;
- `reports/milvus/`: indexación;
- `reports/rag/`: orquestación (revisión en dev);
- `data/gold/README.md`: gold estándar.

## Infraestructura y reproducibilidad

| # | Decisión | Alternativas | Motivo y evidencia |
|---|---|---|---|
| 1 | Un solo módulo carga el encoder (`app/embeddings.py`) y lo usan la API y el pipeline | Un servicio de embeddings aparte | Las consultas y los pasajes se codifican exactamente igual, con menos infraestructura |
| 2 | Python 3.12 en las imágenes y en el entorno local | 3.11 | Todas las dependencias instalan en 3.12, y una sola versión evita diferencias entre local y Docker |
| 3 | Un Dockerfile e imagen por servicio (`api`, `frontend`, `pipeline`); `docker-compose.yml` solo incluye los archivos de `docker/` | Una imagen compartida; un solo compose | Cada imagen lleva solo sus dependencias (el frontend no instala torch). `pipeline` incluye las dependencias de `api`, así que el encoder es idéntico en ambas: se verificó que producen el mismo vector |
| 4 | torch solo CPU en las imágenes | Rueda de PyPI con CUDA | Los contenedores no usan GPU. La imagen de la API bajó de 10,4 GB a 2,1 GB |
| 5 | Todo paso del pipeline corre en el contenedor `pipeline` (perfil `tools`); los notebooks solo leen artefactos | Notebooks que recalculan | Ninguna cifra del informe depende del entorno local |
| 6 | Etapas desacopladas con artefactos intermedios: Parquet → perfilado → muestras → chunks → vectores → Milvus | Ingesta que hace todo de una vez | Cada etapa se repite sin rehacer las anteriores y se puede comparar por separado |
| 7 | `pytest.ini` con `pythonpath = .` | Usar siempre `python -m pytest` | `pytest` a secas fallaba al importar `scripts` |
| 8 | Los registros crudos por ejecución (`reports/**/*.jsonl`) no se versionan | Versionarlos | Contienen texto del corpus |

## Datos y corpus

| # | Decisión | Alternativas | Motivo y evidencia |
|---|---|---|---|
| 9 | El corpus se descarga con `download_corpus` de `open_corpus_co_es` y se copia byte a byte; el paquete queda fijado en `0.1.1a4` | Guardar la salida de `load_corpus`; un rango de versiones | `load_corpus` devuelve una lista de dicts sin los campos nulos de cada fila, y reescribirla cambia tipos y hash. La copia conserva el sha256 publicado (`f4a669f1…b181728`) |
| 10 | El manifiesto del corpus se descarga junto al Parquet | Descargarlo en el perfilado | La adquisición queda en una sola etapa |
| 11 | El Parquet original nunca se modifica; las limpiezas producen artefactos nuevos | Limpiar en el sitio | Trazabilidad: todo offset se refiere al texto original |
| 12 | Los reportes y las muestras versionadas solo tienen agregados e IDs, sin texto, URLs, handles ni nombres de autores | Ejemplos con texto | Privacidad y licencia del corpus. Se verifica con pruebas y con escaneos de fragmentos del corpus |

## Perfilado (Fase A)

| # | Decisión | Alternativas | Motivo y evidencia |
|---|---|---|---|
| 13 | Idioma y tokens se miden sobre el corpus completo; los casi duplicados (MinHash/LSH), sobre la muestra de 10.000 | Todo sobre una muestra | El corpus completo cuesta unos 3 minutos |
| 14 | Detector de idioma lingua (todos los idiomas, alta precisión, primeros 1.000 caracteres) | langdetect, fastText | Determinista y sin descargar modelos; en el piloto dio 96,5 % de acuerdo con `language_base` |
| 15 | Muestras estratificadas por colección × idioma declarado, anidadas (la de 2.000 está dentro de la de 10.000), semilla 42 y con duplicados | Muestrear solo textos únicos; muestras independientes | Representan el corpus tal cual; la deduplicación se hace sobre los chunks. La mayor diferencia de proporción frente al corpus es de 0,03 puntos |
| 16 | Features por documento, sin texto, en un artefacto intermedio no versionado | Recalcular en cada etapa | La segmentación reutiliza idioma detectado, mojibake y boilerplate |
| 17 | `created_at` de Bluesky anterior a 2023 se marca como no plausible | Ignorarlo | 344 posts con fecha retroactiva, de antes de que existiera la plataforma |
| 18 | Los resúmenes de cada fase se escriben a mano; los scripts regeneran tablas, figuras y metadatos | Resúmenes generados | Las decisiones requieren criterio; cada cifra del resumen sale de las tablas |

## Segmentación (Fase B)

| # | Decisión | Alternativas | Motivo y evidencia |
|---|---|---|---|
| 19 | Los chunks se guardan como Parquet intermedio antes de Milvus | Segmentar dentro de la ingesta | Se pueden comparar estrategias sin reindexar |
| 20 | Dos estrategias con el mismo presupuesto de 256 tokens de contenido: A (ventana de tokens con 32 de solapamiento) y B (párrafos y oraciones con una oración de solapamiento) | B con el tope completo del encoder | Comparación en igualdad de condiciones; el chunk más largo mide 261 tokens |
| 21 | El tamaño de los chunks se mide siempre con el tokenizador de multilingual-e5-small, fijado por revisión, aunque el encoder de vectorización sea otro | Medir con el tokenizador de cada encoder | Los chunks no cambian si cambia el encoder, así que una comparación de encoders no se confunde con un cambio de segmentación |
| 22 | Se excluyen los textos de prensa repetidos 100 veces o más | Solo deduplicar | Son portadas y páginas de perfil, no artículos |
| 23 | El mojibake se conserva y se marca con `has_mojibake` | Repararlo; excluir el dominio | Repararlo rompería `text[char_start:char_end] == chunk` |
| 24 | El boilerplate se salta como intervalos y se descartan los fragmentos de menos de 8 tokens en documentos con varios chunks | Reescribir el texto sin boilerplate | Los offsets siguen siendo sobre el original; sale el 7,8 % de los caracteres de prensa |
| 25 | Oraciones con nltk punkt; el húngaro usa el modelo de inglés | pysbd; blingfire | punkt devuelve posiciones y es rápido; pysbd es unas 20 veces más lento y blingfire no funciona en arm64 |
| 26 | `chunk_id` determinista y deduplicación exacta con `provenance_ids` | IDs aleatorios; descartar duplicados sin rastro | Reindexar es idempotente y ningún documento pierde su procedencia |
| 27 | B es la estrategia candidata por defecto, sujeta a la comparación extrínseca | A por defecto | B corta oraciones en el 0,14 % de los chunks de prensa, frente al 93,7 % de A, con un 13,7 % más de chunks |

## Gold estándar

| # | Decisión | Alternativas | Motivo y evidencia |
|---|---|---|---|
| 28 | Relevancia anotada por documento y rango de caracteres, con grados 0/1/2 | Por `chunk_id` | Sirve para cualquier estrategia de segmentación; todos los intervalos caen dentro de chunks de A y de B |
| 29 | Métricas de recuperación a nivel documento: cada chunk cuenta para su documento y los de su procedencia, con su mejor rank; nDCG con ganancia igual al grado | Métricas por chunk | Coherente con la anotación y comparable entre estrategias |
| 30 | `dev` para ajustar y `test` solo para reportar; cada corrida registra el sha256 del gold | Ajustar sobre todo el gold | Evita sobreajustar a las preguntas de test |
| 31 | Documentos sintéticos con inyección de prompt en una colección aparte, con métricas propias | Mezclarlos en la colección general | No contaminan el índice ni las métricas de recuperación |
| 32 | `ventana_fecha = {desde, hasta}` sobre `created_at`; fuera de la ventana un documento no es relevante | La fecha solo en el texto de la pregunta | Permite evaluar el filtro temporal |
| 33 | `idioma_evidencia` se completa con el idioma detectado de la evidencia | Usar la etiqueta del corpus | Coincide con las 14 anotaciones manuales; la etiqueta del corpus tiene errores |
| 34 | Las respuestas de referencia se redactan sin copiar texto del corpus | Citas literales | El gold se versiona y no puede llevar texto del corpus |

## Encoder (Fase C)

| # | Decisión | Alternativas | Motivo y evidencia |
|---|---|---|---|
| 35 | El corpus se vectoriza con `granite-embedding-97m-multilingual-r2`: 384 dimensiones, sin prefijos, coseno, revisión fijada | multilingual-e5-small, Qwen3-Embedding-0.6B, multilingual-e5-base, gte-multilingual-base, EmbeddingGemma-300m, BGE-M3, jina v5 nano | La restricción dura es la CPU y la RAM del contenedor. Granite tiene la misma dimensión que e5-small, 9,4 puntos más de recuperación multilingüe publicada (60,3 frente a 50,9), entrenamiento explícito en español, inglés y húngaro, 32K de contexto y licencia Apache-2.0. Medido: 19,5 chunks/s y 1,9 GB de RAM, frente a 32,7 chunks/s y 1,5 GB de e5-small (`reports/embeddings/tables/costo_encoders.csv`) |
| 36 | Se descarta Qwen3-Embedding-0.6B, el de mejor puntaje publicado (64,64) | Usarlo solo en la muestra | Medido: 0,78 chunks/s (35,6 h por cada 100.000 chunks) y 7 GB de RAM con lote de 32 |
| 37 | Los modelos se cargan en float32 | El formato de los pesos (bfloat16) | En bfloat16 la CPU del contenedor fue unas 16 veces más lenta |
| 38 | Se guardan las 384 dimensiones | Recortar a 256 o 128 | granite no se entrenó con Matryoshka: en dev, recortar a 256 baja el Recall@10 de 0,93 a 0,87, y a 128 lo baja a 0,73 |
| 39 | Consultas sin prefijo | Prefijo `query: ` | Así lo indica la ficha del modelo; con el prefijo el resultado en dev casi no cambia (MRR de 0,734 a 0,746) |
| 40 | Vectores en Parquet, calculados por partes reanudables, antes de Milvus | Vectorizar dentro de la ingesta | La vectorización es la etapa más costosa (34 minutos para la muestra de 10.000) y queda separada de la indexación |
| 41 | El encoder se analiza solo con `dev`, con búsqueda exacta y métricas a nivel documento | Usar `test`; evaluar en Milvus | `test` queda intacto y el análisis no depende de los parámetros del índice |
| 42 | No se compara la calidad de varios encoders con el gold; la elección se apoya en el costo medido y en benchmarks publicados | Ablación con un segundo encoder | Por tiempo, por decisión del equipo |
| 43 | La API adopta granite al indexar en Milvus, con una colección nueva por estrategia | Cambiarla ya | Hasta entonces la API sigue con e5-small y su colección piloto; nunca se mezclan vectores de dos encoders |

## Indexación en Milvus (Fase D)

Detalle y cifras en `reports/milvus/resumen.md`.

| # | Decisión | Alternativas | Motivo y evidencia |
|---|---|---|---|
| 44 | Se indexan un piloto de 2.000 documentos y la muestra de 10.000, cada una con A y B en colecciones separadas. El nombre sigue `rag_<encoder>_<segmentación>_<muestra>_v<n>`, por ejemplo `rag_granite97m_sent256_s10k_v1` | Corpus completo; una sola colección | La muestra de 10.000 contiene los 53 documentos de evidencia del gold; el piloto solo 7, así que sirve para iterar, no para medir. El corpus completo costaría unas 10 horas de vectorización. Con A y B indexadas, la ablación de segmentación queda lista |
| 45 | Campos filtrables: `language` con el idioma declarado o, si falta, el detectado (con `language_source`); `created_at` como época UTC en segundos, nula si falta, con su precisión y su plausibilidad; `provenance_ids` como lista | Solo el idioma declarado; la fecha como texto | Los 763 chunks de Bluesky sin idioma declarado se pueden filtrar. El 95 % de la prensa tiene fecha sin hora, y la precisión lo deja explícito. Las ventanas de fecha de Q51 y Q52 necesitan un rango numérico |
| 46 | Índice HNSW con métrica COSINE, M = 16 y efConstruction = 200 | AUTOINDEX; FLAT; IVF_FLAT | Parámetros documentados y calibrables; valores estándar para decenas de miles de vectores. En dev, Milvus da las mismas métricas que la búsqueda exacta con A y con B |
| 47 | `ef = 64` en búsqueda, por una regla fijada antes de medir: el menor `ef` con Recall@10 medio ≥ 0,99 frente a la búsqueda exacta | `ef = 128` | Con 545 consultas (45 de dev y 500 chunks al azar): Recall@10 de 0,996 y 0,81 ms de mediana. El peor caso (0,40) es una paráfrasis con vecinos casi empatados; `ef = 128` lo sube a 0,90 por 0,15 ms más, una opción si la evaluación lo pide |
| 48 | Índices escalares INVERTED en `language`, `collection_name` y `source`, y STL_SORT en `created_at` | Sin índice escalar | Filtros por igualdad y por rango de fecha; los 6 filtros probados devuelven solo entidades que los cumplen |
| 49 | La ingesta lee los Parquet de chunks y de vectores y hace `upsert` por lotes de 500, con reintentos (esperas de 2, 4, 8… s), progreso para retomar y verificación de conteos al final | `insert`; vectorizar dentro de la ingesta | Repetir la carga no duplica: se cargó de nuevo `sent256_s2k` y siguió con 7.517 entidades. Las 5 colecciones tienen tantas entidades como chunks sus Parquet y las conservan tras reiniciar Milvus |
| 50 | Los vectores del piloto se toman de la muestra de 10.000 por el hash del texto del chunk | Vectorizar el piloto aparte | La muestra de 2.000 está dentro de la de 10.000 y la segmentación es determinista: se reutilizó el 100 % de los vectores (7.517 de B y 6.659 de A) |
| 51 | Colección adversarial aparte con los chunks de los documentos de Q45-Q48 y los 4 documentos sintéticos (36 chunks) | Agregar los sintéticos a la colección general | Aplica la decisión 31: no contaminan el índice ni las métricas de recuperación |
| 52 | La API busca en `rag_granite97m_sent256_s10k_v1` con granite y `ef` desde `.env`; las colecciones las crea solo la ingesta. Se elimina la ingesta piloto por caracteres | Que la API cree la colección si no existe | El esquema y los índices quedan en un solo lugar; la API solo busca. Cierra la decisión 43 |
| 53 | El filtro de fecha usa la fecha del documento canónico de cada chunk | Guardar las fechas de todas las procedencias | Afecta a 176 chunks de 37.919 en `sent256_s10k` (0,5 %): los que tienen procedencias de días distintos |

## Orquestación, API e interfaz (Fases E y F)

Detalle y cifras en `reports/rag/resumen.md`.

| # | Decisión | Alternativas | Motivo y evidencia |
|---|---|---|---|
| 54 | Recuperar (`app/core.py`), armar el contexto y generar (`app/rag.py`) son funciones separadas; `/ask` las compone | Un solo paso dentro del endpoint | La evaluación puede recuperar una vez y generar con cada LLM sobre exactamente el mismo contexto, como pide el experimento principal |
| 55 | Generación con `/api/chat` de Ollama: reglas en el mensaje de sistema y documentos en el del usuario | `/api/generate` con un solo texto | Cada modelo aplica su propia plantilla de chat, y las reglas no quedan mezcladas con el texto recuperado |
| 56 | Prompt versionado en `prompts/rag_v1.yaml`; cada respuesta registra su id, su versión y el sha256 del archivo | Prompt dentro del código | Trazabilidad: cambiar el texto crea una versión nueva y los resultados dicen con cuál se obtuvieron |
| 57 | `num_ctx = 4096` y hasta 512 tokens de respuesta para todos los modelos | El valor por defecto de cada modelo | Si el prompt no cabe, Ollama lo recorta sin avisar. Un valor fijo deja a todos los LLM en las mismas condiciones |
| 58 | El presupuesto se mide con el tokenizador del encoder multiplicado por un factor único (`LLM_TOKEN_FACTOR = 1,3`), calibrado con el conteo real de Ollama en los 4 LLM. Los fragmentos que no caben se omiten completos, desde el de peor rank | Tokenizador de cada LLM; recortar por caracteres | Ollama 0.40 no cuenta tokens antes de generar, y los tokenizadores de Llama y Gemma en Hugging Face exigen aceptar su licencia. Un único factor da el mismo contexto a todos los modelos. Medido con los 20 prompts de `dev` (k = 10): el mayor cociente fue 1,237 (`qwen2.5:1.5b`), frente a 1,222 de Llama y 1,053 de Gemma; el factor suma un 5 % de margen |
| 59 | La abstención la decide el prompt con una frase fija (español e inglés), que la API detecta; sin umbral de score | Umbral de similitud para no llamar al LLM | `dev` tiene solo 3 preguntas sin respuesta, pocas para calibrar un umbral; queda como ablación de la evaluación |
| 60 | Citas `[n]` obligatorias. La API separa citas válidas e inválidas y marca las respuestas sin citas | Confiar en el modelo | La evidencia citada queda trazable hasta el chunk y el documento |
| 61 | Defensa contra inyección: documentos entre etiquetas `<documento>` que el texto recuperado no puede abrir ni cerrar, regla explícita y un canario para detectar si el modelo revela sus instrucciones | Sin defensa | Los casos de prueba usan textos propios y Q45 (`dev`); Q46-Q48 (`test`) quedan intactas para la evaluación |
| 62 | Si los filtros no dejan fragmentos, la API se abstiene sin llamar al LLM | Llamar al LLM sin contexto | Evita una respuesta sin fuente y ahorra la generación |
| 63 | Una línea JSONL por consulta en `data/logs/ask/`, con ids de los chunks, tokens estimados y reales, tiempos por etapa, prompt, semilla y digest del modelo | Guardarlo en `reports/` | La respuesta puede citar texto del corpus, así que no se versiona |
| 64 | Filtros de idioma, fuente y fechas validados antes de armar la expresión de Milvus; colección elegible entre las `rag_*`. `window_filter` vive en `app/filters.py` y el pipeline lo importa | Expresión de filtro libre | Ningún texto del usuario entra a la expresión, y API y pipeline usan la misma implementación |
| 65 | La interfaz lee los modelos de `/models`, muestra el estado de Milvus y Ollama, los tiempos, las citas y las evidencias con offsets; cada error tiene su mensaje y su código HTTP | Lista de modelos fija y un error genérico | Lo pide el enunciado (Fase F) |
| 66 | Ollama usa 8 hilos por petición (`LLM_NUM_THREAD`), mantiene un solo modelo en memoria (`OLLAMA_MAX_LOADED_MODELS=1`) y sin caché de prompts en RAM (`LLAMA_ARG_CACHE_RAM=0`); Milvus se reinicia solo (`restart: unless-stopped`) | Valores por defecto | Fallos observados en la calibración con 8 GB para Docker. Con los 10 núcleos ocupados por Ollama, etcd se atrasó y Milvus perdió su sesión y terminó. Con `llama3.2:3b`, llama-server guardaba cada prompt en RAM (unos 380 MB por prompt de 3.000 tokens, hasta 8 GB por defecto) y el proceso del modelo se quedó sin memoria al sexto prompt |
| 67 | El prompt y la calibración se revisan solo con `dev` (`check_rag` rechaza `test`) | Usar todo el gold | `test` queda intacto para la evaluación |
| 68 | El prompt por defecto es `rag_v1`. `rag_v2` a `rag_v4` quedan versionados con sus resultados en dev | `rag_v4`, que cita más | Criterio fijado antes de ver `rag_v4`: se descarta el prompt que induzca datos ajenos a los documentos o deje más de una respuesta hecha solo de citas, y entre los demás gana la mayor abstención correcta. `rag_v2` copió los datos de su ejemplo de formato (Q15) y `rag_v3` contestó solo con citas en 2 de las 3 preguntas sin respuesta. Entre `rag_v1` y `rag_v4`, la abstención correcta fue 0,95 frente a 0,75, aunque `rag_v4` cita un documento relevante en 0,67 de las respuestas y `rag_v1` en 0,20. Con 20 preguntas la diferencia es indicativa; la ablación de prompt queda para la evaluación |
| 69 | La API solo busca en colecciones del mismo encoder que `MILVUS_COLLECTION` (prefijo `rag_granite97m_`) y `/collections` solo lista esas | Aceptar cualquier colección `rag_*` | En Milvus sigue la colección piloto `rag_e5small_char1000_v1`, de 384 dimensiones como granite: buscar en ella con una consulta de granite daría resultados sin sentido y sin error |
| 70 | Los títulos de más de 300 caracteres se recortan en el prompt | Incluirlos completos | El p99 es de 146 caracteres, pero 16 chunks de la muestra de 10.000 traen títulos de hasta 7.998 caracteres (texto de la página). En Q22 uno de ellos ocupó unos 2.100 tokens y dejó fuera 3 de los 5 fragmentos |
| 71 | Las citas escritas con palabras ("según el documento 2") cuentan como citas pero llevan el aviso `citas_en_texto`; la frase de abstención junto a citas válidas es `abstencion_parcial`, no abstención; una respuesta hecha solo de `[n]` lleva `solo_citas` | Contar solo `[n]`; tratar cualquier aparición de la frase como abstención | Las tres situaciones aparecieron en dev con `qwen2.5:1.5b`. Así la métrica refleja si la respuesta es trazable, y el formato se mide aparte |

## Pooling y segunda anotación del gold

| # | Decisión | Alternativas | Motivo y evidencia |
|---|---|---|---|
| 72 | El pooling une, a nivel documento, los 20 primeros documentos de las colecciones A y B de la muestra de 10.000 (`sent256_s10k` y `win256_s10k`), consultando con la pregunta original y aplicando `ventana_fecha` como filtro. Un chunk cuenta para su documento canónico y para los de `provenance_ids` | Solo B; top-10; añadir un recuperador léxico como fuente de candidatos | Es lo que pide el protocolo del gold, y cubre las dos estrategias que luego se comparan. Al venir de recuperadores vectoriales, el gold queda sesgado hacia ellos: se declara como limitación. Se guarda `pooling.jsonl` (solo ids y rangos) con la cobertura de los relevantes ya anotados |
| 73 | Las hojas de anotación son libros `.xlsx` locales (`data/annotation/`, no versionados), con el pasaje recuperado en su contexto, orden aleatorio estable (semilla 42) y lista desplegable 0/1/2 | CSV; hojas con rangos; documentos completos | El pasaje lleva texto del corpus, que no se versiona. El orden aleatorio y la ausencia de rangos evitan que la posición influya en el juicio. Excel limita las celdas a unos 32.000 caracteres, así que se muestra el pasaje con contexto y no el documento entero |
| 74 | Doble anotación: submuestra de 16 preguntas de `test` estratificada por tipo (entre las 30 con segundo anotador asignado), top-10 de cada colección, en dos fases: primero juicios ciegos y la respuesta propia; después la validez de la respuesta de referencia | Todas las preguntas; ver la referencia desde el inicio | El mínimo del plan es de 16 preguntas. Ver la respuesta de referencia antes de anotar anularía la independencia del segundo juicio |
| 75 | Acuerdo: kappa ponderado cuadrático sobre los grados 0/1/2, kappa binario (grado ≥ 1) y acuerdo exacto, sobre los pares (pregunta, documento) que ambos juzgaron. Si todos los grados coinciden en un valor, el kappa no está definido y se reporta el acuerdo exacto | Solo acuerdo exacto | El kappa corrige el acuerdo por azar, y ponderar tiene en cuenta que 0 frente a 2 es peor que 1 frente a 2. En las preguntas sin respuesta todo es 0 |
| 76 | Los desacuerdos se resuelven entre los dos anotadores (`grado_final`); se conservan los grados originales de ambos en `anotacion_1.jsonl` y `anotacion_2.jsonl`. El resto de la pregunta queda con el grado del anotador 1 | Promediar; el anotador 1 decide | Trazabilidad: el acuerdo se puede recalcular y el gold final tiene un solo grado por documento |
| 77 | `freeze_gold` bloquea si queda pooling pendiente, hay desacuerdos sin resolver, una pregunta sin respuesta queda con documentos relevantes o un relevante cae fuera de `sample_10000`; sin `--write` es un ensayo. Los documentos relevantes nuevos entran sin offsets y las preguntas con doble anotación llevan `doble_anotacion` | Congelar a mano | Las condiciones del congelamiento quedan comprobadas por código. Sin offsets, esos documentos cuentan para Recall, MRR y nDCG, pero no para el acierto de evidencia |
| 78 | Las tablas ya publicadas de `dev` se calcularon con el borrador (sha256 `daf22034…`). Tras congelar, se citan los dos hashes o se recalculan | Reemplazar sin dejar rastro | El hash de cada corrida identifica contra qué gold se midió |
