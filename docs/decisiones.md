# Registro de decisiones

Este documento reúne las decisiones de diseño del proyecto, con las alternativas consideradas y la evidencia que las respalda. Es la fuente para la sección de arquitectura y decisiones del informe. Las cifras salen de los reportes versionados de cada fase:

- `reports/eda/`: perfilado del corpus;
- `reports/chunking/`: segmentación;
- `reports/embeddings/`: encoder;
- `reports/milvus/`: indexación;
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
