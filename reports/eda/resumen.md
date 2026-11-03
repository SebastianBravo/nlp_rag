# Perfilado del corpus `dragons_rag` (Fase A)

Este resumen interpreta los resultados del perfilado y las decisiones que implican para las fases siguientes. Todas las cifras salen de `profile.json` y de `tables/`, que regenera:

```bash
docker compose --profile tools run --rm pipeline python -m scripts.profile_corpus --input data/documentos_rag.parquet
```

Parámetros en `configs/eda.yaml` (semilla 42). La ejecución de referencia tardó unos 3 minutos y registra en `profile.json` el sha256 del Parquet, las versiones de cada paquete y la configuración del encoder. Los reportes solo contienen agregados: ni texto del corpus, ni URLs completas, ni handles o nombres de autores.

## Hallazgos principales

1. **El archivo está íntegro.** El sha256 coincide con el manifiesto y con el catálogo, el esquema es el publicado y no hay diferencias en ningún conteo.
2. **El 83,2 % de las noticias supera el límite de 512 tokens del encoder**; ningún post de Bluesky lo supera (máximo 323). Las noticias necesitan chunking por tokens; Bluesky puede ir como chunk único.
3. **El 8,7 % de las filas son duplicados exactos**, y algunos grupos son grandes: un mismo post de campaña aparece 839 veces y una página de portada de un medio, 526.
4. **Un tercio de las noticias contiene boilerplate** de su sitio, y el 1 % tiene mojibake, casi todo de un único dominio.
5. **El idioma declarado falta en el 20,3 % de Bluesky** y coincide con el detectado en el 96,7 % de los casos comparables.
6. **La prensa está concentrada:** 5 dominios aportan el 47,3 % de las noticias.

## 1. Integridad (A1)

| Verificación | Resultado |
|---|---|
| sha256 frente al manifiesto y al catálogo | Coincide (`f4a669f1…b181728`) |
| Columnas: nombres y orden frente al catálogo | 33 de 33, en el mismo orden |
| Valores no vacíos frente al manifiesto | Iguales en las 33 columnas |
| Unicidad | `document_id` y `mongo_id` únicos; `url` y `uri` tienen 227 filas repetidas, todas de prensa |
| Recálculo de `char_count`, `word_count` y `text_sha256` | 0 diferencias |
| Conteos publicados | 169.248 filas: 64.670 de `bluesky_posts` y 104.578 de `url_crawls` (103.380 de Media Cloud y 1.198 de GDELT) |

- **Definiciones confirmadas:** `char_count = len(text)`, `word_count = len(text.split())` y `text_sha256` es el sha256 del texto en UTF-8.
- **Tipos:**
  - `keyword_matches` y `top_terms` son listas;
  - `word_count` y `char_count` son enteros;
  - el resto, incluidas `created_at` y `synced_at`, es texto. Las fechas hay que interpretarlas antes de usarlas.
- **Texto:** ninguno está vacío ni tiene solo espacios.
- **`title`:** falta en todos los posts de Bluesky y está vacío en 4 noticias.

Tablas: `integridad_*.csv`.

## 2. Distribuciones (A2)

### Longitud

| Colección | Caracteres p50 / p95 / máx. | Tokens p50 / p95 / máx. | > 256 tokens | > 512 tokens |
|---|---|---|---|---|
| Bluesky | 232 / 298 / 422 | 62 / 86 / 323 | 1 documento | 0 |
| Prensa | 3.882 / 10.112 / 363.528 | 896 / 2.335 / 88.532 | 96,6 % | 83,2 % |

Los tokens se cuentan como los procesa el encoder: prefijo `passage: ` más tokens especiales, sin truncar. Se usa la misma función, `app.embeddings.count_tokens`, que medirá los chunks en la Fase B. Figuras: `tokens_ecdf.png` y `longitud_caracteres.png`.

### Idioma

`language_base` se comparó con el idioma detectado por lingua sobre los primeros 1.000 caracteres de cada documento.

- **Acuerdo donde ambos existen:** 96,7 % en total (156.094 documentos), 94,5 % en Bluesky y 97,7 % en prensa.
- **El acuerdo cae en textos cortos:** 75,5 % en posts de menos de 50 caracteres y por encima del 92 % desde 50 caracteres.
- **Declarado nulo en 13.137 posts de Bluesky (20,3 %).** El detector clasifica el 81,2 % como inglés y el 16,3 % como español.
- **Idiomas minoritarios declarados** (de, pt, fr, nl, ja, it…; 2.388 documentos): el detector asigna inglés al 57,9 %. El idioma declarado es poco fiable fuera de en, es y hu.
- **Declarado `en` y detectado `es`:** 2.703 documentos (3,3 %).

Figuras: `idioma_matriz.png` y `idioma_acuerdo_longitud.png`.

### Fechas (`created_at`)

- **Bluesky:**
  - todas las fechas tienen hora;
  - el 95,4 % es de 2025-2026 (mediana 2026-04-30);
  - **344 posts tienen fecha anterior a 2023**, antes de que existiera la plataforma, con mínimo en 2008. Son fechas retroactivas fijadas por el cliente.
- **Prensa:**
  - el 94,9 % trae solo la fecha (99.297), 5.258 traen hora y 23 son nulas;
  - va de 2006-08-24 a 2026-09-21, con mediana 2023-12-07.
- **Ninguna fecha es posterior a `synced_at`** ni a la versión del corpus (2026-10-19).

Figura: `fechas_mensual.png`.

## 3. Duplicados (A3)

### Exactos (`text_sha256`)

- 154.477 textos únicos y **14.771 filas extra** (8,7 %), que confirman el manifiesto. Por colección:
  - Bluesky: 8.393 (13,0 % de la colección);
  - prensa: 6.378 (6,1 %).
- Ningún texto se repite entre colecciones.
- 4.434 textos tienen copias. Los 179 grupos con 10 o más copias suman 8.218 filas, y el mayor tiene 839.
- **Bluesky:** los grupos grandes son posts de plantilla de campañas, el texto que se comparte al firmar una petición o unirse a una acción.
- **Prensa:** los grupos grandes son páginas que no son artículos:
  - `www.rcnradio.com`: 526 copias de una página de portada; 495 de sus 497 filas son extra;
  - `confidencialcolombia.com`: 268 copias de una página de perfil; 196 de 197 filas.

Tablas: `duplicados_exactos_*.csv`.

### Casi duplicados (muestra de 10.000)

Se calcularon con MinHash/LSH sobre 5-gramas de palabras y se confirmaron con el Jaccard real ≥ 0,8.

- **4.398 pares en total.**
  - 4.031 (91,7 %) son duplicados exactos. Un grupo de *n* copias aporta *n(n−1)/2* pares.
  - 362 son del mismo dominio.
  - 5 son entre dominios distintos: noticias sindicadas, por ejemplo entre `scotsman.com` y `edinburghnews.scotsman.com`.
- **Documentos con algún casi duplicado no exacto:** 1,37 % (3,2 % en Bluesky y 0,23 % en prensa).
- En Bluesky, "mismo dominio" significa misma plataforma (`bsky.app`).

Tablas: `casi_duplicados_*.csv`. Los ejemplos solo traen IDs, dominios y medidas.

## 4. URLs y calidad del texto (A4)

### URLs

- Todas las URLs son http(s) válidas, y el 94,1 % de las de prensa usa https.
- 2.050 URLs de prensa (2,0 %) llevan parámetros de tracking, casi todos `utm_*`; 23 llevan `fbclid`.
- En 16.570 noticias (15,8 %) `final_url` difiere de `url` por redirecciones.

Tablas: `urls_*.csv` y `dominios_top.csv`.

### Ruido en el texto

| Señal | Bluesky | Prensa | Observación |
|---|---|---|---|
| Muy corto (< 5 palabras) | 651 (1,0 %) | 2 | |
| Solo URL | 116 (0,18 %) | 0 | |
| Etiquetas HTML | 8 | 111 (0,11 %) | 101 son de `www.infobae.com` |
| Entidades HTML | 19 | 0 | |
| Mojibake (`Ã©`, `â€`) | 0 | 1.049 (1,0 %) | 942 son de `www.elcolombiano.com`: el 99,4 % de sus 948 documentos |
| Carácter de reemplazo U+FFFD | 2 | 15 | |

Ningún texto está vacío ni tiene solo emojis o símbolos. Tablas: `ruido_*.csv`; figura: `ruido.png`.

### Boilerplate

Se considera boilerplate una línea de al menos 20 caracteres repetida en 20 o más documentos del mismo dominio.

- **Alcance:** aparece en el **32,8 % de las noticias** (34.341 de 104.564) y ocupa en promedio el 11,7 % de sus líneas.
- **Total:** 1.378 líneas distintas.
- **Por dominio:**
  - `www.larepublica.co`: 98,9 % de sus documentos;
  - `www.portafolio.co`: 90,8 %;
  - `www.eltiempo.com`: 77,8 %;
  - `www.dailymail.co.uk`: 77,7 %;
  - `www.infobae.com`: 42,5 %, donde el boilerplate es una cuarta parte de las líneas.

Tabla: `boilerplate_dominio.csv`.

## 5. Sesgos de recolección (A5)

- **Concentración de la prensa:**
  - 978 dominios; los 5 principales aportan el 47,3 % y los 20 principales, el 69,5 % (HHI 0,064);
  - `www.infobae.com` (18,9 %) y `www.eltiempo.com` (11,7 %) suman casi un tercio.

  Figura: `dominios_top.png`.
- **Idioma por colección:**
  - prensa: 60,6 % es, 35,3 % en, 3,9 % hu;
  - Bluesky: 70,8 % en, 20,3 % sin declarar, 5,4 % es.
- **Cobertura temporal dispar:** Bluesky cubre sobre todo 2025-2026, y la prensa se reparte desde 2016. Las comparaciones entre colecciones solo son justas en la ventana común 2024-2026.
- **Campos de adquisición:** describen cómo se recolectó cada documento, no su contenido, y no deben usarse como etiquetas semánticas.
  - En Bluesky, `country_key` asigna el 59,8 % a Reino Unido, el 24,2 % a Colombia y el 16,0 % a Hungría, pero solo 71 posts declaran húngaro. `country_key` es la consulta de recolección, no el idioma ni el lugar.
  - `collection_keyword_label` (18 marcos temáticos) y `keyword_family` son familias de búsqueda.
  - En prensa, `strategy` es `whitelist` en el 93,8 %.

Tablas: `sesgo_*.csv`.

## 6. Muestras (A6)

- **Diseño:** muestreo estratificado por colección × idioma declarado (en, es, hu, otro, desconocido), con 9 estratos, asignación proporcional y semilla 42.
- **Tamaños:** 2.000 (piloto) y 10.000, anidadas: la de 2.000 está contenida en la de 10.000.
- **Archivos:** `data/samples/sample_<n>_ids.txt` y `sample_<n>_meta.json`, este último con semilla, fecha, sha256 del Parquet y distribución. El corpus completo solo tiene `sample_full_meta.json`.
- **Representatividad:**
  - la mayor diferencia de proporción por estrato es de 0,03 puntos porcentuales;
  - la prueba KS de longitud en caracteres no muestra diferencias (p = 0,40 y 0,53 en Bluesky; 0,24 y 0,066 en prensa);
  - la mediana de la muestra está entre 0,97 y 1,02 veces la del corpus.

Tablas: `muestras_*.csv`.

## 7. Decisiones que implica

| # | Hallazgo | Decisión | Fase |
|---|---|---|---|
| 1 | Ningún post de Bluesky supera 323 tokens | Bluesky va como chunk único | B |
| 2 | El 83,2 % de las noticias supera 512 tokens; el chunking actual por caracteres no controla tokens | Chunking por tokens medido con `count_tokens`, con prefijo y tokens especiales dentro del presupuesto | B |
| 3 | 8,7 % de filas duplicadas exactas, sin cruces entre colecciones | Deduplicar por hash de chunk conservando todas las procedencias | B |
| 4 | Páginas de portada o perfil repetidas cientos de veces (`rcnradio.com`, `confidencialcolombia.com`) | Marcar los dominios con más del 90 % de filas duplicadas y revisar si se excluyen | B |
| 5 | Boilerplate en el 32,8 % de las noticias | Excluir las líneas boilerplate como intervalos al segmentar, sin reescribir el texto, para que los offsets sigan siendo del original | B |
| 6 | Mojibake en el 1 % de las noticias, casi todo de un dominio | Repararlo cambiaría los offsets: se reporta y se decide en la Fase B si se excluye o se repara en un artefacto aparte | B |
| 7 | `language_base` falta en el 20,3 % de Bluesky y es poco fiable fuera de en, es y hu | Campo de idioma para filtros: `language_base`, o el detectado cuando falta, guardando el origen | D |
| 8 | Fechas en texto, el 94,9 % de la prensa sin hora y 344 posts retroactivos | Convertir a epoch UTC con indicador de precisión y marcar las fechas no plausibles | D |
| 9 | Tracking en el 2,0 % de las URLs de prensa | Mostrar las URLs sin `utm_*`, conservando la original | F |
| 10 | Prensa concentrada en pocos dominios, idiomas desiguales y cobertura temporal distinta | El gold estándar cubre ambas colecciones, es/en/hu y la ventana 2024-2026, y no se apoya en un solo dominio | Evaluación |

## Limitaciones del perfilado

- El detector de idioma lee solo los primeros 1.000 caracteres y es menos fiable en textos muy cortos.
- Los casi duplicados se calcularon sobre la muestra de 10.000; en el corpus completo la tasa puede ser distinta.
- Las heurísticas de boilerplate (línea repetida por dominio) y de mojibake (patrones de UTF-8 leído como Latin-1) son aproximaciones. Una línea legítima repetida, como la firma de un autor, cuenta como boilerplate.
- El conteo de tokens depende de la versión del tokenizador registrada en `profile.json`.
