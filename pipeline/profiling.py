"""Perfilado del corpus (Fase A): funciones que reciben y devuelven DataFrames.

Ninguna salida contiene texto del corpus, URLs completas, handles ni nombres de autores:
solo conteos, proporciones, percentiles, identificadores de documento y dominios.
"""
import hashlib
from fnmatch import fnmatch
from functools import lru_cache
from urllib.parse import parse_qsl, urlsplit
import numpy as np
import pandas as pd

URL = r'(?:https?://|www\.)\S+'
ONLY_URLS = rf'\s*(?:{URL}\s*)+'
ALNUM = r'[^\W_]'
HTML_TAG = r'</?[A-Za-z][A-Za-z0-9]*(?:\s[^<>]*)?/?>'
HTML_ENTITY = r'&(?:[A-Za-z]{2,8}|#\d{2,6}|#x[0-9A-Fa-f]{2,5});'
# UTF-8 leído como Latin-1/cp1252: "é" -> "Ã©", "’" -> "â€™", espacio duro -> "Â ".
MOJIBAKE = r'[ÃÂ][\u0080-¿]|â€'
DATE_ONLY = r'\d{4}-\d{2}-\d{2}'
PERCENTILES = (5, 50, 95, 99)
NOISE_FLAGS = {'is_blank': 'vacío o solo espacios', 'is_short': 'muy corto', 'only_urls': 'solo URL',
               'no_alnum': 'solo emojis o símbolos', 'has_html_tag': 'etiquetas HTML',
               'has_html_entity': 'entidades HTML', 'has_mojibake': 'mojibake',
               'has_replacement_char': 'carácter de reemplazo (U+FFFD)'}
RECOUNT = {'char_count': 'char_count_calc', 'word_count': 'word_count_calc', 'text_sha256': 'text_sha256_calc'}

def _groups(df, by):
    """Cada grupo de `by` y al final el total, para tablas por colección con fila 'total'."""
    return [*((str(k), g) for k, g in df.groupby(by, sort=True)), ('total', df)]

def group_rare(values, min_share, other='otro', missing='desconocido'):
    """Conserva los valores con al menos `min_share` de las filas; el resto pasa a `other` y los nulos a `missing`."""
    shares = values.value_counts(normalize=True)
    keep = set(shares[shares >= min_share].index)
    return values.map(lambda v: missing if pd.isna(v) else (v if v in keep else other)).astype(str)

# --- Features por documento: se calculan por lotes y no conservan el texto ---

def text_features(df, short_words=5):
    """Longitudes y hash recalculados y banderas de ruido de cada documento (`document_id`, `text`)."""
    text = df['text']
    filled = text.fillna('')
    words = filled.str.split().str.len()
    blank = filled.str.strip().eq('')
    return pd.DataFrame({
        'document_id': df['document_id'],
        'text_is_null': text.isna(),
        'char_count_calc': filled.str.len(),
        'word_count_calc': words,
        'text_sha256_calc': [hashlib.sha256(t.encode('utf-8')).hexdigest() for t in filled],
        'n_lines': filled.str.count('\n') + 1,
        'n_urls': filled.str.count(URL),
        'is_blank': blank,
        'is_short': words < short_words,
        'only_urls': filled.str.fullmatch(ONLY_URLS),
        'no_alnum': ~blank & ~filled.str.contains(ALNUM),
        'has_html_tag': filled.str.contains(HTML_TAG),
        'has_html_entity': filled.str.contains(HTML_ENTITY),
        'has_mojibake': filled.str.contains(MOJIBAKE),
        'has_replacement_char': filled.str.contains('�', regex=False),
    })

def normalize_lines(lines):
    """Normalización de líneas para comparar boilerplate: sin bordes, en minúsculas y con espacios colapsados."""
    return lines.fillna('').str.strip().str.lower().str.replace(r'\s+', ' ', regex=True)

def hash_lines(normalized):
    """Hash determinista de 64 bits por línea normalizada (el mismo en el perfilado y en la segmentación)."""
    return pd.util.hash_pandas_object(normalized, index=False).to_numpy()

def line_hashes(df, min_chars=20):
    """Una fila por documento y línea normalizada distinta (hash de 64 bits), para detectar boilerplate."""
    lines = pd.DataFrame({'document_id': df['document_id'].to_numpy(),
                          'line': df['text'].fillna('').str.split('\n').to_numpy()}).explode('line')
    norm = normalize_lines(lines['line'])
    lines = lines.assign(line=norm)[norm.str.len() >= min_chars]
    lines['line_hash'] = hash_lines(lines['line'])
    return lines[['document_id', 'line_hash']].drop_duplicates().reset_index(drop=True)

@lru_cache(maxsize=1)
def _lingua():
    from lingua import LanguageDetectorBuilder
    return LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()

def detect_languages(texts, max_chars=1000):
    """Código ISO 639-1 que detecta lingua sobre el inicio de cada texto (None si no es concluyente)."""
    results = _lingua().detect_languages_in_parallel_of([t[:max_chars] for t in texts])
    return [r.iso_code_639_1.name.lower() if r is not None else None for r in results]

# --- A1. Integridad ---

def schema_report(schema, expected_columns):
    """Columnas y orden del archivo frente a los del catálogo, incluidas las que faltan o sobran."""
    expected = pd.DataFrame({'column': list(expected_columns), 'expected_position': range(len(expected_columns))})
    out = schema.merge(expected, on='column', how='outer')
    out['status'] = np.select([out['position'].isna(), out['expected_position'].isna(),
                               out['position'] != out['expected_position']],
                              ['falta', 'no_esperada', 'otra_posicion'], 'ok')
    return out.sort_values(['expected_position', 'position']).reset_index(drop=True)

def completeness(df):
    """Por columna: filas, nulos, vacíos ('' o lista vacía), solo espacios y con contenido.

    Los conteos se pueden sumar entre lotes.
    """
    rows = []
    for col in df.columns:
        values = df[col]
        present = values[values.notna()]
        first = present.iloc[0] if len(present) else None
        empty = whitespace = 0
        if isinstance(first, str):
            empty = int(present.eq('').sum())
            whitespace = int(present.str.strip().eq('').sum()) - empty
        elif isinstance(first, (list, tuple, np.ndarray)):
            empty = int(present.map(len).eq(0).sum())
        nulls = len(values) - len(present)
        rows.append({'column': col, 'rows': len(values), 'nulls': nulls, 'empty': empty,
                     'whitespace_only': whitespace, 'nonempty': len(values) - nulls - empty - whitespace})
    return pd.DataFrame(rows)

def compare_nonempty(comp, manifest_nonempty=None):
    out = comp.copy()
    out['manifest_nonempty'] = out['column'].map(manifest_nonempty or {})
    out['diff_vs_manifest'] = out['nonempty'] - out['manifest_nonempty']
    return out

def uniqueness(df, columns):
    rows = []
    for col in columns:
        values = df[col].dropna()
        distinct = values.nunique()
        rows.append({'column': col, 'non_null': len(values), 'distinct': distinct, 'duplicate_rows': len(values) - distinct})
    return pd.DataFrame(rows)

def recount(stored, features):
    """Filas cuyo valor guardado difiere del recalculado.

    Definiciones: char_count = len(text), word_count = len(text.split()), text_sha256 = sha256 del texto en UTF-8.
    """
    merged = stored[['document_id', *RECOUNT]].merge(features[['document_id', *RECOUNT.values()]],
                                                     on='document_id', validate='one_to_one')
    out = pd.DataFrame([{'column': col, 'rows': len(merged), 'mismatches': int((merged[col] != merged[calc]).sum())}
                        for col, calc in RECOUNT.items()])
    out['match_rate'] = 1 - out['mismatches'] / out['rows']
    return out

def expected_counts(df, expected):
    """Conteos observados frente a los publicados. `expected`: {'total': n, 'collection_name': {...}, 'source': {...}}."""
    rows = [{'field': 'total', 'value': 'total', 'observed': len(df), 'expected': expected.get('total')}]
    for field in ('collection_name', 'source'):
        observed = df[field].value_counts()
        reference = expected.get(field) or {}
        for value in sorted(set(observed.index) | set(reference)):
            rows.append({'field': field, 'value': value, 'observed': int(observed.get(value, 0)),
                         'expected': reference.get(value)})
    out = pd.DataFrame(rows)
    out['diff'] = out['observed'] - out['expected']
    return out

# --- A2. Distribuciones ---

def counts(df, columns):
    out = df.groupby(columns, dropna=False).size().rename('documents').reset_index()
    out['share'] = out['documents'] / len(df)
    return out.sort_values('documents', ascending=False).reset_index(drop=True)

def percentiles(df, columns, by='collection_name', q=PERCENTILES):
    rows = []
    for group, sub in _groups(df, by):
        for col in columns:
            values = sub[col].dropna()
            rows.append({by: group, 'metric': col, 'n': len(values), 'mean': values.mean(), 'min': values.min(),
                         **{f'p{p}': values.quantile(p / 100) for p in q}, 'max': values.max()})
    return pd.DataFrame(rows)

def share_above(df, column, thresholds, by='collection_name'):
    rows = []
    for group, sub in _groups(df, by):
        rows.append({by: group, 'n': len(sub), **{f'share_gt_{t}': (sub[column] > t).mean() for t in thresholds}})
    return pd.DataFrame(rows)

def language_matrix(df, min_share=0.01):
    """Idioma declarado (`language_base`) × detectado. Los idiomas poco frecuentes se agrupan en 'otro'."""
    declared = group_rare(df['language_base'], min_share)
    keep = set(declared) - {'otro', 'desconocido'}
    detected = df['language_detected'].map(lambda v: 'indeterminado' if pd.isna(v) else (v if v in keep else 'otro'))
    return pd.crosstab(declared.rename('declarado'), detected.rename('detectado'))

def language_agreement(df, by='collection_name'):
    """Tasa de acuerdo entre `language_base` y el idioma detectado, donde ambos existen."""
    rows = []
    for group, sub in _groups(df, by):
        both = sub['language_base'].notna() & sub['language_detected'].notna()
        agree = sub.loc[both, 'language_base'] == sub.loc[both, 'language_detected']
        rows.append({by: group, 'n': len(sub), 'declared_null': int(sub['language_base'].isna().sum()),
                     'detected_null': int(sub['language_detected'].isna().sum()), 'compared': int(both.sum()),
                     'agreement': agree.mean() if both.any() else np.nan})
    return pd.DataFrame(rows)

def agreement_by_length(df, bins, by='collection_name', length='char_count'):
    both = df[df['language_base'].notna() & df['language_detected'].notna()]
    agree = (both['language_base'] == both['language_detected']).rename('agree')
    labels = [f'{int(lo)}–{int(hi)}' for lo, hi in zip(bins[:-2], bins[1:-1])] + [f'≥{int(bins[-2])}']
    bucket = pd.cut(both[length], bins=bins, right=False, labels=labels).rename('length_bin')
    out = agree.groupby([both[by], bucket], observed=True).agg(['size', 'mean'])
    return out.rename(columns={'size': 'n', 'mean': 'agreement'}).reset_index()

def parse_dates(values):
    return pd.to_datetime(values, utc=True, errors='coerce', format='mixed')

def _date_quantiles(values, q):
    values = values.dropna()
    return {f'p{p}': values.quantile(p / 100) if len(values) else pd.NaT for p in q}

def date_report(df, version_date, plausible_start=None, by='collection_name'):
    """Formato, rango y anomalías de `created_at`.

    Cuenta nulos, fechas sin hora, fechas posteriores a `synced_at` o a la versión del corpus, y fechas
    anteriores al inicio plausible de cada colección (`plausible_start`: {colección: fecha}).
    """
    created, synced = parse_dates(df['created_at']), parse_dates(df['synced_at'])
    kind = pd.Series(np.select([df['created_at'].isna(), df['created_at'].str.fullmatch(DATE_ONLY, na=False), created.notna()],
                               ['nulo', 'solo_fecha', 'fecha_hora'], 'no_interpretable'), index=df.index)
    cutoff = pd.Timestamp(version_date, tz='UTC') + pd.DateOffset(days=1)
    starts = df[by].map({k: pd.Timestamp(v, tz='UTC') for k, v in (plausible_start or {}).items()})
    data = pd.DataFrame({by: df[by], 'kind': kind, 'created': created, 'after_synced': created > synced,
                         'after_version': created >= cutoff, 'before_plausible_start': created < starts})
    rows = []
    for group, sub in _groups(data, by):
        kinds = sub['kind'].value_counts()
        rows.append({by: group, 'n': len(sub), **{k: int(kinds.get(k, 0)) for k in ('nulo', 'solo_fecha', 'fecha_hora', 'no_interpretable')},
                     'min': sub['created'].min(), **_date_quantiles(sub['created'], (1, 50, 99)), 'max': sub['created'].max(),
                     'after_synced': int(sub['after_synced'].sum()), 'after_version': int(sub['after_version'].sum()),
                     'before_plausible_start': int(sub['before_plausible_start'].sum())})
    return pd.DataFrame(rows)

def period_counts(df, freq='M', by='collection_name'):
    """Documentos por periodo de `created_at` ('M' mensual, 'Y' anual) y su proporción dentro de cada colección."""
    created = parse_dates(df['created_at'])
    period = created.dt.strftime('%Y-%m' if freq == 'M' else '%Y').rename('period')
    out = df.groupby([df[by], period]).size().rename('documents').reset_index()
    out['share'] = out['documents'] / out.groupby(by)['documents'].transform('sum')
    return out

# --- A3. Duplicados ---

def exact_duplicates(df, top=20):
    """Duplicados exactos por `text_sha256`. Es 'extra' toda fila posterior a la primera (por `document_id`) de cada texto.

    `largest` describe los `top` grupos más grandes solo con metadatos (tamaño, colección, dominios, longitud).
    """
    data = df[['document_id', 'collection_name', 'url_domain', 'char_count', 'text_sha256']].sort_values('document_id').copy()
    groups = data.groupby('text_sha256')
    data['group_size'] = groups['document_id'].transform('size')
    data['is_extra'] = groups.cumcount() > 0
    data['cross_collection'] = groups['collection_name'].transform('nunique') > 1
    rows = []
    for group, sub in _groups(data, 'collection_name'):
        rows.append({'collection_name': group, 'rows': len(sub), 'unique_texts': sub['text_sha256'].nunique(),
                     'extra_rows': int(sub['is_extra'].sum()), 'rows_in_duplicate_groups': int((sub['group_size'] > 1).sum()),
                     'rows_in_cross_collection_groups': int(sub['cross_collection'].sum())})
    by_collection = pd.DataFrame(rows)
    by_collection['extra_share'] = by_collection['extra_rows'] / by_collection['rows']
    firsts = data.drop_duplicates('text_sha256')
    group_sizes = firsts['group_size'].value_counts().sort_index().rename('groups').rename_axis('group_size').reset_index()
    group_sizes['rows'] = group_sizes['group_size'] * group_sizes['groups']
    by_domain = data.groupby(['collection_name', 'url_domain']).agg(rows=('document_id', 'size'), extra_rows=('is_extra', 'sum')).reset_index()
    by_domain['extra_share'] = by_domain['extra_rows'] / by_domain['rows']
    by_domain = by_domain.sort_values('extra_rows', ascending=False).reset_index(drop=True)
    largest = (data[data['group_size'] > 1].groupby('text_sha256')
               .agg(group_size=('document_id', 'size'), collection_name=('collection_name', 'first'),
                    collections=('collection_name', 'nunique'), domains=('url_domain', 'nunique'),
                    top_domain=('url_domain', lambda s: s.value_counts().index[0]), char_count=('char_count', 'first'))
               .sort_values(['group_size', 'top_domain'], ascending=[False, True]).head(top).reset_index(drop=True))
    largest.insert(0, 'rank', range(1, len(largest) + 1))
    return {'by_collection': by_collection, 'group_sizes': group_sizes, 'by_domain': by_domain, 'largest': largest,
            'cross_collection_groups': int(firsts['cross_collection'].sum())}

def shingles(text, k=5):
    words = text.lower().split()
    if len(words) <= k: return {' '.join(words)} if words else set()
    return {' '.join(words[i:i + k]) for i in range(len(words) - k + 1)}

def near_duplicate_pairs(texts, k=5, num_perm=128, threshold=0.8, seed=42):
    """Pares con Jaccard de shingles >= `threshold`: MinHash/LSH propone candidatos y se confirma con el Jaccard real.

    `texts`: Serie indexada por `document_id`.
    """
    from datasketch import MinHash, MinHashLSH
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    sets, sketches = {}, {}
    for doc_id, text in texts.items():
        items = shingles(text or '', k)
        if not items: continue
        sketch = MinHash(num_perm=num_perm, seed=seed)
        sketch.update_batch([s.encode('utf-8') for s in items])
        lsh.insert(doc_id, sketch)
        sets[doc_id], sketches[doc_id] = items, sketch
    candidates = {tuple(sorted((a, b))) for a, sketch in sketches.items() for b in lsh.query(sketch) if a != b}
    rows = [(a, b, len(sets[a] & sets[b]) / len(sets[a] | sets[b])) for a, b in sorted(candidates)]
    pairs = pd.DataFrame(rows, columns=['document_id_a', 'document_id_b', 'jaccard'])
    return pairs[pairs['jaccard'] >= threshold].reset_index(drop=True)

def typify_pairs(pairs, meta):
    """Clasifica pares por metadatos, sin texto: exacto, entre colecciones, mismo dominio u otro dominio."""
    info = meta.set_index('document_id')[['collection_name', 'url_domain', 'text_sha256', 'char_count']]
    out = pairs.join(info.add_suffix('_a'), on='document_id_a').join(info.add_suffix('_b'), on='document_id_b')
    exact = out['text_sha256_a'] == out['text_sha256_b']
    same_collection = out['collection_name_a'] == out['collection_name_b']
    same_domain = out['url_domain_a'] == out['url_domain_b']
    out['type'] = np.select([exact, ~same_collection, same_domain], ['exacto', 'entre_colecciones', 'mismo_dominio'], 'otro_dominio')
    out['length_ratio'] = out[['char_count_a', 'char_count_b']].min(axis=1) / out[['char_count_a', 'char_count_b']].max(axis=1)
    return out[['document_id_a', 'document_id_b', 'type', 'jaccard', 'length_ratio',
                'collection_name_a', 'collection_name_b', 'url_domain_a', 'url_domain_b']]

def near_duplicate_summary(typed, sample_meta):
    """Proporción de documentos de la muestra con algún casi duplicado (sin contar los exactos), por colección."""
    near = typed[typed['type'] != 'exacto']
    with_near = set(near['document_id_a']) | set(near['document_id_b'])
    with_any = set(typed['document_id_a']) | set(typed['document_id_b'])
    rows = []
    for group, sub in _groups(sample_meta, 'collection_name'):
        ids = set(sub['document_id'])
        rows.append({'collection_name': group, 'documents': len(ids), 'with_near_duplicate': len(ids & with_near),
                     'with_any_duplicate': len(ids & with_any)})
    out = pd.DataFrame(rows)
    out['share_near_duplicate'] = out['with_near_duplicate'] / out['documents']
    out['share_any_duplicate'] = out['with_any_duplicate'] / out['documents']
    return out

def pair_types(typed):
    out = typed.groupby('type').agg(pairs=('jaccard', 'size'), jaccard_median=('jaccard', 'median'),
                                    length_ratio_median=('length_ratio', 'median')).reset_index()
    out['share'] = out['pairs'] / out['pairs'].sum()
    return out

def pair_examples(typed, per_type=5, seed=42):
    """Ejemplos por tipo: solo IDs, dominios y medidas, nunca el texto."""
    parts = [group.sample(min(per_type, len(group)), random_state=seed) for _, group in typed.groupby('type')]
    if not parts: return typed.head(0)
    return pd.concat(parts).sort_values(['type', 'jaccard'], ascending=[True, False]).reset_index(drop=True)

# --- A4. URLs y calidad del texto ---

def _tracking_keys(parts, patterns):
    if parts is None or not parts.query: return []
    return [k.lower() for k, _ in parse_qsl(parts.query, keep_blank_values=True) if any(fnmatch(k.lower(), p) for p in patterns)]

def url_report(df, tracking_patterns, by='collection_name'):
    parts = df['url'].map(lambda u: urlsplit(u) if isinstance(u, str) else None)
    scheme = parts.map(lambda p: p.scheme.lower() if p else None)
    data = pd.DataFrame({
        by: df[by], 'null': df['url'].isna(),
        'valid': scheme.isin(['http', 'https']) & parts.map(lambda p: bool(p and p.netloc)),
        'https': scheme.eq('https'), 'with_query': parts.map(lambda p: bool(p and p.query)),
        'with_tracking': parts.map(lambda p: bool(_tracking_keys(p, tracking_patterns))),
        'duplicated': df['url'].duplicated(keep='first') & df['url'].notna(),
        'final_url_differs': df['final_url'].notna() & (df['final_url'] != df['url']),
    })
    rows = [{by: group, 'n': len(sub), **{c: int(sub[c].sum()) for c in data.columns if c != by}} for group, sub in _groups(data, by)]
    return pd.DataFrame(rows)

def tracking_params(df, patterns, by='collection_name'):
    """Documentos por parámetro de tracking en la URL (solo el nombre del parámetro)."""
    keys = df['url'].map(lambda u: sorted(set(_tracking_keys(urlsplit(u), patterns))) if isinstance(u, str) else [])
    exploded = pd.DataFrame({by: df[by], 'param': keys}).explode('param').dropna()
    return exploded.groupby([by, 'param']).size().rename('documents').reset_index().sort_values('documents', ascending=False)

def top_domains(df, n=20, collection='url_crawls'):
    sub = df[df['collection_name'] == collection]
    out = sub['url_domain'].value_counts().head(n).rename('documents').rename_axis('url_domain').reset_index()
    out['share'] = out['documents'] / len(sub)
    out['cumulative_share'] = out['share'].cumsum()
    return out

def noise_summary(features, short_words, by='collection_name'):
    rows = []
    for group, sub in _groups(features, by):
        for flag, label in NOISE_FLAGS.items():
            if flag == 'is_short': label = f'{label} (< {short_words} palabras)'
            rows.append({by: group, 'flag': flag, 'label': label, 'documents': int(sub[flag].sum()), 'share': sub[flag].mean()})
    return pd.DataFrame(rows)

def noise_by_domain(features, flags, n=10):
    """Dominios con más documentos marcados por cada señal de ruido (indica si el problema es de una fuente)."""
    parts = []
    for flag in flags:
        out = features.groupby('url_domain')[flag].agg(['size', 'sum']).rename(columns={'size': 'documents', 'sum': 'flagged'})
        out = out[out['flagged'] > 0].sort_values('flagged', ascending=False).head(n).reset_index()
        out.insert(0, 'flag', flag)
        out['share_of_domain'] = out['flagged'] / out['documents']
        out['share_of_flagged'] = out['flagged'] / features[flag].sum()
        parts.append(out)
    columns = ['flag', 'url_domain', 'documents', 'flagged', 'share_of_domain', 'share_of_flagged']
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)

def boilerplate_lines(lines, domains, min_docs=20):
    """Pares (dominio, hash de línea) que aparecen en al menos `min_docs` documentos del dominio."""
    data = lines.merge(domains[['document_id', 'url_domain']], on='document_id')
    out = data.groupby(['url_domain', 'line_hash']).size().rename('documents').reset_index()
    return out[out['documents'] >= min_docs].reset_index(drop=True)

def boilerplate(lines, domains, min_docs=20):
    """Líneas que aparecen en al menos `min_docs` documentos del mismo dominio.

    Devuelve por documento las líneas y las líneas boilerplate, y por dominio cuántas líneas distintas lo son.
    """
    data = lines.merge(domains[['document_id', 'url_domain']], on='document_id')
    data['is_boilerplate'] = data.groupby(['url_domain', 'line_hash'])['document_id'].transform('size') >= min_docs
    per_doc = data.groupby('document_id').agg(n_lines_checked=('line_hash', 'size'),
                                              n_boilerplate_lines=('is_boilerplate', 'sum')).reset_index()
    per_domain = (data[data['is_boilerplate']].drop_duplicates(['url_domain', 'line_hash'])
                  .groupby('url_domain').size().rename('boilerplate_lines').reset_index())
    return per_doc, per_domain

def boilerplate_summary(per_doc, per_domain, meta, n=20):
    data = per_doc.merge(meta[['document_id', 'url_domain']], on='document_id')
    data['has_boilerplate'] = data['n_boilerplate_lines'] > 0
    data['boilerplate_share'] = data['n_boilerplate_lines'] / data['n_lines_checked']
    out = data.groupby('url_domain').agg(documents=('document_id', 'size'), with_boilerplate=('has_boilerplate', 'sum'),
                                         mean_boilerplate_share=('boilerplate_share', 'mean')).reset_index()
    out = out.merge(per_domain, on='url_domain', how='left').fillna({'boilerplate_lines': 0})
    out['share_with_boilerplate'] = out['with_boilerplate'] / out['documents']
    total = pd.DataFrame([{'url_domain': 'total', 'documents': len(data), 'with_boilerplate': int(data['has_boilerplate'].sum()),
                           'mean_boilerplate_share': data['boilerplate_share'].mean(),
                           'boilerplate_lines': per_domain['boilerplate_lines'].sum(),
                           'share_with_boilerplate': data['has_boilerplate'].mean()}])
    return pd.concat([out.sort_values('documents', ascending=False).head(n), total], ignore_index=True)

# --- A5. Sesgos de recolección ---

def domain_concentration(df, collection='url_crawls'):
    shares = df.loc[df['collection_name'] == collection, 'url_domain'].value_counts(normalize=True)
    return pd.DataFrame([{'collection_name': collection, 'domains': len(shares), 'top5_share': shares.head(5).sum(),
                          'top20_share': shares.head(20).sum(), 'hhi': float((shares ** 2).sum())}])

def acquisition_fields(df, columns, by='collection_name'):
    """Distribución descriptiva de los campos de adquisición. No son etiquetas semánticas."""
    parts = []
    for col in columns:
        out = df.groupby([df[by], df[col].fillna('(nulo)').rename('value')]).size().rename('documents').reset_index()
        out.insert(0, 'field', col)
        out['share'] = out['documents'] / out.groupby(by)['documents'].transform('sum')
        parts.append(out)
    return pd.concat(parts, ignore_index=True)

def list_lengths(df, column, by='collection_name'):
    """Distribución del número de elementos de una columna de listas (p. ej. `keyword_matches`)."""
    lengths = df[column].map(lambda v: 0 if v is None else len(v))
    return percentiles(pd.DataFrame({by: df[by], f'{column}_length': lengths}), [f'{column}_length'], by=by)
