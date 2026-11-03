"""Orquestación del perfilado completo (Fase A): integridad, distribuciones, duplicados, calidad, sesgos y muestras.

Lee el Parquet sin modificarlo y escribe:
- `out_dir`: tablas CSV, figuras PNG y profile.json (solo agregados; se versiona);
- `samples_dir`: listas de IDs y metadatos de las muestras (se versiona);
- `work_dir`: features por documento y pares de casi duplicados (no se versiona).
"""
import platform, time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import pandas as pd
from pipeline import io, profiling as prof, report, sampling

PACKAGES = ('pandas', 'pyarrow', 'numpy', 'scipy', 'matplotlib', 'lingua-language-detector', 'datasketch',
            'transformers', 'sentence-transformers', 'open_corpus_co_es')
FEATURE_COLUMNS = ['collection_name', 'language_base', 'url_domain', 'char_count', 'word_count', 'text_sha256']

def _versions():
    out = {'python': platform.python_version()}
    for name in PACKAGES:
        try: out[name] = version(name)
        except PackageNotFoundError: out[name] = None
    return out

def _expected(manifest, catalog):
    """Conteos publicados: primero el manifiesto, si no el catálogo."""
    if manifest:
        return {'total': manifest.get('rows'), **(manifest.get('distributions') or {})}
    if catalog:
        return {'total': catalog.get('filas'), 'source': catalog.get('fuentes') or {}}
    return {}

def run(parquet, cfg, out_dir, samples_dir, work_dir, *, count_tokens, max_tokens, detect_languages,
        catalog=None, encoder_settings=None, log=print):
    start = time.perf_counter()
    parquet, out_dir, samples_dir, work_dir = map(Path, (parquet, out_dir, samples_dir, work_dir))
    tables, figures = {}, []

    log('Integridad del archivo')
    digest = io.sha256_file(parquet)
    manifest = io.load_manifest(io.manifest_path(parquet))
    schema = io.parquet_schema(parquet)
    columns = list(schema['column'])
    expected_columns = (catalog or {}).get('campos') or columns
    meta = io.read_columns(parquet, [c for c in columns if c != 'text'])

    log('Pasada por lotes sobre el texto (features, tokens, idioma, boilerplate)')
    noise_cfg, lang_cfg = cfg['noise'], cfg['language']
    crawl_ids = set(meta.loc[meta['collection_name'].isin(noise_cfg['boilerplate_collections']), 'document_id'])
    feature_parts, text_completeness, line_parts = [], [], []
    for batch in io.iter_batches(parquet, ['document_id', 'text'], cfg['batch_size']):
        text_completeness.append(prof.completeness(batch[['text']]))
        features = prof.text_features(batch, noise_cfg['short_words'])
        texts = batch['text'].fillna('').tolist()
        features['n_tokens'] = count_tokens(texts)
        features['language_detected'] = detect_languages(texts, lang_cfg['max_chars'])
        feature_parts.append(features)
        crawl = batch[batch['document_id'].isin(crawl_ids)]
        if len(crawl): line_parts.append(prof.line_hashes(crawl, noise_cfg['boilerplate_min_chars']))
    features = pd.concat(feature_parts, ignore_index=True)
    per_doc_bp, per_domain_bp = prof.boilerplate(pd.concat(line_parts, ignore_index=True) if line_parts else
                                                 pd.DataFrame(columns=['document_id', 'line_hash']), meta,
                                                 noise_cfg['boilerplate_min_docs'])
    features = (features.merge(meta[['document_id', *FEATURE_COLUMNS]], on='document_id', validate='one_to_one')
                .merge(per_doc_bp, on='document_id', how='left'))
    work_dir.mkdir(parents=True, exist_ok=True)
    features.to_parquet(work_dir / 'document_features.parquet', index=False)

    log('A1 integridad')
    completeness = pd.concat([prof.completeness(meta), *text_completeness]).groupby('column', sort=False).sum().reset_index()
    completeness = completeness.set_index('column').reindex(columns).reset_index()
    tables['integridad_esquema'] = prof.schema_report(schema, expected_columns)
    tables['integridad_completitud'] = prof.compare_nonempty(completeness, (manifest or {}).get('nonempty_counts'))
    tables['integridad_unicidad'] = prof.uniqueness(meta, ['document_id', 'mongo_id', 'url', 'uri', 'text_sha256'])
    tables['integridad_recalculo'] = prof.recount(meta, features)
    tables['integridad_conteos'] = prof.expected_counts(meta, _expected(manifest, catalog))

    log('A2 distribuciones')
    thresholds = cfg['tokens']['thresholds']
    tables['distribucion_coleccion_fuente'] = prof.counts(meta, ['collection_name', 'source'])
    tables['distribucion_longitudes'] = prof.percentiles(features, ['char_count_calc', 'word_count_calc', 'n_tokens'])
    tables['distribucion_tokens_umbral'] = prof.share_above(features, 'n_tokens', [*thresholds, max_tokens] if max_tokens not in thresholds else thresholds)
    tables['idioma_declarado'] = prof.counts(meta, ['collection_name', 'language_base'])
    matrix = prof.language_matrix(features, lang_cfg['min_share'])
    tables['idioma_matriz'] = matrix.reset_index()
    tables['idioma_acuerdo'] = prof.language_agreement(features)
    tables['idioma_acuerdo_longitud'] = prof.agreement_by_length(features, lang_cfg['length_bins'])
    version_date = (catalog or {}).get('fecha_version') or cfg.get('version_date')
    tables['fechas_resumen'] = prof.date_report(meta, version_date, cfg['dates']['plausible_start'])
    tables['fechas_mensual'] = prof.period_counts(meta, 'M')
    tables['fechas_anual'] = prof.period_counts(meta, 'Y')

    log('A6 muestras')
    sample_cfg = cfg['samples']
    samples, strata = sampling.nested_samples(meta, sample_cfg['sizes'], sample_cfg['strata'], sample_cfg['min_share'], cfg['seed'])
    created_at = datetime.now(timezone.utc).isoformat(timespec='seconds')
    sample_info, distributions, checks = {}, [], []
    for n, ids in samples.items():
        dist = sampling.distribution(strata, ids)
        check = sampling.representativeness(meta, ids, sample_cfg['strata'], sample_cfg['min_share'])
        paths = sampling.write_sample(samples_dir, n, ids, {
            'name': f'sample_{n}', 'size': len(ids), 'seed': cfg['seed'], 'created_at_utc': created_at,
            'parquet': parquet.name, 'parquet_sha256': digest, 'strata': sample_cfg['strata'],
            'min_share': sample_cfg['min_share'], 'nested_in': [m for m in samples if m > n],
            'distribution': report.to_jsonable(dist), 'representativeness': report.to_jsonable(check)})
        distributions.append(dist.assign(sample=n))
        checks.append(check.assign(sample=n))
        sample_info[n] = {'size': len(ids), **paths, 'max_abs_share_diff': float(dist['diff'].abs().max())}
    full = sampling.distribution(strata, strata.index)
    sample_info['full'] = {'size': len(strata), **sampling.write_sample(samples_dir, 'full', None, {
        'name': 'sample_full', 'size': len(strata), 'ids': 'todas las filas del Parquet', 'created_at_utc': created_at,
        'parquet': parquet.name, 'parquet_sha256': digest, 'strata': sample_cfg['strata'],
        'min_share': sample_cfg['min_share'], 'distribution': report.to_jsonable(full)})}
    tables['muestras_distribucion'] = pd.concat(distributions, ignore_index=True)
    tables['muestras_representatividad'] = pd.concat(checks, ignore_index=True)

    log('A3 duplicados')
    exact = prof.exact_duplicates(meta)
    tables['duplicados_exactos_coleccion'] = exact['by_collection']
    tables['duplicados_exactos_tamano_grupo'] = exact['group_sizes']
    tables['duplicados_exactos_dominio'] = exact['by_domain'].head(cfg['urls']['top_domains'])
    tables['duplicados_exactos_grupos_top'] = exact['largest']
    near_cfg = cfg['near_duplicates']
    near_ids = samples[near_cfg['sample']]
    texts = io.read_rows(parquet, ['document_id', 'text'], near_ids).set_index('document_id')['text']
    pairs = prof.near_duplicate_pairs(texts, near_cfg['shingle_words'], near_cfg['num_perm'], near_cfg['threshold'], cfg['seed'])
    typed = prof.typify_pairs(pairs, meta)
    typed.to_parquet(work_dir / f'near_duplicate_pairs_{near_cfg["sample"]}.parquet', index=False)
    tables['casi_duplicados_resumen'] = prof.near_duplicate_summary(typed, meta[meta['document_id'].isin(set(near_ids))])
    tables['casi_duplicados_tipos'] = prof.pair_types(typed)
    tables['casi_duplicados_ejemplos'] = prof.pair_examples(typed, near_cfg['examples_per_type'], cfg['seed'])

    log('A4 URLs y calidad del texto')
    url_cfg = cfg['urls']
    tables['urls_resumen'] = prof.url_report(meta, url_cfg['tracking_params'])
    tables['urls_tracking'] = prof.tracking_params(meta, url_cfg['tracking_params'])
    top = prof.top_domains(meta, url_cfg['top_domains'])
    tables['dominios_top'] = top
    noise = prof.noise_summary(features, noise_cfg['short_words'])
    tables['ruido_coleccion'] = noise
    tables['ruido_dominio'] = prof.noise_by_domain(features, [f for f in prof.NOISE_FLAGS if features[f].any()])
    tables['boilerplate_dominio'] = prof.boilerplate_summary(per_doc_bp, per_domain_bp, meta, url_cfg['top_domains'])

    log('A5 sesgos de recolección')
    tables['sesgo_concentracion_dominios'] = prof.domain_concentration(meta)
    tables['sesgo_campos_adquisicion'] = prof.acquisition_fields(meta, cfg['acquisition_fields'])
    tables['sesgo_keyword_matches'] = prof.list_lengths(meta, 'keyword_matches')
    tables['sesgo_cardinalidad'] = prof.uniqueness(meta, cfg['acquisition_fields'] + ['collection_query'])

    log('Figuras y profile.json')
    figures_dir = out_dir / 'figures'
    figures.append(report.token_ecdf(features, thresholds, max_tokens, figures_dir / 'tokens_ecdf.png'))
    figures.append(report.length_histograms(features, figures_dir / 'longitud_caracteres.png'))
    figures.append(report.language_heatmap(matrix, figures_dir / 'idioma_matriz.png'))
    figures.append(report.agreement_by_length_plot(tables['idioma_acuerdo_longitud'], figures_dir / 'idioma_acuerdo_longitud.png'))
    figures.append(report.monthly_counts_plot(tables['fechas_mensual'], figures_dir / 'fechas_mensual.png'))
    figures.append(report.top_domains_plot(top, figures_dir / 'dominios_top.png'))
    figures.append(report.noise_plot(noise, figures_dir / 'ruido.png'))
    table_files = report.write_tables(tables, out_dir / 'tables')

    recount = tables['integridad_recalculo'].set_index('column')['mismatches']
    schema_status = tables['integridad_esquema']['status']
    counts = tables['integridad_conteos']
    agreement = tables['idioma_acuerdo'].set_index('collection_name')['agreement']
    tokens = tables['distribucion_tokens_umbral'].set_index('collection_name')
    exact_total = exact['by_collection'].set_index('collection_name').loc['total']
    near_total = tables['casi_duplicados_resumen'].set_index('collection_name').loc['total']
    concentration = tables['sesgo_concentracion_dominios'].iloc[0]
    profile = {
        'run': {'created_at_utc': created_at, 'duration_seconds': round(time.perf_counter() - start, 1),
                'parquet': {'path': str(parquet), 'sha256': digest, 'bytes': parquet.stat().st_size, **io.parquet_metadata(parquet)},
                'manifest_available': manifest is not None, 'config': cfg, 'versions': _versions(),
                'encoder': {**(encoder_settings or {}), 'max_tokens': max_tokens}, 'language_detector': 'lingua (todos los idiomas, alta precisión)'},
        'integrity': {
            'sha256_matches_manifest': (manifest or {}).get('output_sha256') == digest if manifest else None,
            'sha256_matches_catalog': (catalog or {}).get('sha256') == digest if catalog else None,
            'columns': len(columns), 'schema_ok': bool((schema_status == 'ok').all()),
            'schema_issues': tables['integridad_esquema'].loc[schema_status != 'ok', ['column', 'status']],
            'document_id_unique': bool(tables['integridad_unicidad'].set_index('column').loc['document_id', 'duplicate_rows'] == 0),
            'recount_mismatches': recount.to_dict(),
            'counts_match_published': bool(counts['expected'].notna().all() and pd.to_numeric(counts['diff']).eq(0).all()),
            'nonempty_diff_vs_manifest': tables['integridad_completitud'].set_index('column')['diff_vs_manifest'].dropna().loc[lambda s: s != 0].to_dict()},
        'distributions': {'documents': len(meta), 'by_collection': meta['collection_name'].value_counts().to_dict(),
                          'tokens_share_above': tokens.drop(columns='n').to_dict(orient='index'),
                          'language_agreement': agreement.to_dict()},
        'duplicates': {'unique_texts': int(meta['text_sha256'].nunique()), 'exact_extra_rows': int(exact_total['extra_rows']),
                       'cross_collection_groups': exact['cross_collection_groups'],
                       'near_duplicate_sample': near_cfg['sample'], 'near_duplicate_pairs': len(typed),
                       'share_docs_with_near_duplicate': float(near_total['share_near_duplicate'])},
        'urls': tables['urls_resumen'].set_index('collection_name').loc['total'].to_dict(),
        'noise': noise[noise['collection_name'] == 'total'].set_index('flag')['share'].to_dict(),
        'bias': {'top5_domain_share': float(concentration['top5_share']), 'top20_domain_share': float(concentration['top20_share']),
                 'domains': int(concentration['domains'])},
        'samples': sample_info,
        'outputs': {'tables': table_files, 'figures': figures,
                    'work': [str(work_dir / 'document_features.parquet'), str(work_dir / f'near_duplicate_pairs_{near_cfg["sample"]}.parquet')]},
    }
    profile = report.to_jsonable(profile)
    report.write_json(profile, out_dir / 'profile.json')
    return profile
