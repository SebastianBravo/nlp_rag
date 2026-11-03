"""Orquestación de la segmentación (Fase B): chunks de una estrategia sobre una muestra y comparación entre estrategias.

Escribe:
- `chunks_dir/<estrategia>/<muestra>.parquet`: los chunks con su texto (no se versiona);
- `reports_dir/runs/<estrategia>_<muestra>.json`: parámetros, versiones y métricas (se versiona);
- `reports_dir/tables` y `figures`: la comparación de las estrategias que ya corrieron sobre esa muestra.
"""
import json, platform, time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import numpy as np
import pandas as pd
from pipeline import chunking, dedup, io, report, sampling

META_COLUMNS = ['document_id', 'collection_name', 'source', 'url', 'title', 'language_base', 'created_at', 'url_domain', 'text_sha256']
CHUNK_COLUMNS = ['chunk_id', 'document_id', 'strategy', 'strategy_version', 'chunk_index', 'char_start', 'char_end',
                 'n_tokens', 'text', 'chunk_sha256', 'collection_name', 'source', 'url', 'title', 'language',
                 'language_detected', 'created_at', 'has_mojibake', 'provenance_ids']
PACKAGES = ('pandas', 'pyarrow', 'nltk', 'transformers', 'sentence-transformers', 'tokenizers')

def sample_name(sample):
    return 'full' if str(sample) == 'full' else Path(sample).stem.removesuffix('_ids')

def _versions():
    out = {'python': platform.python_version()}
    for name in PACKAGES:
        try: out[name] = version(name)
        except PackageNotFoundError: out[name] = None
    return out

def _union_length(spans):
    total, current = 0, None
    for s, e in sorted(spans):
        if current and s <= current[1]: current = (current[0], max(current[1], e))
        else:
            if current: total += current[1] - current[0]
            current = (s, e)
    return total + (current[1] - current[0] if current else 0)

def _boilerplate_by_domain(lines):
    if lines is None or lines.empty: return {}
    return {domain: frozenset(group['line_hash']) for domain, group in lines.groupby('url_domain')}

def chunk_stats(chunks, documents, max_tokens, by='collection_name'):
    """Métricas intrínsecas por colección (B4) sobre los chunks antes de deduplicar y el resumen por documento."""
    rows = []
    for group in [*sorted(documents[by].unique()), 'total']:
        docs = documents if group == 'total' else documents[documents[by] == group]
        sub = chunks if group == 'total' else chunks[chunks[by] == group]
        per_doc = docs['n_chunks']
        rows.append({
            by: group, 'documents': len(docs), 'documents_without_chunks': int((per_doc == 0).sum()),
            'chunks': len(sub), 'dropped_fragments': int(docs['dropped_fragments'].sum()),
            'chunks_per_doc_mean': per_doc.mean(), 'chunks_per_doc_p50': per_doc.median(),
            'chunks_per_doc_p95': per_doc.quantile(0.95), 'chunks_per_doc_max': per_doc.max(),
            'single_chunk_docs_share': (per_doc == 1).mean(),
            'n_tokens_p5': sub['n_tokens'].quantile(0.05), 'n_tokens_p50': sub['n_tokens'].median(),
            'n_tokens_p95': sub['n_tokens'].quantile(0.95), 'n_tokens_max': sub['n_tokens'].max(),
            'share_over_max_tokens': (sub['n_tokens'] > max_tokens).mean(),
            'mid_sentence_share': sub['mid_sentence'].mean(),
            'boilerplate_excluded_share': docs['excluded_chars'].sum() / max(docs['chars'].sum(), 1),
            'coverage_share': docs['covered_chars'].sum() / max(docs['chars'].sum(), 1),
            'overlap_ratio': docs['chunk_chars'].sum() / max(docs['covered_chars'].sum(), 1) - 1,
        })
    return pd.DataFrame(rows)

def run(parquet, sample, strategy, cfg, chunks_dir, reports_dir, *, token_offsets, count_tokens, max_tokens,
        features=None, boilerplate_lines=None, encoder_settings=None, splitter_for=None, log=print):
    """`splitter_for(idioma_punkt)` devuelve el separador de oraciones; por defecto, nltk punkt."""
    started = time.perf_counter()
    splitter_for = splitter_for or chunking.punkt_splitter
    parquet, chunks_dir, reports_dir = Path(parquet), Path(chunks_dir), Path(reports_dir)
    params = cfg['strategies'][strategy]
    name = sample_name(sample)

    log(f'Documentos de la muestra {name}')
    meta = io.read_columns(parquet, META_COLUMNS)
    template_cfg = cfg['documents']['exclude_template_groups']
    templates = chunking.template_documents(meta, template_cfg['collection'], template_cfg['min_copies']) if template_cfg else set()
    ids = list(meta['document_id']) if name == 'full' else sampling.read_sample_ids(sample)
    selected = [i for i in ids if i not in templates]
    info = meta.set_index('document_id').loc[selected]
    if features is not None:
        info = info.join(features.set_index('document_id')[['language_detected', 'has_mojibake']])
    else:
        info = info.assign(language_detected=None, has_mojibake=False)
    bp_by_domain = _boilerplate_by_domain(boilerplate_lines) if cfg['boilerplate']['exclude'] else {}
    languages, default_language = cfg['sentences']['languages'], cfg['sentences']['default']

    log(f'Segmentación {strategy} ({params["name"]})')
    wanted, rows, docs = set(selected), [], []
    t_chunk = time.perf_counter()
    for batch in io.iter_batches(parquet, ['document_id', 'text'], cfg['batch_size']):
        batch = batch[batch['document_id'].isin(wanted)]
        if batch.empty: continue
        texts = batch['text'].fillna('').tolist()
        for doc_id, text, offsets in zip(batch['document_id'], texts, token_offsets(texts)):
            row = info.loc[doc_id]
            code = row['language_base'] if pd.notna(row['language_base']) else row['language_detected']
            splitter = splitter_for(languages.get(code, default_language) if isinstance(code, str) else default_language)
            chunks, excluded = chunking.chunk_document(text, offsets, strategy, params, splitter,
                                                       bp_by_domain.get(row['url_domain'], frozenset()))
            chunks, dropped = chunking.drop_fragments(chunks, cfg['min_fragment_tokens'])
            spans = [(c['char_start'], c['char_end']) for c in chunks]
            docs.append({'document_id': doc_id, 'collection_name': row['collection_name'], 'chars': len(text),
                         'excluded_chars': excluded, 'n_chunks': len(chunks), 'dropped_fragments': dropped,
                         'covered_chars': _union_length(spans), 'chunk_chars': sum(e - s for s, e in spans)})
            for index, c in enumerate(chunks):
                rows.append({'document_id': doc_id, 'chunk_index': index, 'char_start': c['char_start'], 'char_end': c['char_end'],
                             'text': text[c['char_start']:c['char_end']], 'content_tokens': c['content_tokens'],
                             'mid_sentence': c['mid_sentence']})
    chunking_seconds = time.perf_counter() - t_chunk
    documents = pd.DataFrame(docs)
    chunks = pd.DataFrame(rows, columns=['document_id', 'chunk_index', 'char_start', 'char_end', 'text', 'content_tokens', 'mid_sentence'])

    log('Tokens del encoder, IDs y deduplicación')
    t_tokens = time.perf_counter()
    chunks['n_tokens'] = [n for start in range(0, len(chunks), 2000) for n in count_tokens(chunks['text'].iloc[start:start + 2000].tolist())]
    tokens_seconds = time.perf_counter() - t_tokens
    if (chunks['n_tokens'] > max_tokens).any():
        raise ValueError(f'{int((chunks["n_tokens"] > max_tokens).sum())} chunks superan {max_tokens} tokens del encoder.')
    chunks['strategy'], chunks['strategy_version'] = strategy, params['version']
    chunks['chunk_id'] = [chunking.chunk_id(d, strategy, params['version'], s, e)
                          for d, s, e in zip(chunks['document_id'], chunks['char_start'], chunks['char_end'])]
    chunks['chunk_sha256'] = dedup.text_sha256(chunks['text'])
    inherited = info[['collection_name', 'source', 'url', 'title', 'language_base', 'language_detected', 'created_at', 'has_mojibake']]
    chunks = chunks.join(inherited.rename(columns={'language_base': 'language'}), on='document_id')
    chunks['has_mojibake'] = chunks['has_mojibake'].eq(True)
    t_dedup = time.perf_counter()
    kept, removed = dedup.deduplicate(chunks)
    dedup_seconds = time.perf_counter() - t_dedup

    out = chunks_dir / strategy / f'{name}.parquet'
    out.parent.mkdir(parents=True, exist_ok=True)
    kept[CHUNK_COLUMNS].to_parquet(out, index=False)
    documents.to_parquet(out.with_suffix('.documents.parquet'), index=False)

    stats = chunk_stats(chunks, documents, max_tokens)
    dedup_table = dedup.dedup_summary(chunks, removed)
    stats = stats.merge(dedup_table.rename(columns={'chunks': 'chunks_before_dedup'}), on='collection_name')
    sample_ids = len(ids)
    run_info = report.to_jsonable({
        'strategy': strategy, 'strategy_version': params['version'], 'params': params, 'sample': name,
        'sample_path': str(sample), 'sample_documents': sample_ids, 'excluded_template_documents': sample_ids - len(selected),
        'created_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'parquet': {'path': str(parquet), 'sha256': io.sha256_file(parquet)},
        'config': cfg, 'encoder': {**(encoder_settings or {}), 'max_tokens': max_tokens}, 'versions': _versions(),
        'seconds': {'chunking': round(chunking_seconds, 2), 'encoder_tokens': round(tokens_seconds, 2),
                    'dedup': round(dedup_seconds, 2), 'total': round(time.perf_counter() - started, 2)},
        'chunks_per_second': round(len(chunks) / max(chunking_seconds, 1e-9), 1),
        'output': str(out), 'chunks_written': len(kept), 'stats': stats,
    })
    report.write_json(run_info, reports_dir / 'runs' / f'{strategy}_{name}.json')
    if cfg.get('example_text'):
        write_example(cfg['example_text'], strategy, params, reports_dir, token_offsets, count_tokens, languages.get('es', default_language))
    compare(reports_dir, chunks_dir, name, max_tokens, params['chunk_tokens'])
    return run_info

def write_example(path, strategy, params, reports_dir, token_offsets, count_tokens, language):
    """Segmenta un texto de ejemplo (sintético, no del corpus) para ilustrar las fronteras de cada estrategia."""
    text = Path(path).read_text(encoding='utf-8').strip()
    chunks, _ = chunking.chunk_document(text, token_offsets([text])[0], strategy, params, chunking.punkt_splitter(language))
    pieces = [text[c['char_start']:c['char_end']] for c in chunks]
    report.write_json({'strategy': strategy, 'params': params, 'text': text,
                       'chunks': [{**c, 'n_tokens': n} for c, n in zip(chunks, count_tokens(pieces))]},
                      Path(reports_dir) / 'ejemplos' / f'ejemplo_{strategy}.json')

def compare(reports_dir, chunks_dir, name, max_tokens, budget):
    """Tabla y figuras que comparan todas las estrategias ya ejecutadas sobre la muestra `name`."""
    runs = sorted(Path(reports_dir, 'runs').glob(f'*_{name}.json'))
    if not runs: return None
    infos = [json.loads(p.read_text(encoding='utf-8')) for p in runs]
    table = pd.concat([pd.DataFrame(i['stats']).assign(strategy=i['strategy'], strategy_version=i['strategy_version'],
                                                       chunking_seconds=i['seconds']['chunking'],
                                                       chunks_per_second=i['chunks_per_second']) for i in infos], ignore_index=True)
    lead = ['strategy', 'strategy_version', 'collection_name']
    table = table[lead + [c for c in table.columns if c not in lead]]
    report.write_tables({f'comparacion_{name}': table}, Path(reports_dir) / 'tables')
    chunk_data, doc_data = {}, {}
    for i in infos:
        path = Path(chunks_dir) / i['strategy'] / f'{name}.parquet'
        if path.is_file():
            chunk_data[i['strategy']] = pd.read_parquet(path, columns=['collection_name', 'n_tokens'])
            doc_data[i['strategy']] = pd.read_parquet(path.with_suffix('.documents.parquet'))
    if chunk_data:
        figures = Path(reports_dir) / 'figures'
        report.chunk_tokens_ecdf(chunk_data, max_tokens, budget, figures / f'tokens_{name}.png')
        report.chunks_per_document_plot(doc_data, figures / f'chunks_por_documento_{name}.png')
        report.strategy_metrics_plot(table, figures / f'metricas_{name}.png')
    return table
