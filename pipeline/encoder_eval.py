"""Análisis del encoder (Fase C) sobre los chunks vectorizados de una muestra, con las preguntas de dev del gold.

Calcula, sin Milvus (búsqueda exacta por coseno):
- métricas de recuperación a nivel documento por variante de consulta (formato del perfil, otros prefijos,
  paráfrasis, traducción y sin filtro de fecha);
- sensibilidad a recortar la dimensión del vector, con y sin renormalizar;
- estabilidad ante paráfrasis y traducción, desglose por idioma y vecinos de ejemplo (solo IDs y metadatos).
Solo usa el split `dev`: `test` queda intacto para la evaluación final.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pipeline import metrics, retrieval

CHUNK_COLUMNS = ['chunk_id', 'document_id', 'provenance_ids', 'char_start', 'text', 'collection_name', 'language_detected', 'created_at']
KS = (1, 5, 10, 20)
BASE = 'perfil'   # la consulta con el formato del perfil del encoder

def load_gold(gold_path, paraphrases_path, split='dev'):
    questions = [json.loads(l) for l in Path(gold_path).read_text(encoding='utf-8').splitlines() if l.strip()]
    questions = [q for q in questions if q['split'] == split and metrics.answerable(q)]
    paraphrases = {}
    if paraphrases_path and Path(paraphrases_path).is_file():
        paraphrases = {p['id']: p for p in map(json.loads, Path(paraphrases_path).read_text(encoding='utf-8').splitlines()) if p}
    return questions, paraphrases

def normalize(matrix):
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)

def run_variant(name, query_vectors, matrix, chunks, questions, doc_dates, spans, texts, use_window=True, k_chunks=300):
    """Rankings de documentos y métricas por pregunta para un conjunto de vectores de consulta."""
    provenance = chunks['provenance_ids'].tolist()
    rankings, extra, top_ids = {}, {}, {}
    for q, vector in zip(questions, query_vectors):
        allowed = retrieval.documents_in_window(doc_dates, q['ventana_fecha']) if use_window and q.get('ventana_fecha') else None
        mask = None
        if allowed is not None:
            mask = np.array([any(d in allowed for d in p) for p in provenance])[None, :]
        idx, _ = retrieval.top_chunks(vector[None, :], matrix, k_chunks, mask)
        rankings[q['id']] = retrieval.rank_documents(idx[0], provenance, allowed, limit=max(KS))
        q_spans = {d: s for d, s in spans[q['id']].items()}
        extra[q['id']] = {f'evidence@{k}': retrieval.evidence_hit(idx[0], chunks, q_spans, texts, k) for k in (5, 10)}
        top_ids[q['id']] = idx[0]
    table = metrics.evaluate(rankings, questions, KS, extra).assign(variant=name)
    return table, rankings, top_ids

def gold_spans(questions):
    """{id: {document_id: [(inicio, fin)]}} de los relevantes con offsets."""
    out = {}
    for q in questions:
        out[q['id']] = {}
        for r in q['relevantes']:
            if r['char_start'] is not None and r['grado'] >= 1:
                out[q['id']].setdefault(r['document_id'], []).append((r['char_start'], r['char_end']))
    return out

def neighbors(rankings, scores_by_doc, questions, doc_info, top=5):
    """Vecinos de ejemplo: los primeros documentos por pregunta, solo con metadatos (sin texto)."""
    rows = []
    for q in questions:
        grades = metrics.relevance(q)
        for rank, doc in enumerate(rankings[q['id']][:top], start=1):
            info = doc_info.get(doc, {})
            rows.append({'id': q['id'], 'idioma_pregunta': q['idioma_pregunta'], 'rank': rank, 'document_id': doc,
                         'collection_name': info.get('collection_name'), 'language_detected': info.get('language_detected'),
                         'created_at': str(info.get('created_at'))[:10], 'score': scores_by_doc[q['id']].get(doc),
                         'grado': grades.get(doc, 0)})
    return pd.DataFrame(rows)

def doc_scores(query_vectors, matrix, chunks, questions, rankings):
    """Mejor similitud de cada documento rankeado (máximo sobre sus chunks)."""
    provenance = chunks['provenance_ids'].tolist()
    out = {}
    for q, vector in zip(questions, query_vectors):
        wanted = set(rankings[q['id']])
        sims = matrix @ vector
        best = {}
        for i in np.argsort(-sims)[:2000]:
            for doc in provenance[i]:
                if doc in wanted and doc not in best: best[doc] = float(sims[i])
        out[q['id']] = best
    return out

def stability(base_vectors, other_vectors, base_rankings, other_rankings, questions, label, k=10):
    rows = []
    for q, a, b in zip(questions, base_vectors, other_vectors):
        top_a, top_b = set(base_rankings[q['id']][:k]), set(other_rankings[q['id']][:k])
        rows.append({'id': q['id'], 'variant': label, 'cosine_query': float(a @ b),
                     f'jaccard_top{k}': len(top_a & top_b) / len(top_a | top_b) if top_a | top_b else np.nan})
    return pd.DataFrame(rows)

def run(chunks_path, vectors_path, gold_path, paraphrases_path, parquet, profile, embed_query, out_dir, cfg,
        doc_languages=None, log=print):
    """Ejecuta todo el análisis y escribe tablas, figuras y metadatos en `out_dir`.

    `embed_query(textos, prefijo)` devuelve vectores de consulta normalizados con el encoder del perfil.
    """
    from datetime import datetime, timezone
    from pipeline import embedding, io, report
    out_dir = Path(out_dir)
    chunks = pd.read_parquet(chunks_path, columns=CHUNK_COLUMNS)
    ids, matrix, _ = embedding.read_vectors(vectors_path)
    if ids != chunks['chunk_id'].tolist(): raise ValueError('Los vectores no están alineados con el Parquet de chunks.')
    questions, paraphrases = load_gold(gold_path, paraphrases_path)
    docs = sorted({d for p in chunks['provenance_ids'] for d in p})
    meta = io.read_rows(parquet, ['document_id', 'created_at', 'collection_name'], docs).set_index('document_id')
    doc_dates = meta['created_at']
    spans = gold_spans(questions)
    relevant_docs = sorted({d for s in spans.values() for d in s})
    texts = io.read_rows(parquet, ['document_id', 'text'], relevant_docs).set_index('document_id')['text'].to_dict()
    languages = doc_languages or {}
    doc_info = meta.assign(language_detected=[languages.get(d) for d in meta.index]).to_dict(orient='index')

    default_prefix = profile['query_prefix'] or ''
    originals = [q['pregunta'] for q in questions]
    variants = {BASE: (originals, default_prefix, True)}
    variants.update({name: (originals, prefix, True) for name, prefix in (cfg.get('query_variants') or {}).items()})
    variants['sin_filtro_fecha'] = (originals, default_prefix, False)
    if paraphrases:
        variants['parafrasis'] = ([paraphrases[q['id']]['parafrasis']['texto'] for q in questions], default_prefix, True)
        variants['traduccion'] = ([paraphrases[q['id']]['traduccion']['texto'] for q in questions], default_prefix, True)

    log('Variantes de consulta')
    per_question, rankings, vectors = [], {}, {}
    for name, (texts_q, prefix, use_window) in variants.items():
        vectors[name] = np.asarray(embed_query(texts_q, prefix), dtype=np.float32)
        table, rankings[name], _ = run_variant(name, vectors[name], matrix, chunks, questions, doc_dates, spans, texts, use_window)
        per_question.append(table)
    per_question = pd.concat(per_question, ignore_index=True)
    summary = pd.concat([metrics.summarize(g).assign(variant=v) for v, g in per_question.groupby('variant', sort=False)], ignore_index=True)

    log('Dimensión (Matryoshka) y normalización')
    mrl = []
    base = vectors[BASE]
    for dim in cfg['dimensions']:
        for renorm in (True, False):
            if dim == matrix.shape[1] and not renorm: continue
            m, q = matrix[:, :dim], base[:, :dim]
            if renorm: m, q = normalize(m), normalize(q)
            table, _, _ = run_variant(f'{dim}', q, m, chunks, questions, doc_dates, spans, texts)
            mrl.append(metrics.summarize(table).assign(dimension=dim, renormalized=renorm,
                                                       mb_per_100k_chunks=round(dim * 4 * 100_000 / 1e6, 1)))
    mrl = pd.concat(mrl, ignore_index=True)

    log('Idiomas, estabilidad y vecinos')
    original = per_question[per_question['variant'] == BASE].assign(consulta='original')
    by_language = metrics.summarize(original, 'idioma_evidencia').assign(consulta='original')
    if 'traduccion' in rankings:
        translated = per_question[per_question['variant'] == 'traduccion']
        by_language = pd.concat([by_language, metrics.summarize(translated, 'idioma_evidencia').assign(consulta='traduccion')], ignore_index=True)
        stab = pd.concat([stability(base, vectors[v], rankings[BASE], rankings[v], questions, v)
                          for v in ('parafrasis', 'traduccion')], ignore_index=True)
    else:
        stab = pd.DataFrame()
    scores = doc_scores(base, matrix, chunks, questions, rankings[BASE])
    nb = neighbors(rankings[BASE], scores, questions, doc_info, cfg.get('neighbors', 5))

    tables = {'metricas_variantes': summary, 'metricas_por_pregunta': per_question, 'matryoshka': mrl,
              'metricas_por_idioma': by_language, 'estabilidad_consultas': stab, 'vecinos_ejemplo': nb}
    files = report.write_tables(tables, out_dir / 'tables')
    figures = [report.variant_metrics_plot(summary, out_dir / 'figures' / 'variantes_consulta.png'),
               report.matryoshka_plot(mrl, out_dir / 'figures' / 'matryoshka.png'),
               report.language_recall_plot(by_language, out_dir / 'figures' / 'idiomas.png')]
    info = report.to_jsonable({
        'created_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'profile': profile,
        'chunks': {'path': str(chunks_path), 'sha256': io.sha256_file(chunks_path), 'n': len(chunks)},
        'vectors': {'path': str(vectors_path), 'dimension': int(matrix.shape[1]),
                    'norm_mean': float(np.linalg.norm(matrix, axis=1).mean()), 'norm_min': float(np.linalg.norm(matrix, axis=1).min())},
        'gold': {'path': str(gold_path), 'sha256': io.sha256_file(gold_path), 'split': 'dev', 'questions': len(questions)},
        'paraphrases': {'path': str(paraphrases_path), 'sha256': io.sha256_file(paraphrases_path) if paraphrases else None},
        'config': cfg, 'variants': {k: {'prefix': v[1], 'date_filter': v[2]} for k, v in variants.items()},
        'outputs': {'tables': files, 'figures': figures}})
    report.write_json(info, out_dir / 'analisis_encoder.json')
    return info, summary
