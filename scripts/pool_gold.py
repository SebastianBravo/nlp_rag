"""Pooling del gold estándar (Milvus A y B) y hojas de anotación.

Consulta las colecciones de ambas estrategias de chunking con la pregunta original (y su ventana de fechas si la
tiene), une los `--depth` primeros documentos de cada una y escribe:
- data/gold/pooling.jsonl: por pregunta, solo ids y rangos (se versiona);
- reports/gold/tables/pooling_cobertura.csv: cuántos relevantes ya anotados aparecen en el pool;
- data/annotation/hoja_anotador1.xlsx: hoja para quien hizo el borrador (preguntas con pooling pendiente, las de la
  submuestra de doble anotación y, con --todas, el resto);
- data/annotation/hoja_anotador2.xlsx: hoja ciega para el segundo anotador (submuestra, top-`--depth2`, sin rangos
  ni juicios previos ni respuestas de referencia). Las hojas llevan texto del corpus: son locales, no se versionan.
Con --fase2 solo genera la hoja de validación de las respuestas de referencia (se entrega cuando el segundo
anotador terminó la primera hoja).
"""
import argparse, json, os, sys
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import yaml
from pymilvus import MilvusClient
from app.embeddings import embed_array
from pipeline import indexing, io, pooling, report, retrieval

def log(message):
    print(f'· {message}', file=sys.stderr, flush=True)

def load_gold(path):
    return [json.loads(l) for l in Path(path).read_text(encoding='utf-8').splitlines() if l.strip()]

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/milvus.yaml')
    p.add_argument('--gold', default='data/gold/gold.jsonl')
    p.add_argument('--input', default='data/documentos_rag.parquet')
    p.add_argument('--depth', type=int, default=20, help='documentos por colección que entran al pool')
    p.add_argument('--depth2', type=int, default=10, help='profundidad de la hoja del segundo anotador')
    p.add_argument('--sample-size', type=int, default=16, help='preguntas de la doble anotación')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--todas', action='store_true', help='la hoja 1 incluye las 52 preguntas')
    p.add_argument('--fase2', action='store_true', help='solo la hoja de validación de respuestas de referencia')
    p.add_argument('--out-dir', default='data/annotation')
    p.add_argument('--collections', nargs=2, default=['sent256_s10k', 'win256_s10k'], help='claves de configs/milvus.yaml (B y A)')
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    questions = load_gold(args.gold)
    by_id = {q['id']: q for q in questions}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample = pooling.second_annotator_sample(questions, args.sample_size, args.seed)

    if args.fase2:
        rows = [{'id': i, 'tipo': by_id[i]['tipo'], 'pregunta': by_id[i]['pregunta'], 'respuesta_referencia': by_id[i]['respuesta_referencia'],
                 'criterio': by_id[i].get('criterio') or '', 'validez': None, 'comentario': ''} for i in sample]
        path = out_dir / 'hoja_anotador2_fase2.xlsx'
        pooling.write_sheets(path, {'validez': (pooling.PHASE2_COLUMNS, rows)}, {'validez': {'validez': pooling.VALIDITY}},
                             {'pregunta': 60, 'respuesta_referencia': 60, 'criterio': 40, 'validez': 12, 'comentario': 40})
        print(json.dumps({'fase2': str(path), 'questions': sample}, ensure_ascii=False)); return

    profile = yaml.safe_load(Path(cfg['encoders_config']).read_text(encoding='utf-8'))['profiles'][cfg['encoder']]
    client = MilvusClient(uri=os.getenv('MILVUS_URI', 'http://milvus:19530'))
    ef, fields = cfg['search']['ef'], ['document_id', 'char_start', 'text', 'provenance_ids']
    names = {'B' if cfg['collections'][k]['strategy'] == 'B' else 'A': cfg['collections'][k]['name'] for k in args.collections}
    for name in names.values():
        if not client.has_collection(name): sys.exit(f'La colección {name} no existe en Milvus: cárgala con scripts.ingest')

    log(f'Vectorizando {len(questions)} preguntas')
    vectors = embed_array([q['pregunta'] for q in questions], 'query', profile)
    per_collection = {key: {} for key in names}
    for key, name in names.items():
        log(f'Consultando {name}')
        for q, vector in zip(questions, vectors):
            window = q.get('ventana_fecha')
            expr = indexing.window_filter(window) if window else ''
            hits = indexing.search(client, name, [vector], 300, max(ef, 300), expr, fields)[0]
            allowed = None
            if window:   # un chunk repetido en otro documento cuenta solo si ese documento cae en la ventana
                docs = sorted({d for _, _, e in hits for d in e['provenance_ids']})
                dates = io.read_rows(args.input, ['document_id', 'created_at'], docs).set_index('document_id')['created_at']
                allowed = retrieval.documents_in_window(dates, window)
            per_collection[key][q['id']] = pooling.rank_with_passages(hits, allowed, args.depth)
    pool = pooling.build_pool(per_collection)

    gold_sha = io.sha256_file(args.gold)
    records = pooling.pooling_records(pool, {k: v for k, v in names.items()}, args.depth, gold_sha, questions)
    pooling_path = Path('data/gold/pooling.jsonl')
    pooling_path.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in records) + '\n', encoding='utf-8')
    coverage = pd.DataFrame([{'id': r['id'], 'tipo': by_id[r['id']]['tipo'], 'split': by_id[r['id']]['split'], 'candidates': r['n_candidates'],
                              'known_relevant': r['known_relevant'], 'known_relevant_in_pool': r['known_relevant_in_pool']} for r in records])
    report.write_tables({'pooling_cobertura': coverage}, Path('reports/gold/tables'))

    pending = [q['id'] for q in questions if q.get('pooling_pendiente')]
    sheet1_ids = sorted(set(pending) | set(sample) | (set(by_id) if args.todas else set()))
    sheet_ids = sorted(set(sheet1_ids) | set(sample))
    log(f'Leyendo el texto de {len(sheet_ids)} preguntas')
    wanted = {c['document_id'] for i in sheet_ids for c in pool[i]}
    wanted |= {c['passage']['canonical'] for i in sheet_ids for c in pool[i]}
    corpus = io.read_rows(args.input, ['document_id', 'text', 'collection_name', 'title', 'created_at'], sorted(wanted)).set_index('document_id')
    docs = corpus.to_dict(orient='index')

    def passages(qid):
        out = {}
        for c in pool[qid]:
            ps = c['passage']
            out[c['document_id']] = pooling.passage_with_context(docs.get(c['document_id'], {}).get('text', ''), ps['text'],
                                                                 ps['char_start'] if ps['canonical'] == c['document_id'] else None)
        return out

    widths = {'pregunta': 45, 'ventana_fecha': 14, 'document_id': 30, 'titulo': 30, 'pasaje': 90, 'comentario': 30, 'n': 5, 'grado': 8}
    rows1, rows2, answers = [], [], []
    for qid in sheet1_ids:
        known = {r['document_id']: r['grado'] for r in by_id[qid]['relevantes']}
        extra = [{'document_id': d, 'ranks': {}, 'best_rank': 0, 'passage': None} for d in known if d not in {c['document_id'] for c in pool[qid]}]
        cands = pool[qid]
        r = pooling.annotation_rows(qid, by_id[qid], cands, docs, passages(qid), known, args.seed)
        for e in extra:   # relevantes ya anotados que el pool no recuperó: también se revisan
            text = docs.get(e['document_id'], {}).get('text', '')
            r += pooling.annotation_rows(qid, by_id[qid], [e], docs, {e['document_id']: text[:1500]}, known, args.seed)
        for n, row in enumerate(r, 1): row['n'] = n
        rows1 += r
    for qid in sample:
        rows2 += pooling.annotation_rows(qid, by_id[qid], pool[qid], docs, passages(qid), None, args.seed, depth=args.depth2)
        window = by_id[qid].get('ventana_fecha')
        answers.append({'id': qid, 'tipo': by_id[qid]['tipo'], 'pregunta': by_id[qid]['pregunta'],
                        'ventana_fecha': f"{window['desde']} a {window['hasta']}" if window else '', 'respuesta_propia': None})
    grade_choice = {'anotacion': {'grado': pooling.GRADES}}
    p1, p2 = out_dir / 'hoja_anotador1.xlsx', out_dir / 'hoja_anotador2.xlsx'
    pooling.write_sheets(p1, {'anotacion': (pooling.SHEET_COLUMNS, rows1)}, grade_choice, widths)
    pooling.write_sheets(p2, {'anotacion': (pooling.SHEET_COLUMNS, rows2), 'respuestas': (pooling.ANSWER_COLUMNS, answers)}, grade_choice,
                         widths | {'respuesta_propia': 60})
    info = {'created_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'gold_sha256': gold_sha, 'depth': args.depth,
            'depth2': args.depth2, 'seed': args.seed, 'collections': names, 'sample': sample, 'pending': pending}
    report.write_json(info, Path('reports/gold/runs/pooling.json'))
    per_q = coverage.set_index('id')['candidates']
    print(json.dumps({'gold_sha256': gold_sha, 'pooling': str(pooling_path), 'hoja1': str(p1), 'hoja1_preguntas': len(sheet1_ids),
                      'hoja1_filas': len(rows1), 'hoja2': str(p2), 'hoja2_preguntas': len(sample), 'hoja2_filas': len(rows2),
                      'candidatos_pendientes': {i: int(per_q[i]) for i in pending}, 'muestra_doble_anotacion': sample,
                      'relevantes_conocidos': int(coverage['known_relevant'].sum()),
                      'relevantes_conocidos_en_pool': int(coverage['known_relevant_in_pool'].sum())}, ensure_ascii=False))

if __name__ == '__main__': main()
