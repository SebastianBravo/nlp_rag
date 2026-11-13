"""Pooling del gold estándar y segunda anotación.

Reúne, para cada pregunta, los documentos que recuperan las colecciones de Milvus de ambas estrategias de
chunking (A y B), prepara hojas de anotación (con orden aleatorio y sin rangos, para que no sesguen al anotador)
y calcula el acuerdo entre dos anotadores (kappa ponderado).

Solo las funciones de hojas tocan el texto del corpus, y solo para las hojas locales (`data/annotation/`, no
versionado). Lo versionable (`data/gold/pooling.jsonl`, `anotacion_*.jsonl`) lleva únicamente ids, rangos y grados.
"""
import hashlib, math
from collections import defaultdict
import numpy as np

GRADES = (0, 1, 2)
SHEET_COLUMNS = ['id', 'tipo', 'pregunta', 'ventana_fecha', 'n', 'document_id', 'fuente', 'fecha', 'titulo', 'pasaje', 'grado', 'comentario']
ANSWER_COLUMNS = ['id', 'tipo', 'pregunta', 'ventana_fecha', 'respuesta_propia']
PHASE2_COLUMNS = ['id', 'tipo', 'pregunta', 'respuesta_referencia', 'criterio', 'validez', 'comentario']
VALIDITY = ('válida', 'parcial', 'inválida')
CELL_LIMIT = 32000   # Excel admite 32.767 caracteres por celda

# ---------------------------------------------------------------- recuperación y unión

def rank_with_passages(hits, allowed=None, limit=20):
    """Documentos en orden de aparición (mejor rank) a partir de los chunks de Milvus, con el chunk que los aportó.

    `hits`: [(chunk_id, puntaje, entidad)] ordenados. Un chunk cuenta para su documento canónico y para los de
    `provenance_ids`; `allowed` (conjunto de ids) filtra por ventana de fechas.
    """
    out, seen = [], set()
    for chunk_id, _, e in hits:
        for doc in list(e['provenance_ids']):
            if doc in seen or (allowed is not None and doc not in allowed): continue
            seen.add(doc)
            out.append({'document_id': doc, 'rank': len(out) + 1, 'chunk_id': chunk_id, 'canonical': e['document_id'],
                        'char_start': e['char_start'], 'text': e['text']})
            if len(out) == limit: return out
    return out

def build_pool(per_collection):
    """Une los rankings de varias colecciones a nivel documento.

    `per_collection`: {clave: {id_pregunta: [fila de rank_with_passages]}}. Devuelve {id_pregunta: [candidato]} con
    `ranks` ({clave: rank}), `best_rank` y el pasaje del chunk mejor posicionado (empates: orden de las claves).
    """
    keys = list(per_collection)
    questions = sorted({q for rows in per_collection.values() for q in rows})
    pool = {}
    for qid in questions:
        merged = {}
        for key in keys:
            for row in per_collection[key].get(qid, []):
                c = merged.setdefault(row['document_id'], {'document_id': row['document_id'], 'ranks': {}, 'passage_rank': None})
                c['ranks'][key] = row['rank']
                if c['passage_rank'] is None or row['rank'] < c['passage_rank']:
                    c['passage_rank'], c['passage'] = row['rank'], {k: row[k] for k in ('chunk_id', 'canonical', 'char_start', 'text')}
        for c in merged.values(): c['best_rank'] = min(c['ranks'].values())
        pool[qid] = sorted(merged.values(), key=lambda c: (c['best_rank'], c['document_id']))
    return pool

def pooling_records(pool, collections, depth, gold_sha256, questions):
    """Una línea por pregunta para `data/gold/pooling.jsonl`: solo ids y rangos, sin texto del corpus.

    Incluye cuántos de los relevantes ya anotados aparecen en el pool (cobertura).
    """
    by_id = {q['id']: q for q in questions}
    rows = []
    for qid, cands in pool.items():
        known = {r['document_id'] for r in by_id[qid]['relevantes'] if r['grado'] >= 1}
        found = known & {c['document_id'] for c in cands}
        rows.append({'id': qid, 'depth': depth, 'collections': collections, 'gold_sha256': gold_sha256,
                     'n_candidates': len(cands), 'known_relevant': len(known), 'known_relevant_in_pool': len(found),
                     'candidates': [{'document_id': c['document_id'], 'ranks': c['ranks']} for c in cands]})
    return rows

# ---------------------------------------------------------------- muestras y orden

def _hash(*parts):
    return hashlib.sha256('|'.join(map(str, parts)).encode('utf-8')).hexdigest()

def shuffled(candidates, seed, qid):
    """Orden pseudoaleatorio y estable de los candidatos de una pregunta (no depende de las demás)."""
    return sorted(candidates, key=lambda c: _hash(seed, qid, c['document_id']))

def second_annotator_sample(questions, n=16, seed=42):
    """Submuestra estratificada por tipo, entre las preguntas de `test` con segundo anotador asignado."""
    eligible = [q for q in questions if q['split'] == 'test' and q.get('anotador_2')]
    if n >= len(eligible): return sorted(q['id'] for q in eligible)
    groups = defaultdict(list)
    for q in eligible: groups[q['tipo']].append(q['id'])
    quotas = {t: n * len(g) / len(eligible) for t, g in groups.items()}
    take = {t: int(v) for t, v in quotas.items()}
    for t in sorted(quotas, key=lambda t: (-(quotas[t] - take[t]), t))[:n - sum(take.values())]: take[t] += 1
    chosen = []
    for t, g in groups.items():
        chosen += sorted(g, key=lambda i: _hash(seed, i))[:take[t]]
    return sorted(chosen)

# ---------------------------------------------------------------- hojas

def passage_with_context(doc_text, chunk_text, start=None, margin=600):
    """Texto del documento alrededor del chunk, con el chunk entre `>>>` y `<<<`.

    `start` es el offset del chunk en su documento canónico; si no coincide con este texto (documento de
    procedencia distinto del canónico), se busca el chunk en `doc_text`. Sin coincidencia, solo el chunk.
    """
    if start is None or doc_text[start:start + len(chunk_text)] != chunk_text: start = doc_text.find(chunk_text)
    if start < 0: return f'>>> {chunk_text} <<<'[:CELL_LIMIT]
    end = start + len(chunk_text)
    before = doc_text[max(0, start - margin):start]
    after = doc_text[end:end + margin]
    return (('…' if start - margin > 0 else '') + before + f'\n>>> {chunk_text} <<<\n' + after +
            ('…' if end + margin < len(doc_text) else ''))[:CELL_LIMIT]

def _date(value):
    return '' if value is None or value != value else str(value)[:10]

def annotation_rows(qid, question, candidates, docs, passages, prefill=None, seed=42, depth=None):
    """Filas de la hoja de anotación de una pregunta, en orden aleatorio.

    `docs`: {document_id: {'text', 'collection_name', 'title', 'created_at'}} del corpus; `passages`: {document_id:
    pasaje}; `prefill`: {document_id: grado} para precargar un juicio previo; `depth`: solo candidatos con
    best_rank <= depth.
    """
    chosen = [c for c in candidates if depth is None or c['best_rank'] <= depth]
    rows = []
    for n, c in enumerate(shuffled(chosen, seed, qid), 1):
        d = docs.get(c['document_id'], {})
        window = question.get('ventana_fecha')
        rows.append({'id': qid, 'tipo': question['tipo'], 'pregunta': question['pregunta'],
                     'ventana_fecha': f"{window['desde']} a {window['hasta']}" if window else '', 'n': n,
                     'document_id': c['document_id'], 'fuente': d.get('collection_name', ''), 'fecha': _date(d.get('created_at')),
                     'titulo': str(d.get('title') or '')[:300], 'pasaje': passages.get(c['document_id'], ''),
                     'grado': (prefill or {}).get(c['document_id']), 'comentario': ''})
    return rows

def write_sheets(path, sheets, validations=None, widths=None):
    """Escribe un libro .xlsx. `sheets`: {nombre: (columnas, filas)}; `validations`: {nombre: {columna: [opciones]}}."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font
    from openpyxl.worksheet.datavalidation import DataValidation
    wb = Workbook()
    wb.remove(wb.active)
    for name, (columns, rows) in sheets.items():
        ws = wb.create_sheet(name)
        ws.append(columns)
        for cell in ws[1]: cell.font = Font(bold=True)
        for r, row in enumerate(rows, 2):
            for c, col in enumerate(columns, 1):
                value = row.get(col)
                cell = ws.cell(row=r, column=c, value=value)
                if isinstance(value, str):
                    cell.data_type = 's'   # un texto que empiece por "=" no debe interpretarse como fórmula
                    cell.alignment = Alignment(wrap_text=True, vertical='top')
        ws.freeze_panes = 'A2'
        for c, col in enumerate(columns, 1):
            ws.column_dimensions[ws.cell(row=1, column=c).column_letter].width = (widths or {}).get(col, 14)
        for col, options in (validations or {}).get(name, {}).items():
            dv = DataValidation(type='list', formula1='"' + ','.join(map(str, options)) + '"', allow_blank=True)
            ws.add_data_validation(dv)
            letter = ws.cell(row=1, column=columns.index(col) + 1).column_letter
            dv.add(f'{letter}2:{letter}{max(len(rows) + 1, 2)}')
    wb.save(path)

def read_sheet(path, sheet):
    """Filas de una hoja como diccionarios (las celdas vacías son None)."""
    from openpyxl import load_workbook
    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb[sheet]
    header, rows = None, []
    for values in ws.iter_rows(values_only=True):
        if header is None: header = list(values); continue
        if all(v is None for v in values): continue
        rows.append(dict(zip(header, values)))
    return rows

def parse_grade(value):
    """Grado 0, 1 o 2 desde una celda; None si está vacía; ValueError si no es válido."""
    if value is None or (isinstance(value, str) and not value.strip()): return None
    try: grade = int(float(str(value).strip().replace(',', '.')))
    except ValueError: raise ValueError(f'grado no válido: {value!r}')
    if grade not in GRADES or float(str(value).strip().replace(',', '.')) != grade: raise ValueError(f'grado no válido: {value!r}')
    return grade

def grades_from_rows(rows):
    """({(id, document_id): grado}, preguntas sin revisar, preguntas a medias, errores) desde las filas de una hoja."""
    by_q = defaultdict(list)
    errors = []
    for r in rows:
        try: grade = parse_grade(r.get('grado'))
        except ValueError as e: errors.append(f"{r['id']} · {r['document_id']}: {e}"); continue
        by_q[r['id']].append((r['document_id'], grade))
    grades, blank, partial = {}, [], []
    for qid, items in by_q.items():
        filled = [(d, g) for d, g in items if g is not None]
        if not filled: blank.append(qid); continue
        if len(filled) < len(items): partial.append(qid)
        grades.update({(qid, d): g for d, g in filled})
    return grades, sorted(blank), sorted(partial), errors

# ---------------------------------------------------------------- acuerdo

def cohen_kappa(a, b, labels=GRADES, weights='quadratic'):
    """Kappa de Cohen (ponderado cuadrático o lineal; 'none' = nominal). NaN si no está definido (todo igual)."""
    n = len(a)
    if n == 0 or n != len(b): return float('nan')
    index = {label: i for i, label in enumerate(labels)}
    k = len(labels)
    observed = np.zeros((k, k))
    for x, y in zip(a, b): observed[index[x], index[y]] += 1
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / n
    i, j = np.indices((k, k))
    w = {'quadratic': ((i - j) / max(k - 1, 1)) ** 2, 'linear': np.abs(i - j) / max(k - 1, 1)}.get(weights, (i != j).astype(float))
    denominator = (w * expected).sum()
    return float('nan') if denominator == 0 else float(1 - (w * observed).sum() / denominator)

def agreement(grades_1, grades_2, types=None):
    """Acuerdo entre dos anotadores sobre los pares (pregunta, documento) que ambos juzgaron.

    Devuelve (resumen, tabla por pregunta, desacuerdos). El kappa binario usa grado >= 1 (relevante o no).
    """
    keys = sorted(set(grades_1) & set(grades_2))
    a = [grades_1[k] for k in keys]
    b = [grades_2[k] for k in keys]
    summary = {'pairs': len(keys), 'questions': len({k[0] for k in keys}),
               'exact_agreement': float(np.mean([x == y for x, y in zip(a, b)])) if keys else float('nan'),
               'kappa_weighted_quadratic': cohen_kappa(a, b), 'kappa_weighted_linear': cohen_kappa(a, b, weights='linear'),
               'kappa_binary': cohen_kappa([int(x >= 1) for x in a], [int(y >= 1) for y in b], labels=(0, 1), weights='none')}
    by_q = defaultdict(list)
    for k, x, y in zip(keys, a, b): by_q[k[0]].append((x, y))
    table = [{'id': q, 'tipo': (types or {}).get(q), 'pairs': len(v), 'exact_agreement': float(np.mean([x == y for x, y in v])),
              'relevant_1': sum(x >= 1 for x, _ in v), 'relevant_2': sum(y >= 1 for _, y in v)} for q, v in sorted(by_q.items())]
    disagreements = [{'id': k[0], 'document_id': k[1], 'grado_1': x, 'grado_2': y} for k, x, y in zip(keys, a, b) if x != y]
    return summary, table, disagreements

# ---------------------------------------------------------------- congelamiento

def merge_relevantes(question, final_grades):
    """`relevantes` de una pregunta con los grados finales {document_id: grado}.

    Los documentos con grado >= 1 se conservan (con sus offsets si ya los tenían, ajustando el grado); los de grado 0
    se quitan; los nuevos entran como documento completo (offsets nulos). Los demás documentos no se tocan.
    """
    out, seen = [], set()
    for r in question['relevantes']:
        d = r['document_id']
        if d not in final_grades: out.append(dict(r)); seen.add(d); continue
        if final_grades[d] >= 1: out.append({**r, 'grado': final_grades[d]}); seen.add(d)
    for d, g in sorted(final_grades.items()):
        if g >= 1 and d not in seen: out.append({'document_id': d, 'char_start': None, 'char_end': None, 'grado': g})
    return out
