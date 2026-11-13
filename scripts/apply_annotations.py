"""Lee las hojas de anotación, calcula el acuerdo entre anotadores y prepara la resolución de desacuerdos.

Entradas: data/annotation/hoja_anotador1.xlsx y hoja_anotador2.xlsx (con la columna `grado` llena) y, opcional,
hoja_anotador2_fase2.xlsx (validez de las respuestas de referencia).
Salidas versionables, sin texto del corpus:
- data/gold/anotacion_1.jsonl y anotacion_2.jsonl: (pregunta, documento, grado) y la validez de las respuestas;
- reports/gold/acuerdo.json y tables/acuerdo_por_pregunta.csv, desacuerdos.csv.
Salida local: data/annotation/desacuerdos.xlsx, con una columna `grado_final` que deben llenar los dos anotadores
tras discutir cada caso.
"""
import argparse, json, sys
from pathlib import Path
import pandas as pd
from pipeline import pooling, report

def load_gold(path):
    return [json.loads(l) for l in Path(path).read_text(encoding='utf-8').splitlines() if l.strip()]

def write_jsonl(path, rows):
    Path(path).write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows) + ('\n' if rows else ''), encoding='utf-8')

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--gold', default='data/gold/gold.jsonl')
    p.add_argument('--dir', default='data/annotation')
    p.add_argument('--sheet1', default='hoja_anotador1.xlsx')
    p.add_argument('--sheet2', default='hoja_anotador2.xlsx')
    p.add_argument('--fase2', default='hoja_anotador2_fase2.xlsx')
    p.add_argument('--allow-partial', action='store_true', help='no falla si hay preguntas a medias (se reportan)')
    args = p.parse_args()
    directory = Path(args.dir)
    questions = load_gold(args.gold)
    types = {q['id']: q['tipo'] for q in questions}

    def read(name):
        rows = pooling.read_sheet(directory / name, 'anotacion')
        grades, blank, partial, errors = pooling.grades_from_rows(rows)
        return rows, grades, blank, partial, errors
    rows1, grades1, blank1, partial1, errors1 = read(args.sheet1)
    rows2, grades2, blank2, partial2, errors2 = read(args.sheet2)
    problems = [f'hoja 1: {e}' for e in errors1] + [f'hoja 2: {e}' for e in errors2]
    for label, partial in (('hoja 1', partial1), ('hoja 2', partial2)):
        if partial and not args.allow_partial: problems.append(f'{label}: preguntas con documentos sin grado: {", ".join(partial)}')
    if problems: sys.exit('Revisa las hojas:\n- ' + '\n- '.join(problems))

    own_answers = {r['id']: r.get('respuesta_propia') for r in pooling.read_sheet(directory / args.sheet2, 'respuestas')}
    no_consta = {i: str(a or '').strip().lower().rstrip('.') in ('no consta', 'no consta en los documentos') for i, a in own_answers.items()}
    validity = []
    fase2 = directory / args.fase2
    if fase2.is_file():
        for r in pooling.read_sheet(fase2, 'validez'):
            v = (r.get('validez') or '').strip().lower() if r.get('validez') else None
            if v and v not in pooling.VALIDITY: sys.exit(f'Validez no reconocida en {r["id"]}: {r["validez"]!r}; opciones: {pooling.VALIDITY}')
            if v: validity.append({'id': r['id'], 'tipo': 'validez', 'validez': v})

    write_jsonl('data/gold/anotacion_1.jsonl', [{'id': q, 'document_id': d, 'grado': g} for (q, d), g in sorted(grades1.items())])
    ann2 = [{'id': q, 'document_id': d, 'grado': g} for (q, d), g in sorted(grades2.items())]
    ann2 += [{'id': i, 'tipo': 'respuesta_propia', 'no_consta': v} for i, v in sorted(no_consta.items()) if i in {q for q, _ in grades2}]
    write_jsonl('data/gold/anotacion_2.jsonl', ann2 + validity)

    reviewed2 = {q for q, _ in grades2}
    common = {k: v for k, v in grades1.items() if k[0] in reviewed2}
    summary, table, disagreements = pooling.agreement(common, grades2, types)
    summary |= {'questions_sheet1': len({q for q, _ in grades1}), 'questions_sheet2': len(reviewed2), 'blank_sheet1': blank1,
                'blank_sheet2': blank2, 'partial_sheet1': partial1, 'partial_sheet2': partial2, 'disagreements': len(disagreements),
                'validity': {v: sum(x['validez'] == v for x in validity) for v in pooling.VALIDITY}}
    abstentions = {q['id'] for q in questions if q['debe_abstenerse']}
    summary['abstention_questions_in_sample'] = sorted(abstentions & reviewed2)
    report.write_tables({'acuerdo_por_pregunta': pd.DataFrame(table), 'desacuerdos': pd.DataFrame(disagreements)}, Path('reports/gold/tables'))
    report.write_json(summary, Path('reports/gold/acuerdo.json'))

    passage = {(r['id'], r['document_id']): r for r in rows1}
    sheet = []
    for d in disagreements:
        r = passage.get((d['id'], d['document_id']), {})
        sheet.append({'id': d['id'], 'tipo': types[d['id']], 'pregunta': r.get('pregunta', ''), 'ventana_fecha': r.get('ventana_fecha', ''),
                      'document_id': d['document_id'], 'titulo': r.get('titulo', ''), 'pasaje': r.get('pasaje', ''), 'grado_1': d['grado_1'],
                      'grado_2': d['grado_2'], 'grado_final': None, 'comentario': ''})
    columns = ['id', 'tipo', 'pregunta', 'ventana_fecha', 'document_id', 'titulo', 'pasaje', 'grado_1', 'grado_2', 'grado_final', 'comentario']
    pooling.write_sheets(directory / 'desacuerdos.xlsx', {'desacuerdos': (columns, sheet)}, {'desacuerdos': {'grado_final': pooling.GRADES}},
                         {'pregunta': 40, 'document_id': 30, 'titulo': 30, 'pasaje': 90, 'comentario': 30})
    print(json.dumps(summary, ensure_ascii=False, default=str))

if __name__ == '__main__': main()
