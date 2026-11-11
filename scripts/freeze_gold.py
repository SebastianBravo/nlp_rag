"""Congela el gold estándar: aplica los grados finales, valida y escribe la versión `congelado`.

Grado final de cada (pregunta, documento): el de `desacuerdos.xlsx` (columna `grado_final`) si hubo desacuerdo;
si no, el del anotador 1 (que coincide con el del anotador 2 en lo que ambos juzgaron). Solo cambian las preguntas
que el anotador 1 revisó. Sin `--write` solo muestra qué cambiaría (ensayo). Después de escribir: commit del gold,
tag `gold-v1` y registrar el hash que imprime este script en el informe.
"""
import argparse, json, sys
from pathlib import Path
from pipeline import io, pooling, report

def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding='utf-8').splitlines() if l.strip()]

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--gold', default='data/gold/gold.jsonl')
    p.add_argument('--ann1', default='data/gold/anotacion_1.jsonl')
    p.add_argument('--ann2', default='data/gold/anotacion_2.jsonl')
    p.add_argument('--disagreements', default='data/annotation/desacuerdos.xlsx')
    p.add_argument('--sample-ids', default='data/samples/sample_10000_ids.txt')
    p.add_argument('--write', action='store_true', help='escribe el gold congelado (sin esto es un ensayo)')
    args = p.parse_args()

    questions = read_jsonl(args.gold)
    by_id = {q['id']: q for q in questions}
    ann1 = [r for r in read_jsonl(args.ann1) if 'grado' in r]
    ann2 = read_jsonl(args.ann2) if Path(args.ann2).is_file() else []
    final = {}
    for r in ann1: final.setdefault(r['id'], {})[r['document_id']] = r['grado']
    resolved, open_cases = 0, []
    if Path(args.disagreements).is_file():
        for r in pooling.read_sheet(args.disagreements, 'desacuerdos'):
            try: grade = pooling.parse_grade(r.get('grado_final'))
            except ValueError as e: sys.exit(f"Desacuerdos, {r['id']} · {r['document_id']}: {e}")
            if grade is None: open_cases.append(f"{r['id']} · {r['document_id']}"); continue
            final.setdefault(r['id'], {})[r['document_id']] = grade
            resolved += 1

    problems, warnings = [], []
    if open_cases: problems.append(f'desacuerdos sin `grado_final` ({len(open_cases)}): {", ".join(open_cases[:10])}')
    sample_ids = set(Path(args.sample_ids).read_text(encoding='utf-8').split()) if Path(args.sample_ids).is_file() else None
    pending = [q['id'] for q in questions if q.get('pooling_pendiente')]
    unreviewed = [i for i in pending if i not in final]
    if unreviewed: problems.append(f'preguntas con pooling pendiente sin revisar en la hoja 1: {", ".join(unreviewed)}')
    for i in final:
        if i not in by_id: problems.append(f'la anotación menciona una pregunta que no existe: {i}')
    if sample_ids is not None:
        outside = sorted({(i, d) for i, g in final.items() for d, v in g.items() if v >= 1 and d not in sample_ids})
        if outside: problems.append(f'documentos relevantes fuera de sample_10000: {outside[:5]}')
    invalid = [r['id'] for r in ann2 if r.get('validez') == 'inválida']
    if invalid: warnings.append(f'respuesta de referencia marcada inválida por el segundo anotador: {", ".join(sorted(invalid))}')
    partial = [r['id'] for r in ann2 if r.get('validez') == 'parcial']
    if partial: warnings.append(f'respuesta de referencia parcial: {", ".join(sorted(partial))}')

    frozen, changes = [], []
    double = {r['id'] for r in ann2 if 'grado' in r}
    for q in questions:
        new = dict(q)
        if q['id'] in final:
            new['relevantes'] = pooling.merge_relevantes(q, final[q['id']])
            relevant_now = any(r['grado'] >= 1 for r in new['relevantes'])
            if q['debe_abstenerse'] and relevant_now:
                problems.append(f"{q['id']} debe abstenerse pero quedó con documentos relevantes: revisa la pregunta o los grados")
            if not q['debe_abstenerse'] and not relevant_now:
                warnings.append(f"{q['id']} no queda con documentos relevantes (¿debería abstenerse?)")
            new['pooling_pendiente'] = False
            if new['relevantes'] != q['relevantes']: changes.append({'id': q['id'], 'before': len(q['relevantes']), 'after': len(new['relevantes'])})
            if relevant_now and (q.get('idioma_evidencia') is None or not q.get('colecciones')):
                warnings.append(f"{q['id']}: completa `idioma_evidencia` y `colecciones` a mano (ahora tiene relevantes)")
        if q['id'] in double: new['doble_anotacion'] = True
        new['estado'] = 'congelado'
        frozen.append(new)
    still = [q['id'] for q in frozen if q.get('pooling_pendiente')]
    if still: problems.append(f'siguen con `pooling_pendiente`: {", ".join(still)}')

    result = {'questions': len(frozen), 'questions_changed': len(changes), 'disagreements_resolved': resolved,
              'double_annotated': len(double), 'changes': changes, 'warnings': warnings, 'problems': problems,
              'gold_sha256_before': io.sha256_file(args.gold), 'written': False}
    if problems:
        print(json.dumps(result, ensure_ascii=False, indent=1)); sys.exit('No se puede congelar: resuelve los problemas anteriores.')
    if args.write:
        Path(args.gold).write_text('\n'.join(json.dumps(q, ensure_ascii=False) for q in frozen) + '\n', encoding='utf-8')
        result |= {'written': True, 'gold_sha256_after': io.sha256_file(args.gold)}
        report.write_json(result, Path('reports/gold/congelamiento.json'))
    print(json.dumps(result, ensure_ascii=False, indent=1))

if __name__ == '__main__': main()
