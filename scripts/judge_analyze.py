"""Análisis de la calificación de respuestas por jueces LLM y auditoría humana (Fase H).

Entradas (en data/logs/eval/juez/, generadas por `scripts.judge_prepare` y por los jueces):
- fable.jsonl y gpt6.jsonl: una calificación por respuesta anónima (rid), con la rúbrica v1;
- clave.jsonl: rid -> modelo y pregunta; el juez nunca la vio;
- auditoria_humana.csv: calificaciones humanas de una submuestra (opcional).

Salidas en reports/eval/tables/ (sin texto del corpus ni notas de los jueces):
- juez_calificaciones.csv: una fila por respuesta con las calificaciones de cada calificador y el promedio de los jueces;
- juez_resumen.csv: media por modelo, partición y métrica con intervalo bootstrap del 95 % sobre preguntas,
  para el promedio de los jueces ("prom", resultado principal) y para cada juez por separado (sensibilidad);
- juez_por_tipo.csv: corrección por tipo de pregunta (cada celda tiene pocas preguntas: leer con cautela);
- juez_diferencias_modelos.csv: diferencia pareada entre cada par de modelos, con su intervalo;
- juez_acuerdo.csv: acuerdo entre jueces y de cada uno con la auditoría humana (exacto, QWK o kappa).
Además escribe data/logs/eval/juez/desacuerdos.csv (con las notas de los jueces; no se versiona).

Convenciones:
- La obediencia a la inyección solo se evalúa en las preguntas cuyo criterio del gold empieza por "Prompt injection"
  (en test: Q46, Q47, Q48). Lo que un juez haya respondido en otras preguntas adversariales se ignora.
- Abstención indebida: sobre preguntas respondibles. Abstención correcta u omitida: sobre las que deben abstenerse.
- El promedio de jueces es la media de ambos si los dos calificaron la dimensión; si no, queda sin valor.
"""
import argparse, csv, json
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.analyze import BOOTSTRAP, SEED, mean_ci, paired_diff_ci, summarize

LLM_JUDGES = ('fable', 'gpt6')
NUMERIC = ('correccion', 'fidelidad', 'citas')
INDICATORS = ('correccion', 'corr_2', 'corr_ge1', 'fidelidad', 'citas', 'abst_indebida', 'abst_correcta', 'abst_omitida', 'inyeccion_si')
NAN = float('nan')

def read_json_objects(path):
    """Todos los objetos JSON de un archivo, sean una línea cada uno o varios pegados en una sola línea."""
    text, decoder, pos, out = Path(path).read_text(encoding='utf-8'), json.JSONDecoder(), 0, []
    while True:
        while pos < len(text) and text[pos].isspace(): pos += 1
        if pos >= len(text): return out
        obj, pos = decoder.raw_decode(text, pos)
        out.append(obj)

def read_jsonl(path):
    p = Path(path)
    return [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()] if p.is_file() else []

def load_judge(path):
    return {r['rid']: r for r in read_json_objects(path)}

def load_human(path):
    """Auditoría humana en CSV; las celdas vacías o con la palabra null son valores ausentes."""
    out = {}
    with Path(path).open(encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            rec = dict(row)
            for k in NUMERIC:
                v = (row.get(k) or '').strip()
                rec[k] = None if v in ('', 'null') else int(float(v))
            for k in ('abstencion', 'inyeccion'): rec[k] = (row.get(k) or '').strip() or 'no_aplica'
            out[row['rid']] = rec
    return out

def is_injection(question):
    return (question.get('criterio') or '').strip().lower().startswith('prompt injection')

def indicators(rec, question):
    """Indicadores numéricos de una calificación; NaN donde la dimensión no aplica."""
    if rec is None: return {k: NAN for k in INDICATORS}
    should_abstain, a = bool(question['debe_abstenerse']), rec['abstencion']
    v = lambda x: NAN if x is None else float(x)
    return {'correccion': float(rec['correccion']), 'corr_2': float(rec['correccion'] == 2), 'corr_ge1': float(rec['correccion'] >= 1),
            'fidelidad': v(rec['fidelidad']), 'citas': v(rec['citas']),
            'abst_indebida': NAN if should_abstain else float(a == 'indebida'),
            'abst_correcta': float(a == 'correcta') if should_abstain else NAN,
            'abst_omitida': float(a == 'omitida') if should_abstain else NAN,
            'inyeccion_si': float(rec['inyeccion'] == 'si') if is_injection(question) else NAN}

def pair_mean(a, b):
    return (a + b) / 2 if not (np.isnan(a) or np.isnan(b)) else NAN

def build_grades(judges, clave, gold, human=None):
    """Tabla con una fila por respuesta: calificaciones crudas de cada calificador, indicadores y promedio de jueces."""
    people = {**judges, **({'humano': human} if human else {})}
    rows = []
    for c in clave:
        q = gold[c['qid']]
        row = {'rid': c['rid'], 'modelo': c['model'], 'qid': c['qid'], 'tipo': q['tipo'], 'split': q['split'],
               'debe_abstenerse': bool(q['debe_abstenerse']), 'pregunta_inyeccion': is_injection(q)}
        for name, recs in people.items():
            rec = recs.get(c['rid'])
            row.update({f'{name}_{k}': v for k, v in indicators(rec, q).items()})
            row[f'{name}_abstencion'] = rec['abstencion'] if rec else None
            row[f'{name}_inyeccion'] = (rec['inyeccion'] if is_injection(q) else 'no_aplica') if rec else None
        names = [n for n in LLM_JUDGES if n in judges]
        if len(names) == 2:
            for k in INDICATORS: row[f'prom_{k}'] = pair_mean(row[f'{names[0]}_{k}'], row[f'{names[1]}_{k}'])
        rows.append(row)
    return pd.DataFrame(rows)

def check_coverage(judges, clave):
    expected = {c['rid'] for c in clave}
    for name, recs in judges.items():
        missing = sorted(expected - set(recs))
        if missing: raise SystemExit(f'El juez {name} no calificó {len(missing)} respuestas: {missing[:8]}…')

def qwk(a, b, k=3):
    """Kappa ponderado cuadrático entre dos calificaciones ordinales 0..k-1; NaN si no está definido."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    keep = ~(np.isnan(a) | np.isnan(b))
    a, b = a[keep].astype(int), b[keep].astype(int)
    if not len(a): return NAN
    observed = np.zeros((k, k))
    for x, y in zip(a, b): observed[x, y] += 1
    weights = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)]) / (k - 1) ** 2
    expected = np.outer(observed.sum(1), observed.sum(0)) / observed.sum()
    return float(1 - (weights * observed).sum() / (weights * expected).sum()) if (weights * expected).sum() else NAN

def cohen_kappa(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs: return NAN
    a, b, n = [p[0] for p in pairs], [p[1] for p in pairs], len(pairs)
    po = sum(x == y for x, y in pairs) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in set(a) | set(b))
    return float((po - pe) / (1 - pe)) if pe < 1 else NAN

def agreement(grades, names):
    """Acuerdo exacto y QWK (corrección, fidelidad, citas) o kappa (abstención, inyección) entre calificadores."""
    rows = []
    for x, y in combinations(names, 2):
        if f'{x}_correccion' not in grades or f'{y}_correccion' not in grades: continue
        both = grades[grades[f'{x}_correccion'].notna() & grades[f'{y}_correccion'].notna()]
        for d in NUMERIC:
            a, b = both[f'{x}_{d}'], both[f'{y}_{d}']
            keep = a.notna() & b.notna()
            rows.append({'calificador_a': x, 'calificador_b': y, 'dimension': d, 'n': int(keep.sum()),
                         'acuerdo_exacto': float((a[keep] == b[keep]).mean()) if keep.any() else NAN,
                         'qwk': qwk(a[keep], b[keep]), 'kappa': NAN})
        for d in ('abstencion', 'inyeccion'):
            a, b = list(both[f'{x}_{d}']), list(both[f'{y}_{d}'])
            n = sum(p is not None and q is not None for p, q in zip(a, b))
            rows.append({'calificador_a': x, 'calificador_b': y, 'dimension': d, 'n': n,
                         'acuerdo_exacto': float(np.mean([p == q for p, q in zip(a, b) if p is not None and q is not None])) if n else NAN,
                         'qwk': NAN, 'kappa': cohen_kappa(a, b)})
    return pd.DataFrame(rows)

def agreement_with_mean(grades):
    """Promedio de los jueces redondeado frente a la auditoría humana (solo dimensiones ordinales)."""
    if 'humano_correccion' not in grades or 'prom_correccion' not in grades: return pd.DataFrame()
    rows, sub = [], grades[grades['humano_correccion'].notna()]
    for d in NUMERIC:
        a, b = sub[f'prom_{d}'].round(), sub[f'humano_{d}']
        keep = a.notna() & b.notna()
        rows.append({'calificador_a': 'prom (redondeado)', 'calificador_b': 'humano', 'dimension': d, 'n': int(keep.sum()),
                     'acuerdo_exacto': float((a[keep] == b[keep]).mean()) if keep.any() else NAN,
                     'qwk': qwk(a[keep].clip(0, 2), b[keep]), 'kappa': NAN})
    return pd.DataFrame(rows)

def summary(grades, names, B=BOOTSTRAP, seed=SEED):
    """Media e intervalo por modelo, partición y métrica, para cada juez y para su promedio."""
    tables = []
    for name in names:
        sub = grades[['modelo', 'split', 'qid', *[f'{name}_{k}' for k in INDICATORS]]]
        sub = sub.rename(columns={f'{name}_{k}': k for k in INDICATORS})
        per_question = sub.groupby(['modelo', 'split', 'qid'], as_index=False).mean(numeric_only=True)
        both = pd.concat([per_question, per_question.assign(split='todas')], ignore_index=True)
        tables.append(summarize(both, ['modelo', 'split'], list(INDICATORS), B, seed).assign(juez=name))
    out = pd.concat(tables, ignore_index=True)
    return out[['juez', 'modelo', 'split', 'metric', 'mean', 'ci_low', 'ci_high', 'n']]

def by_type(grades, B=BOOTSTRAP, seed=SEED):
    sub = grades[['modelo', 'tipo', 'qid', 'prom_correccion', 'prom_corr_2']].rename(columns={'prom_correccion': 'correccion', 'prom_corr_2': 'corr_2'})
    per_question = sub.groupby(['modelo', 'tipo', 'qid'], as_index=False).mean(numeric_only=True)
    return summarize(per_question, ['modelo', 'tipo'], ['correccion', 'corr_2'], B, seed)

def model_differences(grades, names, B=BOOTSTRAP, seed=SEED):
    """Diferencia pareada por pregunta entre cada par de modelos (corrección y proporción de respuestas con 2)."""
    rows = []
    for name in names:
        for metric in ('correccion', 'corr_2'):
            sub = grades[['modelo', 'split', 'qid', f'{name}_{metric}']].rename(columns={f'{name}_{metric}': 'v'})
            per_question = sub.groupby(['modelo', 'split', 'qid'], as_index=False).mean(numeric_only=True)
            for split, frame in (*per_question.groupby('split'), ('todas', per_question)):
                wide = frame.pivot(index='qid', columns='modelo', values='v')
                for a, b in combinations(sorted(wide.columns), 2):
                    mean, lo, hi, n = paired_diff_ci(wide[a], wide[b], B, seed)
                    rows.append({'juez': name, 'split': split, 'metric': metric, 'modelo_a': a, 'modelo_b': b,
                                 'diff_a_menos_b': mean, 'ci_low': lo, 'ci_high': hi, 'n': n,
                                 'incluye_cero': bool(lo <= 0 <= hi) if n else None})
    return pd.DataFrame(rows)

def disagreements(grades, judges, names=LLM_JUDGES):
    """Respuestas en las que los dos jueces difieren en la corrección, con sus notas (para leerlas a mano)."""
    a, b = names
    rows = []
    for _, r in grades[grades[f'{a}_correccion'] != grades[f'{b}_correccion']].iterrows():
        rows.append({'rid': r['rid'], 'modelo': r['modelo'], 'qid': r['qid'], 'tipo': r['tipo'],
                     f'{a}_correccion': r[f'{a}_correccion'], f'{b}_correccion': r[f'{b}_correccion'],
                     f'{a}_nota': judges[a][r['rid']]['nota'], f'{b}_nota': judges[b][r['rid']]['nota']})
    return pd.DataFrame(rows)

def diagnostics(grades, judges, gold, clave):
    """Incoherencias que conviene conocer: abstención del juez contra el gold e inyección fuera de las preguntas de inyección."""
    out = {'respuestas': len(grades), 'incoherencias_abstencion': 0, 'inyeccion_ignorada': 0}
    for name, recs in judges.items():
        for c in clave:
            rec, q = recs[c['rid']], gold[c['qid']]
            if (not q['debe_abstenerse'] and rec['abstencion'] in ('correcta', 'omitida')) or (q['debe_abstenerse'] and rec['abstencion'] == 'indebida'):
                out['incoherencias_abstencion'] += 1
            if not is_injection(q) and rec['inyeccion'] != 'no_aplica': out['inyeccion_ignorada'] += 1
    return out

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gold', default='data/gold/gold.jsonl')
    ap.add_argument('--juez-dir', default='data/logs/eval/juez')
    ap.add_argument('--reports', default='reports/eval')
    ap.add_argument('--bootstrap', type=int, default=BOOTSTRAP)
    args = ap.parse_args(argv)
    jdir, tables = Path(args.juez_dir), Path(args.reports) / 'tables'
    tables.mkdir(parents=True, exist_ok=True)
    gold = {q['id']: q for q in read_jsonl(args.gold)}
    clave = read_jsonl(jdir / 'clave.jsonl')
    judges = {n: load_judge(jdir / f'{n}.jsonl') for n in LLM_JUDGES}
    check_coverage(judges, clave)
    human_path = jdir / 'auditoria_humana.csv'
    human = load_human(human_path) if human_path.is_file() else None
    grades = build_grades(judges, clave, gold, human)
    names = [*LLM_JUDGES, 'prom']
    grades.to_csv(tables / 'juez_calificaciones.csv', index=False)
    result = summary(grades, names, args.bootstrap)
    result.to_csv(tables / 'juez_resumen.csv', index=False)
    by_type(grades, args.bootstrap).to_csv(tables / 'juez_por_tipo.csv', index=False)
    model_differences(grades, names, args.bootstrap).to_csv(tables / 'juez_diferencias_modelos.csv', index=False)
    people = [*LLM_JUDGES, *(['humano'] if human else [])]
    agree = pd.concat([agreement(grades, people), agreement_with_mean(grades)], ignore_index=True)
    agree.to_csv(tables / 'juez_acuerdo.csv', index=False)
    disagreements(grades, judges).to_csv(jdir / 'desacuerdos.csv', index=False)
    print('Diagnóstico:', diagnostics(grades, judges, gold, clave))
    print('\nAcuerdo entre calificadores:')
    print(agree.round(2).to_string(index=False))
    main_table = result[(result['juez'] == 'prom') & (result['split'] == 'test')]
    print('\nPromedio de los jueces, partición test (media por modelo):')
    print(main_table.pivot(index='modelo', columns='metric', values='mean')[list(INDICATORS)].round(2).to_string())
    corr = main_table[main_table['metric'] == 'correccion'].set_index('modelo')
    print('\nCorrección (0-2) con IC 95 %:')
    for model, r in corr.iterrows(): print(f"  {model:14s} {r['mean']:.2f} [{r['ci_low']:.2f}, {r['ci_high']:.2f}]  n={int(r['n'])}")

if __name__ == '__main__':
    main()
