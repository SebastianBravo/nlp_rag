"""Análisis de la Fase H: métricas por celda, intervalos bootstrap y latencias, sin texto del corpus.

Lee el gold, `data/logs/eval/generacion.jsonl` (respuestas completas, no se versiona) y la tabla de recuperación de
`scripts.evaluate retrieval`. Escribe en reports/eval/tables/:
- generacion_celdas.csv: una fila por (condición, modelo, corrida, pregunta) con métricas derivadas y tiempos, sin respuestas;
- generacion_resumen.csv: media por (condición, modelo, split) con intervalo bootstrap del 95 % sobre preguntas;
- generacion_latencia.csv: p50/p95 de latencias y tokens por segundo, en frío y en caliente, más tasa de errores;
- recuperacion_resumen.csv y recuperacion_diferencia_AB.csv: métricas de A y B con intervalos y su diferencia pareada.

El bootstrap remuestrea preguntas (no corridas): con temperatura 0 las corridas casi no cambian la respuesta, así que
se promedian por pregunta antes de remuestrear. Con ~50 preguntas los intervalos son anchos y hay que leerlos como tales.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

SEED, BOOTSTRAP = 7, 10000
NAN = float('nan')
QUALITY = ['abstencion_correcta', 'abstencion_indebida', 'respondio_sin_evidencia', 'contexto_con_relevante',
           'contexto_con_relevante_g2', 'cita_relevante', 'precision_citas', 'sin_citas', 'citas_invalidas',
           'filtra_instrucciones']

def read_jsonl(path):
    p = Path(path)
    return [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()] if p.is_file() else []

def mean_ci(values, B=BOOTSTRAP, seed=SEED):
    """Media, límites del intervalo percentil del 95 % y n (sin NaN)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if not len(v): return NAN, NAN, NAN, 0
    means = v[np.random.default_rng(seed).integers(0, len(v), size=(B, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), len(v)

def paired_diff_ci(a, b, B=BOOTSTRAP, seed=SEED):
    """Diferencia a − b por pregunta (pares completos) con su intervalo bootstrap."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    keep = ~(np.isnan(a) | np.isnan(b))
    return mean_ci(a[keep] - b[keep], B, seed)

def summarize(df, group_cols, metrics, B=BOOTSTRAP, seed=SEED):
    """Tabla larga: una fila por grupo y métrica con media, intervalo y n."""
    rows = []
    for key, g in df.groupby(group_cols, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        for m in metrics:
            if m not in g: continue
            mean, lo, hi, n = mean_ci(g[m], B, seed)
            rows.append({**dict(zip(group_cols, key)), 'metric': m, 'mean': mean, 'ci_low': lo, 'ci_high': hi, 'n': n})
    return pd.DataFrame(rows)

def load_relevance(gold_path):
    """Por pregunta: documentos relevantes (grado >= 1 y >= 2) y si debe abstenerse."""
    out = {}
    for q in read_jsonl(gold_path):
        grades = {}
        for r in q['relevantes']: grades[r['document_id']] = max(grades.get(r['document_id'], 0), r['grado'])
        out[q['id']] = {'g1': {d for d, g in grades.items() if g >= 1}, 'g2': {d for d, g in grades.items() if g >= 2},
                        'abst': bool(q['debe_abstenerse'])}
    return out

def cell_row(rec, rel):
    """Métricas de una línea del JSONL. Una línea con error solo conserva la bandera de error."""
    base = {k: rec.get(k) for k in ('condition', 'model', 'run', 'qid', 'split', 'tipo', 'debe_abstenerse', 'k', 'prompt_id', 'count')}
    base['error'] = bool(rec.get('error'))
    if base['error']: return base
    r = rel[rec['qid']]
    answerable = (not r['abst']) and bool(r['g1'])
    abst = bool(rec.get('abstained'))
    cited = [c['document_id'] for c in rec.get('citations') or []]
    ctx, warnings = set(rec.get('context_doc_ids') or []), set(rec.get('warnings') or [])
    t, tok, res, mem = rec.get('timings_ms') or {}, rec.get('tokens') or {}, rec.get('resources') or {}, rec.get('model_memory') or {}
    total = sum(t.get(k, 0) for k in ('embed', 'search', 'context', 'llm'))
    return {**base,
            'abstained': abst, 'abstencion_correcta': float(abst == r['abst']),
            'abstencion_indebida': float(abst) if answerable else NAN,
            'respondio_sin_evidencia': float(not abst) if r['abst'] else NAN,
            'contexto_con_relevante': float(bool(ctx & r['g1'])) if answerable else NAN,
            'contexto_con_relevante_g2': float(bool(ctx & r['g2'])) if answerable and r['g2'] else NAN,
            'cita_relevante': float(any(c in r['g1'] for c in cited)) if answerable and not abst else NAN,
            'precision_citas': float(np.mean([c in r['g1'] for c in cited])) if answerable and cited else NAN,
            'sin_citas': float('sin_citas' in warnings) if not abst else NAN,
            'citas_invalidas': float(bool(rec.get('invalid_citations'))),
            'filtra_instrucciones': float('filtra_instrucciones' in warnings),
            'n_citas': len(cited), 'respuesta_cortada': float('respuesta_cortada' in warnings),
            'tokens_prompt': tok.get('prompt'), 'tokens_completion': tok.get('completion'), 'tokens_per_s': tok.get('per_second'),
            'embed_ms': t.get('embed'), 'search_ms': t.get('search'), 'llm_load_ms': t.get('llm_load'),
            'llm_prompt_ms': t.get('llm_prompt'), 'llm_generation_ms': t.get('llm_generation'), 'llm_ms': t.get('llm'),
            'total_ms': total, 'cold_start': bool(rec.get('cold_start')), 'model_vram_mb': mem.get('vram_mb'),
            'cpu_pct_mean': res.get('cpu_pct_mean'), 'system_used_mb_peak': res.get('system_used_mb_peak')}

def build_cells(raw, rel):
    """Última línea de cada clave (un error reintentado cuenta por su reintento)."""
    last = {}
    for rec in raw: last[rec['key']] = rec
    return pd.DataFrame([cell_row(r, rel) for r in last.values()])

def generation_summary(cells, B=BOOTSTRAP, seed=SEED):
    ok = cells[~cells['error']]
    metrics = [m for m in QUALITY if m in ok]
    per_question = ok.groupby(['condition', 'model', 'split', 'qid'], as_index=False)[metrics].mean()
    both = pd.concat([per_question, per_question.assign(split='todas')], ignore_index=True)
    return summarize(both, ['condition', 'model', 'split'], metrics, B, seed)

def latency_summary(cells):
    """p50 y p95 por condición, modelo y arranque (frío o caliente), y tasa de errores por condición y modelo."""
    ok = cells[~cells['error']].copy()
    ok['arranque'] = np.where(ok['cold_start'], 'frio', 'caliente')
    q = lambda p: (lambda s: s.quantile(p))
    agg = {f'{c}_p{int(p * 100)}': (c, q(p)) for c in ('total_ms', 'llm_generation_ms', 'tokens_per_s') for p in (0.5, 0.95)}
    table = ok.groupby(['condition', 'model', 'arranque']).agg(n=('qid', 'size'), **agg, model_vram_mb=('model_vram_mb', 'max'),
                                                              cpu_pct_mean=('cpu_pct_mean', 'mean')).reset_index()
    errors = cells.groupby(['condition', 'model'])['error'].agg(celdas='size', errores='sum').reset_index()
    errors['tasa_errores'] = errors['errores'] / errors['celdas']
    return table.merge(errors, on=['condition', 'model'], how='left')

def retrieval_summary(per_question, B=BOOTSTRAP, seed=SEED):
    metrics = [c for c in per_question.columns if '@' in c or c in ('mrr', 'mrr_g2')]
    both = pd.concat([per_question, per_question.assign(split='todas')], ignore_index=True)
    summary = summarize(both, ['indice', 'split'], metrics, B, seed)
    by_type = summarize(per_question, ['indice', 'tipo'], metrics, B, seed)
    wide = per_question.pivot(index='id', columns='indice', values=metrics)
    rows = []
    for m in metrics:
        mean, lo, hi, n = paired_diff_ci(wide[(m, 'A')], wide[(m, 'B')], B, seed)
        rows.append({'metric': m, 'diff_A_menos_B': mean, 'ci_low': lo, 'ci_high': hi, 'n': n,
                     'incluye_cero': bool(lo <= 0 <= hi)})
    return summary, by_type, pd.DataFrame(rows)

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gold', default='data/gold/gold.jsonl')
    ap.add_argument('--raw', default='data/logs/eval/generacion.jsonl')
    ap.add_argument('--reports', default='reports/eval')
    ap.add_argument('--bootstrap', type=int, default=BOOTSTRAP)
    args = ap.parse_args(argv)
    tables = Path(args.reports) / 'tables'
    tables.mkdir(parents=True, exist_ok=True)
    retrieval = tables / 'recuperacion_por_pregunta.csv'
    if retrieval.is_file():
        summary, by_type, diff = retrieval_summary(pd.read_csv(retrieval), args.bootstrap)
        summary.to_csv(tables / 'recuperacion_resumen.csv', index=False)
        by_type.to_csv(tables / 'recuperacion_por_tipo.csv', index=False)
        diff.to_csv(tables / 'recuperacion_diferencia_AB.csv', index=False)
        print('Recuperación A − B (pareada, IC 95 %):')
        print(diff.round(3).to_string(index=False))
    raw = read_jsonl(args.raw)
    if not raw:
        print(f'Sin generaciones en {args.raw}')
        return
    cells = build_cells(raw, load_relevance(args.gold))
    cells.to_csv(tables / 'generacion_celdas.csv', index=False)
    summary = generation_summary(cells, args.bootstrap)
    summary.to_csv(tables / 'generacion_resumen.csv', index=False)
    latency = latency_summary(cells)
    latency.to_csv(tables / 'generacion_latencia.csv', index=False)
    print(f'\n{len(cells)} celdas ({int(cells["error"].sum())} con error).')
    for split, label in (('test', 'partición test (resultado principal)'), ('todas', 'dev y test juntas')):
        table = summary[(summary['condition'] == 'principal') & (summary['split'] == split)]
        print(f'Experimento principal, {label}:')
        print(table.pivot(index='model', columns='metric', values='mean').round(3).to_string())

if __name__ == '__main__':
    main()
