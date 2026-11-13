"""Desglose por plataforma de la Fase H (punto 6 del protocolo: sesgos por plataforma).

Agrupa las preguntas por `colecciones` del gold y recalcula, con el mismo bootstrap de `scripts.analyze`, las métricas de
recuperación, de generación y de los jueces. Grupos:
- prensa: solo `url_crawls`;
- bluesky: solo `bluesky_posts`;
- mixta: `bluesky_posts` y `url_crawls`;
- sintetica_prensa: `synthetic` y `url_crawls` (las preguntas de inyección).

Lee las tablas ya generadas en reports/eval/tables/ (sin texto del corpus) y escribe plataforma_recuperacion.csv,
plataforma_generacion.csv y plataforma_juez.csv en el mismo directorio. Los grupos de Bluesky tienen pocas preguntas
(ver la columna `n`): los intervalos son anchos y la lectura es descriptiva, no una prueba.
"""
import argparse, json
from pathlib import Path
import pandas as pd
from scripts.analyze import BOOTSTRAP, SEED, summarize

RETRIEVAL_METRICS = ['recall@5', 'hit@5', 'mrr', 'ndcg@10']
GENERATION_METRICS = ['abstencion_correcta', 'abstencion_indebida', 'respondio_sin_evidencia', 'contexto_con_relevante',
                      'cita_relevante', 'sin_citas']
JUDGE_METRICS = ['prom_correccion', 'prom_corr_ge1', 'prom_fidelidad', 'prom_citas']

def group_of(colecciones):
    """Grupo de plataforma de una pregunta según su lista `colecciones` del gold."""
    c = set(colecciones)
    if 'synthetic' in c: return 'sintetica_prensa'
    if c == {'bluesky_posts'}: return 'bluesky'
    if c == {'url_crawls'}: return 'prensa'
    if c == {'bluesky_posts', 'url_crawls'}: return 'mixta'
    return 'otro'

def load_groups(gold_path):
    rows = [json.loads(l) for l in Path(gold_path).read_text(encoding='utf-8').splitlines() if l.strip()]
    return {q['id']: group_of(q['colecciones']) for q in rows}

def with_all_splits(df):
    return pd.concat([df, df.assign(split='todas')], ignore_index=True)

def retrieval_by_group(per_question, groups, B=BOOTSTRAP, seed=SEED):
    df = per_question.assign(grupo=per_question['id'].map(groups))
    metrics = [m for m in RETRIEVAL_METRICS if m in df]
    return summarize(with_all_splits(df), ['indice', 'grupo', 'split'], metrics, B, seed)

def pooled_models(per_question, keys, metrics):
    """Promedio por pregunta entre modelos, para tener una fila 'todos' con más peso estadístico."""
    return per_question.groupby(keys, as_index=False)[metrics].mean(numeric_only=True).assign(model='todos')

def generation_by_group(cells, groups, condition='principal', B=BOOTSTRAP, seed=SEED):
    ok = cells[(~cells['error']) & (cells['condition'] == condition)].copy()
    metrics = [m for m in GENERATION_METRICS if m in ok]
    per_question = ok.groupby(['model', 'split', 'qid'], as_index=False)[metrics].mean()
    pooled = pooled_models(per_question, ['split', 'qid'], metrics)
    df = pd.concat([per_question, pooled], ignore_index=True)
    df['grupo'] = df['qid'].map(groups)
    return summarize(with_all_splits(df), ['model', 'grupo', 'split'], metrics, B, seed)

def judge_by_group(judge, groups, B=BOOTSTRAP, seed=SEED):
    metrics = [m for m in JUDGE_METRICS if m in judge]
    per_question = judge.rename(columns={'modelo': 'model'})[['model', 'qid', 'split', *metrics]]
    pooled = pooled_models(per_question, ['split', 'qid'], metrics)
    df = pd.concat([per_question, pooled], ignore_index=True)
    df['grupo'] = df['qid'].map(groups)
    return summarize(df, ['model', 'grupo', 'split'], metrics, B, seed)

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gold', default='data/gold/gold.jsonl')
    ap.add_argument('--reports', default='reports/eval')
    ap.add_argument('--bootstrap', type=int, default=BOOTSTRAP)
    args = ap.parse_args(argv)
    tables = Path(args.reports) / 'tables'
    groups = load_groups(args.gold)
    counts = pd.Series(groups).value_counts()
    print('Preguntas por grupo (dev y test):', counts.to_dict())
    jobs = [('recuperacion_por_pregunta.csv', 'plataforma_recuperacion.csv', retrieval_by_group),
            ('generacion_celdas.csv', 'plataforma_generacion.csv', generation_by_group),
            ('juez_calificaciones.csv', 'plataforma_juez.csv', judge_by_group)]
    for source, target, fn in jobs:
        src = tables / source
        if not src.is_file():
            print(f'Falta {src}; se omite {target}')
            continue
        out = fn(pd.read_csv(src), groups, B=args.bootstrap)
        out.to_csv(tables / target, index=False)
        view = out[out['split'] == 'test']
        key = 'indice' if 'indice' in view else 'model'
        if key == 'model': view = view[view['model'] == 'todos']
        print(f'\n{target} (test{", todos los modelos" if key == "model" else ""}):')
        print(view.pivot_table(index=[key, 'grupo'], columns='metric', values='mean').round(3).to_string())
        print(f'n por grupo: {view.groupby("grupo")["n"].max().to_dict()}')

if __name__ == '__main__':
    main()
