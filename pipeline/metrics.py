"""Métricas de recuperación a nivel documento (Recall@k, MRR, nDCG@k, hit@k) contra el gold estándar.

Convenciones (data/gold/README.md):
- solo cuentan las preguntas respondibles: `debe_abstenerse = false` y con `relevantes`;
- relevante = documento con `grado >= 1`;
- nDCG usa ganancia lineal igual al `grado` (0/1/2) y descuento log2(posición + 1).
"""
import math
import pandas as pd

def answerable(question):
    return not question['debe_abstenerse'] and bool(question['relevantes'])

def relevance(question):
    """{document_id: grado} con el mayor grado anotado por documento (grado >= 1)."""
    grades = {}
    for r in question['relevantes']:
        if r['grado'] >= 1: grades[r['document_id']] = max(grades.get(r['document_id'], 0), r['grado'])
    return grades

def recall_at_k(ranked, relevant, k):
    return len(set(ranked[:k]) & set(relevant)) / len(relevant) if relevant else float('nan')

def hit_at_k(ranked, relevant, k):
    return float(bool(set(ranked[:k]) & set(relevant)))

def reciprocal_rank(ranked, relevant):
    return next((1 / (i + 1) for i, doc in enumerate(ranked) if doc in relevant), 0.0)

def ndcg_at_k(ranked, grades, k):
    dcg = sum(grades.get(doc, 0) / math.log2(i + 2) for i, doc in enumerate(ranked[:k]))
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else float('nan')

def question_metrics(ranked, question, ks=(1, 5, 10, 20)):
    grades = relevance(question)
    row = {f'recall@{k}': recall_at_k(ranked, grades, k) for k in ks}
    row.update({f'hit@{k}': hit_at_k(ranked, grades, k) for k in ks})
    row.update({'mrr': reciprocal_rank(ranked, grades), 'ndcg@10': ndcg_at_k(ranked, grades, 10)})
    return row

def evaluate(rankings, questions, ks=(1, 5, 10, 20), extra=None):
    """Una fila por pregunta respondible con sus métricas. `rankings`: {id: [document_id, ...]}; `extra`: {id: {columna: valor}}."""
    rows = []
    for q in questions:
        if not answerable(q) or q['id'] not in rankings: continue
        rows.append({'id': q['id'], 'tipo': q['tipo'], 'idioma_pregunta': q['idioma_pregunta'],
                     'idioma_evidencia': q['idioma_evidencia'], 'colecciones': '+'.join(q['colecciones'] or []),
                     **question_metrics(rankings[q['id']], q, ks), **((extra or {}).get(q['id'], {}))})
    return pd.DataFrame(rows)

def summarize(per_question, by=None):
    """Promedio de cada métrica, en total y opcionalmente por grupo."""
    metrics = [c for c in per_question.columns if '@' in c or c in ('mrr',) or c.startswith('evidence')]
    total = per_question[metrics].mean().to_frame().T.assign(group='total', n=len(per_question))
    if not by: return total[['group', 'n', *metrics]]
    groups = per_question.groupby(by)[metrics].mean().reset_index().rename(columns={by: 'group'})
    groups['n'] = per_question.groupby(by).size().to_numpy()
    return pd.concat([groups, total], ignore_index=True)[['group', 'n', *metrics]]
