"""Revisión del RAG con las preguntas de `dev` (Fase E): calibración del conteo de tokens y comportamiento del prompt.

Solo produce agregados e ids. Las respuestas, que pueden citar texto del corpus, van al JSONL de `data/logs/`.
`test` no se usa aquí: queda para la evaluación.
"""
import json, math
from pathlib import Path
import pandas as pd
from pipeline import metrics

def load_questions(gold_path, split='dev'):
    if split != 'dev': raise ValueError('La revisión del RAG solo usa dev; test queda para la evaluación')
    questions = [json.loads(l) for l in Path(gold_path).read_text(encoding='utf-8').splitlines() if l.strip()]
    return [q for q in questions if q['split'] == split]

def plan(question, default_collection, adversarial):
    """Colección y filtros de cada pregunta: las adversariales del gold van a su colección aparte."""
    collection = adversarial['name'] if question['id'] in adversarial['questions'] else default_collection
    window = question.get('ventana_fecha')
    return collection, ({'desde': window['desde'], 'hasta': window['hasta']} if window else {})

def calibration_table(rows):
    """Por modelo: cuántos tokens cuenta Ollama por cada token del encoder en los prompts reales."""
    data = pd.DataFrame(rows)
    return (data.groupby('model').agg(prompts=('ratio', 'size'), encoder_tokens_p50=('encoder_tokens', 'median'),
                                      llm_tokens_p50=('llm_tokens', 'median'), ratio_mean=('ratio', 'mean'),
                                      ratio_min=('ratio', 'min'), ratio_max=('ratio', 'max'))
            .reset_index())

def token_factor(table, margin):
    """El mayor cociente de todos los modelos más el margen, redondeado hacia arriba a múltiplos de 0,05.

    Un solo factor para todos los LLM: así el contexto que reciben es idéntico, como pide el experimento principal.
    """
    return math.ceil(table['ratio_max'].max() * (1 + margin) * 20 - 1e-9) / 20

def _docs(evidence):
    return {evidence['document_id'], *(evidence.get('provenance_ids') or [])}

def question_row(question, result, model):
    """Una fila por pregunta y modelo: banderas y cifras, sin texto."""
    relevant = set(metrics.relevance(question))
    context = [e for e in result['evidence'] if e['in_context']]
    cited = [e for e in result['evidence'] if e['cited']]
    synthetic = lambda e: e['document_id'].startswith('synthetic:')
    t, warnings = result['timings_ms'], set(result['warnings'])
    return {
        'prompt': result['prompt']['id'], 'model': model, 'id': question['id'], 'tipo': question['tipo'],
        'debe_abstenerse': question['debe_abstenerse'],
        'collection': result['retrieval']['collection'], 'k': result['retrieval']['k'],
        'n_context': result['context']['n_context'], 'omitidos': result['context']['omitted'],
        'relevante_en_contexto': any(_docs(e) & relevant for e in context) if relevant else None,
        'cita_relevante': any(_docs(e) & relevant for e in cited) if relevant else None,
        'abstuvo': result['abstained'], 'abstencion_correcta': result['abstained'] == question['debe_abstenerse'],
        'citas_validas': len(result['citations']), 'citas_invalidas': len(result['invalid_citations']),
        'sin_citas': 'sin_citas' in warnings, 'citas_en_texto': 'citas_en_texto' in warnings,
        'abstencion_parcial': 'abstencion_parcial' in warnings, 'solo_citas': 'solo_citas' in warnings,
        'respuesta_cortada': 'respuesta_cortada' in warnings,
        'filtra_instrucciones': 'filtra_instrucciones' in warnings,
        'sintetico_en_contexto': any(synthetic(e) for e in context), 'cita_sintetico': any(synthetic(e) for e in cited),
        'tokens_estimados': result['context']['estimated_tokens'], 'tokens_prompt': result['tokens']['prompt'],
        'tokens_respuesta': result['tokens']['completion'], 'tokens_por_s': result['tokens']['per_second'],
        'arranque_en_frio': result['cold_start'], 'ms_recuperacion': t['embed'] + t['search'], 'ms_llm': t['llm'],
        'ms_total': t['total']}

def summary(per_question):
    """Por prompt, modelo y tipo de pregunta (y total): tasas de abstención y citas, tokens y latencias en caliente."""
    def agg(g):
        warm = g[~g['arranque_en_frio']]
        rate = lambda col: g[col].dropna().astype(float).mean() if g[col].notna().any() else float('nan')
        return pd.Series({
            'n': len(g), 'abstencion': rate('abstuvo'), 'abstencion_correcta': rate('abstencion_correcta'),
            'relevante_en_contexto': rate('relevante_en_contexto'), 'cita_relevante': rate('cita_relevante'),
            'con_citas_validas': (g['citas_validas'] > 0).mean(), 'sin_citas': rate('sin_citas'),
            'citas_en_texto': rate('citas_en_texto'), 'abstencion_parcial': rate('abstencion_parcial'),
            'solo_citas': rate('solo_citas'),
            'con_citas_invalidas': (g['citas_invalidas'] > 0).mean(), 'respuesta_cortada': rate('respuesta_cortada'),
            'filtra_instrucciones': rate('filtra_instrucciones'), 'tokens_prompt_p50': g['tokens_prompt'].median(),
            'tokens_reales_sobre_estimados_max': (g['tokens_prompt'] / g['tokens_estimados']).max(),
            'tokens_por_s_p50': g['tokens_por_s'].median(), 'ms_total_p50': warm['ms_total'].median(),
            'ms_total_p95': warm['ms_total'].quantile(0.95), 'ms_llm_p50': warm['ms_llm'].median(),
            'ms_recuperacion_p50': warm['ms_recuperacion'].median(), 'arranques_en_frio': int(g['arranque_en_frio'].sum())})
    by_type = per_question.groupby(['prompt', 'model', 'tipo']).apply(agg, include_groups=False).reset_index()
    total = per_question.groupby(['prompt', 'model']).apply(agg, include_groups=False).reset_index().assign(tipo='total')
    return pd.concat([by_type, total], ignore_index=True)
