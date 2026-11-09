"""Orquestación RAG (Fase E): recuperar, armar el contexto con presupuesto en tokens, generar y validar citas.

Cada paso es una función aparte, así la evaluación puede recuperar una vez y generar con varios LLM sobre
exactamente el mismo contexto. El prompt se lee de `prompts/<id>.yaml` y cada respuesta registra su sha256.
"""
import hashlib, json, math, os, re, time, unicodedata
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
import yaml
from app import core, llm
from app.embeddings import count_tokens, settings

TAG = re.compile(r'<\s*(/?)\s*documento', re.IGNORECASE)
CITATION = re.compile(r'\[\s*(\d+(?:\s*[,;]\s*\d+)*)\s*\]')
# Menciones como "según el documento 2", "documentos 1 y 4" o "document 3": cuentan como cita, con aviso de formato.
TEXT_CITATION = re.compile(r'\bdocument(?:o|os|s)?\s+(?:(?:n\.?|nº|n°|no\.|número|number)\s*)?(\d+(?:\s*(?:,|y|e|and|&)\s*\d+)*)',
                           re.IGNORECASE)
COLD_START_MS = 500
# Títulos más largos se recortan en el prompt: el p99 de la muestra de 10.000 es de 146 caracteres, pero 16 chunks
# traen títulos de hasta 7.998 (texto de la página, no un título) que consumían el presupuesto de otros fragmentos.
TITLE_MAX = 300

@lru_cache(maxsize=8)
def load_prompt(prompt_id=None):
    """Prompt versionado con su sha256; el texto del sistema ya trae las frases de abstención y el canario."""
    prompt_id = prompt_id or os.getenv('PROMPT_ID', 'rag_v1')
    path = Path(os.getenv('PROMPTS_DIR', 'prompts')) / f'{prompt_id}.yaml'
    raw = path.read_bytes()
    spec = yaml.safe_load(raw)
    if spec['id'] != prompt_id: raise ValueError(f'{path} declara id={spec["id"]!r}, no {prompt_id!r}')
    spec['sha256'] = hashlib.sha256(raw).hexdigest()
    spec['system_text'] = spec['sistema'].format(abstencion_es=spec['abstencion']['es'],
                                                 abstencion_en=spec['abstencion']['en'], canario=spec['canario'])
    return spec

def budget_settings():
    """Ventana del LLM y presupuesto: `num_ctx` fijo, respuesta de hasta `num_predict` tokens y factor de conversión."""
    return {'num_ctx': int(os.getenv('LLM_NUM_CTX', '4096')), 'num_predict': int(os.getenv('LLM_NUM_PREDICT', '512')),
            'token_factor': float(os.getenv('LLM_TOKEN_FACTOR', '1.3'))}

def default_seed():
    return int(os.getenv('LLM_SEED', '42'))

def llm_options(budget, temperature=0.0, seed=None):
    """Opciones de Ollama. `num_thread` deja núcleos libres: con todos ocupados, etcd y Milvus perdieron su sesión."""
    options = {'temperature': temperature, 'seed': default_seed() if seed is None else seed,
               'num_ctx': budget['num_ctx'], 'num_predict': budget['num_predict']}
    threads = int(os.getenv('LLM_NUM_THREAD', '8') or 0)
    return {**options, 'num_thread': threads} if threads else options

def neutralize(text):
    """Evita que un texto recuperado abra o cierre las etiquetas <documento> del prompt."""
    return TAG.sub(lambda m: '‹' + m.group(1) + 'documento', text or '')

def format_document(n, chunk, prompt):
    created = chunk.get('created_at')
    date = datetime.fromtimestamp(created, timezone.utc).date().isoformat() if created is not None else prompt['sin_fecha']
    title = ' '.join(neutralize(chunk.get('title')).split())
    if len(title) > TITLE_MAX: title = title[:TITLE_MAX].rstrip() + '…'
    return prompt['documento'].format(
        n=n, fuente=prompt['fuentes'].get(chunk.get('collection_name'), prompt['fuente_desconocida']), fecha=date,
        idioma=chunk.get('language') or 'und', titulo=title or prompt['sin_titulo'], texto=neutralize(chunk['text']).strip())

def build_context(question, chunks, prompt, budget, count=None):
    """Mensajes para el LLM con los chunks que caben en el presupuesto, en orden de rank.

    Los tokens se cuentan con el tokenizador del encoder (ya cargado en la API) y se multiplican por
    `token_factor`, calibrado con el conteo real de Ollama en los LLM del experimento. Un chunk que no cabe
    se omite junto con los de peor rank: el contexto es siempre un prefijo del ranking.
    """
    count = count or (lambda texts: count_tokens(texts, 'passage'))
    limit = budget['num_ctx'] - budget['num_predict']
    docs = [format_document(i + 1, c, prompt) for i, c in enumerate(chunks)]
    system = prompt['system_text']
    counts = count([system, prompt['usuario'].format(documentos='', pregunta=question), *docs])
    used, kept = counts[0] + counts[1], 0
    for tokens in counts[2:]:
        if (used + tokens) * budget['token_factor'] > limit: break
        used, kept = used + tokens, kept + 1
    return {'system': system, 'user': prompt['usuario'].format(documentos='\n\n'.join(docs[:kept]), pregunta=question),
            'n_context': kept, 'omitted': len(chunks) - kept, 'proxy_tokens': used,
            'estimated_tokens': math.ceil(used * budget['token_factor']), 'limit': limit}

def _numbers(pattern, answer):
    return {int(x) for m in pattern.finditer(answer) for x in re.findall(r'\d+', m.group(1))}

def text_citations(answer):
    """Documentos citados solo con palabras ("según el documento 2"), no con [n]."""
    return _numbers(TEXT_CITATION, answer) - _numbers(CITATION, answer)

def parse_citations(answer, n_context):
    """Números citados como [n], [n, m], [n][m] o "documento n": (válidos, inválidos), sin repetir."""
    cited = _numbers(CITATION, answer) | _numbers(TEXT_CITATION, answer)
    return sorted(n for n in cited if 1 <= n <= n_context), sorted(n for n in cited if not 1 <= n <= n_context)

def _normalize(text):
    text = ''.join(ch for ch in unicodedata.normalize('NFKD', text.lower()) if not unicodedata.combining(ch))
    return re.sub(r'[^a-z0-9]+', ' ', text).strip()

def is_abstention(answer, prompt):
    """La respuesta contiene alguna de las frases fijas de abstención del prompt."""
    text = _normalize(answer)
    return any(_normalize(phrase) in text for phrase in prompt['abstencion'].values())

def generate(question, chunks, model, prompt=None, budget=None, temperature=0.0, seed=None, count=None, chat=None):
    """Genera la respuesta a partir de chunks ya recuperados y la valida (citas, abstención y avisos)."""
    prompt, budget = prompt or load_prompt(), budget or budget_settings()
    chat = chat or llm.chat
    options = llm_options(budget, temperature, seed)
    t0 = time.perf_counter()
    ctx = build_context(question, chunks, prompt, budget, count)
    t1 = time.perf_counter()
    out = chat(model, ctx['system'], ctx['user'], options) if ctx['n_context'] else None
    t2 = time.perf_counter()
    answer = out['text'].strip() if out else prompt['abstencion']['es']
    valid, invalid = parse_citations(answer, ctx['n_context'])
    # Con la frase de abstención y citas válidas, el modelo respondió una parte: no cuenta como abstención.
    phrase = is_abstention(answer, prompt)
    abstained = phrase and not valid
    warnings = [name for name, hit in (
        ('sin_evidencia', not ctx['n_context']),
        ('contexto_recortado', ctx['omitted'] > 0),
        ('sin_citas', not abstained and not valid),
        ('citas_invalidas', bool(invalid)),
        ('citas_en_texto', bool(text_citations(answer))),
        ('abstencion_parcial', phrase and bool(valid)),
        ('solo_citas', bool(out) and not re.sub(r'[\s.,;:]', '', CITATION.sub('', answer))),
        ('respuesta_cortada', bool(out) and out['done_reason'] == 'length'),
        ('posible_truncamiento', bool(out) and (out['prompt_tokens'] or 0) > ctx['limit']),
        ('filtra_instrucciones', prompt['canario'].lower() in answer.lower())) if hit]
    per_second = round(out['completion_tokens'] / (out['generation_ms'] / 1e3), 2) if out and out['generation_ms'] else None
    return {
        'answer': answer, 'model': model, 'abstained': abstained, 'warnings': warnings,
        'citations': [{'n': n, 'chunk_id': chunks[n - 1]['chunk_id'], 'document_id': chunks[n - 1]['document_id'],
                       'url': chunks[n - 1].get('url') or None} for n in valid],
        'invalid_citations': invalid,
        'context': {k: ctx[k] for k in ('n_context', 'omitted', 'proxy_tokens', 'estimated_tokens', 'limit')},
        'tokens': {'prompt': out['prompt_tokens'] if out else 0, 'completion': out['completion_tokens'] if out else 0,
                   'per_second': per_second},
        'timings_ms': {'context': round((t1 - t0) * 1e3, 1), 'llm': round((t2 - t1) * 1e3, 1),
                       **({'llm_load': out['load_ms'], 'llm_prompt': out['prompt_ms'], 'llm_generation': out['generation_ms']}
                          if out else {})},
        'cold_start': bool(out) and out['load_ms'] > COLD_START_MS,
        'prompt': {'id': prompt['id'], 'version': prompt['version'], 'sha256': prompt['sha256']},
        'options': options, 'token_factor': budget['token_factor']}

def answer(question, model, k=5, filters=None, collection=None, temperature=0.0, seed=None, prompt_id=None):
    """Recupera en Milvus y genera; las evidencias indican si entraron al contexto y si se citaron."""
    t0 = time.perf_counter()
    retrieval = core.retrieve(question, k, filters, collection)
    result = generate(question, retrieval['chunks'], model, load_prompt(prompt_id), budget_settings(), temperature, seed)
    cited = {c['n'] for c in result['citations']}
    evidence = [{**c, 'rank': i + 1, 'in_context': i < result['context']['n_context'], 'cited': i + 1 in cited}
                for i, c in enumerate(retrieval['chunks'])]
    timings = {**retrieval['timings_ms'], **result['timings_ms'], 'total': round((time.perf_counter() - t0) * 1e3, 1)}
    return {**result, 'query': question, 'evidence': evidence, 'timings_ms': timings,
            'retrieval': {'collection': retrieval['collection'], 'filter': retrieval['filter'], 'ef': retrieval['ef'], 'k': k}}

def run_record(run_id, request, result=None, error=None, model_digest=None):
    """Línea del registro JSONL: parámetros, ids recuperados, tokens, tiempos y respuesta (sin el texto de los chunks)."""
    record = {'run_id': run_id, 'created_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
              'request': request, 'model_digest': model_digest, 'encoder': settings(), 'error': error}
    if result:
        record.update({key: result[key] for key in ('prompt', 'options', 'token_factor', 'retrieval', 'context', 'tokens',
                                                    'timings_ms', 'cold_start', 'answer', 'abstained', 'citations',
                                                    'invalid_citations', 'warnings')})
        record['evidence'] = [{key: e[key] for key in ('rank', 'chunk_id', 'document_id', 'score', 'in_context', 'cited')}
                              for e in result['evidence']]
    return record

def write_log(record, log_dir=None):
    """Agrega el registro a `<LOG_DIR>/ask/<fecha>.jsonl`. Puede contener texto del corpus: `data/` no se versiona."""
    path = Path(log_dir or os.getenv('LOG_DIR', 'data/logs')) / 'ask' / f'{record["created_at_utc"][:10]}.jsonl'
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as f: f.write(json.dumps(record, ensure_ascii=False) + '\n')
        return str(path)
    except OSError:
        return None
