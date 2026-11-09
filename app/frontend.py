"""Interfaz Streamlit (Fase F): pregunta, modelo, k y filtros; muestra respuesta, citas, evidencias y tiempos."""
import os
from datetime import date, datetime, timezone
import requests
import streamlit as st

API = os.getenv('API_URL', 'http://localhost:8000')
LANGUAGES = {'Todos': None, 'Español (es)': 'es', 'Inglés (en)': 'en', 'Húngaro (hu)': 'hu', 'Alemán (de)': 'de',
             'Portugués (pt)': 'pt', 'Francés (fr)': 'fr', 'Neerlandés (nl)': 'nl', 'Italiano (it)': 'it'}
SOURCES = {'Todas': None, 'Prensa digital': 'url_crawls', 'Bluesky': 'bluesky_posts'}
SOURCE_LABEL = {'url_crawls': 'prensa', 'bluesky_posts': 'Bluesky', 'synthetic': 'sintético'}
WARNINGS = {
    'sin_evidencia': 'No se recuperó ningún fragmento con estos filtros, así que no se consultó el LLM.',
    'contexto_recortado': 'Algunos fragmentos no entraron al contexto por el presupuesto de tokens.',
    'sin_citas': 'La respuesta no cita ningún documento: trátala con cautela.',
    'citas_invalidas': 'La respuesta cita documentos que no se le entregaron.',
    'citas_en_texto': 'La respuesta cita documentos con palabras ("según el documento n") en lugar de [n].',
    'abstencion_parcial': 'La respuesta contesta una parte y dice que el resto no consta en los documentos.',
    'solo_citas': 'La respuesta solo trae números de documentos, sin contenido.',
    'respuesta_cortada': 'La respuesta llegó al máximo de tokens y puede estar incompleta.',
    'posible_truncamiento': 'El prompt superó el presupuesto y Ollama pudo recortarlo.',
    'filtra_instrucciones': 'La respuesta revela instrucciones internas del sistema.'}

st.set_page_config(page_title='Asistente RAG', layout='wide')

@st.cache_data(ttl=30, show_spinner=False)
def _get(path):
    r = requests.get(API + path, timeout=10)
    return r.status_code, r.json()

def fetch(path):
    try: return _get(path)
    except requests.RequestException as e: return None, {'detail': f'No se pudo conectar con la API en {API}: {e}'}

def error_message(status, body):
    detail = body.get('detail', body) if isinstance(body, dict) else body
    if isinstance(detail, list): detail = '; '.join(f"{'.'.join(map(str, d.get('loc', [])[1:]))}: {d.get('msg')}" for d in detail)
    return {404: f'No encontrado. {detail}', 422: f'Revisa los parámetros. {detail}', 503: f'Un servicio no responde. {detail}',
            504: f'{detail}. Prueba con un modelo más pequeño o con menos fragmentos.'}.get(status, f'Error {status}: {detail}')

def day(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).date().isoformat() if epoch is not None else 'sin fecha'

def num(n):
    return f'{n:,}'.replace(',', '.')

def dot(ok):
    return ':green[●]' if ok else ':red[●]'

with st.sidebar:
    st.header('Estado')
    status, health = fetch('/health')
    if status is None:
        st.error(health['detail'])
    else:
        milvus, ollama = health.get('milvus', {}), health.get('ollama', {})
        st.markdown(f"{dot(milvus.get('ok'))} Milvus: {num(milvus.get('entities') or 0)} fragmentos en `{milvus.get('collection', '?')}`"
                    if 'collection' in milvus else f"{dot(False)} Milvus: {milvus.get('error', 'sin respuesta')}")
        st.markdown(f"{dot(ollama.get('ok'))} Ollama {ollama.get('version', '')}: {ollama.get('models', 0)} modelos"
                    if ollama.get('ok') else f"{dot(False)} Ollama: {ollama.get('error', 'sin respuesta')}")
    if st.button('Actualizar estado'):
        st.cache_data.clear()
        st.rerun()

    st.header('Consulta')
    status, data = fetch('/models')
    models = {m['name']: m for m in data.get('models', [])} if status == 200 else {}
    if status == 200 and not models:
        st.warning('No hay modelos descargados en Ollama. Descarga uno con '
                   '`docker compose exec ollama ollama pull qwen2.5:1.5b` y pulsa "Actualizar estado".')
    elif status is not None and status != 200:
        st.warning(f"No se pudo leer la lista de modelos: {data.get('detail', status)}")
    names = list(models)
    default = data.get('default') if status == 200 else None
    model = st.selectbox('LLM local', names, index=names.index(default) if default in names else (0 if names else None),
                         format_func=lambda n: f"{n} · {models[n].get('quantization') or '?'} · {num(models[n]['size_mb'])} MB",
                         disabled=not names)
    k = st.slider('Fragmentos a recuperar (k)', 1, 15, 5)
    with st.expander('Opciones avanzadas'):
        status, data = fetch('/collections')
        collections = [c['name'] for c in data.get('collections', [])] if status == 200 else []
        default = data.get('default') if status == 200 else None
        collection = st.selectbox('Colección de Milvus', collections or [None],
                                  index=collections.index(default) if default in collections else 0,
                                  format_func=lambda n: '(la de la API)' if n is None
                                  else f'{n} (prueba adversarial)' if '_adv_' in n else n)
        language = LANGUAGES[st.selectbox('Idioma', list(LANGUAGES))]
        source = SOURCES[st.selectbox('Fuente', list(SOURCES))]
        by_date = st.checkbox('Filtrar por fecha')
        desde = st.date_input('Desde', value=date(2024, 1, 1), disabled=not by_date)
        hasta = st.date_input('Hasta', value=date.today(), disabled=not by_date)
        temperature = st.slider('Temperatura', 0.0, 1.0, 0.0, 0.1)
        seed = st.number_input('Semilla', min_value=0, value=42, step=1)

st.title('Asistente RAG sobre prensa y Bluesky')
st.caption('Responde con la evidencia que recupera de Milvus y un LLM local. Contrasta siempre las fuentes originales.')

with st.form('consulta'):
    question = st.text_area('Pregunta', placeholder='¿Qué se acordó en la plenaria de la COP16?', height=100)
    submitted = st.form_submit_button('Consultar', type='primary', disabled=not names)

if submitted:
    filters = {key: value for key, value in {'language': language, 'collection_name': source}.items() if value}
    if by_date: filters.update(desde=desde.isoformat(), hasta=hasta.isoformat())
    payload = {'query': question.strip(), 'model': model, 'k': k, 'temperature': temperature, 'seed': int(seed),
               'collection': collection, 'filters': filters}
    st.session_state.pop('result', None)
    if len(payload['query']) < 3:
        st.warning('Escribe una pregunta de al menos 3 caracteres.')
    elif by_date and desde > hasta:
        st.warning('La fecha inicial es posterior a la final.')
    else:
        with st.spinner(f'Recuperando evidencia y generando la respuesta con {model}…'):
            try:
                r = requests.post(API + '/ask', json=payload, timeout=300)
                if r.ok: st.session_state['result'] = r.json()
                else: st.error(error_message(r.status_code, r.json()))
            except requests.Timeout:
                st.error('La API no respondió en 5 minutos. Prueba con un modelo más pequeño o con menos fragmentos.')
            except requests.RequestException as e:
                st.error(f'No se pudo conectar con la API en {API}: {e}')

out = st.session_state.get('result')
if out:
    st.markdown(f"**Pregunta:** {out['query']}")
    st.subheader('Respuesta')
    if out['abstained']: st.info('El modelo se abstuvo: no encontró evidencia suficiente en los documentos.')
    st.markdown(out['answer'])
    for warning in out['warnings']: st.warning(WARNINGS.get(warning, warning))

    t = out['timings_ms']
    cols = st.columns(5)
    cols[0].metric('Tiempo total', f"{t['total'] / 1e3:.1f} s")
    cols[1].metric('Recuperación', f"{(t['embed'] + t['search']) / 1e3:.2f} s")
    cols[2].metric('Generación', f"{t['llm'] / 1e3:.1f} s")
    cols[3].metric('Tokens/s', out['tokens']['per_second'] or '—')
    cols[4].metric('En el contexto', f"{out['context']['n_context']} de {len(out['evidence'])}")
    prompt, options = out['prompt'], out['options']
    st.caption(f"Modelo `{out['model']}` · prompt `{prompt['id']}` v{prompt['version']} (`{prompt['sha256'][:8]}`) · "
               f"colección `{out['retrieval']['collection']}` · temperatura {options['temperature']} · semilla {options['seed']} · "
               f"tokens del prompt {out['tokens']['prompt']} · ejecución `{out['run_id'][:8]}`"
               + (' · arranque en frío' if out['cold_start'] else ''))

    st.subheader('Evidencias')
    if not out['evidence']: st.write('No se recuperó ningún fragmento.')
    for e in out['evidence']:
        tags = [label for label, hit in (('citada', e['cited']), ('fuera del contexto', not e['in_context'])) if hit]
        title = e.get('title') or e['document_id']
        header = (f"[{e['rank']}] {title[:90]} · {SOURCE_LABEL.get(e['collection_name'], e['collection_name'])} · "
                  f"{day(e.get('created_at'))} · score {e['score']:.3f}" + (f" · {', '.join(tags)}" if tags else ''))
        with st.expander(header, expanded=e['cited']):
            st.markdown(f"Documento `{e['document_id']}` · idioma `{e.get('language')}` · caracteres "
                        f"{num(e['char_start'])}–{num(e['char_end'])} · {len(e.get('provenance_ids') or [])} procedencia(s)")
            if e.get('url'): st.markdown(f"[Abrir la fuente original]({e['url']})")
            st.text(e['text'])
