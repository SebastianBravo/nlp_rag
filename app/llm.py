"""Cliente de Ollama: modelos locales y generación con `/api/chat` (instrucciones del sistema separadas del usuario)."""
import os
import requests

class LLMError(Exception):
    """Error al consultar Ollama; `status` es el código HTTP que devuelve la API."""
    status = 503

class ModelNotFound(LLMError):
    status = 404

class LLMTimeout(LLMError):
    status = 504

def base_url():
    return os.getenv('OLLAMA_URL', 'http://ollama:11434').rstrip('/')

def _get(path, timeout=5):
    try:
        r = requests.get(base_url() + path, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        raise LLMError(f'Ollama no responde en {base_url()}: {e}') from e

def version():
    return _get('/api/version').get('version')

def models():
    """Modelos descargados en Ollama, con su digest, tamaño y cuantización."""
    return sorted(({'name': m['name'], 'digest': m.get('digest', ''), 'size_mb': round(m.get('size', 0) / 1e6),
                    'parameter_size': (m.get('details') or {}).get('parameter_size'),
                    'quantization': (m.get('details') or {}).get('quantization_level')}
                   for m in _get('/api/tags').get('models', [])), key=lambda m: m['name'])

def chat(model, system, user, options, timeout=None):
    """Una respuesta sin streaming. Devuelve el texto y las métricas de Ollama (tokens y duraciones en ms)."""
    timeout = timeout or float(os.getenv('LLM_TIMEOUT', '240'))
    body = {'model': model, 'stream': False, 'options': options,
            'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]}
    try:
        r = requests.post(base_url() + '/api/chat', json=body, timeout=timeout)
    except requests.Timeout as e:
        raise LLMTimeout(f'El modelo {model} no respondió en {timeout:.0f} s') from e
    except requests.RequestException as e:
        raise LLMError(f'Ollama no responde en {base_url()}: {e}') from e
    if r.status_code == 404:
        raise ModelNotFound(f'El modelo {model} no está descargado en Ollama. Descárgalo con: '
                            f'docker compose exec ollama ollama pull {model}')
    if not r.ok:
        raise LLMError(f'Ollama devolvió {r.status_code}: {r.text[:300]}')
    data = r.json()
    ms = lambda key: round(data.get(key, 0) / 1e6, 1)
    return {'text': data.get('message', {}).get('content', ''), 'done_reason': data.get('done_reason'),
            'prompt_tokens': data.get('prompt_eval_count'), 'completion_tokens': data.get('eval_count'),
            'load_ms': ms('load_duration'), 'prompt_ms': ms('prompt_eval_duration'),
            'generation_ms': ms('eval_duration'), 'llm_total_ms': ms('total_duration')}
