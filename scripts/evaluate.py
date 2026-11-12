"""Evaluador de la Fase H: recuperación (índices A y B) y generación (experimento principal y ablaciones).

Dos modos:
- `retrieval`: consulta las colecciones A y B con las preguntas respondibles del gold y escribe métricas por pregunta,
  latencias y rankings (solo ids) en reports/eval/.
- `generation`: para cada corrida, condición, modelo y pregunta, recupera una vez (todos los LLM de una condición reciben
  exactamente el mismo contexto), genera con `rag.generate` y escribe una línea JSONL con la respuesta, las citas, los
  tiempos y los recursos. Las respuestas pueden citar el corpus: van a data/logs/ y no se versionan.

Es reanudable: una línea sin `error` cuenta como hecha; un error se reintenta al relanzar.
Las condiciones y los parámetros están en configs/experimentos.yaml. Ejemplo de prueba rápida:
    python -m scripts.evaluate generation --limit 3 --runs 1 --conditions principal --models llama3.2:1b \
        --out data/logs/eval/prueba.jsonl
"""
import argparse, hashlib, json, os, platform, subprocess, sys, threading
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import yaml

os.environ.setdefault('GRPC_VERBOSITY', 'ERROR')   # sin los avisos de gRPC al lanzar subprocesos
KS = (1, 5, 10, 20)

def read_jsonl(path):
    p = Path(path)
    return [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()] if p.is_file() else []

def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_questions(path, splits=None, limit=None):
    questions = read_jsonl(path)
    if splits: questions = [q for q in questions if q['split'] in splits]
    return questions[:limit] if limit else questions

def filters_of(question):
    window = question.get('ventana_fecha')
    return {'desde': window['desde'], 'hasta': window['hasta']} if window else {}

def rank_docs(chunks, skip_synthetic=False):
    """Documentos en orden de aparición. Cada chunk aporta su documento y los que comparten su texto (procedencias)."""
    seen, ranked = set(), []
    for chunk in chunks:
        if skip_synthetic and chunk.get('collection_name') == 'synthetic': continue
        for doc in [chunk['document_id'], *(chunk.get('provenance_ids') or [])]:
            if doc not in seen: seen.add(doc); ranked.append(doc)
    return ranked

class ResourceSampler:
    """Muestrea la CPU del sistema y la memoria de los procesos de Ollama mientras dura una petición."""
    def __init__(self, interval=0.25, match='ollama'):
        try: import psutil
        except ImportError: psutil = None
        self.ps, self.interval, self.match = psutil, interval, match
        self.samples, self._stop, self._thread = [], threading.Event(), None

    def _sample(self):
        ps, rss = self.ps, 0
        for p in ps.process_iter(['name', 'memory_info']):
            try:
                if self.match in (p.info['name'] or '').lower(): rss += p.info['memory_info'].rss
            except (ps.NoSuchProcess, ps.AccessDenied): continue
        return ps.cpu_percent(interval=None), rss / 2 ** 20, ps.virtual_memory().used / 2 ** 20

    def _loop(self):
        while not self._stop.wait(self.interval): self.samples.append(self._sample())

    def __enter__(self):
        if self.ps:
            self.ps.cpu_percent(interval=None)
            self.samples.append(self._sample())
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread: self._thread.join(timeout=2)
        if self.ps: self.samples.append(self._sample())

    def summary(self):
        if not self.samples: return None
        cpu, rss, used = zip(*self.samples)
        return {'samples': len(cpu), 'cpu_pct_mean': round(sum(cpu) / len(cpu), 1), 'cpu_pct_max': round(max(cpu), 1),
                'ollama_rss_mb_peak': round(max(rss)), 'system_used_mb_peak': round(max(used))}

def run_generation(cfg, deps, questions, conditions, models, runs, out_path, sampler_factory=ResourceSampler, log=print,
                   max_errors=10):
    """Una línea JSONL por (condición, modelo, corrida, pregunta). Devuelve cuántas líneas escribió.

    Se detiene tras `max_errors` errores seguidos (p. ej. Ollama caído): al relanzar se reintentan y se sigue desde ahí."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {r['key'] for r in read_jsonl(out_path) if not r.get('error')}
    budget, seed, digests = deps.budget(), cfg.get('seed', 42), deps.digests()
    cache, prompts, written, consecutive = {}, {}, 0, 0
    total = runs * len(conditions) * len(models) * len(questions)
    with out_path.open('a', encoding='utf-8') as f:
        for run in range(1, runs + 1):
            for cname in conditions:
                cond = {**cfg['base'], **cfg['conditions'][cname]}
                prompt = prompts.setdefault(cond['prompt'], deps.load_prompt(cond['prompt']))
                for model in models:
                    for q in questions:
                        key = f"{cname}|{model}|{run}|{q['id']}"
                        if key in done: continue
                        rec = {'key': key, 'condition': cname, 'model': model, 'model_digest': digests.get(model), 'run': run,
                               'qid': q['id'], 'split': q['split'], 'tipo': q['tipo'], 'debe_abstenerse': q['debe_abstenerse'],
                               'k': cond['k'], 'prompt_id': cond['prompt'], 'count': cond.get('count'), 'error': None}
                        try:
                            ckey = (cond['collection'], cond['k'], q['id'])
                            rec['retrieval_cached'] = ckey in cache
                            if ckey not in cache:
                                collection, filters = deps.plan(q, cond['collection'])
                                cache[ckey] = {**deps.retrieve(q['pregunta'], cond['k'], filters, collection), 'filters': filters}
                            ret = cache[ckey]
                            rec.update(collection=ret['collection'], filters=ret['filters'],
                                       retrieved_doc_ids=rank_docs(ret['chunks']))
                            with sampler_factory() as sampler:
                                res = deps.generate(q['pregunta'], ret['chunks'], model, prompt, budget, 0.0, seed, cond.get('count'))
                            rec.update(res)
                            rec['context_doc_ids'] = rank_docs(ret['chunks'][:res['context']['n_context']])
                            rec['timings_ms'] = {**ret['timings_ms'], **res['timings_ms']}
                            rec['resources'] = sampler.summary()
                            try: rec['model_memory'] = deps.loaded(model)
                            except Exception: rec['model_memory'] = None
                        except Exception as e:
                            rec['error'] = f'{type(e).__name__}: {e}'
                        f.write(json.dumps(rec, ensure_ascii=False, default=str) + '\n')
                        f.flush()
                        written += 1
                        consecutive = consecutive + 1 if rec['error'] else 0
                        if consecutive >= max_errors:
                            raise RuntimeError(f'{max_errors} errores seguidos; último en {key}: {rec["error"]}. '
                                               'Revisa Ollama y Milvus y relanza: se retoma desde aquí.')
                        if written % 25 == 0: log(f'{written} líneas escritas (de hasta {total}); última: {key}')
    return written

def run_retrieval(cfg, deps, questions, runs, out_dir, log=print):
    """Métricas por pregunta (A y B), latencias de cada corrida y rankings con solo ids."""
    out_dir = Path(out_dir)
    (out_dir / 'tables').mkdir(parents=True, exist_ok=True)
    (out_dir / 'runs').mkdir(parents=True, exist_ok=True)
    answerable = [q for q in questions if deps.answerable(q)]
    k = cfg['retrieval']['k']
    tables, latency, rankings = [], [], []
    for label, name in cfg['retrieval']['collections'].items():
        ranked_by_id = {}
        for run in range(1, runs + 1):
            for q in answerable:
                r = deps.retrieve(q['pregunta'], k, filters_of(q), name)
                latency.append({'indice': label, 'run': run, 'id': q['id'], **{f'{key}_ms': v for key, v in r['timings_ms'].items()}})
                if run == 1:
                    ranked_by_id[q['id']] = rank_docs(r['chunks'], skip_synthetic=True)
                    rankings.append({'indice': label, 'id': q['id'], 'ranking': ranked_by_id[q['id']]})
        extra = {q['id']: {'indice': label, 'split': q['split']} for q in answerable}
        tables.append(deps.evaluate(ranked_by_id, answerable, KS, extra))
        log(f'{label}: {len(ranked_by_id)} preguntas respondibles')
    per_question = pd.concat(tables, ignore_index=True)
    per_question.to_csv(out_dir / 'tables' / 'recuperacion_por_pregunta.csv', index=False)
    pd.DataFrame(latency).to_csv(out_dir / 'tables' / 'recuperacion_latencia.csv', index=False)
    with (out_dir / 'runs' / 'recuperacion_rankings.jsonl').open('w', encoding='utf-8') as f:
        for row in rankings: f.write(json.dumps(row, ensure_ascii=False) + '\n')
    return per_question

def environment(config_path, gold_path, deps, extra=None):
    """Hardware, versiones y huellas de la ejecución (sin texto del corpus)."""
    info = {'created_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'platform': platform.platform(),
            'machine': platform.machine(), 'python': sys.version.split()[0], 'cpu_count': os.cpu_count(),
            'gold_sha256': sha256_file(gold_path), 'config_sha256': sha256_file(config_path)}
    try:
        import psutil
        info['memory_gb'] = round(psutil.virtual_memory().total / 2 ** 30, 1)
    except ImportError: pass
    try:
        import torch
        info['torch'] = torch.__version__
        info['torch_mps'] = bool(torch.backends.mps.is_available())
    except ImportError: pass
    try:
        info['git_commit'] = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError): pass
    info.update(deps.versions())
    info.update(extra or {})
    return info

class RealDeps:
    """Conecta el evaluador con el RAG real (Milvus y Ollama). Los simulacros de los tests tienen la misma interfaz."""
    def __init__(self, milvus_config):
        from app import core, llm, rag
        from pipeline import metrics
        self.core, self.llm, self.rag, self.metrics = core, llm, rag, metrics
        self.adversarial = yaml.safe_load(Path(milvus_config).read_text(encoding='utf-8'))['adversarial']
        self.client = core.client()

    def retrieve(self, query, k, filters, collection):
        return self.core.retrieve(query, k, filters, collection, self.client)

    def plan(self, question, default_collection):
        """Las preguntas adversariales del gold van a su colección aparte; las demás, a la de la condición."""
        collection = self.adversarial['name'] if question['id'] in self.adversarial['questions'] else default_collection
        return collection, filters_of(question)

    def generate(self, question, chunks, model, prompt, budget, temperature, seed, count):
        return self.rag.generate(question, chunks, model, prompt, budget, temperature, seed, count)

    def load_prompt(self, prompt_id): return self.rag.load_prompt(prompt_id)
    def budget(self): return self.rag.budget_settings()
    def answerable(self, q): return self.metrics.answerable(q)
    def evaluate(self, rankings, questions, ks, extra): return self.metrics.evaluate(rankings, questions, ks, extra)
    def digests(self): return {m['name']: m['digest'] for m in self.llm.models()}

    def loaded(self, model):
        """Memoria que Ollama reserva para el modelo cargado (en Apple Silicon la memoria es unificada)."""
        for m in self.llm._get('/api/ps').get('models', []):
            if model in (m.get('name'), m.get('model')):
                return {'size_mb': round(m.get('size', 0) / 2 ** 20), 'vram_mb': round(m.get('size_vram', 0) / 2 ** 20)}
        return None

    def versions(self):
        return {'ollama': self.llm.version(), 'models': {m['name']: {'digest': m['digest'][:12], 'quantization': m['quantization'],
                                                                      'parameter_size': m['parameter_size']} for m in self.llm.models()}}

    def preflight(self, models, collections):
        missing = [m for m in models if m not in self.digests()]
        if missing: raise SystemExit(f'Faltan modelos en Ollama: {missing}. Descárgalos con `ollama pull`.')
        absent = [c for c in collections if not self.client.has_collection(c)]
        if absent: raise SystemExit(f'Faltan colecciones en Milvus: {absent}')
        self.retrieve('calentamiento del encoder', 1, {}, collections[0])

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('modo', choices=['retrieval', 'generation'])
    ap.add_argument('--config', default='configs/experimentos.yaml')
    ap.add_argument('--milvus-config', default='configs/milvus.yaml')
    ap.add_argument('--conditions', nargs='*')
    ap.add_argument('--models', nargs='*')
    ap.add_argument('--splits', nargs='*')
    ap.add_argument('--runs', type=int)
    ap.add_argument('--limit', type=int, help='solo las primeras N preguntas (pruebas rápidas)')
    ap.add_argument('--out', help='JSONL de salida de la generación')
    ap.add_argument('--dry-run', action='store_true', help='muestra el plan sin ejecutar')
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    questions = load_questions(cfg['gold'], args.splits or cfg['splits'], args.limit)
    runs = args.runs or cfg['runs']
    reports = Path(cfg['reports_dir'])
    if args.modo == 'generation':
        conditions, models = args.conditions or list(cfg['conditions']), args.models or cfg['models']
        unknown = [c for c in conditions if c not in cfg['conditions']]
        if unknown: ap.error(f'condiciones desconocidas: {unknown}; hay {list(cfg["conditions"])}')
        out = Path(args.out or Path(cfg['out_dir']) / 'generacion.jsonl')
        total = runs * len(conditions) * len(models) * len(questions)
        print(f'{len(questions)} preguntas × {len(models)} modelos × {len(conditions)} condiciones × {runs} corridas = {total} generaciones -> {out}')
        if args.dry_run: return
        deps = RealDeps(args.milvus_config)
        collections = sorted({{**cfg['base'], **cfg['conditions'][c]}['collection'] for c in conditions})
        deps.preflight(models, collections)
        env = environment(args.config, cfg['gold'], deps, {'modo': 'generation', 'conditions': conditions, 'runs': runs, 'out': str(out)})
        (reports / 'runs').mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        (reports / 'runs' / f'entorno_{stamp}.json').write_text(json.dumps(env, indent=2, ensure_ascii=False), encoding='utf-8')
        print('escritas', run_generation(cfg, deps, questions, conditions, models, runs, out), 'líneas')
    else:
        print(f'{len(questions)} preguntas, {runs} corridas por índice')
        if args.dry_run: return
        deps = RealDeps(args.milvus_config)
        deps.preflight([], list(cfg['retrieval']['collections'].values()))
        table = run_retrieval(cfg, deps, questions, runs, reports)
        print(table.groupby('indice')[['recall@5', 'recall@10', 'mrr', 'ndcg@10', 'recall@5_g2', 'mrr_g2']].mean().round(3))

if __name__ == '__main__':
    main()
