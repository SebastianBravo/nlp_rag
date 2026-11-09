"""Revisa el RAG con las preguntas de dev (Fase E) y escribe reports/rag/.

Con --calibrar mide, en cada LLM, cuántos tokens cuenta Ollama por cada token del encoder en los prompts reales
(sin generar) y propone LLM_TOKEN_FACTOR. Sin esa opción responde las preguntas de dev con los modelos dados y
resume abstención, citas, tokens y latencias. Las respuestas completas van a data/logs/check_rag/ (no se versionan).
"""
import argparse, json, os, platform, sys, uuid
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import pandas as pd
import yaml
from app import core, llm, rag
from pipeline import io, rag_check, report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',nargs='+',default=[os.getenv('LLM_MODEL','qwen2.5:1.5b')])
    p.add_argument('--prompts',nargs='+',default=[os.getenv('PROMPT_ID','rag_v1')],help='ids de prompts/; la calibración usa el primero')
    p.add_argument('--calibrar',action='store_true',help='mide tokens del LLM frente a tokens del encoder y propone el factor')
    p.add_argument('--k',type=int,default=5,help='fragmentos por pregunta en la revisión')
    p.add_argument('--calibration-k',type=int,default=10,help='fragmentos por prompt en la calibración')
    p.add_argument('--margin',type=float,default=0.05,help='margen sobre el mayor cociente medido')
    p.add_argument('--gold',default='data/gold/gold.jsonl')
    p.add_argument('--milvus-config',default='configs/milvus.yaml')
    p.add_argument('--out',default='reports/rag')
    p.add_argument('--logs',default='data/logs/check_rag')
    args=p.parse_args()
    log=lambda m: print(f'· {m}',file=sys.stderr,flush=True)
    adversarial=yaml.safe_load(Path(args.milvus_config).read_text(encoding='utf-8'))['adversarial']
    questions=rag_check.load_questions(args.gold)
    prompts=[rag.load_prompt(pid) for pid in args.prompts]
    prompt=prompts[0]
    digests={m['name']:m['digest'] for m in llm.models()}
    missing=[m for m in args.models if m not in digests]
    if missing: raise SystemExit(f'Modelos sin descargar en Ollama: {missing}')
    meta={'created_at_utc':datetime.now(timezone.utc).isoformat(timespec='seconds'),'split':'dev','questions':len(questions),
          'gold_sha256':io.sha256_file(args.gold),'prompts':{x['id']:{k:x[k] for k in ('version','sha256')} for x in prompts},
          'models':{m:digests[m] for m in args.models},'collections':{'default':core.default_collection(),'adversarial':adversarial['name']},
          'versions':{'python':platform.python_version(),'ollama':llm.version(),'pymilvus':version('pymilvus')}}
    out=Path(args.out)

    if args.calibrar:
        # Un margen amplio para que ningún fragmento quede fuera: aquí se mide, no se recorta.
        wide={'num_ctx':8192,'num_predict':1,'token_factor':1.0}
        contexts=[]
        for q in questions:
            collection,filters=rag_check.plan(q,core.default_collection(),adversarial)
            ret=core.retrieve(q['pregunta'],args.calibration_k,filters,collection)
            ctx=rag.build_context(q['pregunta'],ret['chunks'],prompt,wide)
            if ctx['n_context']: contexts.append((q['id'],ctx))
        rows=[]
        for model in args.models:
            log(f'Calibrando {model} con {len(contexts)} prompts')
            for qid,ctx in contexts:
                res=llm.chat(model,ctx['system'],ctx['user'],rag.llm_options(wide))
                rows.append({'model':model,'id':qid,'chunks':ctx['n_context'],'encoder_tokens':ctx['proxy_tokens'],
                             'llm_tokens':res['prompt_tokens'],'ratio':res['prompt_tokens']/ctx['proxy_tokens']})
        table=rag_check.calibration_table(rows)
        factor=rag_check.token_factor(table,args.margin)
        report.write_tables({'calibracion_tokens':table},out/'tables')
        report.write_json({**meta,'k':args.calibration_k,'margin':args.margin,'token_factor':factor,'num_ctx':wide['num_ctx'],
                           'rows':rows},out/'runs'/'calibracion_tokens.json')
        print(json.dumps({'token_factor':factor,'ratio_max':table.set_index('model')['ratio_max'].round(3).to_dict()},ensure_ascii=False))
        return

    budget=rag.budget_settings()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    raw_path=Path(args.logs)/f'dev_{stamp}.jsonl'
    raw_path.parent.mkdir(parents=True,exist_ok=True)
    rows=[]
    with raw_path.open('w',encoding='utf-8') as raw:
        for pid in args.prompts:
            for model in args.models:
                log(f'{pid} · {model}: {len(questions)} preguntas de dev')
                for q in questions:
                    collection,filters=rag_check.plan(q,core.default_collection(),adversarial)
                    result=rag.answer(q['pregunta'],model,args.k,filters,collection,prompt_id=pid)
                    request={'id':q['id'],'query':q['pregunta'],'model':model,'k':args.k,'filters':filters,'collection':collection}
                    raw.write(json.dumps(rag.run_record(uuid.uuid4().hex,request,result,model_digest=digests[model]),ensure_ascii=False)+'\n')
                    rows.append(rag_check.question_row(q,result,model))
    per_question=pd.DataFrame(rows)
    summary=rag_check.summary(per_question)
    report.write_tables({'dev_preguntas':per_question,'dev_resumen':summary},out/'tables')
    report.write_json({**meta,'k':args.k,'budget':budget,'seed':rag.default_seed(),'temperature':0.0},out/'runs'/'dev.json')
    totals=summary[summary['tipo']=='total'].set_index(['prompt','model'])
    cols=['n','abstencion_correcta','cita_relevante','sin_citas','ms_total_p50']
    print(json.dumps(report.to_jsonable({f'{pid} · {m}':v for (pid,m),v in totals[cols].round(3).to_dict(orient='index').items()}),
                     ensure_ascii=False))

if __name__=='__main__': main()
