"""Perfilado del corpus (Fase A): regenera reports/eda/ y las muestras de data/samples/ sin modificar el Parquet."""
import argparse, json, sys
import yaml
from app.embeddings import count_tokens, max_tokens, settings
from pipeline.corpus import catalog_entry
from pipeline.eda import run
from pipeline.profiling import detect_languages

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',default='data/documentos_rag.parquet')
    p.add_argument('--config',default='configs/eda.yaml')
    p.add_argument('--corpus',default='dragons_rag',help='nombre en el catálogo de open_corpus_co_es')
    p.add_argument('--out',default='reports/eda')
    p.add_argument('--samples-dir',default='data/samples')
    p.add_argument('--work-dir',default='data/eda',help='artefactos intermedios por documento (no se versionan)')
    args=p.parse_args()
    with open(args.config,encoding='utf-8') as f: cfg=yaml.safe_load(f)
    profile=run(args.input,cfg,args.out,args.samples_dir,args.work_dir,count_tokens=count_tokens,max_tokens=max_tokens(),
                detect_languages=detect_languages,catalog=catalog_entry(args.corpus),encoder_settings=settings(),
                log=lambda m: print(f'· {m}',file=sys.stderr,flush=True))
    summary={k:profile[k] for k in ('integrity','distributions','duplicates','bias')}
    summary['samples']={k:v['size'] for k,v in profile['samples'].items()}
    summary['duration_seconds']=profile['run']['duration_seconds']
    summary['outputs']=args.out
    print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__': main()
