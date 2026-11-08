"""Segmenta una muestra del corpus con la estrategia A o B (Fase B) y actualiza la comparación en reports/chunking/."""
import argparse, json, sys
from pathlib import Path
import pandas as pd
import yaml
from app.embeddings import count_tokens, max_tokens, settings, token_offsets
from pipeline.segmentation import run

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--strategy',required=True,choices=['A','B'])
    p.add_argument('--sample',default='data/samples/sample_2000_ids.txt',help="lista de IDs de data/samples/ o 'full'")
    p.add_argument('--input',default='data/documentos_rag.parquet')
    p.add_argument('--config',default='configs/chunking.yaml')
    p.add_argument('--out',default='data/chunks',help='Parquet de chunks (no se versiona)')
    p.add_argument('--reports',default='reports/chunking')
    args=p.parse_args()
    with open(args.config,encoding='utf-8') as f: cfg=yaml.safe_load(f)
    required=[cfg['features']]+([cfg['boilerplate']['lines']] if cfg['boilerplate']['exclude'] else [])
    missing=[path for path in required if not Path(path).is_file()]
    if missing: raise SystemExit(f'Faltan {missing}: ejecuta antes scripts.profile_corpus (Fase A).')
    features=pd.read_parquet(cfg['features'],columns=['document_id','language_detected','has_mojibake'])
    lines=pd.read_parquet(cfg['boilerplate']['lines']) if cfg['boilerplate']['exclude'] else None
    with open(cfg['encoders_config'],encoding='utf-8') as f: encoder=yaml.safe_load(f)['profiles'][cfg['encoder_profile']]
    info=run(args.input,args.sample,args.strategy,cfg,args.out,args.reports,token_offsets=lambda t: token_offsets(t,encoder),
             count_tokens=lambda t: count_tokens(t,'passage',encoder),max_tokens=max_tokens(encoder),features=features,
             boilerplate_lines=lines,encoder_settings=settings(encoder),
             log=lambda m: print(f'· {m}',file=sys.stderr,flush=True))
    total=next(s for s in info['stats'] if s['collection_name']=='total')
    print(json.dumps({'strategy':info['strategy'],'sample':info['sample'],'documents':total['documents'],
                      'excluded_template_documents':info['excluded_template_documents'],'chunks':total['chunks'],
                      'chunks_written':info['chunks_written'],'n_tokens_max':total['n_tokens_max'],
                      'mid_sentence_share':total['mid_sentence_share'],'removed_share':total['removed_share'],
                      'seconds':info['seconds'],'output':info['output']},ensure_ascii=False))
if __name__=='__main__': main()
