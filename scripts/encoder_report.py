"""Análisis del encoder (Fase C) con las preguntas de dev del gold: variantes de consulta, Matryoshka, idiomas y vecinos.

Lee los chunks y sus vectores (scripts.embed_chunks) y escribe reports/embeddings/. No toca el split test.
"""
import argparse, json, sys
from pathlib import Path
import pandas as pd
import yaml
from app.embeddings import embed_array
from pipeline.encoder_eval import run

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--encoder',default='qwen3-0.6b')
    p.add_argument('--strategy',default='B',choices=['A','B'])
    p.add_argument('--sample',default='sample_10000')
    p.add_argument('--config',default='configs/encoders.yaml')
    p.add_argument('--input',default='data/documentos_rag.parquet')
    p.add_argument('--gold',default='data/gold/gold.jsonl')
    p.add_argument('--paraphrases',default='data/gold/parafrasis_dev.jsonl')
    p.add_argument('--features',default='data/eda/document_features.parquet')
    p.add_argument('--out',default='reports/embeddings')
    args=p.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    profile=cfg['profiles'][args.encoder]
    languages=pd.read_parquet(args.features,columns=['document_id','language_detected']).set_index('document_id')['language_detected'].to_dict()
    info,summary=run(Path('data/chunks')/args.strategy/f'{args.sample}.parquet',
                     Path('data/embeddings')/args.encoder/args.strategy/f'{args.sample}.parquet',
                     args.gold,args.paraphrases,args.input,profile,
                     lambda texts,prefix: embed_array(texts,'query',{**profile,'query_prefix':prefix}),
                     args.out,cfg['analysis'],languages,log=lambda m: print(f'· {m}',file=sys.stderr,flush=True))
    print(json.dumps({'questions':info['gold']['questions'],'variants':summary[['variant','recall@10','ndcg@10','mrr']].round(3).to_dict(orient='records'),
                      'outputs':args.out},ensure_ascii=False))
if __name__=='__main__': main()
