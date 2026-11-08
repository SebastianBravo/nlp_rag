"""Vectoriza los chunks de una estrategia y una muestra con un perfil de encoder (Fase C).

Escribe data/embeddings/<perfil>/<estrategia>/<muestra>.parquet (no se versiona) y los metadatos de la
ejecución en reports/embeddings/runs/. Si se interrumpe, al relanzarlo reutiliza las partes ya calculadas.
"""
import argparse, json, platform, resource, sys, time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from app.embeddings import count_tokens, embed_array, max_tokens, settings
from pipeline import embedding, io, report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--encoder',default='qwen3-0.6b',help='perfil de configs/encoders.yaml')
    p.add_argument('--strategy',default='B',choices=['A','B'])
    p.add_argument('--sample',default='sample_10000')
    p.add_argument('--config',default='configs/encoders.yaml')
    p.add_argument('--chunks-dir',default='data/chunks')
    p.add_argument('--out',default='data/embeddings')
    p.add_argument('--reports',default='reports/embeddings')
    p.add_argument('--part-size',type=int,default=1024)
    args=p.parse_args()
    profile=yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))['profiles'][args.encoder]
    chunks_path=Path(args.chunks_dir)/args.strategy/f'{args.sample}.parquet'
    out=Path(args.out)/args.encoder/args.strategy/f'{args.sample}.parquet'
    chunks=pd.read_parquet(chunks_path,columns=['chunk_id','text'])
    log=lambda m: print(f'· {m}',file=sys.stderr,flush=True)
    log(f'{len(chunks)} chunks de {chunks_path} con {profile["model"]}')
    started=time.perf_counter()
    computed=embedding.embed_in_parts(chunks['chunk_id'].tolist(),chunks['text'].tolist(),
                                      lambda t: embed_array(t,'passage',profile,profile.get('batch_size',32)),
                                      lambda t: count_tokens(t,'passage',profile),out,args.part_size,log)
    seconds=time.perf_counter()-started
    ids,matrix,n_tokens=embedding.read_vectors(out)
    limit=max_tokens(profile)
    if (n_tokens>limit).any(): raise SystemExit(f'{int((n_tokens>limit).sum())} chunks superan {limit} tokens del encoder.')
    norms=np.linalg.norm(matrix,axis=1)
    parts=-(-len(chunks)//args.part_size)
    info=report.to_jsonable({
        'created_at_utc':datetime.now(timezone.utc).isoformat(timespec='seconds'),'encoder':args.encoder,
        'settings':settings(profile),'strategy':args.strategy,'sample':args.sample,
        'chunks':{'path':str(chunks_path),'sha256':io.sha256_file(chunks_path),'n':len(chunks)},
        'output':str(out),'dimension':int(matrix.shape[1]),'max_tokens':limit,
        'n_tokens':{'p50':float(np.median(n_tokens)),'p95':float(np.percentile(n_tokens,95)),'max':int(n_tokens.max())},
        'norm':{'mean':float(norms.mean()),'min':float(norms.min()),'max':float(norms.max())},
        'parts':{'total':parts,'computed_now':computed},
        'seconds':round(seconds,1),'chunks_per_second':round(len(chunks)/seconds,2) if computed==parts else None,
        'peak_rss_mb':round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024),
        'versions':{'python':platform.python_version(),**{n:version(n) for n in ('torch','transformers','sentence-transformers')}}})
    report.write_json(info,Path(args.reports)/'runs'/f'{args.encoder}_{args.strategy}_{args.sample}.json')
    print(json.dumps({k:info[k] for k in ('encoder','strategy','sample','dimension','n_tokens','norm','seconds','chunks_per_second','peak_rss_mb','output')},ensure_ascii=False))
if __name__=='__main__': main()
