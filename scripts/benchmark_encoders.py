"""Mide el costo en CPU de cada perfil de encoder dentro del contenedor (Fase C): chunks/s y RAM pico.

Cada perfil corre en un proceso propio, con los mismos chunks y el mismo tamaño de lote. Escribe
reports/embeddings/tables/costo_encoders.csv y reports/embeddings/runs/costo_encoders.json.
"""
import argparse, json, platform, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import yaml

def measure(cfg, name):
    import resource
    from app.embeddings import embed_array, encoder
    from pipeline.embedding import measure_cost
    bench, profile = cfg['benchmark'], cfg['profiles'][name]
    texts = pd.read_parquet(bench['chunks'], columns=['text'])['text'].sample(bench['n'], random_state=bench['seed']).tolist()
    before = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    model = encoder(profile)
    row = measure_cost(texts, lambda t: embed_array(t, 'passage', profile, bench['batch_size']))
    params = sum(p.numel() for p in model.parameters())
    return {'profile': name, 'model': profile['model'], 'revision': profile['revision'], 'parameters_m': round(params / 1e6, 1),
            'max_seq_length': model.max_seq_length, 'rss_before_load_mb': before, **row}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/encoders.yaml')
    p.add_argument('--out',default='reports/embeddings')
    p.add_argument('--only',help='uso interno: mide un solo perfil en este proceso')
    args=p.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    if args.only:
        print(json.dumps(measure(cfg,args.only))); return
    rows=[]
    for name in cfg['benchmark']['profiles']:
        print(f'· {name}',file=sys.stderr,flush=True)
        out=subprocess.run([sys.executable,'-m','scripts.benchmark_encoders','--config',args.config,'--only',name],
                           check=True,capture_output=True,text=True).stdout
        rows.append(json.loads(out.strip().splitlines()[-1]))
    table=pd.DataFrame(rows)
    table['hours_per_100k_chunks']=(100_000/table['chunks_per_second']/3600).round(2)
    table['mb_per_100k_vectors']=(table['dimension']*4*100_000/1e6).round(1)
    out_dir=Path(args.out)
    (out_dir/'tables').mkdir(parents=True,exist_ok=True)
    table.to_csv(out_dir/'tables'/'costo_encoders.csv',index=False)
    (out_dir/'runs').mkdir(parents=True,exist_ok=True)
    info={'created_at_utc':datetime.now(timezone.utc).isoformat(timespec='seconds'),'benchmark':cfg['benchmark'],
          'python':platform.python_version(),'results':rows}
    (out_dir/'runs'/'costo_encoders.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(table[['profile','parameters_m','dimension','chunks_per_second','peak_rss_mb','hours_per_100k_chunks']].to_json(orient='records',force_ascii=False))
if __name__=='__main__': main()
