"""Carga una colección de Milvus desde los Parquet de chunks (Fase B) y de vectores (Fase C).

Repetir la carga no duplica (upsert por chunk_id) y, si se interrumpe, retoma desde el último lote hecho.
La colección adversarial une los documentos sintéticos del gold (Q45-Q48) con los reales de esas preguntas.
"""
import argparse, json, os, platform, sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import pandas as pd
import yaml
from pymilvus import MilvusClient
from app.embeddings import count_tokens, embed_array, settings, token_offsets
from pipeline import chunking, embedding, indexing, report

def main():
    cfg=yaml.safe_load(Path('configs/milvus.yaml').read_text(encoding='utf-8'))
    keys=[*cfg['collections'],cfg['adversarial']['key']]
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection',required=True,choices=keys,help='clave de configs/milvus.yaml')
    p.add_argument('--fresh',action='store_true',help='borra la colección y su progreso antes de cargar')
    p.add_argument('--chunks-dir',default='data/chunks')
    p.add_argument('--embeddings-dir',default='data/embeddings')
    p.add_argument('--reports',default='reports/milvus')
    args=p.parse_args()
    profiles=yaml.safe_load(Path(cfg['encoders_config']).read_text(encoding='utf-8'))['profiles']
    encoder,chunker=profiles[cfg['encoder']],profiles[cfg['chunker']]
    log=lambda m: print(f'· {m}',file=sys.stderr,flush=True)
    vectors_path=lambda strategy,sample: Path(args.embeddings_dir)/cfg['encoder']/strategy/f'{sample}.parquet'
    chunks_path=lambda strategy,sample: Path(args.chunks_dir)/strategy/f'{sample}.parquet'
    if args.collection==cfg['adversarial']['key']:
        adv=cfg['adversarial']
        name,base_c,base_v=adv['name'],chunks_path(adv['strategy'],adv['base_sample']),vectors_path(adv['strategy'],adv['base_sample'])
        base=pd.read_parquet(base_c)
        ids,matrix,_=embedding.read_vectors(base_v)
        if ids!=base['chunk_id'].tolist(): raise SystemExit('Vectores y chunks de la muestra base no están alineados.')
        gold=[json.loads(l) for l in Path(adv['gold']).read_text(encoding='utf-8').splitlines() if l.strip()]
        questions=[q for q in gold if q['id'] in adv['questions']]
        docs=[json.loads(l) for l in Path(adv['synthetic']).read_text(encoding='utf-8').splitlines() if l.strip()]
        ch=yaml.safe_load(Path('configs/chunking.yaml').read_text(encoding='utf-8'))
        params,sentences=ch['strategies'][adv['strategy']],ch['sentences']
        def chunk_fn(doc):
            splitter=chunking.punkt_splitter(sentences['languages'].get(doc.get('language_base'),sentences['default']))
            spans,_=chunking.chunk_document(doc['text'],token_offsets([doc['text']],chunker)[0],adv['strategy'],params,splitter)
            return indexing.chunk_frame(doc,spans,adv['strategy'],params['version'],lambda t: count_tokens(t,'passage',chunker))
        chunks,vectors,missing=indexing.adversarial_chunks(questions,docs,base,matrix,chunk_fn,lambda t: embed_array(t,'passage',encoder))
        if missing: raise SystemExit(f'Documentos de las preguntas sin chunks en la muestra base: {missing}')
        inputs=[base_c,base_v,Path(adv['synthetic']),Path(adv['gold'])]
        meta={'strategy':adv['strategy'],'sample':adv['base_sample'],'synthetic_chunks':int((chunks['collection_name']=='synthetic').sum())}
    else:
        spec=cfg['collections'][args.collection]
        name,cp,vp=spec['name'],chunks_path(spec['strategy'],spec['sample']),vectors_path(spec['strategy'],spec['sample'])
        chunks=pd.read_parquet(cp)
        ids,vectors,_=embedding.read_vectors(vp)
        if ids!=chunks['chunk_id'].tolist(): raise SystemExit(f'{vp} no está alineado con {cp}: vuelve a ejecutar embed_chunks.')
        inputs=[cp,vp]
        meta={'strategy':spec['strategy'],'sample':spec['sample']}
    client=MilvusClient(uri=os.getenv('MILVUS_URI','http://milvus:19530'))
    info=indexing.ingest(client,name,chunks,vectors,cfg,Path(cfg['ingest']['checkpoint_dir'])/f'{name}.json',
                         indexing.fingerprint(*inputs),args.fresh,log)
    run=report.to_jsonable({'created_at_utc':datetime.now(timezone.utc).isoformat(timespec='seconds'),'key':args.collection,**meta,**info,
                            'inputs':[str(x) for x in inputs],'encoder':settings(encoder),'index':cfg['index'],'ingest':cfg['ingest'],
                            'versions':{'python':platform.python_version(),'pymilvus':version('pymilvus')}})
    report.write_json(run,Path(args.reports)/'runs'/f'ingest_{args.collection}.json')
    print(json.dumps({k:run[k] for k in ('collection','entities','chunks','dimension','seconds','batches','retries','created')},ensure_ascii=False))
if __name__=='__main__': main()
