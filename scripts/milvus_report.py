"""Verifica las colecciones Milvus (Fase D) y escribe reports/milvus/.

Sin opciones: conteos frente a los Parquet, calibración de ef frente a la búsqueda exacta, filtros y métricas
en dev (Milvus frente a búsqueda exacta). Con --snapshot ETIQUETA guarda conteos y resultados fijos; con
--compare A B compara dos snapshots (persistencia tras reiniciar Milvus).
"""
import argparse, json, os, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from pymilvus import MilvusClient
from app.embeddings import embed_array
from pipeline import embedding, encoder_eval, indexing, io, metrics, milvus_eval, report, retrieval

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/milvus.yaml')
    p.add_argument('--out',default='reports/milvus')
    p.add_argument('--snapshot')
    p.add_argument('--compare',nargs=2)
    p.add_argument('--input',default='data/documentos_rag.parquet')
    args=p.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    out=Path(args.out)
    log=lambda m: print(f'· {m}',file=sys.stderr,flush=True)
    if args.compare:
        a,b=(json.loads((out/'runs'/f'snapshot_{x}.json').read_text()) for x in args.compare)
        table=milvus_eval.compare_snapshots(a['collections'],b['collections'])
        report.write_tables({'persistencia':table},out/'tables')
        print(table.to_json(orient='records',force_ascii=False)); return
    profile=yaml.safe_load(Path(cfg['encoders_config']).read_text(encoding='utf-8'))['profiles'][cfg['encoder']]
    client=MilvusClient(uri=os.getenv('MILVUS_URI','http://milvus:19530'))
    names={k:v['name'] for k,v in cfg['collections'].items()}|{cfg['adversarial']['key']:cfg['adversarial']['name']}
    questions,paraphrases=encoder_eval.load_gold('data/gold/gold.jsonl','data/gold/parafrasis_dev.jsonl')
    originals=embed_array([q['pregunta'] for q in questions],'query',profile)
    ef=cfg['search']['ef']
    if args.snapshot:
        snap={'created_at_utc':datetime.now(timezone.utc).isoformat(timespec='seconds'),'ef':ef,
              'collections':milvus_eval.snapshot(client,names,originals[:5],ef)}
        report.write_json(snap,out/'runs'/f'snapshot_{args.snapshot}.json')
        print(json.dumps({k:v['entities'] for k,v in snap['collections'].items()})); return

    tables={}
    log('Conteos')
    rows=[]
    for key,spec in cfg['collections'].items():
        n=len(pd.read_parquet(Path('data/chunks')/spec['strategy']/f"{spec['sample']}.parquet",columns=['chunk_id']))
        rows.append({'key':key,'collection':spec['name'],'parquet_chunks':n})
    adv_run=out/'runs'/f"ingest_{cfg['adversarial']['key']}.json"
    if adv_run.is_file():
        rows.append({'key':cfg['adversarial']['key'],'collection':cfg['adversarial']['name'],'parquet_chunks':json.loads(adv_run.read_text())['chunks']})
    for r in rows:
        r['milvus_entities']=indexing.count(client,r['collection']) if client.has_collection(r['collection']) else None
        r['match']=r['milvus_entities']==r['parquet_chunks']
    tables['conteos']=pd.DataFrame(rows)

    log('Calibración de ef')
    s=cfg['search']; spec=cfg['collections'][s['calibration_collection']]
    chunks_path=Path('data/chunks')/spec['strategy']/f"{spec['sample']}.parquet"
    ids,matrix,_=embedding.read_vectors(Path('data/embeddings')/cfg['encoder']/spec['strategy']/f"{spec['sample']}.parquet")
    rng=np.random.default_rng(s['seed'])
    variants=[originals]+[embed_array([paraphrases[q['id']][v]['texto'] for q in questions],'query',profile) for v in ('parafrasis','traduccion')]
    queries=np.vstack([*variants,matrix[rng.choice(len(matrix),s['random_queries'],replace=False)]])
    calibration=milvus_eval.calibrate_ef(client,spec['name'],queries,matrix,ids,s['ef_candidates'])
    chosen=milvus_eval.choose_ef(calibration,s['target_recall'])
    tables['calibracion_ef']=calibration.assign(chosen=calibration['ef']==chosen,configured=calibration['ef']==ef)

    log('Filtros')
    windows={q['id']:q['ventana_fecha'] for q in questions if q.get('ventana_fecha')}
    gold_all=[json.loads(l) for l in Path('data/gold/gold.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    windows.update({q['id']:q['ventana_fecha'] for q in gold_all if q.get('ventana_fecha')})
    def in_window(w):
        lo,hi=indexing.window_bounds(w)
        return lambda e: e['created_at'] is not None and lo<=e['created_at']<hi
    sample_doc=questions[0]['relevantes'][0]['document_id']
    checks=[('idioma húngaro','language == "hu"',lambda e: e['language']=='hu'),
            ('solo Bluesky','collection_name == "bluesky_posts"',lambda e: e['collection_name']=='bluesky_posts'),
            ('prensa en español','collection_name == "url_crawls" and language == "es"',lambda e: e['collection_name']=='url_crawls' and e['language']=='es'),
            ('procedencia contiene un documento',f'array_contains(provenance_ids, "{sample_doc}")',lambda e: sample_doc in e['provenance_ids'])]
    checks+=[(f'ventana de {qid}',indexing.window_filter(w),in_window(w)) for qid,w in sorted(windows.items())]
    tables['filtros']=milvus_eval.filter_checks(client,cfg['collections']['sent256_s10k']['name'],originals[0],checks,ef)

    log('Métricas en dev: Milvus frente a búsqueda exacta')
    spans=encoder_eval.gold_spans(questions)
    rel_docs=sorted({d for v in spans.values() for d in v})
    texts=io.read_rows(args.input,['document_id','text'],rel_docs).set_index('document_id')['text'].to_dict()
    results=[]
    for key in ('sent256_s10k','win256_s10k'):
        spec=cfg['collections'][key]
        cp=Path('data/chunks')/spec['strategy']/f"{spec['sample']}.parquet"
        chunks=pd.read_parquet(cp,columns=encoder_eval.CHUNK_COLUMNS)
        _,mat,_=embedding.read_vectors(Path('data/embeddings')/cfg['encoder']/spec['strategy']/f"{spec['sample']}.parquet")
        docs=sorted({d for pv in chunks['provenance_ids'] for d in pv})
        doc_dates=io.read_rows(args.input,['document_id','created_at'],docs).set_index('document_id')['created_at']
        exact,_,_=encoder_eval.run_variant('exacta',originals,mat,chunks,questions,doc_dates,spans,texts)
        rk,extra=milvus_eval.milvus_rankings(client,spec['name'],questions,originals,ef,doc_dates,spans,texts)
        milvus=metrics.evaluate(rk,questions,(1,5,10,20),extra)
        for label,table in (('exacta',exact),('milvus',milvus)):
            results.append(metrics.summarize(table).assign(collection=spec['name'],strategy=spec['strategy'],search=label))
    tables['metricas_dev']=pd.concat(results,ignore_index=True)

    files=report.write_tables(tables,out/'tables')
    figure=report.ef_calibration_plot(tables['calibracion_ef'],out/'figures'/'calibracion_ef.png')
    info=report.to_jsonable({'created_at_utc':datetime.now(timezone.utc).isoformat(timespec='seconds'),'config':cfg,
                             'ef_configured':ef,'ef_chosen':chosen,'gold_sha256':io.sha256_file('data/gold/gold.jsonl'),
                             'dev_questions':len(questions),'calibration_queries':int(len(queries)),'outputs':{'tables':files,'figures':[figure]}})
    report.write_json(info,out/'verificacion.json')
    summary=tables['metricas_dev'][['collection','search','recall@10','mrr','ndcg@10']].round(3).to_dict(orient='records')
    print(json.dumps({'ef_configured':ef,'ef_chosen':chosen,'counts_ok':bool(tables['conteos']['match'].all()),
                      'filters_ok':bool(tables['filtros']['all_match'].all()),'dev':summary},ensure_ascii=False))
if __name__=='__main__': main()
