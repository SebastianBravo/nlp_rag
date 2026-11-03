"""Descarga el corpus del taller y lo deja en data/documentos_rag.parquet (no sobrescribe salvo con --force).

También descarga su manifiesto de procedencia a data/documentos_rag.manifest.json.
"""
import argparse, json
from pipeline.corpus import fetch_corpus, fetch_manifest
from pipeline.io import manifest_path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus',default='dragons_rag')
    p.add_argument('--output',default='data/documentos_rag.parquet')
    p.add_argument('--force',action='store_true',help='vuelve a descargar y reemplaza --output y el manifiesto')
    args=p.parse_args()
    summary=fetch_corpus(args.corpus,args.output,force=args.force)
    try: summary['manifest']=fetch_manifest(args.corpus,manifest_path(args.output),force=args.force)
    except Exception as e: summary['manifest']={'status':'error','detail':str(e)}
    print(json.dumps(summary,ensure_ascii=False))
    if not summary['sha256_matches_catalog']: raise SystemExit('El archivo no coincide con el catálogo; usa --force para reemplazarlo.')
if __name__=='__main__': main()
