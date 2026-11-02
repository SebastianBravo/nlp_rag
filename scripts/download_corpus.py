"""Descarga el corpus del taller y lo deja en data/documentos_rag.parquet (no sobrescribe salvo con --force)."""
import argparse, json
from pipeline.corpus import fetch_corpus

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus',default='dragons_rag')
    p.add_argument('--output',default='data/documentos_rag.parquet')
    p.add_argument('--force',action='store_true',help='vuelve a descargar y reemplaza --output')
    args=p.parse_args()
    summary=fetch_corpus(args.corpus,args.output,force=args.force)
    print(json.dumps(summary,ensure_ascii=False))
    if not summary['sha256_matches_catalog']: raise SystemExit('El archivo no coincide con el catálogo; usa --force para reemplazarlo.')
if __name__=='__main__': main()
