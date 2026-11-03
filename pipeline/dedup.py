"""Deduplicación exacta de chunks con procedencia trazable (Fase B3)."""
import hashlib
import pandas as pd

def text_sha256(texts):
    return [hashlib.sha256(t.encode('utf-8')).hexdigest() for t in texts]

def deduplicate(chunks):
    """Conserva un chunk canónico por `chunk_sha256`: el de menor (`document_id`, `char_start`).

    `provenance_ids` lista, ordenados y sin repetir, todos los documentos que contienen ese texto.
    Devuelve los chunks conservados y los eliminados.
    """
    ordered = chunks.sort_values(['chunk_sha256', 'document_id', 'char_start'], kind='mergesort')
    provenance = ordered.groupby('chunk_sha256', sort=False)['document_id'].agg(lambda ids: sorted(set(ids)))
    first = ~ordered['chunk_sha256'].duplicated(keep='first')
    kept = ordered[first].copy()
    kept['provenance_ids'] = kept['chunk_sha256'].map(provenance)
    kept = kept.sort_values(['document_id', 'char_start'], kind='mergesort').reset_index(drop=True)
    return kept, ordered[~first].reset_index(drop=True)

def dedup_summary(chunks, removed, by='collection_name'):
    """Chunks antes y después de deduplicar, por colección."""
    rows = []
    for group in [*sorted(chunks[by].unique()), 'total']:
        before = chunks if group == 'total' else chunks[chunks[by] == group]
        gone = removed if group == 'total' else removed[removed[by] == group]
        rows.append({by: group, 'chunks': len(before), 'removed': len(gone), 'kept': len(before) - len(gone),
                     'removed_share': len(gone) / len(before) if len(before) else 0.0})
    return pd.DataFrame(rows)
