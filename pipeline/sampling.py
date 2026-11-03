"""Muestras estratificadas, anidadas y reproducibles del corpus (Fase A6).

Cada documento recibe una clave aleatoria fija (semilla) y, dentro de su estrato, se toman los de
clave más baja según la asignación proporcional. Como la clave es la misma para todos los tamaños,
cada muestra contiene a las más pequeñas.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from pipeline.profiling import group_rare

def stratum_labels(df, columns, min_share):
    """Etiqueta de estrato por documento, p. ej. 'url_crawls|es'. Los valores raros se agrupan en 'otro'."""
    labels = [group_rare(df[col], min_share) for col in columns]
    return pd.Series(['|'.join(parts) for parts in zip(*labels)], index=df.index, name='stratum')

def allocate(sizes, n):
    """Asignación proporcional por restos mayores; suma exactamente `n`. Empates: orden del estrato."""
    if n > sizes.sum(): raise ValueError(f'La muestra ({n}) supera el corpus ({sizes.sum()}).')
    quotas = sizes / sizes.sum() * n
    alloc = np.floor(quotas).astype(int)
    remainder = (quotas - alloc).sort_index()
    order = remainder.sort_values(ascending=False, kind='mergesort').index
    alloc[order[:n - alloc.sum()]] += 1
    return alloc

def nested_samples(df, sizes, columns, min_share, seed):
    """IDs de cada tamaño de muestra (ordenados) y la etiqueta de estrato de cada documento."""
    data = df[['document_id']].assign(stratum=stratum_labels(df, columns, min_share))
    data = data.sort_values('document_id', kind='mergesort').reset_index(drop=True)
    data['key'] = np.random.default_rng(seed).random(len(data))
    data['rank'] = data.groupby('stratum')['key'].rank(method='first') - 1
    strata = data['stratum'].value_counts().sort_index()
    samples, previous = {}, None
    for n in sorted(sizes):
        alloc = allocate(strata, n)
        if previous is not None and (alloc < previous).any():
            raise ValueError('La asignación no es monótona entre tamaños; las muestras no quedarían anidadas.')
        samples[n] = sorted(data.loc[data['rank'] < data['stratum'].map(alloc), 'document_id'])
        previous = alloc
    return samples, data.set_index('document_id')['stratum']

def distribution(strata, ids):
    """Documentos y proporción por estrato en el corpus y en la muestra."""
    corpus = strata.value_counts().rename('corpus_documents')
    sample = strata.loc[list(ids)].value_counts().rename('sample_documents')
    out = pd.concat([corpus, sample], axis=1).fillna(0).astype(int).rename_axis('stratum').reset_index()
    out['corpus_share'] = out['corpus_documents'] / out['corpus_documents'].sum()
    out['sample_share'] = out['sample_documents'] / max(out['sample_documents'].sum(), 1)
    out['diff'] = out['sample_share'] - out['corpus_share']
    return out.sort_values('corpus_documents', ascending=False).reset_index(drop=True)

def representativeness(meta, ids, columns, min_share, length='char_count', by='collection_name'):
    """Diferencia máxima de proporciones por variable y prueba KS de la longitud por colección."""
    sample = meta[meta['document_id'].isin(set(ids))]
    rows = []
    for col in columns:
        corpus_share = group_rare(meta[col], min_share).value_counts(normalize=True)
        sample_share = group_rare(sample[col], min_share).value_counts(normalize=True)
        diff = sample_share.reindex(corpus_share.index, fill_value=0) - corpus_share
        rows.append({'check': f'proporciones {col}', 'group': 'total', 'value': float(diff.abs().max()), 'p_value': np.nan})
    for group, corpus_sub in meta.groupby(by):
        sample_sub = sample[sample[by] == group]
        if len(sample_sub) < 2: continue
        test = ks_2samp(sample_sub[length], corpus_sub[length])
        rows.append({'check': f'KS {length}', 'group': group, 'value': float(test.statistic), 'p_value': float(test.pvalue)})
        rows.append({'check': f'mediana {length} muestra/corpus', 'group': group,
                     'value': float(sample_sub[length].median() / corpus_sub[length].median()), 'p_value': np.nan})
    return pd.DataFrame(rows)

def write_sample(samples_dir, name, ids, meta):
    """Escribe sample_<name>_ids.txt (si hay IDs) y sample_<name>_meta.json. Solo IDs y metadatos."""
    samples_dir = Path(samples_dir)
    samples_dir.mkdir(parents=True, exist_ok=True)
    paths = {'meta': samples_dir / f'sample_{name}_meta.json'}
    if ids is not None:
        paths['ids'] = samples_dir / f'sample_{name}_ids.txt'
        paths['ids'].write_text(''.join(f'{i}\n' for i in ids), encoding='utf-8')
    paths['meta'].write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')
    return {k: str(v) for k, v in paths.items()}

def read_sample_ids(path):
    return [line.strip() for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
