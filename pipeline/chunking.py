"""Segmentación en chunks (Fase B). Cada chunk es un intervalo continuo del texto original.

Las funciones reciben los offsets de tokens del encoder (`app.embeddings.token_offsets`) y un
separador de oraciones que devuelve intervalos, así que se pueden probar sin el modelo. Siempre
se cumple `text[char_start:char_end] == chunk`.

- Estrategia A: ventana fija de tokens con solapamiento.
- Estrategia B: párrafos y oraciones agrupados hasta el presupuesto de tokens, con una oración de
  solapamiento. Solo corta a mitad de oración cuando una oración sola supera el presupuesto.
"""
import hashlib
from bisect import bisect_left, bisect_right
from functools import lru_cache
import pandas as pd
from pipeline.profiling import hash_lines, normalize_lines

def chunk_id(document_id, strategy, version, start, end):
    """ID determinista: reindexar el mismo chunk siempre produce el mismo ID."""
    return hashlib.sha256(f'{document_id}|{strategy}|{version}|{start}|{end}'.encode('utf-8')).hexdigest()[:32]

def _trim(text, start, end):
    while start < end and text[start].isspace(): start += 1
    while end > start and text[end - 1].isspace(): end -= 1
    return start, end

def line_spans(text):
    """Intervalos (inicio, fin) de cada línea, sin el salto de línea."""
    spans, pos = [], 0
    for line in text.split('\n'):
        spans.append((pos, pos + len(line)))
        pos += len(line) + 1
    return spans

def content_segments(text, boilerplate_hashes=frozenset()):
    """Tramos continuos del texto sin líneas boilerplate, recortados de espacios. Devuelve también los intervalos excluidos."""
    lines = line_spans(text)
    flags = [False] * len(lines)
    if boilerplate_hashes:
        hashes = hash_lines(normalize_lines(pd.Series([text[s:e] for s, e in lines])))
        flags = [h in boilerplate_hashes for h in hashes]
    segments, excluded, current = [], [], None
    for (start, end), is_boilerplate in zip(lines, flags):
        if is_boilerplate:
            if current: segments.append(current)
            current = None
            excluded.append((start, end))
        else:
            current = (current[0], end) if current else (start, end)
    if current: segments.append(current)
    segments = [_trim(text, s, e) for s, e in segments]
    return [(s, e) for s, e in segments if e > s], excluded

class TokenIndex:
    """Cuenta tokens de contenido dentro de un intervalo de caracteres con búsqueda binaria."""
    def __init__(self, offsets):
        self.offsets = offsets
        self.starts = [s for s, _ in offsets]
        self.ends = [e for _, e in offsets]
    def within(self, start, end):
        """Índices [i, j) de los tokens contenidos por completo en [start, end)."""
        return bisect_left(self.starts, start), bisect_right(self.ends, end)
    def count(self, start, end):
        i, j = self.within(start, end)
        return max(j - i, 0)

def window_spans(tokens, start, end, size, overlap):
    """Estrategia A: ventanas de `size` tokens con `overlap` de solapamiento dentro de [start, end)."""
    if not 0 <= overlap < size: raise ValueError('overlap debe ser menor que size')
    i, j = tokens.within(start, end)
    if j - i <= 0: return []
    if j - i <= size: return [(start, end)]
    spans, pos = [], i
    while True:
        stop = min(pos + size, j)
        spans.append((tokens.offsets[pos][0], tokens.offsets[stop - 1][1]))
        if stop == j: return spans
        pos += size - overlap

def sentence_units(text, start, end, split_sentences, tokens, budget):
    """Oraciones de [start, end), separando primero por líneas (párrafos).

    Devuelve las unidades (inicio, fin, párrafo, cortada) y los inicios y fines de oración. Una oración
    que supera el presupuesto se parte en ventanas de tokens y sus piezas quedan marcadas como cortadas.
    """
    units, starts, ends = [], set(), set()
    for paragraph, (line_start, line_end) in enumerate(line_spans(text[start:end])):
        line_start, line_end = _trim(text, start + line_start, start + line_end)
        if line_end <= line_start: continue
        for s, e in split_sentences(text[line_start:line_end]):
            s, e = _trim(text, line_start + s, line_start + e)
            if e <= s: continue
            starts.add(s)
            ends.add(e)
            if tokens.count(s, e) <= budget:
                units.append((s, e, paragraph, False))
            else:
                units.extend((ps, pe, paragraph, True) for ps, pe in window_spans(tokens, s, e, budget, 0))
    return units, starts, ends

def structure_spans(units, tokens, budget, overlap_sentences=1, paragraph_min_fill=0.5):
    """Estrategia B: agrupa unidades consecutivas hasta `budget` tokens.

    Si el grupo termina dentro de un párrafo y contiene un cambio de párrafo, corta en ese cambio
    siempre que el chunk conserve al menos `paragraph_min_fill` del presupuesto. El siguiente chunk
    repite las últimas `overlap_sentences` oraciones si caben con la siguiente.
    """
    spans, i, n = [], 0, len(units)
    fits = lambda a, b: tokens.count(units[a][0], units[b][1]) <= budget
    while i < n:
        j = i
        while j + 1 < n and fits(i, j + 1): j += 1
        if j + 1 < n and units[j][2] == units[j + 1][2]:
            breaks = [k for k in range(i, j) if units[k][2] != units[k + 1][2]
                      and tokens.count(units[i][0], units[k][1]) >= paragraph_min_fill * budget]
            if breaks: j = breaks[-1]
        spans.append((units[i][0], units[j][1]))
        if j + 1 >= n: break
        nxt = max(j + 1 - overlap_sentences, i + 1)
        i = nxt if nxt <= j and fits(nxt, j + 1) else j + 1
    return spans

@lru_cache(maxsize=None)
def punkt(language):
    import nltk
    return nltk.tokenize.PunktTokenizer(language)

def punkt_splitter(language):
    """Separador de oraciones de nltk (punkt) para un idioma de punkt (p. ej. 'spanish')."""
    tokenizer = punkt(language)
    return lambda text: list(tokenizer.span_tokenize(text))

def chunk_document(text, offsets, strategy, params, split_sentences, boilerplate_hashes=frozenset()):
    """Chunks de un documento como lista de dicts con intervalo, tokens de contenido y si corta una oración.

    También devuelve los caracteres excluidos por boilerplate.
    """
    tokens = TokenIndex(offsets)
    budget = params['chunk_tokens']
    segments, excluded = content_segments(text, boilerplate_hashes)
    chunks = []
    for seg_start, seg_end in segments:
        units, starts, ends = sentence_units(text, seg_start, seg_end, split_sentences, tokens, budget)
        if strategy == 'A':
            spans = window_spans(tokens, seg_start, seg_end, budget, params['overlap_tokens'])
        elif strategy == 'B':
            spans = [(seg_start, seg_end)] if tokens.count(seg_start, seg_end) <= budget else \
                structure_spans(units, tokens, budget, params['overlap_sentences'], params['paragraph_min_fill'])
        else:
            raise ValueError(f'Estrategia desconocida: {strategy}')
        for s, e in spans:
            chunks.append({'char_start': s, 'char_end': e, 'content_tokens': tokens.count(s, e),
                           'mid_sentence': s not in starts or e not in ends})
    return chunks, sum(e - s for s, e in excluded)

def drop_fragments(chunks, min_tokens):
    """Descarta fragmentos de menos de `min_tokens` solo si el documento tiene más de un chunk."""
    if len(chunks) <= 1: return chunks, 0
    kept = [c for c in chunks if c['content_tokens'] >= min_tokens]
    return (kept or chunks[:1]), len(chunks) - len(kept or chunks[:1])

def template_documents(meta, collection, min_copies):
    """IDs de documentos de `collection` cuyo texto se repite al menos `min_copies` veces (páginas plantilla)."""
    sub = meta[meta['collection_name'] == collection]
    sizes = sub.groupby('text_sha256')['document_id'].transform('size')
    return set(sub.loc[sizes >= min_copies, 'document_id'])
