import hashlib
import random
import re
from pathlib import Path
import pandas as pd
import pytest
from pipeline import chunking, dedup
from pipeline.profiling import hash_lines, normalize_lines

A = {'chunk_tokens': 20, 'overlap_tokens': 4}
B = {'chunk_tokens': 20, 'overlap_sentences': 1, 'paragraph_min_fill': 0.5}

def word_offsets(text):
    """Tokenizador falso: cada palabra o signo de puntuación es un token."""
    return [m.span() for m in re.finditer(r'\w+|[^\w\s]', text)]

def regex_sentences(text):
    return [m.span() for m in re.finditer(r'[^.!?]+[.!?]*', text)]

def sentence(tag, words):
    return ' '.join(f'{tag}w{k}' for k in range(words)) + '.'

def chunk(text, strategy, params=None, boilerplate=frozenset()):
    params = params or (A if strategy == 'A' else B)
    chunks, excluded = chunking.chunk_document(text, word_offsets(text), strategy, params, regex_sentences, boilerplate)
    return chunks, excluded

def test_chunk_id_is_deterministic():
    expected = hashlib.sha256(b'd1|A|1|0|10').hexdigest()[:32]
    assert chunking.chunk_id('d1', 'A', 1, 0, 10) == expected == chunking.chunk_id('d1', 'A', 1, 0, 10)
    assert chunking.chunk_id('d1', 'A', 1, 0, 11) != expected and chunking.chunk_id('d1', 'B', 1, 0, 10) != expected

@pytest.mark.parametrize('strategy', ['A', 'B'])
def test_random_texts_respect_offsets_budget_and_coverage(strategy):
    rng = random.Random(7)
    for _ in range(40):
        paragraphs = [' '.join(sentence(f'p{p}s{s}', rng.randint(1, 30)) for s in range(rng.randint(1, 5))) for p in range(rng.randint(1, 4))]
        text = '  ' + '\n'.join(paragraphs) + '\n  '
        chunks, _ = chunk(text, strategy)
        tokens = chunking.TokenIndex(word_offsets(text))
        for c in chunks:
            piece = text[c['char_start']:c['char_end']]
            assert piece == piece.strip() and piece
            assert c['content_tokens'] == tokens.count(c['char_start'], c['char_end']) <= 20
        covered = [any(c['char_start'] <= s and e <= c['char_end'] for c in chunks) for s, e in word_offsets(text)]
        assert all(covered)

@pytest.mark.parametrize('strategy', ['A', 'B'])
def test_short_document_is_a_single_chunk(strategy):
    text = '  Un post corto de Bluesky con un enlace.  '
    chunks, _ = chunk(text, strategy)
    assert [(text[c['char_start']:c['char_end']], c['mid_sentence']) for c in chunks] == [(text.strip(), False)]

@pytest.mark.parametrize('strategy', ['A', 'B'])
@pytest.mark.parametrize('text', ['', '   \n\t  '])
def test_empty_or_blank_text_has_no_chunks(strategy, text):
    assert chunk(text, strategy)[0] == []

def test_window_overlap_and_tail():
    text = ' '.join(f'w{i}' for i in range(50))
    chunks, _ = chunk(text, 'A')
    tokens = chunking.TokenIndex(word_offsets(text))
    ranges = [tokens.within(c['char_start'], c['char_end']) for c in chunks]
    assert ranges == [(0, 20), (16, 36), (32, 50)]
    assert all(prev[1] - nxt[0] == 4 for prev, nxt in zip(ranges, ranges[1:]))

def test_text_shorter_than_overlap_is_single_chunk():
    assert len(chunk('uno dos', 'A', {'chunk_tokens': 20, 'overlap_tokens': 4})[0]) == 1

def test_structure_keeps_sentences_whole_with_one_sentence_overlap():
    sentences = [sentence(f's{i}', 6) for i in range(6)]          # 7 tokens cada una
    text = ' '.join(sentences)
    chunks, _ = chunk(text, 'B')
    pieces = [text[c['char_start']:c['char_end']] for c in chunks]
    assert not any(c['mid_sentence'] for c in chunks)
    assert pieces == [' '.join(sentences[0:2]), ' '.join(sentences[1:3]), ' '.join(sentences[2:4]),
                      ' '.join(sentences[3:5]), ' '.join(sentences[4:6])]

def test_sentence_longer_than_budget_is_split_and_flagged():
    text = sentence('largo', 49) + ' ' + sentence('corta', 3)
    chunks, _ = chunk(text, 'B')
    assert all(c['content_tokens'] <= 20 for c in chunks)
    assert sum(c['mid_sentence'] for c in chunks) >= 2
    assert text[chunks[-1]['char_start']:chunks[-1]['char_end']].endswith(sentence('corta', 3))

def test_structure_prefers_paragraph_breaks():
    first = ' '.join([sentence('a', 5), sentence('b', 5)])        # 12 tokens
    second = ' '.join([sentence('c', 5), sentence('d', 5)])
    text = f'{first}\n{second}'
    chunks, _ = chunk(text, 'B')
    assert text[chunks[0]['char_start']:chunks[0]['char_end']] == first
    assert not any(c['mid_sentence'] for c in chunks)

def test_boilerplate_lines_are_skipped_as_spans():
    footer = 'Suscríbete a nuestro boletín semanal de noticias'
    text = f'Primera parte del artículo con contenido.\n{footer}\nSegunda parte del artículo.'
    hashes = frozenset(hash_lines(normalize_lines(pd.Series([footer.upper()]))))
    chunks, excluded = chunk(text, 'A', boilerplate=hashes)
    pieces = [text[c['char_start']:c['char_end']] for c in chunks]
    assert pieces == ['Primera parte del artículo con contenido.', 'Segunda parte del artículo.']
    assert excluded == len(footer)

def test_drop_fragments_only_when_document_has_several_chunks():
    small = {'content_tokens': 3}
    big = {'content_tokens': 30}
    assert chunking.drop_fragments([small], 8) == ([small], 0)
    assert chunking.drop_fragments([big, small], 8) == ([big], 1)

def test_template_documents_only_for_repeated_texts_of_the_collection():
    meta = pd.DataFrame({'document_id': list('abcdef'), 'collection_name': ['url_crawls'] * 4 + ['bluesky_posts'] * 2,
                         'text_sha256': ['t', 't', 't', 'u', 't', 't']})
    assert chunking.template_documents(meta, 'url_crawls', 3) == {'a', 'b', 'c'}

def test_dedup_keeps_canonical_and_all_provenances():
    chunks = pd.DataFrame({'document_id': ['c', 'a', 'b', 'a'], 'char_start': [0, 10, 0, 0],
                           'chunk_sha256': ['h', 'h', 'h', 'x'], 'collection_name': ['n', 'n', 'p', 'n']})
    kept, removed = dedup.deduplicate(chunks)
    row = kept.set_index('chunk_sha256').loc['h']
    assert (row['document_id'], row['char_start'], row['provenance_ids']) == ('a', 10, ['a', 'b', 'c'])
    assert len(removed) == 2 and len(kept) == 2
    again, removed_again = dedup.deduplicate(kept.drop(columns='provenance_ids'))
    assert again['chunk_sha256'].tolist() == kept['chunk_sha256'].tolist() and removed_again.empty
    summary = dedup.dedup_summary(chunks, removed).set_index('collection_name')
    assert summary.loc['total', 'removed'] == 2 and summary.loc['p', 'removed'] == 1

def _chunker_profile():
    import yaml
    cfg = yaml.safe_load(Path('configs/chunking.yaml').read_text(encoding='utf-8'))
    return yaml.safe_load(Path(cfg['encoders_config']).read_text(encoding='utf-8'))['profiles'][cfg['encoder_profile']]

def _encoder_available():
    try:
        import nltk
        from huggingface_hub import try_to_load_from_cache
        nltk.data.find('tokenizers/punkt_tab')
        profile = _chunker_profile()
        return isinstance(try_to_load_from_cache(profile['model'], 'config.json', revision=profile['revision']), str)
    except Exception:
        return False

@pytest.mark.skipif(not _encoder_available(), reason='requiere el modelo del encoder en caché y punkt_tab (contenedor pipeline)')
@pytest.mark.parametrize('strategy', ['A', 'B'])
def test_real_tokenizer_and_punkt_respect_encoder_limit(strategy):
    import yaml
    from app.embeddings import count_tokens, max_tokens, token_offsets
    cfg = yaml.safe_load(Path('configs/chunking.yaml').read_text(encoding='utf-8'))
    text = Path(cfg['example_text']).read_text(encoding='utf-8')
    profile = _chunker_profile()
    chunks, _ = chunking.chunk_document(text, token_offsets([text], profile)[0], strategy, cfg['strategies'][strategy],
                                        chunking.punkt_splitter('spanish'))
    pieces = [text[c['char_start']:c['char_end']] for c in chunks]
    assert len(chunks) > 1 and max(count_tokens(pieces, 'passage', profile)) <= max_tokens(profile)
    assert all(c['content_tokens'] <= cfg['strategies'][strategy]['chunk_tokens'] for c in chunks)
    if strategy == 'B': assert not any(c['mid_sentence'] for c in chunks)
