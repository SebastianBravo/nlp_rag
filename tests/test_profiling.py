import numpy as np
import pandas as pd
import pytest
from pipeline import profiling as prof

def test_text_features_flags():
    df = pd.DataFrame({'document_id': list('abcdefgh'), 'text': [
        'Un texto normal con varias palabras.', '   ', 'https://x.co/a www.y.org', '🔥🔥 !!',
        'Hola <b>mundo</b> con más palabras aquí', 'Precio &amp; calidad del producto final', 'informaciÃ³n pÃºblica del día de hoy',
        None]})
    f = prof.text_features(df, short_words=5).set_index('document_id')
    assert f.loc['a', ['text_is_null', 'char_count_calc', 'word_count_calc']].tolist() == [False, 36, 6]
    assert f.loc['b', 'is_blank'] and not f.loc['b', 'no_alnum']
    assert f.loc['c', 'only_urls'] and f.loc['c', 'n_urls'] == 2
    assert f.loc['d', 'no_alnum'] and f.loc['d', 'is_short']
    assert f.loc['e', 'has_html_tag'] and not f.loc['e', 'has_html_entity']
    assert f.loc['f', 'has_html_entity']
    assert f.loc['g', 'has_mojibake'] and not f.loc['a', 'has_mojibake']
    assert f.loc['h', 'text_is_null'] and f.loc['h', 'is_blank']

def test_mojibake_ignores_valid_accents():
    df = pd.DataFrame({'document_id': ['a', 'b'], 'text': ['NÃO é mojibake: información, árbol, Ñandú', 'â€œcitaâ€']})
    assert prof.text_features(df)['has_mojibake'].tolist() == [False, True]

def test_completeness_counts_are_additive():
    df = pd.DataFrame({'s': ['x', '', '  ', None], 'l': [np.array(['a']), np.array([]), None, np.array(['b'])]})
    comp = prof.completeness(df).set_index('column')
    assert comp.loc['s', ['nulls', 'empty', 'whitespace_only', 'nonempty']].tolist() == [1, 1, 1, 1]
    assert comp.loc['l', ['nulls', 'empty', 'nonempty']].tolist() == [1, 1, 2]
    summed = pd.concat([prof.completeness(df.iloc[:2]), prof.completeness(df.iloc[2:])]).groupby('column').sum()
    assert summed.loc['s', 'nonempty'] == 1 and summed.loc['l', 'rows'] == 4

def test_recount_detects_mismatches():
    stored = pd.DataFrame({'document_id': ['a', 'b'], 'char_count': [3, 9], 'word_count': [1, 1], 'text_sha256': ['h', 'x']})
    calc = pd.DataFrame({'document_id': ['a', 'b'], 'char_count_calc': [3, 3], 'word_count_calc': [1, 1], 'text_sha256_calc': ['h', 'y']})
    out = prof.recount(stored, calc).set_index('column')['mismatches']
    assert out.to_dict() == {'char_count': 1, 'word_count': 0, 'text_sha256': 1}

def test_schema_report_statuses():
    schema = pd.DataFrame({'position': [0, 1, 2], 'column': ['a', 'c', 'x'], 'arrow_type': ['string'] * 3})
    out = prof.schema_report(schema, ['a', 'b', 'c']).set_index('column')['status']
    assert out.to_dict() == {'a': 'ok', 'b': 'falta', 'c': 'otra_posicion', 'x': 'no_esperada'}

def test_expected_counts_reports_differences():
    df = pd.DataFrame({'collection_name': ['p', 'p', 'n'], 'source': ['s', 's', 'm']})
    out = prof.expected_counts(df, {'total': 3, 'collection_name': {'p': 2, 'n': 2}})
    diff = out.set_index(['field', 'value'])['diff']
    assert diff[('total', 'total')] == 0 and diff[('collection_name', 'n')] == -1

def test_group_rare():
    values = pd.Series(['es'] * 90 + ['en'] * 9 + ['hu'] + [None] * 5)
    assert prof.group_rare(values, 0.05).value_counts().to_dict() == {'es': 90, 'en': 9, 'desconocido': 5, 'otro': 1}

def test_language_agreement_and_matrix():
    df = pd.DataFrame({'collection_name': ['a'] * 4 + ['b'] * 2, 'char_count': [10, 60, 150, 600, 20, 30],
                       'language_base': ['es', 'es', 'en', 'en', None, 'es'],
                       'language_detected': ['es', 'pt', 'en', None, 'es', 'es']})
    agreement = prof.language_agreement(df).set_index('collection_name')
    assert agreement.loc['a', 'compared'] == 3 and agreement.loc['a', 'agreement'] == pytest.approx(2 / 3)
    assert agreement.loc['total', 'declared_null'] == 1
    matrix = prof.language_matrix(df, min_share=0.2)
    assert matrix.loc['es', 'es'] == 2 and matrix.loc['en', 'indeterminado'] == 1
    by_length = prof.agreement_by_length(df, [0, 50, 100, 1000])
    assert by_length['n'].sum() == 4

def test_date_report_formats_and_future_dates():
    df = pd.DataFrame({'collection_name': ['a'] * 5,
                       'created_at': ['2024-01-05', '2025-08-04T18:20:26.202Z', None, '2026-12-01', 'no es fecha'],
                       'synced_at': ['2026-10-06 10:00:00+00:00'] * 5})
    row = prof.date_report(df, '2026-10-19', {'a': '2025-01-01'}).set_index('collection_name').loc['a']
    assert (row['solo_fecha'], row['fecha_hora'], row['nulo'], row['no_interpretable']) == (2, 1, 1, 1)
    assert row['after_synced'] == 1 and row['after_version'] == 1 and row['before_plausible_start'] == 1

def test_exact_duplicates_counts_extra_rows_and_cross_collection():
    df = pd.DataFrame({'document_id': ['a', 'b', 'c', 'd', 'e'], 'collection_name': ['p', 'p', 'n', 'n', 'n'],
                       'url_domain': ['x', 'x', 'y', 'y', 'y'], 'char_count': [5, 5, 5, 8, 9],
                       'text_sha256': ['h1', 'h1', 'h1', 'h2', 'h3']})
    out = prof.exact_duplicates(df)
    total = out['by_collection'].set_index('collection_name').loc['total']
    assert total['extra_rows'] == 2 and total['unique_texts'] == 3
    assert out['cross_collection_groups'] == 1
    assert out['group_sizes'].set_index('group_size')['groups'].to_dict() == {1: 2, 3: 1}
    largest = out['largest'].iloc[0]
    assert (largest['group_size'], largest['collections'], largest['domains'], largest['top_domain']) == (3, 2, 2, 'x')
    assert 'text_sha256' not in out['largest'].columns

def test_near_duplicates_finds_only_similar_texts():
    base = ' '.join(f'palabra{i}' for i in range(60))
    texts = pd.Series({'a': base, 'b': base + ' extra', 'c': ' '.join(f'otra{i}' for i in range(60)), 'd': ''})
    pairs = prof.near_duplicate_pairs(texts, k=5, num_perm=128, threshold=0.8, seed=1)
    assert pairs[['document_id_a', 'document_id_b']].values.tolist() == [['a', 'b']]
    meta = pd.DataFrame({'document_id': list('abcd'), 'collection_name': ['p'] * 4, 'url_domain': ['x', 'y', 'x', 'x'],
                         'text_sha256': list('1234'), 'char_count': [100, 106, 100, 0]})
    typed = prof.typify_pairs(pairs, meta)
    assert typed['type'].tolist() == ['otro_dominio'] and typed['length_ratio'].iloc[0] == pytest.approx(100 / 106)
    summary = prof.near_duplicate_summary(typed, meta).set_index('collection_name')
    assert summary.loc['total', 'with_near_duplicate'] == 2

def test_url_report_and_tracking():
    df = pd.DataFrame({'collection_name': ['a'] * 4,
                       'url': ['https://x.com/a?utm_source=t&id=1', 'http://x.com/a', 'ftp://x.com/f', 'https://x.com/a?utm_source=t&id=1'],
                       'final_url': [None, 'https://x.com/b', None, None]})
    row = prof.url_report(df, ['utm_*', 'fbclid']).set_index('collection_name').loc['total']
    assert (row['valid'], row['https'], row['with_tracking'], row['duplicated'], row['final_url_differs']) == (3, 2, 2, 1, 1)
    assert prof.tracking_params(df, ['utm_*'])['param'].tolist() == ['utm_source']

def test_boilerplate_by_domain():
    footer = 'Suscríbete a nuestro boletín semanal de noticias'
    df = pd.DataFrame({'document_id': ['a', 'b', 'c'], 'text': [f'Artículo uno con contenido propio\n{footer}',
                       f'Artículo dos con otro contenido distinto\n{footer.upper()}', 'Artículo tres sin pie de página alguno']})
    lines = prof.line_hashes(df, min_chars=20)
    per_doc, per_domain = prof.boilerplate(lines, pd.DataFrame({'document_id': ['a', 'b', 'c'], 'url_domain': ['x'] * 3}), min_docs=2)
    assert per_doc.set_index('document_id')['n_boilerplate_lines'].to_dict() == {'a': 1, 'b': 1, 'c': 0}
    assert per_domain['boilerplate_lines'].tolist() == [1]

def test_domain_concentration():
    df = pd.DataFrame({'collection_name': ['url_crawls'] * 4, 'url_domain': ['a', 'a', 'b', 'c']})
    row = prof.domain_concentration(df).iloc[0]
    assert row['domains'] == 3 and row['hhi'] == pytest.approx(0.25 + 0.0625 * 2)
