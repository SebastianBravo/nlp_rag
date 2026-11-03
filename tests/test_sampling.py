import pandas as pd
import pytest
from pipeline import sampling

def corpus(n_es=600, n_en=300, n_hu=90, n_none=10):
    langs = ['es'] * n_es + ['en'] * n_en + ['hu'] * n_hu + [None] * n_none
    return pd.DataFrame({'document_id': [f'doc{i:05d}' for i in range(len(langs))],
                         'collection_name': ['news' if i % 3 else 'posts' for i in range(len(langs))],
                         'language_base': langs, 'char_count': [100 + (i * 37) % 900 for i in range(len(langs))]})

def test_allocate_is_proportional_and_exact():
    alloc = sampling.allocate(pd.Series({'a': 500, 'b': 300, 'c': 200}), 7)
    assert alloc.sum() == 7 and alloc.to_dict() == {'a': 4, 'b': 2, 'c': 1}
    with pytest.raises(ValueError):
        sampling.allocate(pd.Series({'a': 2}), 3)

def test_nested_samples_are_nested_deterministic_and_stratified():
    df = corpus()
    samples, strata = sampling.nested_samples(df, [100, 500], ['collection_name', 'language_base'], 0.05, seed=7)
    assert len(samples[100]) == 100 and len(samples[500]) == 500
    assert set(samples[100]) <= set(samples[500])
    again, _ = sampling.nested_samples(df.sample(frac=1, random_state=3), [100, 500], ['collection_name', 'language_base'], 0.05, seed=7)
    assert again == samples
    other, _ = sampling.nested_samples(df, [100, 500], ['collection_name', 'language_base'], 0.05, seed=8)
    assert other[100] != samples[100]
    dist = sampling.distribution(strata, samples[500])
    assert dist['diff'].abs().max() < 0.01
    assert 'news|otro' not in set(strata) and 'news|desconocido' in set(strata)

def test_representativeness_reports_shares_and_ks():
    df = corpus()
    samples, _ = sampling.nested_samples(df, [300], ['collection_name', 'language_base'], 0.05, seed=1)
    checks = sampling.representativeness(df, samples[300], ['collection_name', 'language_base'], 0.05)
    shares = checks[checks['check'].str.startswith('proporciones')]
    assert len(shares) == 2 and shares['value'].max() < 0.02
    assert set(checks.loc[checks['check'] == 'KS char_count', 'group']) == {'news', 'posts'}

def test_write_and_read_sample(tmp_path):
    paths = sampling.write_sample(tmp_path, 3, ['b', 'a', 'c'], {'seed': 1})
    assert sampling.read_sample_ids(paths['ids']) == ['b', 'a', 'c']
    full = sampling.write_sample(tmp_path, 'full', None, {'size': 3})
    assert 'ids' not in full and (tmp_path / 'sample_full_meta.json').is_file()
