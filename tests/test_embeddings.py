import numpy as np
import pytest
from app import embeddings

class FakeTokenizer:
    """Un token por palabra más dos especiales; registra cómo se llamó."""
    def __init__(self): self.kwargs = None
    def __call__(self, texts, **kwargs):
        self.kwargs = kwargs
        return {'input_ids': [[0] + [1] * len(t.split()) + [2] for t in texts]}

class FakeEncoder:
    """Registra lo que recibiría SentenceTransformer.encode sin descargar el modelo."""
    max_seq_length = 512
    def __init__(self, vectors=None):
        self.calls = []
        self.tokenizer = FakeTokenizer()
        self.vectors = vectors
    def encode(self, texts, normalize_embeddings, **kwargs):
        self.calls.append((texts, normalize_embeddings))
        return self.vectors if self.vectors is not None else np.zeros((len(texts), 3))

@pytest.fixture
def fake(monkeypatch):
    enc = FakeEncoder()
    monkeypatch.setattr(embeddings, 'encoder', lambda profile=None: enc)
    for var in ('EMBEDDING_QUERY_PREFIX', 'EMBEDDING_PASSAGE_PREFIX', 'EMBEDDING_NORMALIZE', 'EMBEDDING_DIMENSION'):
        monkeypatch.delenv(var, raising=False)
    return enc

def test_defaults_keep_e5_behavior(fake):
    assert embeddings.embed(['hola'], kind='query') == [[0.0, 0.0, 0.0]]
    embeddings.embed(['hola'])
    assert fake.calls == [(['query: hola'], True), (['passage: hola'], True)]

def test_prefixes_and_normalization_come_from_env(fake, monkeypatch):
    monkeypatch.setenv('EMBEDDING_QUERY_PREFIX', '')
    monkeypatch.setenv('EMBEDDING_PASSAGE_PREFIX', 'doc: ')
    monkeypatch.setenv('EMBEDDING_NORMALIZE', 'false')
    embeddings.embed(['a'], kind='query')
    embeddings.embed(['b'])
    assert fake.calls == [(['a'], False), (['doc: b'], False)]

def test_invalid_normalize_flag_fails(fake, monkeypatch):
    monkeypatch.setenv('EMBEDDING_NORMALIZE', 'quizas')
    with pytest.raises(ValueError):
        embeddings.embed(['a'])

def test_profile_overrides_env(fake, monkeypatch):
    monkeypatch.setenv('EMBEDDING_QUERY_PREFIX', 'query: ')
    profile = {'model': 'm', 'revision': 'abc', 'query_prefix': 'Instruct: x\nQuery:', 'passage_prefix': None, 'normalize': True}
    embeddings.embed(['¿qué?'], kind='query', profile=profile)
    embeddings.embed(['pasaje'], profile=profile)
    assert fake.calls == [(['Instruct: x\nQuery:¿qué?'], True), (['pasaje'], True)]
    assert embeddings.settings(profile)['revision'] == 'abc' and embeddings.settings(profile)['dimension'] is None

def test_matryoshka_dimension_truncates_and_renormalizes(fake):
    fake.vectors = np.array([[3.0, 4.0, 12.0], [0.0, 2.0, 5.0]])
    vectors = embeddings.embed_array(['a', 'b'], profile={'model': 'm', 'normalize': True, 'dimension': 2})
    assert vectors.shape == (2, 2) and vectors.dtype == np.float32
    assert np.allclose(vectors, [[0.6, 0.8], [0.0, 1.0]])

def test_count_tokens_includes_prefix_and_special_tokens_without_truncation(fake):
    assert embeddings.count_tokens(['uno dos tres']) == [6]
    assert embeddings.count_tokens(['uno'], kind='query') == [4]
    assert fake.tokenizer.kwargs['truncation'] is False and fake.tokenizer.kwargs['add_special_tokens'] is True
    assert embeddings.max_tokens() == 512

def test_core_uses_shared_encoder():
    from app import core
    assert core.embed is embeddings.embed
