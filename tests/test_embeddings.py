import numpy as np
import pytest
from app import embeddings

class FakeEncoder:
    """Registra lo que recibiría SentenceTransformer.encode sin descargar el modelo."""
    def __init__(self): self.calls = []
    def encode(self, texts, normalize_embeddings):
        self.calls.append((texts, normalize_embeddings))
        return np.zeros((len(texts), 3))

@pytest.fixture
def fake(monkeypatch):
    enc = FakeEncoder()
    monkeypatch.setattr(embeddings, 'encoder', lambda: enc)
    for var in ('EMBEDDING_QUERY_PREFIX', 'EMBEDDING_PASSAGE_PREFIX', 'EMBEDDING_NORMALIZE'):
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

def test_core_uses_shared_encoder():
    from app import core
    assert core.embed is embeddings.embed
