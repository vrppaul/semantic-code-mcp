"""Tests for embedding generation."""

import pytest

from semantic_code_mcp.config import Settings
from semantic_code_mcp.indexer.embedder import Embedder


class TestEmbedder:
    """Tests for Embedder."""

    @pytest.fixture
    def settings(self) -> Settings:
        """Create test settings."""
        return Settings()

    def test_creates_embedder(self, settings: Settings):
        """Can create an embedder instance."""
        embedder = Embedder(settings)
        assert embedder is not None

    def test_model_not_loaded_initially(self, settings: Settings):
        """Model is not loaded until first use (lazy loading)."""
        embedder = Embedder(settings)
        assert not embedder.is_loaded

    def test_model_loaded_after_embed(self, settings: Settings):
        """Model is loaded after first embedding call."""
        embedder = Embedder(settings)
        embedder.embed_text("hello world")
        assert embedder.is_loaded

    def test_embed_single_text(self, settings: Settings):
        """Can embed a single text string."""
        embedder = Embedder(settings)
        embedding = embedder.embed_text("def hello(): pass")

        assert isinstance(embedding, list)
        assert len(embedding) == 384  # MiniLM embedding dimension
        assert all(isinstance(x, float) for x in embedding)

    def test_embed_batch(self, settings: Settings):
        """Can embed multiple texts in a batch."""
        embedder = Embedder(settings)
        texts = [
            "def hello(): pass",
            "class Foo: pass",
            "def bar(x): return x * 2",
        ]
        embeddings = embedder.embed_batch(texts)

        assert len(embeddings) == 3
        assert all(len(e) == 384 for e in embeddings)

    def test_embed_empty_batch_returns_empty(self, settings: Settings):
        """Empty batch returns empty list."""
        embedder = Embedder(settings)
        embeddings = embedder.embed_batch([])
        assert embeddings == []

    def test_similar_texts_have_similar_embeddings(self, settings: Settings):
        """Semantically similar texts should have similar embeddings."""
        embedder = Embedder(settings)

        # Similar code
        emb1 = embedder.embed_text("def add(a, b): return a + b")
        emb2 = embedder.embed_text("def sum(x, y): return x + y")

        # Different code
        emb3 = embedder.embed_text("class DatabaseConnection: pass")

        # Compute cosine similarity
        def cosine_sim(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b, strict=True))
            norm_a = sum(x * x for x in a) ** 0.5
            norm_b = sum(x * x for x in b) ** 0.5
            return dot / (norm_a * norm_b)

        sim_similar = cosine_sim(emb1, emb2)
        sim_different = cosine_sim(emb1, emb3)

        # Similar texts should have higher similarity
        assert sim_similar > sim_different

    def test_embedding_dimension(self, settings: Settings):
        """Embeddings have correct dimension for the model."""
        embedder = Embedder(settings)
        embedding = embedder.embed_text("test")
        assert embedder.embedding_dim == 384
        assert len(embedding) == embedder.embedding_dim

    def test_load_explicitly(self, settings: Settings):
        """Can explicitly load the model."""
        embedder = Embedder(settings)
        assert not embedder.is_loaded

        embedder.load()

        assert embedder.is_loaded

    def test_model_name_from_settings(self, settings: Settings):
        """Uses model name from settings."""
        embedder = Embedder(settings)
        assert embedder.model_name == settings.embedding_model
