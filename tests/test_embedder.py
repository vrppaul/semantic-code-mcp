"""Tests for embedding generation."""

from sentence_transformers import SentenceTransformer

from semantic_code_mcp.embedder import Embedder


class TestEmbedder:
    """Tests for Embedder with pre-loaded model."""

    def test_creates_embedder(self, model: SentenceTransformer):
        """Can create an embedder instance with a model."""
        embedder = Embedder(model)
        assert embedder is not None

    def test_embed_single_text(self, embedder: Embedder):
        """Can embed a single text string."""
        embedding = embedder.embed_text("def hello(): pass")

        assert isinstance(embedding, list)
        assert len(embedding) == 384  # MiniLM embedding dimension
        assert all(isinstance(x, float) for x in embedding)

    def test_embed_batch(self, embedder: Embedder):
        """Can embed multiple texts in a batch."""
        texts = [
            "def hello(): pass",
            "class Foo: pass",
            "def bar(x): return x * 2",
        ]
        embeddings = embedder.embed_batch(texts)

        assert len(embeddings) == 3
        assert all(len(e) == 384 for e in embeddings)

    def test_embed_empty_batch_returns_empty(self, embedder: Embedder):
        """Empty batch returns empty list."""
        embeddings = embedder.embed_batch([])
        assert embeddings == []

    def test_similar_texts_have_similar_embeddings(self, embedder: Embedder):
        """Semantically similar texts should have similar embeddings."""
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

    def test_embedding_dimension(self, embedder: Embedder):
        """Embeddings have correct dimension for the model."""
        embedding = embedder.embed_text("test")
        assert embedder.embedding_dim == 384
        assert len(embedding) == embedder.embedding_dim
