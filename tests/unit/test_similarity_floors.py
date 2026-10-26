"""Tests for the similarity floors a search takes from the settings."""

from pathlib import Path

from semantic_code_mcp.config import Settings
from semantic_code_mcp.services.projects import similarity_floors
from semantic_code_mcp.services.search import SimilarityFloors


def test_each_floor_of_the_settings_holds_the_hits_it_names():
    settings = Settings(
        cache_dir=Path("/cache"), min_similarity=0.3, min_similarity_when_words_match=0.1
    )

    assert similarity_floors(settings) == SimilarityFloors(meaning_only=0.3, words_match=0.1)
