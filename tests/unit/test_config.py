"""Tests for settings and where a project's index lives."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from semantic_code_mcp.config import LOCAL_INDEX_DIRECTORY, Settings, get_index_path

CACHE = Path("/cache")


def test_settings_are_read_from_prefixed_environment_variables(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SEMANTIC_CODE_MCP_EMBEDDING_MODEL", "custom-model")
    monkeypatch.setenv("SEMANTIC_CODE_MCP_LOCAL_INDEX", "true")

    settings = Settings()

    assert (settings.embedding_model, settings.local_index) == ("custom-model", True)


def test_a_project_env_file_is_not_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    (tmp_path / ".env").write_text("SEMANTIC_CODE_MCP_EMBEDDING_MODEL=from-dotenv\n")
    monkeypatch.chdir(tmp_path)

    assert Settings().embedding_model != "from-dotenv"


def test_the_global_index_of_a_project_lives_under_the_cache():
    index_path = get_index_path(Settings(cache_dir=CACHE), Path("/home/user/project"))

    assert index_path.parent == CACHE


def test_the_local_index_lives_in_the_project():
    project = Path("/home/user/project")

    assert get_index_path(Settings(local_index=True), project) == project / LOCAL_INDEX_DIRECTORY


@pytest.mark.parametrize("local_index", [False, True], ids=["global index", "local index"])
def test_two_spellings_of_one_project_share_its_index(local_index: bool):
    settings = Settings(cache_dir=CACHE, local_index=local_index)

    assert get_index_path(settings, Path("/home/user/other/../project")) == get_index_path(
        settings, Path("/home/user/project")
    )


def test_two_projects_have_different_indexes():
    settings = Settings(cache_dir=CACHE)

    assert get_index_path(settings, Path("/one")) != get_index_path(settings, Path("/two"))


def test_a_floor_for_word_matches_above_the_floor_for_meaning_alone_is_refused():
    """A hit whose words match the query's is never held to more than one whose do not."""
    with pytest.raises(ValidationError):
        Settings(min_similarity=0.1, min_similarity_when_words_match=0.2)
