"""Tests for configuration module."""

from pathlib import Path

from semantic_code_mcp.config import Settings, get_index_path


class TestSettings:
    """Tests for Settings model."""

    def test_default_settings(self):
        """Settings have sensible defaults."""
        settings = Settings()
        assert settings.embedding_model == "all-MiniLM-L6-v2"
        assert settings.embedding_device == "auto"
        assert settings.chunking_target_tokens == 800
        assert settings.chunking_max_tokens == 1500
        assert "node_modules" in str(settings.ignore_patterns)
        assert settings.use_gitignore is True

    def test_cache_dir_default(self):
        """Default cache dir is ~/.cache/semantic-code-mcp."""
        settings = Settings()
        assert "semantic-code-mcp" in str(settings.cache_dir)

    def test_custom_cache_dir(self):
        """Cache dir can be overridden."""
        settings = Settings(cache_dir=Path("/custom/cache"))
        assert settings.cache_dir == Path("/custom/cache")

    def test_local_index_flag(self):
        """local_index flag can be set."""
        settings = Settings(local_index=True)
        assert settings.local_index is True

    def test_settings_from_env(self, monkeypatch):
        """Settings can be loaded from environment variables."""
        monkeypatch.setenv("SEMANTIC_CODE_MCP_EMBEDDING_MODEL", "custom-model")
        monkeypatch.setenv("SEMANTIC_CODE_MCP_LOCAL_INDEX", "true")
        settings = Settings()
        assert settings.embedding_model == "custom-model"
        assert settings.local_index is True


class TestGetIndexPath:
    """Tests for index path resolution."""

    def test_global_cache_path(self):
        """With default settings, index is in global cache."""
        settings = Settings(cache_dir=Path("/cache"))
        project_path = Path("/home/user/myproject")
        index_path = get_index_path(settings, project_path)

        # Should be in cache dir with hash of project path
        assert str(index_path).startswith("/cache/")
        assert index_path != Path("/cache")  # Should have subdirectory

    def test_local_index_path(self):
        """With local_index=True, index is in project .semantic-code/."""
        settings = Settings(local_index=True)
        project_path = Path("/home/user/myproject")
        index_path = get_index_path(settings, project_path)

        assert index_path == project_path / ".semantic-code"

    def test_same_project_same_hash(self):
        """Same project path produces same index path."""
        settings = Settings(cache_dir=Path("/cache"))
        project_path = Path("/home/user/myproject")

        path1 = get_index_path(settings, project_path)
        path2 = get_index_path(settings, project_path)

        assert path1 == path2

    def test_different_projects_different_hash(self):
        """Different project paths produce different index paths."""
        settings = Settings(cache_dir=Path("/cache"))

        path1 = get_index_path(settings, Path("/project1"))
        path2 = get_index_path(settings, Path("/project2"))

        assert path1 != path2
