"""Configuration and settings."""

import hashlib
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_prefix="SEMANTIC_CODE_MCP_",
        env_file=".env",
        extra="ignore",
    )

    # Embedding settings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_device: str = "auto"

    # Storage settings
    cache_dir: Path = Field(default_factory=lambda: Path.home() / ".cache" / "semantic-code-mcp")
    local_index: bool = False

    # Chunking settings
    chunking_target_tokens: int = 800
    chunking_max_tokens: int = 1500

    # Performance settings
    status_cache_ttl: float = 5.0  # Seconds to cache status check results

    # Ignore patterns
    ignore_patterns: list[str] = Field(
        default_factory=lambda: [
            "node_modules/**",
            ".venv/**",
            "__pycache__/**",
            ".git/**",
            "*.pyc",
            ".pytest_cache/**",
        ]
    )
    use_gitignore: bool = True


def get_index_path(settings: Settings, project_path: Path) -> Path:
    """Get the index storage path for a project.

    Args:
        settings: Application settings.
        project_path: Path to the project root.

    Returns:
        Path where the index should be stored.
    """
    if settings.local_index:
        return project_path / ".semantic-code"

    # Hash the absolute path for global cache
    path_hash = hashlib.sha256(str(project_path.resolve()).encode()).hexdigest()[:16]
    return settings.cache_dir / path_hash
