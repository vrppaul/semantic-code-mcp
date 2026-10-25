"""Helpers shared by tests."""

import subprocess
from pathlib import Path


def run_git(repository: Path, *arguments: str) -> None:
    """Run a git command in a repository with a fixed test identity."""
    subprocess.run(  # nosec B603, B607
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
    )


def make_repository(root: Path) -> Path:
    """Create an empty git repository, creating the directory if needed."""
    root.mkdir(parents=True, exist_ok=True)
    run_git(root, "init", "--quiet")
    return root


def commit_all(repository: Path) -> None:
    """Stage and commit everything in a repository."""
    run_git(repository, "add", "--all")
    run_git(repository, "commit", "--quiet", "--message", "snapshot")
