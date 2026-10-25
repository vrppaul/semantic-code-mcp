"""Tests for the mtime cache that tells an index run which files changed."""

import os
from pathlib import Path

import pytest

from semantic_code_mcp.domain.models import FileChanges
from semantic_code_mcp.storage.cache import CACHE_FILENAME, FileChangeCache

EARLIER_MTIME_SECONDS = 1_000_000_000
LATER_MTIME_SECONDS = 2_000_000_000


@pytest.fixture
def source_files(tmp_path: Path) -> list[str]:
    paths = [tmp_path / "first.py", tmp_path / "second.py"]
    for path in paths:
        path.write_text("x = 1\n")
        os.utime(path, (EARLIER_MTIME_SECONDS, EARLIER_MTIME_SECONDS))
    return [str(path) for path in paths]


def test_every_file_is_new_to_an_empty_cache(tmp_path: Path, source_files: list[str]):
    changes = FileChangeCache(tmp_path).get_changes(source_files)

    assert changes == FileChanges(new=source_files, modified=[], deleted=[])


def test_a_recorded_file_with_the_same_mtime_is_unchanged(tmp_path: Path, source_files: list[str]):
    cache = FileChangeCache(tmp_path)
    cache.update_files(source_files)

    assert cache.get_changes(source_files) == FileChanges(new=[], modified=[], deleted=[])


def test_a_file_with_a_newer_mtime_is_modified(tmp_path: Path, source_files: list[str]):
    cache = FileChangeCache(tmp_path)
    cache.update_files(source_files)
    os.utime(source_files[0], (LATER_MTIME_SECONDS, LATER_MTIME_SECONDS))

    assert cache.get_changes(source_files) == FileChanges(
        new=[], modified=[source_files[0]], deleted=[]
    )


def test_a_recorded_file_no_longer_listed_is_deleted(tmp_path: Path, source_files: list[str]):
    cache = FileChangeCache(tmp_path)
    cache.update_files(source_files)

    assert cache.get_changes(source_files[:1]) == FileChanges(
        new=[], modified=[], deleted=[source_files[1]]
    )


def test_removed_files_are_new_again(tmp_path: Path, source_files: list[str]):
    cache = FileChangeCache(tmp_path)
    cache.update_files(source_files)
    cache.remove_files(source_files[:1])

    assert cache.get_stale_files(source_files) == source_files[:1]


def test_what_is_recorded_survives_a_new_instance(tmp_path: Path, source_files: list[str]):
    FileChangeCache(tmp_path).update_files(source_files)

    assert FileChangeCache(tmp_path).get_stale_files(source_files) == []


def test_clear_forgets_every_file(tmp_path: Path, source_files: list[str]):
    cache = FileChangeCache(tmp_path)
    cache.update_files(source_files)
    cache.clear()

    assert cache.get_stale_files(source_files) == source_files


def test_a_corrupted_cache_file_counts_as_empty(tmp_path: Path, source_files: list[str]):
    (tmp_path / CACHE_FILENAME).write_text("{not json")

    assert FileChangeCache(tmp_path).get_stale_files(source_files) == source_files
