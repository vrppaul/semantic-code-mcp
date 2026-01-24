"""Tests for file change detection cache."""

from pathlib import Path

from semantic_code_mcp.storage.cache import FileChangeCache


class TestFileChangeCache:
    """Tests for FileChangeCache."""

    def test_create_cache_in_directory(self, tmp_path: Path):
        """Cache creates its storage file in the given directory."""
        cache = FileChangeCache(tmp_path)
        assert cache.cache_path == tmp_path / "file_mtimes.json"

    def test_empty_cache_returns_no_files(self, tmp_path: Path):
        """New cache has no tracked files."""
        cache = FileChangeCache(tmp_path)
        assert cache.get_tracked_files() == []

    def test_update_single_file(self, tmp_path: Path):
        """Can track a single file's mtime."""
        cache = FileChangeCache(tmp_path)

        # Create a test file
        test_file = tmp_path / "test.py"
        test_file.write_text("print('hello')")

        cache.update_file(str(test_file))

        tracked = cache.get_tracked_files()
        assert str(test_file) in tracked

    def test_update_multiple_files(self, tmp_path: Path):
        """Can track multiple files."""
        cache = FileChangeCache(tmp_path)

        files = []
        for i in range(3):
            f = tmp_path / f"file{i}.py"
            f.write_text(f"# file {i}")
            files.append(str(f))

        cache.update_files(files)

        tracked = cache.get_tracked_files()
        assert len(tracked) == 3
        for f in files:
            assert f in tracked

    def test_persistence_across_instances(self, tmp_path: Path):
        """Cache persists to disk and can be reloaded."""
        test_file = tmp_path / "test.py"
        test_file.write_text("print('hello')")

        # First instance
        cache1 = FileChangeCache(tmp_path)
        cache1.update_file(str(test_file))

        # Second instance should load from disk
        cache2 = FileChangeCache(tmp_path)
        tracked = cache2.get_tracked_files()

        assert str(test_file) in tracked

    def test_detect_new_file(self, tmp_path: Path):
        """Detects files that are not in the cache."""
        cache = FileChangeCache(tmp_path)

        # Track one file
        file1 = tmp_path / "file1.py"
        file1.write_text("# file 1")
        cache.update_file(str(file1))

        # Create another file without tracking
        file2 = tmp_path / "file2.py"
        file2.write_text("# file 2")

        # Check both files
        changes = cache.get_changes([str(file1), str(file2)])

        assert str(file1) not in changes.new
        assert str(file2) in changes.new
        assert str(file1) not in changes.modified
        assert str(file2) not in changes.modified

    def test_detect_modified_file(self, tmp_path: Path):
        """Detects files whose mtime has changed."""
        cache = FileChangeCache(tmp_path)

        test_file = tmp_path / "test.py"
        test_file.write_text("# version 1")
        cache.update_file(str(test_file))

        # Modify the file (need to ensure mtime changes)
        import time

        time.sleep(0.01)  # Ensure mtime differs
        test_file.write_text("# version 2")

        changes = cache.get_changes([str(test_file)])

        assert str(test_file) in changes.modified
        assert str(test_file) not in changes.new

    def test_detect_deleted_file(self, tmp_path: Path):
        """Detects files that were tracked but no longer exist."""
        cache = FileChangeCache(tmp_path)

        test_file = tmp_path / "test.py"
        test_file.write_text("# to be deleted")
        cache.update_file(str(test_file))

        # Delete the file
        test_file.unlink()

        changes = cache.get_changes([])  # Empty current files list

        assert str(test_file) in changes.deleted

    def test_unchanged_file_not_in_changes(self, tmp_path: Path):
        """Unchanged files are not reported in changes."""
        cache = FileChangeCache(tmp_path)

        test_file = tmp_path / "test.py"
        test_file.write_text("# unchanged")
        cache.update_file(str(test_file))

        changes = cache.get_changes([str(test_file)])

        assert str(test_file) not in changes.new
        assert str(test_file) not in changes.modified
        assert str(test_file) not in changes.deleted

    def test_remove_file_from_cache(self, tmp_path: Path):
        """Can remove a file from tracking."""
        cache = FileChangeCache(tmp_path)

        test_file = tmp_path / "test.py"
        test_file.write_text("# test")
        cache.update_file(str(test_file))

        cache.remove_file(str(test_file))

        assert str(test_file) not in cache.get_tracked_files()

    def test_remove_files_batch(self, tmp_path: Path):
        """Can remove multiple files from tracking."""
        cache = FileChangeCache(tmp_path)

        files = []
        for i in range(3):
            f = tmp_path / f"file{i}.py"
            f.write_text(f"# file {i}")
            files.append(str(f))

        cache.update_files(files)
        cache.remove_files(files[:2])  # Remove first two

        tracked = cache.get_tracked_files()
        assert files[0] not in tracked
        assert files[1] not in tracked
        assert files[2] in tracked

    def test_clear_cache(self, tmp_path: Path):
        """Can clear all tracked files."""
        cache = FileChangeCache(tmp_path)

        for i in range(3):
            f = tmp_path / f"file{i}.py"
            f.write_text(f"# file {i}")
            cache.update_file(str(f))

        cache.clear()

        assert cache.get_tracked_files() == []

    def test_has_changes_true_when_stale(self, tmp_path: Path):
        """has_changes returns True when files are stale."""
        cache = FileChangeCache(tmp_path)

        test_file = tmp_path / "test.py"
        test_file.write_text("# version 1")
        cache.update_file(str(test_file))

        import time

        time.sleep(0.01)
        test_file.write_text("# version 2")

        assert cache.has_changes([str(test_file)]) is True

    def test_has_changes_false_when_fresh(self, tmp_path: Path):
        """has_changes returns False when no files changed."""
        cache = FileChangeCache(tmp_path)

        test_file = tmp_path / "test.py"
        test_file.write_text("# unchanged")
        cache.update_file(str(test_file))

        assert cache.has_changes([str(test_file)]) is False

    def test_has_changes_true_when_new_files(self, tmp_path: Path):
        """has_changes returns True when new files exist."""
        cache = FileChangeCache(tmp_path)

        file1 = tmp_path / "file1.py"
        file1.write_text("# tracked")
        cache.update_file(str(file1))

        file2 = tmp_path / "file2.py"
        file2.write_text("# new")

        assert cache.has_changes([str(file1), str(file2)]) is True

    def test_get_stale_files_convenience_method(self, tmp_path: Path):
        """get_stale_files returns list of files needing re-index."""
        cache = FileChangeCache(tmp_path)

        # Tracked and unchanged
        file1 = tmp_path / "unchanged.py"
        file1.write_text("# unchanged")
        cache.update_file(str(file1))

        # Tracked but modified
        file2 = tmp_path / "modified.py"
        file2.write_text("# version 1")
        cache.update_file(str(file2))
        import time

        time.sleep(0.01)
        file2.write_text("# version 2")

        # New file
        file3 = tmp_path / "new.py"
        file3.write_text("# new")

        stale = cache.get_stale_files([str(file1), str(file2), str(file3)])

        assert str(file1) not in stale
        assert str(file2) in stale
        assert str(file3) in stale

    def test_cache_handles_missing_directory(self, tmp_path: Path):
        """Cache creates directory if it doesn't exist."""
        cache_dir = tmp_path / "subdir" / "cache"
        cache = FileChangeCache(cache_dir)

        test_file = tmp_path / "test.py"
        test_file.write_text("# test")
        cache.update_file(str(test_file))

        assert cache_dir.exists()
        assert (cache_dir / "file_mtimes.json").exists()

    def test_cache_handles_corrupted_file(self, tmp_path: Path):
        """Cache handles corrupted JSON file gracefully."""
        cache_file = tmp_path / "file_mtimes.json"
        cache_file.write_text("not valid json {{{")

        # Should not raise, starts with empty cache
        cache = FileChangeCache(tmp_path)
        assert cache.get_tracked_files() == []
