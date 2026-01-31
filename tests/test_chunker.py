"""Tests for tree-sitter based code chunker."""

from pathlib import Path

from semantic_code_mcp.chunkers.python import PythonChunker
from semantic_code_mcp.models import ChunkType


class TestPythonChunker:
    """Tests for PythonChunker."""

    def test_extract_simple_function(self, tmp_path: Path):
        """Extracts a simple function definition."""
        code = """def hello():
    print("hello")
"""
        file_path = tmp_path / "test.py"
        file_path.write_text(code)

        chunker = PythonChunker()
        chunks = chunker.chunk_file(str(file_path))

        assert len(chunks) == 1
        assert chunks[0].name == "hello"
        assert chunks[0].chunk_type == ChunkType.function
        assert "def hello():" in chunks[0].content
        assert 'print("hello")' in chunks[0].content

    def test_extract_function_with_docstring(self, tmp_path: Path):
        """Function extraction includes docstring."""
        code = '''def greet(name: str) -> str:
    """Greet someone by name.

    Args:
        name: The name to greet.

    Returns:
        A greeting message.
    """
    return f"Hello, {name}!"
'''
        file_path = tmp_path / "test.py"
        file_path.write_text(code)

        chunker = PythonChunker()
        chunks = chunker.chunk_file(str(file_path))

        assert len(chunks) == 1
        assert "Greet someone by name" in chunks[0].content
        assert "Args:" in chunks[0].content

    def test_extract_class(self, tmp_path: Path):
        """Extracts a class definition."""
        code = '''class Person:
    """A person with a name."""

    def __init__(self, name: str):
        self.name = name
'''
        file_path = tmp_path / "test.py"
        file_path.write_text(code)

        chunker = PythonChunker()
        chunks = chunker.chunk_file(str(file_path))

        # Should get class and its method
        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.klass]
        method_chunks = [c for c in chunks if c.chunk_type == ChunkType.method]

        assert len(class_chunks) == 1
        assert class_chunks[0].name == "Person"
        assert "A person with a name" in class_chunks[0].content

        assert len(method_chunks) == 1
        assert method_chunks[0].name == "__init__"

    def test_extract_method(self, tmp_path: Path):
        """Extracts methods from a class."""
        code = '''class Calculator:
    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    def subtract(self, a: int, b: int) -> int:
        """Subtract b from a."""
        return a - b
'''
        file_path = tmp_path / "test.py"
        file_path.write_text(code)

        chunker = PythonChunker()
        chunks = chunker.chunk_file(str(file_path))

        method_chunks = [c for c in chunks if c.chunk_type == ChunkType.method]
        assert len(method_chunks) == 2

        names = {c.name for c in method_chunks}
        assert "add" in names
        assert "subtract" in names

    def test_line_numbers_are_correct(self, tmp_path: Path):
        """Line numbers accurately reflect position in file."""
        code = """# Comment at top

def first():
    pass


def second():
    pass
"""
        file_path = tmp_path / "test.py"
        file_path.write_text(code)

        chunker = PythonChunker()
        chunks = chunker.chunk_file(str(file_path))

        first_fn = next(c for c in chunks if c.name == "first")
        second_fn = next(c for c in chunks if c.name == "second")

        # Line numbers are 1-indexed
        assert first_fn.line_start == 3
        assert first_fn.line_end == 4
        assert second_fn.line_start == 7
        assert second_fn.line_end == 8

    def test_file_path_is_set(self, tmp_path: Path):
        """Chunks have correct file path."""
        file_path = tmp_path / "mymodule.py"
        file_path.write_text("def foo(): pass")

        chunker = PythonChunker()
        chunks = chunker.chunk_file(str(file_path))

        assert chunks[0].file_path == str(file_path)

    def test_empty_file_returns_no_chunks(self, tmp_path: Path):
        """Empty file produces no chunks."""
        file_path = tmp_path / "empty.py"
        file_path.write_text("")

        chunker = PythonChunker()
        chunks = chunker.chunk_file(str(file_path))

        assert chunks == []

    def test_file_with_only_comments(self, tmp_path: Path):
        """File with only comments produces no chunks."""
        code = """# This is a comment
# Another comment
"""
        file_path = tmp_path / "comments.py"
        file_path.write_text(code)

        chunker = PythonChunker()
        chunks = chunker.chunk_file(str(file_path))

        assert chunks == []

    def test_multiple_functions(self, tmp_path: Path):
        """Extracts multiple functions from same file."""
        code = """def one():
    pass

def two():
    pass

def three():
    pass
"""
        file_path = tmp_path / "test.py"
        file_path.write_text(code)

        chunker = PythonChunker()
        chunks = chunker.chunk_file(str(file_path))

        assert len(chunks) == 3
        names = {c.name for c in chunks}
        assert names == {"one", "two", "three"}

    def test_nested_function_is_included_in_parent(self, tmp_path: Path):
        """Nested functions are part of parent function content."""
        code = """def outer():
    def inner():
        pass
    inner()
"""
        file_path = tmp_path / "test.py"
        file_path.write_text(code)

        chunker = PythonChunker()
        chunks = chunker.chunk_file(str(file_path))

        # We only extract top-level functions, not nested ones
        assert len(chunks) == 1
        assert chunks[0].name == "outer"
        assert "def inner():" in chunks[0].content

    def test_async_function(self, tmp_path: Path):
        """Extracts async function definitions."""
        code = '''async def fetch_data(url: str) -> dict:
    """Fetch data from URL."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()
'''
        file_path = tmp_path / "test.py"
        file_path.write_text(code)

        chunker = PythonChunker()
        chunks = chunker.chunk_file(str(file_path))

        assert len(chunks) == 1
        assert chunks[0].name == "fetch_data"
        assert chunks[0].chunk_type == ChunkType.function

    def test_decorated_function(self, tmp_path: Path):
        """Extracts functions with decorators, including decorator in content."""
        code = '''@app.route("/api/users")
@require_auth
def get_users():
    """Get all users."""
    return users
'''
        file_path = tmp_path / "test.py"
        file_path.write_text(code)

        chunker = PythonChunker()
        chunks = chunker.chunk_file(str(file_path))

        assert len(chunks) == 1
        assert chunks[0].name == "get_users"
        assert "@app.route" in chunks[0].content
        assert "@require_auth" in chunks[0].content

    def test_decorated_class(self, tmp_path: Path):
        """Extracts classes with decorators."""
        code = """@dataclass
class User:
    name: str
    age: int
"""
        file_path = tmp_path / "test.py"
        file_path.write_text(code)

        chunker = PythonChunker()
        chunks = chunker.chunk_file(str(file_path))

        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.klass]
        assert len(class_chunks) == 1
        assert "@dataclass" in class_chunks[0].content

    def test_handles_syntax_error_gracefully(self, tmp_path: Path):
        """Returns empty list for files with syntax errors."""
        code = """def broken(
    # missing closing paren and colon
"""
        file_path = tmp_path / "broken.py"
        file_path.write_text(code)

        chunker = PythonChunker()
        # Should not raise, returns empty or partial results
        chunks = chunker.chunk_file(str(file_path))
        # We accept either empty list or best-effort extraction
        assert isinstance(chunks, list)

    def test_class_with_class_variables(self, tmp_path: Path):
        """Class extraction includes class variables."""
        code = '''class Config:
    """Application configuration."""

    DEBUG = True
    VERSION = "1.0.0"

    def __init__(self):
        pass
'''
        file_path = tmp_path / "test.py"
        file_path.write_text(code)

        chunker = PythonChunker()
        chunks = chunker.chunk_file(str(file_path))

        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.klass]
        assert len(class_chunks) == 1
        assert "DEBUG = True" in class_chunks[0].content
        assert "VERSION" in class_chunks[0].content

    def test_staticmethod_and_classmethod(self, tmp_path: Path):
        """Extracts static and class methods."""
        code = """class Factory:
    @staticmethod
    def create():
        pass

    @classmethod
    def from_dict(cls, data):
        pass
"""
        file_path = tmp_path / "test.py"
        file_path.write_text(code)

        chunker = PythonChunker()
        chunks = chunker.chunk_file(str(file_path))

        method_chunks = [c for c in chunks if c.chunk_type == ChunkType.method]
        assert len(method_chunks) == 2

        names = {c.name for c in method_chunks}
        assert "create" in names
        assert "from_dict" in names

    def test_chunk_from_string(self):
        """Can chunk code from string without file."""
        code = """def hello():
    pass
"""
        chunker = PythonChunker()
        chunks = chunker.chunk_string(code, file_path="<string>")

        assert len(chunks) == 1
        assert chunks[0].name == "hello"
        assert chunks[0].file_path == "<string>"

    # --- Module docstring tests ---

    def test_module_docstring_with_function(self):
        """File with module docstring + function produces MODULE chunk + FUNCTION chunk."""
        code = '''"""This module does important things."""

def hello():
    pass
'''
        chunker = PythonChunker()
        chunks = chunker.chunk_string(code, file_path="important.py")

        module_chunks = [c for c in chunks if c.chunk_type == ChunkType.module]
        func_chunks = [c for c in chunks if c.chunk_type == ChunkType.function]

        assert len(module_chunks) == 1
        assert len(func_chunks) == 1
        assert module_chunks[0].name == "important"
        assert "This module does important things" in module_chunks[0].content

    def test_module_docstring_multiline_line_numbers(self):
        """Multi-line module docstring has correct line numbers."""
        code = '''"""First line.

Second paragraph with more detail.
Third line.
"""

def foo():
    pass
'''
        chunker = PythonChunker()
        chunks = chunker.chunk_string(code, file_path="multi.py")

        module_chunk = next(c for c in chunks if c.chunk_type == ChunkType.module)
        assert module_chunk.line_start == 1
        assert module_chunk.line_end == 5

    def test_no_module_docstring_starts_with_import(self):
        """File starting with import produces zero MODULE chunks."""
        code = """import os

def hello():
    pass
"""
        chunker = PythonChunker()
        chunks = chunker.chunk_string(code, file_path="no_doc.py")

        module_chunks = [c for c in chunks if c.chunk_type == ChunkType.module]
        assert len(module_chunks) == 0

    def test_docstring_after_import_not_extracted(self):
        """Docstring after import is not a module docstring (PEP 257)."""
        code = '''import os

"""This is not a module docstring."""

def hello():
    pass
'''
        chunker = PythonChunker()
        chunks = chunker.chunk_string(code, file_path="after_import.py")

        module_chunks = [c for c in chunks if c.chunk_type == ChunkType.module]
        assert len(module_chunks) == 0

    def test_file_with_only_docstring(self):
        """File with only a module docstring produces one MODULE chunk."""
        code = '''"""Just a docstring, nothing else."""
'''
        chunker = PythonChunker()
        chunks = chunker.chunk_string(code, file_path="only_doc.py")

        assert len(chunks) == 1
        assert chunks[0].chunk_type == ChunkType.module
        assert chunks[0].name == "only_doc"

    def test_module_docstring_name_from_file_stem(self):
        """Module chunk name is derived from file stem."""
        code = '''"""Docstring."""
'''
        chunker = PythonChunker()
        chunks = chunker.chunk_string(code, file_path="/some/path/my_module.py")

        module_chunk = next(c for c in chunks if c.chunk_type == ChunkType.module)
        assert module_chunk.name == "my_module"

    def test_comments_before_docstring_still_extracted(self):
        """Comments before docstring don't prevent extraction."""
        code = '''# Copyright 2024
# License: MIT

"""Module docstring after comments."""

def foo():
    pass
'''
        chunker = PythonChunker()
        chunks = chunker.chunk_string(code, file_path="commented.py")

        module_chunks = [c for c in chunks if c.chunk_type == ChunkType.module]
        assert len(module_chunks) == 1
        assert "Module docstring after comments" in module_chunks[0].content

    def test_property_decorator(self, tmp_path: Path):
        """Extracts property methods."""
        code = '''class User:
    @property
    def full_name(self) -> str:
        """Get the user's full name."""
        return f"{self.first} {self.last}"

    @full_name.setter
    def full_name(self, value: str):
        self.first, self.last = value.split()
'''
        file_path = tmp_path / "test.py"
        file_path.write_text(code)

        chunker = PythonChunker()
        chunks = chunker.chunk_file(str(file_path))

        method_chunks = [c for c in chunks if c.chunk_type == ChunkType.method]
        # Both getter and setter are methods named "full_name"
        assert len(method_chunks) == 2
        assert all(c.name == "full_name" for c in method_chunks)
