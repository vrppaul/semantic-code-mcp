"""Tests for tree-sitter based code chunker."""

from pathlib import Path

from semantic_code_mcp.chunkers.base import FALLBACK_WINDOW_LINES, MAX_SOURCE_BYTES
from semantic_code_mcp.chunkers.python import PythonChunker
from semantic_code_mcp.domain.models import ChunkKind
from tests.helpers import (
    ChunkOwner,
    ChunkSummary,
    LineRange,
    class_named,
    function_named,
    line_ranges,
    method_named,
    module_named,
    owners_of,
    summarize,
)


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
        chunks = chunker.chunk_file(file_path.parent, Path(file_path.name))

        assert len(chunks) == 1
        assert chunks[0].name == "hello"
        assert chunks[0].kind == ChunkKind.function
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
        chunks = chunker.chunk_file(file_path.parent, Path(file_path.name))

        assert len(chunks) == 1
        assert "Greet someone by name" in chunks[0].content
        assert "Args:" in chunks[0].content

    def test_a_class_and_each_of_its_methods_are_chunks_of_their_own(self):
        code = '''class Person:
    """A person with a name."""

    def __init__(self, name: str):
        self.name = name
'''
        chunks = PythonChunker().chunk_string(code, Path("person.py"))

        assert summarize(chunks) == [class_named("Person"), method_named("__init__")]
        assert "A person with a name" in chunks[0].content

    def test_each_method_of_a_class_is_a_chunk_in_its_order(self):
        code = '''class Calculator:
    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    def subtract(self, a: int, b: int) -> int:
        """Subtract b from a."""
        return a - b
'''
        chunks = PythonChunker().chunk_string(code, Path("calculator.py"))

        assert summarize(chunks) == [
            class_named("Calculator"),
            method_named("add"),
            method_named("subtract"),
        ]

    def test_a_chunk_s_lines_count_from_one_and_cover_its_definition(self):
        code = """# Comment at top

def first():
    pass


def second():
    pass
"""
        chunks = PythonChunker().chunk_string(code, Path("numbered.py"))

        assert line_ranges(chunks) == [
            LineRange(line_start=3, line_end=4),
            LineRange(line_start=7, line_end=8),
        ]

    def test_chunks_carry_their_file_s_path_from_the_project_root(self, tmp_path: Path):
        (tmp_path / "package").mkdir()
        (tmp_path / "package" / "mymodule.py").write_text("def foo(): pass")

        chunks = PythonChunker().chunk_file(tmp_path, Path("package/mymodule.py"))

        assert [chunk.path for chunk in chunks] == [Path("package/mymodule.py")]

    def test_empty_file_returns_no_chunks(self, tmp_path: Path):
        """Empty file produces no chunks."""
        file_path = tmp_path / "empty.py"
        file_path.write_text("")

        chunker = PythonChunker()
        chunks = chunker.chunk_file(file_path.parent, Path(file_path.name))

        assert chunks == []

    def test_file_with_only_comments(self, tmp_path: Path):
        """File with only comments becomes one module chunk."""
        code = """# This is a comment
# Another comment
"""
        file_path = tmp_path / "comments.py"
        file_path.write_text(code)

        chunker = PythonChunker()
        chunks = chunker.chunk_file(file_path.parent, Path(file_path.name))

        assert summarize(chunks) == [ChunkSummary(kind=ChunkKind.module, name="comments")]

    def test_multiple_functions(self):
        """Extracts multiple functions from same file."""
        code = """def one():
    pass

def two():
    pass

def three():
    pass
"""
        chunks = PythonChunker().chunk_string(code, Path("several.py"))

        assert summarize(chunks) == [
            function_named("one"),
            function_named("two"),
            function_named("three"),
        ]

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
        chunks = chunker.chunk_file(file_path.parent, Path(file_path.name))

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
        chunks = chunker.chunk_file(file_path.parent, Path(file_path.name))

        assert len(chunks) == 1
        assert chunks[0].name == "fetch_data"
        assert chunks[0].kind == ChunkKind.function

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
        chunks = chunker.chunk_file(file_path.parent, Path(file_path.name))

        assert len(chunks) == 1
        assert chunks[0].name == "get_users"
        assert "@app.route" in chunks[0].content
        assert "@require_auth" in chunks[0].content

    def test_decorated_class(self):
        """A class's chunk holds its decorators."""
        code = """@dataclass
class User:
    name: str
    age: int
"""
        chunks = PythonChunker().chunk_string(code, Path("user.py"))

        assert summarize(chunks) == [class_named("User")]
        assert chunks[0].content.startswith("@dataclass")

    def test_a_file_in_which_no_definition_parses_is_indexed_whole(self):
        code = """def broken(
    # missing closing paren and colon
"""
        chunks = PythonChunker().chunk_string(code, Path("broken.py"))

        assert summarize(chunks) == [module_named("broken")]

    def test_class_with_class_variables(self):
        """Class extraction includes class variables."""
        code = '''class Config:
    """Application configuration."""

    DEBUG = True
    VERSION = "1.0.0"

    def __init__(self):
        pass
'''
        chunks = PythonChunker().chunk_string(code, Path("config.py"))

        assert summarize(chunks) == [class_named("Config"), method_named("__init__")]
        assert 'DEBUG = True\n    VERSION = "1.0.0"' in chunks[0].content

    def test_staticmethod_and_classmethod(self):
        """Extracts static and class methods."""
        code = """class Factory:
    @staticmethod
    def create():
        pass

    @classmethod
    def from_dict(cls, data):
        pass
"""
        chunks = PythonChunker().chunk_string(code, Path("factory.py"))

        assert summarize(chunks) == [
            class_named("Factory"),
            method_named("create"),
            method_named("from_dict"),
        ]

    def test_chunk_from_string(self):
        """Can chunk code from string without file."""
        code = """def hello():
    pass
"""
        chunker = PythonChunker()
        chunks = chunker.chunk_string(code, Path("<string>"))

        assert len(chunks) == 1
        assert chunks[0].name == "hello"
        assert chunks[0].path == Path("<string>")

    # --- Module docstring tests ---

    def test_module_docstring_with_function(self):
        """File with module docstring + function produces MODULE chunk + FUNCTION chunk."""
        code = '''"""This module does important things."""

def hello():
    pass
'''
        chunks = PythonChunker().chunk_string(code, Path("important.py"))

        assert summarize(chunks) == [module_named("important"), function_named("hello")]
        assert chunks[0].content == '"""This module does important things."""'

    def test_module_docstring_multiline_line_numbers(self):
        """Multi-line module docstring has correct line numbers."""
        code = '''"""First line.

Second paragraph with more detail.
Third line.
"""

def foo():
    pass
'''
        chunks = PythonChunker().chunk_string(code, Path("multi.py"))

        assert line_ranges(chunks) == [
            LineRange(line_start=1, line_end=5),
            LineRange(line_start=7, line_end=8),
        ]

    def test_no_module_docstring_starts_with_import(self):
        """File starting with import produces zero MODULE chunks."""
        code = """import os

def hello():
    pass
"""
        chunks = PythonChunker().chunk_string(code, Path("no_doc.py"))

        assert summarize(chunks) == [function_named("hello")]

    def test_docstring_after_import_not_extracted(self):
        """Docstring after import is not a module docstring (PEP 257)."""
        code = '''import os

"""This is not a module docstring."""

def hello():
    pass
'''
        chunks = PythonChunker().chunk_string(code, Path("after_import.py"))

        assert summarize(chunks) == [function_named("hello")]

    def test_file_with_only_docstring(self):
        """File with only a module docstring produces one MODULE chunk."""
        code = '''"""Just a docstring, nothing else."""
'''
        chunker = PythonChunker()
        chunks = chunker.chunk_string(code, Path("only_doc.py"))

        assert len(chunks) == 1
        assert chunks[0].kind == ChunkKind.module
        assert chunks[0].name == "only_doc"

    def test_module_docstring_name_from_file_stem(self):
        """Module chunk name is derived from file stem."""
        code = '''"""Docstring."""
'''
        chunks = PythonChunker().chunk_string(code, Path("some/path/my_module.py"))

        assert summarize(chunks) == [module_named("my_module")]

    def test_comments_before_docstring_still_extracted(self):
        """Comments before docstring don't prevent extraction."""
        code = '''# Copyright 2024
# License: MIT

"""Module docstring after comments."""

def foo():
    pass
'''
        chunks = PythonChunker().chunk_string(code, Path("commented.py"))

        assert summarize(chunks) == [module_named("commented"), function_named("foo")]
        assert chunks[0].content == '"""Module docstring after comments."""'

    def test_property_decorator(self):
        """A property's getter and setter are methods of one name."""
        code = '''class User:
    @property
    def full_name(self) -> str:
        """Get the user's full name."""
        return f"{self.first} {self.last}"

    @full_name.setter
    def full_name(self, value: str):
        self.first, self.last = value.split()
'''
        chunks = PythonChunker().chunk_string(code, Path("user.py"))

        assert summarize(chunks) == [
            class_named("User"),
            method_named("full_name"),
            method_named("full_name"),
        ]


class TestPythonChunkerSilentDrops:
    """Code that used to vanish from the index without an error."""

    def test_comment_between_decorator_and_function(self):
        code = """import functools


@functools.cache
# cached for the whole session
def compute(value):
    return value * 2
"""
        chunks = PythonChunker().chunk_string(code, Path("sample.py"))

        assert summarize(chunks) == [ChunkSummary(kind=ChunkKind.function, name="compute")]
        assert chunks[0].line_start == 4
        assert chunks[0].content.startswith("@functools.cache")

    def test_comment_between_decorator_and_class(self):
        code = """@dataclass
# frozen on purpose
class Config:
    def load(self):
        return {}
"""
        chunks = PythonChunker().chunk_string(code, Path("sample.py"))

        assert summarize(chunks) == [
            ChunkSummary(kind=ChunkKind.klass, name="Config"),
            ChunkSummary(kind=ChunkKind.method, name="load"),
        ]

    def test_file_without_definitions_becomes_one_module_chunk(self):
        code = """import os

DEBUG = os.environ.get("DEBUG") == "1"
DATABASES = {"default": {"ENGINE": "sqlite"}}
"""
        chunks = PythonChunker().chunk_string(code, Path("project/settings.py"))

        assert summarize(chunks) == [ChunkSummary(kind=ChunkKind.module, name="settings")]
        assert line_ranges(chunks) == [LineRange(line_start=1, line_end=4)]
        assert chunks[0].content == code.rstrip("\n")

    def test_file_with_a_docstring_and_no_definitions_is_indexed_whole(self):
        code = '''"""Project settings."""

import os

DEBUG = os.environ.get("DEBUG") == "1"
'''
        chunks = PythonChunker().chunk_string(code, Path("project/settings.py"))

        assert summarize(chunks) == [ChunkSummary(kind=ChunkKind.module, name="settings")]
        assert line_ranges(chunks) == [LineRange(line_start=1, line_end=5)]

    def test_long_file_without_definitions_is_split_into_windows(self):
        line_count = 2 * FALLBACK_WINDOW_LINES + 1
        code = "\n".join(f"SETTING_{number} = {number}" for number in range(line_count)) + "\n"

        chunks = PythonChunker().chunk_string(code, Path("generated.py"))

        assert line_ranges(chunks) == [
            LineRange(line_start=1, line_end=FALLBACK_WINDOW_LINES),
            LineRange(line_start=FALLBACK_WINDOW_LINES + 1, line_end=2 * FALLBACK_WINDOW_LINES),
            LineRange(line_start=line_count, line_end=line_count),
        ]

    def test_undecodable_bytes_do_not_stop_chunking(self, tmp_path: Path):
        file_path = tmp_path / "latin1.py"
        file_path.write_bytes(b"# caf\xe9 au lait\ndef order():\n    return 1\n")

        chunks = PythonChunker().chunk_file(file_path.parent, Path(file_path.name))

        assert [chunk.name for chunk in chunks] == ["order"]

    def test_binary_file_is_skipped(self, tmp_path: Path):
        file_path = tmp_path / "utf16.py"
        file_path.write_bytes("def order():\n    return 1\n".encode("utf-16"))

        assert PythonChunker().chunk_file(file_path.parent, Path(file_path.name)) == []

    def test_file_over_the_size_limit_is_skipped(self, tmp_path: Path):
        file_path = tmp_path / "huge.py"
        file_path.write_text("x = 1\n" * (MAX_SOURCE_BYTES // len("x = 1\n") + 1))

        assert PythonChunker().chunk_file(file_path.parent, Path(file_path.name)) == []


def test_a_definition_names_the_classes_it_is_defined_in():
    code = """class Outer:
    def method(self):
        pass

    class Inner:
        def deep(self):
            pass


def top():
    pass
"""

    chunks = PythonChunker().chunk_string(code, Path("module.py"))

    assert owners_of(chunks) == [
        ChunkOwner(name="Outer", parent=None),
        ChunkOwner(name="method", parent="Outer"),
        ChunkOwner(name="Inner", parent="Outer"),
        ChunkOwner(name="deep", parent="Outer.Inner"),
        ChunkOwner(name="top", parent=None),
    ]
