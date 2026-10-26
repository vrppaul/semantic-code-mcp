"""Tests for chunk ids: the key a chunk keeps while the file around it changes."""

from pathlib import Path

from semantic_code_mcp.chunkers.python import PythonChunker
from semantic_code_mcp.domain.models import Chunk

MODULE = """def unchanged():
    return 1


def edited():
    return 2
"""
# The same module with a line inserted above both functions and the second one edited
MODULE_EDITED = """import os


def unchanged():
    return 1


def edited():
    return 3
"""
TWO_IDENTICAL_METHODS = """class First:
    def describe(self):
        return type(self).__name__


class Second:
    def describe(self):
        return type(self).__name__
"""


def ids_by_name(chunks: list[Chunk]) -> dict[str, str]:
    return {chunk.name: chunk.chunk_id for chunk in chunks}


def test_a_chunk_keeps_its_id_when_lines_are_inserted_above_it():
    before = ids_by_name(PythonChunker().chunk_string(MODULE, Path("module.py")))

    after = ids_by_name(PythonChunker().chunk_string(MODULE_EDITED, Path("module.py")))

    assert after["unchanged"] == before["unchanged"]


def test_an_edited_chunk_gets_a_new_id():
    before = ids_by_name(PythonChunker().chunk_string(MODULE, Path("module.py")))

    after = ids_by_name(PythonChunker().chunk_string(MODULE_EDITED, Path("module.py")))

    assert after["edited"] != before["edited"]


def test_identical_chunks_in_one_file_get_ids_of_their_own():
    chunks = PythonChunker().chunk_string(TWO_IDENTICAL_METHODS, Path("module.py"))

    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)


def test_the_same_chunk_in_another_file_has_another_id():
    here = PythonChunker().chunk_string(MODULE, Path("here.py"))

    there = PythonChunker().chunk_string(MODULE, Path("there.py"))

    assert {chunk.chunk_id for chunk in here}.isdisjoint(chunk.chunk_id for chunk in there)
