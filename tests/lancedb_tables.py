"""What the LanceDB store's tests read of its tables in an index's directory: its chunks
table's fragments, and its full-text index's rows and parts."""

import lancedb
from lance.fragment import LanceFragment

from semantic_code_mcp.storage.index_directories import IndexDirectory
from semantic_code_mcp.storage.lancedb import CHUNKS_TABLE, FULL_TEXT_INDEX_NAME

# Where LanceDB keeps each version of every index of a table, a directory each: while the
# chunks table has no index but the full-text one, the newest is the full-text index's
TABLE_INDEX_VERSIONS = "_indices"


def chunks_table(directory: IndexDirectory) -> lancedb.table.Table:
    """The chunks table of the index in the directory, at its latest version."""
    return lancedb.connect(str(directory.path)).open_table(CHUNKS_TABLE)


def chunks_outside_the_full_text_index(directory: IndexDirectory) -> int:
    """How many chunks of an index its full-text index does not hold yet."""
    return chunks_table(directory).index_stats(FULL_TEXT_INDEX_NAME).num_unindexed_rows


def chunks_fragments(directory: IndexDirectory) -> list[LanceFragment]:
    """The fragments the latest version of an index's chunks table reads, in order."""
    return chunks_table(directory).to_lance().get_fragments()


def full_text_index_documents(directory: IndexDirectory) -> int:
    """How many rows an index's full-text index counts in its word statistics: those the
    table holds, and those writes replaced, until a rebuild drops them."""
    statistics = chunks_table(directory).to_lance().stats.index_stats(FULL_TEXT_INDEX_NAME)
    return sum(part["num_docs"] for part in statistics["indices"])


def newest_full_text_index_files(directory: IndexDirectory) -> int:
    """How many files the newest version of an index's full-text index holds: each part an
    upkeep folds in adds some, and a rebuild starts again from one part."""
    versions = directory.path / f"{CHUNKS_TABLE}.lance" / TABLE_INDEX_VERSIONS
    newest = max(versions.iterdir(), key=lambda version: version.stat().st_mtime_ns)
    return len(list(newest.rglob("*")))
