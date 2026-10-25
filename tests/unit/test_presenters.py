"""Tests for what the tools answer, made from what the services report."""

from semantic_code_mcp.domain.models import ChunkType, SearchReport, SearchResult, SearchTimings
from semantic_code_mcp.server.presenters import MAX_RESULT_LINES, present_hit, present_search

TIMINGS = SearchTimings(embedding_ms=1.0, search_ms=2.0, total_ms=3.0)


def hit_of(content: str) -> SearchResult:
    return SearchResult(
        file_path="/project/module.py",
        line_start=1,
        line_end=content.count("\n") + 1,
        content=content,
        chunk_type=ChunkType.function,
        name="function",
        score=0.5,
    )


def test_a_long_hit_is_cut_to_the_line_limit_and_says_so():
    long_content = "\n".join(f"line_{number} = {number}" for number in range(MAX_RESULT_LINES + 10))

    view = present_hit(hit_of(long_content))

    assert view.truncated
    assert (
        view.content.split("\n")[:MAX_RESULT_LINES] == long_content.split("\n")[:MAX_RESULT_LINES]
    )


def test_a_short_hit_is_whole():
    view = present_hit(hit_of("def function():\n    return 1"))

    assert (view.truncated, view.content) == (False, "def function():\n    return 1")


def test_debug_details_only_when_asked():
    report = SearchReport(
        results=[hit_of("x = 1")], filtered_count=0, timings=TIMINGS, index_report=None
    )

    assert present_search(report, include_debug=False).debug is None
    assert present_search(report, include_debug=True).debug is not None
