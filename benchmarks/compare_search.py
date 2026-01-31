#!/usr/bin/env python3
"""Benchmark comparing semantic search vs traditional grep+read approach.

Measures:
- Token usage (estimated)
- Time to find relevant code
- Number of tool calls needed
"""

import subprocess  # nosec B404 - needed to simulate grep for benchmarking
import time
from dataclasses import dataclass
from pathlib import Path

from semantic_code_mcp.config import Settings
from semantic_code_mcp.container import Container


@dataclass
class SearchResult:
    """Result of a search benchmark."""

    query: str
    approach: str
    time_seconds: float
    tool_calls: int
    tokens_in_results: int
    files_touched: int
    snippets_returned: int


def estimate_tokens(text: str) -> int:
    """Rough token estimate (1 token ≈ 4 chars)."""
    return len(text) // 4


def benchmark_semantic_search(
    project_path: Path,
    query: str,
    limit: int = 10,
) -> SearchResult:
    """Benchmark semantic search approach."""
    settings = Settings()
    container = Container(settings)
    search_service = container.create_search_service(project_path)

    import asyncio

    start = time.perf_counter()
    outcome = asyncio.run(search_service.search(query, project_path, limit=limit))
    elapsed = time.perf_counter() - start
    results = outcome.results

    # Count tokens in returned snippets
    total_content = "\n".join(r.content for r in results)
    tokens = estimate_tokens(total_content)

    # Unique files in results
    files = {r.file_path for r in results}

    return SearchResult(
        query=query,
        approach="semantic",
        time_seconds=elapsed,
        tool_calls=1,  # Single search call
        tokens_in_results=tokens,
        files_touched=len(files),
        snippets_returned=len(results),
    )


def benchmark_grep_approach(
    project_path: Path,
    query: str,
    keywords: list[str],
) -> SearchResult:
    """Benchmark traditional grep+read approach.

    Simulates what Claude Code would do:
    1. Grep for keywords
    2. Read matching files
    """
    start = time.perf_counter()
    tool_calls = 0
    total_tokens = 0
    files_read = set()

    for keyword in keywords:
        # Simulate grep call
        tool_calls += 1
        try:
            result = subprocess.run(  # nosec B603, B607
                ["grep", "-r", "-l", keyword, str(project_path)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            matching_files = [f for f in result.stdout.strip().split("\n") if f]
        except (subprocess.TimeoutExpired, OSError):
            matching_files = []

        # Simulate reading each matching file
        for file_path in matching_files[:5]:  # Limit to avoid explosion
            if file_path in files_read:
                continue
            files_read.add(file_path)
            tool_calls += 1

            try:
                content = Path(file_path).read_text()
                total_tokens += estimate_tokens(content)
            except OSError:
                continue  # Skip unreadable files

    elapsed = time.perf_counter() - start

    return SearchResult(
        query=query,
        approach="grep+read",
        time_seconds=elapsed,
        tool_calls=tool_calls,
        tokens_in_results=total_tokens,
        files_touched=len(files_read),
        snippets_returned=0,  # Full files, not snippets
    )


def run_benchmark(project_path: Path, queries: list[dict]) -> list[SearchResult]:
    """Run benchmark for all queries."""
    results = []

    for q in queries:
        query = q["query"]
        keywords = q["keywords"]

        print(f"\n{'=' * 60}")
        print(f"Query: {query}")
        print(f"Keywords for grep: {keywords}")
        print("=" * 60)

        # Semantic search
        semantic = benchmark_semantic_search(project_path, query)
        results.append(semantic)
        print("\nSemantic Search:")
        print(f"  Time: {semantic.time_seconds:.3f}s")
        print(f"  Tool calls: {semantic.tool_calls}")
        print(f"  Tokens: {semantic.tokens_in_results:,}")
        print(f"  Files: {semantic.files_touched}")
        print(f"  Snippets: {semantic.snippets_returned}")

        # Grep approach
        grep = benchmark_grep_approach(project_path, query, keywords)
        results.append(grep)
        print("\nGrep+Read Approach:")
        print(f"  Time: {grep.time_seconds:.3f}s")
        print(f"  Tool calls: {grep.tool_calls}")
        print(f"  Tokens: {grep.tokens_in_results:,}")
        print(f"  Files: {grep.files_touched}")

        # Comparison
        if grep.tokens_in_results > 0:
            savings = (1 - semantic.tokens_in_results / grep.tokens_in_results) * 100
            print(f"\n  Token savings: {savings:.1f}%")
            print(f"  Tool call reduction: {grep.tool_calls - semantic.tool_calls}")

    return results


def print_summary(results: list[SearchResult]) -> None:
    """Print summary statistics."""
    semantic = [r for r in results if r.approach == "semantic"]
    grep = [r for r in results if r.approach == "grep+read"]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_semantic_tokens = sum(r.tokens_in_results for r in semantic)
    total_grep_tokens = sum(r.tokens_in_results for r in grep)
    total_semantic_calls = sum(r.tool_calls for r in semantic)
    total_grep_calls = sum(r.tool_calls for r in grep)
    total_semantic_time = sum(r.time_seconds for r in semantic)
    total_grep_time = sum(r.time_seconds for r in grep)

    print(f"\nTotal across {len(semantic)} queries:")
    print("\n  Semantic Search:")
    print(f"    Tokens: {total_semantic_tokens:,}")
    print(f"    Tool calls: {total_semantic_calls}")
    print(f"    Time: {total_semantic_time:.2f}s")

    print("\n  Grep+Read:")
    print(f"    Tokens: {total_grep_tokens:,}")
    print(f"    Tool calls: {total_grep_calls}")
    print(f"    Time: {total_grep_time:.2f}s")

    if total_grep_tokens > 0:
        savings = (1 - total_semantic_tokens / total_grep_tokens) * 100
        print(f"\n  Overall token savings: {savings:.1f}%")
        print(f"  Overall tool call reduction: {total_grep_calls - total_semantic_calls}")


# Test queries with equivalent grep keywords
TEST_QUERIES = [
    {
        "query": "how does embedding generation work",
        "keywords": ["embed", "embedding", "sentence_transformers"],
    },
    {
        "query": "file change detection for incremental indexing",
        "keywords": ["mtime", "cache", "change", "stale"],
    },
    {
        "query": "AST parsing and code chunking",
        "keywords": ["tree_sitter", "chunk", "parse", "AST"],
    },
    {
        "query": "vector similarity search implementation",
        "keywords": ["search", "vector", "similarity", "cosine"],
    },
    {
        "query": "MCP server tool definitions",
        "keywords": ["@mcp.tool", "FastMCP", "Context"],
    },
]


if __name__ == "__main__":
    import sys

    project = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    print(f"Benchmarking on: {project.resolve()}")

    results = run_benchmark(project, TEST_QUERIES)
    print_summary(results)
