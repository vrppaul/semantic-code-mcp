"""Search result ranking and boosting logic."""

import re
import time

# Pattern for extracting words from queries
WORD_PATTERN = re.compile(r"\w+")

# Time constants
ONE_WEEK_SECONDS = 7 * 24 * 60 * 60


def extract_query_words(query: str) -> set[str]:
    """Extract searchable words from a query string.

    Args:
        query: The search query.

    Returns:
        Set of lowercase words (>2 chars) from the query.
    """
    words = WORD_PATTERN.findall(query.lower())
    return {w for w in words if len(w) > 2}


def calculate_boost(
    content: str,
    query_words: set[str],
    mtime: float | None,
) -> float:
    """Calculate score boost for a search result.

    Combines keyword matching and recency boosting.

    Args:
        content: The code content to check for matches.
        query_words: Set of query words to look for.
        mtime: File modification time (Unix timestamp), or None.

    Returns:
        Boost value to add to the base similarity score (0.0 to ~0.5).
    """
    content_lower = content.lower()
    boost = 0.0

    if query_words:
        # Check for exact identifier matches (stronger signal)
        exact_matches = 0
        partial_matches = 0

        for word in query_words:
            if word in content_lower:
                # Check if it's an exact identifier match (word boundaries)
                # Use regex to find whole-word matches
                pattern = rf"\b{re.escape(word)}\b"
                if re.search(pattern, content_lower):
                    exact_matches += 1
                else:
                    partial_matches += 1

        # Exact identifier matches get strong boost (+15% each, up to 45%)
        exact_boost = min(0.45, exact_matches * 0.15)

        # Partial matches get smaller boost (+3% each, up to 15%)
        partial_boost = min(0.15, partial_matches * 0.03)

        boost += exact_boost + partial_boost

    # Recency boost: up to 5% for files modified in last week
    if mtime is not None:
        now = time.time()
        age_seconds = now - mtime
        if age_seconds < ONE_WEEK_SECONDS:
            # Linear decay: 5% for just modified, 0% for 1 week old
            recency_boost = 0.05 * (1 - age_seconds / ONE_WEEK_SECONDS)
            boost += recency_boost

    # Cap total boost
    return min(0.5, boost)
