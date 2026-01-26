"""Tests for search result ranking and boosting."""

from semantic_code_mcp.search.ranking import calculate_boost, extract_query_words


class TestExtractQueryWords:
    """Tests for query word extraction."""

    def test_extracts_words_from_query(self):
        """Extracts alphanumeric words from query."""
        words = extract_query_words("find the authentication logic")
        assert "find" in words
        assert "authentication" in words
        assert "logic" in words

    def test_filters_short_words(self):
        """Words with 2 or fewer characters are filtered out."""
        words = extract_query_words("a is an or if")
        assert len(words) == 0

    def test_handles_underscores(self):
        """Underscored identifiers are extracted."""
        words = extract_query_words("find duration_ms in logs")
        assert "duration_ms" in words

    def test_lowercases_words(self):
        """Words are lowercased for matching."""
        words = extract_query_words("Find DurationMs")
        assert "find" in words
        assert "durationms" in words


class TestCalculateBoost:
    """Tests for score boost calculation."""

    def test_no_boost_when_no_matches(self):
        """No boost when content has no query words."""
        boost = calculate_boost(
            content="def hello(): pass",
            query_words={"authentication", "login"},
            mtime=None,
        )
        assert boost == 0.0

    def test_small_boost_for_single_word_match(self):
        """Single word match gives small boost."""
        boost = calculate_boost(
            content="def login(): pass",
            query_words={"authentication", "login"},
            mtime=None,
        )
        assert 0.0 < boost <= 0.15  # One exact match = 0.15

    def test_larger_boost_for_multiple_word_matches(self):
        """Multiple word matches give larger boost."""
        boost_one = calculate_boost(
            content="def login(): pass",
            query_words={"authentication", "login", "user"},
            mtime=None,
        )
        boost_two = calculate_boost(
            content="def login(): user = authenticate()",
            query_words={"login", "user", "authenticate"},
            mtime=None,
        )
        assert boost_two > boost_one

    def test_exact_phrase_match_gets_strong_boost(self):
        """Exact multi-word phrase match gets strong boost (+30-50%)."""
        # Query contains "duration_ms" - if content has it exactly, big boost
        boost = calculate_boost(
            content='log.debug("timing", duration_ms=100)',
            query_words={"duration_ms", "timing"},
            mtime=None,
        )
        # Exact match of "duration_ms" should give significant boost
        assert boost >= 0.3

    def test_exact_identifier_match_stronger_than_partial(self):
        """Exact identifier match is stronger than partial word match."""
        exact_boost = calculate_boost(
            content="duration_ms = time.time()",
            query_words={"duration_ms"},
            mtime=None,
        )
        partial_boost = calculate_boost(
            content="duration = time.time()  # in ms",
            query_words={"duration_ms"},
            mtime=None,
        )
        assert exact_boost > partial_boost

    def test_boost_capped_at_reasonable_maximum(self):
        """Boost doesn't exceed reasonable maximum (e.g., 0.5)."""
        boost = calculate_boost(
            content="duration_ms duration_ms duration_ms timing logging",
            query_words={"duration_ms", "timing", "logging"},
            mtime=None,
        )
        assert boost <= 0.5

    def test_recency_boost_for_recent_files(self):
        """Recently modified files get small recency boost."""
        import time

        now = time.time()
        recent_boost = calculate_boost(
            content="def foo(): pass",
            query_words={"foo"},
            mtime=now - 3600,  # 1 hour ago
        )
        old_boost = calculate_boost(
            content="def foo(): pass",
            query_words={"foo"},
            mtime=now - (30 * 24 * 3600),  # 30 days ago
        )
        assert recent_boost > old_boost
