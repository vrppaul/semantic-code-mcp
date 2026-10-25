"""Logging configuration: structlog to stderr, as JSON unless debugging."""

import logging
import sys

import structlog


def configure_logging(*, debug: bool) -> None:
    """Configure structlog once, at start-up.

    Args:
        debug: Debug level and readable console output; otherwise info level and JSON.
    """
    # The console renderer prints an exception itself; JSON holds one only as text
    renderers = (
        [structlog.dev.ConsoleRenderer()]
        if debug
        else [structlog.processors.format_exc_info, structlog.processors.JSONRenderer()]
    )
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            *renderers,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.DEBUG if debug else logging.INFO
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
