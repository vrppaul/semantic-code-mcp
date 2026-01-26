"""Logging configuration for semantic-code-mcp."""

import logging
import sys

import structlog

_configured = False


def configure_logging(debug: bool = False) -> None:
    """Configure structlog for the application.

    Args:
        debug: If True, enable debug level and console output.
               If False, use info level and JSON output.
    """
    global _configured
    if _configured:
        return

    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer() if debug else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.DEBUG if debug else logging.INFO
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )

    _configured = True
