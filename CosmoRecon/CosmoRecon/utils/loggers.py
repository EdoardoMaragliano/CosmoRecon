"""Logging utilities for CosmoRecon."""

import logging


def setup_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """Create or retrieve a module-level logger.

    Uses propagation to the root logger rather than adding per-module
    handlers, avoiding duplicate log messages when the calling script
    configures root logging via ``logging.basicConfig()``.

    Parameters
    ----------
    name : str
        Logger name (typically ``__name__``).
    level : int
        Logging level (default ``DEBUG``).

    Returns
    -------
    logging.Logger
    """
    log = logging.getLogger(name)
    log.setLevel(level)
    # Only add a StreamHandler when no handlers exist at root level,
    # to ensure library loggers produce output even if the user has not
    # configured logging.  If root already has handlers (i.e. a script
    # called basicConfig), propagation will carry messages upward.
    if not logging.root.handlers and not log.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s [%(name)s] [%(levelname)s] %(message)s'
        )
        handler.setFormatter(formatter)
        log.addHandler(handler)
    return log
