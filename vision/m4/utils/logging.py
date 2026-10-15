# [MODULE DOC] Library-level logging setup for m4, modelled on transformers.utils.logging. Keeps a single
# stderr
# StreamHandler on the library root logger ("m4"), guarded by a threading lock, with default level from env
# var M4_VERBOSITY (debug/info/warning/error/critical; default WARNING). Re-exports logging level constants
# and, on import, monkey-patches logging.Logger with a warning_once method (global side effect).
import functools
import logging
import os
import sys
import threading
from logging import CRITICAL  # NOQA
from logging import DEBUG  # NOQA
from logging import ERROR  # NOQA
from logging import FATAL  # NOQA
from logging import INFO  # NOQA
from logging import NOTSET  # NOQA
from logging import WARN  # NOQA
from logging import WARNING  # NOQA
from typing import Optional


_lock = threading.Lock()
_default_handler: Optional[logging.Handler] = None

log_levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

_default_log_level = logging.WARNING

_tqdm_active = True


def _get_default_logging_level():
    # [FUNCTION DOC] HELPER: returns the level for M4_VERBOSITY if set to a known name; logs a warning on the
    # root logger for
    # unknown values and falls back to _default_log_level (WARNING).
    """
    If M4_VERBOSITY env var is set to one of the valid choices return that as the new default level. If it is
    not - fall back to `_default_log_level`
    """
    env_level_str = os.getenv("M4_VERBOSITY", None)
    if env_level_str:
        if env_level_str in log_levels:
            return log_levels[env_level_str]
        else:
            logging.getLogger().warning(
                f"Unknown option M4_VERBOSITY={env_level_str}, has to be one of: { ', '.join(log_levels.keys()) }"
            )
    return _default_log_level


def _get_library_name() -> str:
    # [FUNCTION DOC] HELPER: returns the top-level package name from this module's __name__ (e.g. "m4").
    return __name__.split(".")[0]


def _get_library_root_logger() -> logging.Logger:
    # [FUNCTION DOC] HELPER: returns the logging.Logger for the library root name.
    return logging.getLogger(_get_library_name())


def _configure_library_root_logger() -> None:
    # [FUNCTION DOC] HELPER: idempotently (under _lock) creates the global _default_handler (StreamHandler to
    # stderr, flush
    # bound to sys.stderr.flush), attaches it to the library root logger, sets the default level and disables
    # propagation. Mutates module global _default_handler.
    global _default_handler

    with _lock:
        if _default_handler:
            # This library has already configured the library root logger.
            return
        _default_handler = logging.StreamHandler()  # Set sys.stderr as stream.
        _default_handler.flush = sys.stderr.flush

        # Apply our default configuration to the library root logger.
        library_root_logger = _get_library_root_logger()
        library_root_logger.addHandler(_default_handler)
        library_root_logger.setLevel(_get_default_logging_level())
        library_root_logger.propagate = False


def _reset_library_root_logger() -> None:
    # [FUNCTION DOC] HELPER: under _lock, if a default handler exists removes it from the library root logger,
    # resets the
    # level to NOTSET and clears the global _default_handler.
    global _default_handler

    with _lock:
        if not _default_handler:
            return

        library_root_logger = _get_library_root_logger()
        library_root_logger.removeHandler(_default_handler)
        library_root_logger.setLevel(logging.NOTSET)
        _default_handler = None


def get_log_levels_dict():
    # [FUNCTION DOC] Returns the mapping of level names to logging level ints.
    return log_levels


def get_logger(name: Optional[str] = None) -> logging.Logger:
    # [FUNCTION DOC] Returns logging.getLogger(name), defaulting name to the library name, after ensuring the
    # library root
    # logger is configured.
    """
    Return a logger with the specified name.

    This function is not supposed to be directly accessed unless you are writing a custom m4 module.
    """

    if name is None:
        name = _get_library_name()

    _configure_library_root_logger()
    return logging.getLogger(name)


def get_verbosity() -> int:
    # [FUNCTION DOC] Configures the library logger if needed and returns its effective level as an int.
    """
    Return the current level for the 🤗 M4's root logger as an int.

    Returns:
        `int`: The logging level.

    <Tip>

    🤗 M4 has following logging levels:

    - 50: `m4.logging.CRITICAL` or `m4.logging.FATAL`
    - 40: `m4.logging.ERROR`
    - 30: `m4.logging.WARNING` or `m4.logging.WARN`
    - 20: `m4.logging.INFO`
    - 10: `m4.logging.DEBUG`

    </Tip>"""

    _configure_library_root_logger()
    return _get_library_root_logger().getEffectiveLevel()


def set_verbosity(verbosity: int) -> None:
    # [FUNCTION DOC] Configures the library logger if needed and sets its level to the given int.
    """
    Set the verbosity level for the 🤗 M4's root logger.

    Args:
        verbosity (`int`):
            Logging level, e.g., one of:

            - `m4.logging.CRITICAL` or `m4.logging.FATAL`
            - `m4.logging.ERROR`
            - `m4.logging.WARNING` or `m4.logging.WARN`
            - `m4.logging.INFO`
            - `m4.logging.DEBUG`
    """

    _configure_library_root_logger()
    _get_library_root_logger().setLevel(verbosity)


def set_verbosity_info():
    # [FUNCTION DOC] Sets library verbosity to INFO.
    """Set the verbosity to the `INFO` level."""
    return set_verbosity(INFO)


def set_verbosity_warning():
    # [FUNCTION DOC] Sets library verbosity to WARNING.
    """Set the verbosity to the `WARNING` level."""
    return set_verbosity(WARNING)


def set_verbosity_debug():
    # [FUNCTION DOC] Sets library verbosity to DEBUG.
    """Set the verbosity to the `DEBUG` level."""
    return set_verbosity(DEBUG)


def set_verbosity_error():
    # [FUNCTION DOC] Sets library verbosity to ERROR.
    """Set the verbosity to the `ERROR` level."""
    return set_verbosity(ERROR)


def disable_default_handler() -> None:
    # [FUNCTION DOC] Removes the default handler from the library root logger (asserts it exists after
    # configuration).
    """Disable the default handler of the HuggingFace M4's root logger."""

    _configure_library_root_logger()

    assert _default_handler is not None
    _get_library_root_logger().removeHandler(_default_handler)


def enable_default_handler() -> None:
    # [FUNCTION DOC] Re-adds the default handler to the library root logger (asserts it exists).
    """Enable the default handler of the HuggingFace M4's root logger."""

    _configure_library_root_logger()

    assert _default_handler is not None
    _get_library_root_logger().addHandler(_default_handler)


def add_handler(handler: logging.Handler) -> None:
    # [FUNCTION DOC] Adds the given handler to the library root logger; asserts handler is not None.
    """adds a handler to the HuggingFace M4's root logger."""

    _configure_library_root_logger()

    assert handler is not None
    _get_library_root_logger().addHandler(handler)


def remove_handler(handler: logging.Handler) -> None:
    # [FUNCTION DOC] Removes the given handler from the library root logger.
    # KNOWN ISSUE: the assertion requires the handler to NOT already be attached (inverted condition), so
    # removing an attached handler raises AssertionError and the removal is a no-op otherwise.
    """removes given handler from the HuggingFace M4's root logger."""

    _configure_library_root_logger()

    assert handler is not None and handler not in _get_library_root_logger().handlers
    _get_library_root_logger().removeHandler(handler)


def disable_propagation() -> None:
    # [FUNCTION DOC] Sets propagate False on the library root logger.
    """
    Disable propagation of the library log outputs. Note that log propagation is disabled by default.
    """

    _configure_library_root_logger()
    _get_library_root_logger().propagate = False


def enable_propagation() -> None:
    # [FUNCTION DOC] Sets propagate True on the library root logger (may cause double logging if the root
    # logger also has
    # handlers).
    """
    Enable propagation of the library log outputs. Please disable the HuggingFace M4's default handler to
    prevent double logging if the root logger has been configured.
    """

    _configure_library_root_logger()
    _get_library_root_logger().propagate = True


def enable_explicit_format() -> None:
    # [FUNCTION DOC] Sets the formatter "[LEVEL|file:line] time >> message" on every handler of the library
    # root logger.
    # Note: does not call _configure_library_root_logger first.
    """
    Enable explicit formatting for every HuggingFace M4's logger. The explicit formatter is as follows:
    ```
        [LEVELNAME|FILENAME|LINE NUMBER] TIME >> MESSAGE
    ```
    All handlers currently bound to the root logger are affected by this method.
    """
    handlers = _get_library_root_logger().handlers

    for handler in handlers:
        formatter = logging.Formatter("[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s")
        handler.setFormatter(formatter)


def reset_format() -> None:
    # [FUNCTION DOC] Clears the formatter (sets None) on every handler of the library root logger.
    """
    Resets the formatting for HuggingFace M4's loggers.

    All handlers currently bound to the root logger are affected by this method.
    """
    handlers = _get_library_root_logger().handlers

    for handler in handlers:
        handler.setFormatter(None)


@functools.lru_cache(None)
def warning_once(self, *args, **kwargs):
    # [FUNCTION DOC] Logger method installed as logging.Logger.warning_once: calls self.warning(*args,
    # **kwargs), but is
    # wrapped in an unbounded functools.lru_cache so identical (logger, args) combinations warn only once.
    # Arguments must be hashable (unhashable kwargs raise TypeError); the cache holds references forever.
    """
    This method is identical to `logger.warning()`, but will emit the warning with the same message only once

    Note: The cache is for the function arguments, so 2 different callers using the same arguments
    will hit the cache. The assumption here is that all warning messages are unique across the code.
    If they aren't then need to switch to another type of cache that includes the caller frame
    information in the hashing function.
    """
    self.warning(*args, **kwargs)


logging.Logger.warning_once = warning_once
