"""
CmdStanPy logging
"""

import functools
import logging
import types
from contextlib import AbstractContextManager
from typing import Optional, Type


@functools.lru_cache(maxsize=None)
def get_logger() -> logging.Logger:
    """cmdstanpy logger"""
    logger = logging.getLogger("cmdstanpy")
    if not logger.hasHandlers():
        # send all messages to handlers
        logger.setLevel(logging.DEBUG)
        # add a default handler to the logger to INFO and higher
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "%H:%M:%S",
            )
        )
        logger.addHandler(handler)
    return logger


class EnableLogging(AbstractContextManager):
    def __init__(self) -> None:
        self.logger = get_logger()
        self.prev_state = self.logger.disabled
        self.logger.disabled = False

    def __enter__(self) -> "EnableLogging":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[types.TracebackType],
    ) -> None:
        self.logger.disabled = self.prev_state


class DisableLogging(AbstractContextManager):
    def __init__(self) -> None:
        self.logger = get_logger()
        self.prev_state = self.logger.disabled
        self.logger.disabled = True

    def __enter__(self) -> "DisableLogging":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[types.TracebackType],
    ) -> None:
        self.logger.disabled = self.prev_state


def enable_logging() -> EnableLogging:
    """Enable cmdstanpy logging. Can be used as a context manager"""
    return EnableLogging()


def disable_logging() -> DisableLogging:
    """Disable cmdstanpy logging. Can be used as a context manager"""
    return DisableLogging()
