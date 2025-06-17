"""
CmdStanPy logging
"""
import functools
import logging
from contextlib import AbstractContextManager


@functools.lru_cache(maxsize=None)
def get_logger() -> logging.Logger:
    """cmdstanpy logger"""
    logger = logging.getLogger('cmdstanpy')
    if len(logger.handlers) == 0:
        # send all messages to handlers
        logger.setLevel(logging.DEBUG)
        # add a default handler to the logger to INFO and higher
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                "%H:%M:%S",
            )
        )
        logger.addHandler(handler)
    return logger


class enable_logging(AbstractContextManager):
    """Enable cmdstanpy logging. Can be used as a context manager"""

    def __init__(self) -> None:
        self.logger = get_logger()
        self.prev_state = self.logger.disabled
        self.logger.disabled = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.logger.disabled = self.prev_state


class disable_logging(AbstractContextManager):
    """Disable cmdstanpy logging. Can be used as a context manager"""

    def __init__(self) -> None:
        self.logger = get_logger()
        self.prev_state = self.logger.disabled
        self.logger.disabled = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.logger.disabled = self.prev_state
