"""The global configuration for the test suite"""
import os
import subprocess
from typing import Generator

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
DATAFILES_PATH = os.path.join(HERE, 'data')


@pytest.fixture(scope='session', autouse=True)
def cleanup_test_files() -> Generator[None, None, None]:
    """Remove compiled models and output files after test run."""
    yield
    subprocess.Popen(
        ['git', 'clean', '-fX', DATAFILES_PATH],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
