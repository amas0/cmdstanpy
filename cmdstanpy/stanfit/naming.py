"""Helpers for the file naming conventions of CmdStan output files."""

import re
from pathlib import Path

CONFIG_SUFFIX = '_config.json'
METRIC_SUFFIX = '_metric.json'

# The trailing number in a chain's file name both orders the chains and
# gives their ids.
CHAIN_ID_RE = re.compile(r'[_-](\d+)$')


def accompanying_json(csv_file: Path, kind: str) -> Path:
    """Path of the JSON of the given kind CmdStan writes for ``csv_file``."""
    return csv_file.with_name(f'{csv_file.stem}_{kind}.json')


def csv_for_config(config_file: Path) -> Path:
    """The Stan CSV file a config JSON is named for.  CmdStan derives the
    config file name from the (first) output CSV it was given."""
    return config_file.with_name(
        config_file.name.removesuffix(CONFIG_SUFFIX) + '.csv'
    )
