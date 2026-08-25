"""Container objects for results of CmdStan run(s)."""

import glob
import os
import re
from collections.abc import Callable
from pathlib import Path

from cmdstanpy.utils import get_logger

from .gq import CmdStanGQ, PrevFit
from .laplace import CmdStanLaplace
from .mcmc import CmdStanMCMC
from .metadata import InferenceMetadata, parse_config
from .mle import CmdStanMLE
from .pathfinder import CmdStanPathfinder
from .runset import RunSet
from .vb import CmdStanVB

__all__ = [
    "RunSet",
    "InferenceMetadata",
    "CmdStanMCMC",
    "CmdStanMLE",
    "CmdStanVB",
    "CmdStanGQ",
    "CmdStanLaplace",
    "CmdStanPathfinder",
    "PrevFit",
]


# CmdStanPy names each chain's files `<name>_<chain_id>.<ext>`; the
# trailing number both orders the chains and gives their ids.
_CHAIN_ID_RE = re.compile(r'_(\d+)$')


def _chain_sort_key(csv_file: Path) -> tuple[int, str]:
    """Sort key ordering Stan CSV files by their trailing chain id."""
    match = _CHAIN_ID_RE.search(csv_file.stem)
    return (int(match.group(1)) if match else 0, csv_file.name)


def _chain_ids(csvfiles: list[Path]) -> list[int]:
    """Chain ids from the files' trailing numbers, falling back to 1..N
    when the files are not named that way."""
    matches = [_CHAIN_ID_RE.search(f.stem) for f in csvfiles]
    ids = [int(m.group(1)) for m in matches if m]
    if len(ids) == len(csvfiles) and len(set(ids)) == len(ids):
        return ids
    return list(range(1, len(csvfiles) + 1))


def _holds_draws(csv_file: Path) -> bool:
    """
    Whether a CSV found in an output directory holds draws, as opposed to
    latent dynamics or profiling data. Those are written as
    ``<base>_diagnostic[_<id>].csv`` and ``<base>_profile[_<id>].csv``,
    so they would otherwise be mistaken for additional chains.
    """
    return not _CHAIN_ID_RE.sub('', csv_file.stem).endswith(
        ('_diagnostic', '_profile')
    )


def _sidecar(csv_file: Path, kind: str) -> Path:
    """Path to the JSON of the given kind CmdStan writes for ``csv_file``."""
    return csv_file.with_name(f'{csv_file.stem}_{kind}.json')


def _resolve_csv_files(path: str | list[str] | os.PathLike) -> list[Path]:
    """Expand a path spec (list, glob, directory, or file) into CSV paths."""
    if isinstance(path, list):
        csvfiles = [Path(file) for file in path]
    elif isinstance(path, str) and '*' in path:
        parent = Path(path).parent
        if parent.name and not parent.is_dir():
            raise ValueError(
                'Invalid path specification, {}  unknown '
                'directory: {}'.format(path, parent)
            )
        csvfiles = [Path(file) for file in glob.glob(path)]
    elif isinstance(path, (str, os.PathLike)):
        spec = Path(path)
        if spec.is_dir():
            csvfiles = [f for f in spec.glob('*.csv') if _holds_draws(f)]
        elif spec.exists():
            csvfiles = [spec]
        else:
            raise ValueError('Invalid path specification: {}'.format(path))
    else:
        raise ValueError('Invalid path specification: {}'.format(path))

    if len(csvfiles) == 0:
        raise ValueError('No CSV files found in directory {}'.format(path))
    for file in csvfiles:
        if not (file.exists() and file.suffix == '.csv'):
            raise ValueError(
                'Bad CSV file path spec, includes non-csv file: {}'.format(file)
            )
    # Sort by trailing chain id so that chains are ordered 1..N regardless of
    # the order the filesystem reported them in; ``glob`` and ``iterdir`` are
    # both unordered, and callers rely on index i being chain i.
    return sorted(csvfiles, key=_chain_sort_key)


# Methods whose output is a single Stan CSV file, and the builder for each.
_SINGLE_FILE_METHODS: dict[
    str,
    Callable[..., CmdStanMLE | CmdStanVB | CmdStanLaplace | CmdStanPathfinder],
] = {
    'optimize': CmdStanMLE.from_files,
    'variational': CmdStanVB.from_files,
    'laplace': CmdStanLaplace.from_files,
    'pathfinder': CmdStanPathfinder.from_files,
}


def from_output_files(
    path: str | list[str] | os.PathLike | None = None,
    method: str | None = None,
) -> (
    CmdStanMCMC
    | CmdStanMLE
    | CmdStanVB
    | CmdStanPathfinder
    | CmdStanLaplace
    | None
):
    """
    Instantiate a CmdStan object from the output files of a CmdStan run.
    Output files are specified as either a list of Stan CSV files or a single
    filepath which can be either a directory name, a Stan CSV filename, or
    a pathname pattern (i.e., a Python glob).  The optional argument 'method'
    checks that the files were produced by that method.

    The files are assumed to follow CmdStanPy's own output naming: each
    chain's CSV ends in ``_<chain_id>``, and the config (and, for sample,
    metric) JSONs CmdStan writes sit next to the CSVs they belong to, with
    a run that shared one process across chains having a single config
    named for its first chain. Outputs written by running CmdStan directly
    with a single output name and ``num_chains`` follow a different naming
    scheme and are not currently supported.

    :param path: directory path
    :param method: method name (optional)

    :return: a CmdStanMCMC, CmdStanMLE, CmdStanVB, CmdStanLaplace, or
        CmdStanPathfinder object, according to the method that was run
    """
    if path is None:
        raise ValueError('Must specify path to Stan CSV files.')
    if method is not None and method not in [
        'sample',
        'optimize',
        'variational',
        'laplace',
        'pathfinder',
    ]:
        raise ValueError(
            'Bad method argument {}, must be one of: "sample", "optimize", '
            '"variational", "laplace", "pathfinder"'.format(method)
        )

    csvfiles = _resolve_csv_files(path)

    config_file0 = _sidecar(csvfiles[0], 'config')
    if not config_file0.exists():
        raise ValueError(
            f'Config file not found alongside {csvfiles[0]}. '
            'Reconstructing a fit from output files requires the config JSON '
            'written by CmdStan 2.34 or later.'
        )
    try:
        with open(config_file0) as f:
            stan_config0 = parse_config(f.read())
        method_name = stan_config0.method_config.method
    except (IOError, OSError, PermissionError) as e:
        raise ValueError(
            'Cannot read config file: {}'.format(config_file0)
        ) from e
    if method is not None and method != method_name:
        raise ValueError(
            'Expecting Stan CSV output files from method {}, '
            ' found outputs from method {}'.format(method, method_name)
        )
    try:
        if method_name == 'sample':
            # Collect the sidecar JSONs that exist next to each CSV. A
            # single-process run has just one config, named for the first
            # chain's output file; a per-process run has one per chain.
            config_files = [
                cf
                for cf in dict.fromkeys(
                    _sidecar(csv_file, 'config') for csv_file in csvfiles
                )
                if cf.exists()
            ]
            metric_files = [
                mf
                for mf in (
                    _sidecar(csv_file, 'metric') for csv_file in csvfiles
                )
                if mf.exists()
            ]
            fit = CmdStanMCMC.from_files(
                csv_files=csvfiles,
                config_files=config_files,
                metric_files=metric_files or None,
                chain_ids=_chain_ids(csvfiles),
            )
            fit.draws()
            return fit

        builder = _SINGLE_FILE_METHODS.get(method_name)
        if builder is not None:
            if len(csvfiles) != 1:
                raise ValueError(
                    f'Expecting a single {method_name} Stan CSV file, '
                    f'found {len(csvfiles)}'
                )
            return builder(csv_file=csvfiles[0], config_file=config_file0)

        get_logger().warning(
            'Unable to process CSV output files from method %s.',
            (method_name),
        )
        return None
    except (IOError, OSError, PermissionError) as e:
        raise ValueError(
            'An error occurred processing the CSV files:\n\t{}'.format(str(e))
        ) from e
