"""Container objects for results of CmdStan run(s)."""

import glob
import os

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


def from_output_files(  # pylint: disable=too-many-return-statements
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
    Instantiate a CmdStan object from a the Stan CSV files from a CmdStan run.
    CSV files are specified from either a list of Stan CSV files or a single
    filepath which can be either a directory name, a Stan CSV filename, or
    a pathname pattern (i.e., a Python glob).  The optional argument 'method'
    checks that the CSV files were produced by that method.
    Stan CSV files from CmdStan methods 'sample', 'optimize', and 'variational'
    result in objects of class CmdStanMCMC, CmdStanMLE, and CmdStanVB,
    respectively.

    :param path: directory path
    :param method: method name (optional)

    :return: either a CmdStanMCMC, CmdStanMLE, or CmdStanVB object
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
            'Bad method argument {}, must be one of: '
            '"sample", "optimize", "variational"'.format(method)
        )

    csvfiles = []
    if isinstance(path, list):
        csvfiles = path
    elif isinstance(path, str) and '*' in path:
        splits = os.path.split(path)
        if splits[0] is not None:
            if not (os.path.exists(splits[0]) and os.path.isdir(splits[0])):
                raise ValueError(
                    'Invalid path specification, {}  unknown '
                    'directory: {}'.format(path, splits[0])
                )
        csvfiles = glob.glob(path)
    elif isinstance(path, (str, os.PathLike)):
        if os.path.exists(path) and os.path.isdir(path):
            for file in os.listdir(path):
                if os.path.splitext(file)[1] == ".csv":
                    csvfiles.append(os.path.join(path, file))
        elif os.path.exists(path):
            csvfiles.append(os.fspath(path))
        else:
            raise ValueError('Invalid path specification: {}'.format(path))
    else:
        raise ValueError('Invalid path specification: {}'.format(path))

    if len(csvfiles) == 0:
        raise ValueError('No CSV files found in directory {}'.format(path))
    for file in csvfiles:
        if not (os.path.exists(file) and os.path.splitext(file)[1] == ".csv"):
            raise ValueError(
                'Bad CSV file path spec, includes non-csv file: {}'.format(file)
            )

    config_file0 = os.path.splitext(csvfiles[0])[0] + '_config.json'
    if not os.path.exists(config_file0):
        raise ValueError(
            f'Config file not found at expected path: {config_file0}. '
            'Reconstructing a fit from output files requires the config JSON '
            'written by CmdStan 2.36 or later.'
        )
    try:
        with open(config_file0) as f:
            method_name = parse_config(f.read()).method_config.method
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
            config_files: list[str] = []
            metric_files: list[str] = []
            for cf in csvfiles:
                stem = os.path.splitext(cf)[0]
                cfg = stem + '_config.json'
                if not os.path.exists(cfg):
                    raise ValueError(
                        'Sample config file not found at expected path: '
                        f'{cfg}'
                    )
                config_files.append(cfg)
                metric = stem + '_metric.json'
                if os.path.exists(metric):
                    metric_files.append(metric)
            fit = CmdStanMCMC.from_files(
                csv_files=csvfiles,
                config_files=config_files,
                metric_files=metric_files or None,
            )
            fit.draws()
            return fit
        elif method_name == 'optimize':
            if len(csvfiles) != 1:
                raise ValueError(
                    'Expecting a single optimize Stan CSV file, '
                    f'found {len(csvfiles)}'
                )
            csv_file = csvfiles[0]
            config_file = os.path.splitext(csv_file)[0] + '_config.json'
            return CmdStanMLE.from_files(
                csv_file=csv_file, config_file=config_file
            )
        elif method_name == 'variational':
            if len(csvfiles) != 1:
                raise ValueError(
                    'Expecting a single variational Stan CSV file, '
                    f'found {len(csvfiles)}'
                )
            csv_file = csvfiles[0]
            config_file = os.path.splitext(csv_file)[0] + '_config.json'
            if not os.path.exists(config_file):
                raise ValueError(
                    'Variational config file not found at expected path: '
                    f'{config_file}'
                )
            return CmdStanVB.from_files(
                csv_file=csv_file, config_file=config_file
            )
        elif method_name == 'laplace':
            if len(csvfiles) != 1:
                raise ValueError(
                    'Expecting a single Laplace Stan CSV file, '
                    f'found {len(csvfiles)}'
                )
            csv_file = csvfiles[0]
            config_file = os.path.splitext(csv_file)[0] + '_config.json'
            if not os.path.exists(config_file):
                raise ValueError(
                    'Laplace config file not found at expected path: '
                    f'{config_file}'
                )
            return CmdStanLaplace.from_files(
                csv_file=csv_file, config_file=config_file
            )
        elif method_name == 'pathfinder':
            if len(csvfiles) != 1:
                raise ValueError(
                    'Expecting a single Pathfinder Stan CSV file, '
                    f'found {len(csvfiles)}'
                )
            csv_file = csvfiles[0]
            config_file = os.path.splitext(csv_file)[0] + '_config.json'
            if not os.path.exists(config_file):
                raise ValueError(
                    'Pathfinder config file not found at expected path: '
                    f'{config_file}'
                )
            return CmdStanPathfinder.from_files(
                csv_file=csv_file, config_file=config_file
            )
        else:
            get_logger().warning(
                'Unable to process CSV output files from method %s.',
                (method_name),
            )
            return None
    except (IOError, OSError, PermissionError) as e:
        raise ValueError(
            'An error occurred processing the CSV files:\n\t{}'.format(str(e))
        ) from e
