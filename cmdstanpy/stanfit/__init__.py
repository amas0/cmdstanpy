"""Container objects for results of CmdStan run(s)."""

import glob
import os

from cmdstanpy.cmdstan_args import CmdStanArgs, SamplerArgs, VariationalArgs
from cmdstanpy.utils import check_sampler_csv, get_logger, stancsv

from .gq import CmdStanGQ, PrevFit
from .laplace import CmdStanLaplace
from .mcmc import CmdStanMCMC
from .metadata import InferenceMetadata
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


def from_csv(  # pylint: disable=too-many-return-statements
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

    try:
        comments, *_ = stancsv.parse_comments_header_and_draws(csvfiles[0])
        config_dict = stancsv.parse_config(comments)
    except (IOError, OSError, PermissionError) as e:
        raise ValueError('Cannot read CSV file: {}'.format(csvfiles[0])) from e
    if 'model' not in config_dict or 'method' not in config_dict:
        raise ValueError("File {} is not a Stan CSV file.".format(csvfiles[0]))
    if method is not None and method != config_dict['method']:
        raise ValueError(
            'Expecting Stan CSV output files from method {}, '
            ' found outputs from method {}'.format(
                method, config_dict['method']
            )
        )
    model: str = config_dict['model']  # type: ignore
    try:
        if config_dict['method'] == 'sample':
            save_warmup = config_dict['save_warmup'] == 1
            chains = len(csvfiles)
            num_samples: int = config_dict['num_samples']  # type: ignore
            num_warmup: int = config_dict['num_warmup']  # type: ignore
            thin: int = config_dict['thin']  # type: ignore
            sampler_args = SamplerArgs(
                iter_sampling=num_samples,
                iter_warmup=num_warmup,
                thin=thin,
                save_warmup=save_warmup,
            )
            # bugfix 425, check for fixed_params output
            try:
                check_sampler_csv(
                    csvfiles[0],
                    iter_sampling=num_samples,
                    iter_warmup=num_warmup,
                    thin=thin,
                    save_warmup=save_warmup,
                )
            except ValueError:
                try:
                    check_sampler_csv(
                        csvfiles[0],
                        iter_sampling=num_samples,
                        iter_warmup=num_warmup,
                        thin=thin,
                        save_warmup=save_warmup,
                    )
                    sampler_args = SamplerArgs(
                        iter_sampling=num_samples,
                        iter_warmup=num_warmup,
                        thin=thin,
                        save_warmup=save_warmup,
                        fixed_param=True,
                    )
                except ValueError as e:
                    raise ValueError(
                        'Invalid or corrupt Stan CSV output file, '
                    ) from e

            cmdstan_args = CmdStanArgs(
                model_name=model,
                model_exe=model,
                chain_ids=[x + 1 for x in range(chains)],
                method_args=sampler_args,
            )
            runset = RunSet(args=cmdstan_args, chains=chains)
            runset._csv_files = csvfiles
            for i in range(len(runset._retcodes)):
                runset._set_retcode(i, 0)
            fit = CmdStanMCMC(runset)
            fit.draws()
            return fit
        elif config_dict['method'] == 'optimize':
            if len(csvfiles) != 1:
                raise ValueError(
                    'Expecting a single optimize Stan CSV file, '
                    f'found {len(csvfiles)}'
                )
            csv_file = csvfiles[0]
            config_file = os.path.splitext(csv_file)[0] + '_config.json'
            if os.path.exists(config_file):
                return CmdStanMLE.from_files(
                    csv_file=csv_file, config_file=config_file
                )
            # Legacy path: no config file, build config from CSV metadata
            if 'algorithm' not in config_dict:
                raise ValueError(
                    "Cannot find optimization algorithm in file {}.".format(
                        csv_file
                    )
                )
            from .metadata import OptimizeConfig, StanConfig

            opt_config = OptimizeConfig(
                algorithm=config_dict['algorithm'],  # type: ignore
                save_iterations=config_dict.get('save_iterations', 0) == 1,
                jacobian=config_dict.get('jacobian', 0) == 1,
            )
            stan_config = StanConfig[OptimizeConfig].model_validate(
                {
                    'model_name': config_dict['model'],
                    'stan_major_version': str(
                        config_dict.get('stan_version_major', '')
                    ),
                    'stan_minor_version': str(
                        config_dict.get('stan_version_minor', '')
                    ),
                    'stan_patch_version': str(
                        config_dict.get('stan_version_patch', '')
                    ),
                    'method_config': opt_config.model_dump(),
                }
            )
            return CmdStanMLE(
                metadata=InferenceMetadata.from_csv(csv_file),
                model_name=stan_config.model_name,
                csv_file=csv_file,
                config=stan_config,
            )
        elif config_dict['method'] == 'variational':
            if 'algorithm' not in config_dict:
                raise ValueError(
                    "Cannot find variational algorithm in file {}.".format(
                        csvfiles[0]
                    )
                )
            variational_args = VariationalArgs(
                algorithm=config_dict['algorithm'],  # type: ignore
                iter=config_dict['iter'],  # type: ignore
                grad_samples=config_dict['grad_samples'],  # type: ignore
                elbo_samples=config_dict['elbo_samples'],  # type: ignore
                eta=config_dict['eta'],  # type: ignore
                tol_rel_obj=config_dict['tol_rel_obj'],  # type: ignore
                eval_elbo=config_dict['eval_elbo'],  # type: ignore
                output_samples=config_dict['output_samples'],  # type: ignore
            )
            cmdstan_args = CmdStanArgs(
                model_name=model,
                model_exe=model,
                chain_ids=None,
                method_args=variational_args,
            )
            runset = RunSet(args=cmdstan_args)
            runset._csv_files = csvfiles
            for i in range(len(runset._retcodes)):
                runset._set_retcode(i, 0)
            return CmdStanVB(runset)
        elif config_dict['method'] == 'laplace':
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
        elif config_dict['method'] == 'pathfinder':
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
                (config_dict['method']),
            )
            return None
    except (IOError, OSError, PermissionError) as e:
        raise ValueError(
            'An error occurred processing the CSV files:\n\t{}'.format(str(e))
        ) from e
