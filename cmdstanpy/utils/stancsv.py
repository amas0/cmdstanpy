"""
Utility functions for reading the Stan CSV format
"""

import io
import json
import math
import re
import warnings
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    MutableMapping,
    Optional,
    TextIO,
    Tuple,
    Union,
)

import numpy as np
import numpy.typing as npt
import pandas as pd

from cmdstanpy import _CMDSTAN_SAMPLING, _CMDSTAN_THIN, _CMDSTAN_WARMUP


def parse_stan_csv_comments_and_draws(
    lines: Iterator[bytes],
) -> Tuple[List[bytes], List[bytes]]:
    """Parses lines of a Stan CSV file into comment lines and draws lines, where
    a draws line is just a non-commented line.

    Returns a (comment_lines, draws_lines) tuple.
    """
    comment_lines, draws_lines = [], []

    for line in lines:
        if line.startswith(b"#"):  # is comment line
            comment_lines.append(line)
        else:
            draws_lines.append(line)
    return comment_lines, draws_lines


def csv_bytes_list_to_numpy(
    csv_bytes_list: List[bytes], includes_header: bool = True
) -> npt.NDArray[np.float64]:
    """Efficiently converts a list of bytes representing whose concatenation
    represents a CSV file into a numpy array. Includes header specifies
    whether the bytes contains an initial header line."""
    try:
        import polars as pl

        try:
            out = (
                pl.read_csv(
                    io.BytesIO(b"".join(csv_bytes_list)),
                    has_header=includes_header,
                )
                .to_numpy()
                .astype(np.float64)
            )
            if out.shape[0] == 0:
                raise ValueError("No data found to parse")
        except pl.exceptions.NoDataError as exc:
            raise ValueError("No data found to parse") from exc
    except ImportError:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            out = np.loadtxt(
                csv_bytes_list,
                skiprows=int(includes_header),
                delimiter=",",
                dtype=np.float64,
                ndmin=1,
            )
        if out.shape == (0,):
            raise ValueError("No data found to parse")  # pylint: disable=W0707
        if len(out.shape) == 1:
            out = out.reshape(1, -1)

    return out  # type: ignore


def parse_hmc_adaptation_lines(
    comment_lines: List[bytes],
) -> Tuple[float, Optional[npt.NDArray[np.float64]]]:
    """Extracts step size/mass matrix information from the Stan CSV comment
    lines by parsing the adaptation section. If unit metric is used, the mass
    matrix field will be None, otherwise an appropriate numpy array.

    Returns a (step_size, mass_matrix) tuple"""
    step_size, mass_matrix = None, None

    cleaned_lines = (ln.lstrip(b"# ") for ln in comment_lines)
    in_matrix_block = False
    matrix_lines = []
    for line in cleaned_lines:
        if in_matrix_block and line.strip():
            # Stop when we get to timing block
            if line.startswith(b"Elapsed Time"):
                break
            matrix_lines.append(line)
        elif line.startswith(b"Step size"):
            _, ss_str = line.split(b" = ")
            step_size = float(ss_str)
        elif line.startswith(b"Diagonal") or line.startswith(b"Elements"):
            in_matrix_block = True
        elif line.startswith(b"No free"):
            break
    if step_size is None:
        raise ValueError("Unable to parse adapated step size")
    if matrix_lines:
        mass_matrix = csv_bytes_list_to_numpy(
            matrix_lines, includes_header=False
        )
    return step_size, mass_matrix


def parse_timing_lines(
    comment_lines: List[bytes],
) -> Dict[str, float]:
    """Parse the timing lines into a dictionary with key corresponding
    to the phase, e.g. Warm-up, Sampling, Total, and value the elapsed seconds
    """
    out: Dict[str, float] = {}

    cleaned_lines = (ln.lstrip(b"# ") for ln in comment_lines)
    in_timing_block = False
    for line in cleaned_lines:
        if line.startswith(b"Elapsed Time") and not in_timing_block:
            in_timing_block = True

        if not in_timing_block:
            continue
        match = re.findall(r"([\d\.]+) seconds \((.+)\)", str(line))
        if match:
            seconds = float(match[0][0])
            phase = match[0][1]
            out[phase] = seconds
    return out


def check_sampler_csv(
    path: str,
    is_fixed_param: bool = False,
    iter_sampling: Optional[int] = None,
    iter_warmup: Optional[int] = None,
    save_warmup: bool = False,
    thin: Optional[int] = None,
) -> Dict[str, Any]:
    """Capture essential config, shape from stan_csv file."""
    meta = scan_sampler_csv(path, is_fixed_param)
    if thin is None:
        thin = _CMDSTAN_THIN
    elif thin > _CMDSTAN_THIN:
        if 'thin' not in meta:
            raise ValueError(
                'bad Stan CSV file {}, '
                'config error, expected thin = {}'.format(path, thin)
            )
        if meta['thin'] != thin:
            raise ValueError(
                'bad Stan CSV file {}, '
                'config error, expected thin = {}, found {}'.format(
                    path, thin, meta['thin']
                )
            )
    draws_sampling = iter_sampling
    if draws_sampling is None:
        draws_sampling = _CMDSTAN_SAMPLING
    draws_warmup = iter_warmup
    if draws_warmup is None:
        draws_warmup = _CMDSTAN_WARMUP
    draws_warmup = int(math.ceil(draws_warmup / thin))
    draws_sampling = int(math.ceil(draws_sampling / thin))
    if meta['draws_sampling'] != draws_sampling:
        raise ValueError(
            'bad Stan CSV file {}, expected {} draws, found {}'.format(
                path, draws_sampling, meta['draws_sampling']
            )
        )
    if save_warmup:
        if not ('save_warmup' in meta and meta['save_warmup'] in (1, 'true')):
            raise ValueError(
                'bad Stan CSV file {}, '
                'config error, expected save_warmup = 1'.format(path)
            )
        if meta['draws_warmup'] != draws_warmup:
            raise ValueError(
                'bad Stan CSV file {}, '
                'expected {} warmup draws, found {}'.format(
                    path, draws_warmup, meta['draws_warmup']
                )
            )
    return meta


def scan_sampler_csv(path: str, is_fixed_param: bool = False) -> Dict[str, Any]:
    """Process sampler stan_csv output file line by line."""
    dict: Dict[str, Any] = {}
    lineno = 0
    with open(path, 'r') as fd:
        try:
            lineno = scan_config(fd, dict, lineno)
            lineno = scan_column_names(fd, dict, lineno)
            if not is_fixed_param:
                lineno = scan_warmup_iters(fd, dict, lineno)
                lineno = scan_hmc_params(fd, dict, lineno)
            lineno = scan_sampling_iters(fd, dict, lineno, is_fixed_param)
            lineno = scan_time(fd, dict, lineno)
        except ValueError as e:
            raise ValueError("Error in reading csv file: " + path) from e
    return dict


def scan_optimize_csv(path: str, save_iters: bool = False) -> Dict[str, Any]:
    """Process optimizer stan_csv output file line by line."""
    dict: Dict[str, Any] = {}
    lineno = 0
    # scan to find config, header, num saved iters
    with open(path, 'r') as fd:
        lineno = scan_config(fd, dict, lineno)
        lineno = scan_column_names(fd, dict, lineno)
        iters = 0
        for line in fd:
            iters += 1
    if save_iters:
        all_iters: np.ndarray = np.empty(
            (iters, len(dict['column_names'])), dtype=float, order='F'
        )
    # rescan to capture estimates
    with open(path, 'r') as fd:
        for i in range(lineno):
            fd.readline()
        for i in range(iters):
            line = fd.readline().strip()
            if len(line) < 1:
                raise ValueError(
                    'cannot parse CSV file {}, error at line {}'.format(
                        path, lineno + i
                    )
                )
            xs = line.split(',')
            if save_iters:
                all_iters[i, :] = [float(x) for x in xs]
            if i == iters - 1:
                mle: np.ndarray = np.array(xs, dtype=float)
    # pylint: disable=possibly-used-before-assignment
    dict['mle'] = mle
    if save_iters:
        dict['all_iters'] = all_iters
    return dict


def scan_generic_csv(path: str) -> Dict[str, Any]:
    """Process laplace stan_csv output file line by line."""
    dict: Dict[str, Any] = {}
    lineno = 0
    with open(path, 'r') as fd:
        lineno = scan_config(fd, dict, lineno)
        lineno = scan_column_names(fd, dict, lineno)
    return dict


def scan_variational_csv(path: str) -> Dict[str, Any]:
    """Process advi stan_csv output file line by line."""
    dict: Dict[str, Any] = {}
    lineno = 0
    with open(path, 'r') as fd:
        lineno = scan_config(fd, dict, lineno)
        lineno = scan_column_names(fd, dict, lineno)
        line = fd.readline().lstrip(' #\t').rstrip()
        lineno += 1
        if line.startswith('Stepsize adaptation complete.'):
            line = fd.readline().lstrip(' #\t\n')
            lineno += 1
            if not line.startswith('eta'):
                raise ValueError(
                    'line {}: expecting eta, found:\n\t "{}"'.format(
                        lineno, line
                    )
                )
            _, eta = line.split('=')
            dict['eta'] = float(eta)
            line = fd.readline().lstrip(' #\t\n')
            lineno += 1
        xs = line.split(',')
        variational_mean = [float(x) for x in xs]
        dict['variational_mean'] = np.array(variational_mean)
        dict['variational_sample'] = pd.read_csv(
            path,
            comment='#',
            skiprows=lineno,
            header=None,
            float_precision='high',
        ).to_numpy()
    return dict


def scan_config(fd: TextIO, config_dict: Dict[str, Any], lineno: int) -> int:
    """
    Scan initial stan_csv file comments lines and
    save non-default configuration information to config_dict.
    """
    cur_pos = fd.tell()
    line = fd.readline().strip()
    while len(line) > 0 and line.startswith('#'):
        lineno += 1
        if line.endswith('(Default)'):
            line = line.replace('(Default)', '')
        line = line.lstrip(' #\t')
        key_val = line.split('=')
        if len(key_val) == 2:
            if key_val[0].strip() == 'file' and not key_val[1].endswith('csv'):
                config_dict['data_file'] = key_val[1].strip()
            elif key_val[0].strip() != 'file':
                raw_val = key_val[1].strip()
                val: Union[int, float, str]
                try:
                    val = int(raw_val)
                except ValueError:
                    try:
                        val = float(raw_val)
                    except ValueError:
                        if raw_val == "true":
                            val = 1
                        elif raw_val == "false":
                            val = 0
                        else:
                            val = raw_val
                config_dict[key_val[0].strip()] = val
        cur_pos = fd.tell()
        line = fd.readline().strip()
    fd.seek(cur_pos)
    return lineno


def scan_warmup_iters(
    fd: TextIO, config_dict: Dict[str, Any], lineno: int
) -> int:
    """
    Check warmup iterations, if any.
    """
    if 'save_warmup' not in config_dict:
        return lineno
    cur_pos = fd.tell()
    line = fd.readline().strip()
    draws_found = 0
    while len(line) > 0 and not line.startswith('#'):
        lineno += 1
        draws_found += 1
        cur_pos = fd.tell()
        line = fd.readline().strip()
    fd.seek(cur_pos)
    config_dict['draws_warmup'] = draws_found
    return lineno


def scan_column_names(
    fd: TextIO, config_dict: MutableMapping[str, Any], lineno: int
) -> int:
    """
    Process columns header, add to config_dict as 'column_names'
    """
    line = fd.readline().strip()
    lineno += 1
    config_dict['raw_header'] = line.strip()
    names = line.split(',')
    config_dict['column_names'] = tuple(munge_varnames(names))
    return lineno


def munge_varname(name: str) -> str:
    if '.' not in name and ':' not in name:
        return name

    tuple_parts = name.split(':')
    for i, part in enumerate(tuple_parts):
        if '.' not in part:
            continue
        part = part.replace('.', '[', 1)
        part = part.replace('.', ',')
        part += ']'
        tuple_parts[i] = part

    return '.'.join(tuple_parts)


def munge_varnames(names: List[str]) -> List[str]:
    """
    Change formatting for indices of container var elements
    from use of dot separator to array-like notation, e.g.,
    rewrite label ``y_forecast.2.4`` to ``y_forecast[2,4]``.
    """
    if names is None:
        raise ValueError('missing argument "names"')
    return [munge_varname(name) for name in names]


def scan_hmc_params(
    fd: TextIO, config_dict: Dict[str, Any], lineno: int
) -> int:
    """
    Scan step size, metric from  stan_csv file comment lines.
    """
    metric = config_dict['metric']
    line = fd.readline().strip()
    lineno += 1
    if not line == '# Adaptation terminated':
        raise ValueError(
            'line {}: expecting metric, found:\n\t "{}"'.format(lineno, line)
        )
    line = fd.readline().strip()
    lineno += 1
    label, step_size = line.split('=')
    if not label.startswith('# Step size'):
        raise ValueError(
            'line {}: expecting step size, '
            'found:\n\t "{}"'.format(lineno, line)
        )
    try:
        float(step_size.strip())
    except ValueError as e:
        raise ValueError(
            'line {}: invalid step size: {}'.format(lineno, step_size)
        ) from e
    before_metric = fd.tell()
    line = fd.readline().strip()
    lineno += 1
    if metric == 'unit_e':
        if line.startswith("# No free parameters"):
            return lineno
        else:
            fd.seek(before_metric)
            return lineno - 1

    if not (
        (
            metric == 'diag_e'
            and line == '# Diagonal elements of inverse mass matrix:'
        )
        or (
            metric == 'dense_e' and line == '# Elements of inverse mass matrix:'
        )
    ):
        raise ValueError(
            'line {}: invalid or missing mass matrix '
            'specification'.format(lineno)
        )
    line = fd.readline().lstrip(' #\t')
    lineno += 1
    num_unconstrained_params = len(line.split(','))
    if metric == 'diag_e':
        return lineno
    else:
        for _ in range(1, num_unconstrained_params):
            line = fd.readline().lstrip(' #\t')
            lineno += 1
            if len(line.split(',')) != num_unconstrained_params:
                raise ValueError(
                    'line {}: invalid or missing mass matrix '
                    'specification'.format(lineno)
                )
        return lineno


def scan_sampling_iters(
    fd: TextIO, config_dict: Dict[str, Any], lineno: int, is_fixed_param: bool
) -> int:
    """
    Parse sampling iteration, save number of iterations to config_dict.
    Also save number of divergences, max_treedepth hits
    """
    draws_found = 0
    num_cols = len(config_dict['column_names'])
    if not is_fixed_param:
        idx_divergent = config_dict['column_names'].index('divergent__')
        idx_treedepth = config_dict['column_names'].index('treedepth__')
        max_treedepth = config_dict['max_depth']
        ct_divergences = 0
        ct_max_treedepth = 0

    cur_pos = fd.tell()
    line = fd.readline().strip()
    while len(line) > 0 and not line.startswith('#'):
        lineno += 1
        draws_found += 1
        data = line.split(',')
        if len(data) != num_cols:
            raise ValueError(
                'line {}: bad draw, expecting {} items, found {}\n'.format(
                    lineno, num_cols, len(line.split(','))
                )
                + 'This error could be caused by running out of disk space.\n'
                'Try clearing up TEMP or setting output_dir to a path'
                ' on another drive.',
            )
        cur_pos = fd.tell()
        line = fd.readline().strip()
        if not is_fixed_param:
            ct_divergences += int(data[idx_divergent])  # type: ignore
            if int(data[idx_treedepth]) == max_treedepth:  # type: ignore
                ct_max_treedepth += 1

    fd.seek(cur_pos)
    config_dict['draws_sampling'] = draws_found
    if not is_fixed_param:
        config_dict['ct_divergences'] = ct_divergences
        config_dict['ct_max_treedepth'] = ct_max_treedepth
    return lineno


def scan_time(fd: TextIO, config_dict: Dict[str, Any], lineno: int) -> int:
    """
    Scan time information from the trailing comment lines in a Stan CSV file.

    #  Elapsed Time: 0.001332 seconds (Warm-up)
    #                0.000249 seconds (Sampling)
    #                0.001581 seconds (Total)


    It extracts the time values and saves them in the config_dict: key 'time',
    value a dictionary with keys 'warmup', 'sampling', and 'total'.
    Returns the updated line number after reading the time info.

    :param fd: Open file descriptor at comment row following all sample data.
    :param config_dict: Dictionary to which the time info is added.
    :param lineno: Current line number
    """
    time = {}
    keys = ['warmup', 'sampling', 'total']
    while True:
        pos = fd.tell()
        line = fd.readline()
        if not line:
            break
        lineno += 1
        stripped = line.strip()
        if not stripped.startswith('#'):
            fd.seek(pos)
            lineno -= 1
            break
        content = stripped.lstrip('#').strip()
        if not content:
            continue
        tokens = content.split()
        if len(tokens) < 3:
            raise ValueError(f"Invalid time at line {lineno}: {content}")
        if 'Warm-up' in content:
            key = 'warmup'
            time_str = tokens[2]
        elif 'Sampling' in content:
            key = 'sampling'
            time_str = tokens[0]
        elif 'Total' in content:
            key = 'total'
            time_str = tokens[0]
        else:
            raise ValueError(f"Invalid time at line {lineno}: {content}")
        try:
            t = float(time_str)
        except ValueError as e:
            raise ValueError(f"Invalid time at line {lineno}: {content}") from e
        time[key] = t

    if not all(key in time for key in keys):
        raise ValueError(f"Invalid time, stopped at {lineno}")

    config_dict['time'] = time
    return lineno


def read_metric(path: str) -> List[int]:
    """
    Read metric file in JSON or Rdump format.
    Return dimensions of entry "inv_metric".
    """
    if path.endswith('.json'):
        with open(path, 'r') as fd:
            metric_dict = json.load(fd)
        if 'inv_metric' in metric_dict:
            dims_np: np.ndarray = np.asarray(metric_dict['inv_metric'])
            return list(dims_np.shape)
        else:
            raise ValueError(
                'metric file {}, bad or missing'
                ' entry "inv_metric"'.format(path)
            )
    else:
        dims = list(read_rdump_metric(path))
        if dims is None:
            raise ValueError(
                'metric file {}, bad or missing'
                ' entry "inv_metric"'.format(path)
            )
        return dims


def read_rdump_metric(path: str) -> List[int]:
    """
    Find dimensions of variable named 'inv_metric' in Rdump data file.
    """
    metric_dict = rload(path)
    if metric_dict is None or not (
        'inv_metric' in metric_dict
        and isinstance(metric_dict['inv_metric'], np.ndarray)
    ):
        raise ValueError(
            'metric file {}, bad or missing entry "inv_metric"'.format(path)
        )
    return list(metric_dict['inv_metric'].shape)


def rload(fname: str) -> Optional[Dict[str, Union[int, float, np.ndarray]]]:
    """Parse data and parameter variable values from an R dump format file.
    This parser only supports the subset of R dump data as described
    in the "Dump Data Format" section of the CmdStan manual, i.e.,
    scalar, vector, matrix, and array data types.
    """
    data_dict = {}
    with open(fname, 'r') as fd:
        lines = fd.readlines()
    # Variable data may span multiple lines, parse accordingly
    idx = 0
    while idx < len(lines) and '<-' not in lines[idx]:
        idx += 1
    if idx == len(lines):
        return None
    start_idx = idx
    idx += 1
    while True:
        while idx < len(lines) and '<-' not in lines[idx]:
            idx += 1
        next_var = idx
        var_data = ''.join(lines[start_idx:next_var]).replace('\n', '')
        lhs, rhs = [item.strip() for item in var_data.split('<-')]
        lhs = lhs.replace('"', '')  # strip optional Jags double quotes
        rhs = rhs.replace('L', '')  # strip R long int qualifier
        data_dict[lhs] = parse_rdump_value(rhs)
        if idx == len(lines):
            break
        start_idx = next_var
        idx += 1
    return data_dict


def parse_rdump_value(rhs: str) -> Union[int, float, np.ndarray]:
    """Process right hand side of Rdump variable assignment statement.
    Value is either scalar, vector, or multi-dim structure.
    Use regex to capture structure values, dimensions.
    """
    pat = re.compile(
        r'structure\(\s*c\((?P<vals>[^)]*)\)'
        r'(,\s*\.Dim\s*=\s*c\s*\((?P<dims>[^)]*)\s*\))?\)'
    )
    val: Union[int, float, np.ndarray]
    try:
        if rhs.startswith('structure'):
            parse = pat.match(rhs)
            if parse is None or parse.group('vals') is None:
                raise ValueError(rhs)
            vals = [float(v) for v in parse.group('vals').split(',')]
            val = np.array(vals, order='F')
            if parse.group('dims') is not None:
                dims = [int(v) for v in parse.group('dims').split(',')]
                val = np.array(vals).reshape(dims, order='F')
        elif rhs.startswith('c(') and rhs.endswith(')'):
            val = np.array([float(item) for item in rhs[2:-1].split(',')])
        elif '.' in rhs or 'e' in rhs:
            val = float(rhs)
        else:
            val = int(rhs)
    except TypeError as e:
        raise ValueError('bad value in Rdump file: {}'.format(rhs)) from e
    return val
