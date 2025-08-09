"""Container for metadata parsed from the output of a CmdStan run"""

import copy
import os
from typing import Any, Dict, Iterator, Tuple, Union

import stanio

from cmdstanpy.utils import stancsv


class InferenceMetadata:
    """
    CmdStan configuration and contents of output file parsed out of
    the Stan CSV file header comments and column headers.
    Assumes valid CSV files.
    """

    def __init__(
        self, config: Dict[str, Union[str, int, float, Tuple[str, ...]]]
    ) -> None:
        """Initialize object from CSV headers"""
        self._cmdstan_config = config

        assert isinstance(config['raw_header'], str)
        vars = stanio.parse_header(config['raw_header'])

        self._method_vars = {
            k: v for (k, v) in vars.items() if k.endswith('__')
        }
        self._stan_vars = {
            k: v for (k, v) in vars.items() if not k.endswith('__')
        }

    @classmethod
    def from_csv(
        cls, stan_csv: Union[str, os.PathLike, Iterator[bytes]]
    ) -> 'InferenceMetadata':
        comments, draws = stancsv.parse_stan_csv_comments_and_draws(stan_csv)
        return cls(stancsv.extract_config_and_header_info(comments, draws))

    def __repr__(self) -> str:
        return 'Metadata:\n{}\n'.format(self._cmdstan_config)

    def __getitem__(self, key: str) -> Union[str, int, float, Tuple[str, ...]]:
        return self._cmdstan_config[key]

    @property
    def cmdstan_config(self) -> Dict[str, Any]:
        """
        Returns a dictionary containing a set of name, value pairs
        parsed out of the Stan CSV file header.  These include the
        command configuration and the CSV file header row information.
        Uses deepcopy for immutability.
        """
        return copy.deepcopy(self._cmdstan_config)

    @property
    def method_vars(self) -> Dict[str, stanio.Variable]:
        """
        Method variable names always end in `__`, e.g. `lp__`.
        """
        return self._method_vars

    @property
    def stan_vars(self) -> Dict[str, stanio.Variable]:
        """
        These are the user-defined variables in the Stan program.
        """
        return self._stan_vars
