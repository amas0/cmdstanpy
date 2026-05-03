"""Container for the results of running autodiff variational inference"""

from __future__ import annotations

import os
import shutil
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from cmdstanpy.utils import stancsv

from .metadata import (
    InferenceMetadata,
    StanConfig,
    VariationalConfig,
    parse_config,
)


@dataclass
class CmdStanVB:
    """
    Container for outputs from CmdStan variational run.
    Created by :meth:`CmdStanModel.variational`.
    """

    metadata: InferenceMetadata
    model_name: str
    csv_file: str
    config: StanConfig[VariationalConfig]
    config_file: str | None = None  # None if config object passed directly
    stdout_file: str | None = None
    _variational_mean: np.ndarray = field(
        default_factory=lambda: np.array(()), init=False
    )
    _variational_sample: np.ndarray = field(
        default_factory=lambda: np.array(()), init=False
    )
    _eta: float | None = field(default=None, init=False)

    @classmethod
    def from_files(
        cls,
        csv_file: str | os.PathLike,
        config_file: str | os.PathLike,
        stdout_file: str | os.PathLike | None = None,
    ) -> CmdStanVB:
        with open(config_file) as f:
            stan_config = parse_config(f.read(), VariationalConfig)

        metadata = InferenceMetadata.from_csv(csv_file)
        return cls(
            metadata=metadata,
            model_name=stan_config.model_name,
            csv_file=os.fspath(csv_file),
            config=stan_config,
            config_file=os.fspath(config_file),
            stdout_file=(
                os.fspath(stdout_file) if stdout_file is not None else None
            ),
        )

    def _assemble_draws(self) -> None:
        if self._variational_mean.shape != (0,):
            return

        try:
            (
                comment_lines,
                _,
                draw_lines,
            ) = stancsv.parse_comments_header_and_draws(self.csv_file)
            self._eta = stancsv.parse_variational_eta(comment_lines)
            draws_np = stancsv.csv_bytes_list_to_numpy(draw_lines)
        except Exception as exc:
            raise ValueError(
                f"An error occurred when parsing Stan csv {self.csv_file}"
            ) from exc

        self._variational_mean = draws_np[0]
        self._variational_sample = draws_np[1:]

    def create_inits(
        self, seed: int | None = None, chains: int = 4
    ) -> list[dict[str, np.ndarray]] | dict[str, np.ndarray]:
        """
        Create initial values for the parameters of the model
        by randomly selecting draws from the variational approximation
        draws.

        :param seed: Used for random selection, defaults to None
        :param chains: Number of initial values to return, defaults to 4
        :return: The initial values for the parameters of the model.

        If ``chains`` is 1, a dictionary is returned, otherwise a list
        of dictionaries is returned, in the format expected for the
        ``inits`` argument of :meth:`CmdStanModel.sample`.
        """
        self._assemble_draws()
        rng = np.random.default_rng(seed)
        idxs = rng.choice(
            self._variational_sample.shape[0], size=chains, replace=False
        )
        if chains == 1:
            draw = self._variational_sample[idxs[0]]
            return {
                name: var.extract_reshape(draw)
                for name, var in self.metadata.stan_vars.items()
            }
        else:
            return [
                {
                    name: var.extract_reshape(self._variational_sample[idx])
                    for name, var in self.metadata.stan_vars.items()
                }
                for idx in idxs
            ]

    def __repr__(self) -> str:
        mc = self.config.method_config
        lines = [
            f'CmdStanVB: model={self.model_name}'
            f' method={mc.method} algorithm={mc.algorithm}',
            f' csv_file:\n\t{self.csv_file}',
        ]
        if self.config_file is not None:
            lines.append(f' config_file:\n\t{self.config_file}')
        if self.stdout_file is not None:
            lines.append(f' output_file:\n\t{self.stdout_file}')
        return '\n'.join(lines)

    def __getattr__(self, attr: str) -> np.ndarray:
        """Synonymous with ``fit.stan_variable(attr)"""
        if attr.startswith("_"):
            raise AttributeError(f"Unknown variable name {attr}")
        try:
            return self.stan_variable(attr)
        except ValueError as e:
            # pylint: disable=raise-missing-from
            raise AttributeError(*e.args)

    def __getstate__(self) -> dict:
        # This function returns the mapping of objects to serialize with pickle.
        # See https://docs.python.org/3/library/pickle.html#object.__getstate__
        # for details. We call _assemble_draws to ensure draws have been
        # loaded prior to serialization.
        self._assemble_draws()
        state = self.__dict__.copy()
        # StanConfig[VariationalConfig] is a generic alias with no module-level
        # name, so we serialize it as a plain dict for pickle compatibility.
        state['config'] = self.config.model_dump()
        return state

    def __setstate__(self, state: dict) -> None:
        config_dict = state.pop('config')
        self.__dict__.update(state)
        self.config = StanConfig[VariationalConfig].model_validate(config_dict)

    @property
    def columns(self) -> int:
        """
        Total number of information items returned by sampler.
        Includes approximation information and names of model parameters
        and computed quantities.
        """
        return len(self.column_names)

    @property
    def column_names(self) -> tuple[str, ...]:
        """
        Names of information items returned by sampler for each draw.
        Includes approximation information and names of model parameters
        and computed quantities.
        """
        return self.metadata.column_names

    @property
    def eta(self) -> float:
        """
        Step size scaling parameter 'eta'
        """
        self._assemble_draws()
        return self._eta  # type: ignore[return-value]

    @property
    def variational_params_np(self) -> np.ndarray:
        """
        Returns inferred parameter means as numpy array.
        """
        self._assemble_draws()
        return self._variational_mean

    @property
    def variational_params_pd(self) -> pd.DataFrame:
        """
        Returns inferred parameter means as pandas DataFrame.
        """
        self._assemble_draws()
        return pd.DataFrame([self._variational_mean], columns=self.column_names)

    @property
    def variational_params_dict(self) -> dict[str, np.ndarray]:
        """Returns inferred parameter means as Dict."""
        self._assemble_draws()
        return OrderedDict(zip(self.column_names, self._variational_mean))

    def stan_variable(self, var: str, *, mean: bool = False) -> np.ndarray:
        """
        Return a numpy.ndarray which contains the estimates for the
        for the named Stan program variable where the dimensions of the
        numpy.ndarray match the shape of the Stan program variable, with
        a leading axis added for the number of draws from the variational
        approximation.

        * If the variable is a scalar variable, the return array has shape
          ( draws, ).
        * If the variable is a vector, the return array has shape
          ( draws, len(vector))
        * If the variable is a matrix, the return array has shape
          ( draws, size(dim 1), size(dim 2) )
        * If the variable is an array with N dimensions, the return array
          has shape ( draws, size(dim 1), ..., size(dim N))

        This functionaltiy is also available via a shortcut using ``.`` -
        writing ``fit.a`` is a synonym for ``fit.stan_variable("a")``

        :param var: variable name

        :param mean: if True, return the variational mean. Otherwise,
            return the variational sample. Defaults to False.

        See Also
        --------
        CmdStanVB.stan_variables
        CmdStanMCMC.stan_variable
        CmdStanMLE.stan_variable
        CmdStanPathfinder.stan_variable
        CmdStanGQ.stan_variable
        CmdStanLaplace.stan_variable
        """
        self._assemble_draws()

        if mean:
            draws = self._variational_mean
        else:
            draws = self._variational_sample

        try:
            out: np.ndarray = self.metadata.stan_vars[var].extract_reshape(
                draws
            )
            return out
        except KeyError:
            # pylint: disable=raise-missing-from
            raise ValueError(
                f'Unknown variable name: {var}\n'
                'Available variables are '
                + ", ".join(self.metadata.stan_vars.keys())
            )

    def stan_variables(self, *, mean: bool = False) -> dict[str, np.ndarray]:
        """
        Return a dictionary mapping Stan program variables names
        to the corresponding numpy.ndarray containing the inferred values.

        See Also
        --------
        CmdStanVB.stan_variable
        CmdStanMCMC.stan_variables
        CmdStanMLE.stan_variables
        CmdStanGQ.stan_variables
        CmdStanPathfinder.stan_variables
        CmdStanLaplace.stan_variables
        """
        result = {}
        for name in self.metadata.stan_vars:
            result[name] = self.stan_variable(name, mean=mean)
        return result

    @property
    def variational_sample(self) -> np.ndarray:
        """Returns the set of approximate posterior output draws."""
        self._assemble_draws()
        return self._variational_sample

    @property
    def variational_sample_pd(self) -> pd.DataFrame:
        """
        Returns the set of approximate posterior output draws as
        a pandas DataFrame.
        """
        self._assemble_draws()
        return pd.DataFrame(self._variational_sample, columns=self.column_names)

    def save_csvfiles(self, dir: str | None = None) -> None:
        """
        Move output CSV file, and any associated config and stdout files,
        to the specified directory. Updates the corresponding attributes on
        this object to point at the new locations.

        :param dir: directory path

        See Also
        --------
        cmdstanpy.from_csv
        """
        dest = Path(dir) if dir is not None else Path.cwd()
        dest.mkdir(parents=True, exist_ok=True)

        for attr in ('csv_file', 'config_file', 'stdout_file'):
            src = getattr(self, attr)
            if src is None:
                continue
            dst = dest / Path(src).name
            if dst.exists():
                raise ValueError(f'File exists, not overwriting: {dst}')
            shutil.move(src, dst)
            setattr(self, attr, os.fspath(dst))
