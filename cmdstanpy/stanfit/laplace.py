"""
Container for the result of running a laplace approximation.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Hashable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, MutableMapping

import numpy as np
import pandas as pd

try:
    import xarray as xr

    XARRAY_INSTALLED = True
except ImportError:
    XARRAY_INSTALLED = False

from cmdstanpy.utils import stancsv
from cmdstanpy.utils.data_munging import build_xarray_data

from .metadata import InferenceMetadata, LaplaceRunConfig
from .mle import CmdStanMLE

# TODO list:
# - docs and example notebook


@dataclass
class CmdStanLaplace:
    """
    Container for outputs from the Laplace approximation.
    Created by :meth:`CmdStanModel.laplace_sample`.
    """

    metadata: InferenceMetadata
    model_name: str
    csv_file: str
    config: LaplaceRunConfig
    mode: CmdStanMLE
    config_file: str | None = None  # None if config object passed directly
    stdout_file: str | None = None
    _draws: np.ndarray = field(default_factory=lambda: np.array(()), init=False)

    @classmethod
    def from_files(
        cls,
        csv_file: str | os.PathLike,
        config_file: str | os.PathLike,
        stdout_file: str | os.PathLike | None = None,
        mode: CmdStanMLE | None = None,
    ) -> CmdStanLaplace:
        # Local import to avoid circular dependency with stanfit.__init__
        from cmdstanpy.stanfit import from_output_files

        with open(config_file) as f:
            stan_config = LaplaceRunConfig.from_json(f.read())

        metadata = InferenceMetadata.from_csv(csv_file)
        if mode is None:
            mle = from_output_files(
                stan_config.method_config.mode, method='optimize'
            )
            if not isinstance(mle, CmdStanMLE):
                raise TypeError(
                    f'Expected CmdStanMLE from mode file '
                    f'{stan_config.method_config.mode!r}, got '
                    f'{type(mle).__name__}'
                )
            mode = mle
        return cls(
            metadata=metadata,
            csv_file=os.fspath(csv_file),
            model_name=stan_config.model_name,
            config=stan_config,
            mode=mode,
            config_file=os.fspath(config_file),
            stdout_file=(
                os.fspath(stdout_file) if stdout_file is not None else None
            ),
        )

    def create_inits(
        self, seed: int | None = None, chains: int = 4
    ) -> list[dict[str, np.ndarray]] | dict[str, np.ndarray]:
        """
        Create initial values for the parameters of the model
        by randomly selecting draws from the Laplace approximation.

        :param seed: Used for random selection, defaults to None
        :param chains: Number of initial values to return, defaults to 4
        :return: The initial values for the parameters of the model.

        If ``chains`` is 1, a dictionary is returned, otherwise a list
        of dictionaries is returned, in the format expected for the
        ``inits`` argument of :meth:`CmdStanModel.sample`.
        """
        self._assemble_draws()
        rng = np.random.default_rng(seed)
        idxs = rng.choice(self._draws.shape[0], size=chains, replace=False)
        if chains == 1:
            draw = self._draws[idxs[0]]
            return {
                name: var.extract_reshape(draw)
                for name, var in self.metadata.stan_vars.items()
            }
        else:
            return [
                {
                    name: var.extract_reshape(self._draws[idx])
                    for name, var in self.metadata.stan_vars.items()
                }
                for idx in idxs
            ]

    def _assemble_draws(self) -> None:
        if self._draws.shape != (0,):
            return

        try:
            *_, draws = stancsv.parse_comments_header_and_draws(self.csv_file)
            self._draws = stancsv.csv_bytes_list_to_numpy(draws)
        except Exception as exc:
            raise ValueError(
                f"An error occurred when parsing Stan csv {self.csv_file}"
            ) from exc

    def stan_variable(self, var: str) -> np.ndarray:
        """
        Return a numpy.ndarray which contains the estimates for the
        for the named Stan program variable where the dimensions of the
        numpy.ndarray match the shape of the Stan program variable.

        This functionaltiy is also available via a shortcut using ``.`` -
        writing ``fit.a`` is a synonym for ``fit.stan_variable("a")``

        :param var: variable name

        See Also
        --------
        CmdStanMLE.stan_variables
        CmdStanMCMC.stan_variable
        CmdStanPathfinder.stan_variable
        CmdStanVB.stan_variable
        CmdStanGQ.stan_variable
        """
        self._assemble_draws()
        try:
            out: np.ndarray = self.metadata.stan_vars[var].extract_reshape(
                self._draws
            )
            return out
        except KeyError:
            # pylint: disable=raise-missing-from
            raise ValueError(
                f'Unknown variable name: {var}\n'
                'Available variables are '
                + ", ".join(self.metadata.stan_vars.keys())
            )

    def stan_variables(self) -> dict[str, np.ndarray]:
        """
        Return a dictionary mapping Stan program variables names
        to the corresponding numpy.ndarray containing the inferred values.

        :param inc_warmup: When ``True`` and the warmup draws are present in
            the MCMC sample, then the warmup draws are included.
            Default value is ``False``

        See Also
        --------
        CmdStanGQ.stan_variable
        CmdStanMCMC.stan_variables
        CmdStanMLE.stan_variables
        CmdStanPathfinder.stan_variables
        CmdStanVB.stan_variables
        """
        result = {}
        for name in self.metadata.stan_vars:
            result[name] = self.stan_variable(name)
        return result

    def method_variables(self) -> dict[str, np.ndarray]:
        """
        Returns a dictionary of all sampler variables, i.e., all
        output column names ending in `__`.  Assumes that all variables
        are scalar variables where column name is variable name.
        Maps each column name to a numpy.ndarray (draws x chains x 1)
        containing per-draw diagnostic values.
        """
        self._assemble_draws()
        return {
            name: var.extract_reshape(self._draws)
            for name, var in self.metadata.method_vars.items()
        }

    def draws(self) -> np.ndarray:
        """
        Return a numpy.ndarray containing the draws from the
        approximate posterior distribution. This is a 2-D array
        of shape (draws, parameters).
        """
        self._assemble_draws()
        return self._draws

    def draws_pd(
        self,
        vars: list[str] | str | None = None,
    ) -> pd.DataFrame:
        if vars is not None:
            if isinstance(vars, str):
                vars_list = [vars]
            else:
                vars_list = vars

        self._assemble_draws()
        cols = []
        if vars is not None:
            for var in dict.fromkeys(vars_list):
                if var in self.metadata.method_vars:
                    cols.append(var)
                elif var in self.metadata.stan_vars:
                    info = self.metadata.stan_vars[var]
                    cols.extend(
                        self.column_names[info.start_idx : info.end_idx]
                    )
                else:
                    raise ValueError(f'Unknown variable: {var}')

        else:
            cols = list(self.column_names)

        return pd.DataFrame(self._draws, columns=self.column_names)[cols]

    def draws_xr(
        self,
        vars: str | list[str] | None = None,
    ) -> xr.Dataset:
        """
        Returns the sampler draws as a xarray Dataset.

        :param vars: optional list of variable names.

        See Also
        --------
        CmdStanMCMC.draws_xr
        CmdStanGQ.draws_xr
        """
        if not XARRAY_INSTALLED:
            raise RuntimeError(
                'Package "xarray" is not installed, cannot produce draws array.'
            )

        if vars is None:
            vars_list = list(self.metadata.stan_vars.keys())
        elif isinstance(vars, str):
            vars_list = [vars]
        else:
            vars_list = vars

        self._assemble_draws()

        attrs: MutableMapping[Hashable, Any] = {
            "stan_version": f"{self.config.stan_major_version}."
            f"{self.config.stan_minor_version}."
            f"{self.config.stan_patch_version}",
            "model": self.model_name,
        }

        data: MutableMapping[Hashable, Any] = {}
        coordinates: MutableMapping[Hashable, Any] = {
            "draw": np.arange(self._draws.shape[0]),
        }

        for var in vars_list:
            build_xarray_data(
                data,
                self.metadata.stan_vars[var],
                self._draws[:, np.newaxis, :],
            )
        return (
            xr.Dataset(data, coords=coordinates, attrs=attrs)
            .transpose('draw', ...)
            .squeeze()
        )

    def __repr__(self) -> str:
        mode_repr = '\n'.join(
            ['\t' + line for line in repr(self.mode).splitlines()]
        )[1:]
        lines = [
            f'CmdStanLaplace: model={self.model_name}',
            f' mode=({mode_repr})',
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
        # for details. We call _assemble_draws to ensure posterior samples have
        # been loaded prior to serialization.
        self._assemble_draws()
        return self.__dict__

    @property
    def column_names(self) -> tuple[str, ...]:
        """
        Names of all outputs from the sampler, comprising sampler parameters
        and all components of all model parameters, transformed parameters,
        and quantities of interest. Corresponds to Stan CSV file header row,
        with names munged to array notation, e.g. `beta[1]` not `beta.1`.
        """
        return self.metadata.column_names

    def save_output_files(self, dir: str | None = None) -> None:
        """
        Move output CSV file, and any associated config and stdout files,
        to the specified directory. Updates the corresponding attributes on
        this object to point at the new locations.

        :param dir: directory path

        See Also
        --------
        cmdstanpy.from_output_files
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
