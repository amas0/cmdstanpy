"""
Container for the result of running Pathfinder.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cmdstanpy.stanfit.metadata import (
    InferenceMetadata,
    PathfinderConfig,
    StanConfig,
    parse_config,
)
from cmdstanpy.utils import stancsv


@dataclass
class CmdStanPathfinder:
    """
    Container for outputs from the Pathfinder algorithm.
    Created by :meth:`CmdStanModel.pathfinder()`.
    """

    metadata: InferenceMetadata
    model_name: str
    csv_file: str
    config: StanConfig[PathfinderConfig]
    config_file: str | None = None
    stdout_file: str | None = None
    _draws: np.ndarray = field(default_factory=lambda: np.array(()), init=False)

    @classmethod
    def from_files(
        cls,
        csv_file: str | os.PathLike,
        config_file: str | os.PathLike,
        stdout_file: str | os.PathLike | None = None,
    ) -> CmdStanPathfinder:
        with open(config_file) as f:
            stan_config = parse_config(f.read(), PathfinderConfig)

        metadata = InferenceMetadata.from_csv(csv_file)
        return cls(
            metadata=metadata,
            csv_file=os.fspath(csv_file),
            model_name=stan_config.model_name,
            config=stan_config,
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
        by randomly selecting draws from the Pathfinder approximation.

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

    def __repr__(self) -> str:
        lines = [
            f'CmdStanPathfinder: model={self.model_name}',
            f' csv_file:\n\t{self.csv_file}',
        ]
        if self.config_file is not None:
            lines.append(f' config_file:\n\t{self.config_file}')
        if self.stdout_file is not None:
            lines.append(f' output_file:\n\t{self.stdout_file}')
        return '\n'.join(lines)

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
        CmdStanPathfinder.stan_variables
        CmdStanMLE.stan_variable
        CmdStanMCMC.stan_variable
        CmdStanVB.stan_variable
        CmdStanGQ.stan_variable
        CmdStanLaplace.stan_variable
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

        See Also
        --------
        CmdStanPathfinder.stan_variable
        CmdStanMCMC.stan_variables
        CmdStanMLE.stan_variables
        CmdStanVB.stan_variables
        CmdStanGQ.stan_variables
        CmdStanLaplace.stan_variables
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

    @property
    def is_resampled(self) -> bool:
        """
        Returns True if the draws were resampled from several Pathfinder
        approximations, False otherwise.
        """
        return (
            self.config.method_config.num_paths > 1
            and self.config.method_config.psis_resample
            and self.config.method_config.calculate_lp
        )

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
