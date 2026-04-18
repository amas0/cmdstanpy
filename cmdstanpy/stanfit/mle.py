"""Container for the result of running optimization"""

from __future__ import annotations

import os
import shutil
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from cmdstanpy.utils import get_logger, stancsv

from .metadata import (
    InferenceMetadata,
    OptimizeConfig,
    StanConfig,
    parse_config,
)


@dataclass
class CmdStanMLE:
    """
    Container for outputs from CmdStan optimization.
    Created by :meth:`CmdStanModel.optimize`.
    """

    metadata: InferenceMetadata
    model_name: str
    csv_file: str
    config: StanConfig[OptimizeConfig]
    converged: bool = True
    config_file: str | None = None  # None if config object passed directly
    stdout_file: str | None = None
    _mle: np.ndarray = field(default_factory=lambda: np.array(()), init=False)
    _all_iters: np.ndarray = field(
        default_factory=lambda: np.array(()), init=False
    )

    @classmethod
    def from_files(
        cls,
        csv_file: str | os.PathLike,
        config_file: str | os.PathLike,
        stdout_file: str | os.PathLike | None = None,
        converged: bool = True,
    ) -> CmdStanMLE:
        with open(config_file) as f:
            stan_config = parse_config(f.read(), OptimizeConfig)

        metadata = InferenceMetadata.from_csv(csv_file)
        return cls(
            metadata=metadata,
            model_name=stan_config.model_name,
            csv_file=os.fspath(csv_file),
            config=stan_config,
            converged=converged,
            config_file=os.fspath(config_file),
            stdout_file=(
                os.fspath(stdout_file) if stdout_file is not None else None
            ),
        )

    def _assemble_draws(self) -> None:
        if self._mle.shape != (0,):
            return

        try:
            *_, draws_lines = stancsv.parse_comments_header_and_draws(
                self.csv_file
            )
            all_draws = stancsv.csv_bytes_list_to_numpy(draws_lines)
        except Exception as exc:
            raise ValueError(
                f"An error occurred when parsing Stan csv {self.csv_file}"
            ) from exc

        self._mle = all_draws[-1]
        if self.config.method_config.save_iterations:
            self._all_iters = all_draws

    def create_inits(
        self, seed: int | None = None, chains: int = 4
    ) -> dict[str, np.ndarray]:
        """
        Create initial values for the parameters of the model
        from the MLE.

        :param seed: Unused. Kept for compatibility with other
        create_inits methods.
        :param chains: Unused. Kept for compatibility with other
        create_inits methods.
        :return: The initial values for the parameters of the model.

        Returns a dictionary of MLE estimates in the format expected
        for the ``inits`` argument of :meth:`CmdStanModel.sample`.
        When running multi-chain sampling, all chains will be initialized
        at the same points.
        """
        # pylint: disable=unused-argument

        return self.stan_variables()

    def __repr__(self) -> str:
        mc = self.config.method_config
        lines = [
            f'CmdStanMLE: model={self.model_name}'
            f' method={mc.method} algorithm={mc.algorithm}',
            f' csv_file:\n\t{self.csv_file}',
        ]
        if self.config_file is not None:
            lines.append(f' config_file:\n\t{self.config_file}')
        if self.stdout_file is not None:
            lines.append(f' output_file:\n\t{self.stdout_file}')
        if not self.converged:
            lines.append(
                ' Warning: invalid estimate, optimization failed to converge.'
            )
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
        # for details. We call _assemble_draws to ensure estimates have been
        # loaded prior to serialization.
        self._assemble_draws()
        state = self.__dict__.copy()
        # StanConfig[OptimizeConfig] is a generic alias with no module-level
        # name, so we serialize it as a plain dict for pickle compatibility.
        state['config'] = self.config.model_dump()
        return state

    def __setstate__(self, state: dict) -> None:
        config_dict = state.pop('config')
        self.__dict__.update(state)
        self.config = StanConfig[OptimizeConfig].model_validate(config_dict)

    @property
    def column_names(self) -> tuple[str, ...]:
        """
        Names of estimated quantities, includes joint log probability,
        and all parameters, transformed parameters, and generated quantities.
        """
        return self.metadata.column_names

    @property
    def optimized_params_np(self) -> np.ndarray:
        """
        Returns all final estimates from the optimizer as a numpy.ndarray
        which contains all optimizer outputs, i.e., the value for `lp__`
        as well as all Stan program variables.
        """
        if not self.converged:
            get_logger().warning(
                'Invalid estimate, optimization failed to converge.'
            )
        self._assemble_draws()
        return self._mle

    @property
    def optimized_iterations_np(self) -> np.ndarray | None:
        """
        Returns all saved iterations from the optimizer and final estimate
        as a numpy.ndarray which contains all optimizer outputs, i.e.,
        the value for `lp__` as well as all Stan program variables.

        """
        if not self.config.method_config.save_iterations:
            get_logger().warning(
                'Intermediate iterations not saved to CSV output file. '
                'Rerun the optimize method with "save_iterations=True".'
            )
            return None
        if not self.converged:
            get_logger().warning(
                'Invalid estimate, optimization failed to converge.'
            )
        self._assemble_draws()
        return self._all_iters

    @property
    def optimized_params_pd(self) -> pd.DataFrame:
        """
        Returns all final estimates from the optimizer as a pandas.DataFrame
        which contains all optimizer outputs, i.e., the value for `lp__`
        as well as all Stan program variables.
        """
        if not self.converged:
            get_logger().warning(
                'Invalid estimate, optimization failed to converge.'
            )
        self._assemble_draws()
        return pd.DataFrame([self._mle], columns=self.column_names)

    @property
    def optimized_iterations_pd(self) -> pd.DataFrame | None:
        """
        Returns all saved iterations from the optimizer and final estimate
        as a pandas.DataFrame which contains all optimizer outputs, i.e.,
        the value for `lp__` as well as all Stan program variables.

        """
        if not self.config.method_config.save_iterations:
            get_logger().warning(
                'Intermediate iterations not saved to CSV output file. '
                'Rerun the optimize method with "save_iterations=True".'
            )
            return None
        if not self.converged:
            get_logger().warning(
                'Invalid estimate, optimization failed to converge.'
            )
        self._assemble_draws()
        return pd.DataFrame(self._all_iters, columns=self.column_names)

    @property
    def optimized_params_dict(self) -> dict[str, np.float64]:
        """
        Returns all estimates from the optimizer, including `lp__` as a
        Python Dict.  Only returns estimate from final iteration.
        """
        if not self.converged:
            get_logger().warning(
                'Invalid estimate, optimization failed to converge.'
            )
        self._assemble_draws()
        return OrderedDict(zip(self.column_names, self._mle))

    def stan_variable(
        self,
        var: str,
        *,
        inc_iterations: bool = False,
        warn: bool = True,
    ) -> np.ndarray:
        """
        Return a numpy.ndarray which contains the estimates for the
        for the named Stan program variable where the dimensions of the
        numpy.ndarray match the shape of the Stan program variable.

        This functionaltiy is also available via a shortcut using ``.`` -
        writing ``fit.a`` is a synonym for ``fit.stan_variable("a")``

        :param var: variable name

        :param inc_iterations: When ``True`` and the intermediate estimates
            are included in the output, i.e., the optimizer was run with
            ``save_iterations=True``, then intermediate estimates are included.
            Default value is ``False``.

        See Also
        --------
        CmdStanMLE.stan_variables
        CmdStanMCMC.stan_variable
        CmdStanPathfinder.stan_variable
        CmdStanVB.stan_variable
        CmdStanGQ.stan_variable
        CmdStanLaplace.stan_variable
        """
        if var not in self.metadata.stan_vars:
            raise ValueError(
                f'Unknown variable name: {var}\n'
                'Available variables are ' + ", ".join(self.metadata.stan_vars)
            )
        save_iterations = self.config.method_config.save_iterations
        if warn and inc_iterations and not save_iterations:
            get_logger().warning(
                'Intermediate iterations not saved to CSV output file. '
                'Rerun the optimize method with "save_iterations=True".'
            )
        if warn and not self.converged:
            get_logger().warning(
                'Invalid estimate, optimization failed to converge.'
            )
        self._assemble_draws()
        if inc_iterations and save_iterations:
            data = self._all_iters
        else:
            data = self._mle

        try:
            out: np.ndarray = self.metadata.stan_vars[var].extract_reshape(data)
            return out
        except KeyError:
            # pylint: disable=raise-missing-from
            raise ValueError(
                f'Unknown variable name: {var}\n'
                'Available variables are '
                + ", ".join(self.metadata.stan_vars.keys())
            )

    def stan_variables(
        self, inc_iterations: bool = False
    ) -> dict[str, np.ndarray]:
        """
        Return a dictionary mapping Stan program variables names
        to the corresponding numpy.ndarray containing the inferred values.

        :param inc_iterations: When ``True`` and the intermediate estimates
            are included in the output, i.e., the optimizer was run with
            ``save_iterations=True``, then intermediate estimates are included.
            Default value is ``False``.


        See Also
        --------
        CmdStanMLE.stan_variable
        CmdStanMCMC.stan_variables
        CmdStanPathfinder.stan_variables
        CmdStanVB.stan_variables
        CmdStanGQ.stan_variables
        CmdStanLaplace.stan_variables
        """
        if not self.converged:
            get_logger().warning(
                'Invalid estimate, optimization failed to converge.'
            )
        result = {}
        for name in self.metadata.stan_vars:
            result[name] = self.stan_variable(
                name, inc_iterations=inc_iterations, warn=False
            )
        return result

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
