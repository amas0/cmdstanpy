"""Shared base classes for containers of CmdStan inference results."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Generic

import numpy as np

from cmdstanpy.utils import stancsv

from .metadata import InferenceMetadata, MethodT, StanConfig


@dataclass(kw_only=True)
class StanFit(Generic[MethodT]):
    """
    Base container for the outputs of a CmdStan run: the metadata parsed
    from the Stan CSV header, the run configuration parsed from the config
    JSON, and the locations of the output files on disk.

    Draws are loaded lazily from the output CSV file(s) on first access.
    """

    metadata: InferenceMetadata
    model_name: str
    config: StanConfig[MethodT]

    # Attributes naming output files; each holds a path, an optional path,
    # or a list of paths. ``save_output_files`` moves every named file.
    _FILE_ATTRS: ClassVar[tuple[str, ...]] = ()

    def _assemble(self) -> None:
        """Load lazily-parsed outputs from the output files."""
        raise NotImplementedError

    def stan_variable(self, var: str) -> np.ndarray:
        """
        Return a numpy.ndarray which contains the values for the named
        Stan program variable.
        """
        raise NotImplementedError

    def stan_variables(self) -> dict[str, np.ndarray]:
        """
        Return a dictionary mapping Stan program variables names
        to the corresponding numpy.ndarray containing the inferred values.

        See Also
        --------
        StanFit.stan_variable
        """
        result = {}
        for name in self.metadata.stan_vars:
            result[name] = self.stan_variable(name)
        return result

    def _extract_stan_var(self, var: str, draws: np.ndarray) -> np.ndarray:
        """Reshape the draws for one Stan program variable, raising a
        ValueError naming the available variables when it is unknown."""
        try:
            out: np.ndarray = self.metadata.stan_vars[var].extract_reshape(
                draws
            )
            return out
        except KeyError:
            raise ValueError(
                f'Unknown variable name: {var}\n'
                'Available variables are '
                + ", ".join(self.metadata.stan_vars.keys())
            ) from None

    @property
    def column_names(self) -> tuple[str, ...]:
        """
        Names of all outputs from the method, comprising method diagnostics
        and all components of all model parameters, transformed parameters,
        and quantities of interest. Corresponds to Stan CSV file header row,
        with names munged to array notation, e.g. `beta[1]` not `beta.1`.
        """
        return self.metadata.column_names

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
        # for details. We call _assemble to ensure lazily-parsed outputs have
        # been loaded prior to serialization.
        self._assemble()
        return self.__dict__

    def save_output_files(self, dir: str | None = None) -> None:
        """
        Move the output CSV file(s), and any associated config, metric, and
        stdout files, to the specified directory. Updates the corresponding
        attributes on this object to point at the new locations.

        :param dir: directory path

        See Also
        --------
        cmdstanpy.from_output_files
        """
        dest = Path(dir) if dir is not None else Path.cwd()
        try:
            dest.mkdir(parents=True, exist_ok=True)
            probe = dest / f'.cmdstanpy-write-test-{os.getpid()}'
            probe.touch()
            probe.unlink()
        except OSError as exc:
            raise RuntimeError(f'Cannot save to path: {dest}') from exc

        def _move(src: str, is_csv: bool) -> str:
            if not os.path.exists(src):
                if is_csv:
                    raise ValueError(f'Cannot access CSV file {src}')
                return src  # leave e.g. a deleted stdout file as-is
            dst = dest / Path(src).name
            if dst.exists():
                raise ValueError(f'File exists, not overwriting: {dst}')
            shutil.move(src, dst)
            return os.fspath(dst)

        for attr in self._FILE_ATTRS:
            val = getattr(self, attr)
            is_csv = attr in ('csv_file', 'csv_files')
            if val is None:
                continue
            if isinstance(val, list):
                setattr(self, attr, [_move(f, is_csv) for f in val])
            else:
                setattr(self, attr, _move(val, is_csv))


@dataclass(kw_only=True)
class SingleFileFit(StanFit[MethodT]):
    """
    Base container for the outputs of a CmdStan run whose draws are held
    in a single Stan CSV file.
    """

    csv_file: str
    config_file: str | None = None  # None if config object passed directly
    stdout_file: str | None = None

    _FILE_ATTRS: ClassVar[tuple[str, ...]] = (
        'csv_file',
        'config_file',
        'stdout_file',
    )

    _draws: np.ndarray = field(default_factory=lambda: np.array(()), init=False)

    @classmethod
    def _from_files_kwargs(
        cls,
        csv_file: str | os.PathLike,
        config_file: str | os.PathLike,
        stdout_file: str | os.PathLike | None,
        config_cls: type[StanConfig[MethodT]],
    ) -> dict[str, Any]:
        """Common constructor arguments parsed from the output files."""
        with open(config_file) as f:
            stan_config = config_cls.from_json(f.read())

        return {
            'metadata': InferenceMetadata.from_csv(csv_file),
            'model_name': stan_config.model_name,
            'csv_file': os.fspath(csv_file),
            'config': stan_config,
            'config_file': os.fspath(config_file),
            'stdout_file': (
                os.fspath(stdout_file) if stdout_file is not None else None
            ),
        }

    def _assemble(self) -> None:
        if self._draws.shape != (0,):
            return

        try:
            *_, draws = stancsv.parse_comments_header_and_draws(self.csv_file)
            self._draws = stancsv.csv_bytes_list_to_numpy(draws)
        except Exception as exc:
            raise ValueError(
                f"An error occurred when parsing Stan csv {self.csv_file}"
            ) from exc

    def draws(self) -> np.ndarray:
        """
        Return a numpy.ndarray containing the draws. This is a 2-D array
        of shape (draws, parameters).
        """
        self._assemble()
        return self._draws

    def stan_variable(self, var: str) -> np.ndarray:
        """
        Return a numpy.ndarray which contains the values for the
        named Stan program variable where the dimensions of the
        numpy.ndarray match the shape of the Stan program variable.

        This functionality is also available via a shortcut using ``.`` -
        writing ``fit.a`` is a synonym for ``fit.stan_variable("a")``

        :param var: variable name

        See Also
        --------
        StanFit.stan_variables
        """
        self._assemble()
        return self._extract_stan_var(var, self._draws)

    def method_variables(self) -> dict[str, np.ndarray]:
        """
        Returns a dictionary of all method variables, i.e., all
        output column names ending in `__`.  Assumes that all variables
        are scalar variables where column name is variable name.
        Maps each column name to a numpy.ndarray containing
        per-draw diagnostic values.
        """
        self._assemble()
        return {
            name: var.extract_reshape(self._draws)
            for name, var in self.metadata.method_vars.items()
        }

    def create_inits(
        self, seed: int | None = None, chains: int = 4
    ) -> list[dict[str, np.ndarray]] | dict[str, np.ndarray]:
        """
        Create initial values for the parameters of the model
        by randomly selecting draws from the fit.

        :param seed: Used for random selection, defaults to None
        :param chains: Number of initial values to return, defaults to 4
        :return: The initial values for the parameters of the model.

        If ``chains`` is 1, a dictionary is returned, otherwise a list
        of dictionaries is returned, in the format expected for the
        ``inits`` argument of :meth:`CmdStanModel.sample`.
        """
        self._assemble()
        rng = np.random.default_rng(seed)
        idxs = rng.choice(self._draws.shape[0], size=chains, replace=False)
        if chains == 1:
            draw = self._draws[idxs[0]]
            return {
                name: var.extract_reshape(draw)
                for name, var in self.metadata.stan_vars.items()
            }
        return [
            {
                name: var.extract_reshape(self._draws[idx])
                for name, var in self.metadata.stan_vars.items()
            }
            for idx in idxs
        ]
