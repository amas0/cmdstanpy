"""
Container for the result of running a laplace approximation.
"""

from __future__ import annotations

import os
from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any, MutableMapping

import numpy as np
import pandas as pd

try:
    import xarray as xr

    XARRAY_INSTALLED = True
except ImportError:
    XARRAY_INSTALLED = False

from cmdstanpy.utils.data_munging import build_xarray_data

from .base import SingleFileFit
from .metadata import LaplaceConfig, LaplaceRunConfig
from .mle import CmdStanMLE

# TODO list:
# - docs and example notebook


@dataclass(kw_only=True)
class CmdStanLaplace(SingleFileFit[LaplaceConfig]):
    """
    Container for outputs from the Laplace approximation.
    Created by :meth:`CmdStanModel.laplace_sample`.
    """

    mode: CmdStanMLE

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

        kwargs = cls._from_files_kwargs(
            csv_file, config_file, stdout_file, LaplaceRunConfig
        )
        if mode is None:
            mode_file = kwargs['config'].method_config.mode
            mle = from_output_files(mode_file, method='optimize')
            if not isinstance(mle, CmdStanMLE):
                raise TypeError(
                    f'Expected CmdStanMLE from mode file '
                    f'{mode_file!r}, got {type(mle).__name__}'
                )
            mode = mle
        return cls(mode=mode, **kwargs)

    def draws_pd(
        self,
        vars: list[str] | str | None = None,
    ) -> pd.DataFrame:
        if vars is not None:
            if isinstance(vars, str):
                vars_list = [vars]
            else:
                vars_list = vars

        self._assemble()
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

        self._assemble()

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
