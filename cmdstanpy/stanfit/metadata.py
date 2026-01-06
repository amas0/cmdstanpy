"""Container for metadata parsed from the output of a CmdStan run"""

from __future__ import annotations

import copy
import math
import os
from typing import Any, Iterator, Literal

import stanio
from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeInt,
    field_validator,
    model_validator,
)

from cmdstanpy.utils import stancsv


class InferenceMetadata:
    """
    CmdStan configuration and contents of output file parsed out of
    the Stan CSV file header comments and column headers.
    Assumes valid CSV files.
    """

    def __init__(
        self, config: dict[str, str | int | float | tuple[str, ...]]
    ) -> None:
        """Initialize object from CSV headers"""
        self._cmdstan_config = config

        vars = stanio.parse_header(config['raw_header'])  # type: ignore

        self._method_vars = {
            k: v for (k, v) in vars.items() if k.endswith('__')
        }
        self._stan_vars = {
            k: v for (k, v) in vars.items() if not k.endswith('__')
        }

    @classmethod
    def from_csv(
        cls, stan_csv: str | os.PathLike | Iterator[bytes]
    ) -> InferenceMetadata:
        try:
            comments, header, _ = stancsv.parse_comments_header_and_draws(
                stan_csv
            )
            return cls(stancsv.construct_config_header_dict(comments, header))
        except Exception as exc:
            raise ValueError(
                f"An error occurred when parsing Stan csv {stan_csv}"
            ) from exc

    def __repr__(self) -> str:
        return 'Metadata:\n{}\n'.format(self._cmdstan_config)

    def __getitem__(self, key: str) -> str | int | float | tuple[str, ...]:
        return self._cmdstan_config[key]

    @property
    def cmdstan_config(self) -> dict[str, Any]:
        """
        Returns a dictionary containing a set of name, value pairs
        parsed out of the Stan CSV file header.  These include the
        command configuration and the CSV file header row information.
        Uses deepcopy for immutability.
        """
        return copy.deepcopy(self._cmdstan_config)

    @property
    def column_names(self) -> tuple[str, ...]:
        col_names = self['column_names']
        return col_names  # type: ignore

    @property
    def method_vars(self) -> dict[str, stanio.Variable]:
        """
        Method variable names always end in `__`, e.g. `lp__`.
        """
        return self._method_vars

    @property
    def stan_vars(self) -> dict[str, stanio.Variable]:
        """
        These are the user-defined variables in the Stan program.
        """
        return self._stan_vars


class MetricInfo(BaseModel):
    """Structured representation of HMC-NUTS metric information,
    as output by CmdStan"""

    stepsize: float
    metric_type: Literal["diag_e", "dense_e", "unit_e"]
    inv_metric: list[float] | list[list[float]]

    @field_validator("stepsize")
    @classmethod
    def validate_stepsize(cls, v: float) -> float:
        if not math.isnan(v) and v <= 0:
            raise ValueError("stepsize must be greater than 0 or NaN")
        return v

    @model_validator(mode="after")
    def validate_inv_metric_shape(self) -> MetricInfo:
        if not self.inv_metric:  # Empty inv_metric, e.g. from no parameters
            return self

        is_1d = isinstance(self.inv_metric[0], float)

        if self.metric_type in ("diag_e", "unit_e") and not is_1d:
            raise ValueError(
                "inv_metric must be 1D for diag_e and unit_e metric type"
            )
        if self.metric_type == "dense_e":
            if is_1d:
                raise ValueError("Dense inv_metric must be 2D")

            if any(not row for row in self.inv_metric):
                raise ValueError("Dense inv_metric cannot contain empty rows")

            n_rows = len(self.inv_metric)
            if not all(
                len(row) == n_rows for row in self.inv_metric  # type: ignore
            ):
                raise ValueError("Dense inv_metric must be square")

        return self


class ConfigInfo(BaseModel):
    """Structured representation of a config JSON file output as part of a
    Stan inference run. Only required fields are fully validated."""

    model_config = ConfigDict(extra="allow")

    stan_major_version: NonNegativeInt
    stan_minor_version: NonNegativeInt
    stan_patch_version: NonNegativeInt
    model_name: str
