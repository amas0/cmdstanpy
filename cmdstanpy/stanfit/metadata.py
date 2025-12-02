"""Container for metadata parsed from the output of a CmdStan run"""

from __future__ import annotations

import copy
import json
import os
from typing import Any, Iterator, Literal

import numpy as np
import stanio
from pydantic import BaseModel, Field, field_validator, model_validator

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
    ) -> 'InferenceMetadata':
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

    chain_id: int = Field(gt=0)
    stepsize: float = Field(gt=0)
    metric_type: Literal["diag_e", "dense_e", "unit_e"]
    inv_metric: np.ndarray

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("inv_metric", mode="before")
    def convert_inv_metric(  # pylint: disable=no-self-argument
        cls, v: Any
    ) -> np.ndarray:

        return np.asarray(v)

    @model_validator(mode="after")
    def validate_inv_metric_shape(self) -> MetricInfo:
        if (
            self.metric_type in ("diag_e", "unit_e")
            and self.inv_metric.ndim != 1
        ):
            raise ValueError(
                "inv_metric must be 1D for diag_e and unit_e metric type"
            )
        if self.metric_type == "dense_e":
            if self.inv_metric.ndim != 2:
                raise ValueError("Dense inv_metric must be 2D")
            if self.inv_metric.shape[0] != self.inv_metric.shape[1]:
                raise ValueError("Dense inv_metric must be square")

        return self

    @classmethod
    def from_json(cls, file: str | os.PathLike, chain_id: int) -> MetricInfo:
        """Parse and validate a metric json given a file path and chain_id"""
        with open(file) as f:
            info_dict = json.load(f)

        info_dict['chain_id'] = chain_id
        return cls.model_validate(info_dict)  # type: ignore
