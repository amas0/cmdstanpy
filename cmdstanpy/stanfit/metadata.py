"""Container for metadata parsed from the output of a CmdStan run"""

from __future__ import annotations

import copy
import json
import math
import os
from typing import Annotated, Any, Iterator, Literal

import stanio
from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
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


class SampleConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    method: Literal["sample"] = "sample"
    algorithm: str
    num_samples: int
    num_warmup: int
    save_warmup: bool = False
    thin: int = 1
    max_depth: int | None = None


class OptimizeConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    method: Literal["optimize"] = "optimize"
    algorithm: str
    save_iterations: bool = False
    jacobian: bool = False


class StanConfig(BaseModel):
    """Common representation of a config JSON file output as part of a
    Stan inference run. Separate method-specific config classes handle
    the variation of output between methods."""

    model_config = ConfigDict(extra="allow")

    model_name: str
    stan_major_version: str
    stan_minor_version: str
    stan_patch_version: str

    method_config: Annotated[
        SampleConfig | OptimizeConfig,
        Discriminator('method'),
    ]


def flatten_value_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively flatten CmdStan's nested value/subdict structure.

    CmdStan uses a pattern where a field contains:
        {"value": "val", "val": {"k1": v1, "k2": v2}}

    This flattens it to the parent level as:
        {"field": "val", "k1": v1, "k2": v2}

    The flattening is applied recursively to any nested dicts.
    """
    result: dict[str, Any] = {}

    for key, value in data.items():
        if not isinstance(value, dict):
            result[key] = value
            continue

        # Check if this is a value/subdict pattern
        if 'value' in value:
            value_name = value['value']
            result[key] = value_name

            # Get the nested dict matching the value name and flatten it
            nested = value.get(value_name, {})
            if isinstance(nested, dict):
                # Recursively flatten the nested dict first
                flattened_nested = flatten_value_dict(nested)
                # Merge into result (nested keys go to parent level)
                for nested_key, nested_value in flattened_nested.items():
                    if nested_key not in result:
                        result[nested_key] = nested_value
        else:
            # Regular dict without value pattern - recurse into it
            result[key] = flatten_value_dict(value)

    return result


def flatten_config(data: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested CmdStan config JSON structure.

    CmdStan outputs config JSON with deeply nested structure like:
        {"method": {"value": "sample", "sample": {"num_samples": 1000, ...}}}

    This flattens it to:
        {"method_config": {"method": "sample", "num_samples": 1000, ...}, ...}
    """
    method_data = data.get('method')
    if not isinstance(method_data, dict):
        return data

    result = {k: v for k, v in data.items() if k != "method"}
    method_name = method_data.get('value')

    # Build method_config from the method-specific nested dict
    nested_method = method_data.get(method_name, {})
    method_config = flatten_value_dict(nested_method)
    method_config['method'] = method_name

    result['method_config'] = method_config
    return result


def parse_config(json_data: str | bytes) -> StanConfig:
    """Parse a CmdStan config JSON string into a StanConfig."""

    raw = json.loads(json_data)
    flat = flatten_config(raw)
    return StanConfig.model_validate(flat)  # type: ignore
