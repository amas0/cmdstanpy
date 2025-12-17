"""Metadata tests"""

import json
import math
import os
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from cmdstanpy.cmdstan_args import CmdStanArgs, SamplerArgs
from cmdstanpy.stanfit import InferenceMetadata, RunSet
from cmdstanpy.stanfit.metadata import MetricInfo
from cmdstanpy.utils import EXTENSION, check_sampler_csv

HERE = os.path.dirname(os.path.abspath(__file__))
DATAFILES_PATH = os.path.join(HERE, 'data')

DATAFILES_PATH = os.path.join(HERE, 'data')
GOODFILES_PATH = os.path.join(DATAFILES_PATH, 'runset-good')
BADFILES_PATH = os.path.join(DATAFILES_PATH, 'runset-bad')


def test_good() -> None:
    # construct fit using existing sampler output
    exe = os.path.join(DATAFILES_PATH, 'bernoulli' + EXTENSION)
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
    sampler_args = SamplerArgs(
        iter_sampling=100, max_treedepth=11, adapt_delta=0.95
    )
    cmdstan_args = CmdStanArgs(
        model_name='bernoulli',
        model_exe=exe,
        chain_ids=[1, 2, 3, 4],
        seed=12345,
        data=jdata,
        output_dir=DATAFILES_PATH,
        method_args=sampler_args,
    )
    runset = RunSet(args=cmdstan_args)
    runset._csv_files = [
        os.path.join(DATAFILES_PATH, 'runset-good', 'bern-1.csv'),
        os.path.join(DATAFILES_PATH, 'runset-good', 'bern-2.csv'),
        os.path.join(DATAFILES_PATH, 'runset-good', 'bern-3.csv'),
        os.path.join(DATAFILES_PATH, 'runset-good', 'bern-4.csv'),
    ]
    retcodes = runset._retcodes
    for i in range(len(retcodes)):
        runset._set_retcode(i, 0)
    config = check_sampler_csv(
        path=runset.csv_files[i],
        iter_sampling=100,
        iter_warmup=1000,
        save_warmup=False,
        thin=1,
    )
    expected = 'Metadata:\n{}\n'.format(config)
    metadata = InferenceMetadata(config)
    actual = '{}'.format(metadata)
    assert expected == actual
    assert config == metadata.cmdstan_config

    hmc_vars = {
        'lp__',
        'accept_stat__',
        'stepsize__',
        'treedepth__',
        'n_leapfrog__',
        'divergent__',
        'energy__',
    }

    method_vars_cols = metadata.method_vars
    assert hmc_vars == method_vars_cols.keys()
    bern_model_vars = {'theta'}
    assert bern_model_vars == metadata.stan_vars.keys()


class TestMetricInfoValidators:
    """Test custom validators for MetricInfo model"""

    def test_valid_diag_e_metric(self) -> None:
        """Test valid diag_e metric with 1D list"""
        metric = MetricInfo(
            stepsize=0.5,
            metric_type="diag_e",
            inv_metric=[1.0, 2.0, 3.0],
        )
        assert metric.stepsize == 0.5
        assert isinstance(metric.inv_metric, list)
        assert isinstance(metric.inv_metric[0], float)
        assert len(metric.inv_metric) == 3

    def test_valid_unit_e_metric(self) -> None:
        """Test valid unit_e metric with 1D list"""
        metric = MetricInfo(
            stepsize=0.1,
            metric_type="unit_e",
            inv_metric=[1.0, 1.0, 1.0],
        )
        assert metric.metric_type == "unit_e"
        assert isinstance(metric.inv_metric[0], float)
        assert len(metric.inv_metric) == 3

    def test_valid_dense_e_metric(self) -> None:
        """Test valid dense_e metric with 2D square list"""
        metric = MetricInfo(
            stepsize=0.3,
            metric_type="dense_e",
            inv_metric=[[1.0, 0.5], [0.5, 1.0]],
        )
        assert metric.metric_type == "dense_e"
        assert isinstance(metric.inv_metric[0], list)
        assert len(metric.inv_metric) == 2
        assert len(metric.inv_metric[0]) == 2

    def test_inv_metric_stays_as_list(self) -> None:
        """Test that inv_metric remains as list type"""
        metric = MetricInfo(
            stepsize=0.5,
            metric_type="diag_e",
            inv_metric=[1.0, 2.0, 3.0],
        )
        assert isinstance(metric.inv_metric, list)

    def test_inv_metric_nested_list(self) -> None:
        """Test that inv_metric handles nested lists correctly"""
        metric = MetricInfo(
            stepsize=0.5,
            metric_type="dense_e",
            inv_metric=[[1.0, 0.0], [0.0, 1.0]],
        )
        assert isinstance(metric.inv_metric, list)
        assert isinstance(metric.inv_metric[0], list)

    def test_stepsize_positive(self) -> None:
        """Test valid positive stepsize"""
        metric = MetricInfo(
            stepsize=0.5,
            metric_type="diag_e",
            inv_metric=[1.0],
        )
        assert metric.stepsize == 0.5

    def test_stepsize_nan_allowed(self) -> None:
        """Test that NaN stepsize is allowed"""
        metric = MetricInfo(
            stepsize=math.nan,
            metric_type="diag_e",
            inv_metric=[1.0],
        )
        assert math.isnan(metric.stepsize)

    def test_stepsize_zero_raises_error(self) -> None:
        """Test that zero stepsize raises ValueError"""
        with pytest.raises(ValidationError) as exc_info:
            MetricInfo(
                stepsize=0.0,
                metric_type="diag_e",
                inv_metric=[1.0],
            )
        assert "stepsize must be greater than 0 or NaN" in str(exc_info.value)

    def test_stepsize_negative_raises_error(self) -> None:
        """Test that negative stepsize raises ValueError"""
        with pytest.raises(ValidationError) as exc_info:
            MetricInfo(
                stepsize=-0.5,
                metric_type="diag_e",
                inv_metric=[1.0],
            )
        assert "stepsize must be greater than 0 or NaN" in str(exc_info.value)

    def test_diag_e_with_2d_list_raises_error(self) -> None:
        """Test that diag_e with 2D list raises ValueError"""
        with pytest.raises(ValidationError) as exc_info:
            MetricInfo(
                stepsize=0.5,
                metric_type="diag_e",
                inv_metric=[[1.0, 2.0]],
            )
        assert "inv_metric must be 1D for diag_e and unit_e" in str(
            exc_info.value
        )

    def test_unit_e_with_2d_list_raises_error(self) -> None:
        """Test that unit_e with 2D list raises ValueError"""
        with pytest.raises(ValidationError) as exc_info:
            MetricInfo(
                stepsize=0.5,
                metric_type="unit_e",
                inv_metric=[[1.0], [1.0]],
            )
        assert "inv_metric must be 1D for diag_e and unit_e" in str(
            exc_info.value
        )

    def test_dense_e_with_1d_list_raises_error(self) -> None:
        """Test that dense_e with 1D list raises ValueError"""
        with pytest.raises(ValidationError) as exc_info:
            MetricInfo(
                stepsize=0.5,
                metric_type="dense_e",
                inv_metric=[1.0, 2.0],
            )
        assert "Dense inv_metric must be 2D" in str(exc_info.value)

    def test_dense_e_non_square_raises_error(self) -> None:
        """Test that dense_e with non-square list raises ValueError"""
        with pytest.raises(ValidationError) as exc_info:
            MetricInfo(
                stepsize=0.5,
                metric_type="dense_e",
                inv_metric=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            )
        assert "Dense inv_metric must be square" in str(exc_info.value)


class TestMetricInfoModelValidateJson:
    """Test model_validate_json class method"""

    def test_from_json_diag_e(self) -> None:
        """Test loading diag_e metric from JSON file"""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            json.dump(
                {
                    'stepsize': 0.5,
                    'metric_type': 'diag_e',
                    'inv_metric': [1.0, 2.0, 3.0],
                },
                f,
            )
            temp_path = f.name

        try:
            with open(temp_path) as f:
                metric = MetricInfo.model_validate_json(f.read())
            assert metric.stepsize == 0.5
            assert metric.metric_type == "diag_e"
            assert metric.inv_metric == [1.0, 2.0, 3.0]
        finally:
            Path(temp_path).unlink()

    def test_from_json_dense_e(self) -> None:
        """Test loading dense_e metric from JSON file"""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            json.dump(
                {
                    'stepsize': 0.3,
                    'metric_type': 'dense_e',
                    'inv_metric': [[1.0, 0.5], [0.5, 1.0]],
                },
                f,
            )
            temp_path = f.name

        try:
            with open(temp_path) as f:
                metric = MetricInfo.model_validate_json(f.read())
            assert metric.stepsize == 0.3
            assert metric.metric_type == "dense_e"
            assert len(metric.inv_metric) == 2
            assert len(metric.inv_metric[0]) == 2  # type: ignore
        finally:
            Path(temp_path).unlink()

    def test_from_json_invalid_data_raises_error(self) -> None:
        """Test that invalid data in JSON raises ValidationError"""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            json.dump(
                {
                    'stepsize': -0.5,  # Invalid: negative stepsize
                    'metric_type': 'diag_e',
                    'inv_metric': [1.0, 2.0, 3.0],
                },
                f,
            )
            temp_path = f.name

        try:
            with pytest.raises(ValidationError):
                with open(temp_path) as f:
                    MetricInfo.model_validate_json(f.read())
        finally:
            Path(temp_path).unlink()

    def test_from_json_pathlike(self) -> None:
        """Test from_json works with PathLike objects"""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            json.dump(
                {
                    'stepsize': 0.5,
                    'metric_type': 'unit_e',
                    'inv_metric': [1.0, 1.0],
                },
                f,
            )
            temp_path = Path(f.name)

        try:
            with open(temp_path) as f:
                metric = MetricInfo.model_validate_json(f.read())
            assert metric.metric_type == "unit_e"
        finally:
            temp_path.unlink()

    def test_invalid_metric_type_raises_error(self) -> None:
        with pytest.raises(ValidationError):
            MetricInfo(
                stepsize=0.5,
                metric_type="not_a_metric",  # type: ignore
                inv_metric=[1.0],
            )

    def test_from_json_invalid_metric_type_raises_error(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            json.dump(
                {
                    'stepsize': 0.5,
                    'metric_type': 'not_a_metric',
                    'inv_metric': [1.0, 2.0, 3.0],
                },
                f,
            )
            temp_path = f.name

        try:
            with pytest.raises(ValidationError):
                with open(temp_path) as fh:
                    MetricInfo.model_validate_json(fh.read())
        finally:
            Path(temp_path).unlink()
