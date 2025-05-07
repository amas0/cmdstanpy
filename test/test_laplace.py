"""Tests for the Laplace sampling method."""

import os
from tempfile import TemporaryDirectory

import numpy as np
import pytest

import cmdstanpy
from cmdstanpy.stanfit import from_csv

HERE = os.path.dirname(os.path.abspath(__file__))
DATAFILES_PATH = os.path.join(HERE, 'data')


def test_laplace_from_opt_csv():
    model_file = os.path.join(DATAFILES_PATH, 'optimize', 'rosenbrock.stan')
    model = cmdstanpy.CmdStanModel(stan_file=model_file)
    fit = model.laplace_sample(
        data={},
        mode=os.path.join(DATAFILES_PATH, 'optimize', 'rosenbrock_mle.csv'),
        jacobian=False,
    )
    assert 'x' in fit.stan_variables()
    assert 'y' in fit.stan_variables()
    assert isinstance(fit.mode, cmdstanpy.CmdStanMLE)


def test_laplace_from_csv():
    model_file = os.path.join(DATAFILES_PATH, 'optimize', 'rosenbrock.stan')
    model = cmdstanpy.CmdStanModel(stan_file=model_file)
    fit = model.laplace_sample(
        data={},
        seed=1234,
    )
    fit2 = from_csv(fit._runset.csv_files)
    assert isinstance(fit2, cmdstanpy.CmdStanLaplace)
    assert 'x' in fit2.stan_variables()
    assert 'y' in fit2.stan_variables()
    assert isinstance(fit2.mode, cmdstanpy.CmdStanMLE)

    with TemporaryDirectory() as dir:
        model.laplace_sample(data={}, seed=1234, output_dir=dir)

        fit3 = from_csv(
            [
                os.path.join(dir, f)
                for f in os.listdir(dir)
                if f.endswith(".csv") and "opt" not in f
            ]
        )
        assert isinstance(fit3, cmdstanpy.CmdStanLaplace)
        assert 'x' in fit3.stan_variables()
        assert 'y' in fit3.stan_variables()
        assert isinstance(fit3.mode, cmdstanpy.CmdStanMLE)


def test_laplace_runs_opt():
    model_file = os.path.join(DATAFILES_PATH, 'optimize', 'rosenbrock.stan')
    model = cmdstanpy.CmdStanModel(stan_file=model_file)
    fit1 = model.laplace_sample(data={}, seed=1234, opt_args={'iter': 1003})
    assert isinstance(fit1.mode, cmdstanpy.CmdStanMLE)

    assert fit1.mode.metadata.cmdstan_config['seed'] == 1234
    assert fit1.metadata.cmdstan_config['seed'] == 1234
    assert fit1.mode.metadata.cmdstan_config['iter'] == 1003


def test_laplace_bad_jacobian_mismatch():
    model_file = os.path.join(DATAFILES_PATH, 'optimize', 'rosenbrock.stan')
    model = cmdstanpy.CmdStanModel(stan_file=model_file)
    with pytest.raises(ValueError):
        model.laplace_sample(
            data={},
            mode=os.path.join(DATAFILES_PATH, 'optimize', 'rosenbrock_mle.csv'),
            jacobian=True,
        )


def test_laplace_bad_two_modes():
    model_file = os.path.join(DATAFILES_PATH, 'optimize', 'rosenbrock.stan')
    model = cmdstanpy.CmdStanModel(stan_file=model_file)
    with pytest.raises(ValueError):
        model.laplace_sample(
            data={},
            mode=os.path.join(DATAFILES_PATH, 'optimize', 'rosenbrock_mle.csv'),
            opt_args={'iter': 1003},
            jacobian=False,
        )


def test_laplace_outputs():
    model_file = os.path.join(DATAFILES_PATH, 'optimize', 'rosenbrock.stan')
    model = cmdstanpy.CmdStanModel(stan_file=model_file)
    fit = model.laplace_sample(data={}, seed=1234, draws=123)

    variables = fit.stan_variables()
    assert 'x' in variables
    assert 'y' in variables
    assert variables['x'].shape == (123,)

    np.testing.assert_array_equal(variables['x'], fit.x)

    fit_pd = fit.draws_pd()
    assert 'x' in fit_pd.columns
    assert 'y' in fit_pd.columns
    assert fit_pd['x'].shape == (123,)


def test_laplace_create_inits():
    stan = os.path.join(DATAFILES_PATH, 'bernoulli.stan')
    bern_model = cmdstanpy.CmdStanModel(stan_file=stan)
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')

    laplace = bern_model.laplace_sample(data=jdata)

    inits = laplace.create_inits()
    assert isinstance(inits, list)
    assert len(inits) == 4
    assert isinstance(inits[0], dict)
    assert 'theta' in inits[0]

    inits_10 = laplace.create_inits(chains=10)
    assert isinstance(inits_10, list)
    assert len(inits_10) == 10

    inits_1 = laplace.create_inits(chains=1)
    assert isinstance(inits_1, dict)
    assert 'theta' in inits_1
    assert len(inits_1) == 1

    seeded = laplace.create_inits(seed=1234)
    seeded2 = laplace.create_inits(seed=1234)
    assert all(
        init1['theta'] == init2['theta']
        for init1, init2 in zip(seeded, seeded2)
    )


def test_laplace_init_sampling():
    stan = os.path.join(DATAFILES_PATH, 'logistic.stan')
    logistic_model = cmdstanpy.CmdStanModel(stan_file=stan)
    logistic_data = os.path.join(DATAFILES_PATH, 'logistic.data.R')

    laplace = logistic_model.laplace_sample(data=logistic_data)
    inits = laplace.create_inits()

    fit = logistic_model.sample(data=logistic_data, inits=inits)

    assert fit.chains == 4
    assert fit.draws().shape == (1000, 4, 9)
