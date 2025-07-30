"""testing stancsv parsing"""

import os
from pathlib import Path
from test import without_import
from typing import List

import numpy as np
import pytest

import cmdstanpy
from cmdstanpy.utils import stancsv

HERE = os.path.dirname(os.path.abspath(__file__))
DATAFILES_PATH = os.path.join(HERE, 'data')


def test_csv_bytes_to_numpy_no_header():
    lines = [
        b"-6.76206,1,0.787025,1,1,0,6.81411,0.229458\n",
        b"-6.81411,0.983499,0.787025,1,1,0,6.8147,0.20649\n",
        b"-6.85511,0.994945,0.787025,2,3,0,6.85536,0.310589\n",
        b"-6.85511,0.812189,0.787025,1,1,0,7.16517,0.310589\n",
    ]
    expected = np.array(
        [
            [-6.76206, 1, 0.787025, 1, 1, 0, 6.81411, 0.229458],
            [-6.81411, 0.983499, 0.787025, 1, 1, 0, 6.8147, 0.20649],
            [-6.85511, 0.994945, 0.787025, 2, 3, 0, 6.85536, 0.310589],
            [-6.85511, 0.812189, 0.787025, 1, 1, 0, 7.16517, 0.310589],
        ],
        dtype=np.float64,
    )
    arr_out = stancsv.csv_bytes_list_to_numpy(lines, includes_header=False)
    assert np.array_equal(arr_out, expected)
    assert arr_out[0].dtype == np.float64


def test_csv_bytes_to_numpy_no_header_no_polars():
    lines = [
        b"-6.76206,1,0.787025,1,1,0,6.81411,0.229458\n",
        b"-6.81411,0.983499,0.787025,1,1,0,6.8147,0.20649\n",
        b"-6.85511,0.994945,0.787025,2,3,0,6.85536,0.310589\n",
        b"-6.85511,0.812189,0.787025,1,1,0,7.16517,0.310589\n",
    ]
    expected = np.array(
        [
            [-6.76206, 1, 0.787025, 1, 1, 0, 6.81411, 0.229458],
            [-6.81411, 0.983499, 0.787025, 1, 1, 0, 6.8147, 0.20649],
            [-6.85511, 0.994945, 0.787025, 2, 3, 0, 6.85536, 0.310589],
            [-6.85511, 0.812189, 0.787025, 1, 1, 0, 7.16517, 0.310589],
        ],
        dtype=np.float64,
    )
    with without_import("polars", cmdstanpy.utils.stancsv):
        arr_out = stancsv.csv_bytes_list_to_numpy(lines, includes_header=False)
        assert np.array_equal(arr_out, expected)
        assert arr_out[0].dtype == np.float64


def test_csv_bytes_to_numpy_with_header():
    lines = [
        (
            b"lp__,accept_stat__,stepsize__,treedepth__,"
            b"n_leapfrog__,divergent__,energy__,theta\n"
        ),
        b"-6.76206,1,0.787025,1,1,0,6.81411,0.229458\n",
        b"-6.81411,0.983499,0.787025,1,1,0,6.8147,0.20649\n",
        b"-6.85511,0.994945,0.787025,2,3,0,6.85536,0.310589\n",
        b"-6.85511,0.812189,0.787025,1,1,0,7.16517,0.310589\n",
    ]
    expected = np.array(
        [
            [-6.76206, 1, 0.787025, 1, 1, 0, 6.81411, 0.229458],
            [-6.81411, 0.983499, 0.787025, 1, 1, 0, 6.8147, 0.20649],
            [-6.85511, 0.994945, 0.787025, 2, 3, 0, 6.85536, 0.310589],
            [-6.85511, 0.812189, 0.787025, 1, 1, 0, 7.16517, 0.310589],
        ],
        dtype=np.float64,
    )
    arr_out = stancsv.csv_bytes_list_to_numpy(lines, includes_header=True)
    assert np.array_equal(arr_out, expected)


def test_csv_bytes_to_numpy_single_element():
    lines = [
        b"-6.76206\n",
    ]
    expected = np.array(
        [
            [-6.76206],
        ],
        dtype=np.float64,
    )
    arr_out = stancsv.csv_bytes_list_to_numpy(lines, includes_header=False)
    assert np.array_equal(arr_out, expected)


def test_csv_bytes_to_numpy_single_element_no_polars():
    lines = [
        b"-6.76206\n",
    ]
    expected = np.array(
        [
            [-6.76206],
        ],
        dtype=np.float64,
    )
    with without_import("polars", cmdstanpy.utils.stancsv):
        arr_out = stancsv.csv_bytes_list_to_numpy(lines, includes_header=False)
        assert np.array_equal(arr_out, expected)


def test_csv_bytes_to_numpy_with_header_no_polars():
    lines = [
        (
            b"lp__,accept_stat__,stepsize__,treedepth__,"
            b"n_leapfrog__,divergent__,energy__,theta\n"
        ),
        b"-6.76206,1,0.787025,1,1,0,6.81411,0.229458\n",
        b"-6.81411,0.983499,0.787025,1,1,0,6.8147,0.20649\n",
        b"-6.85511,0.994945,0.787025,2,3,0,6.85536,0.310589\n",
        b"-6.85511,0.812189,0.787025,1,1,0,7.16517,0.310589\n",
    ]
    expected = np.array(
        [
            [-6.76206, 1, 0.787025, 1, 1, 0, 6.81411, 0.229458],
            [-6.81411, 0.983499, 0.787025, 1, 1, 0, 6.8147, 0.20649],
            [-6.85511, 0.994945, 0.787025, 2, 3, 0, 6.85536, 0.310589],
            [-6.85511, 0.812189, 0.787025, 1, 1, 0, 7.16517, 0.310589],
        ],
        dtype=np.float64,
    )
    with without_import("polars", cmdstanpy.utils.stancsv):
        arr_out = stancsv.csv_bytes_list_to_numpy(lines, includes_header=True)
        assert np.array_equal(arr_out, expected)


def test_csv_bytes_to_numpy_empty():
    lines = [b""]
    with pytest.raises(ValueError):
        stancsv.csv_bytes_list_to_numpy(lines)


def test_csv_bytes_to_numpy_empty_no_polars():
    lines = [b""]
    with without_import("polars", cmdstanpy.utils.stancsv):
        with pytest.raises(ValueError):
            stancsv.csv_bytes_list_to_numpy(lines)


def test_csv_bytes_to_numpy_header_no_draws():
    lines = [
        (
            b"lp__,accept_stat__,stepsize__,treedepth__,"
            b"n_leapfrog__,divergent__,energy__,theta\n"
        ),
    ]
    with pytest.raises(ValueError):
        stancsv.csv_bytes_list_to_numpy(lines)


def test_csv_bytes_to_numpy_header_no_draws_no_polars():
    lines = [
        (
            b"lp__,accept_stat__,stepsize__,treedepth__,"
            b"n_leapfrog__,divergent__,energy__,theta\n"
        ),
    ]
    with without_import("polars", cmdstanpy.utils.stancsv):
        with pytest.raises(ValueError):
            stancsv.csv_bytes_list_to_numpy(lines)


def test_parse_comments_and_draws():
    lines: List[bytes] = [b"# 1\n", b"2\n", b"3\n", b"# 4\n"]
    comment_lines, draws_lines = stancsv.parse_stan_csv_comments_and_draws(
        iter(lines)
    )

    assert comment_lines == [b"# 1\n", b"# 4\n"]
    assert draws_lines == [b"2\n", b"3\n"]


def test_parsing_adaptation_lines():
    lines = [
        b"# Adaptation terminated\n",
        b"# Step size = 0.787025\n",
        b"# Diagonal elements of inverse mass matrix:\n",
        b"# 1\n",
    ]
    step_size, mass_matrix = stancsv.parse_hmc_adaptation_lines(lines)
    assert step_size == 0.787025
    print(mass_matrix)
    assert mass_matrix == 1


def test_parsing_adaptation_lines_diagonal():
    lines = [
        b"diag_e",  # Will be present in the Stan CSV config
        b"# Adaptation terminated\n",
        b"# Step size = 0.787025\n",
        b"# Diagonal elements of inverse mass matrix:\n",
        b"# 1,2,3\n",
    ]
    step_size, mass_matrix = stancsv.parse_hmc_adaptation_lines(lines)
    assert step_size == 0.787025
    assert mass_matrix is not None
    assert np.array_equal(mass_matrix, np.array([1, 2, 3]))


def test_parsing_adaptation_lines_dense():
    lines = [
        b"# Adaptation terminated\n",
        b"# Step size = 0.775147\n",
        b"# Elements of inverse mass matrix:\n",
        b"# 2.84091, 0.230843, 0.0509365\n",
        b"# 0.230843, 3.92459, 0.126989\n",
        b"# 0.0509365, 0.126989, 3.82718\n",
    ]
    step_size, mass_matrix = stancsv.parse_hmc_adaptation_lines(lines)
    expected = np.array(
        [
            [2.84091, 0.230843, 0.0509365],
            [0.230843, 3.92459, 0.126989],
            [0.0509365, 0.126989, 3.82718],
        ],
        dtype=np.float64,
    )
    assert step_size == 0.775147
    assert mass_matrix is not None
    assert np.array_equal(mass_matrix, expected)


def test_parsing_adaptation_lines_missing_step_size():
    lines = [
        b"# Adaptation terminated\n",
        b"# Elements of inverse mass matrix:\n",
        b"# 2.84091, 0.230843, 0.0509365\n",
        b"# 0.230843, 3.92459, 0.126989\n",
        b"# 0.0509365, 0.126989, 3.82718\n",
    ]
    with pytest.raises(ValueError):
        stancsv.parse_hmc_adaptation_lines(lines)


def test_parsing_adaptation_lines_no_free_params():
    lines = [
        b"# Adaptation terminated\n",
        b"# Step size = 1.77497\n",
        b"# No free parameters for unit metric\n",
    ]
    _, mass_matrix = stancsv.parse_hmc_adaptation_lines(lines)
    assert mass_matrix is None


def test_csv_polars_and_numpy_equiv():
    lines = [
        b"-6.76206,1,0.787025,1,1,0,6.81411,0.229458\n",
        b"-6.81411,0.983499,0.787025,1,1,0,6.8147,0.20649\n",
        b"-6.85511,0.994945,0.787025,2,3,0,6.85536,0.310589\n",
        b"-6.85511,0.812189,0.787025,1,1,0,7.16517,0.310589\n",
    ]
    arr_out_polars = stancsv.csv_bytes_list_to_numpy(
        lines, includes_header=False
    )
    with without_import("polars", cmdstanpy.utils.stancsv):
        arr_out_numpy = stancsv.csv_bytes_list_to_numpy(
            lines, includes_header=False
        )
    assert np.array_equal(arr_out_polars, arr_out_numpy)


def test_csv_polars_and_numpy_equiv_one_line():
    lines = [
        b"-6.76206,1,0.787025,1,1,0,6.81411,0.229458\n",
    ]
    arr_out_polars = stancsv.csv_bytes_list_to_numpy(
        lines, includes_header=False
    )
    with without_import("polars", cmdstanpy.utils.stancsv):
        arr_out_numpy = stancsv.csv_bytes_list_to_numpy(
            lines, includes_header=False
        )
    assert np.array_equal(arr_out_polars, arr_out_numpy)


def test_csv_polars_and_numpy_equiv_one_element():
    lines = [
        b"-6.76206\n",
    ]
    arr_out_polars = stancsv.csv_bytes_list_to_numpy(
        lines, includes_header=False
    )
    with without_import("polars", cmdstanpy.utils.stancsv):
        arr_out_numpy = stancsv.csv_bytes_list_to_numpy(
            lines, includes_header=False
        )
    assert np.array_equal(arr_out_polars, arr_out_numpy)


def test_parse_stan_csv_from_file():
    csv_path = os.path.join(DATAFILES_PATH, "bernoulli_output_1.csv")

    comment_lines, draws_lines = stancsv.parse_stan_csv_comments_and_draws(
        csv_path
    )
    assert all(ln.startswith(b"#") for ln in comment_lines)
    assert all(not ln.startswith(b"#") for ln in draws_lines)

    (
        comment_lines_path,
        draws_lines_path,
    ) = stancsv.parse_stan_csv_comments_and_draws(Path(csv_path))
    assert all(ln.startswith(b"#") for ln in comment_lines)
    assert all(not ln.startswith(b"#") for ln in draws_lines)

    assert comment_lines == comment_lines_path
    assert draws_lines == draws_lines_path
