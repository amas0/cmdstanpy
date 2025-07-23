"""testing stancsv parsing"""

from typing import List

import numpy as np
import polars as pl
import pytest

from cmdstanpy.utils import stancsv


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
        dtype=np.float32,
    )
    arr_out = stancsv.csv_bytes_list_to_numpy(lines, includes_header=False)
    assert np.array_equiv(arr_out, expected)
    assert arr_out[0].dtype == np.float32


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
        dtype=np.float32,
    )
    arr_out = stancsv.csv_bytes_list_to_numpy(lines, includes_header=True)
    assert np.array_equiv(arr_out, expected)


def test_csv_bytes_to_numpy_empty():
    lines = [b""]
    with pytest.raises(pl.exceptions.NoDataError):
        stancsv.csv_bytes_list_to_numpy(lines)


def test_csv_bytes_to_numpy_header_no_draws():
    lines = [
        (
            b"lp__,accept_stat__,stepsize__,treedepth__,"
            b"n_leapfrog__,divergent__,energy__,theta\n"
        ),
    ]
    arr_out = stancsv.csv_bytes_list_to_numpy(lines)
    assert arr_out.shape == (0, 8)


def test_parsing_with_rules():
    lines: List[bytes] = [b"# 1\n", b"2\n", b"3\n", b"# 4\n"]
    comment_lines = []
    non_comment_lines = []
    rules = (
        stancsv.ParsingRule(action=comment_lines.append),
        stancsv.ParsingRule(action=non_comment_lines.append),
        stancsv.ParsingRule(action=comment_lines.append),
    )
    stancsv.parse_general_stan_csv_from_lines(iter(lines), rules)
    assert comment_lines == [b"# 1\n", b"# 4\n"]
    assert non_comment_lines == [b"2\n", b"3\n"]


def test_parsing_with_rules_not_start_in_comment():
    lines: List[bytes] = [b"1\n", b"2\n", b"3\n", b"# 4\n"]
    comment_lines = []
    non_comment_lines = []
    rules = (
        stancsv.ParsingRule(action=non_comment_lines.append),
        stancsv.ParsingRule(action=comment_lines.append),
    )
    stancsv.parse_general_stan_csv_from_lines(
        iter(lines), rules, start_in_comment=False
    )
    assert comment_lines == [b"# 4\n"]
    assert non_comment_lines == [b"1\n", b"2\n", b"3\n"]


def test_parsing_with_rules_entry_action():
    lines: List[bytes] = [b"# 1\n", b"2\n", b"# 4\n"]
    parsed, entry = [], []
    rules = (
        stancsv.ParsingRule(action=parsed.append),
        stancsv.ParsingRule(action=parsed.append, entry_action=entry.append),
        stancsv.ParsingRule(action=parsed.append),
    )
    stancsv.parse_general_stan_csv_from_lines(iter(lines), rules)
    assert parsed == [b"# 1\n", b"# 4\n"]
    assert entry == [b"2\n"]


def test_parsing_insufficient_rules():
    lines: List[bytes] = [b"# 1\n", b"2\n", b"# 4\n"]
    parsed = []
    rules = (
        stancsv.ParsingRule(action=parsed.append),
        stancsv.ParsingRule(action=parsed.append),
    )
    with pytest.raises(IndexError):
        stancsv.parse_general_stan_csv_from_lines(iter(lines), rules)


def test_parsing_timing_lines():
    lines = [
        b"# \n",
        b"#  Elapsed Time: 0.001332 seconds (Warm-up)\n",
        b"#                0.000249 seconds (Sampling)\n",
        b"#                0.001581 seconds (Total)\n",
        b"# \n",
    ]
    out = stancsv.parse_timing_lines(lines)
    assert len(out) == 3
    assert out['Warm-up'] == 0.001332
    assert out['Sampling'] == 0.000249
    assert out['Total'] == 0.001581


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
        b"# Adaptation terminated\n",
        b"# Step size = 0.787025\n",
        b"# Diagonal elements of inverse mass matrix:\n",
        b"# 1,2,3\n",
    ]
    step_size, mass_matrix = stancsv.parse_hmc_adaptation_lines(lines)
    assert step_size == 0.787025
    assert mass_matrix is not None
    assert np.array_equiv(mass_matrix, np.array([1, 2, 3]))


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
        dtype=np.float32,
    )
    assert step_size == 0.775147
    assert mass_matrix is not None
    assert np.array_equiv(mass_matrix, expected)


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
