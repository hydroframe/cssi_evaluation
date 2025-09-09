"""Test /model_evaluation/evaluation_metrics.py module."""

# pylint: disable=E0401
import sys
import os
import math
import pytest
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from cssi_evaluation import evaluation_metrics


def test_rmse():
    """Test RMSE calculation."""
    # Example from Sci-kit Learn API documentation
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.root_mean_squared_error.html
    sim = np.array([3, -0.5, 2, 7])
    obs = np.array([2.5, 0.0, 2, 8])

    calc = evaluation_metrics.RMSE(obs, sim)
    assert math.isclose(calc, 0.612372, rel_tol=0.000001)


def test_mse():
    """Test MSE calculation."""
    # Example from Sci-kit Learn API documentation
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
    sim = np.array([3, -0.5, 2, 7])
    obs = np.array([2.5, 0.0, 2, 8])

    calc = evaluation_metrics.MSE(obs, sim)
    assert math.isclose(calc, 0.375, rel_tol=0.001)


def test_pearson_r():
    """Test Pearson R calculation."""
    # Example from SciPy API documentation
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
    sim = np.array([1, 2, 3, 4, 5])
    obs = np.array([10, 9, 2.5, 6, 4])

    calc = evaluation_metrics.pearson_R(obs, sim)
    assert math.isclose(calc, -0.742610, rel_tol=0.000001)


def test_spearman_rank():
    """Test spearman rank calculation."""
    obs = np.array([8, 4, 3, 9, 9])
    sim = np.array([1, 4, 3, 3, 9])

    calc = evaluation_metrics.spearman_rank(obs, sim)
    assert math.isclose(calc, 0.2368421, rel_tol=0.0000001)


def test_nse():
    """Test Nash-Sutcliffe Efficiency (NSE) calculation."""
    sim = np.arange(1, 11, 1)  # array from 1-10
    obs = np.arange(2, 12, 1)  # array from 2-11

    calc = evaluation_metrics.NSE(obs, sim)
    assert math.isclose(
        calc, 0.8787878, rel_tol=0.0000001
    )  # value based on using R HydroGOF package


def test_kge():
    """Test Kling-Gupta Efficiency (KGE) calculation."""
    sim = np.arange(1, 11, 1)  # array from 1-10
    obs = np.arange(2, 12, 1)  # array from 2-11

    calc = evaluation_metrics.KGE(obs, sim)
    assert math.isclose(
        calc, 0.8461538, rel_tol=0.0000001
    )  # value based on using R HydroGOF package


def test_r_squared():
    """Test R-squared calculation."""
    # Example from scikit-learn API documentation
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html.
    sim = np.array([2.5, 0.0, 2, 8])
    obs = np.array([3, -0.5, 2, 7])

    calc = evaluation_metrics.R_squared(obs, sim)
    assert math.isclose(calc, 0.948608, rel_tol=0.000001)


def test_bias_from_r():
    """Test bias from R calculation."""
    sim = np.arange(1, 11, 1)  # array from 1-10
    obs = np.arange(2, 12, 1)  # array from 2-11

    calc = evaluation_metrics.bias_from_R(obs, sim)
    assert math.isclose(
        calc, 0.9428571, rel_tol=0.0000001
    )  # value based on using R HydroGOF package


def test_bias():
    """Test bias calculation."""
    sim = np.arange(1, 11, 1)  # array from 1-10
    obs = np.arange(2, 12, 1)  # array from 2-11

    calc = evaluation_metrics.bias(obs, sim)
    assert math.isclose(
        calc, -0.15384615, rel_tol=0.000001
    )  # value aligned with using R HydroGOF package


def test_percent_bias():
    """Test percent bias calculation."""
    sim = np.arange(1, 11, 1)  # array from 1-10
    obs = np.arange(2, 12, 1)  # array from 2-11

    calc = evaluation_metrics.percent_bias(obs, sim)
    assert math.isclose(
        calc, -15.384615, rel_tol=0.000001
    )  # value aligned with using R HydroGOF package


def test_absolute_relative_bias():
    """Test absolute relative bias calculation."""
    sim = np.arange(1, 11, 1)  # array from 1-10
    obs = np.arange(2, 12, 1)  # array from 2-11

    calc = evaluation_metrics.absolute_relative_bias(obs, sim)
    assert math.isclose(
        calc, 0.15384615, rel_tol=0.000001
    )  # value aligned with using R HydroGOF package


def test_total_difference():
    """Test total difference calculaiton."""
    sim = np.array([3, -0.5, 2, 7])
    obs = np.array([2.5, 0.0, 2, 8])

    calc = evaluation_metrics.total_difference(obs, sim)
    assert math.isclose(calc, -1.0, rel_tol=0.0001)


if __name__ == "__main__":
    pytest.main()
