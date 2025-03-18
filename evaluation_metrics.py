"""
Evaluation Metrics 

- Root Mean Square Error (RMSE)
- Mean Square Error (MSE)
- Pearson Correlation Coefficient (R)
- Spearman's Rank Correlation Coefficient
- Nash-Sutcliffe Efficiency (NSE)
- Kling-Gupta Efficiency (KGE)
- R-squared
- Bias from R
- Bias
- Percent Bias
- Absolute Relative Bias
- Total Difference
- Condon cateogory (low/high bias, poor/good shape)
"""

# pylint: disable=C0103

from scipy import stats
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score, root_mean_squared_error
import numpy as np


def RMSE(x, y):
    """
     Calculate RMSE.

    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.root_mean_squared_error.html.

     Parameters
     ----------
     x: array-like of shape (n_samples,) or (n_samples, n_outputs)
         Ground Truth (correct) target values.
     y: array-like of shape (n_samples,) or (n_samples, n_outputs)
         Estimated values.

     Returns
     -------
     float or ndarray of floats
    """
    result = root_mean_squared_error(x, y)
    return result


def MSE(x, y):
    """
    Calculate MSE.

    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html.

    Parameters
    ----------
    x: array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground Truth (correct) target values.
    y: array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated values.

    Returns
    -------
    float or ndarray of floats
    """
    result = mse(x, y)
    return result


def pearson_R(x, y):
    """
    Calculate Pearson's R.

    Refer to https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html.

    Parameters
    ----------
    x: array-like of shape (n_samples,)
        Ground Truth (correct) target values.
    y: array-like of shape (n_samples,)
        Estimated values.

    Returns
    -------
    float or ndarray of floats
    """
    result = stats.pearsonr(x, y)[0]
    return result


def spearman_rank(x, y):
    """
    Calculate Spearman's Rank Correlation Coefficient.

    Refer to https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html.

    Parameters
    ----------
    x: array-like of shape (n_samples,)
        Ground Truth (correct) target values.
    y: array-like of shape (n_samples,)
        Estimated values.

    Returns
    -------
    float or ndarray of floats
    """
    result = stats.spearmanr(x, y)[0]
    return result


def NSE(x, y):
    """
    Calculate Nash-Sutcliffe Efficiency (NSE).

    Refer to https://doi.org/10.1016/0022-1694(70)90255-6.

    Parameters
    ----------
    x: array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground Truth (correct) target values.
    y: array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated values.

    Returns
    -------
    float or ndarray of floats
    """
    return 1 - (np.sum((x - y) ** 2) / np.sum((x - np.mean(x)) ** 2))


def KGE(x, y):
    """
    Calculate Kling-Gupta Efficiency (KGE).

    Refer to Equation 9 in https://doi.org/10.1016/j.jhydrol.2009.08.003.

    Parameters
    ----------
    x: array-like of shape (n_samples,)
        Ground Truth (correct) target values.
    y: array-like of shape (n_samples,)
        Estimated values.

    Returns
    -------
    float or ndarray of floats
    """
    r = pearson_R(x, y)
    alpha = np.std(y) / np.std(x)
    beta = np.average(y) / np.average(x)
    result = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

    return result


def R_squared(x, y):
    """
    Calculate R**2.

    Refer to https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html.

    Parameters
    ----------
    x: array-like of shape (n_samples,)
        Ground Truth (correct) target values.
    y: array-like of shape (n_samples,)
        Estimated values.

    Returns
    -------
    float or ndarray of floats
    """
    return r2_score(x, y)


def bias_from_R(x, y):
    """
    Calculate the bias from R.

    The bias from R indicates systematic additive and multiplicative biases in the
    generated values, with a value between 0 and 1, where bias = 1 means no bias.
    Refer to Equation 16 on the paper (https://www.nature.com/articles/srep19401).

    Parameters
    ----------
    x: array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground Truth (correct) target values.
    y: array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated values.

    Returns
    -------
    float or ndarray of floats
    """
    std_x = np.std(x)
    mean_x = np.mean(x)

    std_y = np.std(y)
    mean_y = np.mean(y)

    result = 2 / (
        std_x / std_y + std_y / std_x + (mean_x - mean_y) ** 2 / (std_x * std_y)
    )
    return result


def bias(x, y):
    """
    Calculate bias.

    Parameters
    ----------
    x: array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground Truth (correct) target values.
    y: array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated values.

    Returns
    -------
    float or ndarray of floats
    """
    return np.sum(y - x) / np.sum(x)


def percent_bias(x, y):
    """
    Calculate percent bias.

    Parameters
    ----------
    x: array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground Truth (correct) target values.
    y: array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated values.

    Returns
    -------
    float or ndarray of floats
    """
    return 100 * (np.sum(y - x) / np.sum(x))


def absolute_relative_bias(x, y):
    """
    Calculate absolute relative bias.

    Parameters
    ----------
    x: array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground Truth (correct) target values.
    y: array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated values.

    Returns
    -------
    float or ndarray of floats
    """
    return abs((np.sum(y - x) / np.sum(x)))


def total_difference(x, y):
    """
    Calculate total difference (model minus observed).

    Parameters
    ----------
    x: array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground Truth (correct) target values.
    y: array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated values.

    Returns
    -------
    float or ndarray of floats
    """
    return np.sum(y) - np.sum(x)


def condon(abs_rel_bias, srho):
    """
    Calculate the Condon category (low/high bias, poor/good shape).

    Parameters
    ----------
    abs_rel_bias : Absolute relative bias
    srho: Spearman's rho

    Returns
    -------
    String of categorical mapping
    """
    if abs_rel_bias < 1:
        if srho >= 0.5:
            return "Low bias, good shape"
        elif srho < 0.5:
            return "Low bias, poor shape"
        else:
            return "Undefined"
    elif abs_rel_bias >= 1:
        if srho >= 0.5:
            return "High bias, good shape"
        elif srho < 0.5:
            return "High bias, poor shape"
        else:
            return "Undefined"
    else:
        return "Undefined"
