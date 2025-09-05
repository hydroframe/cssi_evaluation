"""
Model evaluation utility functions.

Note that these functions are not intended to be used stand-alone; they act as sub-processes
within the model_evaluation.evaluate method.
"""

import datetime
import numpy as np
from hf_hydrodata import get_gridded_data

HYDRODATA = "/hydrodata"


def get_conus_mask(grid):
    """
    Get the CONUS mask for a given grid.
    """
    options = {"dataset": f"{grid}_domain", "variable": "mask"}
    conus_mask = get_gridded_data(options).squeeze()

    return conus_mask.astype(int)


def check_mask_shape(mask, ij_bounds):
    """
    Function to check size of mask matches with ij bounds provided.

    Parameters
    ----------
    mask : array
        Array representing a domain mask.
    ij_bounds : tuple
        Tuple of (i_min, j_min, i_max, j_max) of where the mask is located within the
        conus domain.

    Returns
    -------
    None
        Raises ValueError if size of mask doesn't match size of ij bounds.
    """
    j_bound_length = ij_bounds[3] - ij_bounds[1]
    i_bound_length = ij_bounds[2] - ij_bounds[0]

    try:
        assert i_bound_length == mask.shape[1]
        assert j_bound_length == mask.shape[0]
    except Exception as exc:
        raise ValueError(
            f"The mask shape is {mask.shape} but the ij_bounds is shape {j_bound_length, i_bound_length}"
        ) from exc


def get_domain_indices(ij_bounds, conus_indices):
    """
    Get the domain indices for a subset grid from a larger grid. Typically this larger
    grid will be either the CONUS1 or CONUS2 grids.

    Parameters
    ----------
    ij_bounds : tuple
        (imin, jmin, imax, jmax) for the subset domain relative to the larger domain.
        Typically this larger domain is CONUS1 or CONUS2.
    conus_indices : tuple
        (i, j) indices cooresponding to the grid cell in the larger domain. Typically
        this larger domain is CONUS1 or CONUS2.

    Returns
    -------
    domain_indices : tuple
        (i, j) indices cooresponding to the grid cell in the subset domain.
    """
    mapped_j = int(conus_indices[1]) - ij_bounds[1]  # subtract jmin
    mapped_i = int(conus_indices[0]) - ij_bounds[0]  # subtract imin

    return (mapped_i, mapped_j)


def remove_sparse_columns(df, min_obs_pct=None, min_obs_count=None):
    """
    Removes columns from a DataFrame that have fewer non-missing values than the specified threshold.

    Parameters:
    - df: pd.DataFrame
    - min_obs_pct: float, optional — e.g. 0.95 means keep columns with at least 95% non-missing values
    - min_obs_count: int, optional — e.g. 100 means keep columns with at least 100 non-missing values

    Returns:
    - pd.DataFrame: filtered DataFrame with columns removed

    Raises:
    - ValueError: if neither or both thresholds are specified
    """
    if (min_obs_pct is None) == (min_obs_count is None):
        raise ValueError("You must specify exactly one of min_obs_pct or min_obs_count")

    if min_obs_pct is not None:
        threshold = df.shape[0] * min_obs_pct
    else:
        threshold = min_obs_count

    # Keep only columns where non-missing count >= threshold
    valid_cols = df.columns[df.notna().sum() >= threshold]
    return df[valid_cols]


def convert_dates_to_timesteps(
    start_date, end_date, temporal_resolution, initial_timestep=None
):
    """
    Convert start and end dates to timesteps relative to a water year.

    Parameters
    ----------
    start_date : datetime
        The starting date (daily) or date+hour (hourly) for the ParFlow simulations.
    end_date : datetime
        The ending date (daily) or date+hour (hourly) for the ParFlow simulations.
    temporal_resolution : str
        "hourly" or "daily"
    initial_timestep : datetime; default=None
        The starting date (daily) or date+hour (hourly) for the ParFlow simulations.
        If None, defaults to the first of the water year containing start_date.

    Returns
    -------
    tuple
        (ts_start, ts_end) representing the starting and ending timesteps relative to the
        water year and temporal resolution. The ending timestep is inclusive.
    """
    water_year = get_water_year(start_date)

    # Default to counting timestep 1 as the first of the water year
    if initial_timestep is None:
        initial_timestep = datetime.datetime(water_year - 1, 10, 1)

    start_delta = start_date - initial_timestep
    end_delta = end_date - initial_timestep

    # Add 1 after to start daily indexing at 1 instead of 0
    if temporal_resolution == "daily":
        ts_start = start_delta.days + 1
        ts_end = end_delta.days + 1

    elif temporal_resolution == "hourly":
        ts_start = start_delta.days * 24 + 1
        ts_end = (end_delta.days + 1) * 24

    else:
        raise ValueError(
            f"Value of {temporal_resolution} for temporal_resolution is not recognized."
        )

    return (ts_start, ts_end)


def get_water_year(date):
    """Return the water year for a given date."""
    if date.month in range(1, 10):
        return date.year
    else:
        return date.year + 1


def initialize_metrics_df(obs_metadata_df, metrics_list):
    """
    Initialize DataFrame table to store metrics output.

    Parameters
    ----------
    obs_metadata_df: DataFrame
        Pandas DataFrame consisting of at least site_id, x, and y CONUS grid mapping
        values.
    metrics: list
        List of string names of metrics to use for evaluation. Must be present in METRICS_DICT
        dictionary in the model_evaluation.py module.

    Returns
    -------
    DataFrame
        DataFrame containing site ID, x and y CONUS grid mapping values, along with empty columns
        for each of the evaluation metrics defined in metrics.
    """
    metrics_df = obs_metadata_df[
        ["site_id", "site_name", "latitude", "longitude", "domain_i", "domain_j"]
    ].copy()
    for m in metrics_list:
        metrics_df[f"{m}"] = np.nan

    return metrics_df
