"""
Model evaluation utility functions.

Note that these functions are not intended to be used stand-alone; they act as sub-processes
within the model_evaluation.evaluate method.
"""

import numpy as np

HYDRODATA = "/hydrodata"


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
