# These are functions that need consideration for where they should be reorganized within the codebase. 

#### From Amy's `model_evaluation.py` script, but have been moved here since they are more specific to the ParFlow model outputs and workflow.
# They *should* belong in the evaluation_utils.py file since they are focused on handling observations data, 
# but they are currently implemented in a way that is specific to the PF-CONUS outputs and workflow. 
# They could be adapted for other models/datasets, but the current implementation is specific to PF-CONUS outputs.

### LOCATION OF ORIGINAL FUNCTION DEFINITIONS
# model_evaluation.explore_available_observations()
# model_evaluation.get_observations()

import datetime
import warnings
import pandas as pd
import hf_hydrodata as hf
import numpy as np
import parflow as pf
from parflow import Run
from parflow.tools.io import read_pfb
import parflow.tools.hydrology as hydro

import cssi_evaluation.utils as utils
import cssi_evaluation.evaluation_metrics as evaluation_metrics

METRICS_DICT = {
    "r2": evaluation_metrics.R_squared,
    "spearman_rho": evaluation_metrics.spearman_rank,
    "mse": evaluation_metrics.MSE,
    "rmse": evaluation_metrics.RMSE,
    "bias": evaluation_metrics.bias,
    "percent_bias": evaluation_metrics.percent_bias,
    "abs_rel_bias": evaluation_metrics.absolute_relative_bias,
    "total_difference": evaluation_metrics.total_difference,
    "pearson_r": evaluation_metrics.pearson_R,
    "nse": evaluation_metrics.NSE,
    "kge": evaluation_metrics.KGE,
    "bias_from_r": evaluation_metrics.bias_from_R,
    "condon": evaluation_metrics.condon,
}

DATE_SUFFIX = datetime.date.today().strftime("%Y%m%d")

warnings.simplefilter(action="ignore", category=FutureWarning)

# *This function is specific to PF-CONUS, but could be adapted for other models/datasets
def explore_available_observations(mask, ij_bounds, grid, **kwargs):
    """
    Given a list of HUC(s) and a grid, return information on data availability.

    This function accepts additional filters on temporal_resolution, dataset, variable,
    date_start, and date_end. The returned DataFrame includes metadata about each site and
    each variable that site has data available for (there are cases where a single site might
    have multiple types of data available).

    Parameters
    ----------
    mask : array
        Array representing a domain mask.
    ij_bounds : tuple
        Tuple of (i_min, j_min, i_max, j_max) of where the mask is located within the
        conus domain.
    grid : str
        "conus1" or "conus2"
    kwargs
        Additional keyword arguments to pass in to `hf_hydrodata.get_site_variables`. These include
        `temporal_resolution`, `dataset`, `variable`, `date_start`, and `date_end`.

    Returns
    -------
    metadata_df : DataFrame
        DataFrame with one row per filtered site, containing location and site attribute
        information.
    """
    options = kwargs

    utils.check_mask_shape(mask, ij_bounds)

    # Query site metadata for the CONUS grid bounds
    options["grid"] = grid
    options["grid_bounds"] = [
        ij_bounds[0],
        ij_bounds[1],
        ij_bounds[2] - 1,
        ij_bounds[3] - 1,
    ]
    data_available_df = hf.get_site_variables(options)

    # Shift i/j coordinates so that they index starting from the regional
    # bounding box origin instead of the overall CONUS grid origin
    data_available_df["domain_i"] = data_available_df.apply(
        lambda x: utils.get_domain_indices(ij_bounds, (x[f"{grid}_i"], x[f"{grid}_j"]))[
            0
        ],
        axis=1,
    )
    data_available_df["domain_j"] = data_available_df.apply(
        lambda x: utils.get_domain_indices(ij_bounds, (x[f"{grid}_i"], x[f"{grid}_j"]))[
            1
        ],
        axis=1,
    )

    # Filter sites to only those within HUC mask
    data_available_df["mask"] = mask[
        data_available_df["domain_j"], data_available_df["domain_i"]
    ]
    data_available_df = data_available_df[data_available_df["mask"] == 1]

    return data_available_df


def get_observations( # *This function is specific to PF-CONUS, but could be adapted for other models/datasets
    mask,
    ij_bounds,
    grid,
    date_start,
    date_end,
    variable,
    temporal_resolution,
    output_type="wide",
    write_csvs=False,
    csv_paths=None,
    remove_sites_no_data=True,
    missing_pct_threshold=None,
    missing_count_threshold=None,
    **kwargs,
):
    """
    Given a mask, its ij bounds, and a grid, return observations of a given variable from a given
    dataset that are located within the HUC(s).

    This one returns metadata + data. Needs to have a variable passed in. Otherwise, a site
    might be trying to return multiple types of data in a single DataFrame. Not possible if
    one column per site.

    Parameters
    ----------
    mask : array
        Array representing a domain mask.
    ij_bounds : tuple
        Tuple of (i_min, j_min, i_max, j_max) of where the mask is located within the
        conus domain.
    grid : str
        "conus1" or "conus2"
    date_start : datetime
        Starting date of observations data returned.
    date_end : datetime
        Ending date of observations data returned.
    variable : str
        Variable requested
    temporal_resolution : str
        "hourly" or "daily"
    output_type : str; default="wide"
        "wide" or "long" where "wide" represents a DataFrame that is one column per site
        and "long" represents a DataFrame that is one record per site*date combination.
        This impacts the observations DataFrame only.
    write_csvs : bool; default=False
        Indicator for whether to additionally write out calculated metrics to disk as .csv.
    csv_paths : tuple of str; default=None
        Tuple of paths to where to save .csvs of observations metadata and data if `write_csv=True`.
    remove_sites_no_data : bool; default=True
        Indicator for whether to filter data and metadata DataFrames to only include sites with non-NaN
        observation measurements over the requested time range. The default is to exclude these sites.
    missing_pct_threshold : float; default=None
        Float representing the minimum percentage of non-missing values required for a site to be
        included in the returned DataFrames. E.g. 0.95 means a site must have at least 95% of the
        requested timesteps with non-missing values. If None, this filter is not applied.
    missing_count_threshold : int; default=None
        Integer representing the minimum count of non-missing values required for a site to be
        included in the returned DataFrames. E.g. 100 means a site must have at least 100
        non-missing values over the requested time range. If None, this filter is not applied.
    kwargs
        Additional keyword arguments to pass in to `hf_hydrodata.get_point_metadata` and
        `hf_hydrodata.get_point_data`. These include `dataset` and `aggregation`.


    Returns
    -------
    metadata_df : DataFrame
        DataFrame with one row per filtered site, containing location and site attribute
        information.
    obs_data_df : DataFrame
        DataFrame containing the time series observartions for each filtered site for the
        requested time range. One column per site and one row per timestep.
    """
    try:
        assert variable in ["streamflow", "water_table_depth", "swe", "latent_heat"]
    except Exception as exc:
        raise ValueError(
            f"{variable} is not supported. Supported variables include: 'streamflow', 'water_table_depth', 'swe', 'latent_heat'."
        ) from exc

    # Setting an ij_bounds of None means that the user wants to use the full CONUS grid, so we don't need
    # to check the mask shape or adjust the ij_bounds.
    if ij_bounds is not None:
        utils.check_mask_shape(mask, ij_bounds)

        # Update bounds so they use inclusive upper bounds for hf_hydrodata. Otherwise, one index too many will be requested.
        ij_bounds = [
            ij_bounds[0],
            ij_bounds[1],
            ij_bounds[2] - 1,
            ij_bounds[3] - 1,
        ]

    # Define variables if provided (move to separate function)
    if kwargs.get("dataset") is None:
        if variable in ["streamflow", "water_table_depth"]:
            kwargs["dataset"] = "usgs_nwis"
        elif variable == "swe":
            kwargs["dataset"] = "snotel"
        elif variable == "latent_heat":
            kwargs["dataset"] = "ameriflux"
    if kwargs.get("aggregation") is None:
        if variable in ["streamflow", "water_table_depth"]:
            kwargs["aggregation"] = "mean"
        elif variable == "swe":
            kwargs["aggregation"] = "sod"
        elif variable == "latent_heat":
            kwargs["aggregation"] = "sum"

    # Query site metadata for the CONUS grid bounds
    metadata_df = hf.get_point_metadata(
        dataset=kwargs["dataset"],
        variable=variable,
        temporal_resolution=temporal_resolution,
        aggregation=kwargs["aggregation"],
        date_start=date_start,
        date_end=date_end,
        grid=grid,
        grid_bounds=ij_bounds,
    )

    # Shift i/j coordinates so that they index starting from the regional
    # bounding box origin instead of the overall CONUS grid origin
    if ij_bounds is not None:
        metadata_df["domain_i"] = metadata_df.apply(
            lambda x: utils.get_domain_indices(
                ij_bounds, (x[f"{grid}_i"], x[f"{grid}_j"])
            )[0],
            axis=1,
        )
        metadata_df["domain_j"] = metadata_df.apply(
            lambda x: utils.get_domain_indices(
                ij_bounds, (x[f"{grid}_i"], x[f"{grid}_j"])
            )[1],
            axis=1,
        )

        # Filter sites to only those within HUC mask
        metadata_df["mask"] = mask[metadata_df["domain_j"], metadata_df["domain_i"]]
        metadata_df = metadata_df[metadata_df["mask"] == 1]
        metadata_df.drop(columns=("mask"), inplace=True)
    else:
        metadata_df["domain_i"] = metadata_df[f"{grid}_i"].astype(int)
        metadata_df["domain_j"] = metadata_df[f"{grid}_j"].astype(int)

    # Add context variables to metadata DF
    metadata_df["grid"] = grid
    metadata_df["dataset"] = kwargs["dataset"]
    metadata_df["variable"] = variable
    metadata_df["temporal_resolution"] = temporal_resolution
    metadata_df["aggregation"] = kwargs["aggregation"]

    # Create list of filtered sites to pass in to get time series
    site_list = list(metadata_df["site_id"])

    # Query point observations time series for only sites within HUC mask
    obs_data_df = hf.get_point_data(
        dataset=kwargs["dataset"],
        variable=variable,
        temporal_resolution=temporal_resolution,
        aggregation=kwargs["aggregation"],
        date_start=date_start,
        date_end=date_end,
        site_ids=site_list,
    )

    if remove_sites_no_data is True:
        obs_data_df = obs_data_df.dropna(axis=1, how="all")
        nan_sites = [s for s in site_list if s not in list(obs_data_df.columns)]
        metadata_df = metadata_df[~metadata_df.site_id.isin(nan_sites)]

    # Only proceed if observation time series has enough non-NaN values
    if (missing_pct_threshold) or (missing_count_threshold):
        obs_data_df = utils.remove_sparse_columns(
            obs_data_df,
            min_obs_pct=missing_pct_threshold,
            min_obs_count=missing_count_threshold,
        )
        missing_data_sites = [
            s for s in site_list if s not in list(obs_data_df.columns)
        ]
        metadata_df = metadata_df[~metadata_df.site_id.isin(missing_data_sites)]

    if output_type == "long":
        # Reshape observations dataframe and attach i/j locations
        obs_data_df = obs_data_df.melt(
            id_vars=["date"], var_name="site_id", value_name=variable
        )
        sim_loc_info = metadata_df[["site_id", f"{grid}_i", f"{grid}_j"]]
        obs_data_df = pd.merge(obs_data_df, sim_loc_info, on="site_id", how="inner")

    # Additionally write to disk if requested
    if write_csvs is True:
        metadata_df.to_csv(csv_paths[0], index=False)
        obs_data_df.to_csv(csv_paths[1], index=False)

    return metadata_df, obs_data_df