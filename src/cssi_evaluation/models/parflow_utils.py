"""
ParFlow model utilities.

Functions for preprocessing PF outputs, handling coordinate/grid conversions,
and preparing datasets for comparison with observations.
"""
"""
Main method for model evaluation.
These functions are from Amy's `model_evaluation.py` script, but have been 
moved here since they are more specific to the ParFlow model outputs and workflow. 
They could be adapted for other models/datasets, but the current implementation is specific to 
PF-CONUS outputs. 
In particular the `explore_available_observations` and `get_observations` functions could be adapted 
for other models/datasets, and moved to the more general `evaluation_utils.py` since they are focused on 
handling observations data. 
"""

# IMPORTS NEEDED FOR FUNCTION DEFINITIONS***
# From model_evaluation.py, but some may not be needed and need to clean this up.
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

def get_parflow_output(
    obs_metadata_df,
    parflow_output_dir,
    parflow_runname,
    date_start,
    date_end,
    variable,
    temporal_resolution,
    initial_timestep=None,
    write_csv=False,
    csv_path=None,
):
    """
    Subset ParFlow outputs to only observation locations.

    Parameters
    ----------
    obs_metadata_df : DataFrame
        DataFrame with one row per filtered site, containing location and site attribute
        information. This is output from the function `get_observations`.
    parflow_output_dir : str
        String representing the directory path to where ParFlow outputs are stored.
    parflow_runname : str
        Name used to define the ParFlow run. Note that in standard ParFlow file naming
        conventions, this is used at the beginning of certain output file names.
    date_start : datetime
        The starting date (daily) or date+hour (hourly) for the ParFlow simulations.
    date_end : datetime
        The ending date (daily) or date+hour (hourly) for the ParFlow simulations.
    variable : str
        Variable requested
    temporal_resolution : str
        "hourly" or "daily"
    initial_timestep : datetime; default=None
        The starting date (daily) or date+hour (hourly) for the ParFlow simulations.
        If None, defaults to the first of the water year containing start_date.
    write_csv : bool; default=False
        Indicator for whether to additionally write out calculated metrics to disk as .csv.
    csv_path : str; default=None
        Path to where to save .csv of ParFlow outputs to disk if `write_csv=True`.

    Returns
    -------
    DataFrame
        DataFrame containing the time series observartions for each matched-site-grid-cell
        for the requested time range. The columns represent each site location requested
        and the rows contain the time series from the ParFlow grid cell that contains
        that site.
    """

    water_year = utils.get_water_year(date_start)

    if temporal_resolution == "hourly":
        timesteps = np.arange(
            date_start,
            date_end + datetime.timedelta(hours=24),
            datetime.timedelta(hours=1),
        ).astype("datetime64[h]")
    elif temporal_resolution == "daily":
        timesteps = np.arange(
            date_start,
            date_end + datetime.timedelta(days=1),
            datetime.timedelta(days=1),
        ).astype("datetime64[D]")
    else:
        raise ValueError("temporal_resolution must be either 'hourly' or 'daily'.")
    nt = len(timesteps)

    ts_start, ts_end = utils.convert_dates_to_timesteps(
        date_start, date_end, temporal_resolution, initial_timestep
    )

    if temporal_resolution == "hourly":
        run = Run.from_definition(f"{parflow_output_dir}/{parflow_runname}.pfidb")
        data = run.data_accessor

        dx = data.dx
        dy = data.dy
        dz = data.dz

        mask = data.mask
        mannings = (read_pfb(f"{parflow_output_dir}/mannings.pfb")).squeeze()
        slopex = (data.slope_x).squeeze()
        slopey = (data.slope_y).squeeze()

    # Initialize array for final output: one column per mapped site, one row per timestep
    num_sites = len(obs_metadata_df)
    pf_matched_obs = np.zeros((nt, num_sites))

    # Iterate through all hours, starting at 1 and ending at the last hour in the date range
    # (t-1) is used below to set the dataframe from index 0 (hour starts at index 1)
    # Note: pf_variable below will be a NumPy array of shape (ny, nx) for a single timestep
    ts_idx = 0
    for t in range(ts_start, (ts_end + 1)):
        if variable == "streamflow":
            if temporal_resolution == "hourly":
                pressure = pf.read_pfb(
                    f"{parflow_output_dir}/{parflow_runname}.out.press.{str(t).zfill(5)}.pfb"
                )

                # convert streamflow from m^3/h to m^3/s
                pf_variable = (
                    hydro.calculate_overland_flow_grid(
                        pressure, slopex, slopey, mannings, dx, dy, mask=mask
                    )
                    / 3600
                )
            else:
                pf_variable = pf.read_pfb(
                    f"{parflow_output_dir}/flow.{water_year}.daily.{str(t).zfill(3)}.pfb"
                ).squeeze()
                pf_variable = pf_variable / 3600  # convert from m^3/h to cms

        elif variable == "water_table_depth":
            if temporal_resolution == "hourly":
                pressure = pf.read_pfb(
                    f"{parflow_output_dir}/{parflow_runname}.out.press.{str(t).zfill(5)}.pfb"
                )
                saturation = pf.read_pfb(
                    f"{parflow_output_dir}/{parflow_runname}.out.satur.{str(t).zfill(5)}.pfb"
                )

                pf_variable = hydro.calculate_water_table_depth(
                    pressure, saturation, dz
                )
            else:
                pf_variable = pf.read_pfb(
                    f"{parflow_output_dir}/WTD.{water_year}.daily.{str(t).zfill(3)}.pfb"
                ).squeeze()

        elif variable == "swe":
            if temporal_resolution == "hourly":
                clm = pf.read_pfb(
                    f"{parflow_output_dir}/{parflow_runname}.out.clm_output.{str(t).zfill(5)}.C.pfb"
                )
                pf_variable = clm[
                    10, :, :
                ]  # SWE is the 11th layer in CLM files (Python index 10)
            else:
                pf_variable = pf.read_pfb(
                    f"{parflow_output_dir}/swe_out.{water_year}.daily.{str(t).zfill(3)}.pfb"
                ).squeeze()

        elif variable == "latent_heat":
            if temporal_resolution == "hourly":
                clm = pf.read_pfb(
                    f"{parflow_output_dir}/{parflow_runname}.out.clm_output.{str(t).zfill(5)}.C.pfb"
                )
                pf_variable = clm[
                    0, :, :
                ]  # latent heat is the 1st layer in CLM files (Python index 0)
            else:
                pf_variable = pf.read_pfb(
                    f"{parflow_output_dir}/eflx_lh_tot.{water_year}.daily.{str(t).zfill(3)}.pfb"
                ).squeeze()

        else:
            raise ValueError(
                "variable must be one of: 'streamflow', 'water_table_depth', 'swe', or 'latent_heat'"
            )

        # Select out only locations with observations
        for obs_idx in range(num_sites):
            pf_matched_obs[ts_idx, obs_idx] = pf_variable[
                obs_metadata_df.iloc[obs_idx].loc["domain_j"],
                obs_metadata_df.iloc[obs_idx].loc["domain_i"],
            ]
        ts_idx += 1

    # Format final output array into DataFrame
    pf_matched_obs_df = pd.DataFrame(pf_matched_obs)
    pf_matched_obs_df.columns = list(obs_metadata_df["site_id"])
    pf_matched_obs_df = pf_matched_obs_df.set_index(timesteps).reset_index(names="date")

    # Additionally write to disk if requested
    if write_csv is True:
        pf_matched_obs_df.to_csv(csv_path, index=False)

    return pf_matched_obs_df