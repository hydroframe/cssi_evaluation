"""
ParFlow model utilities.

Functions for preprocessing PF outputs, handling coordinate/grid conversions,
and preparing datasets for comparison with observations.
"""

# IMPORTS NEEDED FOR FUNCTION DEFINITIONS***
import datetime
import warnings
import pandas as pd
import hf_hydrodata as hf
import numpy as np
import parflow as pf
from parflow import Run
from parflow.tools.io import read_pfb
import parflow.tools.hydrology as hydro

from cssi_evaluation.utils.dataPrep_utils import (
    get_water_year,
    convert_dates_to_timesteps,
)

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

    water_year = get_water_year(date_start)

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

    ts_start, ts_end = convert_dates_to_timesteps(
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


def get_conus_mask(grid):
    """
    Get the CONUS-wide mask for a given grid ("conus1" or "conus2").
    """
    options = {"dataset": f"{grid}_domain", "variable": "mask"}
    conus_mask = hf.get_gridded_data(options).squeeze()

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
