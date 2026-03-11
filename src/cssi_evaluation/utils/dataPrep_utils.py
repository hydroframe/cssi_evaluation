"""
Data preparation utilities.

General-purpose utilities for preparing datasets for evaluation,
including time alignment, filtering, and dataframe restructuring.
"""

### LOCATION OF ORIGINAL FUNCTIONS
# snow_utils.compute_water_year()
# utils.get_water_year()  duplicate 
# utils.remove_sparse_columns()
# nwm_utils.combine()
# utils.convert_dates_to_timesteps()

import os
import datetime
import pandas as pd
import numpy as np
from typing import Any, Union

# From CUAHSI's utils/snow_utils.py
def compute_water_year(
    df: pd.DataFrame, inplace: bool = False
) -> Union[pd.Series, None]:
    """
    Computes the water year for a given time-index.

    Parameters
    ==========
    df: pandas.DataFrame
        A pandas dataframe containing a datetime[64] index.
    inplace: bool -> False
        A flag to indicate if the water year computation should be returned
        as a column in the input dataframe.

    Returns
    =======
    Union[pandas.Series, pandas.DataFrame]
        If inplace is False, a pandas series containing water year is returned.
        If inplace is True, water year is added to the original dataframe and None is returned
    """

    water_year = df.index.map(lambda x: x.year + 1 if x.month > 9 else x.year)

    if inplace:
        df["Water_Year"] = water_year
        return None

    return water_year.to_series()

# From CUAHSI's nwm_utils.py -- MIGHT NOT NEED THIS, BUT KEEPING IT FOR USE IN CURRENT NOTEBOOKS
def combine(obs_files, mod_files, StartDate, EndDate):

    # Create a dictionary to store dataframes
    dataframes = {}
    
    # Read SNOTEL files
    for file in obs_files:
        location = os.path.basename(file).split('_')[1]  # Extract location from filename
        network = os.path.basename(file).split('_')[-1].split('.')[0] # Extract network from filename
        df = pd.read_csv(file)
        df['Date'] = pd.to_datetime(df['Date']).dt.date  # .dt.date is required because times are not excatly the same between SNOTEL and NWM
        dataframes[f'{network}_{location}'] = df.set_index('Date')
    
    # Read NWM files
    for file in mod_files:
        location = os.path.basename(file).split('_')[1]  # Extract location from filename
        df = pd.read_csv(file)
        df['Date_Local'] = pd.to_datetime(df['Date_Local']).dt.date  # .dt.date is required because times are not excatly the same between SNOTEL and NWM
        dataframes[f'NWM_{location}'] = df.set_index('Date_Local')
    
    # Merge dataframes on Date
    combined_df = pd.DataFrame(index=pd.date_range(start=StartDate, end=EndDate))  
    for key, df in dataframes.items():
        if 'SNTL' in key:
            combined_df[f'{key}_swe_m'] = df['Snow Water Equivalent (m) Start of Day Values']
        if 'CCSS' in key:
            combined_df[f'{key}_swe_m'] = df['Snow Water Equivalent (m) Start of Day Values']
        elif 'NWM' in key:
            combined_df[f'{key}_swe_m'] = df['NWM_SWE_meters']

    return combined_df



### BELOW ARE UTILS FROM AMY'S FILES 

"""
Model evaluation utility functions.

Note that these functions are not intended to be used stand-alone; they act as sub-processes
within the model_evaluation.evaluate method.
"""

import datetime
import numpy as np
from hf_hydrodata import get_gridded_data

HYDRODATA = "/hydrodata"

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