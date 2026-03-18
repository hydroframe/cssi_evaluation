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
# nwm_utils.convert_latlon_to_yx()
# nwm_utils.convert_utc_to_local()

import os
import datetime
import pandas as pd
import numpy as np
from typing import Any, Union
import pyproj
import pytz

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

def get_water_year(date):
    """Return the water year for a given date."""
    if date.month in range(1, 10):
        return date.year
    else:
        return date.year + 1

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

# from Irene's nwm_utils.py

def convert_latlon_to_yx(lat, lon, input_crs, output_crs):
    """
    This function takes latitude and longitude values along with
    input and output coordinate reference system (CRS) and 
    uses Python's pyproj package to convert the provided values 
    (as single float values, not arrays) to the corresponding y and x 
    coordinates in the output CRS.

    Parameters
    ----------
    lat : float
        Latitude value.
    lon : float
        Longitude value.
    input_crs : str
        CRS of the input coordinates (e.g., 'EPSG:4326').
    output_crs : str
        CRS of the output coordinates (e.g., model grid CRS).

    Returns
    -------
    tuple
        (y, x) coordinates in the output CRS.
    """
    
    # Create a transformer
    transformer = pyproj.Transformer.from_crs(input_crs, output_crs, always_xy=True)

    # Perform the transformation
    x, y = transformer.transform(lon, lat)

    return y, x 

def convert_utc_to_local(state, df):
    """
    Convert a UTC datetime column to local time based on U.S. state.

    This function converts the 'Date' column in a dataframe from UTC
    to the local timezone corresponding to the provided U.S. state.
    The converted timestamps are stored in a new column called
    'Date_Local'.

    Parameters
    ----------
    state : str
        U.S. state name or two-letter abbreviation used to determine
        the appropriate timezone.
    df : pandas.DataFrame
        DataFrame containing a 'Date' column with UTC timestamps.

    Returns
    -------
    pandas.DataFrame
        DataFrame with an additional 'Date_Local' column containing
        timezone-adjusted timestamps.
    """

    state_abbreviations = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI",
    "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN",
    "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA",
    "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY"
    }

    state_timezones = {
    'AL': 'US/Central', 'AK': 'US/Alaska', 'AZ': 'US/Mountain', 'AR': 'US/Central',
    'CA': 'US/Pacific', 'CO': 'US/Mountain', 'CT': 'US/Eastern', 'DE': 'US/Eastern',
    'FL': 'US/Eastern', 'GA': 'US/Eastern', 'HI': 'US/Hawaii', 'ID': 'US/Mountain',
    'IL': 'US/Central', 'IN': 'US/Eastern', 'IA': 'US/Central', 'KS': 'US/Central',
    'KY': 'US/Eastern', 'LA': 'US/Central', 'ME': 'US/Eastern', 'MD': 'US/Eastern',
    'MA': 'US/Eastern', 'MI': 'US/Eastern', 'MN': 'US/Central', 'MS': 'US/Central',
    'MO': 'US/Central', 'MT': 'US/Mountain', 'NE': 'US/Central', 'NV': 'US/Pacific',
    'NH': 'US/Eastern', 'NJ': 'US/Eastern', 'NM': 'US/Mountain', 'NY': 'US/Eastern',
    'NC': 'US/Eastern', 'ND': 'US/Central', 'OH': 'US/Eastern', 'OK': 'US/Central',
    'OR': 'US/Pacific', 'PA': 'US/Eastern', 'RI': 'US/Eastern', 'SC': 'US/Eastern',
    'SD': 'US/Central', 'TN': 'US/Central', 'TX': 'US/Central', 'UT': 'US/Mountain',
    'VT': 'US/Eastern', 'VA': 'US/Eastern', 'WA': 'US/Pacific', 'WV': 'US/Eastern',
    'WI': 'US/Central', 'WY': 'US/Mountain'
    }    

    if len(state) == 2:
        state_abbr = state
    else:
        state_abbr = state_abbreviations.get(state, "State not found")

    # Extract the state abbreviation from the filename
    # state_abbr = os.path.basename(filename).split('_')[2]  
    timezone = state_timezones.get(state_abbr)

    if timezone:
        # Convert the 'Date' column to datetime
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
        
        # Convert to local time zone
        local_tz = pytz.timezone(timezone)
        df['Date_Local'] = df['Date'].dt.tz_convert(local_tz)

         # Save the timezone-aware Date_Local column
        df['Date_Local'] = df['Date_Local'].astype(str)
        df['Date_Local'] = df['Date_Local'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S%z'))
        df['Date_Local'] = df['Date_Local'].apply(lambda x: x.replace(tzinfo=None))

    else:
        print(f"Timezone for state abbreviation {state_abbr} not found.")
        
    return df