"""
Snow evaluation utility functions
"""

import pandas as pd


def compute_water_year(df: pd.DataFrame, inplace: bool = False) -> pd.Series:
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
    
    water_year = df.index.map(lambda x: x.year+1 if x.month>9 else x.year)

    if inplace:
        df['Water_Year'] = water_year
        return None
        
    return water_year.to_series()

def same_day_swe_comparison(df: pd.DataFrame, obs_swe_cols: list[str], mod_swe_cols: list[str]) -> pd.DataFrame:
    """
    Computes a Same-Day SWE Comparison between modeled and observed values.

    Parameters
    ==========
    df: pandas.DataFrame
        A pandas dataframe containing columns associated with modeled and observed SWE. The 
        dataframe must have an datetime[64] index.
    obs_swe_cols: list[str]
        Names of the columns associated with observed SWE
    mod_swe_cols: list[str]
        Names of the columns associated with modeled SWE

    Returns
    =======
    df: pandas.DataFrame
        A dataframe containing observed max observed SWE and the modeled SWE at the index of the
        maximum observed. The format of the DataFrame will be:

                Observed     Modeled              Water_Year       Station
        <date>  <max value>  <value at obs max>   <water year>    <obs station name>
        <date>  <max value>  <value at obs max>   <water year>    <obs station name>
        ...

        Example:

                	Observed	Modeled	  Water_Year	Station
        2019-04-18	0.98044	    1.0293	  2019	        CCSS_DAN_swe_m
        2019-04-20	2.12090	    1.3598	  2019	        CCSS_HRS_swe_m
        2019-03-28	0.80264	    0.6708	  2019	        CCSS_KIB_swe_m
        2019-04-07	1.78562	    0.9965	  2019	        CCSS_PDS_swe_m
        ...
        
    """

    # compute water year if it doesn't already exist in the dataframe.
    # this is needed to properly align the same-day comparison
    if "Water_Year" not in df.columns:
        compute_water_year(df, inplace=True)

    # check to make sure that the input columns are the same length. 
    # Raise an exception if they aren't, because our computation will fail.
    if len(obs_swe_cols) != len(mod_swe_cols):
        raise Exception('Modeled and observed inputs must be the same length')

    # make sure our column data is represented as float64, otherwise
    # the pandas operations below will fail.
    df = df.apply(pd.to_numeric, errors='coerce').astype('float64')
    df['Water_Year'] = df['Water_Year'].astype(int) # keep wateryear an integer

    # Loop over each pairwise grouping of obs and mod columns that
    # have been provided as inputs. Group data for these stations
    # by water year and determine when the maximum value occurs in
    # the observation series. Save this value along with the corresponding
    # mod value at the same time.
    dfs = []
    for obs, mod in zip(obs_swe_cols, mod_swe_cols):

        # get the data for the current obs and mod columns
        # but drop all NaN data that may exist.
        dat = df.dropna(subset=[obs, mod, 'Water_Year']).copy()
        
        # if all data is NaN for the current obs, mod combination
        # just skip it.
        if dat.empty:
            print(f'Skipping ({obs}, {mod}) because all data is NaN')
            continue
        
        idx = dat.groupby('Water_Year')[obs].idxmax()
        dat = dat.loc[idx, [obs, mod,'Water_Year']].copy()
    
        dat.rename(columns={obs:'Observed', mod:'Modeled'}, inplace=True)
        dat['Station'] = obs
    
        dfs.append(dat)

    # concatenate all dataframes together and return
    return pd.concat(dfs)