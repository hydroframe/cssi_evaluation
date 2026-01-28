"""
Snow evaluation utility functions
"""

import pandas as pd
from typing import Any, Union


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


def modeled_swe_at_observed_peak(
    df: pd.DataFrame, obs_swe_cols: list[str], mod_swe_cols: list[str]
) -> pd.DataFrame:
    """
    Extract modeled SWE values on the dates of observed peak (maximum) SWE.

    This function evaluates model performance by comparing observed peak SWE
    to the modeled SWE on the same calendar date. For each station and water year,
    the date of maximum observed SWE is identified, and the modeled SWE value
    at that date is extracted.

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
        raise Exception("Modeled and observed inputs must be the same length")

    # make sure our column data is represented as float64, otherwise
    # the pandas operations below will fail.
    df = df.apply(pd.to_numeric, errors="coerce").astype("float64")  # type: ignore[assignment]
    df["Water_Year"] = df["Water_Year"].astype(int)  # keep wateryear an integer

    # Loop over each pairwise grouping of obs and mod columns that
    # have been provided as inputs. Group data for these stations
    # by water year and determine when the maximum value occurs in
    # the observation series. Save this value along with the corresponding
    # mod value at the same time.
    dfs = []
    for obs, mod in zip(obs_swe_cols, mod_swe_cols):

        # get the data for the current obs and mod columns
        # but drop all NaN data that may exist.
        dat = df.dropna(subset=[obs, mod, "Water_Year"]).copy()

        # if all data is NaN for the current obs, mod combination
        # just skip it.
        if dat.empty:
            print(f"Skipping ({obs}, {mod}) because all data is NaN")
            continue

        idx = dat.groupby("Water_Year")[obs].idxmax()
        dat = dat.loc[idx, [obs, mod, "Water_Year"]].copy()

        dat.rename(columns={obs: "Observed", mod: "Modeled"}, inplace=True)
        dat["Station"] = obs

        dfs.append(dat)

    # concatenate all dataframes together and return
    return pd.concat(dfs)


def modeled_vs_observed_peak_swe(
    df: pd.DataFrame, obs_swe_cols: list[str], mod_swe_cols: list[str]
) -> pd.DataFrame:
    """
    Extract and compare modeled and observed peak (maximum) SWE values and their timing.

    This function identifies the dates and magnitudes of peak SWE
    independently for both observed and modeled time series. For each station
    and water year, it extracts the maximum observed SWE and its occurrence date,
    as well as the maximum modeled SWE and its occurrence date.

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
        A dataframe containing maximum observed and modeled SWE at their respective times of
        occurence. The format of the DataFrame will be:

           Observed     Observed_Date   Modeled      Modeled_Date    Water_Year    Station
        0  <max value>  <obs max date>  <max value>  <mod max date>  <water year>  <obs station name>
        1  <max value>  <obs max date>  <max value>  <mod max date>  <water year>  <obs station name>
        ...

        Example:

            Observed  Observed_Date	 Modeled  Modeled_Date  Water_Year  Station
        0	0.98044	  2019-04-18	 1.0393	  2019-04-10    2019	    CCSS_DAN_swe_m
        1	0.41910	  2020-04-21	 0.5206	  2020-04-18    2020	    CCSS_DAN_swe_m
        2	2.12090	  2019-04-20	 1.5498	  2019-04-03    2019	    CCSS_HRS_swe_m
        3	0.89662	  2020-04-10	 0.5745	  2020-04-10    2020	    CCSS_HRS_swe_m
        ...


    """

    # compute water year if it doesn't already exist in the dataframe.
    # this is needed to properly align the same-day comparison
    if "Water_Year" not in df.columns:
        compute_water_year(df, inplace=True)

    # check to make sure that the input columns are the same length.
    # Raise an exception if they aren't, because our computation will fail.
    if len(obs_swe_cols) != len(mod_swe_cols):
        raise Exception("Modeled and observed inputs must be the same length")

    # make sure our column data is represented as float64, otherwise
    # the pandas operations below will fail.
    df = df.apply(pd.to_numeric, errors="coerce").astype("float64")  # type: ignore[assignment]
    df["Water_Year"] = df["Water_Year"].astype(int)  # keep wateryear an integer

    # Loop over each pairwise grouping of obs and mod columns that
    # have been provided as inputs. Group data for these stations
    # by water year and determine when the maximum value occurs in
    # both the observation and modeled series. Save these values
    # along with their corresponding times
    dfs = []
    for obs, mod in zip(obs_swe_cols, mod_swe_cols):

        # get the data for the current obs and mod columns
        # but drop all NaN data that may exist.
        dat = df.dropna(subset=[obs, mod, "Water_Year"]).copy()

        # if all data is NaN for the current obs, mod combination
        # just skip it.
        if dat.empty:
            print(f"Skipping ({obs}, {mod}) because all data is NaN")
            continue

        obs_idx = dat.groupby("Water_Year")[obs].idxmax()
        obs_dat = dat.loc[obs_idx, [obs, "Water_Year"]].copy()
        obs_dat = obs_dat.rename(columns={obs: "Observed"})
        obs_dat["Observed_Date"] = obs_idx.values

        mod_idx = dat.groupby("Water_Year")[mod].idxmax()
        mod_dat = dat.loc[mod_idx, [mod, "Water_Year"]].copy()
        mod_dat = mod_dat.rename(columns={mod: "Modeled"})
        mod_dat["Modeled_Date"] = mod_idx.values

        dfs.append(
            # combine the observation and modeled sub-dataframes into one
            # by joining them on Water_Year. Then add
            obs_dat.merge(mod_dat, on="Water_Year", how="outer").assign(
                # create a new "Station" column containing the value of the obs
                Station=obs
            )
        )

    # concatenate all dataframes together and return
    return pd.concat(dfs).reset_index().drop("index", axis=1)


def compute_melt_period(
    swe_series: pd.Series, min_zero_days: int = 10
) -> dict[str, Any]:
    """
    computes the snow melt period for the input Series.

    Parameters
    ==========
    swe_series: pandas.Series
        A pandas series containing SWE values indexed by datetime.
    min_zero_days: int -> 10
        The minimum number of consecutive days with zero SWE to consider
        when determining the melt end date.

    Returns
    =======
    dict[str, Any]
        A dictionary containing melt period information with the following keys:
        peak_date, peak_swe_m, melt_end_date, melt_period_days, melt_rate_m/d

    """

    peak_date = swe_series.idxmax()
    peak_swe = swe_series.max()

    after_peak = swe_series.loc[peak_date:]

    zero_streak = 0
    melt_end_date = None

    for date, value in after_peak.items():
        if value == 0:
            zero_streak += 1
        else:
            zero_streak = 0

        if zero_streak >= min_zero_days:
            melt_end_date = date
            break

    if melt_end_date is None:
        raise ValueError(
            f"Could not find a period of at least {min_zero_days} consecutive zero SWE days after the peak."
        )

    melt_period_days = (melt_end_date - peak_date).days
    melt_rate = peak_swe / melt_period_days

    return {
        "peak_date": peak_date,
        "peak_swe_m": peak_swe,
        "melt_end_date": melt_end_date,
        "melt_period_days": melt_period_days,
        "melt_rate_m/d": melt_rate,
    }


def compute_melt_period_statistics(
    df: pd.DataFrame, min_zero_days: int = 10
) -> pd.DataFrame:
    """
    Computes melt period statistics for each station and water year in the input DataFrame.

    Parameters
    ==========

    Returns
    =======
    pandas.DataFrame
        A pandas DataFrame containing melt period statistics with the following columns:
        Water_Year, Station, Peak_SWE_Date, Peak_SWE_m, Melt_End_Date, Melt_Period_Days,
        Melt_Rate_m_per_day

    """

    # TODO: move ccss columns as an input parameter
    result = []

    # Identify CCSS SWE columns
    ccss_columns = [
        col for col in df.columns if col.startswith("CCSS_") and col.endswith("_swe_m")
    ]

    for wy, group in df.groupby("Water_Year"):
        for station_col in ccss_columns:

            # TODO: refactore dropna handling similar to other functions
            # Clean series
            swe_series = pd.to_numeric(group[station_col], errors="coerce").dropna()

            # Skip if insufficient data
            if swe_series.empty or swe_series.max() == 0:
                continue

            try:
                # Compute melt period stats
                stats = compute_melt_period(swe_series, min_zero_days=min_zero_days)
                result.append(
                    {
                        "Water_Year": wy,
                        "Station": station_col,
                        "Peak_SWE_Date": stats["peak_date"],
                        "Peak_SWE_m": stats["peak_swe_m"],
                        "Melt_End_Date": stats["melt_end_date"],
                        "Melt_Period_Days": stats["melt_period_days"],
                        "Melt_Rate_m_per_day": stats["melt_rate_m/d"],
                    }
                )
            except ValueError:
                # If melt period cannot be determined, skip
                continue

    return pd.DataFrame(result)

