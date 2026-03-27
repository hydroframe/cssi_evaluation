"""
Evaluation workflow utilities.

Helper functions that orchestrate model-observation comparisons,
metric calculation, and summary statistics.
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from cssi_evaluation.utils import metric_utils as metrics

### LOCATION OF ORIGINAL FUNCTIONS
# nwm_utils.report_max_dates_and_values()
# utils.initialize_metrics_df()
# model_evaluation.calculate_metrics()
# nwm_utils.compute_stats()
# nwm_utils.compute_stats_period()

METRICS_DICT = {
    "rmse": metrics.RMSE,
    "mse": metrics.MSE,
    "pearson_r": metrics.pearson_R,
    "spearman_rho": metrics.spearman_rank,
    "nse": metrics.NSE,
    "kge": metrics.KGE,
    "r_squared": metrics.R_squared,
    "bias_from_r": metrics.bias_from_R,
    "bias": metrics.bias,
    "percent_bias": metrics.percent_bias,
    "abs_rel_bias": metrics.absolute_relative_bias,
    "total_difference": metrics.total_difference,
    "condon": metrics.condon,
}


def initialize_metrics_df(obs_metadata_df, metrics_list):
    """
    Initialize DataFrame table to store metrics output.

    Parameters
    ----------
    obs_metadata_df: DataFrame
        Pandas DataFrame consisting of at least site_id, site_name, latitude, longitude
        values.
    metrics: list
        List of string names of metrics to use for evaluation. Must be present in METRICS_DICT
        dictionary in this module.

    Returns
    -------
    DataFrame
        DataFrame containing site ID, x and y CONUS grid mapping values, along with empty columns
        for each of the evaluation metrics defined in metrics.
    """
    if ("domain_i" not in obs_metadata_df.columns) or (
        "domain_j" not in obs_metadata_df.columns
    ):
        metrics_df = obs_metadata_df[
            ["site_id", "site_name", "latitude", "longitude"]
        ].copy()
    else:
        metrics_df = obs_metadata_df[
            ["site_id", "site_name", "latitude", "longitude", "domain_i", "domain_j"]
        ].copy()
    for m in metrics_list:
        if m == "condon":
            metrics_df[f"{m}"] = ""
        else:
            metrics_df[f"{m}"] = np.nan

    return metrics_df


def calculate_metrics(
    obs_data_df,
    model_data_df,
    obs_metadata_df,
    metrics_list=None,
    write_csv=False,
    csv_path=None,
):
    """
    Calculate comparison metrics between observations and matching model outputs.

    Parameters
    ----------
    obs_data_df : DataFrame
        DataFrame containing the time series observartions for each filtered site for the
        requested time range. One column per site and one row per timestep. This is
        output from the function `get_observations`.
    model_data_df : DataFrame
        DataFrame containing the time series observartions for each matched-site-grid-cell
        for the requested time range. The columns represent each site location requested
        and the rows contain the time series from the model region that contains
        that site.
    obs_metadata_df : DataFrame
        DataFrame with one row per filtered site, containing location and site attribute
        information. It should include at least the columns 'site_id', 'site_name', 'latitude', and 'longitude'.
    metrics_list : list; default=None
        List of metrics to calculate. Defaults to calculating all metrics if none explicitly
        provided.
    write_csv : bool; default=False
        Indicator for whether to additionally write out calculated metrics to disk as .csv.
    csv_path : str; default=None
        Path to where to save .csv of calculated metrics to disk if `write_csv=True`.

    Returns
    -------
    DataFrame
        DataFrame containing one row per site and one column per calculated metric. Contains
        additional site attribute columns for lat/lon and domain grid location.
    """
    # If no metrics_list provided, calculate all available metrics
    if metrics_list is None:
        metrics_list = list(METRICS_DICT.keys())

    # Initialize empty metrics DataFrame to store calculated comparison metrics.
    metrics_df = initialize_metrics_df(obs_metadata_df, metrics_list)

    num_sites = obs_data_df.shape[1] - 1  # first column is 'date'

    for i in range(num_sites):
        site_id = obs_data_df.columns[(i + 1)]

        obs_data = obs_data_df.loc[:, [site_id]].to_numpy()
        model_data = model_data_df.loc[:, [site_id]].to_numpy()

        # Trim arrays to remove matching indices where NaN observations are
        nan_mask = ~np.isnan(obs_data)
        obs_data = obs_data[nan_mask]
        model_data = model_data[nan_mask]

        try:
            assert len(obs_data) == len(model_data)
        except Exception as exc:
            raise ValueError(
                f"""The number of observation timesteps ({len(obs_data)}) does not 
                match the number of model result timesteps ({len(model_data)})."""
            ) from exc

        # Calculate metrics
        for m in metrics_list:
            # too few observations to compare
            if len(model_data) < 2:
                metrics_df.loc[metrics_df["site_id"] == site_id, f"{m}"] = np.nan

            elif m == "condon":
                try:
                    assert ("abs_rel_bias" in METRICS_DICT) and (
                        "spearman_rho" in METRICS_DICT
                    )
                except Exception as exc:
                    raise ValueError(
                        """Please include 'abs_rel_bias' and 'spearman_rho' in the metrics list
                        in order to calculate the Condon category."""
                    ) from exc
                metrics_df.loc[metrics_df["site_id"] == site_id, f"{m}"] = METRICS_DICT[
                    m
                ](
                    metrics_df.loc[
                        metrics_df["site_id"] == site_id, "abs_rel_bias"
                    ].values[0],
                    metrics_df.loc[
                        metrics_df["site_id"] == site_id, "spearman_rho"
                    ].values[0],
                )
            else:
                metrics_df.loc[metrics_df["site_id"] == site_id, f"{m}"] = METRICS_DICT[
                    m
                ](obs_data, model_data)

    # Additionally write to disk if requested
    if write_csv is True:
        metrics_df.to_csv(csv_path, index=False)

    return metrics_df


def report_max_dates_and_values(df, col_obs, col_mod):  # we may no longer need this function, but keeping it here for now just in case.
    # Ensure the index is datetime
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        raise ValueError("DataFrame index must be datetime")

    # Find max values and associated dates
    max_obs = df[col_obs].max()
    date_obs = df[col_obs].idxmax()

    max_mod = df[col_mod].max()
    date_mod = df[col_mod].idxmax()

    # Create a summary table as a DataFrame (nice for Jupyter display)
    summary_table = pd.DataFrame(
        {
            "Data Source": [col_obs, col_mod],
            "Peak SWE (mm)": [max_obs, max_mod],
            "Date of Maximum": [date_obs.date(), date_mod.date()],
        }
    )

    return summary_table


def compute_stats(df, ts1, ts2):
    df = df[[f"{ts1}", f"{ts2}"]]
    df = df[
        [ts1, ts2]
    ].dropna()  # Both Pearson and Spearman correlations cannot handle NaN values, so make sure to drop nan values before any calculatoin.

    # Compute statistics for each time series
    stats = {
        "Mean [L]": [df[f"{ts1}"].mean(), df[f"{ts2}"].mean()],
        "Median [L]": [df[f"{ts1}"].median(), df[f"{ts2}"].median()],
        "Standard Deviation [L]": [df[f"{ts1}"].std(), df[f"{ts2}"].std()],
        "Variance [L^2]": [df[f"{ts1}"].var(), df[f"{ts2}"].var()],
        "Min [L]": [df[f"{ts1}"].min(), df[f"{ts2}"].min()],
        "Max [L]": [df[f"{ts1}"].max(), df[f"{ts2}"].max()],
    }

    # Calculate correlation coefficients
    pearson_corr, _ = pearsonr(df[f"{ts1}"], df[f"{ts2}"])
    spearman_corr, _ = spearmanr(df[f"{ts1}"], df[f"{ts2}"])

    # Compute Mean Bias (mean error)
    bias = df[ts2].mean() - df[ts1].mean()

    # Compute Nash-Sutcliffe Efficiency (NSE)
    obs_mean = df[ts1].mean()
    numerator = np.sum((df[ts2] - df[ts1]) ** 2)
    denominator = np.sum((df[ts1] - obs_mean) ** 2)
    nse = 1 - (numerator / denominator)

    # Compute Kling-Gupta Efficiency (KGE)
    r = pearson_corr
    alpha = df[ts2].std() / df[ts1].std()
    beta = df[ts2].mean() / df[ts1].mean()
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

    # Create a DataFrame for the statistics
    stats_table = pd.DataFrame(stats, index=["observed", "modeled"])

    # Add Pearson and Spearman correlations as additional rows
    stats_table.loc[""] = [""] * len(stats_table.columns)  # Blank row for formatting
    stats_table.loc["Pearson Correlation [-]"] = [pearson_corr, "", "", "", "", ""]
    stats_table.loc["Spearman Correlation [-]"] = [spearman_corr, "", "", "", "", ""]
    stats_table.loc["Bias (Modeled - Observed) [L]"] = [bias, "", "", "", "", ""]
    stats_table.loc["Nash-Sutcliffe Efficiency (NSE) [-]"] = [nse, "", "", "", "", ""]
    stats_table.loc["Kling-Gupta Efficiency (KGE) [-]"] = [kge, "", "", "", "", ""]

    return stats_table


def compute_stats_period(df, ts_obs, ts_mod, months):
    """
    Create a wrapper for compute_stats to filter dataframe by months before computing stats.
    For example, to compute stats for melt season (April to July), use months=[4,5,6,7].

    Parameters:
    - df: DataFrame containing the data.
    - ts_obs: Column name for observed timeseries in df.
    - ts_mod: Column name for modeled timeseries in df.
    - months: List of months to filter the data.

    Returns:
    - DataFrame with computed statistics.
    """
    df_sub = df[df.index.month.isin(months)]

    return compute_stats(df_sub, ts_obs, ts_mod)
