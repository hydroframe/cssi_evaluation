"""Functions to support plotting within the model evaluation module."""
import os
from matplotlib.lines import Line2D
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm

import utils

SITE_COLORS = {
    "stream gauge": "blue",
    "groundwater well": "green",
    "SNOTEL station": "purple",
    "flux tower": "orange",
}

CONDON_COLORS = {
    "Low bias, good shape": (0 / 255, 128 / 255, 0 / 255),
    "High bias, good shape": (0, 0, 1),
    "Low bias, poor shape": (138 / 255, 43 / 255, 226 / 255),
    "High bias, poor shape": (187 / 255, 34 / 255, 34 / 255),
}
CONDON_LABELS = [
    "Low bias, good shape",
    "High bias, good shape",
    "Low bias, poor shape",
    "High bias, poor shape",
]


STATES_SHP = "/hydrodata/national_mapping/NaturalEarth/US_states.shp"

font_prop = fm.FontProperties(fname="Roboto-Regular.ttf", size=7.9)
legend_tick_prop = fm.FontProperties(fname="Roboto-Regular.ttf", size=6.5)


def plot_obs_locations(obs_metadata_df, mask, file_path):
    """
    Plot domain mask with locations of sites within the domain.

    Parameters
    ----------
    obs_metadata_df : DataFrame
        Observation metadata DataFrame containing the site_type and domain
        indices for each site.
    mask : array
        NumPy array representing the domain mask.
    file_path : str
        File path for saving plot.

    Returns
    -------

    """
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))

    ax1.imshow(mask, origin="lower", cmap="Greys_r")
    ax1.scatter(
        obs_metadata_df["domain_i"],
        obs_metadata_df["domain_j"],
        c=obs_metadata_df["site_type"].map(SITE_COLORS),
        alpha=0.6,
    )
    ax1.set_aspect("equal")
    ax1.set_title(f"Locations of Observations", fontsize=15)

    handles = [
        Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=v, label=k, markersize=8
        )
        for k, v in SITE_COLORS.items()
    ]
    ax1.legend(
        title="site type", handles=handles, bbox_to_anchor=(1.05, 1), loc="upper left"
    )

    plt.savefig(f"{file_path}", bbox_inches="tight")


def plot_time_series(
    obs_data_df,
    parflow_data_df,
    obs_metadata_df,
    variable,
    site_list=None,
    output_dir=".",
):
    """
    Plot a time series of ParFlow vs. observations for each site.

    Parameters
    ----------
    obs_data_df : DataFrame
        DataFrame containing the time series observartions for each filtered site for the
        requested time range. One column per site and one row per timestep. This is
        output from the function `get_observations`.
    parflow_data_df : DataFrame
        DataFrame containing the time series observartions for each matched-site-grid-cell
        for the requested time range. The columns represent each site location requested
        and the rows contain the time series from the ParFlow grid cell that contains
        that site. This is output from the function `get_parflow_output`.
    obs_metadata_df : DataFrame
        DataFrame with one row per filtered site, containing location and site attribute
        information. This is output from the function `get_observations`.
    variable : str
        Type of variable being compared and plotted (ie. 'streamflow', 'water_table_depth', 'swe')
    site_list : list of str; default=None
        Optional list of strings to indicate only a subset of sites should have plots made.
        The site ID values in this list must exist in obs_metadata_df and each of the two data
        DataFrames or else an error will be raised.
    output_dir : str; default="."
        String path to where plots should be saved. Default is current working directory.

    Returns
    -------
    None
        Saves one plot per time series to output_dir.
    """
    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if site_list is not None:
        num_sites = len(site_list)
    else:
        num_sites = len(obs_metadata_df)
        site_list = list(obs_metadata_df["site_id"])

    # Create plot for each site
    for i in range(num_sites):
        site_id = site_list[i]
        site_name = obs_metadata_df[obs_metadata_df["site_id"] == site_id][
            "site_name"
        ].values[0]

        # Get time series for a single site
        obs_data = obs_data_df.loc[:, [site_id]]
        pf_data = parflow_data_df.loc[:, [site_id]]

        # Format dates for x-axis
        dt_series = pd.to_datetime(obs_data_df["date"])
        dt_list = list(dt_series)

        # Plot time series
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))

        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))

        ax.plot(dt_list, pf_data, label="ParFlow")
        ax.plot(dt_list, obs_data, label="Observation")
        ax.set_xticks(ax.get_xticks()[::3])
        ax.legend()

        if variable == "streamflow":
            ax.set_ylabel("Streamflow [cms]")
        elif variable == "water_table_depth":
            ax.set_ylabel("Water Table Depth [m]")
        elif variable == "swe":
            ax.set_ylabel("Snow Water Equivalent [mm]")

        plt.title(f"{site_name}")
        plt.savefig(f"{output_dir}/{variable}_{site_id}.png", bbox_inches="tight")


def plot_compare_scatter(
    obs_data_df, parflow_data_df, variable, log_scale=False, output_dir="."
):
    """
    Plot a time series of ParFlow vs. observations for each site.

    Parameters
    ----------
    obs_data_df : DataFrame
        DataFrame containing the time series observartions for each filtered site for the
        requested time range. One column per site and one row per timestep. This is
        output from the function `get_observations`.
    parflow_data_df : DataFrame
        DataFrame containing the time series observartions for each matched-site-grid-cell
        for the requested time range. The columns represent each site location requested
        and the rows contain the time series from the ParFlow grid cell that contains
        that site. This is output from the function `get_parflow_output`.
    obs_metadata_df : DataFrame
        DataFrame with one row per filtered site, containing location and site attribute
        information. This is output from the function `get_observations`.
    variable : str
        Type of variable being compared and plotted (ie. 'streamflow', 'water_table_depth', 'swe')
    log_scale : bool; default=False
        Produce plot with log scale axes.
    output_dir : str; default="."
        String path to where plots should be saved. Default is current working directory.

    Returns
    -------
    None
        Saves one plot to output_dir.
    """
    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Calculate mean values per site
    obs_mean = pd.DataFrame(obs_data_df.iloc[:, 1:].mean(axis=0)).reset_index()
    obs_mean.columns = ["site_id", "obs_mean"]
    pf_mean = pd.DataFrame(parflow_data_df.iloc[:, 1:].mean(axis=0)).reset_index()
    pf_mean.columns = ["site_id", "pf_mean"]
    merged_mean = pd.merge(obs_mean, pf_mean, on="site_id")

    # Scatterplot
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    ax.scatter(merged_mean["obs_mean"], merged_mean["pf_mean"], c="dimgrey")

    ax.plot(
        range(round(max(merged_mean["obs_mean"].max(), merged_mean["pf_mean"].max()))),
        color="lightcoral",
    )
    if log_scale is True:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)

    plt.ylabel("ParFlow", fontsize=18)
    plt.xlabel("Observations", fontsize=18)
    plt.title(f"ParFlow vs. observations comparison: {variable}", fontsize=20)
    plt.savefig(f"{output_dir}/{variable}_comparison_scatter.png", bbox_inches="tight")


def plot_metric_map(mask, metrics_df, variable, metrics_list, output_dir="."):
    """
    Create a map overlaid by a scatterplot where each point represents a site's value on a given
    metric.

    Parameters
    ----------
    mask : array
        Array representing the domain mask, output from something like subsettools.define_huc_domain().
    metrics_df : DataFrame
        DataFrame containing one row per site and one column per calculated metric. Contains
            additional site attribute columns for lat/lon and domain grid location.
    variable : str
        Type of variable being compared and plotted (ie. 'streamflow', 'water_table_depth', 'swe')
    metrics_list : list of str
        List containing the names of metrics to create plots for. One plot per metric will be
        produced and saved.
    output_dir : str; default="."
        String path to where plots should be saved. Default is current working directory.

    Returns
    -------
    None
        Saves one plot per metric in metrics_list to output_dir.
    """
    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for metric in metrics_list:
        # Expand for full list of metrics
        if metric in ["rmse", "mse", "percent_bias", "abs_rel_bias"]:
            metric_cmap = "Oranges"
        elif metric in ["r2"]:
            metric_cmap = "Blues"
        elif metric in ["spearman_rho"]:
            metric_cmap = "RdYlGn"
        elif metric in ["bias", "total_difference"]:
            metric_cmap = "RdYlBu"
        elif metric == "condon":
            metric_cmap = CONDON_COLORS
        else:
            raise ValueError(
                f"The metric {metric} is not currently supported. Please reach out to explore how we might add support for it."
            )

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        if metric == "condon":
            ax.imshow(mask, origin="lower", cmap="Greys_r", alpha=0.1)
            points = ax.scatter(
                metrics_df["domain_i"],
                metrics_df["domain_j"],
                c=metrics_df[metric].map(metric_cmap),
                s=20,
            )
            # add categorical legend
            custom = [
                Line2D(
                    [],
                    [],
                    marker=".",
                    color=metric_cmap["Low bias, good shape"],
                    linestyle="None",
                ),
                Line2D(
                    [],
                    [],
                    marker=".",
                    color=metric_cmap["High bias, good shape"],
                    linestyle="None",
                ),
                Line2D(
                    [],
                    [],
                    marker=".",
                    color=metric_cmap["Low bias, poor shape"],
                    linestyle="None",
                ),
                Line2D(
                    [],
                    [],
                    marker=".",
                    color=metric_cmap["High bias, poor shape"],
                    linestyle="None",
                ),
            ]
            legend = ax.legend(
                handles=custom,
                labels=CONDON_LABELS,
                loc="lower left",
                markerscale=1,
                frameon=False,
                borderpad=0,
                labelspacing=0.4,
            )
            ax.add_artist(legend)

        else:
            ax.imshow(mask, origin="lower", cmap="Greys_r", alpha=0.1)
            points = ax.scatter(
                metrics_df["domain_i"],
                metrics_df["domain_j"],
                c=metrics_df[metric],
                s=20,
                cmap=metric_cmap,
            )
            plt.colorbar(points, label=metric, shrink=0.75)

        plt.title(f"{variable} {metric}")
        plt.savefig(f"{output_dir}/{variable}_map_{metric}.png", bbox_inches="tight")


def plot_condon_diagram(metrics_df, variable, output_dir="."):
    """
    Create a Condon diagram.

    Parameters
    ----------
    metrics_df : DataFrame
        DataFrame containing one row per site and one column per calculated metric. Contains
        additional site attribute columns for lat/lon and domain grid location. Must include
        'condon', 'spearman_rho', and 'abs_rel_bias' for this plot.
    variable : str
        Type of variable being compared and plotted (ie. 'streamflow', 'water_table_depth', 'swe')
    output_dir : str; default="."
        String path to where plots should be saved. Default is current working directory.

    Returns
    -------
    None
        Saves plot to output_dir.
    """
    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df_plot = metrics_df[metrics_df["condon"] != "Undefined"]

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(
        df_plot["abs_rel_bias"],
        df_plot["spearman_rho"],
        c=df_plot["condon"].map(CONDON_COLORS),
        s=4,
        zorder=1,
        alpha=0.4,
    )

    custom = [
        Line2D(
            [],
            [],
            marker=".",
            color=CONDON_COLORS["Low bias, good shape"],
            linestyle="None",
        ),
        Line2D(
            [],
            [],
            marker=".",
            color=CONDON_COLORS["High bias, good shape"],
            linestyle="None",
        ),
        Line2D(
            [],
            [],
            marker=".",
            color=CONDON_COLORS["Low bias, poor shape"],
            linestyle="None",
        ),
        Line2D(
            [],
            [],
            marker=".",
            color=CONDON_COLORS["High bias, poor shape"],
            linestyle="None",
        ),
    ]
    legend = ax.legend(
        handles=custom,
        labels=CONDON_LABELS,
        loc=8,
        markerscale=1,
        frameon=False,
        borderpad=0,
        labelspacing=0.4,
    )
    ax.add_artist(legend)

    ax.vlines(1, -1, 1, colors="k")
    ax.hlines(0.5, 0, 10, colors="k")
    ax.set_xlabel("Absolute Relative Bias", fontsize=12)
    ax.set_ylabel("Spearman's Rho", fontsize=12)
    ax.set_xlim(0, 10)
    ax.set_ylim(-1, 1)
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.set_yticks([-1.0, -0.5, 0, 0.5, 1])
    ax.tick_params(axis="both", labelsize=12)

    # Add text for the percentage in each category
    total_obs = df_plot.shape[0]
    ax.text(
        0,
        0.9,
        str(
            round(
                df_plot[df_plot["condon"] == "Low bias, good shape"].shape[0]
                / total_obs
            )
            * 100
        )
        + "%",
        weight="bold",
        fontsize=12,
    )
    ax.text(
        9.3,
        0.9,
        str(
            round(
                df_plot[df_plot["condon"] == "High bias, good shape"].shape[0]
                / total_obs
            )
            * 100
        )
        + "%",
        weight="bold",
        fontsize=12,
    )
    ax.text(
        0,
        -1,
        str(
            round(
                df_plot[df_plot["condon"] == "Low bias, poor shape"].shape[0]
                / total_obs
            )
            * 100
        )
        + "%",
        weight="bold",
        fontsize=12,
    )
    ax.text(
        9.3,
        -1,
        str(
            round(
                df_plot[df_plot["condon"] == "High bias, poor shape"].shape[0]
                / total_obs
            )
            * 100
        )
        + "%",
        weight="bold",
        fontsize=12,
    )

    plt.title(f"{variable.capitalize()} Performance Category")
    plt.savefig(f"{output_dir}/{variable}_condon_diagram.png", bbox_inches="tight")
