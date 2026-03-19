"""
Plotting utilities.

Functions for visualizing model-observation comparisons, including
time series plots, scatter plots, spatial maps, and evaluation diagrams.
"""

# LOCATION OF ORIGINAL FUNCTION
# plots.plot_condon_diagram()
# plots.plot_time_series()
# plots.plot_compare_scatter()
# nwm_utils.plot_custom_scatter_SWE()
# nwm_utils.comparison_plots()
# nwm_utils.plot_grid_vector_monthly_data()
# nwm_utils.plot_sites_within_domain() overlaps with plot_obs_locations
# nwm_utils.plot_grid_vector_data()
# plots.plot_metric_map()
# plots.plot_obs_locations()
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import xarray as xr
import geopandas as gpd
import holoviews as hv
import hvplot.pandas
import hvplot.xarray
import geoviews as gv
import geoviews.tile_sources as gts
import folium
import xyzservices.providers as xyz

gv.extension("bokeh")

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
    model_data_df,
    obs_metadata_df,
    variable,
    site_list=None,
    output_dir=".",
):
    """
    Plot a time series of model results vs. observations for each site.

    Parameters
    ----------
    obs_data_df : DataFrame
        DataFrame containing the time series observartions for each filtered site for the
        requested time range. One column per site and one row per timestep.
    model_data_df : DataFrame
        DataFrame containing the time series observartions for each matched-site-region
        for the requested time range. The columns represent each site location requested
        and the rows contain the time series from the model grid cell that contains
        that site.
    obs_metadata_df : DataFrame
        DataFrame with one row per filtered site, containing location and site attribute
        information.
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
        pf_data = model_data_df.loc[:, [site_id]]

        # Format dates for x-axis
        dt_series = pd.to_datetime(obs_data_df["date"])
        dt_list = list(dt_series)

        # Plot time series
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))

        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))

        ax.plot(dt_list, pf_data, label="Model")
        ax.plot(dt_list, obs_data, label="Observation")
        ax.set_xticks(ax.get_xticks()[::3])
        ax.legend()

        if variable == "streamflow":
            ax.set_ylabel("Streamflow [cms]")
        elif variable == "water_table_depth":
            ax.set_ylabel("Water Table Depth [m]")
        elif variable == "swe":
            ax.set_ylabel("Snow Water Equivalent [mm]")
        elif variable == "latent_heat":
            ax.set_ylabel("Latent Heat Flux [W/m^2]")

        plt.title(f"{site_name}")
        plt.savefig(f"{output_dir}/{variable}_{site_id}.png", bbox_inches="tight")
        plt.close()


def plot_compare_scatter(
    obs_data_df, model_data_df, variable, log_scale=False, output_dir="."
):
    """
    Plot a time series of model vs. observations for each site.

    Parameters
    ----------
    obs_data_df : DataFrame
        DataFrame containing the time series observartions for each filtered site for the
        requested time range. One column per site and one row per timestep.
    model_data_df : DataFrame
        DataFrame containing the time series observartions for each matched-site-grid-cell
        for the requested time range. The columns represent each site location requested
        and the rows contain the time series from the model grid cell that contains
        that site.
    obs_metadata_df : DataFrame
        DataFrame with one row per filtered site, containing location and site attribute
        information.
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
    pf_mean = pd.DataFrame(model_data_df.iloc[:, 1:].mean(axis=0)).reset_index()
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

    plt.ylabel("Model", fontsize=18)
    plt.xlabel("Observations", fontsize=18)
    plt.title(f"Model vs. observations comparison: {variable}", fontsize=20)
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
            df_plot = metrics_df[metrics_df["condon"] != "Undefined"]
            points = ax.scatter(
                df_plot["domain_i"],
                df_plot["domain_j"],
                c=df_plot[metric].map(metric_cmap),
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
                * 100
            )
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
                * 100
            )
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
                * 100
            )
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
                * 100
            )
        )
        + "%",
        weight="bold",
        fontsize=12,
    )

    plt.title(f"{variable.capitalize()} Performance Category")
    plt.savefig(f"{output_dir}/{variable}_condon_diagram.png", bbox_inches="tight")


# from Irene's nwm_utils.py


def plot_sites_within_domain(gdf_sites, domain_gdf, zoom_start=10):
    """
    Create and return a folium map showing observation sites within a given watershed boundary.

    Parameters:
    - gdf_sites: GeoDataFrame containing site locations.
    - domain_gdf: GeoDataFrame containing the watershed boundary.
    - zoom_start: Initial zoom level for the map (default=10).

    Returns:
    - folium.Map object ready to display.
    """
    # Ensure CRS compatibility
    if gdf_sites.crs != domain_gdf.crs:
        gdf_sites = gdf_sites.to_crs(domain_gdf.crs)

    # Helper to find appropriate column names
    def find_column(columns, candidates):
        return next((c for c in candidates if c in columns), None)

    # Candidate column names (ordered by preference)
    site_name_col = find_column(
        gdf_sites.columns, ["site_name", "station_name", "name", "Site Name"]
    )
    site_id_col = find_column(
        gdf_sites.columns, ["site_id", "station_id", "site_code", "code", "Site Code"]
    )

    # Calculate center of the domain
    minx, miny, maxx, maxy = domain_gdf.total_bounds
    center_lat = (miny + maxy) / 2
    center_lon = (minx + maxx) / 2

    # Create folium map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start)

    # Add site markers
    for _, row in gdf_sites.iterrows():
        site_name = row[site_name_col] if site_name_col else "Site"
        site_id = row[site_id_col] if site_id_col else ""

        folium.Marker(
            location=[row.geometry.y, row.geometry.x],
            popup=f"<b>{site_name}</b><br>Site ID: {site_id}",
            tooltip=f"{site_name} ({site_id})",
            icon=folium.Icon(color="green", icon="info-sign"),
        ).add_to(m)

    # Add watershed boundary
    folium.GeoJson(
        domain_gdf.to_json(),
        name="Watershed Boundary",
        style_function=lambda x: {
            "color": "lightcyan",
            "weight": 2,
            "fillOpacity": 0.3,
        },
    ).add_to(m)

    # Add Esri Imagery basemap
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/Tile/{z}/{y}/{x}",
        attr="Esri, Maxar, Earthstar Geographics, and the GIS User Community",
        name="Esri Imagery",
        overlay=False,
        control=True,
    ).add_to(m)

    # Layer control
    folium.LayerControl(collapsed=False).add_to(m)

    return m


def comparison_plots(df, ts_obs, ts_mod):
    """
    Create a set of comparison plots (timeseries overlay and scatter plot with 1:1 line)

    Parameters:
    df: dataframe with combined observed and modeled timeseries for each site
    ts_obs (str): column heading for observed timeseries in df
    ts_mod (str): column heading for modeled timeseries in df
    """

    df = df.copy()
    df.index.name = (
        "date"  # change the index name to "Date" for better hover tooltip display
    )

    # Timeseries plot (Overlay)
    observed_plot = df.hvplot.line(
        y=ts_obs,
        ylabel="Snow Water Equivalent (mm)",
        xlabel="",
        label="Observed SWE",
        color="blue",
        line_width=2,
        width=500,
        height=400,
    )

    modeled_plot = df.hvplot.line(
        y=ts_mod,
        ylabel="Snow Water Equivalent (mm)",
        xlabel="",
        label="Modeled SWE",
        color="orchid",
        line_width=2,
        width=500,
        height=400,
    )

    # Overlay (combines both lines into a single visual object)
    timeseries_plot = (observed_plot * modeled_plot).opts(
        title="Observed vs Modeled SWE\nDaily Time Series",
        legend_position="top_right",
    )

    # Scatter plot
    scatter_plot = df.hvplot.scatter(
        x=ts_obs,
        y=ts_mod,
        xlabel="Observed SWE (mm)",
        ylabel="Modeled SWE (mm)",
        color="black",
        width=500,
        height=400,
        size=15,
        hover_cols=["date"],  # This will add the date (index) to hover tooltip
    )

    # Add 1:1 line (perfect match line)
    swe_max = max(df[ts_obs].max(), df[ts_mod].max())
    one_to_one_line = (
        hv.Curve(([0, swe_max], [0, swe_max]))
        .opts(
            color="gray",
            line_dash="solid",
            line_width=1,
        )
        .relabel("1:1 Line")
    )  # This is the correct way to set a label for a Curve

    # Combine scatter plot and 1:1 line into an Overlay
    scatter_with_line = (scatter_plot * one_to_one_line).opts(
        title="Observed vs Modeled SWE\nScatter with 1:1 Line",
        legend_position="bottom_right",
    )

    # Combine both into a 1-row, 2-column layout
    layout = (timeseries_plot + scatter_with_line).opts(shared_axes=False)

    return layout


def plot_custom_scatter_SWE(
    df,
    obs_col,
    mod_col,
    *,
    highlight_months=None,
    month_col="month",
    size=15,
    width=500,
    height=400,
):
    """
    Flexible scatter plot with optional month highlighting and 1:1 line.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe
    obs_col : str
        Column name for observed SWE
    mod_col : str
        Column name for modeled SWE
    highlight_months : list[int], optional
        Months to highlight (e.g., [10, 11])
    month_col : str
        Column containing month values
    """

    df = df.copy()

    # Handle highlighting
    if highlight_months is not None and month_col in df.columns:
        df["color"] = df[month_col].apply(
            lambda m: "teal" if m in highlight_months else "tomato"
        )
        color = "color"
    else:
        color = "black"

    scatter = df.hvplot.scatter(
        x=obs_col,
        y=mod_col,
        xlabel="Observed SWE (mm)",
        ylabel="Modeled SWE (mm)",
        title="Observed vs. Modeled SWE at " + obs_col,
        size=15,
        width=500,
        height=400,
        hover_cols=["index", "month"],
        color="color",
    )

    # 1:1 line
    swe_max = max(df[obs_col].max(), df[mod_col].max())
    one_to_one = (
        hv.Curve(([0, swe_max], [0, swe_max]))
        .opts(
            color="gray",
            line_dash="dashed",
            line_width=1,
        )
        .relabel("1:1 Line")
    )

    return (scatter * one_to_one).opts(legend_position="bottom_right")


def plot_grid_vector_data(ds_clip, data_var, time_index, shp, sites):
    hv.extension("bokeh")
    da = ds_clip[data_var]

    # Select one timestep
    if isinstance(time_index, int):
        da = da.isel(time=time_index)
    else:
        da = da.sel(time=time_index)

    # Create an interactive map plot
    clipped = da.rio.reproject("EPSG:4326")
    clipped = clipped.rename({"x": "longitude", "y": "latitude"})
    hvplot_map = clipped.hvplot(
        x="longitude",
        y="latitude",
        geo=True,
        project=True,
        tiles=gts.ESRI,
        cmap="kbc",
        alpha=0.6,
        frame_height=400,
        title=f"Snow Water Equivalent, at {pd.to_datetime(time_index).strftime('%Y-%m-%d %H:%M')}",
        clim=(0, 300),
    )

    shp = shp.to_crs("EPSG:4326").reset_index(drop=True)
    sites = sites.to_crs("EPSG:4326").reset_index(drop=True)

    # Plot the shapefile outline
    shp_plot = shp.hvplot(geo=True, project=True, color="none", line_width=2)

    # Plot sites (scatter)
    points_plot = sites.hvplot.points(
        x="longitude",
        y="latitude",
        geo=True,
        project=True,
        color="red",
        size=100,
        hover_cols=["name"],
    )

    # Combine the two by overlaying
    combined_map = (hvplot_map * shp_plot * points_plot).opts(framewise=True)

    return combined_map


def plot_grid_vector_monthly_data(ds_clip, data_var, shp, sites):
    hv.extension("bokeh")

    # Create an interactive map plot
    clipped = ds_clip[data_var].rio.reproject("EPSG:4326")
    clipped = clipped.rename({"x": "longitude", "y": "latitude"})

    # Plot the shapefile outline
    shp_plot = shp.hvplot(geo=True, project=True, color="none", line_width=2)

    # Plot sites (scatter)
    points_plot = sites.hvplot.points(
        x="longitude",
        y="latitude",
        geo=True,
        project=True,
        color="red",
        size=100,
        hover_cols=["name"],
    )

    # Split into individual plots (list of plots)
    plots = []
    for t in clipped.time.values:
        base_plot = clipped.sel(time=t).hvplot(
            x="longitude",
            y="latitude",
            geo=True,
            project=True,
            tiles=gts.ESRI,
            title=f'SWE (mm) on {pd.to_datetime(t).strftime("%Y-%m-%d")}',
            frame_height=200,
            frame_width=300,
        )
        # Overlay shapefile and points on top of SWE map
        combined_plot = base_plot * shp_plot * points_plot
        plots.append(combined_plot)

    layout = hv.Layout(plots).cols(3)

    return layout
