"""
National Water Model (NWM) utilities.

Functions for preprocessing NWM outputs, downloading data,handling coordinate conversions,
and preparing datasets for comparison with observations.
"""

import sys
from pathlib import Path

# Dynamically get repo root relative to this file
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]  # Adjust based on how deep src is (here 3 levels up to project root)
SRC_PATH = REPO_ROOT / "src"

# Add src to sys.path if not already there
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import pandas as pd
import xarray as xr
import time
import pyproj
from cssi_evaluation.utils import dataPrep_utils


def getNWMSWE(
    gdf_in_bbox,
    input_crs,
    network,
    conus_bucket_url,
    StartDate,
    EndDate,
    OutputFile=None
):
    """
    Retrieve modeled SWE (SNEQV) for multiple sites from a Zarr dataset
    and return a merged dataframe similar to CCSS observations.

    Parameters
    ----------
    gdf_in_bbox : GeoDataFrame
        Sites to retrieve data for (must include: code, name, latitude, longitude, state)
    conus_bucket_url : str
        URL or path to the modeled Zarr dataset
    StartDate, EndDate : str
        Date range to extract (YYYY-MM-DD)
    OutputFile : str, optional
        If provided, save the merged dataframe to CSV

    Returns
    -------
    merged_df : pandas.DataFrame
        Merged dataframe with Date + one column per site: SITEID:STATE:NWM
    """

    # Open modeled dataset
    ds = xr.open_zarr(
        store=conus_bucket_url,
        consolidated=True,
        storage_options={
            "anon": True,
            "client_kwargs": {"region_name": "us-east-1"}
        }
    )

    #input_crs = 'EPSG:4269' # NAD83 lat/lon. Given as argument now
    output_crs = pyproj.CRS(ds.crs.esri_pe_string)  # modeled CRS
    dataframes = []

    for i in range(len(gdf_in_bbox)):
        site = gdf_in_bbox.iloc[i]
        site_name = site["name"]
        site_code = site["code"]
        state = site["state"]

        # Convert lat/lon → dataset coordinates
        snotel_y, snotel_x = dataPrep_utils.convert_latlon_to_yx(
            site.latitude,
            site.longitude,
            input_crs,
            output_crs
        )

        dl_start_time = time.time()

        # Subset dataset
        ds_subset = ds[['SNEQV']].sel(
            y=snotel_y,
            x=snotel_x,
            method='nearest'
        ).sel(time=slice(StartDate, EndDate)).compute()

        elapsed = time.time() - dl_start_time
        print(f"✅ Retrieved {site_name} ({site_code}) in {elapsed:.2f}s")

        # Convert to dataframe
        df = ds_subset.to_dataframe().reset_index()
        df = df.drop(columns=['x', 'y'])
        df["time"] = pd.to_datetime(df["time"])
        df.rename(columns={"time": "Date", "SNEQV": "NWM_SWE_meters"}, inplace=True)
        df["NWM_SWE_meters"] = pd.to_numeric(df["NWM_SWE_meters"]) / 1000  # mm → m

        # Convert to local time
        df_local = dataPrep_utils.convert_utc_to_local(state, df)

        # Aggregate to daily values
        df_local.index = pd.to_datetime(df_local['Date_Local'])
        df_local = df_local.groupby(pd.Grouper(freq='D')).first()

        # Keep only SWE column and rename to SITEID:STATE:NETWORK
        col_name = f"{site_code}:{state[:2].upper()}:{network}"
        df_final = df_local[["NWM_SWE_meters"]].rename(columns={"NWM_SWE_meters": col_name})
        df_final.index.name = "Date"

        dataframes.append(df_final)

    # Merge all sites into one dataframe
    merged_df = pd.concat(dataframes, axis=1).reset_index()

    # save to CSV
    if OutputFile:
        merged_df.to_csv(OutputFile, index=False)
        print(f"\n✅ Merged modeled data saved to: {OutputFile}")

    return merged_df

def getNWMQ(
    gdf_sites,
    conus_bucket_url,
    StartDate,
    EndDate,
    OutputFile=None,
):
    """
    Retrieve modeled NWM streamflow for multiple USGS gauges from the
    retrospective channel-routing Zarr store.

    Parameters
    ----------
    gdf_sites : GeoDataFrame or DataFrame
        Table of USGS gauges to retrieve data for. Must contain a `site_id`
        column matching the NWM `gage_id` values and ideally a `site_name`
        column for progress messages.
    conus_bucket_url : str
        URL or path to the modeled NWM channel-routing Zarr dataset.
    StartDate, EndDate : str
        Date range to extract (YYYY-MM-DD).
    OutputFile : str, optional
        If provided, save the merged dataframe to CSV.

    Returns
    -------
    merged_df : pandas.DataFrame
        DataFrame with `date` plus one column per site_id containing modeled
        streamflow in m3/s.
    """

    site_ids = gdf_sites["site_id"].astype(str).tolist()
    formatted_gauge_ids = [f"{site_id:>15}".encode("ascii") for site_id in site_ids]

    ds = xr.open_zarr(
        store=conus_bucket_url,
        consolidated=True,
        storage_options={
            "anon": True,
            "client_kwargs": {"region_name": "us-east-1"},
        },
    )

    mask = ds.gage_id.isin(formatted_gauge_ids)
    ds_subset = ds.where(mask.compute(), drop=True)

    model_long_df = (
        ds_subset[["streamflow"]]
        .sel(time=slice(StartDate, EndDate))
        .streamflow.to_dataframe()
        .reset_index()
    )

    model_long_df["gage_id"] = (
        model_long_df["gage_id"].str.decode("utf-8").str.strip()
    )
    model_long_df.rename(columns={"time": "date", "gage_id": "site_id"}, inplace=True)
    model_long_df["date"] = pd.to_datetime(model_long_df["date"])

    merged_df = (
        model_long_df.pivot(index="date", columns="site_id", values="streamflow")
        .reset_index()
        .sort_values("date")
    )
    merged_df.columns.name = None

    matched_sites = [col for col in merged_df.columns if col != "date"]
    print(f"Matched {len(matched_sites)} NWM gauges out of {len(site_ids)} requested.")

    missing_sites = sorted(set(site_ids) - set(matched_sites))
    if missing_sites:
        print("Sites not found in NWM gage_id:", missing_sites)

    if OutputFile:
        merged_df.to_csv(OutputFile, index=False)
        print(f"\nSaved modeled streamflow data to: {OutputFile}")

    return merged_df
