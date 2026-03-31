"""
SUMMA model utilities.

Functions for reading SUMMA and mizuRoute NetCDF outputs, handling HRU
mapping, segment selection, and preparing datasets for comparison with
observations.

Inspired by the SYMFLUENCE SUMMAResultExtractor and MizuRouteResultExtractor
for variable naming conventions, spatial aggregation, and segment selection
patterns.
"""

import warnings
import numpy as np
import pandas as pd
import xarray as xr

warnings.simplefilter(action="ignore", category=FutureWarning)

# SUMMA variable names for each supported variable type
SUMMA_VARIABLE_NAMES = {
    "swe": ["scalarSWE"],
    "streamflow": ["averageRoutedRunoff", "basin__TotalRunoff", "scalarTotalRunoff"],
    "latent_heat": ["scalarLatHeatTotal"],
    "total_et": ["scalarTotalET"],
    "soil_moisture": ["scalarTotalSoilWat"],
    "aquifer_storage": ["scalarAquiferStorage"],
}

# Default file patterns for each variable type
SUMMA_FILE_PATTERNS = {
    "swe": ["*_day.nc"],
    "streamflow": ["*_timestep.nc", "*_day.nc"],
    "latent_heat": ["*_day.nc"],
    "total_et": ["*_day.nc"],
    "soil_moisture": ["*_day.nc"],
    "aquifer_storage": ["*_day.nc"],
}


def get_summa_output(
    summa_output_dir,
    variable,
    date_start,
    date_end,
    temporal_resolution="daily",
    attributes_file=None,
    site_mapping=None,
    obs_metadata_df=None,
    file_pattern=None,
    write_csv=False,
    csv_path=None,
):
    """
    Extract SUMMA outputs for evaluation against observations.

    This function reads SUMMA NetCDF output files and returns a DataFrame
    compatible with the cssi_evaluation pipeline (same format as
    ``get_parflow_output`` and ``getNWMSWE``).

    Parameters
    ----------
    summa_output_dir : str or Path
        Directory containing SUMMA output NetCDF files.
    variable : str
        Variable to extract. One of: 'swe', 'streamflow', 'latent_heat',
        'total_et', 'soil_moisture', 'aquifer_storage'.
    date_start : str or datetime-like
        Start date for extraction (inclusive), e.g. '2002-10-01'.
    date_end : str or datetime-like
        End date for extraction (inclusive), e.g. '2007-09-30'.
    temporal_resolution : str, default 'daily'
        'daily' or 'hourly'. Determines which output file to read
        (*_day.nc vs *_timestep.nc) and how to resample.
    attributes_file : str or Path, optional
        Path to SUMMA attributes.nc file. Used for area-weighted spatial
        aggregation when the domain has multiple HRUs. If None and
        multiple HRUs exist, the function falls back to simple selection.
    site_mapping : dict, optional
        Mapping of site_id -> HRU index (0-based) for extracting specific
        HRU values in multi-HRU domains. For example:
        ``{'SNOTEL_site1': 0, 'SNOTEL_site2': 3}``.
        If None and multiple HRUs exist, each HRU is treated as a separate site.
    obs_metadata_df : DataFrame, optional
        DataFrame with observation site metadata (site_id, latitude, longitude).
        If provided alongside a multi-HRU domain without site_mapping, the
        function will attempt nearest-HRU matching using lat/lon from
        attributes.nc.
    file_pattern : str, optional
        Glob pattern to locate the output file within summa_output_dir.
        If None, uses default patterns based on variable and temporal_resolution.
    write_csv : bool, default False
        Whether to write the output DataFrame to CSV.
    csv_path : str, optional
        Path for the CSV output file. Required if write_csv is True.

    Returns
    -------
    DataFrame
        DataFrame with a 'date' column and one column per site/HRU.
        Compatible with ``calculate_metrics()`` in evaluation_utils.
    """
    from pathlib import Path

    summa_output_dir = Path(summa_output_dir)

    if variable not in SUMMA_VARIABLE_NAMES:
        raise ValueError(
            f"Variable '{variable}' not supported. "
            f"Must be one of: {list(SUMMA_VARIABLE_NAMES.keys())}"
        )

    # --- Locate the output file ---
    output_file = _find_output_file(
        summa_output_dir, variable, temporal_resolution, file_pattern
    )

    # --- Read the NetCDF dataset ---
    ds = _open_netcdf(output_file)

    # Find the appropriate SUMMA variable name in the file
    var_name = _find_variable_in_dataset(ds, variable)

    # Extract the variable data
    var_data = ds[var_name]

    # --- Time subsetting ---
    var_data = var_data.sel(time=slice(str(date_start), str(date_end)))

    # --- Spatial handling ---
    num_hrus = ds.sizes.get("hru", 1)
    num_grus = ds.sizes.get("gru", 1)
    spatial_dim = _get_spatial_dim(var_data)

    if spatial_dim is None or (num_hrus == 1 and num_grus == 1):
        # Single HRU/GRU: simple extraction
        result_df = _extract_single_hru(var_data, variable, spatial_dim)
    elif site_mapping is not None:
        # User provided explicit HRU mapping
        result_df = _extract_mapped_sites(var_data, variable, site_mapping, spatial_dim)
    elif obs_metadata_df is not None and attributes_file is not None:
        # Match observation sites to nearest HRUs
        result_df = _extract_nearest_hru(
            var_data, variable, obs_metadata_df, attributes_file, spatial_dim
        )
    else:
        # Multi-HRU: area-weighted aggregation
        result_df = _extract_aggregated(
            var_data, variable, attributes_file, spatial_dim
        )

    ds.close()

    # --- Resample to daily if needed ---
    if temporal_resolution == "daily" and _is_subdaily(result_df):
        date_col = result_df["date"]
        data_cols = result_df.drop(columns=["date"])
        data_cols.index = pd.to_datetime(date_col)
        data_cols = data_cols.resample("D").mean()
        result_df = data_cols.reset_index(names="date")

    if write_csv and csv_path is not None:
        result_df.to_csv(csv_path, index=False)

    return result_df


def get_summa_streamflow(
    summa_output_dir,
    date_start,
    date_end,
    catchment_area=None,
    attributes_file=None,
    temporal_resolution="hourly",
    file_pattern=None,
    output_units="cms",
    write_csv=False,
    csv_path=None,
):
    """
    Extract SUMMA streamflow and convert to volumetric discharge (m³/s).

    SUMMA outputs runoff as depth per unit time (m/s) which must be
    multiplied by the catchment area to get volumetric discharge.

    Parameters
    ----------
    summa_output_dir : str or Path
        Directory containing SUMMA output files.
    date_start, date_end : str or datetime-like
        Date range for extraction.
    catchment_area : float, optional
        Catchment area in m². If None, attempts to read from attributes.nc.
    attributes_file : str or Path, optional
        Path to SUMMA attributes.nc for catchment area lookup.
    temporal_resolution : str, default 'hourly'
        'hourly' or 'daily'.
    file_pattern : str, optional
        Custom glob pattern for the output file.
    output_units : str, default 'cms'
        Output units: 'cms' (m³/s) or 'depth' (m/s, no area scaling).
    write_csv : bool, default False
        Whether to write output to CSV.
    csv_path : str, optional
        Path for CSV output.

    Returns
    -------
    DataFrame
        DataFrame with 'date' column and streamflow column(s).
    """
    from pathlib import Path

    summa_output_dir = Path(summa_output_dir)

    # Get catchment area
    if catchment_area is None and output_units == "cms":
        catchment_area = _get_catchment_area(summa_output_dir, attributes_file)

    # Extract raw streamflow
    result_df = get_summa_output(
        summa_output_dir=summa_output_dir,
        variable="streamflow",
        date_start=date_start,
        date_end=date_end,
        temporal_resolution=temporal_resolution,
        attributes_file=attributes_file,
        file_pattern=file_pattern,
    )

    # Apply unit conversion
    if output_units == "cms" and catchment_area is not None:
        data_cols = [c for c in result_df.columns if c != "date"]
        for col in data_cols:
            values = result_df[col].values
            # Check if values are mass flux (kg m⁻² s⁻¹) vs depth flux (m s⁻¹)
            mean_val = np.nanmean(np.abs(values))
            if mean_val > 1e-6:
                # Likely mass flux, convert to depth first
                values = values / 1000.0
            result_df[col] = values * catchment_area

    # Resample to daily if requested
    if temporal_resolution == "daily" and _is_subdaily(result_df):
        date_col = result_df["date"]
        data_cols = result_df.drop(columns=["date"])
        data_cols.index = pd.to_datetime(date_col)
        data_cols = data_cols.resample("D").mean()
        result_df = data_cols.reset_index(names="date")

    if write_csv and csv_path is not None:
        result_df.to_csv(csv_path, index=False)

    return result_df


# mizuRoute routed streamflow variable names, tried in priority order
MIZUROUTE_STREAMFLOW_VARS = [
    "IRFroutedRunoff",
    "KWTroutedRunoff",
    "dlayRunoff",
]


def get_mizuroute_streamflow(
    mizuroute_output_dir,
    date_start,
    date_end,
    segment_id=None,
    latitude=None,
    longitude=None,
    river_network_shp=None,
    topology_file=None,
    temporal_resolution="hourly",
    file_pattern=None,
    write_csv=False,
    csv_path=None,
):
    """
    Extract routed streamflow from mizuRoute output at a specific river segment.

    mizuRoute produces streamflow already in volumetric units (m³/s) at each
    river segment. This function selects a segment by ID, by lat/lon proximity
    to the river network, or defaults to the outlet (segment with highest
    mean discharge).

    Parameters
    ----------
    mizuroute_output_dir : str or Path
        Directory containing mizuRoute output NetCDF file(s).
    date_start, date_end : str or datetime-like
        Date range for extraction.
    segment_id : int, optional
        Specific river segment (reach) ID to extract. This corresponds to
        the ``reachID`` values in the mizuRoute output or ``segId`` in the
        topology file.
    latitude, longitude : float, optional
        Lat/lon coordinates of a point of interest (e.g., a stream gauge).
        Used to find the nearest river segment. Requires either
        ``river_network_shp`` or ``topology_file`` for spatial lookup.
    river_network_shp : str or Path, optional
        Path to a river network shapefile. Segments are matched by
        nearest-geometry distance to the (longitude, latitude) point.
    topology_file : str or Path, optional
        Path to mizuRoute topology NetCDF file. Used as fallback for
        segment lookup when no shapefile is provided. The file must
        contain ``segId`` and segment-level coordinates or be paired
        with SUMMA attributes for GRU centroids.
    temporal_resolution : str, default 'hourly'
        'hourly' or 'daily'. If 'daily', hourly output is resampled
        to daily means.
    file_pattern : str, optional
        Glob pattern to locate the mizuRoute output file. If None,
        searches for ``*.h.*.nc`` and ``*.nc`` patterns.
    write_csv : bool, default False
        Whether to write the output DataFrame to CSV.
    csv_path : str, optional
        Path for the CSV output file.

    Returns
    -------
    DataFrame
        DataFrame with a 'date' column and a 'streamflow' column (m³/s).
        Compatible with ``calculate_metrics()`` in evaluation_utils.
    """
    from pathlib import Path

    mizuroute_output_dir = Path(mizuroute_output_dir)

    # --- Locate the output file ---
    output_file = _find_mizuroute_file(mizuroute_output_dir, file_pattern)

    # --- Read the dataset ---
    ds = _open_netcdf(output_file)

    # Find the streamflow variable
    var_name = None
    for candidate in MIZUROUTE_STREAMFLOW_VARS:
        if candidate in ds.data_vars:
            var_name = candidate
            break
    if var_name is None:
        ds.close()
        raise ValueError(
            f"No routed streamflow variable found in {output_file}. "
            f"Tried: {MIZUROUTE_STREAMFLOW_VARS}. "
            f"Available: {list(ds.data_vars)}"
        )

    var_data = ds[var_name]

    # --- Time subsetting ---
    var_data = var_data.sel(time=slice(str(date_start), str(date_end)))

    # --- Determine the spatial dimension ---
    if "seg" in var_data.dims:
        spatial_dim = "seg"
    elif "reachID" in var_data.dims:
        spatial_dim = "reachID"
    else:
        # Scalar (single-segment output)
        series = var_data.to_pandas()
        result_df = pd.DataFrame({"date": series.index, "streamflow": series.values})
        ds.close()
        return _resample_if_needed(result_df, temporal_resolution, write_csv, csv_path)

    # --- Get reach IDs from the dataset ---
    if "reachID" in ds.data_vars:
        reach_ids = ds["reachID"].values
    elif "reachID" in ds.coords:
        reach_ids = ds["reachID"].values
    else:
        reach_ids = np.arange(ds.sizes[spatial_dim])

    # --- Select the target segment ---
    seg_idx = _resolve_segment_index(
        segment_id=segment_id,
        latitude=latitude,
        longitude=longitude,
        river_network_shp=river_network_shp,
        topology_file=topology_file,
        reach_ids=reach_ids,
        var_data=var_data,
        spatial_dim=spatial_dim,
    )

    selected_reach_id = reach_ids[seg_idx] if segment_id is None else segment_id

    # Extract the time series at the selected segment
    series = var_data.isel({spatial_dim: seg_idx}).to_pandas()
    ds.close()

    result_df = pd.DataFrame({"date": series.index, "streamflow": series.values})

    return _resample_if_needed(result_df, temporal_resolution, write_csv, csv_path)


# ---------------------------------------------------------------------------
# Internal helper functions
# ---------------------------------------------------------------------------


def _open_netcdf(filepath):
    """Open a NetCDF file, trying multiple engines to handle HDF5 version mismatches."""
    for engine in [None, "h5netcdf", "scipy"]:
        try:
            return xr.open_dataset(filepath, engine=engine)
        except Exception:
            continue
    raise OSError(f"Could not open {filepath} with any available NetCDF engine")


def _find_output_file(summa_output_dir, variable, temporal_resolution, file_pattern):
    """Locate the SUMMA output NetCDF file."""
    from pathlib import Path

    summa_output_dir = Path(summa_output_dir)

    if file_pattern is not None:
        matches = sorted(summa_output_dir.glob(file_pattern))
        if matches:
            return matches[0]
        raise FileNotFoundError(
            f"No files matching '{file_pattern}' in {summa_output_dir}"
        )

    # Streamflow (averageRoutedRunoff) is typically only in timestep files,
    # so always prefer timestep files for streamflow regardless of resolution.
    if variable == "streamflow":
        preferred = ["*_timestep.nc"]
        fallback = ["*_day.nc"]
    elif temporal_resolution == "hourly":
        preferred = ["*_timestep.nc"]
        fallback = ["*_day.nc"]
    else:
        preferred = ["*_day.nc"]
        fallback = ["*_timestep.nc"]

    for pattern in preferred + fallback:
        matches = sorted(summa_output_dir.glob(pattern))
        if matches:
            return matches[0]

    # Last resort: any .nc file
    matches = sorted(summa_output_dir.glob("*.nc"))
    nc_files = [f for f in matches if "restart" not in f.name.lower()]
    if nc_files:
        return nc_files[0]

    raise FileNotFoundError(
        f"No SUMMA output files found in {summa_output_dir}"
    )


def _find_variable_in_dataset(ds, variable):
    """Find the matching SUMMA variable name in a dataset."""
    var_names = SUMMA_VARIABLE_NAMES[variable]
    for name in var_names:
        if name in ds.data_vars:
            return name

    raise ValueError(
        f"Could not find variable for '{variable}' in dataset. "
        f"Tried: {var_names}. Available: {list(ds.data_vars)}"
    )


def _get_spatial_dim(var_data):
    """Determine the spatial dimension name ('hru' or 'gru')."""
    for dim in ["hru", "gru"]:
        if dim in var_data.dims:
            return dim
    return None


def _extract_single_hru(var_data, variable, spatial_dim):
    """Extract data from a single-HRU domain."""
    if spatial_dim is not None:
        series = var_data.isel({spatial_dim: 0}).to_pandas()
    else:
        series = var_data.to_pandas()

    df = pd.DataFrame({"date": series.index, variable: series.values})
    return df


def _extract_mapped_sites(var_data, variable, site_mapping, spatial_dim):
    """Extract data for specific HRUs using a site_id -> HRU index mapping."""
    records = {"date": var_data.time.values}
    for site_id, hru_idx in site_mapping.items():
        records[site_id] = var_data.isel({spatial_dim: hru_idx}).values
    return pd.DataFrame(records)


def _extract_nearest_hru(var_data, variable, obs_metadata_df, attributes_file, spatial_dim):
    """Match observation sites to nearest HRUs using lat/lon."""
    attrs = xr.open_dataset(attributes_file)

    hru_lats = attrs["latitude"].values
    hru_lons = attrs["longitude"].values
    attrs.close()

    records = {"date": var_data.time.values}
    for _, site in obs_metadata_df.iterrows():
        site_lat = site["latitude"]
        site_lon = site["longitude"]
        # Find nearest HRU by Euclidean distance in lat/lon space
        distances = np.sqrt(
            (hru_lats - site_lat) ** 2 + (hru_lons - site_lon) ** 2
        )
        nearest_idx = int(np.argmin(distances))
        records[site["site_id"]] = var_data.isel({spatial_dim: nearest_idx}).values

    return pd.DataFrame(records)


def _extract_aggregated(var_data, variable, attributes_file, spatial_dim):
    """Area-weighted aggregation across all HRUs."""
    if attributes_file is not None:
        from pathlib import Path

        attributes_file = Path(attributes_file)
        if attributes_file.exists():
            attrs = xr.open_dataset(attributes_file)

            area_var = "HRUarea" if spatial_dim == "hru" else "GRUarea"
            if area_var in attrs:
                areas = attrs[area_var]
                if areas.sizes.get(spatial_dim, 0) == var_data.sizes.get(spatial_dim, 0):
                    total_area = float(areas.sum())
                    weights = areas / total_area
                    aggregated = (var_data * weights).sum(dim=spatial_dim)
                    attrs.close()
                    series = aggregated.to_pandas()
                    return pd.DataFrame(
                        {"date": series.index, variable: series.values}
                    )
            attrs.close()

    # Fallback: simple mean across spatial dimension
    aggregated = var_data.mean(dim=spatial_dim)
    series = aggregated.to_pandas()
    return pd.DataFrame({"date": series.index, variable: series.values})


def _get_catchment_area(summa_output_dir, attributes_file):
    """Resolve catchment area from attributes.nc."""
    from pathlib import Path

    if attributes_file is not None:
        attrs_path = Path(attributes_file)
    else:
        # Try standard SUMMA project layout
        attrs_path = summa_output_dir.parent / "settings" / "SUMMA" / "attributes.nc"
        if not attrs_path.exists():
            # Try two levels up (common in SYMFLUENCE layout)
            for parent in summa_output_dir.parents:
                candidate = parent / "settings" / "SUMMA" / "attributes.nc"
                if candidate.exists():
                    attrs_path = candidate
                    break

    if attrs_path.exists():
        attrs = xr.open_dataset(attrs_path)
        if "HRUarea" in attrs:
            area = float(attrs["HRUarea"].sum())
            attrs.close()
            return area
        attrs.close()

    warnings.warn(
        "Could not determine catchment area. Streamflow will remain in depth units (m/s). "
        "Provide catchment_area or attributes_file to convert to m³/s.",
        stacklevel=2,
    )
    return None


def _is_subdaily(df):
    """Check if DataFrame has sub-daily time resolution."""
    dates = pd.to_datetime(df["date"])
    if len(dates) < 2:
        return False
    diff = dates.iloc[1] - dates.iloc[0]
    return diff < pd.Timedelta(days=1)


def _resample_if_needed(result_df, temporal_resolution, write_csv, csv_path):
    """Resample to daily if needed and optionally write CSV."""
    if temporal_resolution == "daily" and _is_subdaily(result_df):
        date_col = result_df["date"]
        data_cols = result_df.drop(columns=["date"])
        data_cols.index = pd.to_datetime(date_col)
        data_cols = data_cols.resample("D").mean()
        result_df = data_cols.reset_index(names="date")

    if write_csv and csv_path is not None:
        result_df.to_csv(csv_path, index=False)

    return result_df


def _find_mizuroute_file(mizuroute_output_dir, file_pattern):
    """Locate the mizuRoute output NetCDF file."""
    from pathlib import Path

    mizuroute_output_dir = Path(mizuroute_output_dir)

    if file_pattern is not None:
        matches = sorted(mizuroute_output_dir.glob(file_pattern))
        if matches:
            return matches[0]
        raise FileNotFoundError(
            f"No files matching '{file_pattern}' in {mizuroute_output_dir}"
        )

    # mizuRoute standard patterns
    for pattern in ["*.h.*.nc", "*_routed.nc", "*.nc"]:
        matches = sorted(mizuroute_output_dir.glob(pattern))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"No mizuRoute output files found in {mizuroute_output_dir}"
    )


def _resolve_segment_index(
    segment_id,
    latitude,
    longitude,
    river_network_shp,
    topology_file,
    reach_ids,
    var_data,
    spatial_dim,
):
    """Resolve which segment index to extract from.

    Priority:
    1. Explicit segment_id
    2. Lat/lon with river network shapefile (nearest geometry)
    3. Lat/lon with topology file (nearest segment centroid)
    4. Outlet detection (segment with highest mean discharge)
    """
    # 1. Explicit segment ID
    if segment_id is not None:
        matching = np.where(reach_ids == segment_id)[0]
        if len(matching) == 0:
            raise ValueError(
                f"segment_id={segment_id} not found in reach IDs: {reach_ids}"
            )
        return int(matching[0])

    # 2. Lat/lon with shapefile
    if latitude is not None and longitude is not None and river_network_shp is not None:
        return _find_nearest_segment_shp(
            latitude, longitude, river_network_shp, reach_ids
        )

    # 3. Lat/lon with topology file
    if latitude is not None and longitude is not None and topology_file is not None:
        return _find_nearest_segment_topology(
            latitude, longitude, topology_file, reach_ids
        )

    # 4. Outlet: segment with highest mean discharge
    segment_means = var_data.mean(dim="time").values
    outlet_idx = int(np.argmax(segment_means))
    return outlet_idx


def _find_nearest_segment_shp(latitude, longitude, river_network_shp, reach_ids):
    """Find nearest river segment using shapefile geometry."""
    import geopandas as gpd
    from shapely.geometry import Point

    shp = gpd.read_file(river_network_shp)

    # Project to UTM for accurate distance calculation
    utm_crs = shp.estimate_utm_crs()
    shp_proj = shp.to_crs(utm_crs)

    point = gpd.GeoSeries([Point(longitude, latitude)], crs="EPSG:4326")
    point_proj = point.to_crs(utm_crs)

    distances = shp_proj.geometry.distance(point_proj.iloc[0])
    closest_shp_idx = int(distances.idxmin())

    # Map shapefile row back to reach_ids index
    # Try common column names for segment ID
    seg_id_col = None
    for col in ["LINKNO", "segId", "seg_id", "COMID", "reach_id", "reachID"]:
        if col in shp.columns:
            seg_id_col = col
            break

    if seg_id_col is not None:
        closest_seg_id = int(shp.loc[closest_shp_idx, seg_id_col])
        matching = np.where(reach_ids == closest_seg_id)[0]
        if len(matching) > 0:
            return int(matching[0])

    # Fallback: use positional index if shapefile order matches
    return closest_shp_idx


def _find_nearest_segment_topology(latitude, longitude, topology_file, reach_ids):
    """Find nearest segment using topology file coordinates.

    The topology file typically contains segment IDs (``segId``) and HRU
    coordinates.  We use the HRU-to-segment mapping and HRU centroids
    to find the segment nearest to the query point.  If the topology
    also has an accompanying SUMMA attributes file with lat/lon per HRU,
    those coordinates are used.
    """
    from pathlib import Path

    topo = xr.open_dataset(topology_file)

    seg_ids = topo["segId"].values if "segId" in topo else reach_ids

    # Try to find segment centroids via HRU mapping
    if "hruToSegId" in topo and "hruId" in topo:
        hru_to_seg = topo["hruToSegId"].values

        # Look for SUMMA attributes.nc alongside topology for lat/lon
        attrs_path = Path(topology_file).parent.parent / "SUMMA" / "attributes.nc"
        if attrs_path.exists():
            attrs = xr.open_dataset(attrs_path)
            hru_lats = attrs["latitude"].values
            hru_lons = attrs["longitude"].values
            attrs.close()

            # Compute area-weighted centroid per segment
            seg_lats = {}
            seg_lons = {}
            for i, seg in enumerate(hru_to_seg):
                seg_lats.setdefault(seg, []).append(hru_lats[i])
                seg_lons.setdefault(seg, []).append(hru_lons[i])

            best_dist = float("inf")
            best_idx = 0
            for seg in seg_ids:
                if seg in seg_lats:
                    clat = np.mean(seg_lats[seg])
                    clon = np.mean(seg_lons[seg])
                    dist = (clat - latitude) ** 2 + (clon - longitude) ** 2
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = int(np.where(reach_ids == seg)[0][0])

            topo.close()
            return best_idx

    topo.close()

    # Last fallback: use reach_ids order and position
    warnings.warn(
        "Could not determine segment coordinates. Selecting outlet segment.",
        stacklevel=2,
    )
    return 0
