import os
import sys
import pytz
import time
import urllib3
import datetime
import numpy as np
import pandas as pd
import pyproj
import folium
import hvplot.pandas
import holoviews as hv
import hvplot.xarray
from holoviews import opts
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xyzservices.providers as xyz
from scipy.stats import pearsonr, spearmanr
pd.options.mode.chained_assignment = None

import geoviews as gv
import geoviews.tile_sources as gts
gv.extension('bokeh')

def getSNOTELData(SiteName, SiteID, StateAbb, StartDate, EndDate, OutputFolder):
	url1 = 'https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/customMultiTimeSeriesGroupByStationReport/daily/start_of_period/'
	url2 = f'{SiteID}:{StateAbb}:SNTL%7Cid=%22%22%7Cname/'
	url3 = f'{StartDate},{EndDate}/'
	url4 = 'WTEQ::value?fitToScreen=false'
	url = url1+url2+url3+url4
    
	dl_start_time = time.time()
	
	http = urllib3.PoolManager()
	response = http.request('GET', url)
	data = response.data.decode('utf-8')
	i=0
	for line in data.split("\n"):
		if line.startswith("#"):
			i=i+1
	data = data.split("\n")[i:]
	
	df = pd.DataFrame.from_dict(data)
	df = df[0].str.split(',', expand=True)
	df.rename(columns={0:df[0][0], 
					   1:df[1][0]}, inplace=True)
	df.drop(0, inplace=True)
	df.dropna(inplace=True)
	df.reset_index(inplace=True, drop=True)
	df["Date"] = pd.to_datetime(df["Date"])
	df.rename(columns={df.columns[1]:'Snow Water Equivalent (m) Start of Day Values'}, inplace=True)
	df.iloc[:, 1:] = df.iloc[:, 1:].apply(lambda x: pd.to_numeric(x) * 0.0254)  # convert in to m
	df['Water_Year'] = pd.to_datetime(df['Date']).map(lambda x: x.year+1 if x.month>9 else x.year)
	
	df.to_csv(f'./{OutputFolder}/df_{SiteID}_{StateAbb}_SNTL.csv', index=False)

	dl_elapsed = time.time() - dl_start_time
	print(f'✅ Retrieved data for {SiteName}, {SiteID} in {dl_elapsed:.2f} seconds\n')

def getCCSSData(SiteName, SiteID, StartDate, EndDate, OutputFolder):
    StateAbb = 'Ca'
    url1 = 'https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/customSingleStationReport/daily/start_of_period/' 
    url2 = f'{SiteID}:CA:MSNT%257Cid=%2522%2522%257Cname/'
    url3 = f'{StartDate},{EndDate}/'
    url4 = 'WTEQ::value?fitToScreen=false'
    url = url1+url2+url3+url4

    dl_start_time = time.time()

    http = urllib3.PoolManager()
    response = http.request('GET', url)
    data = response.data.decode('utf-8')
    i=0
    for line in data.split("\n"):
        if line.startswith("#"):
            i=i+1
    data = data.split("\n")[i:]

    df = pd.DataFrame.from_dict(data)
    print(df.columns)
    df = df[0].str.split(',', expand=True)
    df.rename(columns={0:df[0][0], 
                        1:df[1][0]}, inplace=True)
    df.drop(0, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(inplace=True, drop=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df.rename(columns={df.columns[1]:'Snow Water Equivalent (m) Start of Day Values'}, inplace=True)
    df.iloc[:, 1:] = df.iloc[:, 1:].apply(lambda x: pd.to_numeric(x) * 0.0254)  # convert in to m
    df['Water_Year'] = pd.to_datetime(df['Date']).map(lambda x: x.year+1 if x.month>9 else x.year)

    df.to_csv(f'./{OutputFolder}/df_{SiteID}_{StateAbb}_CCSS.csv', index=False)

    dl_elapsed = time.time() - dl_start_time
    print(f'✅ Retrieved data for {SiteName}, {SiteID} in {dl_elapsed:.2f} seconds\n')

def convert_latlon_to_yx(lat, lon, input_crs, ds, output_crs):
    """
    This function takes latitude and longitude values along with
    input and output coordinate reference system (CRS) and 
    uses Python's pyproj package to convert the provided values 
    (as single float values, not arrays) to the corresponding y and x 
    coordinates in the output CRS.
    
    Parameters:
    lat: The latitude value
    lon: The longitude value
    input_crs: The input coordinate reference system ('EPSG:4326')
    output_crs: The output coordinate reference system
    
    Returns:
    y, x: a tuple of the transformed coordinates in the specified output
    """
    # Create a transformer
    transformer = pyproj.Transformer.from_crs(input_crs, output_crs, always_xy=True)

    # Perform the transformation
    x, y = transformer.transform(lon, lat)

    return y, x 

def convert_utc_to_local(state, df):
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

def report_max_dates_and_values(df, col_obs, col_mod):
    # Ensure the index is datetime
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        raise ValueError("DataFrame index must be datetime")

    # Find max values and associated dates
    max_obs = df[col_obs].max()
    date_obs = df[col_obs].idxmax()

    max_mod = df[col_mod].max()
    date_mod = df[col_mod].idxmax()

    # Create a summary table as a DataFrame (nice for Jupyter display)
    summary_table = pd.DataFrame({
        'Data Source': [col_obs, col_mod],
        'Peak SWE (mm)': [max_obs, max_mod],
        'Date of Maximum': [date_obs.date(), date_mod.date()]
    })

    return summary_table

def compute_melt_period(swe_series, min_zero_days=10):
    
    # Find peak date and maximum SWE
    peak_date = swe_series.idxmax()
    peak_swe = swe_series.max()

    # Subset data to only include days after peak date
    after_peak = swe_series.loc[peak_date:]

    # Find first date where SWE becomes zero and stays zero for at least `min_zero_days`
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
        raise ValueError("Could not find a period of at least 10 consecutive zero SWE days after the peak.")

    # Calculate melt period length (days between peak and melt completion)
    melt_period_days = (melt_end_date - peak_date).days

    # Calculate melt rate (m/day)
    melt_rate = peak_swe / melt_period_days

    # Return results in a dictionary
    return {
        'peak_date': peak_date,
        'peak_swe_m': peak_swe,
        'melt_end_date': melt_end_date,
        'melt_period_days': melt_period_days,
        'melt_rate_m/d': melt_rate
    }

def prep_nwm_swe_dataframe(ds, state):
    df = ds.to_dataframe()
    df.drop(columns=['crs'], inplace=True)
    df.reset_index(inplace=True)
    df["time"] = pd.to_datetime(df["time"])
    df.rename(columns={df.columns[0]:'Date', df.columns[1]:'NWM_SWE_meters'}, inplace=True)
    df.iloc[:, 1:] = df.iloc[:, 1:].apply(lambda x: pd.to_numeric(x)/1000)  # convert mm to m   
    df_local = convert_utc_to_local(state, df)   
    df_local.index = pd.to_datetime(df_local['Date_Local'])
    df_local = df_local.groupby(pd.Grouper(freq='D')).first()
    df_local = df_local.reset_index(drop=True)  
    #df_local.to_csv(output_path, index=False)

    return df_local

def compute_spatial_agg_from_obs(folder_path, agg):
    # List all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    if len(csv_files) == 0:
        raise ValueError("No CSV files found in the specified folder.")

    # Read all files into a list of DataFrames
    dfs = []
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path, parse_dates=['Date'])
        dfs.append(df)

    # Concatenate all files into a single DataFrame
    combined_df = pd.concat(dfs)

    # Group by Date and Water_Year, compute mean SWE
    averaged_df = combined_df.groupby(['Date', 'Water_Year'], as_index=False).agg({
        'Snow Water Equivalent (m) Start of Day Values': agg
    })

    # Save to output CSV
    # averaged_df.to_csv(output_file, index=False)
    return averaged_df

    print(f"Averaged CSV saved to: {output_file}")
    


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
    import folium
    import geopandas as gpd

    # Ensure CRS compatibility
    if gdf_sites.crs != domain_gdf.crs:
        gdf_sites = gdf_sites.to_crs(domain_gdf.crs)

    # Helper to find appropriate column names
    def find_column(columns, candidates):
        return next((c for c in candidates if c in columns), None)

    # Candidate column names (ordered by preference)
    site_name_col = find_column(
        gdf_sites.columns,
        ['site_name', 'station_name', 'name', 'Site Name']
    )
    site_id_col = find_column(
        gdf_sites.columns,
        ['site_id', 'station_id', 'site_code', 'code', 'Site Code']
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
            icon=folium.Icon(color="green", icon="info-sign")
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


def compute_stats(df, ts1, ts2):
    df = df[[f'{ts1}', f'{ts2}']]
    df.dropna(inplace=True)  # Both Pearson and Spearman correlations cannot handle NaN values, so make sure to drop nan values before any calculatoin.
    
    # Compute statistics for each time series
    stats = {
        'Mean [L]': [df[f'{ts1}'].mean(), df[f'{ts2}'].mean()],
        'Median [L]': [df[f'{ts1}'].median(), df[f'{ts2}'].median()],
        'Standard Deviation [L]': [df[f'{ts1}'].std(), df[f'{ts2}'].std()],
        'Variance [L^2]': [df[f'{ts1}'].var(), df[f'{ts2}'].var()],
        'Min [L]': [df[f'{ts1}'].min(), df[f'{ts2}'].min()],
        'Max [L]': [df[f'{ts1}'].max(), df[f'{ts2}'].max()]
    }

    # Calculate correlation coefficients
    pearson_corr, _ = pearsonr(df[f'{ts1}'], df[f'{ts2}'])
    spearman_corr, _ = spearmanr(df[f'{ts1}'], df[f'{ts2}'])

    # Compute Mean Bias (mean error)
    bias = df[ts2].mean() - df[ts1].mean()

    # Compute Nash-Sutcliffe Efficiency (NSE)
    obs_mean = df[ts1].mean()
    numerator = np.sum((df[ts2] - df[ts1])**2)
    denominator = np.sum((df[ts1] - obs_mean)**2)
    nse = 1 - (numerator / denominator)

    # Compute Kling-Gupta Efficiency (KGE)
    r = pearson_corr
    alpha = df[ts2].std() / df[ts1].std()
    beta = df[ts2].mean() / df[ts1].mean()
    kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

    # Create a DataFrame for the statistics
    stats_table = pd.DataFrame(stats, index=['observed', 'modeled'])

    # Add Pearson and Spearman correlations as additional rows
    stats_table.loc[''] = [''] * len(stats_table.columns)  # Blank row for formatting
    stats_table.loc['Pearson Correlation [-]'] = [pearson_corr, '', '', '', '', '']
    stats_table.loc['Spearman Correlation [-]'] = [spearman_corr, '', '', '', '', '']
    stats_table.loc['Bias (Modeled - Observed) [L]'] = [bias, '', '', '', '', '']
    stats_table.loc['Nash-Sutcliffe Efficiency (NSE) [-]'] = [nse, '', '', '', '', '']
    stats_table.loc['Kling-Gupta Efficiency (KGE) [-]'] = [kge, '', '', '', '', '']

    return stats_table

def compute_stats_period(
    df, ts_obs, ts_mod, months
):
    df_sub = df[df.index.month.isin(months)]
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

    return compute_stats(df_sub, ts_obs, ts_mod)


def comparison_plots(df, ts_obs, ts_mod):
    '''
    Create a set of comparison plots (timeseries overlay and scatter plot with 1:1 line)

    Parameters:
    df: dataframe with combined observed and modeled timeseries for each site  
    ts_obs (str): column heading for observed timeseries in df
    ts_mod (str): column heading for modeled timeseries in df
    '''

    df = df.copy()
    df.index.name = "date" # change the index name to "Date" for better hover tooltip display

    # Timeseries plot (Overlay)
    observed_plot = df.hvplot.line(
        y=ts_obs,
        ylabel='Snow Water Equivalent (mm)',
        xlabel='',
        label='Observed SWE',
        color='blue',
        line_width=2,
        width=500,
        height=400,
    )

    modeled_plot = df.hvplot.line(
    y=ts_mod,
    ylabel='Snow Water Equivalent (mm)',
    xlabel='',
    label='Modeled SWE',
    color='orchid',
    line_width=2,
    width=500,
    height=400,
    )

    # Overlay (combines both lines into a single visual object)
    timeseries_plot = (observed_plot * modeled_plot).opts(
        title='Observed vs Modeled SWE\nDaily Time Series',
        legend_position='top_right',
    )

    # Scatter plot
    scatter_plot = df.hvplot.scatter(
        x=ts_obs,
        y=ts_mod,
        xlabel='Observed SWE (mm)',
        ylabel='Modeled SWE (mm)',
        color='black',
        width=500,
        height=400,
        size=15,
        hover_cols=['date']  # This will add the date (index) to hover tooltip
    )

    # Add 1:1 line (perfect match line)
    swe_max = max(df[ts_obs].max(), df[ts_mod].max())
    one_to_one_line = hv.Curve(([0, swe_max], [0, swe_max])).opts(
        color='gray',
        line_dash='solid',
        line_width=1,
    ).relabel('1:1 Line')  # This is the correct way to set a label for a Curve
    
    # Combine scatter plot and 1:1 line into an Overlay
    scatter_with_line = (scatter_plot * one_to_one_line).opts(
        title='Observed vs Modeled SWE\nScatter with 1:1 Line',
        legend_position='bottom_right'
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
        xlabel='Observed SWE (mm)',
        ylabel='Modeled SWE (mm)',
        title='Observed vs. Modeled SWE at ' + obs_col,
        size=15,
        width=500,
        height=400,
        hover_cols=['index', 'month'],
        color='color'
    )

    # 1:1 line
    swe_max = max(df[obs_col].max(), df[mod_col].max())
    one_to_one = hv.Curve(([0, swe_max], [0, swe_max])).opts(
        color="gray",
        line_dash="dashed",
        line_width=1,
    ).relabel("1:1 Line")

    return (scatter * one_to_one).opts(legend_position="bottom_right")


def plot_grid_vector_data(ds_clip, data_var, time_index, shp, sites):
    hv.extension('bokeh')
    da = ds_clip[data_var]

    # Select one timestep
    if isinstance(time_index, int):
        da = da.isel(time=time_index)
    else:
        da = da.sel(time=time_index)

    # Create an interactive map plot
    clipped = da.rio.reproject("EPSG:4326")
    clipped = clipped.rename({'x': 'longitude', 'y': 'latitude'})
    hvplot_map = clipped.hvplot(
        x='longitude',
        y='latitude', 
        geo=True,
        project=True,
        tiles=gts.ESRI,
        cmap='kbc',
        alpha=0.6,
        frame_height=400,
        title=f"Snow Water Equivalent, at {pd.to_datetime(time_index).strftime('%Y-%m-%d %H:%M')}",
        clim=(0, 300)
    )
    
    shp = shp.to_crs("EPSG:4326").reset_index(drop=True)
    sites = sites.to_crs("EPSG:4326").reset_index(drop=True)

    # Plot the shapefile outline
    shp_plot = shp.hvplot(
    geo=True, project=True,
    color='none', line_width=2
    )

    # Plot sites (scatter)
    points_plot = sites.hvplot.points(
    x='longitude', y='latitude',
    geo=True, project=True,
    color='red', size=100, hover_cols=['name']
    )

    # Combine the two by overlaying
    combined_map = (hvplot_map * shp_plot * points_plot).opts(framewise=True)
    
    return combined_map

def plot_grid_vector_monthly_data(ds_clip, data_var, shp, sites):
    hv.extension('bokeh')

    # Create an interactive map plot
    clipped = ds_clip[data_var].rio.reproject("EPSG:4326")
    clipped = clipped.rename({'x': 'longitude', 'y': 'latitude'})
    
    # Plot the shapefile outline
    shp_plot = shp.hvplot(
    geo=True, project=True,
    color='none', line_width=2
    )

    # Plot sites (scatter)
    points_plot = sites.hvplot.points(
    x='longitude', y='latitude',
    geo=True, project=True,
    color='red', size=100, hover_cols=['name']
    )

    # Split into individual plots (list of plots)
    plots = []
    for t in clipped.time.values:
        base_plot = clipped.sel(time=t).hvplot(
            x='longitude', y='latitude',
            geo=True, project=True,
            tiles=gts.ESRI,
            title=f'SWE (mm) on {pd.to_datetime(t).strftime("%Y-%m-%d")}',
            frame_height=200, frame_width=300
        )
        # Overlay shapefile and points on top of SWE map
        combined_plot = base_plot * shp_plot * points_plot
        plots.append(combined_plot)
        
    layout = hv.Layout(plots).cols(3)
    
    return layout