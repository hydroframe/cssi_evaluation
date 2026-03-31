"""
External data access functions for observations, specifically for observations not currently available in HydroData.
"""
### LOCATION OF ORIGINAL FUNCTIONS
# nwm_utils.getSNOTELData()
# nwm_utils.getCCSSData()

import time
import urllib3
import requests
import pandas as pd

pd.options.mode.chained_assignment = None


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

def getWSCData(station_id, start_date=None, end_date=None, resample="D", output_file=None):
    """
    Download daily streamflow data from a Water Survey of Canada (WSC) station
    via the ECCC GeoMet API.

    Parameters
    ----------
    station_id : str
        WSC station number, e.g. '05BB001' (Bow River at Banff).
    start_date : str or datetime-like, optional
        Start date for subsetting (inclusive). If None, returns all available data.
    end_date : str or datetime-like, optional
        End date for subsetting (inclusive).
    resample : str, default 'D'
        Pandas resample frequency. 'D' for daily, 'h' for hourly (interpolated),
        'W' for weekly, etc. Set to None to skip resampling.
    output_file : str or Path, optional
        If provided, save the resulting DataFrame to this CSV path.

    Returns
    -------
    DataFrame
        DataFrame with a DatetimeIndex named 'datetime' and a 'discharge_cms'
        column (m³/s). Compatible with cssi_evaluation workflows.

    Examples
    --------
    >>> df = getWSCData('05BB001', '2004-01-01', '2009-12-31')
    >>> df.head()
                discharge_cms
    datetime
    2004-01-01           9.84
    2004-01-02           9.71
    ...
    """
    base_url = "https://api.weather.gc.ca/collections/hydrometric-daily-mean/items"
    page_limit = 10000

    dl_start_time = time.time()
    all_rows = []
    offset = 0

    while True:
        params = {
            "STATION_NUMBER": station_id,
            "f": "json",
            "limit": page_limit,
            "offset": offset,
        }

        response = requests.get(base_url, params=params, timeout=60)
        response.raise_for_status()

        data = response.json()
        features = data.get("features", [])

        if not features:
            if offset == 0:
                raise ValueError(
                    f"No data found for WSC station {station_id}. "
                    "Verify the station number at "
                    "https://wateroffice.ec.gc.ca/search/real_time_e.html"
                )
            break

        for feat in features:
            all_rows.append(feat.get("properties", {}))

        if len(features) < page_limit:
            break

        offset += page_limit

    df = pd.DataFrame(all_rows)

    # Identify the date and discharge columns
    date_col = _find_column(df.columns, ["DATE", "date", "datetime"])
    value_col = _find_column(df.columns, ["DISCHARGE", "VALUE", "discharge_cms", "flow"])

    if date_col is None or value_col is None:
        raise ValueError(
            f"Could not identify date/discharge columns in GeoMet response. "
            f"Columns found: {list(df.columns)}"
        )

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[date_col, value_col])

    df = df.set_index(date_col).sort_index()
    df = df[[value_col]].rename(columns={value_col: "discharge_cms"})

    # Subset to date range
    if start_date is not None:
        df = df.loc[str(start_date):]
    if end_date is not None:
        df = df.loc[:str(end_date)]

    # Resample
    if resample is not None:
        df = df.resample(resample).mean()
        df = df.interpolate(method="time", limit_direction="both", limit=30)

    df.index.name = "datetime"

    dl_elapsed = time.time() - dl_start_time
    print(
        f"Retrieved {len(df)} records for WSC station {station_id} "
        f"in {dl_elapsed:.2f} seconds"
    )

    if output_file is not None:
        df.to_csv(output_file)
        print(f"Saved to {output_file}")

    return df


def _find_column(columns, candidates):
    """Find first matching column name (case-insensitive)."""
    col_lower = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in col_lower:
            return col_lower[candidate.lower()]
    return None


def getCCSSData(gdf, StateAbb, StartDate, EndDate, OutputFile):

    url1 = 'https://wcc.sc.egov.usda.gov/reportGenerator/view_csv/customSingleStationReport/daily/start_of_period/'
    url4 = 'WTEQ::value?fitToScreen=false'

    http = urllib3.PoolManager()
    dataframes = []

    for i in range(len(gdf)):
        SiteName = gdf.name.iloc[i]
        SiteID = gdf.code.iloc[i]

        url2 = f'{SiteID}:{StateAbb}:MSNT%257Cid=%2522%2522%257Cname/'
        url3 = f'{StartDate},{EndDate}/'
        url = url1 + url2 + url3 + url4

        dl_start_time = time.time()

        response = http.request('GET', url)
        data = response.data.decode('utf-8')

        # Remove comments
        i_skip = sum(1 for line in data.split("\n") if line.startswith("#"))
        data = data.split("\n")[i_skip:]

        df = pd.DataFrame(data)[0].str.split(',', expand=True)

        df.rename(columns={0: df[0][0], 1: df[1][0]}, inplace=True)
        df = df.drop(0).dropna().reset_index(drop=True)

        df["Date"] = pd.to_datetime(df["Date"])
        df.iloc[:, 1] = pd.to_numeric(df.iloc[:, 1]) * 0.0254  # inches → meters

        # ✅ Keep ONLY SWE column
        df = df[["Date", df.columns[1]]]

        # ✅ Rename column to desired format
        col_name = f"{SiteID}:{StateAbb}:CCSS"
        df.rename(columns={df.columns[1]: col_name}, inplace=True)

        df = df.set_index("Date")

        dataframes.append(df)

        print(f'✅ Retrieved {SiteName} ({SiteID}) in {time.time() - dl_start_time:.2f}s')

    # ✅ Merge (no multi-index!)
    merged_df = pd.concat(dataframes, axis=1)

    # ✅ Reset index so Date becomes a column
    merged_df.reset_index(inplace=True)

    # Optional: sort by date
    merged_df.sort_values("Date", inplace=True)

    merged_df.to_csv(OutputFile, index=False)

    print(f"\n✅ Final format saved: {OutputFile}")

    return merged_df