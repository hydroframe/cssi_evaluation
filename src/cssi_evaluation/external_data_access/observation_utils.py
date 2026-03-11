"""
External data access functions for observations, specifically for observations not currently available in HydroData.
"""
### LOCATION OF ORIGINAL FUNCTIONS
# nwm_utils.getSNOTELData()
# nwm_utils.getCCSSData()



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
