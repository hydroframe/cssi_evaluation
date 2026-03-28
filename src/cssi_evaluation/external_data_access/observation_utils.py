"""
External data access functions for observations, specifically for observations not currently available in HydroData.
"""
### LOCATION OF ORIGINAL FUNCTIONS
# nwm_utils.getSNOTELData()
# nwm_utils.getCCSSData()

import time
import urllib3
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