import requests
import os
from zipfile import ZipFile
from io import BytesIO
import shapefile

def fetch_huc8_shapefile(
    huc_2_code,
    huc_8_code,
    out_dir="domain_data",
    verbose=False
):
    """
    Download WBD HUC2 data, extract HUC8 shapefile,
    select a specific HUC8, and save it locally.
    This code is adapted directly from the HydroData readthedocs.io documentation:
    https://hf-hydrodata.readthedocs.io/en/latest/point_data/examples/example_shapefile.html

    Returns
    -------
    huc8_shape : shapefile.Shape
    huc8_record : dict
    huc8_geo : dict (geo-interface)
    crs_wkt : str
    """

    # Download HUC2 WBD zip
    url = (
        "https://prd-tnm.s3.amazonaws.com/"
        f"StagedProducts/Hydrography/WBD/HU2/Shape/WBD_{huc_2_code}_HU2_Shape.zip"
    )
    response = requests.get(url)
    response.raise_for_status()

    # Open zip in memory
    myzipfile = ZipFile(BytesIO(response.content))

    # Extract only HUC8-level files
    members = [
        "Shape/WBDHU8.shp",
        "Shape/WBDHU8.shx",
        "Shape/WBDHU8.dbf",
        "Shape/WBDHU8.prj",
    ]
    myzipfile.extractall(members=members)

    # Read shapefile + CRS
    huc2_shp = shapefile.Reader("Shape/WBDHU8.shp")

    with open("Shape/WBDHU8.prj") as f:
        crs_wkt = f.read().strip()

    if verbose:
        print(f"CRS: {crs_wkt}")

    # Find requested HUC8
    huc8_shape = None
    huc8_record = None

    for sr in huc2_shp.shapeRecords():
        record = sr.record.as_dict()
        if record["huc8"] == huc_8_code:
            huc8_shape = sr.shape
            huc8_record = record

            if verbose:
                print(f'Selected HUC8: {record["states"]}, {record["huc8"]}, {record["name"]}')
            break

    if huc8_shape is None:
        raise ValueError(f"HUC8 {huc_8_code} not found in HUC2 {huc_2_code}")

    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Save selected HUC8 shapefile
    out_name = f'{out_dir}/{huc8_record["name"]}_{huc8_record["huc8"]}'
    with shapefile.Writer(out_name) as w:
        w.fields = huc2_shp.fields[1:]
        w.record(huc8_record)
        w.shape(huc8_shape)

    # Write projection file
    prj_path = f"{out_name}.prj"
    with open(prj_path, "w") as f:
        f.write(crs_wkt)

    # Get geo-interface, convert the pyshp geometry into a standard GeoJSON-style geometry dictionary
    huc8_geo = huc8_shape.__geo_interface__

    return huc8_shape, huc8_record, huc8_geo, crs_wkt

