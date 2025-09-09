"""Test /model_evaluation/model_evaluation.py module."""

# pylint: disable=E0401
import sys
import os
import pytest
import pandas as pd
import subsettools

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from cssi_evaluation import model_evaluation


def test_get_observations():
    """Test get_observations function."""

    # Define inputs to workflow
    grid = "conus2"
    huc_list = ["02040106"]
    ij_bounds, mask = subsettools.define_huc_domain(huc_list, grid)
    start_date = "2003-04-01 01:00:00"
    end_date = "2003-04-03 00:00:00"
    temporal_resolution = "hourly"
    variable = "streamflow"

    obs_metadata_df, obs_data_df = model_evaluation.get_observations(
        mask, ij_bounds, grid, start_date, end_date, variable, temporal_resolution
    )
    assert obs_metadata_df.shape == (14, 32)
    assert obs_data_df.shape == (48, 15)
    assert "01447500" in obs_data_df.columns


def test_get_observations_nan_filter_sites_removed():
    """Test get_observations function remove_sites_no_data parameter set to True (default)."""

    # Define inputs to workflow
    grid = "conus2"
    huc_list = ["01100006"]
    ij_bounds, mask = subsettools.define_huc_domain(huc_list, grid)
    start_date = "2003-11-01"
    end_date = "2003-11-05"
    temporal_resolution = "daily"
    variable = "streamflow"

    # Removing all-NaN observation sites
    obs_metadata_df, obs_data_df = model_evaluation.get_observations(
        mask, ij_bounds, grid, start_date, end_date, variable, temporal_resolution
    )

    assert obs_metadata_df.shape == (7, 32)
    assert obs_data_df.shape == (5, 8)
    assert "01209500" not in obs_data_df.columns


def test_get_observations_nan_filter_sites_included():
    """Test get_observations function remove_sites_no_data parameter set to False."""

    # Define inputs to workflow
    grid = "conus2"
    huc_list = ["01100006"]
    ij_bounds, mask = subsettools.define_huc_domain(huc_list, grid)
    start_date = "2003-11-01"
    end_date = "2003-11-05"
    temporal_resolution = "daily"
    variable = "streamflow"

    # Leaving these sites in
    obs_metadata_df, obs_data_df = model_evaluation.get_observations(
        mask,
        ij_bounds,
        grid,
        start_date,
        end_date,
        variable,
        temporal_resolution,
        remove_sites_no_data=False,
    )

    assert obs_metadata_df.shape == (8, 32)
    assert obs_data_df.shape == (5, 9)
    assert "01209500" in obs_data_df.columns


# This test needs to be refactored to use a local ParFlow output directory for testing purposes.
# def test_get_parflow_output():
#     """Test get_parflow_output function."""
#     start_date = "2003-04-01 01:00:00"
#     end_date = "2003-04-03 00:00:00"
#     temporal_resolution = "hourly"
#     variable = "streamflow"

#     parflow_runname = "02040106"
#     parflow_output_dir = f"/scratch/network/hydrogen_collaboration/example_parflow_runs/02040106_2003-04-01_2003-04-03/outputs/{parflow_runname}_conus2_2003WY"

#     test_obs_metadata_df = pd.read_csv(
#         os.path.join(os.path.dirname(__file__), "test_data/test_obs_metadata_df.csv"),
#         dtype={"site_id": str},
#     )

#     parflow_data_df = model_evaluation.get_parflow_output(
#         test_obs_metadata_df,
#         parflow_output_dir,
#         parflow_runname,
#         start_date,
#         end_date,
#         variable,
#         temporal_resolution,
#     )

#     assert parflow_data_df.shape == (48, 15)
#     assert "01447500" in parflow_data_df.columns


def test_calculate_metrics():
    """Test calculate_metrics function."""
    test_obs_data_df = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "test_data/test_obs_data_df.csv")
    )
    test_parflow_data_df = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "test_data/test_parflow_data_df.csv")
    )
    test_obs_metadata_df = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "test_data/test_obs_metadata_df.csv"),
        dtype={"site_id": str},
    )

    metrics_df = model_evaluation.calculate_metrics(
        test_obs_data_df, test_parflow_data_df, test_obs_metadata_df
    )

    assert metrics_df.shape[0] == 13
    assert "site_id" in metrics_df.columns
    assert "01447500" in list(metrics_df["site_id"])


def test_explore_available_observations():
    """Test explore_available_observations function."""
    grid = "conus2"
    huc_list = ["02040106"]
    ij_bounds, mask = subsettools.define_huc_domain(huc_list, grid)
    start_date = "2003-04-01 01:00:00"
    end_date = "2003-04-03 00:00:00"

    obs_available_metadata_df = model_evaluation.explore_available_observations(
        mask, ij_bounds, grid, date_start=start_date, date_end=end_date
    )

    assert obs_available_metadata_df.shape[0] >= 1306
    assert "stream gauge" in list(obs_available_metadata_df["site_type"])
    assert "groundwater well" in list(obs_available_metadata_df["site_type"])
    assert obs_available_metadata_df["latitude"].min() >= 40.4
    assert obs_available_metadata_df["latitude"].max() <= 41.3
    assert obs_available_metadata_df["longitude"].min() >= -76
    assert obs_available_metadata_df["longitude"].max() <= -75.2


if __name__ == "__main__":
    pytest.main()
