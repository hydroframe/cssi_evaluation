"""Test utils.py module."""

# pylint: disable=E0401
import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from cssi_evaluation import utils


def test_check_mask_shape():
    """Test check_mask_shape function."""
    mask = np.zeros(shape=(8, 2))
    ij_bounds = [100, 100, 102, 108]

    # check no exception raised
    utils.check_mask_shape(mask, ij_bounds)


def test_check_mask_shape_error():
    """Test check_mask_shape function exception handling."""
    with pytest.raises(Exception) as exc:
        mask = np.zeros(shape=(8, 2))
        ij_bounds = [100, 100, 101, 108]

        utils.check_mask_shape(mask, ij_bounds)
    assert (
        str(exc.value) == "The mask shape is (8, 2) but the ij_bounds is shape (8, 1)"
    )


def test_get_domain_indices():
    """Test get_domain_indices function."""
    ij_bounds = [1000, 1000, 1500, 1500]
    conus_indices = [1250, 1300]

    assert utils.get_domain_indices(ij_bounds, conus_indices) == (250, 300)


def test_initialize_metrics_df():
    """Test initialize_metrics_df function."""
    mock_obs_metadata_df = pd.DataFrame(
        data={
            "site_id": [1, 2, 3],
            "site_name": ["a", "b", "c"],
            "latitude": [50, 60, 70],
            "longitude": [115, 116, 117],
            "domain_i": [0, 5, 10],
            "domain_j": [40, 41, 42],
        }
    )

    mock_metrics_df = utils.initialize_metrics_df(
        mock_obs_metadata_df, ["mse", "rmse", "spearman_rho"]
    )
    assert mock_metrics_df.shape == (3, 9)
    assert "mse" in mock_metrics_df.columns
    assert "rmse" in mock_metrics_df.columns
    assert "spearman_rho" in mock_metrics_df.columns
    assert "site_id" in mock_metrics_df.columns


if __name__ == "__main__":
    pytest.main()
