"""Test /utils/evaluation_utils.py module."""

# pylint: disable=E0401
import sys
import os
import pytest
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from cssi_evaluation.utils import evaluation_utils


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

    metrics_df = evaluation_utils.calculate_metrics(
        test_obs_data_df, test_parflow_data_df, test_obs_metadata_df
    )

    assert metrics_df.shape[0] == 13
    assert "site_id" in metrics_df.columns
    assert "01447500" in list(metrics_df["site_id"])


if __name__ == "__main__":
    pytest.main()
