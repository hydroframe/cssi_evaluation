"""
Streamflow evaluation utility functions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_streamflow_diagnostics(
    streamflow_data_df,
    model_df,
    site_id,
    metrics_row=None,
    output_dir="."
):
    """
    Create a 3-panel diagnostic plot:
    1) Hydrograph
    2) Flow Duration Curve (FDC)
    3) Q-Q plot

    Parameters
    ----------
    streamflow_data_df : DataFrame
        Observed data with 'date' column
    model_df : DataFrame
        Modeled data with 'date' column
    site_id : str
        Gage/site ID column name
    metrics_row : Series (optional)
        Row containing metrics (rho, bias, nse, condon)
    output_dir : str; default="."
        String path to where plots should be saved. Default is current working directory.
    """

    # ensure datetime
    dates = pd.to_datetime(streamflow_data_df['date'])

    obs = streamflow_data_df[site_id].values
    mod = model_df[site_id].values

    # remove any NaNs in observed and modeled dataframes
    mask = ~np.isnan(obs) & ~np.isnan(mod)
    obs = obs[mask]
    mod = mod[mask]
    dates = dates[mask]

    # create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    # =========================================================
    # 1. HYDROGRAPH
    # =========================================================
    ax = axes[0]
    ax.plot(dates, obs, label="Observed", linewidth=2)
    ax.plot(dates, mod, label="Modeled", linewidth=2, alpha=0.8)

    ax.set_title(f"Hydrograph at gage {site_id}")
    ax.set_ylabel("Streamflow")
    ax.legend()

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # --- metrics annotation ---
    if metrics_row is not None:
        textstr = (
            f"Srho: {metrics_row['spearman_rho']:.2f}\n"
            f"Bias: {metrics_row['bias']:.2f}\n"
            f"NSE: {metrics_row['nse']:.2f}\n"
            f"Condon: {metrics_row['condon']}"
        )
        ax.text(
            0.02, 0.98, textstr,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", alpha=0.2)
        )

    # =========================================================
    # 2. FLOW DURATION CURVE (FDC)
    # =========================================================
    ax = axes[1]

    # sort descending
    obs_sorted = np.sort(obs)[::-1]
    mod_sorted = np.sort(mod)[::-1]

    # exceedance probability
    p = np.arange(1, len(obs_sorted)+1) / (len(obs_sorted)+1)

    ax.plot(p, obs_sorted, label="Observed", linewidth=2)
    ax.plot(p, mod_sorted, label="Modeled", linewidth=2, alpha=0.8)

    ax.set_yscale("log")  # important!
    ax.set_xlabel("Exceedance Probability")
    ax.set_ylabel("Streamflow")
    ax.set_title("Flow Duration Curve")
    ax.legend()

    # =========================================================
    # 3. Q–Q PLOT
    # =========================================================
    ax = axes[2]

    q = np.linspace(0, 1, 100)
    obs_q = np.quantile(obs, q)
    mod_q = np.quantile(mod, q)

    ax.scatter(obs_q, mod_q, s=20, alpha=0.7)

    # 1:1 line
    min_val = min(obs_q.min(), mod_q.min())
    max_val = max(obs_q.max(), mod_q.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--')

    ax.set_xlabel("Observed Quantiles")
    ax.set_ylabel("Modeled Quantiles")
    ax.set_title("Q-Q Plot")

    # =========================================================
    plt.tight_layout()
    plt.savefig(f"{output_dir}/streamflow_diagnostics_3panel_{site_id}.png", bbox_inches="tight", dpi=300)