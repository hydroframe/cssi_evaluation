# cssi_evaluation

`cssi_evaluation` contains tools for evaluating hydrologic model outputs against
observations and derived metrics.

- **Models** – model-specific preprocessing and data handling
- **Variables** – evaluation functions for specific hydrologic variables
- **Utilities** – general-purpose functions used across models and variables
- **External Data** - access to observational datasets not currently available through the Princeton HydroData catalog

These modules are currently designed to be imported and used within evaluation Jupyter notebooks.

## Package Structure
```bash
/src/cssi_evaluation/
│
├── models/                         # Model-specific integrations and preprocessing
│   ├── __init__.py
│   ├── parflow_utils.py
│   ├── nwm_utils.py
│   └── README.md                   # Instructions for contributing a new model integration
│
├── variables/                      # Variable-specific evaluation logic
│   ├── __init__.py
│   ├── snow_utils.py
│   ├── streamflow_utils.py
│   └── groundwater_utils.py        # Place holder example            
│
├── utils/                          # General utilities usable across models
│   ├── __init__.py
│   ├── metric_utils.py
│   ├── evaluation_utils.py
│   ├── dataPrep_utils.py                 # May overlap with HydroData tools
│   └── plot_utils.py
│
└── external_data_access/           # Accessing observational datasets external to HydroData
    ├── __init__.py
    └── observation_utils.py
```



## Directory Descriptions

### `models/`

Contains functions specific to individual hydrologic models. These functions
typically handle:

- model output preprocessing
- variable extraction
- unit conversions
- **data formatting to match the evaluation framework**

Each supported model should have its own module (e.g., `parflow_utils.py`, `nwm_utils.py`).

See `models/README.md` for instructions on contributing a new model integration.

---

### `variables/`

Contains evaluation function organized by **hydrologic variable**.

These modules compute diagnostics and metrics specific to a variable, such as:

- snow metrics (peak SWE, melt timing, snow duration)
- streamflow metrics (flow statistics, timing metrics)
- groundwater storage or depletion metrics

Functions in these modules typically operate on data that has already been
standardized into a common structure.

---

### `utils/`

Contains general-purpose functions used across models and variables.

These utilities include:

- evaluation workflows
- statistical metrics
- data preparation and restructuring
- plotting and visualization tools

These functions should be **model-agnostic** whenever possible.

---

### `external_data_access/`

Provides functions for accessing observational datasets that are external to
HydroData or the modeling framework.

Examples may include:

- USGS streamflow observations
- snow observations
- other observational datasets used for validation



## Example Usage --- Need to modify the below example, just a place holder for now

```python
# from cssi_evaluation.variables import snow
# from cssi_evaluation.utils.metrics import nse

# peak_swe = snow.compute_peak_swe(swe_data)
# nse_score = nse(simulated_streamflow, observed_streamflow)