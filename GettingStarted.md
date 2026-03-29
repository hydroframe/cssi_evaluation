## Clone the repository
Run the following command to clone the repository and navigate into it:

    git clone https://github.com/hydroframe/cssi_evaluation.git

    cd cssi_evaluation

## Set up the recommended local environment

Run these commands from the repository root. This assumes that conda and mamba are already installed (e.g., conda install mamba -n base -c conda-forge).

```bash

$ mamba env create -f cssi_env.yml

$ mamba activate cssi_evaluation

$ python -m ipykernel install --user --name=cssi_evaluation
```

The environment file installs the local `cssi_evaluation` package in editable mode with the
`[notebooks,dev]` extras, so package imports, notebooks, and development tools are available in
the same environment.

## Notes

- `pyproject.toml` is the package metadata file used by `pip install -e .`. It contains core library dependencies. 
- `cssi_env.yml` is the single Conda environment file intended for running examples and it contains heavy Conday/system dependencies.

## Running the Examples

Examples that demonstrate the capabilities of the evaluation framework are located in the `examples` directory:

```text
cssi_evaluation/
├── src/cssi_evaluation/         # Core framework code
├── examples/                    # Example notebooks and supporting assets
│   ├── collect_observations/    # Demonstrates observation data collection capabilties
│   ├── nwm/                     # National Water Model examples
│   ├── parflow/                 # Parflow examples
├── docs/                        # Project documentation and notes
├── tests/                       # Package tests
├── pyproject.toml               # Package metadata and dependencies
└── README.md
```

To begin exploring these examples, issue the following command to start a Jupyter Lab interface:

```bash
$ jupyter lab
```

This will start up an interactive web interface for you to explore the capabilities of the framework. Simply navigate through the `examples` directory to begin testing.

