## Clone the repository
Run the following command to clone the repository and navigate into it:

    git clone https://github.com/hydroframe/cssi_evaluation.git

    cd cssi_evaluation

## Set up the recommended local environment

Run these commands from the repository root. This assumes that conda and mamba are already installed (e.g., conda install mamba -n base -c conda-forge).

    mamba env create -f cssi_env.yml

    conda activate cssi_evaluation

    python -m ipykernel install --user --name=cssi_evaluation

The environment file installs the local `cssi_evaluation` package in editable mode with the
`[notebooks,dev]` extras, so package imports, notebooks, and development tools are available in
the same environment.

## Notes

- `pyproject.toml` is the package metadata file used by `pip install -e .`. It contains core library dependencies. 
- `cssi_env.yml` is the single Conda environment file intended for running examples and it contains heavy Conday/system dependencies.
