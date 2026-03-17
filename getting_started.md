## Creating the recommended local environment

Run these commands from the repository root. This assumes that conda and mamba are already installed (e.g., conda install mamba -n base -c conda-forge).

    mamba env create -f cssi_env.yml

    conda activate cssi_evaluation

    python -m ipykernel install --user --name=cssi_evaluation

The environment file installs the local `cssi_evaluation` package in editable mode with the
`[notebooks,dev]` extras, so package imports, notebooks, and development tools are available in
the same environment.


## Creating the recommended environment on **Verde**

Follow these steps:

    # 1. Go to a writable folder (Verde user home)
    cd ~

    # 2. Download and extract micromamba (Linux x86_64)
    curl -L https://micromamba.snakepit.net/api/micromamba/linux-64/latest | tar -xvj bin/micromamba

    # 3. Add micromamba to PATH temporarily (for this session)
    export PATH=$PWD/bin:$PATH

    # 4. Go to the cloned repository root
    cd /home/<USER>/cssi_evaluation

    # 5. Create the environment from the repository YAML
    micromamba env create -f cssi_env.yml

    # 6. Initialize micromamba for your shell
    eval "$(micromamba shell hook --shell bash)"

    # 7. Activate your environment
    micromamba activate cssi_evaluation

    # 8. Register the Jupyter kernel
    python -m ipykernel install --user --name=cssi_evaluation


## Notes

- `pyproject.toml` is the package metadata file used by `pip install -e .`. It contains core library deps. 
- `cssi_env.yml` is the single Conda environment file intended for running examples and it contains heavy Conday/system deps.
