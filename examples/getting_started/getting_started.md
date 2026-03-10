## Creating your HydroLearnEnv Virtual Environment
Follow these steps: 

    mamba env create -f nwm_env.yml 

    conda activate nwm_env

    python -m ipykernel install --user --name=nwm_env


## Creating your HydroLearnEnv Virtual Environment on **Verde**
Follow these steps:  

    # 1. Go to a writable folder (Verde user home)
    cd ~

    # 2. Download and extract micromamba (Linux x86_64)
    curl -L https://micromamba.snakepit.net/api/micromamba/linux-64/latest | tar -xvj bin/micromamba

    # 3. Add micromamba to PATH temporarily (for this session)
    export PATH=$PWD/bin:$PATH

    # 4. Create the environment from the YAML (navigate to /home/<USER>/cssi_evaluation/examples/notebooks/snow)   
    micromamba env create -f nwm_env.yml

    # 5. Initialize micromamba for your shell
    eval "$(micromamba shell hook --shell bash)"  

    # 6. Register Jupyter Kernel
    python -m ipykernel install --user --name=nwm_env

    # 7. Activate your environment
    micromamba activate nwm_env #make sure to restart the Jupyter session if running

