# Graph Neural Networks for particle track reconstruction

## Installation
Please install [miniconda](https://docs.conda.io/en/latest/miniconda.html). Jupyter lab is needed to run [Jupyter@NERSC](https://jupyter.nersc.gov)

After installed the conda, please write following to `~/.condarc`
```
envs_dirs:
  - ~/.conda/envs
report_errors: true
```
Then install
```bash
source ~/miniconda3/bin/activate
conda create --name heptrkx python=3.7
conda activate heptrkx

conda install -c conda-forge jupyterlab
git clone https://github.com/xju2/heptrkx-gnn-tracking.git heptrkx
cd heptrkx
pip install -e .
```

For your reference, I exported all packages from my setup, attached the [environment.yml](https://github.com/xju2/heptrkx-gnn-tracking/blob/tf2/environment.yml).
Note that it may contain packages that are not used in this project.
