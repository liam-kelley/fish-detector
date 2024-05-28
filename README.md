# fish-detector

Welcome to fish detector.

## Setup

Install the cuda 12.1 toolkit. Other versions may work, but this is the one I used.
If you use another other, please change the pytorch-cuda version in the conda install command below.

```bash

conda create -n fish python=3.9
conda activate fish
conda install numpy
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 lightning -c pytorch -c nvidia -c conda-forge
python -m pip install pandas
python -m pip install scipy
python -m pip install librosa
python -m pip install cupy-cuda12x
python -m pip install matplotlib
python -m pip install IPython
python -m pip install wandb
python -m pip install tqdm
python -m pip install PyYAML

```

## Usage

Good luck
