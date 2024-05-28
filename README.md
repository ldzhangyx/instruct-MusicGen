
<div align="center">

# instruct-MusicGen

[![python](https://img.shields.io/badge/-Python_3.11.7-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

This is the official repository for the paper "Instruct-MusicGen: Unlocking Text-to-Music Editing for Music Language Models via Instruction Tuning".

**We will release the source code and model weights very soon.**

## Installation

#### Pip

```bash
# clone project
git clone https://github.com/xxx/instruct-MusicGen/
cd instruct-MusicGen

# [OPTIONAL] create conda environment
conda create -n myenv python=3.11.7
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

#### Conda

```bash
# clone project
git clone https://github.com/xxx/instruct-MusicGen/
cd instruct-MusicGen

# create conda environment and install dependencies
conda env create -f environment.yaml -n myenv

# activate conda environment
conda activate myenv
```

## How to run

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```


You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=50 data.batch_size=4
```
