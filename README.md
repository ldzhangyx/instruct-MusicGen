
<div align="center">

# instruct-MusicGen

[![python](https://img.shields.io/badge/-Python_3.11.7-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.2405.18386-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

This is the official repository for the paper "Instruct-MusicGen: Unlocking Text-to-Music Editing for Music Language Models via Instruction Tuning".

If there is any problem related to the code running, please open an issue and I will help you as mush as I can.

## Official pretrained ckpt

To promote transparency and reproducibility in research, I have retrained a similar model using publicly available datasets after the internship. This model has been trained on public data and adheres to the same methodology described in the paper.

**Note that this is NOT the official ckpt, which can match the results in this paper. This pretrained model has NO relation with Sony.
**
https://drive.google.com/file/d/1F4GaKuMqPjftjqOo2t5wdhix0R2bQStT/view?usp=drive_link



## Demo page

https://bit.ly/instruct-musicgen

## Installation

#### Pip

```bash
# clone project
git clone https://github.com/ldzhangyx/instruct-MusicGen/
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
git clone https://github.com/ldzhangyx/instruct-MusicGen/
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

You may need to change essential parameters in `config/config.yaml` to fit your own dataset.


You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=50 data.batch_size=4
```

## Evaluation

### Step 1: Generate evaluation datasets

```bash
python src/data/slakh_datamodule.py
```

### Step 2: Generate music files

For `add`, `remove`, `extract` operation, please change the parameters in both `test_step()` in `src/models/instructmusicgenadapter_module.py` and `__getitem__()` in `src/data/slakh_datamodule.py`.

Currently it should be completed manually. But we will provide a script to automate this process soon.


```bash
python src/eval.py
```

### Step 3: Evaluate

Please make sure the generated music files are in the corresponding locations.

```bash
python evaluation/utils.py  # to generate a csv file for CLAP calculation
python evaluation/main.py
```

## Inference script

After preparing the checkpoint and the input audio file, you can generate audio via

```bash
python src/inference.py
```

## Citation

```
@article{zhang2024instruct,
  title={Instruct-MusicGen: Unlocking Text-to-Music Editing for Music Language Models via Instruction Tuning},
  author={Zhang, Yixiao and Ikemiya, Yukara and Choi, Woosung and Murata, Naoki and Mart{\'\i}nez-Ram{\'\i}rez, Marco A and Lin, Liwei and Xia, Gus and Liao, Wei-Hsiang and Mitsufuji, Yuki and Dixon, Simon},
  journal={arXiv preprint arXiv:2405.18386},
  year={2024}
}
```
