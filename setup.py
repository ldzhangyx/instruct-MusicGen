#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="instruct-musicgen",
    version="0.1.0",
    description="Instruct-MusicGen: Unlocking Text-to-Music Editing for Music Language Models via Instruction Tuning",
    author="Yixiao Zhang",
    author_email="ldzhangyx@gmail.com",
    url="https://github.com/ldzhangyx/instruct-MusicGen",
    install_requires=["lightning", "hydra-core"],
    packages=find_packages(),
    python_requires=">=3.11.7",
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = src.train:main",
            "eval_command = src.eval:main",
        ]
    },
)
