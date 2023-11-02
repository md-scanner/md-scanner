# MD Scanner Code Monorepo

MD-Scanner is a Computer Vision Pipeline that generate accurate Markdown files from document images. Notably, it seems to performs better than Meta AI's [Nougat](https://facebookresearch.github.io/nougat/).

![](./us_vs_fb.jpg)

## How to setup (Linux)

From the repository root directory, create a python virtual enviornment and activate it:
```
python -m venv .venv
source ./.venv/bin/activate
```

Install torch v1.10.1 + CUDA 11.1 ([described here](https://pytorch.org/get-started/previous-versions/#linux-and-windows-15)):
```
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
```

Install detectron2 for torch v1.10 + CUDA 11 ([described here](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)):
```
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html
```

Install other python requirements:
```
pip install -r requirements.txt
```

## How to run (Linux)

```
python pipeline.py <input-image> <output-md>
```

## Repository Contents

This repository contains all of the code we wrote for the MarkDown Scanner, except for some Section Classification-specific code. Those things are in [this repository](https://github.com/md-scanner/unilm). There you can also find the [`pipeline.py` file](https://github.com/md-scanner/unilm/blob/master/dit/pipeline.py) running the whole pipeline except the preprocessing step.

For dataset generation, there a lot of shell scripts (some are Bash-specific, some others are not) meant to run on UNIX-like environments.

Other requirements are:

* **Python 3.7** (compatibility with Torch 1.10.1 + CUDA 11.1, required by detectron2)
* for the `md-retriever`: Python 3 and some packages (TODO `requirements.txt`).
* for the `md-renderer`: ImageMagick and:
    1. Pandoc and the themes from [this repository](https://github.com/cab-1729/Pandoc-Themes) and associated fonts for the MD->Latex->PDF->JPG conversion (no BBs generated), using the `gen_pdfs.sh` shell script.
    2. Docker (and an active Internet connection to allow it to fetch images remotely) for the MD->HTML->PDF->JPG/PNG conversion (with generated BBs), using the `main.sh` Bash script.
* for the `bb-extractor`: CMake, a C++ toolchain, OpenCV and its development headers;
* for `to-coco-format`: Python 3 and `pandas`

**NOTE: You can find the latest training checkpoints for our models and the generated DB in the repository [releases](https://github.com/md-scanner/md-scanner/releases).**

## Steps to generate the dataset

* retrieve Markdown files from the GitHub API using `md-retriever`;
* generate images from those Markdown files using `md-renderer`;
* extract bounding box coordinates to CSV files using `extract-bbs`.

To convert the dataset to (PubLayNet-like) COCo format by generating a JSON file from the CSVs, use `to-coco-format`. 

## Resources:
- PubLayNet (Jupyter Notebook): https://github.com/ibm-aur-nlp/PubLayNet/blob/master/exploring_PubLayNet_dataset.ipynb
- PubLayNet: https://arxiv.org/pdf/1908.07836.pdf
- FUNSD: https://arxiv.org/pdf/1905.13538.pdf


