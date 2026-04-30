# OT Score: Official Implementation

Official PyTorch implementation for the paper:

[**OT Score: An OT based Confidence Score for Prototype-Assisted Source Free Unsupervised Domain Adaptation**](https://arxiv.org/abs/2505.11669)

This repository contains code for source-free unsupervised domain adaptation (SFUDA) with the proposed **OT score**, including source model training, target adaptation, and OT-score-related feature/prototype utilities.

## Overview

The OT score is a confidence score based on semi-discrete optimal transport. It is designed to estimate the reliability of target pseudo-labels in prototype-assisted source-free unsupervised domain adaptation.

This repository provides:

- `train_source.py`: train the source model and save source checkpoints
- `train_target.py`: adapt the target model from a trained source checkpoint
- `utils/ot_score_utils.py`: OT score computation and feature/prototype utilities

## Repository Structure

```text
.
├── train_source.py
├── train_target.py
├── environment.yaml
├── utils/
│   └── ot_score_utils.py
├── output/
└── README.md
```

## 1. Environment Setup

Create the conda environment from `environment.yaml`:

```bash
conda env create -f environment.yaml
conda activate OTscore
```

If you use a different environment name, for example `DA`, activate that environment consistently:

```bash
conda activate DA
```

## 2. Dataset Preparation

This code assumes that `--datadir` contains both:

1. Image folders referenced by the list files
2. DomainNet list files, such as:

```text
clipart_list.txt
real_list.txt
painting_list.txt
sketch_list.txt
```

Each list file should follow the format:

```text
relative/or/absolute/image_path label_id
```

For example:

```text
clipart/airplane/image_0001.jpg 0
clipart/bicycle/image_0002.jpg 1
```

In the commands below, replace:

```text
/path/to/domainnet/
```

with your own dataset root directory.

## 3. Train Source Model

To train a source model for one source-target task on DomainNet, run:

```bash
python train_source.py \
  --dset domainnet \
  --s 0 \
  --t 1 \
  --datadir "/path/to/domainnet/" \
  --output output
```

By default, this only trains the source model and does not run target-side testing.

To additionally run source-side testing after training, use:

```bash
python train_source.py \
  --dset domainnet \
  --s 0 \
  --t 1 \
  --datadir "/path/to/domainnet/" \
  --output output \
  --test_target
```

Source checkpoints will be saved to:

```text
output/<dset>/<SourceInitial>/
  source_F_<seed>.pt
  source_B_<seed>.pt
  source_C_<seed>.pt
```

For example:

```text
output/domainnet/C/
  source_F_2026.pt
  source_B_2026.pt
  source_C_2026.pt
```

## 4. Target Adaptation

After training the source model, run target adaptation with:

```bash
python train_target.py \
  --dset domainnet \
  --s 0 \
  --t 1 \
  --datadir "/path/to/domainnet/" \
  --output_src output \
  --output output \
  --seed 2026
```

The target adaptation script loads source checkpoints from:

```text
<output_src>/<dset>/<SourceInitial>/source_*.pt
```

Target checkpoints will be saved to:

```text
<output>/<dset>/<SourceInitial><TargetInitial>/
  target_F_<seed>.pt
  target_B_<seed>.pt
  target_C_<seed>.pt
```

For example:

```text
output/domainnet/CR/
  target_F_2026.pt
  target_B_2026.pt
  target_C_2026.pt
```

## 5. Example Workflow

A typical workflow is:

### Step 1: Train the source model

```bash
python train_source.py \
  --dset domainnet \
  --s 0 \
  --t 1 \
  --datadir "/path/to/domainnet/" \
  --output output \
  --seed 2026
```

### Step 2: Adapt the target model

```bash
python train_target.py \
  --dset domainnet \
  --s 0 \
  --t 1 \
  --datadir "/path/to/domainnet/" \
  --output_src output \
  --output output \
  --seed 2026
```

## 6. OT Score Utilities

The OT score and related feature/prototype utilities are implemented in:

```text
utils/ot_score_utils.py
```

These utilities are used during target adaptation to compute confidence scores for pseudo-labeled target samples.

The OT score can be used for:

- identifying low-confidence target pseudo-labels
- reweighting target samples during adaptation
- providing a label-free proxy for target-domain performance

## 7. Reference

If you find this repository useful, please cite our paper:

```bibtex
@article{zhang2026otscore,
  title={OT Score: An OT based Confidence Score for Prototype-Assisted Source Free Unsupervised Domain Adaptation},
  author={Zhang, Yiming and Liu, Sitong and Cloninger, Alex},
  journal={Transactions on Machine Learning Research},
  year={2026}
}
```
