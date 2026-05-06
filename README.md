# Federated Learning Experiment Framework

This project provides a flexible framework for simulating Federated Learning (FL) experiments using [Flower](https://flower.ai/) and PyTorch. It supports various models, datasets (including Image, Tabular, and Audio), and aggregation strategies.

## Overview

The main script `flower_baseline.py` simulates multiple training settings on top of [Flower](https://flower.ai/): centralized, federated, split, continual federated, and continual federated split learning. Clients train locally on their partitions, and Flower strategies orchestrate aggregation or sequential updates depending on the selected method.

## Available Features

### Models

| Model | Description | Best For |
| :--- | :--- | :--- |
| **CNN** | A standard 3-layer Convolutional Neural Network. | simple image datasets (CIFAR-10, MNIST) |
| **MLP** | Multi-Layer Perceptron. | Tabular data (Adult) or simple vector data |
| **M5** | A standard Audio CNN (refer to [Dai et al.](https://arxiv.org/abs/1610.00087)). | Audio classification (SpeechCommands) |
| **ResNet** | ResNet-18 architecture. | Complex image datasets |
| **SqueezeNet** | Lightweight SqueezeNet 1.1. | Efficient image classification |
| **ShuffleNet** | ShuffleNet V2 x0.5. | Mobile/Edge efficient image classification |

### Datasets

| Dataset | Type | Description |
| :--- | :--- | :--- |
| **CIFAR10** | Image | 10 classes of 32x32 color images. Standard benchmark. |
| **MNIST** | Image | Handwritten digits (grayscale). |
| **STL10** | Image | 10 classes, larger images (96x96). |
| **OxfordPet** | Image | 37 categories of pet breeds. Fine-grained classification. |
| **Adult** | Tabular | Census Income dataset. Binary classification (income >50K). |
| **SpeechCommands** | Audio | Short audio commands (1 sec). 35 classes. |

### Strategies (Baselines)

| Strategy | Description |
| :--- | :--- |
| **FedAvg** | Federated Averaging. The standard FL baseline. |
| **FedExp** | **(Ours)** Custom aggregation strategy weighting clients by inverse training loss. |
| **FedAvgM** | FedAvg with Server Momentum. |
| **FedMedian** | Uses median aggregation instead of mean (robustness). |
| **FedAdam** | Adaptive optimization (Adam) on the server side. |
| **FedProx** | Adds a proximal term to handle data heterogeneity. |

### Learning Methods

| Method | Description |
| :--- | :--- |
| **CL** | Centralized Learning with the full training set assigned to a single Flower client. |
| **FL** | Standard Federated Learning with all clients participating each round. |
| **SL** | Split Learning with round-robin client updates on a split model. |
| **CFL** | Continual Federated Learning with per-client experience streams and replay. |
| **CFSL** | Continual Federated Split Learning combining split models, Flower aggregation, and continual streams. |

## Installation

Ensure you have Python 3.9+ and install the dependencies:

```bash
pip install -r requirements.txt
```

For a practical command reference covering local runs, smoke tests, and Leonardo RCM interactive execution, see [NIGHTLY_ALPHA05_FL_SFLV1_SFLV2.md](/Users/ivan/Desktop/DML-Analysis/NIGHTLY_ALPHA05_FL_SFLV1_SFLV2.md:1).

## How to Run

Run the simulation using `python3 flower_baseline.py`. Below are common usage examples:

| Scenario | Command |
| :--- | :--- |
| **Run FedAvg on CIFAR-10** | `python3 flower_baseline.py --baseline fedavg --dataset cifar10 --model cnn --learning-type FL` |
| **Run Centralized Learning** | `python3 flower_baseline.py --dataset cifar10 --model cnn --learning-type CL` |
| **Run Split Learning** | `python3 flower_baseline.py --dataset cifar10 --model cnn --learning-type SL` |
| **Run Continual Federated Learning** | `python3 flower_baseline.py --baseline fedavg --dataset cifar10 --model cnn --learning-type CFL` |
| **Run Fair Comparison Profile** | `python3 flower_baseline.py --baseline fedavg --dataset mnist --model cnn --learning-type CFL --comparison-profile fair` |
| **Run SpeechCommands with M5** | `python3 flower_baseline.py --baseline fedavg --dataset speechcommands --model m5` |
| **Run on Tabular Data** | `python3 flower_baseline.py --baseline fedavg --dataset adult --model mlp` |
| **Set Random Seed** | `python3 flower_baseline.py --baseline fedexp --seed 57` |
| **Non-IID Data (Dirichlet)** | `python3 flower_baseline.py --data-distr 0.5 --num_clients 10` |
| **Run Multiple Baselines** | `python3 flower_baseline.py --baseline fedavg fedexp --rounds 20` |

For repeated studies across datasets/models/client settings, use `python3 run_experiments.py`. Example:

```bash
python3 run_experiments.py \
  --dataset mnist \
  --model cnn \
  --learning-type FL SL CFL CFSL \
  --num-clients 20 \
  --clients-per-round 5 10 20 \
  --rounds 10 \
  --epochs 1 \
  --repetitions 5
```

This runner:
- launches one experiment per unique configuration and repetition;
- assigns a deterministic seed to each repetition;
- archives the generated CSV/PDF artifacts into `study_runs/<study_name>/`;
- refreshes `run_summary.csv`, `aggregate_metrics.csv`, and `pairwise_statistics.csv`;
- computes pairwise Mann-Whitney U tests and A12 effect sizes across repeated runs.

### Arguments

- `--dataset`: Choose dataset (e.g., `cifar10`, `speechcommands`).
- `--model`: Choose model architecture (e.g., `cnn`, `m5`).
- `--baseline`: Strategy to use (e.g., `fedexp`, `fedavg`, or `all`).
- `--num_clients`: Number of FL clients.
- `--rounds`: Number of server rounds.
- `--epochs`: Local training epochs per round.
- `--seed`: Random seed for reproducibility.
- `--data-distr`: Dirichlet alpha parameter (1.0 = IID, <1.0 = Non-IID).
- `--learning-type`: Learning type (`CL`, `FL`, `SL`, `CFL`, `CFSL`).
- `--comparison-profile`: `fair` for budget-matched continual replay and logical SL rounds, `legacy` to reproduce the older behavior.
- `--continual-steps`: Number of continual experiences per client.
- `--continual-replay-ratio`: Replay ratio used in continual modes.

## Visualization

Results are saved in `csv/` and plots in `results/`.

Repeated studies are archived under `study_runs/`. The notebook [Results_Visualization.ipynb](/Users/ivan/Desktop/FedTest/Results_Visualization.ipynb) loads the latest study automatically, considers repetitions, and displays descriptive statistics together with Mann-Whitney U and A12 comparisons.
