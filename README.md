# Federated Learning Experiment Framework

This project provides a flexible framework for simulating Federated Learning (FL) experiments using [Flower](https://flower.ai/) and PyTorch. It supports various models, datasets (including Image, Tabular, and Audio), and aggregation strategies.

## Overview

The main script `flower_baseline.py` simulates a federated learning environment where clients train locally on their data partitions, and a central server aggregates their updates using a specified strategy.

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

## Installation

Ensure you have Python 3.9+ and install the dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

Run the simulation using `python3 flower_baseline.py`. Below are common usage examples:

| Scenario | Command |
| :--- | :--- |
| **Run FedExp on CIFAR-10** | `python3 flower_baseline.py --baseline fedexp --dataset cifar10 --model cnn` |
| **Run SpeechCommands with M5** | `python3 flower_baseline.py --baseline fedavg --dataset speechcommands --model m5` |
| **Run on Tabular Data** | `python3 flower_baseline.py --baseline fedavg --dataset adult --model mlp` |
| **Set Random Seed** | `python3 flower_baseline.py --baseline fedexp --seed 57` |
| **Non-IID Data (Dirichlet)** | `python3 flower_baseline.py --data-distr 0.5 --num_clients 10` |
| **Run Multiple Baselines** | `python3 flower_baseline.py --baseline fedavg fedexp --rounds 20` |

### Arguments

- `--dataset`: Choose dataset (e.g., `cifar10`, `speechcommands`).
- `--model`: Choose model architecture (e.g., `cnn`, `m5`).
- `--baseline`: Strategy to use (e.g., `fedexp`, `fedavg`, or `all`).
- `--num_clients`: Number of FL clients.
- `--rounds`: Number of server rounds.
- `--epochs`: Local training epochs per round.
- `--seed`: Random seed for reproducibility.
- `--data-distr`: Dirichlet alpha parameter (1.0 = IID, <1.0 = Non-IID).
- `--learning_type`: Learning type (e.g., `FL`, `SL`).

## Visualization

Results are saved in `csv/` and plots in `results/`.
You can use `visualization.ipynb` to regenerate plots from the CSV logs.
