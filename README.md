# FedSSIM: Federated Learning with Selective Weight Refinement

This project implements Federated Learning (FL) simulations using two distinct approaches: an experimental method based on **Grad-CAM** for selective weight refinement (**FedSSIM**) and a scalable **baseline** system powered by the **Flower** framework.

## Installation

Ensure you have Python 3.9+ installed, then install the required dependencies:

```bash
pip install -r requirements.txt
```

> [!NOTE]
> The first time you run a simulation with a new dataset, it will be automatically downloaded via Torchvision APIs. This may require additional time and a stable internet connection.

---

## 1. Experimental Approach (FedSSIM / Grad-CAM)

The script `fed_gradcam.py` implements an experimental approach where clients use Grad-CAM to identify channel importance in the last convolutional layer (`conv3`). The server then performs a "Winner-Takes-All" override for those specific channels based on donor client performance.

### FedSSIM Usage

```bash
python3 fed_gradcam.py [options]
```

### FedSSIM Parameters

- `--dataset`: `stl10` or `cifar10` (default: `stl10`)
- `--rounds`: Number of FL rounds (default: 5)
- `--num_clients`: Number of clients (default: 4)
- `--distribution`: `iid` or `non-iid` (default: `non-iid`)
- `--alpha`: Dirichlet parameter for non-IID distribution (default: 0.5)

### FedSSIM Output

- Performance plots are saved as `fedgradcam_results_[dataset]_[dist].pdf`.
- Grad-CAM visualizations for each round/client are saved in the `visualizations/` directory as `.pdf` files.

---

## 2. Baseline Comparisons (Flower Framework)

The script `flower_baseline.py` utilizes **Flower** to run standard FL simulations with a wide variety of aggregation strategies. It supports several image classification architectures.

### Flower Usage

```bash
python3 flower_baseline.py [options]
```

### Flower Parameters

- `--baseline`: One or more strategies (use `all` to run all).
- `--dataset`: `cifar10`, `stl10`, `mnist`, or `oxfordpet` (default: `cifar10`).
- `--model`: `cnn`, `squeezenet`, `shufflenet`, or `resnet` (default: `cnn`).
- `--data-distr`: Data distribution (1.0 for IID, < 1.0 for Dirichlet non-IID).
- `--rounds`: Number of rounds (default: 5).
- `--num_clients`: Number of clients (default: 4).

### Available Strategies

| Strategy | Full Name | Description |
| :--- | :--- | :--- |
| `fedavg` | Federated Averaging | The standard FL algorithm; computes a weighted average of client updates. |
| `fedavgm` | FedAvg with Momentum | Enhances FedAvg with server-side momentum to accelerate convergence. |
| `fedprox` | Federated Proximal | Adds a proximal term to local objectives to handle system heterogeneity. |
| `fedadam` | Federated Adam | Adaptive optimizer that uses estimates of first and second moments of gradients. |
| `fedyogi` | Federated Yogi | Adaptive optimizer designed to be more robust than Adam in certain FL settings. |
| `fedadagrad`| Federated Adagrad | Adaptive optimizer that scales the learning rate based on historical gradients. |
| `fedmedian`| Federated Median | A robust aggregation method that uses the coordinate-wise median. |
| `fedtrimmedavg`| Trimmed Mean | Robustly aggregates by removing a percentage of extreme client values. |
| `faulttolerantfedavg`| Fault-Tolerant FedAvg| Standard FedAvg with robust handling of client failures during a round. |
| `fedexp` | FedExp | Placeholder for the experimental FedExp strategy. |

### Baseline Examples

- **Run FedAvg on OxfordPet using ResNet**:

  ```bash
  python3 flower_baseline.py --baseline fedavg --dataset oxfordpet --model resnet --rounds 10
  ```

- **Compare multiple baselines on MNIST**:

  ```bash
  python3 flower_baseline.py --baseline fedavg fedprox fedexp --dataset mnist --model cnn --rounds 5
  ```

### Flower Output

- **CSV Logs**: Detailed logs per strategy in `csv/baseline_[method]_[dataset].csv`.
- **Plots**: Visualizations saved in `results/` using the format `[metric]_[baselines]_[model]_[num_clients]Clients.pdf`.

---

## 3. Automated Experiment Runner

The script `run_experiments.py` is a utility tool to automate the execution of multiple simulations across all supported datasets and models.

### Runner Usage

```bash
python3 run_experiments.py [options]
```

By default, it runs all combinations of **Datasets** and **Models** with 20 clients and an Alpha value of 0.5. You can use `--baseline all` to run all strategies for every combination (caution: this will result in a large number of simulations).

---

## Technical Requirements

The script automatically detects and utilizes the best available hardware:

- **CUDA**: NVIDIA GPUs
- **MPS**: Apple Silicon (Metal Performance Shaders)
- **CPU**: Fallback for other systems
