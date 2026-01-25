import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import flwr as fl
from flwr.common import Context
import argparse
import numpy as np
import os
import csv
import time
from collections import OrderedDict
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import logging
import urllib.request
import json
import base64
logging.getLogger("flwr").setLevel(logging.ERROR)

from captum.attr import visualization as vit
from fedexp import FedExpStrategy, set_inplace, get_layer_importance

# --- 1. Dataset-Specific Hyperparameters ---
DATASET_HYPERPARAMS = {
    'mnist': {'lr': 0.01, 'batch_size': 32, 'epochs': 2},
    'cifar10': {'lr': 0.01, 'batch_size': 64, 'epochs': 3},
    'stl10': {'lr': 0.01, 'batch_size': 32, 'epochs': 3},
    'oxfordpet': {'lr': 0.005, 'batch_size': 16, 'epochs': 5}
}


class CNN(nn.Module):
    def __init__(self, num_classes=10, input_size=128, in_channels=3):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1) 
        self.pool = nn.MaxPool2d(2, 2)
        
        dummy_input = torch.zeros(1, in_channels, input_size, input_size)
        x = self.pool(F.relu(self.conv1(dummy_input)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        self.fc_input_dim = x.view(1, -1).size(1)
        
        self.fc1 = nn.Linear(self.fc_input_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, self.fc_input_dim)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_model(name, num_classes=10, input_size=128, in_channels=3):
    if name.lower() == 'cnn':
        return CNN(num_classes=num_classes, input_size=input_size, in_channels=in_channels)
    
    elif name.lower() == 'squeezenet':
        model = models.squeezenet1_1(num_classes=num_classes)
        if in_channels != 3:
            model.features[0] = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2)
        return model
    
    elif name.lower() == 'shufflenet':
        model = models.shufflenet_v2_x0_5(num_classes=num_classes)
        if in_channels != 3:
            model.conv1[0] = nn.Conv2d(in_channels, 24, kernel_size=3, stride=2, padding=1, bias=False)
        return model
    
    elif name.lower() == 'resnet':
        model = models.resnet18(num_classes=num_classes)
        if in_channels != 3:
            model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        return model
    
    else:
        raise ValueError(f"Unsupported model: {name}")

def warmup(model_name, num_classes, img_size, in_channels, device):
    """Performs a dummy forward-backward pass to initialize hardware and libraries."""
    print("  - Warming up hardware (first-run initialization)...")
    model = get_model(model_name, num_classes=num_classes, input_size=img_size, in_channels=in_channels).to(device)
    dummy_input = torch.randn(1, in_channels, img_size, img_size).to(device)
    model.train()
    try:
        output = model(dummy_input)
        loss = output.sum()
        loss.backward()
        if device == 'cuda':
            torch.cuda.synchronize()
        elif device == 'mps':
            torch.mps.synchronize()
    except Exception as e:
        print(f"    Warning: Warmup failed (not critical): {e}")
    print("  - Warmup complete.")

# --- 3. Dataset Loader ---

def get_dataset(name, img_size):
    if name.lower() == 'cifar10':
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        targets = np.array(train_set.targets)
    elif name.lower() == 'stl10':
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_set = datasets.STL10(root='./data', split='train', download=True, transform=transform)
        test_set = datasets.STL10(root='./data', split='test', download=True, transform=transform)
        targets = np.array(train_set.labels)
    elif name.lower() == 'mnist':
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        targets = np.array(train_set.targets)
    elif name.lower() == 'oxfordpet':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_set = datasets.OxfordIIITPet(root='./data', split='trainval', download=True, transform=transform)
        test_set = datasets.OxfordIIITPet(root='./data', split='test', download=True, transform=transform)
        # Note: torchvision's OxfordIIITPet targets are in ._labels
        targets = np.array(train_set._labels)
    else:
        raise ValueError(f"Unsupported dataset: {name}")
    return train_set, test_set, targets

def partition_data(dataset, targets, num_clients, alpha, num_classes):
    indices = [[] for _ in range(num_clients)]
    
    if alpha >= 1.0:
        all_indices = np.arange(len(dataset))
        np.random.shuffle(all_indices)
        indices = np.array_split(all_indices, num_clients)
        return [idx.tolist() for idx in indices]
    else:
        # Non-IID (Dirichlet)
        for k in range(num_classes):
            idx_k = np.where(targets == k)[0]
            if len(idx_k) == 0: continue
            np.random.shuffle(idx_k)
            
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            
            idx_splits = np.split(idx_k, proportions)
            for i in range(num_clients):
                indices[i].extend(idx_splits[i].tolist())
                
        for i in range(num_clients):
            np.random.shuffle(indices[i])
            
    return indices

# --- 2. Flower Client ---

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, valloader, device, epochs, lr):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.epochs = epochs
        self.lr = lr

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        epoch_losses = []
        for ep in range(self.epochs):
            batch_losses = []
            for data, target in self.trainloader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
            epoch_losses.append(sum(batch_losses)/len(batch_losses))
        
        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0

        # --- FedExp Importance Calculation ---
        importance_dict = {}
        # Dynamically find convolutional layers for importance calculation
        # To avoid being too slow, we focus on the last few layers if many exist
        conv_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                conv_layers.append((name, module))
        
        # Take the last 3 conv layers (standard for pathway refinement)
        target_layers = conv_layers[-3:] if len(conv_layers) > 3 else conv_layers
        
        if target_layers:
            self.model.eval()
            set_inplace(self.model, False)
            
            for name, layer in target_layers:
                layer_imp_sum = None
                count = 0
                for i, (data, target) in enumerate(self.trainloader):
                    if i >= 4: break 
                    data, target = data.to(self.device), target.to(self.device)
                    
                    # We only compute for correctly classified samples
                    with torch.no_grad():
                        out = self.model(data)
                        preds = out.argmax(dim=1)
                        mask = preds == target
                    
                    if mask.any():
                        imp = get_layer_importance(self.model, layer, data[mask], target[mask], self.device)
                        if layer_imp_sum is None:
                            layer_imp_sum = imp
                        else:
                            layer_imp_sum += imp
                        count += 1
                
                if count > 0:
                    importance_dict[name] = (layer_imp_sum / count).tolist()
                else:
                    importance_dict[name] = [1.0] * layer.out_channels

        return self.get_parameters(config={}), len(self.trainloader.dataset), {"loss": float(avg_loss), "importance": json.dumps(importance_dict)}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        correct = 0
        total = 0
        loss = 0.0
        with torch.no_grad():
            for data, target in self.valloader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                loss += criterion(outputs, target).item()
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        acc = correct / total if total > 0 else 0
        return float(loss) / len(self.valloader), len(self.valloader.dataset), {"accuracy": float(acc)}

# --- 3. Performance Tracker ---

class PerformanceTracker:
    def __init__(self):
        self.fit_start_time = 0
        self.durations = []

    def start_fit(self, server_round: int):
        self.fit_start_time = time.time()
        return {}

    def stop_evaluate(self, metrics: List[Tuple[int, Dict]]):
        # This is where we finalize the round duration
        if self.fit_start_time > 0:
            duration = time.time() - self.fit_start_time
            self.durations.append(duration)
        
        if not metrics:
            return {"accuracy": 0.0}
            
        accs = [num_examples * m["accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]
        return {"accuracy": sum(accs) / sum(examples)}

# --- 4. Utilities ---

def print_experiment_summary(args, device, client_indices, targets, selected_baselines, num_classes):
    print("\n" + "="*50)
    print("      EXPERIMENT CONFIGURATION SUMMARY")
    print("="*50)
    print(f"Dataset:       {args.dataset.upper()}")
    print(f"Process Unit:  {device.upper()}")
    print(f"Rounds:        {args.rounds}")
    print(f"Local Epochs:  {args.epochs}")
    print(f"Batch Size:    {args.batch_size}")
    print(f"Num Clients:   {args.num_clients}")
    print(f"Data Distr:    Alpha={args.data_distr} ({'IID' if args.data_distr >= 1.0 else 'Non-IID'})")
    print(f"Baselines:     {', '.join([STRATEGY_DISPLAY_NAMES.get(b, b) for b in selected_baselines])}")
    print(f"Model:         {args.model}")
    print("-" * 50)
    
    print("Data Distribution per Client (Label Counts):")
    # Truncate label header if too many classes
    max_labels_to_show = 15
    if num_classes > max_labels_to_show:
        labels_to_show = list(range(max_labels_to_show))
        label_header = " | ".join([f"L{i}" for i in labels_to_show]) + " | ..."
    else:
        label_header = " | ".join([f"L{i}" for i in range(num_classes)])
    
    header = f"Client    | {label_header} | Total"
    print(header)
    print("-" * len(header))
    
    for client_idx in range(len(client_indices)):
        client_targets = targets[client_indices[client_idx]]
        counts = []
        limit = min(num_classes, max_labels_to_show)
        for label in range(limit):
            count = np.sum(client_targets == label)
            counts.append(f"{count:3}")
        
        row = f"Client {client_idx:2} | " + " | ".join(counts)
        if num_classes > max_labels_to_show:
            row += " | ..."
        row += f" | {len(client_targets):4}"
        print(row)
    print("="*50 + "\n")

# --- 5. Strategy Naming Mapping ---

STRATEGY_DISPLAY_NAMES = {
    'fedavg': 'FedAvg',
    'fedavgm': 'FedAvgM',
    'fedadam': 'FedAdam',
    'fedyogi': 'FedYogi',
    'fedadagrad': 'FedAdagrad',
    'fedprox': 'FedProx',
    'fedmedian': 'FedMedian',
    'fedtrimmedavg': 'FedTrimmedAvg',
    'faulttolerantfedavg': 'FaultTolerantFedAvg',
    #'fedexp': 'FedExp',
}

# --- 6. Main Simulation ---

def main():
    parser = argparse.ArgumentParser(description='Flower Baseline Simulator')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'stl10', 'mnist', 'oxfordpet'])
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'squeezenet', 'shufflenet', 'resnet'])
    parser.add_argument('--num_clients', type=int, default=4)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=1, help='Local epochs (auto-tuned if None)')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (auto-tuned if None)')
    parser.add_argument('--baseline', type=str, nargs='+', default=['fedavg'], 
                        help='Aggregation strategy or "all". Available: ' + ', '.join(STRATEGY_DISPLAY_NAMES.keys()))
    parser.add_argument('--lr', type=float, default=None, help='Learning rate (auto-tuned if None)')
    parser.add_argument('--data-distr', type=float, default=1.0, help='Data distribution (1.0 for IID, < 1.0 for Dirichlet non-IID)')
    
    args = parser.parse_args()
    
    # Dynamic Hyperparameter Selection
    hparams = DATASET_HYPERPARAMS.get(args.dataset.lower(), {'lr': 0.01, 'batch_size': 32, 'epochs': 1})
    if args.lr is None: args.lr = hparams['lr']
    if args.batch_size is None: args.batch_size = hparams['batch_size']
    if args.epochs is None: args.epochs = hparams['epochs']
    
    args.vis_dir = 'results' # Fixed to results
    os.makedirs(args.vis_dir, exist_ok=True)
    os.makedirs('csv', exist_ok=True)
    
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    if args.dataset == 'mnist':
        img_size = 28
    elif args.dataset == 'oxfordpet':
        img_size = 224
    elif args.dataset == 'cifar10':
        img_size = 32
    else:
        img_size = 128

    in_channels = 1 if args.dataset == 'mnist' else 3
    num_classes = 37 if args.dataset == 'oxfordpet' else 10
    train_set, test_set, targets = get_dataset(args.dataset, img_size)
    client_indices = partition_data(train_set, targets, args.num_clients, args.data_distr, num_classes)
    
    def client_fn(context: Context) -> fl.client.Client:
        cid = context.node_config["partition-id"]
        model = get_model(args.model, num_classes=num_classes, input_size=img_size, in_channels=in_channels).to(device)
        idx = int(cid)
        ds = Subset(train_set, client_indices[idx])
        trainloader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
        valloader = DataLoader(test_set, batch_size=args.batch_size)
        return FlowerClient(model, trainloader, valloader, device, args.epochs, lr=args.lr).to_client()

    available_baselines = list(STRATEGY_DISPLAY_NAMES.keys())
    requested_baselines = [b.lower() for b in args.baseline]
    if 'all' in requested_baselines:
        selected_baselines = available_baselines
        baseline_name_for_file = "all"
    else:
        selected_baselines = requested_baselines
        baseline_name_for_file = "-".join(requested_baselines)

    # Warmup hardware before starting any baseline
    warmup(args.model, num_classes, img_size, in_channels, device)

    # Printing initial info
    print_experiment_summary(args, device, client_indices, targets, selected_baselines, num_classes)

    all_metric_results = {}
    all_time_results = {}

    for mode in selected_baselines:
        display_name = STRATEGY_DISPLAY_NAMES.get(mode, mode)
        print(f"\n=== Starting Flower simulation: {display_name} Baseline ===")
        tracker = PerformanceTracker()
        
        initial_model = get_model(args.model, num_classes=num_classes, input_size=img_size, in_channels=in_channels).to(device)
        initial_params = [val.cpu().numpy() for _, val in initial_model.state_dict().items()]
        initial_parameters = fl.common.ndarrays_to_parameters(initial_params)

        common_params = {
            "fraction_fit": 1.0,
            "fraction_evaluate": 1.0,
            "min_fit_clients": args.num_clients,
            "min_evaluate_clients": args.num_clients,
            "min_available_clients": args.num_clients,
            "on_fit_config_fn": tracker.start_fit,
            "evaluate_metrics_aggregation_fn": tracker.stop_evaluate,
            "initial_parameters": initial_parameters,
        }

        if mode == 'fedavg':
            strategy = fl.server.strategy.FedAvg(**common_params)
        elif mode == 'fedavgm':
            strategy = fl.server.strategy.FedAvgM(server_learning_rate=1.0, server_momentum=0.9, **common_params)
        elif mode == 'fedyogi':
            strategy = fl.server.strategy.FedYogi(eta=0.01, eta_l=0.0316, beta_1=0.9, beta_2=0.99, tau=0.01, **common_params)
        elif mode == 'fedadam':
            strategy = fl.server.strategy.FedAdam(eta=0.01, eta_l=0.01, beta_1=0.9, beta_2=0.99, tau=0.01, **common_params)
        elif mode == 'fedadagrad':
            strategy = fl.server.strategy.FedAdagrad(eta=0.01, eta_l=0.1, tau=0.01, **common_params)
        elif mode == 'fedprox':
            strategy = fl.server.strategy.FedProx(proximal_mu=0.1, **common_params)
        elif mode == 'fedmedian':
            strategy = fl.server.strategy.FedMedian(**common_params)
        elif mode == 'faulttolerantfedavg':
            strategy = fl.server.strategy.FaultTolerantFedAvg(**common_params)
        elif mode == 'fedtrimmedavg':
            strategy = fl.server.strategy.FedTrimmedAvg(**common_params)
        elif mode == 'fedexp':
            m_params = {
                'name': args.model,
                'num_classes': num_classes,
                'input_size': img_size,
                'in_channels': in_channels,
                'dataset_name': args.dataset
            }
            strategy = FedExpStrategy(test_set=test_set, device=device, model_params=m_params, **common_params)
        else:
            print(f"Skipping unknown baseline: {mode}")
            continue
        
        # Define client resources for simulation
        # Use a fraction of a CPU to allow higher parallelism on limited hardware
        client_resources = {"num_cpus": 0.5} 
        
        if device == "cuda":
            # On NVIDIA, Ray tracks GPUs, so we can share the physical device
            client_resources["num_gpus"] = 1.0 / args.num_clients
        # Note: on MacOS (MPS), Ray does NOT see the GPU as a standard resource.
        # We don't set num_gpus here; PyTorch will still use MPS internally.
        
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=args.num_clients,
            config=fl.server.ServerConfig(num_rounds=args.rounds),
            strategy=strategy,
            client_resources=client_resources,
        )
        
        if "accuracy" in history.metrics_distributed:
            history_acc = [val for _, val in history.metrics_distributed["accuracy"]]
            all_metric_results[display_name] = history_acc
            all_time_results[display_name] = tracker.durations
            
        # Log to CSV
        csv_path = f'csv/baseline_{mode}_{args.dataset}.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Round', 'Accuracy', 'Duration_Sec', 'Clients', 'Epochs', 'Model', 'Dataset'])
            for r in range(len(history_acc)):
                dur = tracker.durations[r] if r < len(tracker.durations) else 0.0
                model_name = args.model
                writer.writerow([r+1, history_acc[r], dur, args.num_clients, args.epochs, model_name, args.dataset])
        print(f"Log saved to {csv_path}")

    # Plotting (Separated)
    if all_metric_results:
        # 1. Accuracy Plot
        plt.figure(figsize=(10, 6))
        for display_name in all_metric_results.keys():
            plt.plot(range(1, len(all_metric_results[display_name]) + 1), all_metric_results[display_name], label=display_name, marker='o')
        plt.xlabel('Round')
        plt.ylabel('Testing Accuracy')
        plt.title(f'Accuracy Comparison - {args.dataset}')
        plt.legend()
        plt.grid(False)
        acc_filename = f"accuracy_{baseline_name_for_file}_{args.model}_{args.num_clients}Clients.pdf"
        acc_plot_path = os.path.join(args.vis_dir, acc_filename)
        plt.savefig(acc_plot_path)
        plt.close()
        print(f"\nAccuracy plot saved to {acc_plot_path}")
        
        # 2. Time Plot
        plt.figure(figsize=(10, 6))
        for display_name in all_time_results.keys():
            plt.plot(range(1, len(all_time_results[display_name]) + 1), all_time_results[display_name], label=display_name, marker='s')
        plt.xlabel('Round')
        plt.ylabel('Round Duration (seconds)')
        plt.title(f'Time per Round Comparison - {args.dataset}')
        plt.legend()
        plt.grid(False)
        time_filename = f"time_{baseline_name_for_file}_{args.model}_{args.num_clients}Clients.pdf"
        time_plot_path = os.path.join(args.vis_dir, time_filename)
        plt.savefig(time_plot_path)
        plt.close()
        print(f"Time plot saved to {time_plot_path}")

if __name__ == "__main__":
    main()
