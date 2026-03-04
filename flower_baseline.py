import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset

import flwr as fl
from flwr.common import (
    Context,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
import argparse
import numpy as np
import os
import csv
import time
from collections import OrderedDict
from typing import List, Tuple, Dict, Optional, Union
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import logging
import urllib.request
import json
import base64
import warnings
import random

# Suppress warnings
logging.getLogger("flwr").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*mode.*deprecated.*Pillow.*")
warnings.filterwarnings("ignore", message=".*urllib3.*OpenSSL.*")

from captum.attr import visualization as vit
from newAux import FedGSW
from strategy_laa import FedLAA
from strategy_ama import FedAMA
from strategy_de import FedDE
from strategy_sbfl import FedSBFL

DATASET_HYPERPARAMS = {
    'mnist': {'lr': 0.01, 'batch_size': 32, 'epochs': 1},
    'cifar10': {'lr': 0.01, 'batch_size': 64, 'epochs': 1},
    'stl10': {'lr': 0.01, 'batch_size': 32, 'epochs': 1},
    'oxfordpet': {'lr': 0.005, 'batch_size': 16, 'epochs': 1},
    'adult': {'lr': 0.01, 'batch_size': 128, 'epochs': 1},
    'speechcommands': {'lr': 0.01, 'batch_size': 32, 'epochs': 1}
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

class MLP(nn.Module):
    def __init__(self, input_dim, num_classes=2, hidden_dims=[64, 32]):
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class M5(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return x.squeeze(2)

def get_model(name, num_classes=10, input_size=128, in_channels=3, input_dim=None):
    if name.lower() == 'cnn':
        return CNN(num_classes=num_classes, input_size=input_size, in_channels=in_channels)
    
    elif name.lower() == 'mlp':
        if input_dim is None:
            raise ValueError("input_dim must be specified for MLP model")
        return MLP(input_dim=input_dim, num_classes=num_classes)

    elif name.lower() == 'm5':
        return M5(n_input=in_channels, n_output=num_classes)
    
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

def get_split_models(name, num_classes=10, input_size=128, in_channels=3, input_dim=None):
    if name.lower() == 'cnn':
        class CNNFront(nn.Module):
            def __init__(self, in_channels):
                super().__init__()
                self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
            def forward(self, x):
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                return x
        
        class CNNBack(nn.Module):
            def __init__(self, num_classes, input_size, in_channels):
                super().__init__()
                self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                dummy_input = torch.zeros(1, in_channels, input_size, input_size)
                dummy_front = CNNFront(in_channels)
                with torch.no_grad():
                    x = self.pool(F.relu(self.conv3(dummy_front(dummy_input))))
                self.fc_input_dim = x.view(1, -1).size(1)
                self.fc1 = nn.Linear(self.fc_input_dim, 64)
                self.fc2 = nn.Linear(64, num_classes)
            def forward(self, x):
                x = self.pool(F.relu(self.conv3(x)))
                x = x.view(-1, self.fc_input_dim)
                x = F.relu(self.fc1(x))
                x = self.fc2(x)
                return x
        return CNNFront(in_channels), CNNBack(num_classes, input_size, in_channels)

    elif name.lower() == 'mlp':
        class MLPFront(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.2))
            def forward(self, x):
                return self.net(x)
        class MLPBack(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, num_classes))
            def forward(self, x):
                return self.net(x)
        return MLPFront(input_dim), MLPBack(num_classes)

    elif name.lower() == 'm5':
        class M5Front(nn.Module):
            def __init__(self, n_input, stride=16, n_channel=32):
                super().__init__()
                self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
                self.bn1 = nn.BatchNorm1d(n_channel)
                self.pool1 = nn.MaxPool1d(4)
                self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
                self.bn2 = nn.BatchNorm1d(n_channel)
                self.pool2 = nn.MaxPool1d(4)
            def forward(self, x):
                x = self.pool1(F.relu(self.bn1(self.conv1(x))))
                x = self.pool2(F.relu(self.bn2(self.conv2(x))))
                return x
        class M5Back(nn.Module):
            def __init__(self, n_output, n_channel=32):
                super().__init__()
                self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
                self.bn3 = nn.BatchNorm1d(2 * n_channel)
                self.pool3 = nn.MaxPool1d(4)
                self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
                self.bn4 = nn.BatchNorm1d(2 * n_channel)
                self.pool4 = nn.MaxPool1d(4)
                self.fc1 = nn.Linear(2 * n_channel, n_output)
            def forward(self, x):
                x = self.pool3(F.relu(self.bn3(self.conv3(x))))
                x = self.pool4(F.relu(self.bn4(self.conv4(x))))
                x = F.avg_pool1d(x, x.shape[-1]).permute(0, 2, 1)
                return self.fc1(x).squeeze(2)
        return M5Front(in_channels), M5Back(num_classes)

    else:
        full_model = get_model(name, num_classes, input_size, in_channels, input_dim)
        if name.lower() == 'squeezenet':
            front = nn.Sequential(*list(full_model.features.children())[:5])
            class SqueezeNetBack(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.features_back = nn.Sequential(*list(model.features.children())[5:])
                    self.classifier = model.classifier
                def forward(self, x):
                    x = self.features_back(x)
                    x = self.classifier(x)
                    return torch.flatten(x, 1)
            return front, SqueezeNetBack(full_model)
        elif name.lower() == 'shufflenet':
            front = nn.Sequential(full_model.conv1, full_model.maxpool, full_model.stage2)
            class ShuffleNetBack(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.stage3 = model.stage3
                    self.stage4 = model.stage4
                    self.conv5 = model.conv5
                    self.fc = model.fc
                def forward(self, x):
                    x = self.stage3(x)
                    x = self.stage4(x)
                    x = self.conv5(x)
                    x = x.mean([2, 3])
                    return self.fc(x)
            return front, ShuffleNetBack(full_model)
        elif name.lower() == 'resnet':
            front = nn.Sequential(full_model.conv1, full_model.bn1, full_model.relu, full_model.maxpool, full_model.layer1)
            class ResNetBack(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.layer2 = model.layer2
                    self.layer3 = model.layer3
                    self.layer4 = model.layer4
                    self.avgpool = model.avgpool
                    self.fc = model.fc
                def forward(self, x):
                    x = self.layer2(x)
                    x = self.layer3(x)
                    x = self.layer4(x)
                    x = self.avgpool(x)
                    x = torch.flatten(x, 1)
                    return self.fc(x)
            return front, ResNetBack(full_model)
        else:
            raise ValueError(f"Split implementation not found for {name}")

class SplitServer:
    def __init__(self, back_model, device, lr):
        self.model = back_model.to(device)
        self.device = device
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()

    def train_step(self, smashed_data, labels):
        self.model.train()
        smashed_data = smashed_data.detach().clone().to(self.device).requires_grad_(True)
        labels = labels.to(self.device).long()

        self.optimizer.zero_grad()
        output = self.model(smashed_data)
        loss = self.criterion(output, labels)
        loss.backward()
        self.optimizer.step()
        
        return smashed_data.grad.clone(), loss.item()

    def eval_forward(self, smashed_data):
        self.model.eval()
        with torch.no_grad():
            smashed_data = smashed_data.to(self.device)
            output = self.model(smashed_data)
        return output

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

class TabularDataset(torch.utils.data.Dataset):
    """Custom dataset for tabular data."""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_dataset(name, img_size):
    if name.lower() == 'adult':
        from sklearn.datasets import fetch_openml
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.model_selection import train_test_split
        
        print("  - Downloading Adult Income dataset...")
        data = fetch_openml('adult', version=2, as_frame=True, parser='auto')
        X = data.data.copy()
        y = data.target.copy()
        
        for col in X.select_dtypes(include=['category', 'object']).columns:
            X[col] = X[col].astype(str)
            X.loc[:, col] = LabelEncoder().fit_transform(X[col])
        
        y = LabelEncoder().fit_transform(y)
        
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        train_set = TabularDataset(X_train, y_train)
        test_set = TabularDataset(X_test, y_test)
        targets = y_train
        
        num_classes = len(np.unique(y_train)) # Dynamically determine num_classes for adult
        
        return train_set, test_set, targets, num_classes
    
    elif name.lower() == 'cifar10':
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        targets = np.array(train_set.targets)
        num_classes = 10
    elif name.lower() == 'stl10':
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_set = datasets.STL10(root='./data', split='train', download=True, transform=transform)
        test_set = datasets.STL10(root='./data', split='test', download=True, transform=transform)
        targets = np.array(train_set.labels)
        num_classes = 10
    elif name.lower() == 'mnist':
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        targets = np.array(train_set.targets)
        num_classes = 10
    elif name.lower() == 'oxfordpet':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_set = datasets.OxfordIIITPet(root='./data', split='trainval', download=True, transform=transform)
        test_set = datasets.OxfordIIITPet(root='./data', split='test', download=True, transform=transform)
        targets = np.array(train_set._labels)
        num_classes = 37
    elif name.lower() == 'speechcommands':
        import torchaudio
        class SubsetSC(torchaudio.datasets.SPEECHCOMMANDS):

            def __init__(self, subset: str = None):
                super().__init__("./data", download=True)
                
                def load_list(filename):
                    filepath = os.path.join(self._path, filename)
                    with open(filepath) as fileobj:
                        return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]
                
                if subset == "validation":
                    self._walker = load_list("validation_list.txt")
                elif subset == "testing":
                    self._walker = load_list("testing_list.txt")
                elif subset == "training":
                    excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
                    excludes = set(excludes)
                    self._walker = [w for w in self._walker if w not in excludes]
        
        if name.lower() == 'speechcommands':
            # Use 4-class subset as requested by user
            # Common commands: 'yes', 'no', 'up', 'down'
            subset_labels = ['yes', 'no', 'up', 'down']
            label_to_index = {label: index for index, label in enumerate(subset_labels)}
            num_classes = len(subset_labels)
            
            # Helper to check if a sample is in the subset
            def is_in_subset(datapoint):
                return datapoint[2] in subset_labels

            # Load and filter raw datasets
            train_set_raw = [d for d in SubsetSC("training") if is_in_subset(d)]
            test_set_raw = [d for d in SubsetSC("testing") if is_in_subset(d)]
            
            if not train_set_raw:
                # Fallback to standard 10+2 if subset empty (shouldn't happen with standard dataset)
                print("Warning: Requested subset empty. Using full dataset.")
                all_labels = sorted(list(set(datapoint[2] for datapoint in SubsetSC("training"))))
                labels = [l for l in all_labels if l != "_background_noise_"]
                label_to_index = {label: index for index, label in enumerate(labels)}
                num_classes = len(labels)
                train_set_raw = [d for d in SubsetSC("training") if d[2] != "_background_noise_"]
                test_set_raw = [d for d in SubsetSC("testing") if d[2] != "_background_noise_"]


        class SPEECHCOMMANDS_Processed(torch.utils.data.Dataset):
            def __init__(self, dataset, label_to_index):
                self.dataset = dataset
                self.label_to_index = label_to_index
                self.new_sample_rate = 8000
                self.transform = torchaudio.transforms.Resample(orig_freq=16000, new_freq=self.new_sample_rate)

            def __getitem__(self, n):
                waveform, sample_rate, label, _, _ = self.dataset[n]
                # Filter out background_noise (often has different shape or no label) if necessary, 
                # but standard SubsetSC usually handles valid files. 
                # Note: M5 expects 1xInputLength. Pad to 1 sec (8000 samples).
                
                waveform = self.transform(waveform)
                
                # Pad or truncate to 1 second
                if waveform.shape[-1] < self.new_sample_rate:
                    waveform = torch.nn.functional.pad(waveform, (0, self.new_sample_rate - waveform.shape[-1]))
                else:
                    waveform = waveform[:, :self.new_sample_rate]
                
                label_idx = self.label_to_index.get(label, 0)
                
                return waveform, torch.tensor(label_idx, dtype=torch.long)


            def __len__(self):
                return len(self.dataset)

        train_set = SPEECHCOMMANDS_Processed(train_set_raw, label_to_index)
        test_set = SPEECHCOMMANDS_Processed(test_set_raw, label_to_index)
        
        # Extract targets for partitioning
        print(f"  - Processing {num_classes}-class SpeechCommands targets...")
        targets = []
        for i in range(len(train_set)):
            _, label = train_set[i]
            # Ensure label is int for np.array
            targets.append(int(label))
        targets = np.array(targets, dtype=np.int64)
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    return train_set, test_set, targets, num_classes

def partition_data(dataset, targets, num_clients, alpha, num_classes):
    indices = [[] for _ in range(num_clients)]
    
    if alpha >= 1.0:
        all_indices = np.arange(len(dataset))
        np.random.shuffle(all_indices)
        indices = np.array_split(all_indices, num_clients)
        return [idx.tolist() for idx in indices]
    else:
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

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, valloader, device, epochs, lr, split_server=None):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.split_server = split_server

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
                data, target = data.to(self.device), target.to(self.device).long()
                
                # Debug logging
                with open("debug_log.txt", "a") as f:
                    output_temp = self.model(data)
                    f.write(f"Round Debug - Output shape: {output_temp.shape}, Target shape: {target.shape}, Target dtype: {target.dtype}\n")
                
                optimizer.zero_grad()
                output = self.model(data)

                if self.split_server is not None:
                    # Split Learning Mode
                    smashed_grad, loss_item = self.split_server.train_step(output, target)
                    output.backward(smashed_grad)
                    optimizer.step()
                    batch_losses.append(loss_item)
                else:
                    # Federated Learning Mode
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    batch_losses.append(loss.item())

            epoch_losses.append(sum(batch_losses)/len(batch_losses))
        
        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0

        return self.get_parameters(config={}), len(self.trainloader.dataset), {"loss": float(avg_loss)}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        correct = 0
        total = 0
        loss = 0.0
        with torch.no_grad():
            for data, target in self.valloader:
                data, target = data.to(self.device), target.to(self.device).long()

                output = self.model(data)

                if self.split_server is not None:
                    # Split Learning Mode
                    server_outputs = self.split_server.eval_forward(output)
                    loss += criterion(server_outputs, target).item()
                    _, predicted = torch.max(server_outputs.data, 1)
                else:
                    # Federated Learning Mode
                    loss += criterion(output, target).item()
                    _, predicted = torch.max(output.data, 1)

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
    print(f"Learning Type: {args.learning_type.upper()}")
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
    #'fedavgm': 'FedAvgM',
    #'fedmedian': 'FedMedian',
    #'fedadam': 'FedAdam',
    #'fedprox': 'FedProx',
    'fedgsw': 'FedGSW',
    'fedlaa': 'FedLAA',
    'fedama': 'FedAMA',
    'fedde': 'FedDE',
    'fedsbfl': 'FedSBFL',
}



# --- 6. Main Simulation ---

def main():
    parser = argparse.ArgumentParser(description='Flower Baseline Simulator')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'stl10', 'mnist', 'oxfordpet', 'adult', 'speechcommands'])
    parser.add_argument('--model', type=str, nargs='+', default=['cnn'], 
                        help='Model(s) to use. Can specify multiple: --model cnn resnet',
                        choices=['cnn', 'squeezenet', 'shufflenet', 'resnet', 'mlp', 'm5'])
    parser.add_argument('--num_clients', type=int, default=4)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=1, help='Local epochs (auto-tuned if None)')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (auto-tuned if None)')
    parser.add_argument('--baseline', type=str, nargs='+', default=['all'], 
                        help='Aggregation strategy or "all". Available: ' + ', '.join(STRATEGY_DISPLAY_NAMES.keys()))
    parser.add_argument('--lr', type=float, default=None, help='Learning rate (auto-tuned if None)')
    parser.add_argument('--data-distr', type=float, default=1.0, help='Data distribution (1.0 for IID, < 1.0 for Dirichlet non-IID)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--learning-type', type=str, default='FL', choices=['FL', 'SL'], help='Learning type: FL or SL')
    
    args = parser.parse_args()
    args.model = [m.lower() for m in args.model]

    # Set random seed
    random_seed = args.seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
    
    hparams = DATASET_HYPERPARAMS.get(args.dataset.lower(), {'lr': 0.01, 'batch_size': 32, 'epochs': 1})
    if args.lr is None: args.lr = hparams['lr']
    if args.batch_size is None: args.batch_size = hparams['batch_size']
    if args.epochs is None: args.epochs = hparams['epochs']
    
    args.vis_dir = 'results'
    os.makedirs(args.vis_dir, exist_ok=True)
    os.makedirs('csv', exist_ok=True)
    
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    is_tabular = args.dataset in ['adult']
    is_audio = args.dataset in ['speechcommands']

    models_to_run = [m.lower() for m in args.model]

    if is_tabular:
        img_size = None
        in_channels = None
        models_to_run = ['mlp']
    elif is_audio:
        img_size = 8000
        in_channels = 1
        models_to_run = ['m5']
    else:
        if args.dataset == 'mnist':
            img_size = 28
        elif args.dataset == 'oxfordpet':
            img_size = 224
        elif args.dataset == 'cifar10':
            img_size = 32
        else:
            img_size = 128

        in_channels = 1 if args.dataset == 'mnist' else 3

        for m in models_to_run:
            if m in ['mlp', 'm5']:
                raise ValueError(f"Model {m} not supported for image dataset {args.dataset}")

        
        # Load dataset
        train_set, test_set, targets, num_classes = get_dataset(args.dataset, img_size if not (is_tabular or is_audio) else None)
        
        if is_tabular:
            input_dim = train_set.X.shape[1]
        elif is_audio:
            input_dim = None
        else:
            input_dim = None
    
    client_indices = partition_data(train_set, targets, args.num_clients, args.data_distr, num_classes)

    available_baselines = list(STRATEGY_DISPLAY_NAMES.keys())
    requested_baselines = [b.lower() for b in args.baseline]
    if 'all' in requested_baselines:
        selected_baselines = available_baselines
        baseline_name_for_file = "all"
    else:
        selected_baselines = requested_baselines
        baseline_name_for_file = "-".join(requested_baselines)

    for model_name in models_to_run:
        args.model = model_name

        if is_tabular:
            input_dim = train_set.X.shape[1]
        else:
            input_dim = None

        split_server = None
        if args.learning_type.upper() == 'SL':
            if is_tabular:
                front_model, back_model = get_split_models(model_name, num_classes=num_classes, input_dim=input_dim)
            elif is_audio:
                front_model, back_model = get_split_models(model_name, num_classes=num_classes, in_channels=in_channels)
            else:
                front_model, back_model = get_split_models(model_name, num_classes=num_classes, input_size=img_size, in_channels=in_channels)
            split_server = SplitServer(back_model, device, args.lr)
        else:
            if is_tabular:
                front_model = get_model(model_name, num_classes=num_classes, input_dim=input_dim)
            elif is_audio:
                front_model = get_model(model_name, num_classes=num_classes, in_channels=in_channels)
            else:
                front_model = get_model(model_name, num_classes=num_classes, input_size=img_size, in_channels=in_channels)

        def client_fn(context: Context) -> fl.client.Client:
            cid = context.node_config["partition-id"]
            if args.learning_type.upper() == 'SL':
                if is_tabular:
                    model, _ = get_split_models(model_name, num_classes=num_classes, input_dim=input_dim)
                elif is_audio:
                    model, _ = get_split_models(model_name, num_classes=num_classes, in_channels=in_channels)
                else:
                    model, _ = get_split_models(model_name, num_classes=num_classes, input_size=img_size, in_channels=in_channels)
            else:
                if is_tabular:
                    model = get_model(model_name, num_classes=num_classes, input_dim=input_dim)
                elif is_audio:
                    model = get_model(model_name, num_classes=num_classes, in_channels=in_channels)
                else:
                    model = get_model(model_name, num_classes=num_classes, input_size=img_size, in_channels=in_channels)
            
            model = model.to(device)
            idx = int(cid)
            ds = Subset(train_set, client_indices[idx])
            trainloader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
            valloader = DataLoader(test_set, batch_size=args.batch_size)
            return FlowerClient(model, trainloader, valloader, device, args.epochs, lr=args.lr, split_server=split_server).to_client()

        warmup_model = front_model.to(device)
        dummy_input = torch.randn(1, input_dim).to(device) if is_tabular else (
            torch.randn(1, in_channels, img_size).to(device) if is_audio else 
            torch.randn(1, in_channels, img_size, img_size).to(device)
        )

        print("  - Warming up hardware (first-run initialization)...")
        warmup_model.train()
        try:
            output = warmup_model(dummy_input)
            loss = output.sum()
            loss.backward()
            if device == 'cuda':
                torch.cuda.synchronize()
            elif device == 'mps':
                torch.mps.synchronize()
        except Exception as e:
            print(f"    Warning: Warmup failed (not critical): {e}")
        print("  - Warmup complete.")

        print_experiment_summary(args, device, client_indices, targets, selected_baselines, num_classes)

        all_metric_results = {}
        all_time_results = {}
        history_acc = []
        durations_fixed = []

        for mode in selected_baselines:
            display_name = STRATEGY_DISPLAY_NAMES.get(mode, mode)
            print(f"\n=== Starting Flower simulation: {display_name} Baseline ===")
            tracker = PerformanceTracker()

            if args.learning_type.upper() == 'SL':
                if is_tabular:
                    initial_model, back_model = get_split_models(model_name, num_classes=num_classes, input_dim=input_dim)
                elif is_audio:
                    initial_model, back_model = get_split_models(model_name, num_classes=num_classes, in_channels=in_channels)
                else:
                    initial_model, back_model = get_split_models(model_name, num_classes=num_classes, input_size=img_size, in_channels=in_channels)
                initial_model = initial_model.to(device)
                split_server.model = back_model.to(device)
                split_server.optimizer = optim.SGD(split_server.model.parameters(), lr=args.lr, momentum=0.9)
            else:
                if is_tabular:
                    initial_model = get_model(model_name, num_classes=num_classes, input_dim=input_dim).to(device)
                elif is_audio:
                    initial_model = get_model(model_name, num_classes=num_classes, in_channels=in_channels).to(device)
                else:
                    initial_model = get_model(model_name, num_classes=num_classes, input_size=img_size, in_channels=in_channels).to(device)

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
            elif mode == 'fedadam':
                strategy = fl.server.strategy.FedAdam(eta=0.01, eta_l=0.01, beta_1=0.9, beta_2=0.99, tau=0.01, **common_params)
            elif mode == 'fedprox':
                strategy = fl.server.strategy.FedProx(proximal_mu=0.1, **common_params)
            elif mode == 'fedmedian':
                strategy = fl.server.strategy.FedMedian(**common_params)
            elif mode == 'fedgsw':
                strategy = FedGSW(**common_params)
            elif mode == 'fedlaa':
                strategy = FedLAA(**common_params)
            elif mode == 'fedama':
                strategy = FedAMA(**common_params)
            elif mode == 'fedde':
                strategy = FedDE(**common_params)
            elif mode == 'fedsbfl':
                strategy = FedSBFL(**common_params)
            else:
                print(f"Skipping unknown baseline: {mode}")
                continue

            client_resources = {"num_cpus": 0.5}
            if device == "cuda":
                client_resources["num_gpus"] = 1.0 / args.num_clients

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

                durations_fixed = list(tracker.durations)
                if len(durations_fixed) > 1:
                    durations_fixed[0] = float(np.mean(durations_fixed[1:]))

                all_time_results[display_name] = durations_fixed

            csv_path = f'csv/baseline_{mode}_{model_name}_{args.dataset}_{args.num_clients}Clients.csv'
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Round', 'Accuracy', 'Duration_Sec', 'Clients', 'Epochs', 'Model', 'Dataset', 'Data Distr. (Alpha)'])
                for r in range(len(history_acc)):
                    dur = durations_fixed[r] if r < len(durations_fixed) else 0.0
                    writer.writerow([r+1, history_acc[r], dur, args.num_clients, args.epochs, model_name, args.dataset, args.data_distr])
            print(f"Log saved to {csv_path}")

        if all_metric_results:
            plt.figure(figsize=(10, 6))
            for display_name in all_metric_results.keys():
                plt.plot(range(1, len(all_metric_results[display_name]) + 1), all_metric_results[display_name], label=display_name, marker='o')
            plt.xlabel('Federated Learning Round')
            plt.ylabel('Testing Accuracy')
            plt.title(f'Testing Accuracy - {model_name} and {args.dataset}')
            plt.legend()
            plt.grid(False)
            acc_filename = f"accuracy_{baseline_name_for_file}_{model_name}_{args.num_clients}Clients.pdf"
            acc_plot_path = os.path.join(args.vis_dir, acc_filename)
            plt.savefig(acc_plot_path)
            plt.close()
            print(f"\nAccuracy plot saved to {acc_plot_path}")

            plt.figure(figsize=(10, 6))
            for display_name in all_time_results.keys():
                plt.plot(range(1, len(all_time_results[display_name]) + 1), all_time_results[display_name], label=display_name, marker='s')
            plt.xlabel('Round')
            plt.ylabel('Round Duration (seconds)')
            plt.title(f'Time per Round Comparison - {args.dataset} (Model: {model_name})')
            plt.legend()
            plt.grid(False)
            time_filename = f"time_{baseline_name_for_file}_{model_name}_{args.num_clients}Clients.pdf"
            time_plot_path = os.path.join(args.vis_dir, time_filename)
            plt.savefig(time_plot_path)
            plt.close()
            print(f"Time plot saved to {time_plot_path}")

if __name__ == "__main__":
    main()
